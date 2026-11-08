"""Инженерен режим, безплатният локален начин: от сканирането на детайла (мрежа) и мерките на потребителя прави CAD
модел от отделни елементи (блокове, цилиндри, профили, отвори, джобове, закръгления), а не една мрежа. Изход:
part.py (кодът на CadQuery: всеки елемент на свой ред, с размерите като числа; отваря се в CQ-editor или FreeCAD),
part.step (тела с истински равнини и цилиндри за SolidWorks, Fusion, FreeCAD, Onshape), part.stl и part.json (отчет).

Как: детайлът се завърта по най-малкия правоъгълник отгоре и се мащабира точно по мерките (общите размери излизат
колкото са измерени). Реже се на тънки сечения по всяка от трите оси. Кръглите вътрешни контури, които продължават през
няколко сечения, са отвори по тази ос. За всяка ос отворите по другите две се запълват в сеченията ѝ, и там, където
сеченията не се променят, е един елемент (изтегляне). Оста с най-малко елементи и най-малка грешка спрямо
сканирането е главната. Всяко сечение става кръг, правоъгълник (по желание със закръглени ъгли) или многоъгълник с
изправени страни; вътрешните контури са отвори и джобове, а отворите по другите оси се пробиват отделно. Измерените
диаметри на отвори и издатини и дебелините на стъпала се налагат точно.

Подходящ е за изработени детайли (призматични и стругови: плочи, скоби, втулки, валове, фланци); свободни форми
(фигури, отливки) излизат като стъпаловидно приближение. Като собствен процес:
python cad_local.py <сканиране.stl/.ply> <мерки.json> <изходна папка>
мерки.json: {"length": мм, "width": мм, "height": мм (всяко по желание, поне едно),
             "features": [{"kind": "hole" | "diameter" | "thickness" | "other", "value": мм, "label": текст}]}"""

import json
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

# Равнините на CadQuery: (име, кои глобални оси са локалните u, v, w). Изтеглянето е по w.
PLANES = {"z": ("XY", (0, 1, 2)), "x": ("YZ", (1, 2, 0)), "y": ("ZX", (2, 0, 1))}
EDGE_SEL = {"z": "|Z", "x": "|X", "y": "|Y"}
SLICES = 160  # сечения по всяка ос
SAME = 0.95  # IoU, над което съседни сечения са един елемент
MIN_SLAB = 0.025  # от дължината по оста: по-тънък елемент се слива със съседа (преход, фаска, шум)
CIRCLE_RMS = 0.03  # средно отклонение от кръга (дял от радиуса; поне 0,35 от допуска)
RECT_IOU = 0.97
ROUND = 0.05  # мм: закръгляне на всички размери
SNAP = 0.25  # измерена стойност се налага, ако се различава до толкова (дял) от намерената
EXACT = 0.12  # общ размер се налага точно по оста си при разлика до толкова след общия мащаб
MIN_HOLE = 0.012  # от размера: по-малки контури са шум
MIN_RUN = 0.03  # от сеченията: толкова поне трябва да продължи отвор по ос, за да е отвор
SCORE_SLAB = 0.012  # наказание за всеки елемент при избора на главната ос


def r(v: float) -> float:
    return round(round(float(v) / ROUND) * ROUND, 3)


def iou(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def resample(ring, n: int = 240) -> np.ndarray:
    L = ring.length
    return np.array([ring.interpolate(L * k / n).coords[0] for k in range(n)])


def fit_circle(pts: np.ndarray) -> tuple[float, float, float, float]:
    """(cx, cy, радиус, средно квадратично отклонение), по най-малките квадрати (Kåsa)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    c, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = c[0], c[1]
    rad = math.sqrt(max(c[2] + cx * cx + cy * cy, 1e-12))
    return cx, cy, rad, float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - rad) ** 2)))


def circle_of(ring, tol: float) -> dict | None:
    # Върховете на опростения контур са точки от повърхността; отсечките между тях минават вътре в кръга.
    pts = np.array(ring.coords)[:-1]
    cx, cy, rad, rms = fit_circle(pts if len(pts) >= 16 else resample(ring))
    area = Polygon(ring).area
    if rms < max(CIRCLE_RMS * rad, 0.35 * tol) and abs(math.pi * rad * rad - area) / area < 0.08:
        return {"kind": "circle", "c": [cx, cy], "r": rad}
    return None


# --- Подреждане и мащаб ---------------------------------------------------------------------------------------------


def oriented(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Завъртян около z така, че най-малкият правоъгълник отгоре е по осите (дългата страна по x), центриран, дъно на
    z = 0."""
    from scipy.spatial import ConvexHull

    pts2 = mesh.vertices[:, :2]
    rect = np.array(Polygon(pts2[ConvexHull(pts2).vertices]).minimum_rotated_rectangle.exterior.coords)[:4]
    e0, e1 = rect[1] - rect[0], rect[2] - rect[1]
    edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.atan2(edge[1], edge[0]), [0, 0, 1]))
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return m


def scales(ext: np.ndarray, meas: dict) -> tuple[float, np.ndarray, dict, list[str]]:
    """Общ мащаб, поправка по всяка ос (за точни общи размери) и коя мярка на коя ос е. ext: x (дълга), y, z."""
    notes = []
    given = {k: float(meas[k]) for k in ("length", "width", "height") if meas.get(k)}
    if not given:
        raise ValueError("нужна е поне една обща мярка (дължина, ширина или височина)")
    if "length" in given and "width" in given and given["width"] > given["length"]:
        given["length"], given["width"] = given["width"], given["length"]
    default = {"length": 0, "width": 1, "height": 2}
    keys = list(given)

    def spread(axes):
        lr = [math.log(given[k] / ext[a]) for k, a in zip(keys, axes)]
        return float(np.std(lr)) if len(lr) > 1 else 0.0

    best = tuple(default[k] for k in keys)
    if len(keys) > 1:
        cand = min(permutations(range(3), len(keys)), key=spread)
        if spread(best) > 0.08 and spread(cand) < spread(best) / 2:
            names = {0: "дължината", 1: "ширината", 2: "височината"}
            notes.append("мерките пасват на снимките по-добре в друг ред: " + ", ".join(
                f"{k} -> {names[a]}" for k, a in zip(keys, cand)))  # fmt: skip
            best = cand
    axis_of = dict(zip(keys, best))
    s = math.exp(np.mean([math.log(given[k] / ext[a]) for k, a in axis_of.items()]))
    f = np.ones(3)
    for k, a in axis_of.items():
        g = given[k] / (s * ext[a])
        if abs(g - 1) <= EXACT:
            f[a] = g
        else:
            notes.append(f"{k} {given[k]:g} мм се разминава със снимките с {100 * (g - 1):+.0f}%: проверете мярката "
                         "(моделът е мащабиран по всички мерки заедно)")  # fmt: skip
    return s, f, axis_of, notes


# --- Сечения --------------------------------------------------------------------------------------------------------


def local(mesh: trimesh.Trimesh, axis: str) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices = mesh.vertices[:, list(PLANES[axis][1])]
    return m


def parts_of(g) -> list[Polygon]:
    if g is None or g.is_empty:
        return []
    geoms = g.geoms if hasattr(g, "geoms") else [g]
    return [p for p in geoms if isinstance(p, Polygon) and p.area > 0]


def clean(g, least: float):
    """Без шума: части и дупки, по-малки от `least` (често от сечение точно през плоска стена)."""
    keep = [Polygon(p.exterior, [h for h in p.interiors if Polygon(h).area > least])
            for p in parts_of(g) if p.area > least]  # fmt: skip
    return unary_union(keep) if keep else None


def sections(m: trimesh.Trimesh, n: int, tol: float, hs: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    """n сечения, перпендикулярни на локалната z, като области на shapely. Отсечките на всяко сечение се съединяват в
    затворени контури; контур в нечетен брой други е дупка."""
    from shapely import line_merge
    from shapely.geometry import MultiLineString
    from shapely.prepared import prep

    lo, hi = m.bounds[:, 2]
    if hs is None:
        hs = lo + (np.arange(n) + 0.5) * (hi - lo) / n
    lines, to3d, _ = trimesh.intersections.mesh_multiplane(m, [0, 0, 0], [0, 0, 1], hs)
    size = float(np.max(m.extents))
    q = 1e-6 * size
    least = (MIN_HOLE * size) ** 2 / 2
    out = []
    for seg, T in zip(lines, to3d):
        g = None
        if len(seg):
            p = trimesh.transform_points(np.c_[seg.reshape(-1, 2), np.zeros(len(seg) * 2)], T)[:, :2]
            p = np.round(p / q) * q
            try:
                merged = line_merge(MultiLineString(list(p.reshape(-1, 2, 2))))
                chains = merged.geoms if hasattr(merged, "geoms") else [merged]
                # Отворен контур (пукнатина в мрежата) се затваря, ако краищата му са близо; иначе се пропуска.
                rings = [Polygon(c.coords) for c in chains if len(c.coords) >= 4
                         and (c.is_closed or np.hypot(*np.subtract(c.coords[0], c.coords[-1])) < 2 * tol)]  # fmt: skip
                rings = [x.simplify(tol / 2).buffer(0) for x in rings if abs(x.area) > least]
                rings = sorted((x for x in rings if x.area > least), key=lambda x: -x.area)
                solid, holes = [], []
                for k, x in enumerate(rings):
                    pt = x.representative_point()
                    depth = sum(prep(rings[j]).contains(pt) for j in range(k))
                    (holes if depth % 2 else solid).append(x)
                if solid:
                    g = unary_union(solid)
                    if holes:
                        g = g.difference(unary_union(holes))
                    g = clean(g.buffer(0), least)
            except Exception:
                g = None
        out.append(g)
    if n:  # само при пълния набор: части под 1% от най-голямото сечение са шум
        big = max((g.area for g in out if g is not None), default=0)
        out = [g if g is not None and g.area > 0.01 * big else None for g in out]
    return hs, out


def hole_runs(axis: str, hs: np.ndarray, polys: list, tol: float, size: float) -> list[dict]:
    """Кръгли отвори по оста: вътрешни кръгове, които продължават поне MIN_RUN от сеченията."""
    step = hs[1] - hs[0]
    runs = []
    for k, g in enumerate(polys):
        for part in parts_of(g):
            for ring in part.interiors:
                h = circle_of(ring, tol)
                if h is None or 2 * h["r"] < MIN_HOLE * size:
                    continue
                for run in runs:
                    if run["end"] == k - 1 and same(run["shape"], h, tol):
                        run["end"] = k
                        run["rs"].append(h["r"])
                        break
                else:
                    runs.append({"shape": h, "start": k, "end": k, "rs": [h["r"]]})
    out = []
    for run in runs:
        if run["end"] - run["start"] + 1 >= max(3, int(MIN_RUN * len(hs))):
            out.append({"axis": axis, "shape": dict(run["shape"], r=float(np.median(run["rs"]))),
                        "z": [hs[run["start"]] - step / 2, hs[run["end"]] + step / 2]})  # fmt: skip
    return out


def footprint(hole: dict, axis: str, h: float, margin: float = 0.0):
    """Сечението на отвор по друга ос (цилиндър) с равнината на `axis` на височина h: правоъгълник в локалните ѝ
    координати, или None."""
    iu, iv, iw = PLANES[hole["axis"]][1]
    au, av, aw = PLANES[axis][1]
    cu, cv = hole["shape"]["c"]
    rad = hole["shape"]["r"]
    centre = {iu: cu, iv: cv}
    other = iv if aw == iu else iu
    d = h - centre[aw]
    if abs(d) >= rad:
        return None
    half = math.sqrt(rad * rad - d * d) + margin / 2  # малко в стените: без процеп до сканираната им повърхност
    span = {iw: [hole["z"][0] - margin, hole["z"][1] + margin], other: [centre[other] - half, centre[other] + half]}
    return box(span[au][0], span[av][0], span[au][1], span[av][1])


# --- Елементи -------------------------------------------------------------------------------------------------------


def segment(polys: list) -> list[list[int]]:
    n = len(polys)
    for i in range(1, n - 1):  # единично неуспяло сечение вътре в детайла: като съседното
        if polys[i] is None and polys[i - 1] is not None and any(polys[j] is not None for j in range(i + 1, min(n, i + 3))):
            polys[i] = polys[i - 1]
    slabs, i = [], 0
    while i < n:
        if polys[i] is None:
            i += 1
            continue
        j = i + 1
        while j < n and polys[j] is not None and iou(polys[j], polys[i]) > SAME and iou(polys[j], polys[j - 1]) > SAME:
            j += 1
        slabs.append([i, j])
        i = j
    small = max(2, int(MIN_SLAB * n))
    changed = True
    while changed and len(slabs) > 1:
        changed = False
        for k, (a, b) in enumerate(slabs):
            if b - a >= small:
                continue
            nb = [q for q in (k - 1, k + 1) if 0 <= q < len(slabs) and (slabs[q][1] == a or slabs[q][0] == b)]
            if not nb:
                continue
            mid = polys[(a + b) // 2]
            q = max(nb, key=lambda q: iou(polys[(slabs[q][0] + slabs[q][1]) // 2], mid))
            slabs[q] = [min(slabs[q][0], a), max(slabs[q][1], b)]
            del slabs[k]
            changed = True
            break
    return slabs


def representative(polys: list, a: int, b: int):
    """Сечението от средата на елемента, най-сходно с останалите (без прехода в краищата)."""
    idx = [k for k in range(a, b) if polys[k] is not None]
    inner = idx[len(idx) // 5 : max(len(idx) // 5 + 1, len(idx) - len(idx) // 5)]
    sample = inner[:: max(1, len(inner) // 9)]
    return max((polys[k] for k in sample), key=lambda g: sum(iou(g, polys[q]) for q in sample))


def robust_circle(ring, tol: float) -> dict | None:
    """Кръгът, на който лежи повечето от контура (вал с канал или скосяване): изрязаното не го измества."""
    pts = resample(ring)
    keep = np.ones(len(pts), bool)
    for _ in range(4):
        cx, cy, rad, _ = fit_circle(pts[keep])
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rad)
        keep = d < max(2 * tol, 0.02 * rad)
        if keep.sum() < 0.55 * len(pts):
            return None
    return {"kind": "circle", "c": [cx, cy], "r": rad}


def notched(poly: Polygon, tol: float) -> tuple[dict, list[dict]] | None:
    """Основна форма (кръг или правоъгълник) минус изрези от края (шпонков канал, лице на вал, прорез в плоча):
    (основата, изрезите), ако обяснява очертанието."""
    least = max((2 * tol) ** 2, 0.004 * poly.area)
    best = None
    rect = poly.minimum_rotated_rectangle
    for base in (robust_circle(poly.exterior, tol), shape_of(rect, tol, plain=True)):
        if base is None:
            continue
        bp = as_poly(base)
        if poly.difference(bp).area > 0.03 * poly.area:
            continue
        cuts = [c for c in parts_of(bp.difference(poly).buffer(-tol / 4).buffer(tol / 4)) if c.area > least]
        if not cuts or len(cuts) > 6:
            continue
        rebuilt = bp.difference(unary_union(cuts))
        fit = iou(rebuilt, poly)
        score = fit - 0.01 * len(cuts)  # по-малко изрези е по-близо до начина, по който е направен детайлът
        if fit > RECT_IOU - 0.01 and (best is None or score > best[0]):
            best = (score, base, [outward(notch_shape(c, tol), base, bp, tol) for c in cuts])
    return None if best is None else (best[1], best[2])


def notch_shape(cut: Polygon, tol: float) -> dict:
    """Изрез от края: правоъгълник и когато външната му страна следва извивката на основата."""
    s = shape_of(cut, tol, plain=True)
    if s["kind"] == "poly" and iou(cut, cut.minimum_rotated_rectangle) > 0.85:
        s = shape_of(cut.minimum_rotated_rectangle, tol, plain=True)
    return s


def outward(cut: dict, base: dict, bp: Polygon, tol: float) -> dict:
    """Правоъгълен изрез от края се продължава навън отвъд основата: без тънка кора по извивката ѝ."""
    if cut["kind"] != "rect":
        return cut
    a = math.radians(cut["angle"])
    ex, ey = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.array(cut["c"])
    d = c - np.array(bp.centroid.coords[0])
    m = 3 * tol + 0.15 * min(cut["w"], cut["h"])
    if abs(d @ ex) * cut["h"] > abs(d @ ey) * cut["w"]:  # навън по ширината
        c = c + ex * np.sign(d @ ex) * m / 2
        return dict(cut, c=list(c), w=cut["w"] + m)
    c = c + ey * np.sign(d @ ey) * m / 2
    return dict(cut, c=list(c), h=cut["h"] + m)


def shape_of(poly: Polygon, tol: float, plain: bool = False) -> dict:
    """Външното очертание като кръг, правоъгълник (по желание закръглен) или многоъгълник с изправени страни.
    plain: без основа с изрези (за самите изрези)."""
    c = circle_of(poly.exterior, tol)
    if c is not None:
        return c
    rect = poly.minimum_rotated_rectangle
    pts = np.array(rect.exterior.coords)[:4]
    e0, e1 = pts[1] - pts[0], pts[2] - pts[1]
    w, h = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    raw = math.degrees(math.atan2(e0[1], e0[0]))
    ang = (raw + 45) % 90 - 45  # в [-45, 45): ширината и височината се разменят при нужда
    if round((raw - ang) / 90) % 2:
        w, h = h, w
    if abs(ang) < 3:
        ang = 0.0
    centre = rect.centroid
    out = {"kind": "rect", "c": [centre.x, centre.y], "w": w, "h": h, "angle": ang}
    if iou(poly, rect) > RECT_IOU:
        return out
    deficit = rect.area - poly.area
    if deficit > 0 and iou(poly, rect) > 0.85:
        rf = math.sqrt(deficit / (4 - math.pi))
        if rf < 0.45 * min(w, h):
            rounded = rect.buffer(-rf).buffer(rf, quad_segs=16)
            if iou(poly, rounded) > RECT_IOU:
                return out | {"fillet": rf}
    if not plain:
        found = notched(poly, tol)
        if found is not None:
            return found[0] | {"notches": found[1]}
    return {"kind": "poly", "pts": straight(poly, tol)}


def straight(poly: Polygon, tol: float) -> list[list[float]]:
    p = poly.simplify(tol, preserve_topology=True)
    if not isinstance(p, Polygon) or p.is_empty:
        p = poly
    pts = np.array(p.exterior.coords)[:-1]
    for _ in range(2):  # почти хоризонтални и вертикални страни стават точно такива
        for k in range(len(pts)):
            a, b = pts[k], pts[(k + 1) % len(pts)]
            d = b - a
            ang = math.degrees(math.atan2(d[1], d[0])) % 180
            if min(ang, 180 - ang) < 4:
                pts[k][1] = pts[(k + 1) % len(pts)][1] = (a[1] + b[1]) / 2
            elif abs(ang - 90) < 4:
                pts[k][0] = pts[(k + 1) % len(pts)][0] = (a[0] + b[0]) / 2
    out = []
    for x, y in pts:
        q = [r(x), r(y)]
        if not out or q != out[-1]:
            out.append(q)
    return out[:-1] if len(out) > 1 and out[0] == out[-1] else out


def holes_of(poly: Polygon, tol: float) -> list[dict]:
    out = []
    for ring in poly.interiors:
        c = circle_of(ring, tol)
        out.append(c if c is not None else {"kind": "poly", "pts": straight(Polygon(ring), tol)})
    return out


def as_poly(s: dict) -> Polygon:
    if s["kind"] == "circle":
        t = np.linspace(0, 2 * np.pi, 97)[:-1]
        return Polygon(np.c_[s["c"][0] + s["r"] * np.cos(t), s["c"][1] + s["r"] * np.sin(t)])
    if s["kind"] == "rect":
        p = box(-s["w"] / 2, -s["h"] / 2, s["w"] / 2, s["h"] / 2)
        if s.get("fillet"):
            p = p.buffer(-s["fillet"]).buffer(s["fillet"], quad_segs=16)
        return affinity.translate(affinity.rotate(p, s["angle"], origin=(0, 0)), *s["c"])
    return Polygon(s["pts"]).buffer(0)


def same(a: dict, b: dict, tol: float) -> bool:
    if a["kind"] == b["kind"] == "circle":
        near = max(tol, 0.1 * a["r"])
        return math.dist(a["c"], b["c"]) < near and abs(a["r"] - b["r"]) < near
    return iou(as_poly(a), as_poly(b)) > SAME


def bores(runs: list[dict]) -> list[dict]:
    """Отворите по оста, слети по кръг: отвор, прекъснат от напречен отвор, е един отвор."""
    out: list[dict] = []
    for run in runs:
        for b in out:
            if same(b["shape"], run["shape"], 0.0):
                b["z"] = [min(b["z"][0], run["z"][0]), max(b["z"][1], run["z"][1])]
                break
        else:
            out.append({"shape": run["shape"], "z": list(run["z"])})
    return out


def to_global(axis: str, pts: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts)
    out[:, list(PLANES[axis][1])] = pts
    return out


def solid_at(mesh: trimesh.Trimesh, pts: np.ndarray) -> np.ndarray:
    try:
        return mesh.contains(pts)
    except Exception:  # без rtree/лъчи: приема се, че е материал (отворът не се продължава)
        return np.ones(len(pts), bool)


def through(mesh: trimesh.Trimesh, axis: str, cuts: list[dict], step: float) -> None:
    """Край на отвор, зад който е въздух, се продължава навън: без кожа с дебелина от части от милиметъра."""
    for c in cuts:
        cu, cv = c["shape"]["c"] if c["shape"]["kind"] == "circle" else as_poly(c["shape"]).representative_point().coords[0]
        probe = to_global(axis, np.array([[cu, cv, c["z"][0] - 1.5 * step], [cu, cv, c["z"][1] + 1.5 * step]]))
        below, above = ~solid_at(mesh, probe)
        c["depth"] = c["z"][1] - c["z"][0]
        c["thru"] = bool(below and above)
        c["z"] = [c["z"][0] - 3 * step * below, c["z"][1] + 3 * step * above]


def joined(mesh: trimesh.Trimesh, holes: list[dict], step: float) -> list[dict]:
    """Части от един отвор на една ос (кръгът е същият), между които е само въздух (през отвор по друга ос): един."""
    out: list[dict] = []
    for h in sorted(holes, key=lambda h: (h["axis"], h["z"][0])):
        for o in out:
            if o["axis"] != h["axis"] or not same(o["shape"], h["shape"], 0.0) or h["z"][0] < o["z"][1]:
                continue
            w = np.linspace(o["z"][1] + step, h["z"][0] - step, 8) if h["z"][0] - o["z"][1] > 2 * step else []
            pts = to_global(h["axis"], np.array([[*h["shape"]["c"], x] for x in w])) if len(w) else np.zeros((0, 3))
            if not len(pts) or not solid_at(mesh, pts).any():
                o["z"][1] = h["z"][1]
                break
        else:
            out.append({"axis": h["axis"], "shape": h["shape"], "z": list(h["z"])})
    return out


def merge_solids(solids: list[dict], tol: float) -> list[dict]:
    """Съседни по оста елементи с едно и също сечение (кръг със същия център и диаметър): един елемент."""
    out: list[dict] = []
    for s in sorted(solids, key=lambda s: s["z"][0]):
        for o in out:
            if abs(o["z"][1] - s["z"][0]) < 1e-6 and same(o["shape"], s["shape"], tol):
                total = (o["z"][1] - o["z"][0]) + (s["z"][1] - s["z"][0])
                if o["shape"]["kind"] == "circle":  # средното по височина
                    wa, wb = (o["z"][1] - o["z"][0]) / total, (s["z"][1] - s["z"][0]) / total
                    o["shape"] = dict(o["shape"], r=wa * o["shape"]["r"] + wb * s["shape"]["r"],
                                      c=[wa * a + wb * b for a, b in zip(o["shape"]["c"], s["shape"]["c"])])  # fmt: skip
                o["z"][1] = s["z"][1]
                break
        else:
            out.append({"shape": s["shape"], "z": list(s["z"])})
    return out


def analyse(mesh: trimesh.Trimesh, axis: str, secs: tuple, cross: list[dict], own: list[dict], tol: float,
            lm: trimesh.Trimesh) -> dict:  # fmt: skip
    """Елементите по `axis` от сеченията ѝ. Отворите по другите оси (`cross`) се запълват (продължени малко до стената
    на отвор по тази ос или до външната стена, но не в отворите по тази ос, `own`, и не извън очертанието)."""
    hs, raw = secs
    step = hs[1] - hs[0]
    lo, hi = hs[0] - step / 2, hs[-1] + step / 2
    margin = 2 * step + 2 * tol
    least = (MIN_HOLE * float(np.max(mesh.extents))) ** 2 / 2

    def filled(h, g):
        if g is not None and cross:
            fill = [f for f in (footprint(c, axis, h, margin) for c in cross) if f is not None]
            if fill:
                add = unary_union(fill).intersection(g.convex_hull)
                keep = [as_poly(b["shape"]) for b in own if b["z"][0] <= h <= b["z"][1]]
                if keep:
                    add = add.difference(unary_union(keep))
                g = clean(unary_union([g, add]).buffer(0), least)
        return g

    polys = [filled(h, g) for h, g in zip(hs, raw)]
    slabs = segment(polys)
    if slabs:
        slabs[0][0], slabs[-1][1] = 0, len(polys)  # до дъното и върха
    reps = [representative(polys, a, b) for a, b in slabs]
    zb = [lo + a * step for a, _ in slabs] + [lo + slabs[-1][1] * step] if slabs else []
    for k in range(1, len(slabs)):  # точното място на стъпалото между два елемента (по-фино от сеченията)
        if slabs[k - 1][1] != slabs[k][0]:
            continue
        A, B = reps[k - 1], reps[k]

        def side(h):
            g = filled(h, sections(lm, 0, tol, np.array([h]))[1][0])
            return iou(g, A) - iou(g, B)

        a_, b_ = zb[k] - 1.5 * step, zb[k] + 1.5 * step
        fa, fb = side(a_), side(b_)
        if fa > 0 > fb:
            for _ in range(6):
                mid = (a_ + b_) / 2
                if side(mid) > 0:
                    a_ = mid
                else:
                    b_ = mid
            zb[k] = (a_ + b_) / 2
    solids, cuts, err = [], [], 0.0
    for k, (a, b) in enumerate(slabs):
        rep = reps[k]
        z0, z1 = zb[k], zb[k + 1]
        built, made_cuts = [], []
        for part in parts_of(rep):
            sh = shape_of(Polygon(part.exterior), tol)
            notches = sh.pop("notches", [])
            built.append(as_poly(sh))
            solids.append({"shape": sh, "z": [z0, z1]})
            for notch in notches:
                made_cuts.append(as_poly(notch))
                cuts.append({"shape": notch, "z": [z0, z1]})
            for hole in holes_of(part, tol):
                made_cuts.append(as_poly(hole))
                cuts.append({"shape": hole, "z": [z0, z1]})
        fitted = unary_union([b_.buffer(0) for b_ in built]) if built else None
        if fitted is not None and made_cuts:
            fitted = fitted.difference(unary_union(made_cuts))
        for k in range(a, b):
            if polys[k] is not None:
                err += (polys[k].symmetric_difference(fitted).area if fitted is not None else polys[k].area) * step
    merged = []  # отвор през няколко елемента: един отвор
    for c in sorted(cuts, key=lambda c: c["z"][0]):
        for m_ in merged:
            if abs(m_["z"][1] - c["z"][0]) < 1.5 * step and same(m_["shape"], c["shape"], tol):
                m_["z"][1] = c["z"][1]
                break
        else:
            merged.append({"shape": c["shape"], "z": list(c["z"])})
    vol = abs(mesh.volume) if mesh.is_volume else float(np.prod(mesh.extents))
    solids = merge_solids(solids, tol)
    return {"axis": axis, "solids": solids, "cuts": merged, "error": err / vol, "slabs": len(solids), "step": step}


# --- Точни мерки ----------------------------------------------------------------------------------------------------


def snap_box(model: dict, bounds: np.ndarray, tol: float) -> None:
    """Стени на елементи до общите граници на детайла (по локалните u, v) стават точно на тях."""
    (ulo, vlo), (uhi, vhi) = bounds[0][:2], bounds[1][:2]
    near = 2.5 * tol

    def snap(x, lo, hi):
        return lo if abs(x - lo) < near else hi if abs(x - hi) < near else x

    for s in (item["shape"] for item in model["solids"]):
        if s["kind"] == "rect" and not s["angle"]:
            x0, x1 = snap(s["c"][0] - s["w"] / 2, ulo, uhi), snap(s["c"][0] + s["w"] / 2, ulo, uhi)
            y0, y1 = snap(s["c"][1] - s["h"] / 2, vlo, vhi), snap(s["c"][1] + s["h"] / 2, vlo, vhi)
            s.update(c=[(x0 + x1) / 2, (y0 + y1) / 2], w=x1 - x0, h=y1 - y0)
        elif s["kind"] == "circle":
            cx, cy, rad = s["c"][0], s["c"][1], s["r"]
            if abs(cx - rad - ulo) < near and abs(cx + rad - uhi) < near:
                s["c"][0], s["r"] = (ulo + uhi) / 2, (uhi - ulo) / 2
            if abs(cy - rad - vlo) < near and abs(cy + rad - vhi) < near:
                s["c"][1], s["r"] = (vlo + vhi) / 2, min(s["r"], (vhi - vlo) / 2) if s["c"][0] != cx else (vhi - vlo) / 2
        elif s["kind"] == "poly":
            s["pts"] = [[r(snap(x, ulo, uhi)), r(snap(y, vlo, vhi))] for x, y in s["pts"]]


def snap_features(model: dict, holes: list[dict], features: list[dict], notes: list[str]) -> None:
    """Измерените диаметри на отвори и издатини и дебелините на стъпала се налагат точно: всеки намерен кръг взима
    най-близката измерена стойност от своя вид (до SNAP)."""
    hits: dict[int, int] = {}
    for kind, pool in (("hole", [c["shape"] for c in model["cuts"] + holes]),
                       ("diameter", [s["shape"] for s in model["solids"]])):  # fmt: skip
        vals = [(k, float(f["value"])) for k, f in enumerate(features) if f.get("kind") == kind and f.get("value")]
        for s in pool:
            if s["kind"] != "circle" or not vals:
                continue
            k, v = min(vals, key=lambda kv: abs(2 * s["r"] - kv[1]))
            if abs(2 * s["r"] - v) <= SNAP * v:
                s["r"] = v / 2
                hits[k] = hits.get(k, 0) + 1
    for k, f in enumerate(features):
        v = float(f.get("value") or 0)
        if f.get("kind") != "thickness" or v <= 0:
            continue
        slabs = sorted({tuple(s["z"]) for s in model["solids"]})
        if not slabs:
            continue
        z0, z1 = min(slabs, key=lambda z: abs((z[1] - z[0]) - v))
        if abs((z1 - z0) - v) <= SNAP * v:
            # Границата на стъпалото се мести (общата височина е измерена и остава): съседният елемент поема разликата.
            d = v - (z1 - z0)
            for item in model["solids"] + model["cuts"]:
                for e in (0, 1):
                    if abs(item["z"][e] - z1) < 1e-6:
                        item["z"][e] = z1 + d
            hits[k] = 1
    names = {"hole": "отвор Ø", "diameter": "диаметър Ø", "thickness": "дебелина"}
    for k, f in enumerate(features):
        v = float(f.get("value") or 0)
        if v <= 0:
            continue
        label = f.get("label") or names.get(f.get("kind"), "мярка")
        if f.get("kind") not in names:
            notes.append(f"{label}: {v:g} мм (само в отчета)")
        elif k in hits:
            notes.append(f"{label} {v:g} мм: наложено точно ({hits[k]} {'елемент' if hits[k] == 1 else 'елемента'})")
        else:
            notes.append(f"{label} {v:g} мм: в сканирането няма такъв елемент (не е в модела)")


# --- Код на CadQuery ------------------------------------------------------------------------------------------------


def expr(axis: str, s: dict, z0: float, z1: float) -> str:
    plane = PLANES[axis][0]
    head = f'cq.Workplane("{plane}").workplane(offset={r(z0)})'
    h = r(z1 - z0)
    if s["kind"] == "circle":
        return f"{head}.center({r(s['c'][0])}, {r(s['c'][1])}).circle({r(s['r'])}).extrude({h})"
    if s["kind"] == "rect":
        rot = f".transformed(rotate=(0, 0, {round(s['angle'], 2)}))" if s["angle"] else ""
        body = f"{head}.center({r(s['c'][0])}, {r(s['c'][1])}){rot}.rect({r(s['w'])}, {r(s['h'])}).extrude({h})"
        if s.get("fillet"):
            body += f'.edges("{EDGE_SEL[axis]}").fillet({r(s["fillet"])})'
        return body
    pts = ", ".join(f"({x}, {y})" for x, y in s["pts"])
    return f"{head}.polyline([{pts}]).close().extrude({h})"


def describe(s: dict, z0: float, z1: float, cut: bool, info: dict | None = None) -> str:
    h = r((info or {}).get("depth") or (z1 - z0))
    if (info or {}).get("thru"):
        what = f"Ø{r(2 * s['r'])} мм" if s["kind"] == "circle" else f"с {len(s.get('pts', []))} страни"
        return f"{'Отвор' if s['kind'] == 'circle' else 'Прорез'} {what}, проходен ({h} мм)"
    if s["kind"] == "circle":
        return f"{'Отвор' if cut else 'Цилиндър'} Ø{r(2 * s['r'])} мм, {'дълбочина' if cut else 'височина'} {h} мм"
    if s["kind"] == "rect":
        extra = f", закръгляне R{r(s['fillet'])}" if s.get("fillet") else ""
        return f"{'Джоб' if cut else 'Блок'} {r(s['w'])} × {r(s['h'])} × {h} мм{extra}"
    return f"{'Джоб' if cut else 'Профил'} с {len(s['pts'])} страни, {'дълбочина' if cut else 'височина'} {h} мм"


def build(model: dict, holes: list[dict], header: str) -> tuple[str, object, list[str], list[str]]:
    """Изпълнява елементите един по един (неуспял се пропуска) и връща (кода, тялото, описанията, пропуснатите)."""
    import cadquery as cq

    axis = model["axis"]
    items = [("union", axis, s["shape"], *s["z"], s) for s in model["solids"]]
    items += [("cut", axis, c["shape"], *c["z"], c) for c in model["cuts"]]
    items += [("cut", h["axis"], h["shape"], *h["z"], h) for h in holes]
    lines, made, skipped, part = [header, "import cadquery as cq", ""], [], [], None
    for op, ax, s, z0, z1, info in items:
        if z1 - z0 < ROUND or (s["kind"] == "circle" and s["r"] < ROUND):
            continue
        e, text = expr(ax, s, z0, z1), describe(s, z0, z1, op == "cut", info)
        try:
            feature = eval(e, {"cq": cq})
        except Exception:
            if not s.get("fillet"):
                skipped.append(text)
                continue
            s = {k: v for k, v in s.items() if k != "fillet"}
            e, text = expr(ax, s, z0, z1), describe(s, z0, z1, op == "cut", info)
            feature = eval(e, {"cq": cq})
        try:
            if part is None and op == "cut":
                skipped.append(text)
                continue
            new = feature if part is None else part.union(feature) if op == "union" else part.cut(feature)
            if not new.val().isValid():
                raise ValueError("invalid solid")
        except Exception:
            skipped.append(text)
            continue
        lines.append(f"# {len(made) + 1}. {text}")
        lines.append(f"part = {e}" if part is None else f"part = part.{op}({e})")
        part = new
        made.append(text)
    if part is None:
        raise ValueError("не се получи нито един елемент")
    lines += ["", "result = part", ""]
    return "\n".join(lines), part, made, skipped


def deviation(scan: trimesh.Trimesh, cad: trimesh.Trimesh) -> dict:
    """Разстоянието между сканирането и модела (в двете посоки), мм."""
    from scipy.spatial import cKDTree

    a = trimesh.sample.sample_surface(scan, 30000, seed=0)[0]
    b = trimesh.sample.sample_surface(cad, 200000, seed=1)[0]
    d = np.r_[cKDTree(b).query(a)[0], cKDTree(a).query(trimesh.sample.sample_surface(cad, 30000, seed=2)[0])[0]]
    return {"mean": round(float(d.mean()), 3), "p90": round(float(np.percentile(d, 90)), 3)}


def run(scan_path: Path, meas: dict, out: Path) -> dict:
    import cadquery as cq

    out.mkdir(parents=True, exist_ok=True)
    mesh = oriented(trimesh.load(scan_path, force="mesh"))
    s, f, _, notes = scales(mesh.extents, meas)
    mesh.apply_scale(s)
    mesh.vertices = mesh.vertices * f
    mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])
    size = float(np.max(mesh.extents))
    tol = max(0.004 * size, 0.1)
    locals_ = {a: local(mesh, a) for a in PLANES}
    secs = {a: sections(locals_[a], SLICES, tol) for a in PLANES}
    runs = {a: hole_runs(a, *secs[a], tol, size) for a in PLANES}
    found = []
    for a in PLANES:
        cross = [h for b in PLANES if b != a for h in runs[b]]
        model = analyse(mesh, a, secs[a], cross, bores(runs[a]), tol, locals_[a])
        found.append((model["error"] + SCORE_SLAB * model["slabs"], model, cross))
    _, best, holes = min(found, key=lambda x: x[0])
    step = best["step"]
    holes = joined(mesh, holes, step)
    best["cuts"] = [{k: v for k, v in c.items() if k != "axis"}
                    for c in joined(mesh, [dict(c, axis=best["axis"]) for c in best["cuts"]], step)]  # fmt: skip
    through(mesh, best["axis"], best["cuts"], step)
    for h in holes:
        through(mesh, h["axis"], [h], step)
    snap_box(best, locals_[best["axis"]].bounds, tol)
    snap_features(best, holes, meas.get("features") or [], notes)
    L, W, H = (r(v) for v in mesh.extents)
    given = ", ".join(f"{k} {meas[k]:g}" for k in ("length", "width", "height") if meas.get(k))
    header = (f'"""Детайл {L} × {W} × {H} мм (Orbitize, инженерен режим, локално). Мерки: {given}. Размерите са в мм; '
              'всеки елемент е на свой ред и може да се промени."""')  # fmt: skip
    code, part, made, skipped = build(best, holes, header)
    (out / "part.py").write_text(code, encoding="utf-8")
    cq.exporters.export(part, str(out / "part.step"))
    cq.exporters.export(part, str(out / "part.stl"), tolerance=max(0.01, size / 2000), angularTolerance=0.1)
    dev = deviation(mesh, trimesh.load(out / "part.stl", force="mesh"))
    if skipped:
        notes.append("пропуснати елементи (не се построиха чисто): " + "; ".join(skipped))
    report = {"engine": "local", "axis": best["axis"], "size_mm": [L, W, H],
              "measured": {k: meas[k] for k in ("length", "width", "height") if meas.get(k)}, "elements": made,
              "deviation_mm": dev, "notes": notes}  # fmt: skip
    (out / "part.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    mesh.export(out / "scan_mm.stl")  # сканирането в същите координати (за сравнение и за Claude)
    return report


if __name__ == "__main__":
    rep = run(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), Path(sys.argv[3]))
    print(json.dumps(rep))
