"""Описанието на детайла и въпросите към потребителя.

Всеки размер е запис в `dims` (стойност в мм и откъде е: измерен с шублер, оценен от снимките или пропуснат от
потребителя, който тогава остава оценката). Геометрията никога не съдържа числа, а само препратки към размерите:

- Общите размери L (дължина, по X), W (ширина, по Y), H (височина, по Z). Детайл от струг ("turned") има L и D
  (най-големият диаметър).
- Нива: координата по една ос, зададена като разстояние от началото или от края на детайла по нея (от по-близкия
  край: така се мери с шублер) или "в средата". Особените нива: x0/xL, y0/yW, z0/zH и средите xc, yc, zc.
- Изгледи: очертание отгоре (X, Y), отпред (X, Z) и отстрани, от десния край (Y, Z) като многоъгълник от нива, с
  радиус по желание във всеки ъгъл, или кръг/елипса в общите размери ("round": изправен или легнал цилиндър).
  Тялото е сечението на трите изтеглени очертания. При детайл от струг: половината профил (X, радиус), завъртян
  около оста X.
- Елементи: отвори, правоъгълни изрези и джобове, канали, заобляния и фаски на ръбове, резба (бележка към отвор).

Координати: X наляво-надясно на снимката отгоре (0 е левият край), Y отпред назад (0 е предният ръб, долу на
снимката), Z отдолу нагоре (0 е масата)."""

import copy
import math
import re

from shapely.geometry import Polygon

LIMIT = 1000.0  # мм: най-големият размер
CENTER_TOL = 0.025  # елемент в рамките на 2,5% от средата е "в средата"
MERGE_TOL = 0.03  # нива от различни изгледи по-близо от 3% от размера по оста са едно и също

AXIS_TOTAL = {"x": "L", "y": "W", "z": "H"}
VIEW_AXES = {"top": ("x", "y"), "side": ("x", "z"), "end": ("y", "z"), "profile": ("x", "r")}
VIEW_NAME = {"top": "отгоре", "side": "отпред", "end": "отстрани", "profile": "отпред"}
FROM_TEXT = {
    "x": ("левия край", "десния край"),
    "y": ("предния ръб", "задния ръб"),
    "z": ("долната страна", "горната страна"),
}
FACES = {  # стената, от която започва елемент по оста си: (ос, страна)
    "top": ("z", "+"), "bottom": ("z", "-"), "front": ("y", "-"), "back": ("y", "+"), "left": ("x", "-"),
    "right": ("x", "+"),
}  # fmt: skip
FACE_NAME = {"top": "горната страна", "bottom": "долната страна", "front": "предната страна",
             "back": "задната страна", "left": "левия край", "right": "десния край"}  # fmt: skip
EDGE_SETS = {"vertical": "вертикалните ръбове", "top": "ръбовете отгоре", "bottom": "ръбовете отдолу",
             "all": "всички ръбове", "ends": "краищата", "flat": "ръбовете отгоре и отдолу"}  # fmt: skip
OVERALL = ("L", "W", "H", "D")  # общите размери: само те не могат да са 0


class DesignError(ValueError):
    """Описание, което не може да се построи: съобщението е за потребителя."""


# --- Размери и нива ------------------------------------------------------------------------------------------------


def new_id(d: dict, prefix: str) -> str:
    k = 1
    while f"{prefix}{k}" in d["dims"] or f"{prefix}{k}" in d["levels"]:
        k += 1
    return f"{prefix}{k}"


def add_dim(d: dict, prefix: str, est: float | None, name: str, ask: str, hint: str = "", kind: str = "length",
            optional: bool = False, show: dict | None = None) -> str:  # fmt: skip
    """Нов размер за измерване. est: оценката от снимките (None: няма; тогава е задължителен)."""
    i = new_id(d, prefix)
    d["dims"][i] = {"v": round(est, 2) if est else None, "est": round(est, 2) if est else None, "src": "photo",
                    "name": name, "ask": ask, "hint": hint, "kind": kind, "optional": optional and est is not None,
                    "show": show or {}}  # fmt: skip
    return i


def total(d: dict, axis: str) -> float:
    if axis == "r":
        return value(d, "D") / 2
    if d["kind"] == "turned" and axis in "yz":
        return value(d, "D")
    return value(d, AXIS_TOTAL[axis])


def value(d: dict, dim: str) -> float:
    v = d["dims"][dim]["v"]
    if v is None:
        raise DesignError(f"Липсва размерът „{d['dims'][dim]['name']}“.")
    return float(v)


def coord(d: dict, lvl: str) -> float:
    """Координатата на ниво в мм."""
    if lvl in d["levels"]:
        return level_coord(d, lvl)
    if lvl[1:] == "0":
        return 0.0
    if lvl[1:] in ("L", "W", "H", "D"):
        return total(d, lvl[0])
    if lvl[1:] == "c":
        return total(d, lvl[0]) / 2
    if lvl[1:2] == "~":  # точка от гладка форма: дял от общия размер по оста (мени се с него, не се пита)
        return total(d, lvl[0]) * float(lvl[2:])
    raise DesignError("Непознато ниво.")


def frac_level(axis: str, at: float, size: float) -> str:
    """Нивото на точка от гладка форма: дял от общия размер."""
    return f"{axis}~{min(max(at / size, 0.0), 1.0):.5f}"


def level_coord(d: dict, lvl: str) -> float:
    spec = d["levels"][lvl]
    v = value(d, spec["dim"])
    if spec["from"] == "diameter":
        return v / 2
    if spec["from"] == "end":
        return total(d, spec["axis"]) - v
    if spec["from"] == "center":  # отместване от средата (за детайли от струг: никога не се ползва засега)
        return total(d, spec["axis"]) / 2 + v
    return v


def end_level(axis: str) -> str:
    return {"x": "xL", "y": "yW", "z": "zH", "r": "rD"}[axis]


def level_for(d: dict, axis: str, at: float, name: str, what: str, allow_center: bool = False,
              view: str | None = None, merge: bool = True) -> str:
    """Ниво за координата `at` (оценка в мм): особено ниво, съществуващо близко ниво от същия вид или ново с размер
    за измерване от по-близкия край. name: името на размера ("Стъпало по дължината"); what: до какво се мери
    ("ръба на стъпалото"). merge=False: само свое ниво (положенията на отвори не се сливат със стъпала)."""
    size = total(d, axis)
    tol = MERGE_TOL * size
    if abs(at) <= tol:  # до 3% от ръба е ръбът (на снимка по-малко не се различава от наклон и шум)
        return f"{axis}0"
    if abs(at - size) <= tol:
        return end_level(axis)
    if allow_center and abs(at - size / 2) <= CENTER_TOL * size:
        return f"{axis}c"
    tag = "edge" if merge else "own"
    if merge:
        for lid, spec in d["levels"].items():
            if (spec["axis"] == axis and spec.get("tag") == "edge" and spec["from"] in ("start", "end")
                    and abs(coord(d, lid) - at) <= tol):  # fmt: skip
                return lid
    start, end = FROM_TEXT[axis]
    from_end = at > size / 2
    dist = size - at if from_end else at
    ref = end if from_end else start
    dim = add_dim(d, axis, dist, f"{name} (от {ref})", f"Разстояние от {ref} до {what}",
                  "Мерете с дълбокомера или с челюстите на шублера.", "position",
                  show={"view": view, "axis": axis, "from": "end" if from_end else "start"})  # fmt: skip
    lid = f"L{dim}"
    d["levels"][lid] = {"axis": axis, "dim": dim, "from": "end" if from_end else "start", "tag": tag}
    d["dims"][dim]["show"]["level"] = lid
    return lid


# --- Ново описание -------------------------------------------------------------------------------------------------


def new(kind: str, L: float, W: float, H: float) -> dict:
    """Празно описание с измерените общи размери (кутия)."""
    d = {"v": 1, "kind": kind, "dims": {}, "levels": {}, "views": {}, "features": [], "notes": []}
    if kind == "turned":
        D = round((W + H) / 2, 3)
        d["dims"]["L"] = overall("Дължина", L, "Цялата дължина по оста.")
        d["dims"]["D"] = overall("Най-голям диаметър", D, "Най-дебелото място.")
        d["views"]["profile"] = {"pts": [["x0", "r0", None], ["xL", "r0", None], ["xL", "rD", None],
                                         ["x0", "rD", None]]}  # fmt: skip
    else:
        d["dims"]["L"] = overall("Дължина", L, "Най-дългата страна, както лежи на масата.")
        d["dims"]["W"] = overall("Ширина", W, "Другата страна, гледано отгоре.")
        d["dims"]["H"] = overall("Височина", H, "От масата до най-високата точка.")
        for view, (a, b) in VIEW_AXES.items():
            if view != "profile":
                d["views"][view] = {"pts": box(a, b)}
    return d


def overall(name: str, v: float, hint: str) -> dict:
    return {"v": round(float(v), 3), "est": None, "src": "measured", "name": name, "ask": f"Измерете: {name.lower()}",
            "hint": hint, "kind": "length", "optional": False, "show": {"overall": True}}  # fmt: skip


def box(a: str, b: str) -> list:
    return [[f"{a}0", f"{b}0", None], [end_level(a), f"{b}0", None], [end_level(a), end_level(b), None],
            [f"{a}0", end_level(b), None]]  # fmt: skip


# --- Геометрията в мм ----------------------------------------------------------------------------------------------


def outline(d: dict, view: str) -> list[tuple[float, float, float]]:
    """Ъглите на очертанието на изгледа: (u, v, радиус) в мм."""
    out = []
    for u, v, r in d["views"][view]["pts"]:
        out.append((coord(d, u), coord(d, v), value(d, r) if r else 0.0))
    return out


def tidy(pts: list) -> list:
    """Ъглите без повторените (стъпало с мярка 0 изчезва) и без тези по права линия (там няма ъгъл)."""
    pts = [p for k, p in enumerate(pts) if math.dist(p[:2], pts[k - 1][:2]) > 1e-6] or pts[:1]
    out = list(pts)
    changed = True
    while changed and len(out) > 3:
        changed = False
        for k in range(len(out)):
            a, b, c = out[k - 1], out[k], out[(k + 1) % len(out)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) < 1e-9 * max(1.0, math.dist(a[:2], c[:2]) ** 2):
                out.pop(k)
                changed = True
                break
    return out


def zero(d: dict, dim) -> bool:
    """Размер, въведен като 0: елементът го няма (отвор, канал, заобляне, фаска, дълбочина)."""
    return isinstance(dim, str) and dim in d["dims"] and d["dims"][dim]["v"] is not None and float(
        d["dims"][dim]["v"]) <= 0  # fmt: skip


def present(d: dict, f: dict) -> bool:
    """Елементът се строи: никой от размерите му (без положенията) не е 0, а границата на части е вътре."""
    if any(zero(d, f.get(k)) for k in ("d", "w", "h", "len", "depth", "r", "c")):
        return False
    if f["t"] == "split":
        for key in ("at", "at2"):
            if f.get(key) is not None:
                x = coord(d, f[key])
                if not 1e-6 < x < total(d, "x") - 1e-6:
                    return False
    return True


def feature_place(d: dict, f: dict) -> dict:
    """Положението на елемент в мм: ос, страна, център (u, v) в равнината на стената и размерите му."""
    axis = f["axis"]
    out = {"axis": axis, "side": f.get("side", "+")}
    for key in ("u", "v"):
        p = f[key]
        c = coord(d, p["lvl"])
        size = f.get("w" if key == "u" else "h") or f.get("len" if key == "u" else "w")
        if f["t"] in ("rect", "slot") and p.get("anchor") in ("min", "max"):
            half = value(d, size) / 2 if size else 0.0
            c = c + half if p["anchor"] == "min" else c - half
        out[key] = c
    return out


def plane_axes(axis: str) -> tuple[str, str]:
    """Осите (u, v) на стена, перпендикулярна на `axis`."""
    return {"z": ("x", "y"), "y": ("x", "z"), "x": ("y", "z")}[axis]


# --- Проверка ------------------------------------------------------------------------------------------------------


def check(d: dict) -> None:
    """DesignError с обяснение, ако описанието не може да се построи."""
    for i, dim in d["dims"].items():
        v = dim["v"]
        if v is None:
            raise DesignError(f"Липсва размерът „{dim['name']}“.")
        if not (0 < float(v) if i in OVERALL else 0 <= float(v)) or float(v) > LIMIT:
            raise DesignError(f"„{dim['name']}“ трябва да е между 0 и {LIMIT:g} мм.")
    for view, spec in d["views"].items():
        if spec.get("off"):
            continue
        pts = tidy(outline(d, view))
        poly = Polygon([(u, v) for u, v, _ in pts])
        if not poly.is_valid or poly.area <= 0:
            names = sorted({d["dims"][d["levels"][p]["dim"]]["name"] for row in spec["pts"] for p in row[:2]
                            if p in d["levels"]})  # fmt: skip
            raise DesignError(f"Очертанието {VIEW_NAME[view]} се пресича само: проверете "
                              f"{', '.join(names) or 'общите размери'}.")  # fmt: skip
        a, b = VIEW_AXES[view]
        if spec.get("round"):
            continue
        for u, v, _ in pts:
            if not (-1e-6 <= u <= total(d, a) + 1e-6 and -1e-6 <= v <= total(d, b) + 1e-6):
                raise DesignError(f"Ниво {VIEW_NAME[view]} е извън общите размери: проверете разстоянията до ръбовете.")
        problem = radius_problem(d, view)
        if problem:
            raise DesignError(problem)
    for f in d["features"]:
        if f["t"] in ("hole", "rect", "slot") and present(d, f):
            place = feature_place(d, f)
            a, b = plane_axes(place["axis"])
            if not (0 <= place["u"] <= total(d, a) and 0 <= place["v"] <= total(d, b)):
                raise DesignError(f"{label(d, f)} е извън детайла: проверете положението му.")


RADIUS_SLACK = 0.15  # измерен радиус, по-голям до 15% от мястото си (полукръгъл край, радиусът се мери трудно):
# свива се, с бележка в отчета, без грешка
SHARP = 0.1  # мм: оценка за радиус, свита под толкова, е остър ъгъл


def corners(d: dict, view: str) -> list[tuple]:
    """Ъглите на изгледа без повторените и тези по права линия: (u, v, радиус, размерът на радиуса)."""
    return tidy([(coord(d, u), coord(d, v), value(d, r) if r else 0.0, r) for u, v, r in d["views"][view]["pts"]])


def mm(x: float) -> str:
    return f"{round(x, 2):g}".replace(".", ",")


def radius_problem(d: dict, view: str) -> str | None:
    """Защо заоблените ъгли на изгледа не стават (None: стават): двете дъги в краищата на една страна не се събират
    на нея или дъгите се застъпват (външният радиус на извивка, по-голям от вътрешния и стената: куката с вътрешен
    радиус 0,1 и външен 20 мм пресичаше сама себе си)."""
    pts = corners(d, view)
    n = len(pts)
    cut = []  # колко от двете страни на всеки ъгъл взима дъгата
    for k, (u, v, r, rid) in enumerate(pts):
        if r <= 0:
            cut.append(0.0)
            continue
        ang = corner_angle(pts[k - 1][:2], (u, v), pts[(k + 1) % n][:2])
        if ang <= 1e-3 or ang >= math.pi - 1e-3:
            return too_big(d, rid, 0.0, view)
        cut.append(r / math.tan(ang / 2))
    for k in range(n):
        k1 = (k + 1) % n
        side = math.dist(pts[k][:2], pts[k1][:2])
        if cut[k] + cut[k1] > side + 1e-6:
            j, other = (k, cut[k1]) if cut[k] >= cut[k1] else (k1, cut[k])
            if pts[k][3] == pts[k1][3]:  # един размер в двата края: двете дъги заедно
                room = pts[j][2] * side / (cut[k] + cut[k1])
            else:
                room = pts[j][2] * max(0.0, side - other) / cut[j]
            return too_big(d, pts[j][3], room, view)
    if any(p[2] > 0 for p in pts) and not Polygon(arc_points([p[:3] for p in pts])).is_valid:
        ids = sorted({p[3] for p in pts if p[2] > 0}, key=lambda i: -value(d, i))
        radii = ", ".join(mm(value(d, i)) for i in ids)
        return (f"Заоблените ъгли {VIEW_NAME[view]} се застъпват (радиуси {radii} мм): "
                "външният радиус на извивка е колкото вътрешния и дебелината на стената. Проверете ги.")  # fmt: skip
    return None


def too_big(d: dict, rid: str, room: float, view: str) -> str:
    name = d["dims"][rid]["name"] if rid in d["dims"] else "Радиусът"
    return (f"„{name}“ {mm(value(d, rid))} мм е твърде голям за ъгъла {VIEW_NAME[view]}: там се събират най-много "
            f"{mm(math.floor(room * 100) / 100)} мм.")  # fmt: skip


def arc_points(pts: list[tuple[float, float, float]], steps: int = 10) -> list[tuple[float, float]]:
    """Ъглите (u, v, r) -> начупена линия със заоблените ъгли като дъги (същото като outline() в part.py). Точка
    по-близо от 0,1 микрона до предишната се пропуска: две дъги, които се допират (полукръгъл край), не се застъпват."""
    out: list[tuple[float, float]] = []

    def put(p):
        if not out or math.dist(out[-1], p) > 1e-4:
            out.append(p)

    for k in range(len(pts)):
        u, v, r = pts[k]
        pu, pv, _ = pts[k - 1]
        nu, nv, _ = pts[(k + 1) % len(pts)]
        a, b = (pu - u, pv - v), (nu - u, nv - v)
        na, nb = math.hypot(*a), math.hypot(*b)
        if r <= 0 or na < 1e-9 or nb < 1e-9:
            put((u, v))
            continue
        a, b = (a[0] / na, a[1] / na), (b[0] / nb, b[1] / nb)
        ang = math.acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1])))
        if ang > math.pi - 1e-3 or ang < 1e-3:
            put((u, v))
            continue
        t = r / math.tan(ang / 2)
        bl, h = math.hypot(a[0] + b[0], a[1] + b[1]), r / math.sin(ang / 2)
        c = (u + (a[0] + b[0]) / bl * h, v + (a[1] + b[1]) / bl * h)
        p1, p2 = (u + a[0] * t, v + a[1] * t), (u + b[0] * t, v + b[1] * t)
        a1, a2 = math.atan2(p1[1] - c[1], p1[0] - c[0]), math.atan2(p2[1] - c[1], p2[0] - c[0])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for i in range(steps):
            s = i / (steps - 1)
            put((c[0] + r * math.cos(a1 + da * s), c[1] + r * math.sin(a1 + da * s)))
    if len(out) > 1 and math.dist(out[0], out[-1]) <= 1e-4:
        out.pop()
    return out


def fit_radii(d: dict) -> list[tuple[str, float, float]]:
    """Радиусите на ъглите, които не се събират, свити колкото е нужно (на място): първо оценките от снимката (от
    най-голямата: "не мога да го измеря" не бива да спира модела, щом мерките се различават от снимката), после
    измерените, но само до 15% (полукръгъл край: радиусът е половината ширина, а двете мерки се разминават).
    Радиус, измерен много по-голям, остава и check казва колко най-много става. Връща (размер, беше, стана)."""
    changed: list[tuple[str, float, float]] = []
    for view, spec in d["views"].items():
        if spec.get("off") or spec.get("round") or not spec.get("pts"):
            continue
        thin = thin_wall(d, view)
        if not radius_problem(d, view) and (thin is None or thin()):
            continue
        ids = sorted({row[2] for row in spec["pts"] if row[2] in d["dims"] and value(d, row[2]) > 0},
                     key=lambda i: -value(d, i))  # fmt: skip
        orig = {i: value(d, i) for i in ids}
        est = [i for i in ids if d["dims"][i]["src"] != "measured"]
        meas = [i for i in ids if d["dims"][i]["src"] == "measured"]

        def ok(view=view) -> bool:
            return radius_problem(d, view) is None

        def put(i: str, x: float) -> None:
            d["dims"][i]["v"] = math.floor(x * 1000) / 1000

        def grow(i: str, ok=ok, top=orig) -> None:  # най-голямата стойност до първоначалната, при която става
            lo, hi = 0.0, top[i]
            put(i, hi)
            if ok():
                return
            for _ in range(24):
                put(i, (lo + hi) / 2)
                lo, hi = (d["dims"][i]["v"], hi) if ok() else (lo, (lo + hi) / 2)
            put(i, lo)

        zeroed = []
        for i in est:
            if ok():
                break
            put(i, 0.0)
            if ok():
                grow(i)
            else:
                zeroed.append(i)
        if not ok() and meas:
            lo, hi = 1 - RADIUS_SLACK, 1.0
            for i in meas:
                put(i, orig[i] * lo)
            if ok():
                for _ in range(24):
                    mid = (lo + hi) / 2
                    for i in meas:
                        put(i, orig[i] * mid)
                    lo, hi = (mid, hi) if ok() else (lo, mid)
                for i in meas:
                    put(i, orig[i] * lo)
        if not ok():  # не става заради измерен радиус: той е в грешката (оценките на 0 не са виновни)
            for i in ids:
                d["dims"][i]["v"] = orig[i] if i in meas else 0.0
            continue
        for i in reversed(zeroed):
            grow(i)
        if est and thin and not thin():  # дъгите вече не се пресичат, но стената на извивката е изтъняла до нищо
            kept = {i: value(d, i) for i in est}
            for i in est:
                put(i, 0.0)
            if ok() and thin():
                for i in est:
                    grow(i, lambda: ok() and thin(), kept)
            else:
                for i in est:
                    d["dims"][i]["v"] = kept[i]
        for i in est:
            if 0 < value(d, i) < SHARP:
                put(i, 0.0)
        changed += [(i, orig[i], value(d, i)) for i in ids if value(d, i) < orig[i]]
    return changed


WALL_KEEP = 0.6  # след свиване стената на извивката е поне толкова от най-тънката стена на детайла по снимката


def pieces(poly, e: float) -> int:
    """На колко парчета се разпада очертанието, свито навътре с e (стена, по-тънка от 2e, го разделя)."""
    g = poly.buffer(-e) if e > 0 else poly
    return 0 if g.is_empty else len(getattr(g, "geoms", [g]))


def thin_wall(d: dict, view: str):
    """Проверка (функция без аргументи) дали очертанието няма стена, много по-тънка от тези по снимката: куката с
    измерен вътрешен радиус 0,1 и оценен външен 19 мм не се пресичаше, но стената в извивката беше 0,3 мм вместо
    5 и заоблянето на ръбовете не ставаше. Най-тънката стена на изгледа с оценките от снимката (там очертанието е
    цяло) е мярката. None: няма с какво да се сравни."""
    photo = copy.deepcopy(d)
    for dim in photo["dims"].values():
        if dim["est"] is not None:
            dim["v"] = dim["est"]
    try:
        if radius_problem(photo, view):
            return None
        poly = Polygon(arc_points([p[:3] for p in corners(photo, view)]))
    except (KeyError, TypeError, ValueError):
        return None
    if not poly.is_valid or poly.area <= 0 or pieces(poly, 0.0) != 1:
        return None
    lo, hi = 0.0, min(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) / 2
    for _ in range(16):  # най-голямото e, при което е още едно парче: половината от най-тънката стена
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pieces(poly, mid) == 1 else (lo, mid)
    e = max(WALL_KEEP * lo, edge_round(d) * 1.25)  # заоблянето на ръбовете от двете страни иска стена над 2r
    if e <= 0 or e >= lo:  # и по снимката не става: няма какво да се пази
        return None

    def thick() -> bool:
        p = Polygon(arc_points([q[:3] for q in corners(d, view)]))
        return p.is_valid and pieces(p, e) == 1

    return thick


def edge_round(d: dict) -> float:
    """Най-голямото заобляне или фаска на ръбовете (отгоре и отдолу: до половината дебелина, както в part.py)."""
    out = 0.0
    for f in d["features"]:
        key = f.get("r" if f["t"] == "fillet" else "c")
        if f["t"] in ("fillet", "chamfer") and key in d["dims"] and d["dims"][key]["v"] is not None and present(d, f):
            r = value(d, key)
            out = max(out, min(r, 0.49 * total(d, "z")) if f["edges"] == "flat" else r)
    return out


def fit_notes(d: dict, changed: list[tuple[str, float, float]]) -> list[str]:
    """Бележките за свитите радиуси в отчета на детайла."""
    out = []
    for i, was, now in changed:
        what = "измерено" if d["dims"][i]["src"] == "measured" else "оценка"
        out.append(f"„{d['dims'][i]['name']}“: {what} {mm(was)} мм, построено {mm(now)} мм, за да се съберат дъгите.")
    return out


def corner_angle(p, c, n) -> float:
    a = (p[0] - c[0], p[1] - c[1])
    b = (n[0] - c[0], n[1] - c[1])
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb))))


# --- Описание с думи -----------------------------------------------------------------------------------------------


def label(d: dict, f: dict) -> str:
    t = f["t"]
    if t == "hole":
        return "Отвор" + (" (резба " + f["thread"] + ")" if f.get("thread") else "")
    if t == "rect":
        return "Правоъгълен " + ("изрез" if f.get("depth") is None else "джоб")
    if t == "slot":
        return "Канал"
    if t == "fillet":
        return f"Заобляне на {EDGE_SETS[f['edges']]}"
    if t == "chamfer":
        return f"Фаска на {EDGE_SETS[f['edges']]}"
    if t == "split":
        return "Граница между две части (например капачка и тяло)" + (", по диагонал по страните" if f.get("at2") else "")
    return t


def describe(d: dict) -> list[dict]:
    """Какво е разпознато, за екрана "Какво виждам": [{id, text, view, kind}] по реда на модела."""
    out = []
    if d["kind"] == "turned":
        pts = d["views"]["profile"]["pts"]
        diam = len({p[1] for p in pts if p[1] not in ("r0",)})
        out.append({"id": "body", "view": "profile", "kind": "body",
                    "text": f"Тяло от струг: {diam} диаметър{'а' if diam > 1 else ''} по дължината"})  # fmt: skip
    if d.get("flat"):
        out.append({"id": "body", "view": "top", "kind": "body",
                    "text": "Плосък детайл: очертанието отгоре с една дебелина навсякъде"})  # fmt: skip
    for view in ("top", "side", "end"):
        spec = d["views"].get(view)
        if not spec or spec.get("off") or (d.get("flat") and view != "top"):
            continue
        pts = spec["pts"]
        rounded = sum(1 for p in pts if p[2])
        plain = len(pts) == 4 and all(p[0][1:] in "0LWH" and p[1][1:] in "0LWH" for p in pts)
        if spec.get("round"):
            text = "кръг" if abs(total(d, VIEW_AXES[view][0]) - total(d, VIEW_AXES[view][1])) < 1e-6 else "елипса"
        elif spec.get("curve"):
            sharp = len(spec.get("corners") or [])
            text = "гладка форма по снимката" + (f", {sharp} остри ъгъла" if sharp else "")
        elif spec.get("free"):
            text = "форма по снимката (оценена)"
        elif plain:
            text = "правоъгълник"
        else:
            text = f"очертание с {len(pts)} ъгъла (стъпала или изрези)"
        if rounded:
            text += f", {rounded} заоблен{'и' if rounded > 1 else ''} ъг{'ъла' if rounded > 1 else 'ъл'}"
        out.append({"id": f"view:{view}", "view": view, "kind": "view", "text": f"Отгоре: {text}" if view == "top"
                    else f"{VIEW_NAME[view].capitalize()}: {text}"})  # fmt: skip
    for f in d["features"]:
        where = f" в {FACE_NAME[f['face']]}" if f.get("face") else ""
        out.append({"id": f["id"], "view": f.get("view"), "kind": f["t"],
                    "text": label(d, f) + where + (" (проходен)" if f["t"] in ("hole", "rect", "slot")
                                                    and f.get("depth") is None else "")})  # fmt: skip
    return out


# --- Въпросите -----------------------------------------------------------------------------------------------------


def used_dims(d: dict) -> list[str]:
    """Размерите, които моделът използва, в реда на въпросите: общите, нивата по изгледи, елементите, радиусите."""
    order: list[str] = []

    def take(i):
        if i and i in d["dims"] and i not in order:
            order.append(i)

    for i in ("L", "W", "H", "D"):
        take(i)
    radii = []
    for view in ("top", "side", "end", "profile"):
        spec = d["views"].get(view)
        if not spec or spec.get("off"):
            continue
        for row in spec["pts"]:
            for lvl in row[:2]:
                if lvl in d["levels"]:
                    take(d["levels"][lvl]["dim"])
            radii.append(row[2])
    for f in d["features"]:
        for key in ("d", "w", "h", "len", "depth", "r", "c"):
            if isinstance(f.get(key), str):
                take(f[key])
        for key in ("u", "v"):
            p = f.get(key)
            if p and p["lvl"] in d["levels"]:
                take(d["levels"][p["lvl"]]["dim"])
        for key in ("at", "at2"):
            if f.get(key) in d["levels"]:
                take(d["levels"][f[key]]["dim"])
    for r in radii:
        take(r)
    return order


def questions(d: dict) -> list[dict]:
    out = []
    for i in used_dims(d):
        dim = d["dims"][i]
        if dim.get("hidden"):
            continue
        out.append({"id": i, "name": dim["name"], "ask": dim["ask"], "hint": dim["hint"], "est": dim["est"],
                    "value": dim["v"], "src": dim["src"], "optional": dim["optional"], "kind": dim["kind"]})  # fmt: skip
    return out


def unused_cleanup(d: dict) -> None:
    """Маха размерите и нивата, които вече нищо не използва (след премахване на елемент)."""
    keep = set(used_dims(d))
    for lvl in list(d["levels"]):
        if d["levels"][lvl]["dim"] not in keep:
            del d["levels"][lvl]
    for i in list(d["dims"]):
        if i not in keep:
            del d["dims"][i]


# --- Промени от приложението ---------------------------------------------------------------------------------------

DIM_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


def set_values(d: dict, values: dict) -> dict:
    """values: {размер: число (измерено) или None (пропуснат: остава оценката)}."""
    d = copy.deepcopy(d)
    for i, v in values.items():
        if i not in d["dims"]:
            continue
        dim = d["dims"][i]
        if v is None:
            if dim["est"] is None:
                raise DesignError(f"„{dim['name']}“ трябва да се измери: няма оценка от снимките.")
            dim["v"], dim["src"] = dim["est"], "skipped"
        else:
            v = float(v)
            if not (0 < v if i in OVERALL else 0 <= v) or v > LIMIT:
                raise DesignError(f"„{dim['name']}“ трябва да е между 0 и {LIMIT:g} мм.")
            dim["v"], dim["src"] = round(v, 3), "measured"
    return d


def remove(d: dict, item: str) -> dict:
    """Маха елемент или опростява изглед до правоъгълник ("view:side")."""
    d = copy.deepcopy(d)
    if item.startswith("view:"):
        view = item[5:]
        if view in d["views"] and view != "profile":
            a, b = VIEW_AXES[view]
            d["views"][view] = {"pts": box(a, b)}  # и кръгъл изглед става правоъгълник
    else:
        d["features"] = [f for f in d["features"] if f["id"] != item]
    unused_cleanup(d)
    return d


def add_feature(d: dict, spec: dict) -> dict:
    """Нов елемент от менюто "+ Добави". spec: {t, face, through, centered: [bool, bool], edges, thread}. Размерите
    му са нови въпроси без оценка (затова задължителни)."""
    d = copy.deepcopy(d)
    t = spec.get("t")
    fid = new_feature_id(d, t)
    if t in ("fillet", "chamfer"):
        edges = spec.get("edges") or "vertical"
        if edges not in EDGE_SETS:
            raise DesignError("Неизвестни ръбове.")
        key = "r" if t == "fillet" else "c"
        name = "Радиус на заоблянето" if t == "fillet" else "Размер на фаската"
        dim = add_dim(d, key, None, name, f"Измерете: {name.lower()} ({EDGE_SETS[edges]})",
                      "Радиусомер или на око с шублера." if t == "fillet" else "Катетът на фаската.",
                      show={"feature": fid})  # fmt: skip
        d["features"].append({"id": fid, "t": t, "edges": edges, key: dim})
        return d
    face = spec.get("face") or "top"
    if face not in FACES:
        raise DesignError("Неизвестна стена.")
    axis, side = FACES[face]
    if d["kind"] == "turned" and axis in ("y", "z"):
        axis, side = "y", "+"  # напречен отвор през оста
    a, b = plane_axes(axis)
    through = bool(spec.get("through", True))
    centered = spec.get("centered") or [True, True]
    f = {"id": fid, "t": t, "axis": axis, "side": side, "face": face}
    what = {"hole": "центъра на отвора", "rect": "ръба на изреза", "slot": "центъра на канала"}.get(t)
    if what is None:
        raise DesignError("Неизвестен елемент.")
    if t == "hole":
        f["d"] = add_dim(d, "d", None, "Диаметър на отвора", "Измерете диаметъра на отвора",
                         "С горните (вътрешните) челюсти на шублера.", "diameter", show={"feature": fid})  # fmt: skip
        if spec.get("thread"):
            f["thread"] = str(spec["thread"])[:12]
    elif t == "rect":
        f["w"] = add_dim(d, "w", None, "Ширина на изреза", "Измерете изреза по " + axis_word(a),
                         show={"feature": fid, "along": a})  # fmt: skip
        f["h"] = add_dim(d, "h", None, "Височина на изреза", "Измерете изреза по " + axis_word(b),
                         show={"feature": fid, "along": b})  # fmt: skip
    else:
        f["len"] = add_dim(d, "s", None, "Дължина на канала", "Измерете канала по дължината му (с краищата)",
                           show={"feature": fid})  # fmt: skip
        f["w"] = add_dim(d, "s", None, "Ширина на канала", "Измерете ширината на канала", show={"feature": fid})
        f["angle"] = 0 if spec.get("along", a) == a else 90
    for key, ax, c in (("u", a, centered[0]), ("v", b, centered[1])):
        if c:
            f[key] = {"lvl": f"{ax}c", "anchor": "center"}
        else:
            start = FROM_TEXT[ax][0]
            dim = add_dim(d, ax, None, f"Положение ({start})", f"Разстояние от {start} до {what}",
                          show={"feature": fid, "axis": ax, "from": "start"})  # fmt: skip
            lid = f"L{dim}"
            d["levels"][lid] = {"axis": ax, "dim": dim, "from": "start"}
            d["dims"][dim]["show"]["level"] = lid
            f[key] = {"lvl": lid, "anchor": "min" if t == "rect" else "center"}
    if not through:
        f["depth"] = add_dim(d, "t", None, "Дълбочина", f"Измерете дълбочината от {FACE_NAME[face]}",
                             "С дълбокомера на шублера.", show={"feature": fid})  # fmt: skip
    d["features"].append(f)
    return d


SPLIT_SLANT = 0.6  # мм: граница, по-разместена от това отгоре и отдолу, минава по диагонал по страните


def add_split(d: dict, x: float, x_bottom: float | None = None) -> dict:
    """Граница между две части по дължината (капачка и тяло, два материала): моделът става две тела, разделени при
    x (оценка в мм) на горната страна и при x_bottom на долната; ако се различават, по страните границата минава
    по диагонал. Мери се от по-близкия край."""
    d = copy.deepcopy(d)
    fid = new_feature_id(d, "split")
    slanted = x_bottom is not None and abs(x_bottom - x) > SPLIT_SLANT
    name = "Граница между частите" + (" отгоре" if slanted else "")
    lvl = level_for(d, "x", x, name, "границата между частите" + (" на горната страна" if slanted else ""),
                    view="top", merge=False)  # fmt: skip
    if lvl in ("x0", end_level("x")):
        return d
    d["dims"][d["levels"][lvl]["dim"]]["show"]["feature"] = fid
    f = {"id": fid, "t": "split", "axis": "x", "at": lvl, "view": "top"}
    if slanted:
        low = level_for(d, "x", x_bottom, "Граница между частите отдолу", "границата между частите на долната страна",
                        view="side", merge=False)  # fmt: skip
        if low not in ("x0", end_level("x")):
            d["dims"][d["levels"][low]["dim"]]["show"]["feature"] = fid
            f["at2"] = low
    d["features"].append(f)
    return d


def axis_word(a: str) -> str:
    return {"x": "дължината", "y": "ширината", "z": "височината"}[a]


def new_feature_id(d: dict, t: str) -> str:
    k = 1
    while any(f["id"] == f"{t}{k}" for f in d["features"]):
        k += 1
    return f"{t}{k}"


def summary(d: dict) -> dict:
    """Измерени и оценени размери (за отчета и чертежа)."""
    measured = [i for i in used_dims(d) if d["dims"][i]["src"] == "measured"]
    estimated = [i for i in used_dims(d) if d["dims"][i]["src"] != "measured"]
    return {"measured": measured, "estimated": estimated}
