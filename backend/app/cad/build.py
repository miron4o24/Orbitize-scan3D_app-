"""Описание на детайла -> кодът на CadQuery -> твърдо тяло -> файловете.

Кодът (part.py) е това, което сървърът изпълнява, затова STEP файлът и кодът са винаги едно и също: всеки размер е
именувана променлива най-горе, а геометрията е изразена чрез тях (промяна на число и ново изпълнение дава новия
детайл). Описанието идва от приложението, затова първо се проверява изцяло (clean): в кода влизат само имена по
DIM_ID, числа и изброени стойности, никога текст от заявката (освен в коментари, без нови редове)."""

import json
import keyword
import re
from pathlib import Path

from . import design as D

RESERVED = {"cq", "math", "plane", "outline", "cut_tool", "rounded_rect", "round_view", "body", "result", "top",
            "side", "end", "profile", "top_solid", "side_solid", "end_solid", "lambda", "None", "True", "False"}
FEATURE_KEYS = {"hole": ("d", "depth"), "rect": ("w", "h", "depth", "r"), "slot": ("len", "w", "depth"),
                "fillet": ("r",), "chamfer": ("c",), "split": ()}  # fmt: skip


# --- Проверка на описанието от приложението ------------------------------------------------------------------------


def _txt(s, n=120) -> str:
    return re.sub(r"[\r\n\t]+", " ", str(s or ""))[:n]


def clean(raw: dict) -> dict:
    """Описанието, проверено по форма (имена, видове, препратки). DesignError при нещо непознато."""
    if not isinstance(raw, dict) or raw.get("kind") not in ("block", "turned"):
        raise D.DesignError("Непознато описание на детайла.")
    d = {"v": 1, "kind": raw["kind"], "dims": {}, "levels": {}, "views": {}, "features": [],
         "notes": [_txt(n, 300) for n in (raw.get("notes") or [])[:20]]}  # fmt: skip
    if raw.get("flat"):
        d["flat"] = True
    if len(raw.get("dims") or {}) > 400 or len(raw.get("levels") or {}) > 400:
        raise D.DesignError("Твърде много размери.")
    for i, dim in (raw.get("dims") or {}).items():
        if not D.DIM_ID.match(i) or i in RESERVED or keyword.iskeyword(i) or not isinstance(dim, dict):
            raise D.DesignError("Непознат размер.")
        v, est = dim.get("v"), dim.get("est")
        d["dims"][i] = {"v": None if v is None else float(v), "est": None if est is None else float(est),
                        "src": dim.get("src") if dim.get("src") in ("measured", "photo", "skipped") else "photo",
                        "name": _txt(dim.get("name"), 60), "ask": _txt(dim.get("ask")), "hint": _txt(dim.get("hint")),
                        "kind": dim.get("kind") if dim.get("kind") in ("length", "position", "diameter", "radius")
                        else "length", "optional": bool(dim.get("optional")),
                        "show": dim.get("show") if isinstance(dim.get("show"), dict) else {},
                        **({"hidden": True} if dim.get("hidden") else {})}  # fmt: skip
    for k in ("L", "D") if raw["kind"] == "turned" else ("L", "W", "H"):
        if k not in d["dims"]:
            raise D.DesignError("Липсват общите размери.")
    for lid, spec in (raw.get("levels") or {}).items():
        if (not D.DIM_ID.match(lid) or lid in RESERVED or keyword.iskeyword(lid) or lid in d["dims"] or not isinstance(spec, dict) or spec.get("axis") not in ("x", "y", "z", "r")
                or spec.get("dim") not in d["dims"] or spec.get("from") not in ("start", "end", "diameter")):
            raise D.DesignError("Непознато ниво.")  # fmt: skip
        d["levels"][lid] = {"axis": spec["axis"], "dim": spec["dim"], "from": spec["from"],
                            "tag": "edge" if spec.get("tag") == "edge" else "own"}
    allowed_views = ("profile",) if raw["kind"] == "turned" else ("top", "side", "end")
    for view, spec in (raw.get("views") or {}).items():
        if view not in allowed_views or not isinstance(spec, dict):
            raise D.DesignError("Непознат изглед.")
        a, b = D.VIEW_AXES[view]
        pts = []
        for row in spec.get("pts") or []:
            u, v, r = (list(row) + [None])[:3]
            if not (level_ok(d, u, a) and level_ok(d, v, b) and (r is None or r in d["dims"])):
                raise D.DesignError("Непознат ъгъл на очертание.")
            pts.append([u, v, r])
        if not 3 <= len(pts) <= 200:
            raise D.DesignError("Очертанието трябва да има поне 3 ъгъла.")
        d["views"][view] = {"pts": pts, **({"free": True} if spec.get("free") else {}),
                            **({"round": True} if spec.get("round") and view != "profile" else {})}  # fmt: skip
        if spec.get("curve") and view != "profile":
            corners = spec.get("corners") or []
            if not isinstance(corners, list) or not all(isinstance(i, int) and 0 <= i < len(pts) for i in corners):
                raise D.DesignError("Непознат ъгъл на очертание.")
            d["views"][view] |= {"curve": True, "free": True, "corners": sorted(set(corners))}
    for f in (raw.get("features") or [])[:60]:
        t = f.get("t") if isinstance(f, dict) else None
        if t not in FEATURE_KEYS or not D.DIM_ID.match(str(f.get("id", ""))):
            raise D.DesignError("Непознат елемент.")
        g = {"id": f["id"], "t": t}
        for key in FEATURE_KEYS[t]:
            if f.get(key) is not None:
                if f[key] not in d["dims"]:
                    raise D.DesignError("Елемент с непознат размер.")
                g[key] = f[key]
        if t == "split":
            if not level_ok(d, f.get("at"), "x"):
                raise D.DesignError("Непозната граница между части.")
            g.update({"axis": "x", "at": f["at"], "view": "top"})
            if f.get("at2") is not None:
                if not level_ok(d, f["at2"], "x"):
                    raise D.DesignError("Непозната граница между части.")
                g["at2"] = f["at2"]
            d["features"].append(g)
            continue
        if t in ("fillet", "chamfer"):
            if f.get("edges") not in D.EDGE_SETS:
                raise D.DesignError("Непознати ръбове.")
            g["edges"] = f["edges"]
            if ("r" if t == "fillet" else "c") not in g:
                raise D.DesignError("Липсва размер на заобляне или фаска.")
        else:
            if f.get("axis") not in ("x", "y", "z") or f.get("side") not in ("+", "-"):
                raise D.DesignError("Непозната посока на елемент.")
            g["axis"], g["side"] = f["axis"], f["side"]
            if f.get("face") in D.FACES:
                g["face"] = f["face"]
            if f.get("view") in D.VIEW_AXES:
                g["view"] = f["view"]
            a, b = D.plane_axes(g["axis"])
            for key, ax in (("u", a), ("v", b)):
                p = f.get(key) or {}
                if not level_ok(d, p.get("lvl"), ax) or p.get("anchor") not in ("min", "max", "center"):
                    raise D.DesignError("Непознато положение на елемент.")
                g[key] = {"lvl": p["lvl"], "anchor": p["anchor"]}
            need = {"hole": ("d",), "rect": ("w", "h"), "slot": ("len", "w")}[t]
            if any(k not in g for k in need):
                raise D.DesignError("Липсва размер на елемент.")
            if t == "slot":
                g["angle"] = 90 if f.get("angle") == 90 else 0
            if t == "hole" and f.get("thread"):
                g["thread"] = re.sub(r"[^A-Za-z0-9.,x×/ -]", "", str(f["thread"]))[:12]
        d["features"].append(g)
    return d


FRAC = re.compile(r"^[xyz]~(0(\.\d{1,6})?|1(\.0{1,6})?)$")


def level_ok(d: dict, lvl, axis: str) -> bool:
    if not isinstance(lvl, str):
        return False
    if lvl in (f"{axis}0", f"{axis}c", D.end_level(axis)):
        return True
    if FRAC.match(lvl):
        return lvl[0] == axis
    return lvl in d["levels"] and d["levels"][lvl]["axis"] == axis


# --- Кодът ---------------------------------------------------------------------------------------------------------

HELPERS = '''
def plane(axis, at):
    """Равнина, перпендикулярна на оста `axis` при координата `at`, с изтегляне в положителна посока по оста
    (по Y: от `at` надолу към по-малки Y). Локалните оси: XY за Z, XZ за Y, YZ за X."""
    if axis == "z":
        return cq.Plane((0, 0, at), (1, 0, 0), (0, 0, 1))
    if axis == "y":
        return cq.Plane((0, at, 0), (1, 0, 0), (0, -1, 0))
    return cq.Plane((at, 0, 0), (0, 1, 0), (1, 0, 0))


def tidy(pts):
    """Без повторените ъгли (размер 0: стъпалото го няма) и без ъглите по права линия."""
    pts = [p for k, p in enumerate(pts) if math.dist(p[:2], pts[k - 1][:2]) > 1e-6] or pts[:1]
    k = 0
    while len(pts) > 3 and k < len(pts):
        a, b, c = pts[k - 1], pts[k], pts[(k + 1) % len(pts)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9 * max(1.0, math.dist(a[:2], c[:2]) ** 2):
            pts.pop(k)
            k = 0
        else:
            k += 1
    return pts


def outline(pl, pts):
    """Затворено очертание от ъгли (u, v, радиус) в равнината `pl`; радиус > 0 заобля ъгъла с дъга, допирателна
    към двете страни."""
    pts = tidy(list(pts))
    n = len(pts)

    def corner(k):
        u, v, r = pts[k]
        if r <= 0:
            return (u, v), None, (u, v)
        pu, pv, _ = pts[k - 1]
        nu, nv, _ = pts[(k + 1) % n]
        a, b = (pu - u, pv - v), (nu - u, nv - v)
        la, lb = math.hypot(*a), math.hypot(*b)
        a, b = (a[0] / la, a[1] / la), (b[0] / lb, b[1] / lb)
        ang = math.acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1])))
        t = r / math.tan(ang / 2)
        bis = (a[0] + b[0], a[1] + b[1])
        lbis = math.hypot(*bis)
        c = r / math.sin(ang / 2)
        mid = (u + bis[0] / lbis * (c - r), v + bis[1] / lbis * (c - r))
        return (u + a[0] * t, v + a[1] * t), mid, (u + b[0] * t, v + b[1] * t)

    cs = [corner(k) for k in range(n)]
    wp = cq.Workplane(pl).moveTo(*cs[0][2])
    here = cs[0][2]
    for k in list(range(1, n)) + [0]:
        p1, mid, p2 = cs[k]
        if math.dist(here, p1) > 1e-7:
            wp = wp.lineTo(*p1)
        if mid is not None:
            wp = wp.threePointArc(mid, p2)
        here = p2
    return wp.close()


def curve(pl, pts, corners):
    """Затворена гладка форма през точките (u, v) в равнината `pl`: сплайн, който се чупи само в острите ъгли
    (индексите `corners`); между два съседни ъгъла без точки помежду им е права."""
    n = len(pts)
    xy = [(u, v) for u, v, *_ in pts]
    if not corners:
        return cq.Workplane(pl).spline(xy, periodic=True, includeCurrent=False).close()
    wp = cq.Workplane(pl).moveTo(*xy[corners[0]])
    for a, b in zip(corners, corners[1:] + [corners[0] + n]):
        run = [xy[k % n] for k in range(a + 1, b + 1)]
        wp = wp.lineTo(*run[0]) if len(run) == 1 else wp.spline(run, includeCurrent=True)
    return wp.close()


def cut_tool(axis, side, u, v, size, depth, sketch):
    """Тяло за изрязване по оста `axis` в (u, v) на стената: през целия детайл (depth None) или до `depth` от
    страната `side` ("+" горе/отзад/вдясно). `sketch(wp)` рисува сечението около (0, 0)."""
    if depth is None:
        lo, hi = -1.0, size + 1.0
    elif side == "+":
        lo, hi = size - depth, size + 1.0
    else:
        lo, hi = -1.0, depth
    at = hi if axis == "y" else lo
    return sketch(cq.Workplane(plane(axis, at)).center(u, v)).extrude(hi - lo)


def round_view(pl, u, v):
    """Кръг или елипса, вписани в правоъгълника u x v на равнината `pl` (от 0 до u и до v)."""
    wp = cq.Workplane(pl).center(u / 2, v / 2)
    if abs(u - v) < 1e-6:
        return wp.circle(u / 2)
    return wp.ellipse(u / 2, v / 2)


def rounded_rect(w, h, r):
    def draw(wp):
        if r and r > 0:
            return wp.placeSketch(cq.Sketch().rect(w, h).vertices().fillet(min(r, w / 2 - 1e-3, h / 2 - 1e-3)))
        return wp.rect(w, h)
    return draw
'''


def num(v: float) -> str:
    return repr(round(float(v), 4))


def code(d: dict) -> str:
    """part.py за описанието (вече проверено от clean и check)."""
    lines = ['"""CAD модел от Orbitize (инженерен режим): размерите са в мм.', ""]
    est = [i for i in D.used_dims(d) if d["dims"][i]["src"] != "measured"]
    lines.append("Всички размери са измерени с шублер." if not est else
                 "Размерите с „оценено“ са от снимките, а не от шублера: проверете ги преди изработка.")  # fmt: skip
    lines += ["Промяна на число по-долу и ново изпълнение (CadQuery) дава новия детайл.", '"""', "",
              "import math", "", "import cadquery as cq", "", "# --- Размери (мм) ---"]  # fmt: skip
    for i in D.used_dims(d):
        dim = d["dims"][i]
        src = {"measured": "измерено", "skipped": "оценено от снимките", "photo": "оценено от снимките"}[dim["src"]]
        lines.append(f"{i} = {num(dim['v'])}  # {_txt(dim['name'], 60)} ({src})")
    lines += ["", "# --- Нива: разстояния от началото на оста ---"]
    names = {}
    for lid, spec in d["levels"].items():
        if spec["dim"] not in d["dims"]:
            continue
        tot = axis_total_expr(d, spec["axis"])
        expr = {"start": spec["dim"], "end": f"{tot} - {spec['dim']}", "diameter": f"{spec['dim']} / 2"}[spec["from"]]
        names[lid] = lid
        lines.append(f"{lid} = {expr}")
    lines += [HELPERS, "", "# --- Тяло ---"]

    def lv(lvl: str) -> str:
        if lvl in names:
            return names[lvl]
        ax, rest = lvl[0], lvl[1:]
        if rest == "0":
            return "0.0"
        if lvl == D.end_level(ax) or rest in ("L", "W", "H", "D"):
            return axis_total_expr(d, ax)
        if rest == "c":
            return f"{axis_total_expr(d, ax)} / 2"
        if rest[:1] == "~":
            return f"{axis_total_expr(d, ax)} * {rest[1:]}"
        return names[lvl]

    def pts_expr(view: str) -> str:
        rows = [f"({lv(u)}, {lv(v)}, {r or '0.0'})" for u, v, r in d["views"][view]["pts"]]
        return "[" + ", ".join(rows) + "]"

    if d["kind"] == "turned":
        lines.append(f"profile = {pts_expr('profile')}  # (по оста X, радиус)")
        lines.append("body = (outline(plane('z', 0), profile).revolve(360, (0, 0, 0), (1, 0, 0))")
        lines.append("        .translate((0, D / 2, D / 2)))  # оста по средата на кутията D x D")
    else:
        parts = []
        for view, axis, size in (("top", "z", "H"), ("side", "y", "W"), ("end", "x", "L")):
            spec = d["views"].get(view)
            if not spec or (is_box(view, spec) and not spec.get("round")):
                continue
            at = "W" if axis == "y" else "0"
            if spec.get("round"):
                a, b = (D.AXIS_TOTAL[k] for k in D.VIEW_AXES[view])
                lines.append(f"{view}_solid = round_view(plane('{axis}', {at}), {a}, {b}).extrude({size})"
                             f"  # {D.VIEW_NAME[view]}: кръг/елипса")  # fmt: skip
            elif spec.get("curve"):
                lines.append(f"{view} = {pts_expr(view)}  # {D.VIEW_NAME[view]}: гладка форма по снимката")
                lines.append(f"{view}_solid = curve(plane('{axis}', {at}), {view}, {spec.get('corners') or []})"
                             f".extrude({size})")  # fmt: skip
            else:
                lines.append(f"{view} = {pts_expr(view)}  # {D.VIEW_NAME[view]}")
                lines.append(f"{view}_solid = outline(plane('{axis}', {at}), {view}).extrude({size})")
            parts.append(f"{view}_solid")
        if not parts:
            lines.append("body = cq.Workplane(plane('z', 0)).moveTo(L / 2, W / 2).rect(L, W).extrude(H)")
        else:
            lines.append("body = " + parts[0])
            for p in parts[1:]:
                lines.append(f"body = body.intersect({p})")
    feats = [f for f in d["features"] if D.present(d, f)]  # с размер 0 ги няма
    edge_feats = [f for f in feats if f["t"] in ("fillet", "chamfer")]
    cut_feats = [f for f in feats if f["t"] in ("rect", "slot")]
    holes = [f for f in feats if f["t"] == "hole"]
    if edge_feats:
        lines += ["", "# --- Заобляния и фаски ---"]
        for f in edge_feats:
            op = f"fillet({f['r']})" if f["t"] == "fillet" else f"chamfer({f['c']})"
            if f["edges"] == "flat":  # до половината от дебелината (точно половината OpenCascade не може)
                op = f"fillet(min({f['r']}, H * 0.49))" if f["t"] == "fillet" else f"chamfer(min({f['c']}, H * 0.49))"
            lines.append(f"body = {edge_select('body', f['edges'], d['kind'])}.{op}  # {D.label(d, f)}")
    for title, group in (("Изрези, джобове и канали", cut_feats), ("Отвори", holes)):
        if not group:
            continue
        lines += ["", f"# --- {title} ---"]
        for f in group:
            a, b = D.plane_axes(f["axis"])
            u = pos_expr(f, "u", lv)
            v = pos_expr(f, "v", lv)
            size = axis_total_expr(d, f["axis"])
            depth = f.get("depth") or "None"
            if f["t"] == "hole":
                sk = f"lambda wp: wp.circle({f['d']} / 2)"
            elif f["t"] == "rect":
                sk = f"rounded_rect({f['w']}, {f['h']}, {f.get('r') or '0.0'})"
            else:
                sk = f"lambda wp: wp.slot2D({f['len']}, {f['w']}, {f.get('angle', 0)})"
            note = D.label(d, f) + (f", {D.FACE_NAME[f['face']]}" if f.get("face") else "")
            lines.append(f"body = body.cut(cut_tool('{f['axis']}', '{f['side']}', {u}, {v}, {size}, {depth}, {sk}))"
                         f"  # {_txt(note, 80)}")  # fmt: skip
    splits = [f for f in feats if f["t"] == "split"]
    if splits:
        lines += ["", "# --- Части (например капачка и тяло): отделни тела ---"]
        for f in splits:
            if f.get("at2"):  # по диагонал: на горната страна при at, на долната при at2
                top, low, h = lv(f["at"]), lv(f["at2"]), axis_total_expr(d, "z")
                lines.append(f"body = body.copyWorkplane(cq.Workplane(cq.Plane(({low}, 0, 0), (0, 1, 0), ({h}, 0, "
                             f"-({top} - {low}))))).split(keepTop=True, keepBottom=True)")  # fmt: skip
            else:
                lines.append(f"body = body.copyWorkplane(cq.Workplane(plane('x', {lv(f['at'])}))).split(keepTop=True, "
                             "keepBottom=True)")  # fmt: skip
    lines += ["", "result = body", ""]
    return "\n".join(lines)


def axis_total_expr(d: dict, axis: str) -> str:
    if axis == "r":
        return "(D / 2)"
    if d["kind"] == "turned" and axis in "yz":
        return "D"
    return D.AXIS_TOTAL[axis]


def pos_expr(f: dict, key: str, lv) -> str:
    p = f[key]
    c = lv(p["lvl"])
    if f["t"] in ("rect", "slot") and p["anchor"] in ("min", "max"):
        if f["t"] == "rect":
            size = f["w"] if key == "u" else f["h"]
        else:  # канал: дължината е по u при ъгъл 0, иначе по v
            along_u = f.get("angle", 0) == 0
            size = f["len"] if (key == "u") == along_u else f["w"]
        return f"({c} + {size} / 2)" if p["anchor"] == "min" else f"({c} - {size} / 2)"
    return c


def edge_select(body: str, edges: str, kind: str) -> str:
    if edges == "flat":
        return f'{body}.faces(">Z or <Z").edges()'
    if edges == "vertical":
        return f'{body}.edges("|Z")'
    if edges == "top":
        return f'{body}.faces(">Z").edges()'
    if edges == "bottom":
        return f'{body}.faces("<Z").edges()'
    if edges == "ends":
        return f'{body}.faces("<X or >X").edges()'
    return f"{body}.edges()"


def is_box(view: str, spec: dict) -> bool:
    """Изгледът е само правоъгълникът на общите размери (сечението с него не променя нищо)."""
    a, b = D.VIEW_AXES[view]
    return {tuple(p) for p in map(tuple, spec["pts"])} == {tuple(p) for p in D.box(a, b)}


# --- Изпълнение и файлове ------------------------------------------------------------------------------------------


def solid(text: str):
    """Изпълнява кода (създаден от `code`, никога от заявка) и връща твърдото тяло."""
    import cadquery as cq

    ns: dict = {}
    exec(compile(text, "part.py", "exec"), ns)  # noqa: S102 (кодът е създаден тук от проверено описание)
    result = ns["result"]
    shape = result.val() if isinstance(result, cq.Workplane) else result
    if isinstance(result, cq.Workplane) and len(result.vals()) > 1:
        shape = cq.Compound.makeCompound(result.vals())
    return shape


def build(raw: dict, out: Path) -> dict:
    """Строи детайла в `out`: part.py, part.step, part.stl, part.glb, design.json. Връща отчета."""
    import cadquery as cq

    d = clean(raw)
    D.check(d)
    text = code(d)
    try:
        shape = solid(text)
    except D.DesignError:
        raise
    except Exception as e:  # OpenCascade: заобляне, по-голямо от ръба, и подобни
        raise D.DesignError(failure_text(d, e)) from e
    if not shape.isValid() or shape.Volume() <= 0:
        raise D.DesignError("Тялото не се получи цяло: проверете размерите на изрезите и заобляниятa.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "part.py").write_text(text, encoding="utf-8")
    (out / "design.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    wp = cq.Workplane().add(shape)
    cq.exporters.export(wp, str(out / "part.step"))
    bb = shape.BoundingBox()
    size = max(bb.xlen, bb.ylen, bb.zlen)
    cq.exporters.export(wp, str(out / "part.stl"), tolerance=max(0.005, size / 4000), angularTolerance=0.08)
    mesh = glb(out / "part.stl", out / "part.glb", shape)
    s = D.summary(d)
    volume = float(shape.Volume())
    if any(v.get("curve") for v in d["views"].values()):  # OpenCascade смята обема на сплайн неточно (до 3%): мрежата
        volume = abs(float(mesh.volume))
    report = {"size_mm": [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)],
              "volume_mm3": round(volume, 1), "solids": len(shape.Solids()),
              "kind": d["kind"], "measured": len(s["measured"]), "estimated": len(s["estimated"]),
              "estimated_names": [d["dims"][i]["name"] for i in s["estimated"]],
              "elements": [x["text"] for x in D.describe(d)], "triangles": int(len(mesh.faces))}  # fmt: skip
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return report


def failure_text(d: dict, e: Exception) -> str:
    msg = str(e)
    if any(f["t"] == "fillet" for f in d["features"]) and ("fillet" in msg.lower() or "BRep_API" in msg):
        return "Заоблянето не може да се направи с този радиус: намалете го или изберете други ръбове."
    if any(f["t"] == "chamfer" for f in d["features"]) and "chamfer" in msg.lower():
        return "Фаската не може да се направи с този размер: намалете я."
    return "Моделът не се построи с тези размери: проверете дали отворите и изрезите са в детайла."


# Частите в ясно различни цветове (и под силната светлина на прегледа): тялото светлосиво, капачката оранжева
# (цветът на приложението), трета част синя.
PART_COLORS = ([200, 204, 212, 255], [242, 120, 32, 255], [40, 110, 220, 255])


def glb(stl: Path, dest: Path, shape=None):
    """GLB за прегледа в приложението, в мм: всяко тяло (капачка, тяло) в свой цвят, с нормали, при които острите
    ръбове (над 30°) са остри, а заоблените повърхности гладки. Без нормали прегледът сам решаваше как да осветява, а
    с общи нормали през ръбовете (както при сканиране) детайлът изглеждаше размазан и закръглен. Записва и
    display.ply (същото, за модела в историята: service.build)."""
    import numpy as np
    import trimesh

    from .. import mesh as M

    solids = sorted(shape.Solids(), key=lambda x: -x.Volume()) if shape is not None else []
    parts = []
    if len(solids) <= 1:
        parts.append(trimesh.load(stl, force="mesh"))
    else:  # най-голямата част (тялото) е сива, по-малките (капачка) в цвят
        for sol in solids:
            bb = sol.BoundingBox()
            vs, fs = sol.tessellate(max(0.005, max(bb.xlen, bb.ylen, bb.zlen) / 4000), 0.08)
            parts.append(trimesh.Trimesh(np.array([(v.x, v.y, v.z) for v in vs]), np.array(fs), process=True))
    shown = []
    for k, m in enumerate(parts):
        trimesh.repair.fix_normals(m)
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(PART_COLORS[k % len(PART_COLORS)], (len(m.faces), 1)))
        shown.append(trimesh.graph.smooth_shade(m, angle=np.radians(30), facet_minarea=None))
    whole = trimesh.util.concatenate(shown) if len(shown) > 1 else shown[0]
    whole.export(dest.with_name("display.ply"))  # в мм, Z нагоре (за модела в историята: service.build)
    upright = whole.copy()
    upright.apply_transform(M.Y_UP)  # glTF е с Y нагоре: прегледът и историята показват детайла еднакво
    dest.write_bytes(M.glb(upright))
    return whole
