"""Описанието на детайла и въпросите към потребителя.

Всеки размер е запис в `dims` (стойност в мм и откъде е: измерен с шублер, оценен от снимките или пропуснат от
потребителя, който тогава остава оценката). Геометрията никога не съдържа числа, а само препратки към размерите:

- Общите размери L (дължина, по X), W (ширина, по Y), H (височина, по Z). Детайл от струг ("turned") има L и D
  (най-големият диаметър).
- Нива: координата по една ос, зададена като разстояние от началото или от края на детайла по нея (от по-близкия
  край: така се мери с шублер) или "в средата". Особените нива: x0/xL, y0/yW, z0/zH и средите xc, yc, zc.
- Изгледи: очертание отгоре (X, Y), отстрани (X, Z) и откъм края (Y, Z) като многоъгълник от нива, с радиус по
  желание във всеки ъгъл. Тялото е сечението на трите изтеглени очертания. При детайл от струг: половината профил
  (X, радиус), завъртян около оста X.
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
VIEW_NAME = {"top": "отгоре", "side": "отстрани", "end": "откъм края", "profile": "отстрани"}
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
             "all": "всички ръбове", "ends": "краищата"}  # fmt: skip


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
    raise DesignError("Непознато ниво.")


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
        if not 0 < float(v) <= LIMIT:
            raise DesignError(f"„{dim['name']}“ трябва да е между 0 и {LIMIT:g} мм.")
    for view, spec in d["views"].items():
        if spec.get("off"):
            continue
        pts = outline(d, view)
        poly = Polygon([(u, v) for u, v, _ in pts])
        if not poly.is_valid or poly.area <= 0:
            names = sorted({d["dims"][d["levels"][p]["dim"]]["name"] for row in spec["pts"] for p in row[:2]
                            if p in d["levels"]})  # fmt: skip
            raise DesignError(f"Очертанието {VIEW_NAME[view]} се пресича само: проверете "
                              f"{', '.join(names) or 'общите размери'}.")  # fmt: skip
        a, b = VIEW_AXES[view]
        for u, v, _ in pts:
            if not (-1e-6 <= u <= total(d, a) + 1e-6 and -1e-6 <= v <= total(d, b) + 1e-6):
                raise DesignError(f"Ниво {VIEW_NAME[view]} е извън общите размери: проверете разстоянията до ръбовете.")
        for k, (u, v, r) in enumerate(pts):
            if r <= 0:
                continue
            pu, pv, _ = pts[k - 1]
            nu, nv, _ = pts[(k + 1) % len(pts)]
            room = min(math.dist((u, v), (pu, pv)), math.dist((u, v), (nu, nv)))
            ang = corner_angle((pu, pv), (u, v), (nu, nv))
            if ang <= 1e-3 or ang >= math.pi - 1e-3 or r / math.tan(ang / 2) > room + 1e-6:
                raise DesignError(f"Радиусът {r:g} мм е твърде голям за ъгъла {VIEW_NAME[view]}.")
    for f in d["features"]:
        if f["t"] in ("hole", "rect", "slot"):
            place = feature_place(d, f)
            a, b = plane_axes(place["axis"])
            if not (0 <= place["u"] <= total(d, a) and 0 <= place["v"] <= total(d, b)):
                raise DesignError(f"{label(d, f)} е извън детайла: проверете положението му.")


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
    return t


def describe(d: dict) -> list[dict]:
    """Какво е разпознато, за екрана "Какво виждам": [{id, text, view, kind}] по реда на модела."""
    out = []
    if d["kind"] == "turned":
        pts = d["views"]["profile"]["pts"]
        diam = len({p[1] for p in pts if p[1] not in ("r0",)})
        out.append({"id": "body", "view": "profile", "kind": "body",
                    "text": f"Тяло от струг: {diam} диаметър{'а' if diam > 1 else ''} по дължината"})  # fmt: skip
    for view in ("top", "side", "end"):
        spec = d["views"].get(view)
        if not spec or spec.get("off"):
            continue
        pts = spec["pts"]
        rounded = sum(1 for p in pts if p[2])
        plain = len(pts) == 4 and all(p[0][1:] in "0LWH" and p[1][1:] in "0LWH" for p in pts)
        if spec.get("free"):
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
            if not 0 < v <= LIMIT:
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
            d["views"][view] = {"pts": box(a, b)}
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
