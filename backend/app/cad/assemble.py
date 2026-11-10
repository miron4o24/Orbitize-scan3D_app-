"""Прочетените изгледи (vision.View) -> описанието на детайла (design.py) с оценките от снимките.

Посоките при снимане (приложението ги казва): отгоре - детайлът лежи, левият му край вляво; отстрани - легнал на
страна, горната му страна към горния край на екрана и същият край вляво; откъм края - изправен на десния си край,
горната страна към горния край на екрана, тоест гледаме левия край (тогава надясно на снимката е към предния ръб).
Обръщания (flips) поправят снимка, направена наобратно."""

import math

import numpy as np
from shapely.geometry import Polygon

from . import design as D
from . import fit

TURNED_SYM = 0.93  # очертание, огледално на себе си поне толкова (IoU), е от струг
TURNED_DIAM = 0.08  # W и H в рамките на 8%: кръгло сечение
RADIUS_GROUP = 0.25  # радиуси в рамките на 25% са един размер
DIAM_GROUP = 0.04


def mirror_iou(outer: np.ndarray, V: float) -> float:
    p = Polygon(outer).buffer(0)
    q = Polygon(np.c_[outer[:, 0], V - outer[:, 1]]).buffer(0)
    return p.intersection(q).area / max(p.union(q).area, 1e-9)


def resting_down(outer: np.ndarray, U: float, V: float) -> bool:
    """Снимката отстрани или откъм края е обърната наопаки (завъртяна на 180°): телефонът, държан над масата, не знае
    накъде е "горе", а детайлът стои на долната си страна, която обикновено е пълна права по цялата дължина. Ако
    горната страна на снимката е по-пълна от долната с поне 20%, снимката е наопаки."""
    tol = 0.012 * V
    poly = Polygon(outer).buffer(0)
    edge_pts = np.asarray(poly.exterior.coords)
    bottom = np.ptp(edge_pts[edge_pts[:, 1] <= tol][:, 0]) if (edge_pts[:, 1] <= tol).sum() > 1 else 0.0
    top = np.ptp(edge_pts[edge_pts[:, 1] >= V - tol][:, 0]) if (edge_pts[:, 1] >= V - tol).sum() > 1 else 0.0
    return top > 1.2 * bottom + 0.02 * U


def oriented(view: dict, name: str, flips: dict) -> dict:
    """Изглед в осите на детайла: (u, v) на изгледа -> координати по осите му, с обръщанията."""
    U, V = view["U"], view["V"]
    fu, fv = flips.get(name, [False, False])
    if name in ("side", "end") and resting_down(np.asarray(view["outer"]), U, V):
        fu, fv = not fu, not fv  # завъртане на 180°: и двете оси
    if name == "end":
        fu = not fu  # гледаме левия край: надясно на снимката е към предния ръб (по-малки Y)

    def map_(p):
        p = np.asarray(p, float)
        return np.c_[U - p[:, 0] if fu else p[:, 0], V - p[:, 1] if fv else p[:, 1]]

    return {"outer": map_(view["outer"]), "holes": [map_(h) for h in view["holes"]], "U": U, "V": V}


def is_turned(views: dict, L: float, W: float, H: float) -> bool:
    if abs(W - H) / max(W, H) > TURNED_DIAM or "top" not in views or "side" not in views:
        return False
    if "end" in views:
        e = views["end"]
        _, _, _, rel = fit.circle_fit(e["outer"])
        if rel > 0.03:
            return False
    top, side = views["top"], views["side"]
    if mirror_iou(top["outer"], top["V"]) < TURNED_SYM or mirror_iou(side["outer"], side["V"]) < TURNED_SYM:
        return False
    a, b = Polygon(top["outer"]).buffer(0), Polygon(side["outer"]).buffer(0)
    return a.intersection(b).area / max(a.union(b).area, 1e-9) > 0.9


def assemble(views: dict, L: float, W: float, H: float, flips: dict | None = None) -> dict:
    """views: {"top"|"side"|"end": {"outer", "holes", "U", "V"}} в мм на изгледа (от vision.View)."""
    flips = flips or {}
    views = {k: oriented(v, k, flips) for k, v in views.items()}
    if is_turned(views, L, W, H):
        return turned(views, L, W, H)
    d = D.new("block", L, W, H)
    radius_dims: list[tuple[float, str]] = []
    for name in ("top", "side", "end"):
        if name not in views:
            continue
        v = views[name]
        a, b = D.VIEW_AXES[name]
        axis = {"top": "z", "side": "y", "end": "x"}[name]
        for h in v["holes"]:
            add_hole(d, h, axis, name, max(v["U"], v["V"]))
        cs = fit.corners(v["outer"], v["U"], v["V"])
        if cs is None:
            d["views"][name] = free_view(d, v, a, b, name)
            d["notes"].append(f"Очертанието {D.VIEW_NAME[name]} не е от прави и дъги: взето е от снимката (оценено).")
            continue
        pts = []
        for u, w, r in cs:
            lu = D.level_for(d, a, clamp(u, v["U"]), step_name(a), step_what(a), view=name)
            lv = D.level_for(d, b, clamp(w, v["V"]), step_name(b), step_what(b), view=name)
            rid = radius_dim(d, radius_dims, r, name) if r > 0 else None
            if pts and pts[-1][:2] == [lu, lv]:  # два ъгъла на едно място (под допуска): един
                continue
            pts.append([lu, lv, rid])
        if len(pts) >= 2 and pts[0][:2] == pts[-1][:2]:
            pts.pop()
        d["views"][name] = {"pts": pts} if len(pts) >= 3 else {"pts": D.box(a, b)}
    group_holes(d)
    try:
        D.check(d)
    except D.DesignError:
        # Оценките от снимката не пасват (например стъпало, почти колкото ръба): изгледите стават правоъгълници.
        for name in ("top", "side", "end"):
            a, b = D.VIEW_AXES[name]
            d["views"][name] = {"pts": D.box(a, b)}
        D.unused_cleanup(d)
        d["notes"].append("Очертанията от снимките не се събраха: започваме от кутия; добавете елементите ръчно.")
    D.unused_cleanup(d)
    return d


def clamp(x: float, top: float) -> float:
    return min(max(x, 0.0), top)


def step_name(axis: str) -> str:
    return {"x": "Стъпало по дължината", "y": "Стъпало по ширината", "z": "Стъпало по височината",
            "r": "Диаметър"}[axis]  # fmt: skip


def step_what(axis: str) -> str:
    return {"x": "ръба на стъпалото", "y": "ръба на стъпалото", "z": "ръба на стъпалото", "r": ""}[axis]


def radius_dim(d: dict, known: list, r: float, view: str) -> str:
    for value, i in known:
        if abs(value - r) <= RADIUS_GROUP * max(value, r):
            return i
    i = D.add_dim(d, "r", r, "Радиус на заоблен ъгъл", "Радиус на заобления ъгъл",
                  "Сравнете с радиусомер или свредла; ако не можете да го измерите, оставете оценката.", "radius",
                  optional=True, show={"view": view, "radius": True})  # fmt: skip
    known.append((r, i))
    return i


def free_view(d: dict, v: dict, a: str, b: str, view: str) -> dict:
    """Свободна форма: опростеното очертание, с нива, които не се питат (оценени от снимката)."""
    poly = Polygon(v["outer"]).buffer(0).simplify(0.01 * max(v["U"], v["V"]))
    pts = []
    for u, w in list(poly.exterior.coords)[:-1]:
        row = []
        for ax, val, size in ((a, u, v["U"]), (b, w, v["V"])):
            lvl = D.level_for(d, ax, clamp(val, size), "Точка от формата", "точката", view=view, merge=False)
            if lvl in d["levels"]:
                d["dims"][d["levels"][lvl]["dim"]]["hidden"] = True
            row.append(lvl)
        pts.append(row + [None])
    return {"pts": pts, "free": True}


def add_hole(d: dict, pts: np.ndarray, axis: str, view: str, size: float) -> None:
    shape = fit.hole_shape(pts, size)
    a, b = D.plane_axes(axis)
    fid = D.new_feature_id(d, shape["t"])
    f = {"id": fid, "t": shape["t"], "axis": axis, "side": "+", "view": view,
         "face": {"z": "top", "y": "front", "x": "left"}[axis]}  # fmt: skip
    if shape["t"] == "hole":
        f["d"] = D.add_dim(d, "d", shape["d"], "Диаметър на отвора", "Измерете диаметъра на отвора",
                           "С горните (вътрешните) челюсти на шублера.", "diameter",
                           show={"feature": fid})  # fmt: skip
        sizes = (0.0, 0.0)
        what = "центъра на отвора"
    elif shape["t"] == "slot":
        f["len"] = D.add_dim(d, "s", shape["len"], "Дължина на канала", "Измерете канала по дължината му (с краищата)",
                             "С вътрешните челюсти.", show={"feature": fid})  # fmt: skip
        f["w"] = D.add_dim(d, "s", shape["w"], "Ширина на канала", "Измерете ширината на канала",
                           "С вътрешните челюсти.", show={"feature": fid})  # fmt: skip
        f["angle"] = shape["angle"]
        sizes = (shape["len"], shape["w"]) if shape["angle"] == 0 else (shape["w"], shape["len"])
        what = "ръба на канала"
    else:
        f["w"] = D.add_dim(d, "w", shape["w"], "Ширина на отвора", f"Измерете отвора по {D.axis_word(a)}",
                           "С вътрешните челюсти.", show={"feature": fid, "along": a})  # fmt: skip
        f["h"] = D.add_dim(d, "h", shape["h"], "Височина на отвора", f"Измерете отвора по {D.axis_word(b)}",
                           "С вътрешните челюсти.", show={"feature": fid, "along": b})  # fmt: skip
        if shape.get("r"):
            f["r"] = D.add_dim(d, "r", shape["r"], "Радиус в ъглите на отвора", "Радиус в ъглите на отвора",
                               "Ако не можете да го измерите, оставете оценката.", "radius", optional=True,
                               show={"feature": fid})  # fmt: skip
        sizes = (shape["w"], shape["h"])
        what = "ръба на отвора"
    for key, ax, c, s in (("u", a, shape["u"], sizes[0]), ("v", b, shape["v"], sizes[1])):
        tot = D.total(d, ax)
        if abs(c - tot / 2) <= D.CENTER_TOL * tot:
            f[key] = {"lvl": f"{ax}c", "anchor": "center"}
            continue
        if f["t"] == "hole":
            lvl = D.level_for(d, ax, c, "Положение на отвора", what, view=view, merge=False)
            f[key] = {"lvl": lvl, "anchor": "center"}
        else:  # ръбът по-близо до края на детайла се мери от него
            near_start = c < tot / 2
            edge = c - s / 2 if near_start else c + s / 2
            lvl = D.level_for(d, ax, clamp(edge, tot), "Положение на " + ("канала" if f["t"] == "slot" else "отвора"),
                              what, view=view, merge=False)  # fmt: skip
            f[key] = {"lvl": lvl, "anchor": "min" if near_start else "max"}
    d["features"].append(f)


def group_holes(d: dict) -> None:
    """Еднакви по диаметър отвори (до 4%) имат един размер: "4 отвора Ø"."""
    holes = [f for f in d["features"] if f["t"] == "hole"]
    groups: list[list[dict]] = []
    for f in holes:
        v = d["dims"][f["d"]]["v"]
        for g in groups:
            if abs(d["dims"][g[0]["d"]]["v"] - v) <= DIAM_GROUP * max(v, 1e-9):
                g.append(f)
                break
        else:
            groups.append([f])
    for g in groups:
        if len(g) < 2:
            continue
        keep = g[0]["d"]
        vals = [d["dims"][f["d"]]["v"] for f in g]
        d["dims"][keep]["v"] = d["dims"][keep]["est"] = round(float(np.mean(vals)), 2)
        d["dims"][keep]["name"] = f"Диаметър на отворите ({len(g)} бр.)"
        for f in g[1:]:
            f["d"] = keep


# --- Детайл от струг ------------------------------------------------------------------------------------------------


def turned(views: dict, L: float, W: float, H: float) -> dict:
    d = D.new("turned", L, W, H)
    profiles = [fit.turned_profile(views[k]["outer"], views[k]["U"], views[k]["V"]) for k in ("top", "side")]
    prof = profiles[0]  # отгоре и отстрани са почти еднакви (проверено); отгоре е по-ясно (лежи стабилно)
    Dm = D.value(d, "D")
    # Радиусите по групи: най-големият е D/2, другите са нови диаметри за измерване.
    radii: list[tuple[float, str]] = [(Dm / 2, "rD")]

    def r_level(r: float) -> str:
        for value, lvl in radii:
            if abs(value - r) <= 0.015 * Dm + 0.02 * max(value, r):
                return lvl
        k = len(radii) + 1
        dim = D.add_dim(d, "d", 2 * r, f"Диаметър {k}", "Измерете този диаметър",
                        "С външните челюсти на шублера.", "diameter", show={"view": "profile", "diameter": True})  # fmt: skip
        lid = f"L{dim}"
        d["levels"][lid] = {"axis": "r", "dim": dim, "from": "diameter"}
        d["dims"][dim]["show"]["level"] = lid
        radii.append((r, lid))
        return lid

    upper = []
    for x, r in prof:
        lx = D.level_for(d, "x", clamp(x, L), "Рамо по дължината", "рамото", view="profile")
        lr = r_level(r)
        if upper and upper[-1][:2] == [lx, lr]:
            continue
        upper.append([lx, lr, None])
    # Затворен профил: по оста от 0 до L, после горният ръб обратно.
    if not upper or upper[0][0] != "x0":
        upper.insert(0, ["x0", upper[0][1] if upper else "rD", None])
    if upper[-1][0] != "xL":
        upper.append(["xL", upper[-1][1], None])
    pts = [["x0", "r0", None], ["xL", "r0", None]] + [p for p in reversed(upper)]
    clean = []
    for p in pts:
        if not clean or clean[-1][:2] != p[:2]:
            clean.append(p)
    d["views"]["profile"] = {"pts": clean}
    if "end" in views:
        e = views["end"]
        cx, cy, _, _ = fit.circle_fit(e["outer"])
        for h in e["holes"]:
            c = fit.robust_circle(h)
            if c is None:
                continue
            hx, hy, hr = c
            if math.hypot(hx - cx, hy - cy) < 0.06 * Dm:
                fid = D.new_feature_id(d, "hole")
                dim = D.add_dim(d, "d", 2 * hr, "Диаметър на отвора по оста", "Измерете диаметъра на отвора по оста",
                                "С вътрешните челюсти.", "diameter", show={"feature": fid})  # fmt: skip
                d["features"].append({"id": fid, "t": "hole", "axis": "x", "side": "-", "face": "left",
                                      "view": "end", "d": dim, "u": {"lvl": "yc", "anchor": "center"},
                                      "v": {"lvl": "zc", "anchor": "center"}})  # fmt: skip
                d["notes"].append("Отворът по оста е взет като проходен: ако не е, сменете го на глух.")
    for k in ("top", "side"):
        for h in views[k]["holes"]:
            c = fit.robust_circle(h)
            if c is not None:
                hx, hy, hr = c
                fid = D.new_feature_id(d, "hole")
                dim = D.add_dim(d, "d", 2 * hr, "Диаметър на напречния отвор", "Измерете диаметъра на напречния отвор",
                                "С вътрешните челюсти.", "diameter", show={"feature": fid})  # fmt: skip
                lx = D.level_for(d, "x", hx, "Положение на напречния отвор", "центъра на отвора", view="profile",
                                 merge=False)  # fmt: skip
                d["features"].append({"id": fid, "t": "hole", "axis": "y", "side": "+", "face": "top", "view": k,
                                      "d": dim, "u": {"lvl": lx, "anchor": "center"},
                                      "v": {"lvl": "zc", "anchor": "center"}})  # fmt: skip
        if any(f["axis"] == "y" for f in d["features"]):
            break  # напречният отвор се вижда и в двата изгледа: веднъж
    group_holes(d)
    D.unused_cleanup(d)
    return d
