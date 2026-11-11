"""Изгледите в мм (photos.views_mm: вече в осите на детайла) -> описанието на детайла (design.py) с оценките от
снимките: стъпала и заоблени ъгли от очертанията, отвори, кръгли изгледи (цилиндър), детайл от струг."""

import math

import numpy as np
from shapely.geometry import Polygon

from . import design as D
from . import fit

ROUND_RMS = 0.02  # кръгъл изглед: отклонение от кръг под 2% от радиуса
TURNED_SYM = 0.93  # очертание, огледално на себе си поне толкова (IoU), е от струг
TURNED_DIAM = 0.08  # W и H в рамките на 8%: кръгло сечение
RADIUS_GROUP = 0.25  # радиуси в рамките на 25% са един размер
DIAM_GROUP = 0.04


def mirror_iou(outer: np.ndarray, V: float) -> float:
    p = Polygon(outer).buffer(0)
    q = Polygon(np.c_[outer[:, 0], V - outer[:, 1]]).buffer(0)
    return p.intersection(q).area / max(p.union(q).area, 1e-9)


def is_round(v: dict) -> bool:
    """Изгледът е кръг или елипса (по нормираните до квадрат очертания)."""
    pts = np.c_[np.asarray(v["outer"])[:, 0] / v["U"], np.asarray(v["outer"])[:, 1] / v["V"]]
    if abs(math.log(v["U"] / v["V"])) > math.log(1.07):
        return False
    _, _, _, rel = fit.circle_fit(pts)
    return rel < ROUND_RMS


def is_turned(views: dict, L: float, W: float, H: float) -> bool:
    """Детайл от струг (вал, втулка, цилиндър легнал по дължината): кръгло сечение (W = H и снимката от края е кръг)
    и очертанието отгоре огледално спрямо оста. Без снимка от края: и отпред трябва да е същото като отгоре."""
    if abs(W - H) / max(W, H) > TURNED_DIAM or "top" not in views:
        return False
    top = views["top"]
    if mirror_iou(top["outer"], top["V"]) < TURNED_SYM:
        return False
    if "end" in views:
        _, _, _, rel = fit.circle_fit(views["end"]["outer"])
        if rel > 0.03:
            return False
        if "side" not in views:
            return True
    elif "side" not in views:
        return False
    side = views["side"]
    if mirror_iou(side["outer"], side["V"]) < TURNED_SYM:
        return False
    a, b = Polygon(top["outer"]).buffer(0), Polygon(side["outer"]).buffer(0)
    return a.intersection(b).area / max(a.union(b).area, 1e-9) > 0.9


def profile_corners(outer: np.ndarray, U: float, V: float) -> list[tuple[float, float, float]] | None:
    """Горната половина на очертанието на детайл от струг като ъгли (x, радиус, радиус на заобляне) отляво надясно,
    от същото разпознаване на прави и дъги като при другите изгледи (заоблените краища стават дъги, а не стъпала)."""
    cs = fit.corners(outer, U, V)
    if cs is None:
        return None
    R = V / 2
    up = [c[1] > R + 0.02 * V for c in cs]
    if sum(up) < 2 or all(up):
        return None
    n = len(cs)
    start = next(k for k in range(n) if up[k] and not up[k - 1])
    run = []
    k = start
    while up[k % n] and len(run) < n:
        run.append(cs[k % n])
        k += 1
    if len(run) != sum(up):
        return None  # горната страна не е едно парче
    return [(u, v - R, r) for u, v, r in reversed(run)]  # обратно на часовника горе е отдясно наляво


def assemble(views: dict, L: float, W: float, H: float) -> dict:
    """views: {"top"|"side"|"end": {"outer", "holes", "U", "V"}} в мм на изгледа, в осите на детайла."""
    views = {k: {"outer": np.asarray(v["outer"], float), "holes": [np.asarray(h, float) for h in v["holes"]],
                 "U": v["U"], "V": v["V"]} for k, v in views.items()}  # fmt: skip
    if is_turned(views, L, W, H):
        return turned(views, L, W, H)
    d = D.new("block", L, W, H)
    radius_dims: list[tuple[float, str]] = []
    top_tabs: list[dict] = []
    for name in ("top", "side", "end"):
        if name not in views:
            continue
        v = views[name]
        a, b = D.VIEW_AXES[name]
        axis = {"top": "z", "side": "y", "end": "x"}[name]
        for h in v["holes"]:
            add_hole(d, h, axis, name, max(v["U"], v["V"]))
        if is_round(v):
            d["views"][name] = {"pts": D.box(a, b), "round": True}
            continue
        cs = fit.corners(v["outer"], v["U"], v["V"], raw=True)
        if cs is not None:
            cs = fit.symmetric_round(cs, v["U"], v["V"])
            if name in ("top", "side"):  # по-тесен/по-тънък край (флашка, химикалка): къси стъпала
                tabs = fit.end_tabs(v["outer"], v["U"], v["V"])
                cs = fit.apply_tabs(cs, tabs, v["U"], v["V"])
                if name == "top":
                    top_tabs = tabs
        if cs is None:
            d["views"][name] = free_view(d, v, a, b, name)
            d["notes"].append(f"Очертанието {D.VIEW_NAME[name]} не е от прави и дъги: взето е от снимката (оценено).")
            continue
        d["views"][name] = view_pts(d, cs, a, b, v["U"], v["V"], name, radius_dims)
    if top_tabs and "side" not in views:
        # По-тесният край обикновено е по-тънък със същото (без снимка отпред не се вижда): оценка, която се мери.
        tabs = [t | {"lo": min((t["lo"] + t["hi"]) / 2, 0.3 * H), "hi": min((t["lo"] + t["hi"]) / 2, 0.3 * H)}
                for t in top_tabs]  # fmt: skip
        cs = fit.apply_tabs([(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, H, 0.0), (0.0, H, 0.0)], tabs, L, H)
        d["views"]["side"] = view_pts(d, cs, "x", "z", L, H, "side", radius_dims)
    group_holes(d)
    try:
        D.check(d)
    except D.DesignError:
        # Оценките от снимката не пасват (например стъпало, почти колкото ръба): изгледите стават правоъгълници.
        for name in ("top", "side", "end"):
            a, b = D.VIEW_AXES[name]
            if not d["views"].get(name, {}).get("round"):
                d["views"][name] = {"pts": D.box(a, b)}
        D.unused_cleanup(d)
        d["notes"].append("Очертанията от снимките не се събраха: започваме от кутия; добавете елементите ръчно.")
    D.unused_cleanup(d)
    return d


def view_pts(d: dict, cs: list, a: str, b: str, U: float, V: float, name: str, radius_dims: list) -> dict:
    """Ъглите (u, v, r) в мм -> изгледът на описанието: нива (стъпала) и размери за радиусите."""
    pts = []
    for u, w, r in cs:
        lu = D.level_for(d, a, clamp(u, U), step_name(a), step_what(a), view=name)
        lv = D.level_for(d, b, clamp(w, V), step_name(b), step_what(b), view=name)
        rid = radius_dim(d, radius_dims, r, name) if r > 0 else None
        if pts and pts[-1][:2] == [lu, lv]:  # два ъгъла на едно място (под допуска): един
            continue
        pts.append([lu, lv, rid])
    if len(pts) >= 2 and pts[0][:2] == pts[-1][:2]:
        pts.pop()
    pts = tidy(pts)
    return {"pts": pts} if len(pts) >= 3 else {"pts": D.box(a, b)}


def tidy(pts: list) -> list:
    """Маха ъглите, които не са ъгли: повторени точки и точки по средата на права по едно ниво (стъпало, по-малко
    от допуска, слято с ръба), докато не остане какво да се маха."""
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for k in range(len(pts)):
            p, c, n = pts[k - 1], pts[k], pts[(k + 1) % len(pts)]
            same = c[:2] == p[:2]
            on_line = (p[0] == c[0] == n[0]) or (p[1] == c[1] == n[1])
            if same or (on_line and not c[2]):
                pts = pts[:k] + pts[k + 1 :]
                changed = True
                break
    return pts


def clamp(x: float, top: float) -> float:
    return min(max(x, 0.0), top)


def step_name(axis: str) -> str:
    return {"x": "Стъпало по дължината", "y": "Стъпало по ширината", "z": "Стъпало по височината",
            "r": "Диаметър"}[axis]  # fmt: skip


def step_what(axis: str) -> str:
    return {"x": "ръба на стъпалото", "y": "ръба на стъпалото", "z": "ръба на стъпалото", "r": ""}[axis]


def radius_dim(d: dict, known: list, r: float, view: str) -> str:
    """Размерът за заоблен ъгъл: близките радиуси в един изглед са един размер (едно мерене); в различните изгледи
    са различни ръбове (например ъглите отгоре и заоблените страни), затова и различни размери."""
    for value, i in known:
        if d["dims"][i]["show"].get("view") == view and abs(value - r) <= RADIUS_GROUP * max(value, r):
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
    top = views["top"]
    prof = profile_corners(top["outer"], top["U"], top["V"])
    if prof is None:
        prof = [(x, r, 0.0) for x, r in fit.turned_profile(top["outer"], top["U"], top["V"])]
    Dm = D.value(d, "D")
    # Радиусите по групи: най-големият е D/2, другите са нови диаметри за измерване.
    radii: list[tuple[float, str]] = [(Dm / 2, "rD")]
    rounds: list[tuple[float, str]] = []

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

    def round_dim(rad: float, at_r: float) -> str | None:
        if rad <= 0:
            return None
        if abs(rad - at_r) <= 0.07 * at_r:  # цял полукръг (заоблен край на капсула): точно половината диаметър
            rad = math.floor(at_r * 100) / 100
            for value, i in rounds:
                if abs(value - rad) < 1e-6:
                    return i
            i = D.add_dim(d, "r", rad, "Радиус на заобления край", "Радиус на заобления край",
                          "Ако краят е полукръг, оставете оценката (половината диаметър).", "radius", optional=True,
                          show={"view": "profile", "radius": True})  # fmt: skip
            rounds.append((rad, i))
            return i
        return radius_dim(d, rounds, rad, "profile")

    upper = []
    for x, r, rad in prof:
        lx = D.level_for(d, "x", clamp(x, L), "Рамо по дължината", "рамото", view="profile")
        lr = r_level(r)
        rid = round_dim(rad, D.coord(d, lr))
        if upper and upper[-1][:2] == [lx, lr]:
            continue
        upper.append([lx, lr, rid])
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
    if "end" not in views:
        d["notes"].append("Няма снимка от края: приех, че сечението е кръгло. Ако не е, снимайте и края.")
    for k in ("top", "side"):
        if k not in views:
            continue
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
