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
RADIUS_GROUP = 0.12  # радиуси в рамките на 12%, заоблени в една посока, са един размер
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


def profile_corners(outer: np.ndarray, U: float, V: float,
                    cs: list | None = None) -> list[tuple[float, float, float]] | None:
    """Горната половина на очертанието на детайл от струг като ъгли (x, радиус, радиус на заобляне) отляво надясно,
    от същото разпознаване на прави и дъги като при другите изгледи (заоблените краища стават дъги, а не стъпала).
    cs: вече намерените ъгли (с по-тесните краища)."""
    cs = fit.corners(outer, U, V) if cs is None else cs
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


def assemble(views: dict, L: float, W: float, H: float, flat: bool = False) -> dict:
    """views: {"top"|"side"|"end": {"outer", "holes", "U", "V"}} в мм на изгледа, в осите на детайла. flat: плосък
    детайл (кука, пластина): само очертанието отгоре, с една дебелина навсякъде."""
    if flat:
        views = {k: v for k, v in views.items() if k == "top"}
    views = {k: {"outer": np.asarray(v["outer"], float), "holes": [np.asarray(h, float) for h in v["holes"]],
                 "U": v["U"], "V": v["V"], "each": [np.asarray(e, float) for e in v.get("each", [])],
                 "blind": [(np.asarray(b["pts"], float), b["face"]) for b in v.get("blind", [])
                           if b.get("face") in D.FACES]}
             for k, v in views.items()}  # fmt: skip
    if is_turned(views, L, W, H):
        return turned(views, L, W, H)
    d = D.new("block", L, W, H)
    radius_dims: list[tuple[float, str, bool]] = []
    top_tabs: list[dict] = []
    for name in ("top", "side", "end"):
        if name not in views:
            continue
        v = views[name]
        a, b = D.VIEW_AXES[name]
        axis = {"top": "z", "side": "y", "end": "x"}[name]
        for h in v["holes"]:
            add_hole(d, h, axis, name, max(v["U"], v["V"]))
        for h, face in v["blind"]:
            add_hole(d, h, axis, name, max(v["U"], v["V"]), face)
        if is_round(v):
            d["views"][name] = {"pts": D.box(a, b), "round": True}
            continue
        if not flat:  # каналите с прави стени: отделни елементи с мерки, очертанието е без тях
            outer, cuts = fit.notches(v["outer"], v["U"], v["V"], v["each"])
            for c in cuts:
                add_notch(d, c, axis, name)
            v = v | {"outer": outer}
        cs = fit.corners(v["outer"], v["U"], v["V"], raw=True)
        if cs is not None:
            cs = fit.symmetric_round(cs, v["U"], v["V"])
            if name in ("top", "side"):  # по-тесен/по-тънък край (флашка, химикалка): къси стъпала
                other = views.get("side" if name == "top" else "top")
                tabs = each_tabs(v, whole_tabs(other) if other is not None else None)
                cs = fit.apply_tabs(cs, tabs, v["U"], v["V"])
                if name == "top":
                    top_tabs = tabs
        if cs is None:
            r = fit.rounded_box(v["outer"], v["U"], v["V"])
            if r is not None:  # овал или правоъгълник със заоблени ъгли, които разпознаването на прави не хваща
                cs = [(0.0, 0.0, r), (v["U"], 0.0, r), (v["U"], v["V"], r), (0.0, v["V"], r)]
        if cs is None:
            d["views"][name] = free_view(d, v, a, b, name)
            continue
        d["views"][name] = view_pts(d, cs, a, b, v["U"], v["V"], name, radius_dims)
    if top_tabs and "side" not in views and not flat:
        # По-тесният край обикновено е по-тънък със същото (без снимка отпред не се вижда): оценка, която се мери.
        tabs = [t | {"lo": min((t["lo"] + t["hi"]) / 2, 0.3 * H), "hi": min((t["lo"] + t["hi"]) / 2, 0.3 * H)}
                for t in top_tabs]  # fmt: skip
        cs = fit.apply_tabs([(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, H, 0.0), (0.0, H, 0.0)], tabs, L, H)
        d["views"]["side"] = view_pts(d, cs, "x", "z", L, H, "side", radius_dims)
    group_holes(d)
    repair(d)
    D.unused_cleanup(d)
    return d


MIN_RADIUS = 0.3  # мм: по-малка оценка за радиус, свита да се събере, е остър ъгъл


def fits(d: dict) -> bool:
    try:
        D.check(d)
        return True
    except D.DesignError:
        return False


def repair(d: dict) -> None:
    """Оценките от снимката, които не се събират, се поправят една по една, а не цялото на кутия: първо радиусите
    (заобляне, по-голямо от стената до него: оценка 3,6 мм на стена от 3 мм развали целия ъгъл-скоба), после само
    изгледите, които още не стават. Кутия е само последното."""
    if fits(d):
        return
    for name in ("top", "side", "end"):
        if d["views"].get(name, {}).get("pts") and not d["views"][name].get("round"):
            shrink_radii(d, name)
    for i, _, now in D.fit_radii(d):  # и застъпените дъги (външният радиус на извивка по-голям от вътрешния и стената)
        d["dims"][i]["est"] = now
    if fits(d):
        return
    boxed = []
    for name in ("end", "side", "top"):  # от по-малко сигурния изглед (краят е най-малък на снимката)
        if fits(d):
            break
        spec = d["views"].get(name, {})
        if spec.get("round") or not spec.get("pts"):
            continue
        a, b = D.VIEW_AXES[name]
        if spec["pts"] != D.box(a, b):
            d["views"][name] = {"pts": D.box(a, b)}
            boxed.append(D.VIEW_NAME[name])
    D.unused_cleanup(d)
    if not fits(d):  # например елемент извън детайла: махат се елементите, изгледите остават
        d["features"] = [f for f in d["features"] if f["t"] not in ("hole", "rect", "slot")]
        D.unused_cleanup(d)
    if boxed:
        d["notes"].append(f"Очертанието {' и '.join(boxed)} от снимките не се събра с другите: там е правоъгълник; "
                          "добавете стъпалата ръчно, ако има.")  # fmt: skip


def shrink_radii(d: dict, view: str) -> None:
    """Радиусите на изгледа, свити колкото да се съберат дъгите на всяка страна (с малко резерв); твърде малките
    стават остри ъгли. Общ размер за няколко ъгъла взема най-малката стойност."""
    import math

    pts = D.outline(d, view)
    rows = d["views"][view]["pts"]
    n = len(pts)
    if n < 3:
        return
    cut, tan = [], []
    for k, (u, v, r) in enumerate(pts):
        pu, pv, _ = pts[k - 1]
        nu, nv, _ = pts[(k + 1) % n]
        ang = D.corner_angle((pu, pv), (u, v), (nu, nv))
        t = math.tan(ang / 2) if 1e-3 < ang < math.pi - 1e-3 else 0.0
        tan.append(t)
        cut.append(r / t if r > 0 and t > 0 else (math.inf if r > 0 else 0.0))
    scale = [1.0] * n
    for k in range(n):
        if pts[k][2] <= 0:
            continue
        for j in (k - 1, k):  # двете страни на ъгъла: (k-1, k) и (k, k+1)
            a, b = j % n, (j + 1) % n
            side = math.dist(pts[a][:2], pts[b][:2])
            need = cut[a] + cut[b]
            if need > 0.95 * side:
                scale[k] = min(scale[k], 0.95 * side / need if math.isfinite(need) else 0.0)
    best: dict[str, float] = {}
    for k, row in enumerate(rows):
        rid = row[2]
        if rid:
            best[rid] = min(best.get(rid, math.inf), pts[k][2] * scale[k])
    for k, row in enumerate(rows):
        rid = row[2]
        if rid and best[rid] < MIN_RADIUS:
            row[2] = None
    for rid, r in best.items():
        if r >= MIN_RADIUS and r < D.value(d, rid):
            d["dims"][rid]["v"] = d["dims"][rid]["est"] = round(r, 2)


def whole_tabs(v: dict) -> set[int]:
    """Краищата, по-тесни на общото очертание на изгледа."""
    return {t["end"] for t in fit.end_tabs(v["outer"], v["U"], v["V"])}


def each_tabs(v: dict, other: set[int] | None = None) -> list[dict]:
    """По-тесните краища на изгледа: от общото очертание и от всяка снимка поотделно (на двете противоположни
    снимки стъпалото може да е от различни страни и обединението да го скрие). За всеки край - средното от
    снимките, които го показват. Когато има две снимки, край, който е по-тесен само на едната (а не на общото
    очертание), е от перспективата: по-високата част по средата (стената на Т-профил) е по-близо до телефона и
    изглежда по-дълга от основата, като щепсел. Истинският по-тесен край (флашка) е и на двете, а ако се е видял
    само на едната (сянка, пръсти), и в другия изглед (other: по-тесните краища отгоре за изгледа отпред и обратно;
    флашката е по-тясна и по-тънка в същия край, а стената на Т-профила изглежда по-дълга само отгоре)."""
    found: dict[int, list] = {}
    whole = {t["end"] for t in fit.end_tabs(v["outer"], v["U"], v["V"])}
    each = list(v.get("each", []))
    seen: dict[int, int] = {}
    for k, outline in enumerate([v["outer"]] + each):
        for t in fit.end_tabs(outline, v["U"], v["V"]):
            found.setdefault(t["end"], []).append(t)
            if k:
                seen[t["end"]] = seen.get(t["end"], 0) + 1
    out = []
    for end, ts in found.items():
        if len(each) >= 2 and end not in whole and seen.get(end, 0) < 2 and other is not None and end not in other:
            continue
        out.append({"end": end, **{k: float(np.mean([t[k] for t in ts])) for k in ("len", "lo", "hi")},
                    **{k: float(np.median([t.get(k, t["len"]) for t in ts])) for k in ("len_lo", "len_hi")}})
    return out


def view_pts(d: dict, cs: list, a: str, b: str, U: float, V: float, name: str, radius_dims: list) -> dict:
    """Ъглите (u, v, r) в мм -> изгледът на описанието: нива (стъпала) и размери за радиусите."""
    pts = []
    n = len(cs)
    for k, (u, w, r) in enumerate(cs):
        lu = D.level_for(d, a, clamp(u, U), step_name(a), step_what(a), view=name)
        lv = D.level_for(d, b, clamp(w, V), step_name(b), step_what(b), view=name)
        (pu, pw, _), (nu, nw, _) = cs[k - 1], cs[(k + 1) % n]
        outward = (u - pu) * (nw - w) - (w - pw) * (nu - u) > 0  # изпъкнал ъгъл (обратно на часовника)
        rid = radius_dim(d, radius_dims, r, name, outward) if r > 0 else None
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


def radius_dim(d: dict, known: list, r: float, view: str, outward: bool = True) -> str:
    """Размерът за заоблен ъгъл: близките радиуси в един изглед, заоблени в една посока, са един размер (едно
    мерене); в различните изгледи са различни ръбове (например ъглите отгоре и заоблените страни), а вътрешната и
    външната дъга на завой (кука) са различни радиуси, затова и различни размери."""
    for value, i, out in known:
        if (d["dims"][i]["show"].get("view") == view and out == outward
                and abs(value - r) <= RADIUS_GROUP * max(value, r)):  # fmt: skip
            return i
    i = D.add_dim(d, "r", r, "Радиус на заоблен ъгъл", "Радиус на заобления ъгъл",
                  "Сравнете с радиусомер или свредла; ако не можете да го измерите, оставете оценката.", "radius",
                  optional=True, show={"view": view, "radius": True})  # fmt: skip
    known.append((r, i, outward))
    return i


def free_view(d: dict, v: dict, a: str, b: str, view: str) -> dict:
    """Свободна форма (кука, дъга): гладка крива през точки от очертанието на снимката (fit.smooth_outline), които
    следват общите размери и не се питат."""
    pts, corners = fit.smooth_outline(v["outer"], v["U"], v["V"])
    rows = [[D.frac_level(a, u, v["U"]), D.frac_level(b, w, v["V"]), None] for u, w in pts]
    return {"pts": rows, "free": True, "curve": True, "corners": corners}


FACE_OF = {("z", "+"): "top", ("z", "-"): "bottom", ("y", "-"): "front", ("y", "+"): "back", ("x", "-"): "left",
           ("x", "+"): "right"}  # fmt: skip


def add_notch(d: dict, c: dict, axis: str, view: str) -> None:
    """Канал по цялата дължина на оста `axis`, отворен към стена (fit.notches): правоъгълен изрез, който излиза
    извън детайла, с мерки за ширината и дълбочината му и за положението му."""
    a, b = D.plane_axes(axis)
    key, sign = c["open"]
    deep = a if key == "u" else b  # оста на дълбочината
    across = b if key == "u" else a
    face = FACE_OF[(deep, sign)]
    fid = D.new_feature_id(d, "rect")
    f = {"id": fid, "t": "rect", "axis": axis, "side": "+", "view": view, "face": face, "open": {key: sign}}
    width = (c["v1"] - c["v0"]) if key == "u" else (c["u1"] - c["u0"])
    depth = (c["u1"] - c["u0"]) if key == "u" else (c["v1"] - c["v0"])
    w = D.add_dim(d, "w", width, "Ширина на канала", f"Измерете ширината на канала в {D.FACE_NAME[face]}",
                  "С вътрешните челюсти на шублера.", show={"feature": fid, "along": across})  # fmt: skip
    t = D.add_dim(d, "h", depth, "Дълбочина на канала", f"Измерете дълбочината на канала от {D.FACE_NAME[face]}",
                  "С дълбокомера на шублера.", show={"feature": fid, "along": deep})  # fmt: skip
    f["w"], f["h"] = (t, w) if key == "u" else (w, t)
    f[key] = {"lvl": f"{deep}0", "anchor": "min"} if sign == "-" else {"lvl": D.end_level(deep), "anchor": "max"}
    other = "v" if key == "u" else "u"
    lo, hi = (c["v0"], c["v1"]) if other == "v" else (c["u0"], c["u1"])
    tot = D.total(d, across)
    mid = (lo + hi) / 2
    if abs(mid - tot / 2) <= D.CENTER_TOL * tot:
        f[other] = {"lvl": f"{across}c", "anchor": "center"}
    else:
        near_start = mid < tot / 2
        lvl = D.level_for(d, across, clamp(lo if near_start else hi, tot), "Положение на канала", "ръба на канала",
                          view=view, merge=False)  # fmt: skip
        f[other] = {"lvl": lvl, "anchor": "min" if near_start else "max"}
    d["features"].append(f)


BLIND_DEPTH = 1.5  # оценката за дълбочината на глух отвор: толкова пъти по-малкия му размер, до половината детайл


BOSS_NAMES = (("на отвора", "на издатината"), ("Отвор", "Издатина"), ("отвора", "издатината"),
              ("Дълбочина", "Височина"), ("дълбочината", "височината"), ("дълбокомера", "дълбокомера или челюстите"))


def add_boss(d: dict, found: dict, size: float) -> None:
    """Издатина на тънък детайл (камерата на телефон, service.bump): очертанието отгоре, на стената си, с
    височина. Като глух отвор (размерите и положението се питат), но стърчи навън: "outside" - извън измерената
    дебелина (тя е на тялото), иначе в нея (тялото е по-тънко около издатината)."""
    add_hole(d, np.asarray(found["pts"], float), "z", "top", size, blind=found["face"])
    f = d["features"][-1]
    f["boss"], f["outside"] = True, bool(found["outside"])
    dim = d["dims"][f["depth"]]
    dim["v"] = dim["est"] = round(float(found["height"]), 2)
    keys = [f[k] for k in ("d", "w", "h", "len", "r", "depth") if f.get(k)]
    keys += [d["levels"][p["lvl"]]["dim"] for p in (f["u"], f["v"]) if p["lvl"] in d["levels"]]
    for i in keys:
        for field in ("name", "ask", "hint"):
            for a, b in BOSS_NAMES:
                d["dims"][i][field] = d["dims"][i][field].replace(a, b)


DETAIL = {  # детайл по стената (app/cad/details.py): думата в размерите, оценката за дълбочината
    "port": ("порта", None), "panel": ("панела", 0.5), "button": ("бутона", 0.3), "dot": ("отворчето", 1.0),
}  # fmt: skip


def add_detail(d: dict, pts: np.ndarray, axis: str, view: str, size: float, face: str, kind: str) -> None:
    """Детайл по стената, видян на снимката (екранът, порт, бутон, светодиод): глух елемент от стената с размерите
    и положението си и с дълбочина за мерене. Порт: дълбок колкото е дълъг (USB-A около 12 мм), до 45% от детайла;
    останалите - плитки (екранът е малко вдлъбнат, бутонът - с процеп). 0 за размер го маха."""
    if kind not in DETAIL:
        return
    add_hole(d, pts, axis, view, size, blind=face, shape=fit.detail_shape(pts))
    f = d["features"][-1]
    f["kind"] = kind
    word, depth = DETAIL[kind]
    if depth is None:
        long = max(d["dims"][f[k]]["v"] for k in ("d", "w", "h", "len") if f.get(k))
        depth = max(3.0, long)
    dim = d["dims"][f["depth"]]
    dim["v"] = dim["est"] = round(float(min(depth, 0.45 * D.total(d, axis))), 2)
    keys = [f[k] for k in ("d", "w", "h", "len", "r", "depth") if f.get(k)]
    keys += [d["levels"][p["lvl"]]["dim"] for p in (f["u"], f["v"]) if p["lvl"] in d["levels"]]
    for i in keys:
        for field in ("name", "ask", "hint"):
            d["dims"][i][field] = d["dims"][i][field].replace("отвора", word).replace("канала", word)


def add_hole(d: dict, pts: np.ndarray, axis: str, view: str, size: float, blind: str | None = None,
             shape: dict | None = None) -> None:
    """Отвор (кръгъл, канал или правоъгълен) по оста `axis`: проходен, или глух от стената `blind` (с дълбочина за
    мерене; оценката е само приблизителна - отвътре отворът не се вижда)."""
    shape = shape or fit.hole_shape(pts, size, D.total(d, axis))
    a, b = D.plane_axes(axis)
    fid = D.new_feature_id(d, shape["t"])
    face = blind or {"z": "top", "y": "front", "x": "left"}[axis]
    f = {"id": fid, "t": shape["t"], "axis": axis, "side": D.FACES[face][1] if blind else "+", "view": view,
         "face": face}  # fmt: skip
    if blind:
        small = shape["d"] if shape["t"] == "hole" else min(shape.get("w", 0.0), shape.get("h") or shape["w"])
        f["depth"] = D.add_dim(d, "t", min(BLIND_DEPTH * small, D.total(d, axis) / 2), "Дълбочина на отвора",
                               f"Измерете дълбочината на отвора от {D.FACE_NAME[face]}", "С дълбокомера на шублера.",
                               show={"feature": fid})  # fmt: skip
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
        if shape.get("shape"):  # по формата от снимката, вписана в ширината и височината
            f["shape"], f["corners"] = shape["shape"], shape["corners"]
        elif shape.get("r"):
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
    cs = fit.corners(top["outer"], top["U"], top["V"])
    if cs is not None:  # по-тесен край (върхът на химикалка): късо стъпало, което разпознаването на прави пропуска
        cs = fit.apply_tabs(cs, each_tabs(top), top["U"], top["V"])
    prof = profile_corners(top["outer"], top["U"], top["V"], cs)
    if prof is None:  # гладък цилиндър или капсула, които разпознаването на прави едва не хваща: еднакви краища
        r = fit.rounded_box(top["outer"], top["U"], top["V"])
        if r is not None:
            prof = [(0.0, top["V"] / 2, r), (top["U"], top["V"] / 2, r)]
    if prof is None:
        prof = [(x, r, 0.0) for x, r in fit.turned_profile(top["outer"], top["U"], top["V"])]
    Dm = D.value(d, "D")
    # Радиусите по групи: най-големият е D/2, другите са нови диаметри за измерване.
    radii: list[tuple[float, str]] = [(Dm / 2, "rD")]
    rounds: list[tuple[float, str, bool]] = []

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
            for value, i, _ in rounds:
                if abs(value - rad) < 1e-6:
                    return i
            i = D.add_dim(d, "r", rad, "Радиус на заобления край", "Радиус на заобления край",
                          "Ако краят е полукръг, оставете оценката (половината диаметър).", "radius", optional=True,
                          show={"view": "profile", "radius": True})  # fmt: skip
            rounds.append((rad, i, True))
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
