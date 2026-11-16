"""Цилиндричен детайл (третият вариант в приложението: капачка, втулка, ролка, вал, буркан): три снимки отгоре с
телефона успореден на масата -
1. "top": предметът изправен на дъното си, горният край към телефона;
2. "bottom": обърнат, дъното към телефона (по желание: без нея дъното е гладко);
3. "front": легнал на една страна, горният край надясно (профилът).

Моделът е тяло от струг (design "turned") с ос по височината: x0 е дъното, xL е горният край (d["upright"]; файловете
се строят изправени). Профилът (всеки диаметър и всяко стъпало по височината) е от снимката отстрани; отворите в
краищата - от снимките отгоре и отдолу: отвор, през който се вижда масата, е проходен; тъмен кръг е глух, с
дълбочина за мерене; отвор на едно и също място отгоре и отдолу е проходен. Всеки размер е въпрос с оценка от
снимките, а моделът е от измереното с шублера, затова е 1:1 колкото мерките.

Мащабът на краищата: снимките са с перспектива (телефонът е на 20-40 см) и очертанието отгоре е най-широкото място
на тялото на неговата височина, а не самият край. Височината на телефона се намира от профила, фокусното разстояние
(EXIF) и големината на очертанието (end_scale), така че горният край (на височина H) и масата (през проходния отвор)
са всеки в своя мащаб: отвор в тесния горен край на шише не излиза по-голям, отколкото е."""

import math

import numpy as np
from shapely.geometry import Polygon

from . import assemble as AS
from . import design as D
from . import fit
from . import photos as P

NAMES = ("top", "bottom", "front")
TITLE = {"top": "отгоре", "bottom": "отдолу", "front": "отстрани"}
SYM_OK = 0.9  # очертанието отстрани е огледално спрямо оста поне толкова (IoU): тази ос е оста на детайла
SYM_BAD = 0.8  # а спрямо другата - под толкова
SYM_WARN = 0.85  # и по-малко от толкова спрямо избраната: предупреждение (не е тяло от струг)
ASPECT_SURE = math.log(1.15)  # височина и диаметър, различни с поне толкова: оста е по пропорциите на снимката
ROUND_WARN = 0.06  # очертанието отгоре е толкова далеч от кръг: предупреждение
ON_AXIS = 0.05  # отвор с център до толкова от диаметъра от оста е по оста
SAME_SPOT = 0.08  # отвори отгоре и отдолу до толкова от диаметъра един от друг са едно място (проходен отвор)
SAME_DIAM = 0.3  # и с диаметри, различни с не повече от толкова: един отвор
COUNTERBORE = 1.3  # тъмен кръг около проходен отвор, по-голям поне толкова пъти: разширение (глух по-голям отвор)
CIRCLE_GROUP = 0.05  # отвори встрани от оста на окръжности, различни до толкова от диаметъра: една окръжност
EVEN_DEG = 8.0  # ъглите на отворите по окръжност, до толкова от равномерните: равномерни
BLIND_DEPTH = 1.0  # оценката за дълбочината на глух отвор: толкова диаметъра (отвътре не се вижда)


def side_axes(f: P.Frame, sizes: tuple[float, float] | None = None, swap: bool = False) -> tuple:
    """Оста на детайла на снимката отстрани (eu, от дъното към горния край) и посоката напряко (ev), в осите на
    снимката с y нагоре. Коя от двете оси на очертанието: по пропорциите, ако височината и диаметърът са измерени и
    ясно различни; иначе тази, спрямо която очертанието е огледално (а спрямо другата не е: стъпала); иначе по-близката
    до хоризонтала (приложението казва "легнал, горният край надясно"). Накъде е горният край: надясно на снимката
    (swap: обратно - обръщане на ръка в приложението)."""
    cands = [f.l, f.s]

    def perp(e):
        return np.array([-e[1], e[0]])

    def sym(e):
        poly = f.normalized(e, perp(e))
        return AS.mirror_iou(np.asarray(poly.exterior.coords), 1.0) if not poly.is_empty else 0.0

    def aspect(e):
        return math.log(f.ext(e) / max(f.ext(perp(e)), 1e-9))

    eu = max(cands, key=lambda e: abs(float(e[0])))
    if sizes and abs(math.log(sizes[0] / sizes[1])) > ASPECT_SURE:
        want = math.log(sizes[0] / sizes[1])
        eu = min(cands, key=lambda e: abs(aspect(e) - want))
    else:
        s = [sym(e) for e in cands]
        for k in (0, 1):
            if s[k] >= SYM_OK and s[1 - k] < SYM_BAD:
                eu = cands[k]
    eu = np.asarray(eu, float).copy()
    if abs(eu[0]) >= 0.3:
        eu = eu if eu[0] > 0 else -eu  # горният край надясно
    else:
        eu = eu if eu[1] > 0 else -eu  # оста е отвесно на снимката: горният край нагоре
    if swap:
        eu = -eu
    return eu, perp(eu), sym(eu)


def rot2(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def layout(shots: dict, sizes: tuple[float, float] | None = None, flips: dict | None = None) -> dict:
    """Като photos.layout, но за трите снимки на цилиндъра: {"orient": {снимка: {"view", "eu", "ev", "primary"}},
    "warnings", "cylinder": True}. Снимката отстрани е изгледът "top" на тялото от струг (профилът), отгоре и отдолу
    - краищата ("end"). sizes: (височина, диаметър), ако са измерени: тогава и мащабът на краищата (end_scale) и
    как е завъртяна снимката отдолу спрямо тази отгоре (по отворите), в "A" (от мм на края към пиксели)."""
    if "front" not in shots:
        raise D.DesignError("Липсва снимката отстрани: без нея не се вижда профилът на детайла.")
    flips = flips or {}
    f = P.Frame(shots["front"])
    eu, ev, symmetry = side_axes(f, sizes, bool(flips.get("side", [False, False])[0]))
    orient = {"front": {"view": "top", "eu": [float(x) for x in eu], "ev": [float(x) for x in ev], "primary": True}}
    warnings = []
    if symmetry < SYM_WARN:
        warnings.append("Отстрани предметът не е еднакъв от двете страни на оста: ако не е кръгъл (струг), "
                        "изберете „Обемен предмет“. Ако е, снимайте го отстрани отново, легнал.")  # fmt: skip
    for name in ("top", "bottom"):
        if name not in shots:
            continue
        orient[name] = {"view": "end", "eu": [1.0 if name == "top" else -1.0, 0.0], "ev": [0.0, 1.0],
                        "primary": name == "top"}  # fmt: skip
        g = P.Frame(shots[name])
        if abs(math.log(g.ext(g.l) / max(g.ext(g.s), 1e-9))) > math.log(1.12) or P.roundness(
                g.normalized(g.l, g.s)) > ROUND_WARN:  # fmt: skip
            warnings.append(f"Снимката {TITLE[name]} не е кръг: снимайте я отново точно отгоре, с предмета в "
                            "средата на рамката.")  # fmt: skip
    lay = {"orient": orient, "warnings": warnings, "cylinder": True, "round": {"top": False, "end": True},
           "iou": {}, "same": {}, "sure": {}, "alts": {}, "ends": {}}  # fmt: skip
    if sizes:
        H, Dm = sizes
        prof = profile_of(side_shot(shots["front"], orient["front"], H, Dm), orient["front"], H, Dm)
        ends = {}
        for name in ("top", "bottom"):
            if name in shots:
                ends[name] = end_scale(shots[name], prof, H, Dm, upside_down=name == "bottom")
        if "top" in ends and "bottom" in ends:
            turn = bottom_turn(shots, ends, Dm)
            o = orient["bottom"]
            o["eu"] = [float(x) for x in rot2(turn) @ np.asarray(o["eu"])]
            o["ev"] = [float(x) for x in rot2(turn) @ np.asarray(o["ev"])]
        for name, sc in ends.items():
            orient[name]["A"] = end_matrix(orient[name], sc, Dm).tolist()
            orient[name]["scale"] = {"c": [float(x) for x in sc["c"]], "face": float(sc["face"]),
                                     "ground": float(sc["ground"])}  # fmt: skip
    return lay


def overall(lay: dict) -> list[dict]:
    """Общите размери за мерене: диаметърът (линия през кръга на снимката отгоре) и височината (по оста на снимката
    отстрани)."""
    o = lay["orient"]
    out = [{"id": "D", "name": "Най-голям диаметър", "ask": "Измерете диаметъра по оранжевата линия.",
            "hint": "Най-дебелото място на детайла, с външните челюсти на шублера.",
            "photo": "top" if "top" in o else "front", "along": "u"}]  # fmt: skip
    out.append({"id": "H", "name": "Височина", "ask": "Измерете височината по оранжевата линия.",
                "hint": "От дъното до горния край, по оста (както стои изправен).", "photo": "front",
                "along": "u"})  # fmt: skip
    return out


def resolve(values: dict) -> tuple[float, float]:
    """(височина, диаметър) от мерките (H или L за височината)."""
    H = values.get("H") or values.get("L")
    Dm = values.get("D")
    if not (H and Dm):
        raise ValueError("missing")
    return float(H), float(Dm)


# --- Мащабът на краищата ---------------------------------------------------------------------------------------------


def profile_of(shot: dict, o: dict, H: float, Dm: float) -> list[tuple[float, float]]:
    """Профилът (x от дъното, радиус) в мм от снимката отстрани."""
    pl = P.placement(shot, o, H, Dm)
    try:
        prof = fit.turned_profile(np.asarray(pl["outer"]), H, Dm)
    except Exception:  # noqa: BLE001 - очертание, което не се срязва: цилиндър
        prof = []
    return prof if len(prof) >= 2 else [(0.0, Dm / 2), (H, Dm / 2)]


SIDE_AXIAL = 0.7  # точка от очертанието, чиято нормала е поне толкова по оста: от челото на края или на стъпало
SIDE_NEAR = 0.04  # радиусът на челото: най-големият в толкова от височината около точката


def side_shot(shot: dict, o: dict, H: float, Dm: float) -> dict:
    """Снимката отстрани с очертанието, върнато на масата (без перспективата). Легналият детайл е с оста на
    височина Dm/2: страните му са на тази височина, а челата (краищата и стъпалата) са кръгове напряко на оста. Челото,
    което не гледа към телефона, се вижда с горната си половина (по-близо до телефона, по-голяма): краят се издува
    като дъга и острият ъгъл изглеждаше заоблен (R 3,9 мм на остър ръб). Челото, което гледа към телефона, се вижда
    до долната си половина: стъпалото изглеждаше по-навътре (8 вместо 6 мм). Всяка точка се връща на масата с
    височината си: страните - оста, челата - оста плюс или минус sqrt(R^2 - y^2) (R - радиусът на челото, y - колко е
    встрани от оста)."""
    w, h_img = shot["size"]
    fpx = (shot.get("f35") or P.F35_DEFAULT) * math.hypot(w, h_img) / 43.27
    q = np.asarray(shot["outer"], float) * [1.0, -1.0]
    eu, ev = np.asarray(o["eu"], float), np.asarray(o["ev"], float)
    a, b = q @ eu, q @ ev
    s = float(np.ptp(b)) / Dm  # пиксела на мм на височината на оста (най-широкото място е там)
    if s <= 0 or len(q) < 20:
        return shot
    zc = fpx / s + Dm / 2
    prof = profile_of(shot, o, H, Dm)
    xs = (a - a.min()) / max(float(np.ptp(a)), 1e-9) * H
    ys = (b - (b.max() + b.min()) / 2) / s
    px = np.array([p[0] for p in prof])
    pr = np.array([p[1] for p in prof])
    grid = np.linspace(0, H, 400)
    rr = np.interp(grid, px, pr)
    near = max(1, int(SIDE_NEAR * len(grid)))
    big = np.array([rr[max(0, k - near): k + near + 1].max() for k in range(len(grid))])
    r_face = np.interp(xs, grid, big)
    n = P.normals(q, False)
    c = np.array([w / 2, -h_img / 2])
    axial = np.abs(n @ eu) >= SIDE_AXIAL
    seen = np.einsum("ij,ij->i", n, c - q) > 0
    rise = np.sqrt(np.maximum(r_face**2 - ys**2, 0.0))
    hh = np.where(axial, Dm / 2 + np.where(seen, -rise, rise), Dm / 2)
    k = max(3, len(q) // 150) | 1
    hh = np.minimum(np.convolve(np.r_[hh[-k:], hh, hh[:k]], np.ones(k) / k, "same")[k:-k], 0.9 * zc)
    back = c + (q - c) * ((zc - hh) / zc)[:, None]
    return shot | {"outer": (back * [1.0, -1.0]).tolist()}


def end_scale(shot: dict, prof: list, H: float, Dm: float, upside_down: bool = False) -> dict:
    """Мащабът на снимка отгоре или отдолу: {"c" (средата на края в пиксели), "face" (пиксела на мм в края, на
    височина H над масата), "ground" (на масата)}. Очертанието е най-голямото от кръговете на профила, всеки на
    своята височина (по-високото е по-близо до телефона и изглежда по-голямо): оттам е височината на телефона Zc.
    upside_down: снимката отдолу (детайлът е обърнат: x от профила е на височина H - x)."""
    w, h = shot["size"]
    fpx = (shot.get("f35") or P.F35_DEFAULT) * math.hypot(w, h) / 43.27
    cx, cy, R, _ = fit.circle_fit(np.asarray(shot["outer"], float))
    simple = {"c": np.array([cx, cy]), "face": R / (Dm / 2), "ground": R / (Dm / 2)}
    xs = np.array([p[0] for p in prof], float)
    rs = np.array([p[1] for p in prof], float)
    if len(xs) < 2 or rs.max() <= 0 or R <= 0:
        return simple
    t = np.linspace(0, H, 200)  # по-гъсто, за да няма скок между ъглите
    rs = np.interp(t, xs, rs)
    heights = H - t if upside_down else t

    def seen(zc):
        return float(np.max(rs * fpx / (zc - heights)))

    lo, hi = H + 1e-3, H + 1e5
    if seen(hi) > R:
        return simple
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if seen(mid) > R else (lo, mid)
    zc = (lo + hi) / 2
    if zc - H < 0.15 * zc:  # телефонът почти до предмета: нещо не е наред, мащабът на очертанието
        return simple
    widest = float(heights[int(np.argmax(rs / (zc - heights)))])
    c0 = np.array([w / 2, h / 2])
    c = c0 + (np.array([cx, cy]) - c0) * (zc - widest) / (zc - H)  # оста на височина H, а не на най-широкото
    return {"c": c, "face": fpx / (zc - H), "ground": fpx / zc}


def end_matrix(o: dict, sc: dict, Dm: float) -> np.ndarray:
    """A (2 x 3): от мм на края (u, v от 0 до Dm, средата в Dm/2) към пиксели на снимката (x надясно, y надолу)."""
    eu, ev = np.asarray(o["eu"], float), np.asarray(o["ev"], float)
    k, (cx, cy) = sc["face"], sc["c"]
    return np.array([[eu[0] * k, ev[0] * k, cx - (eu[0] + ev[0]) * k * Dm / 2],
                     [-eu[1] * k, -ev[1] * k, cy + (eu[1] + ev[1]) * k * Dm / 2]])  # fmt: skip


def to_face(px: np.ndarray, o: dict, sc: dict, k: float, Dm: float) -> np.ndarray:
    """Пиксели на снимката -> мм на края (u, v), с мащаб k пиксела на мм около средата на края."""
    eu, ev = np.asarray(o["eu"], float), np.asarray(o["ev"], float)
    w = (np.asarray(px, float) - sc["c"]) * [1.0, -1.0] / k
    return np.c_[w @ eu, w @ ev] + Dm / 2


# --- Отворите в краищата ---------------------------------------------------------------------------------------------


def end_holes(shot: dict, o: dict, sc: dict, Dm: float) -> list[dict]:
    """Отворите на снимка отгоре или отдолу: [{"u", "v" (мм на края), "d", "through" (вижда се масата)}].
    Проходният е ограничен от далечния си край (на масата), затова диаметърът е в мащаба на масата, а мястото - между
    двата края. Тъмен кръг (сив, не печат: photos.dark_holes) е в мащаба на края."""
    out = []
    mid = (sc["face"] + sc["ground"]) / 2
    for h in shot.get("holes", []):
        pts = np.asarray(h, float)
        if len(pts) < 8 or Polygon(pts).buffer(0).area <= 0:
            continue
        c = fit.robust_circle(pts) or fit.circle_fit(pts)[:3]
        if c is None:
            continue
        hx, hy, r = c
        u, v = to_face(np.array([[hx, hy]]), o, sc, mid, Dm)[0]
        d = min(2 * r / sc["ground"], 0.95 * Dm)
        out.append({"u": float(u), "v": float(v), "d": float(d), "through": True})
    outer_mm = to_face(np.asarray(shot["outer"], float), o, sc, sc["face"], Dm)
    p = {"outer": outer_mm, "marks": [to_face(np.asarray(m, float), o, sc, sc["face"], Dm) for m in
                                      shot.get("marks", []) if len(m) >= 4],
         "mark_info": [i for m, i in zip(shot.get("marks", []), shot.get("mark_info") or [])
                       if len(m) >= 4]}  # fmt: skip
    for m in P.dark_holes(p, [], Dm, Dm, strict=False):
        c = fit.robust_circle(np.asarray(m)) or fit.circle_fit(np.asarray(m))[:3]
        if c is None or fit.circle_fit(np.asarray(m))[3] > 0.12:
            continue  # не е кръг (печат, сянка)
        u, v, r = c
        out.append({"u": float(u), "v": float(v), "d": float(min(2 * r, 0.95 * Dm)), "through": False})
    return out


def bottom_turn(shots: dict, ends: dict, Dm: float) -> float:
    """С колко градуса е завъртяна снимката отдолу спрямо тази отгоре (около оста): при което отворите встрани от
    оста съвпадат (проходните са на едно място и на двете). Без такива: 0 (ъгълът по окръжността и без това се
    взима от снимката отгоре)."""
    o_top = {"eu": [1.0, 0.0], "ev": [0.0, 1.0]}
    top = [h for h in end_holes(shots["top"], o_top, ends["top"], Dm) if off_axis(h, Dm)]
    o0 = {"eu": [-1.0, 0.0], "ev": [0.0, 1.0]}
    bot = [h for h in end_holes(shots["bottom"], o0, ends["bottom"], Dm) if off_axis(h, Dm)]
    if not top or not bot:
        return 0.0
    cap = 0.15 * Dm
    best = (math.inf, 0.0)
    a = np.array([[h["u"] - Dm / 2, h["v"] - Dm / 2] for h in top])
    b0 = np.array([[h["u"] - Dm / 2, h["v"] - Dm / 2] for h in bot])
    for deg in range(360):
        # Осите на снимката отдолу, завъртени с deg (огледални): точката в мм на края се завърта със същото.
        b = b0 @ rot2(deg).T
        dist = np.linalg.norm(a[:, None] - b[None], axis=2)
        cost = float(np.minimum(dist.min(1), cap).sum() + np.minimum(dist.min(0), cap).sum())
        if cost < best[0] - 1e-9:
            best = (cost, float(deg))
    return best[1]


def off_axis(h: dict, Dm: float) -> bool:
    return math.hypot(h["u"] - Dm / 2, h["v"] - Dm / 2) > ON_AXIS * Dm


def end_features(d: dict, shots: dict, lay: dict, Dm: float) -> None:
    """Отворите в краищата като елементи на тялото: проходни (вижда се масата, или са на едно място отгоре и
    отдолу), глухи (тъмен кръг само от едната страна) и разширения около проходен отвор."""
    found = {}
    for name in ("top", "bottom"):
        o = lay["orient"].get(name)
        if o is None or "A" not in o or name not in shots:
            continue
        sc = {"c": np.asarray(o["scale"]["c"], float), "face": o["scale"]["face"], "ground": o["scale"]["ground"]}
        found[name] = end_holes(shots[name], o, sc, Dm)
    top, bot = found.get("top", []), found.get("bottom", [])
    used = set()
    holes = []  # (face, u, v, d, through)
    for h in top:
        match = None
        for j, g in enumerate(bot):
            if j in used:
                continue
            near = math.hypot(h["u"] - g["u"], h["v"] - g["v"]) <= SAME_SPOT * Dm + 0.3 * max(h["d"], g["d"])
            if near and abs(h["d"] - g["d"]) <= SAME_DIAM * max(h["d"], g["d"]):
                match = j
                break
        if match is not None:
            used.add(match)
            g = bot[match]
            dia = (h["d"] + g["d"]) / 2 if h["through"] == g["through"] else (h if h["through"] else g)["d"]
            holes.append(("right", (h["u"] + g["u"]) / 2, (h["v"] + g["v"]) / 2, dia, True))
        else:
            holes.append(("right", h["u"], h["v"], h["d"], h["through"]))
    for j, g in enumerate(bot):
        if j not in used:
            holes.append(("left", g["u"], g["v"], g["d"], g["through"]))
    # Тъмен кръг около проходен отвор (разширение за глава на винт): глух, по-голям; иначе същото място като
    # проходен отвор е същият отвор.
    through = [h for h in holes if h[4]]
    kept = []
    for h in holes:
        if h[4]:
            kept.append(h)
            continue
        inner = [t for t in through if math.hypot(h[1] - t[1], h[2] - t[2]) <= SAME_SPOT * Dm + 0.3 * t[3]]
        if inner and all(h[3] < COUNTERBORE * t[3] for t in inner):
            continue
        kept.append(h)
    for face, u, v, dia, thru in sorted(kept, key=lambda h: (not h[4], -h[3])):
        add_end_hole(d, face, u, v, dia, thru, Dm)
    group_circles(d)


def add_end_hole(d: dict, face: str, u: float, v: float, dia: float, through: bool, Dm: float) -> None:
    """Отвор в края `face` ("right": горният, "left": дъното): по оста или встрани от нея на окръжност (полярно:
    диаметърът на окръжността през центровете и ъгълът от снимката)."""
    fid = D.new_feature_id(d, "hole")
    f = {"id": fid, "t": "hole", "axis": "x", "side": D.FACES[face][1], "face": face, "view": "end",
         "u": {"lvl": "yc", "anchor": "center"}, "v": {"lvl": "zc", "anchor": "center"}}  # fmt: skip
    f["d"] = D.add_dim(d, "d", dia, "Диаметър на отвора", "Измерете диаметъра на отвора",
                       "С горните (вътрешните) челюсти на шублера.", "diameter", show={"feature": fid})  # fmt: skip
    r = math.hypot(u - Dm / 2, v - Dm / 2)
    if r > ON_AXIS * Dm:
        f["polar"] = {"pcd": D.add_dim(d, "p", 2 * r, *PCD_TEXT, "diameter", show={"feature": fid, "pcd": True}),
                      "a": round(math.degrees(math.atan2(v - Dm / 2, u - Dm / 2)) % 360, 1)}  # fmt: skip
    if not through:
        f["depth"] = D.add_dim(d, "t", min(BLIND_DEPTH * dia, 0.9 * D.value(d, "L")), "Дълбочина на отвора",
                               f"Измерете дълбочината на отвора от {D.face_name(d, face)}",
                               "С дълбокомера на шублера (тънката пръчка отзад).", show={"feature": fid})  # fmt: skip
    d["features"].append(f)


PCD_TEXT = ("Окръжност на отворите (през центровете)", "Измерете разстоянието между центровете през оста",
            "От левия ръб на един отвор до левия ръб на срещуположния (това е разстоянието между центровете). При "
            "един отвор: два пъти разстоянието от оста до центъра му.")  # fmt: skip


def group_circles(d: dict) -> None:
    """Отворите встрани от оста на една окръжност (до CIRCLE_GROUP от диаметъра) имат един размер за окръжността,
    а ъглите им, ако са почти равномерни, стават равномерни (4 отвора през 90 градуса)."""
    Dm = D.value(d, "D")
    polar = [f for f in d["features"] if f.get("polar")]
    groups: list[list[dict]] = []
    for f in polar:
        v = d["dims"][f["polar"]["pcd"]]["v"]
        for g in groups:
            if g[0]["face"] == f["face"] and abs(d["dims"][g[0]["polar"]["pcd"]]["v"] - v) <= CIRCLE_GROUP * Dm:
                g.append(f)
                break
        else:
            groups.append([f])
    for g in groups:
        if len(g) < 2:
            continue
        keep = g[0]["polar"]["pcd"]
        vals = [d["dims"][f["polar"]["pcd"]]["v"] for f in g]
        d["dims"][keep]["v"] = d["dims"][keep]["est"] = round(float(np.mean(vals)), 2)
        d["dims"][keep]["name"] = f"Окръжност на отворите ({len(g)} бр., през центровете)"
        for f in g[1:]:
            f["polar"]["pcd"] = keep
        n = len(g)
        g.sort(key=lambda f: f["polar"]["a"])
        rel = np.array([(f["polar"]["a"] - 360.0 * k / n) for k, f in enumerate(g)])
        base = math.degrees(math.atan2(np.sin(np.radians(rel)).mean(), np.cos(np.radians(rel)).mean()))
        off = (rel - base + 180) % 360 - 180
        if np.all(np.abs(off) <= EVEN_DEG):
            for k, f in enumerate(g):
                f["polar"]["a"] = round((base + 360.0 * k / n) % 360, 1)
    AS.group_holes(d)
    D.unused_cleanup(d)


# --- Описанието -------------------------------------------------------------------------------------------------------


def assemble(shots: dict, values: dict, flips: dict | None = None) -> tuple[dict, list[str]]:
    """Описанието на цилиндъра от снимките и измерените височина и диаметър: (описанието, предупрежденията)."""
    H, Dm = resolve(values)
    lay = layout(shots, (H, Dm), flips)
    o = lay["orient"]["front"]
    pl = P.placement(side_shot(shots["front"], o, H, Dm), o, H, Dm)
    side = {"outer": np.asarray(pl["outer"], float), "holes": [np.asarray(h, float) for h in pl["holes"]],
            "U": H, "V": Dm, "each": [], "blind": []}  # fmt: skip
    d = AS.turned({"top": side}, H, Dm, Dm, upright=True)
    d["notes"] = [n for n in d["notes"] if "снимка от края" not in n]
    try:
        end_features(d, shots, lay, Dm)
    except Exception:  # noqa: BLE001 - отворите в краищата са допълнение: без тях, ако снимките не се четат
        import logging

        logging.getLogger("uvicorn.error").exception("отворите в краищата не се прочетоха")
        d["notes"].append("Отворите в краищата не се разчетоха от снимките: добавете ги ръчно, ако има.")
    if "bottom" not in shots:
        d["notes"].append("Няма снимка отдолу: дъното е гладко, освен проходните отвори. Ако има отвор само в дъното, "
                          "добавете го ръчно.")  # fmt: skip
    d["flips"] = {k: v for k, v in (flips or {}).items() if k == "side"}
    return d, lay["warnings"]
