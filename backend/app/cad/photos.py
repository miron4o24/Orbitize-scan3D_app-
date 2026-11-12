"""Снимките на инженерния режим: до шест, всяка отгоре с телефона успореден на масата (нивелирът го пази), а
предметът обърнат с една от страните си нагоре: отгоре, отпред, отзад, отдясно, отляво и отдолу.

Коя снимка кой изглед на детайла е и как е завъртяна (без да разчитаме, че човекът е спазил посоките точно):
1. Снимката отгоре определя осите: дължината (X) е по-дългата страна на очертанието, ширината (Y) е напряко,
   височината (Z) е към телефона.
2. За всяка друга снимка: как би лежал детайлът, ако е обърнат, както казва приложението (отпред: претърколен от
   нас, отдясно: изправен на левия край, ...). Оттам е кой изглед е (отгоре, отпред или отстрани) и накъде са осите
   му на снимката; после всяка ос се изправя по очертанието на тази снимка.
3. Пропорциите проверяват това: дължина/ширина отгоре, дължина/височина отпред и ширина/височина отстрани трябва да
   си пасват (а след мерките и с тях). Ако не, се взима подреждането, което пасва: детайлът е бил обърнат иначе.
4. Накъде е нагоре (височината): детайлът стои на долната си страна, която обикновено е пълна права.
5. Снимките на две противоположни страни са огледални една на друга: втората се нагласява към първата по най-голямо
   съвпадение и очертанията им се обединяват. Перспективата (по-близкото изглежда по-голямо) само смалява
   по-далечните части спрямо най-близката страна, различни при двете снимки, затова обединението е по-вярно.
6. Перспективата се поправя и направо (views_mm): всяка точка се връща на масата с височината си, взета от формата,
   сглобена от другите снимки. За всеки изглед се взима по-простото от двете очертания (с и без поправката).

Всичко е в пиксели на снимката с y нагоре (q = (x, -y)), докато не дойдат мерките: тогава всеки изглед става мм."""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure

NAMES = ("top", "front", "back", "right", "left", "bottom")
LEGACY = {"side": "front", "end": "left"}  # имената от първия вариант (три снимки)
TITLE = {"top": "отгоре", "front": "отпред", "back": "отзад", "right": "отдясно", "left": "отляво", "bottom": "отдолу"}
VIEWS = ("top", "side", "end")  # изгледите на описанието (design.py)
PAIRS = {"top": ("top", "bottom"), "side": ("front", "back"), "end": ("right", "left")}  # първата е основната
AXES = {"top": ("x", "y"), "side": ("x", "z"), "end": ("y", "z")}
SCALE_COST = 0.5  # на единица log от разликата в мащаба между снимките над SCALE_FREE
SCALE_FREE = 1.6  # толкова по-отблизо или по-отдалеч от снимката отгоре е нормално (рамката, различни страни)
PRIOR_COST = 0.08  # колко (в log на пропорциите) струва да не вярваме на посоките от приложението
FUSE_IOU = 0.86  # под това съвпадение две противоположни снимки не се усредняват
MATCH_IOU = 0.8  # под това съвпадение (в мм) противоположните снимки не са едно и също
# Перспективата смалява по-далечното, затова двете противоположни снимки на висок предмет (ъгъл, стоящ на края си)
# се различават само с вдлъбнатини: едното очертание е почти цялото в другото. Тогава са една страна и
# обединението им е истинското очертание, дори при по-малко съвпадение.
CONTAIN = 0.96  # толкова от по-малкото очертание е в другото
CONTAIN_IOU = 0.6  # но не по-малко съвпадение от това
RATIO_WARN = 1.15  # пропорциите на снимката и мерките се разминават повече: предупреждение
RATIO_DROP = 1.25  # още повече: снимката не е ползвана (под ъгъл или друга страна)
MARK_WALL = 0.015  # тъмно петно по-близо до ръба (дял от размера) не е отвор
SURE_MARGIN = 0.01  # с толкова по-добро съвпадение от следващото обръщане е ясно как е обърната снимката
ROUND_RMS = 0.02  # очертание, нормирано до квадрат, по-близко до кръг: кръгло
END_EVIDENCE = 0.3  # снимката отстрани пасва на снимката отгоре обърната с поне толкова по-добре: обърнат е
SQUARE = math.log(1.15)  # ширината и височината се различават по-малко: по пропорциите не личи как е завъртян краят
TURN_EVIDENCE = 0.4  # краят, завъртян с 90 градуса, пасва на снимките отгоре и отпред с поне толкова по-добре
TURN_SHAPE = 0.006  # профил отгоре или отпред, по-равен от това (разсейване), не казва нищо за завъртането


def rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = round(math.cos(a), 12), round(math.sin(a), 12)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# Как е обърнат детайлът от положението на снимката отгоре (оси на снимката: x надясно, y нагоре, z към телефона).
MOVE = {
    "top": np.eye(3),
    "front": rot("x", -90),  # претърколен от нас: предната страна (към нас) гледа нагоре
    "back": rot("x", 90),  # към нас: задната страна нагоре
    "right": rot("y", -90),  # изправен на левия край: десният нагоре
    "left": rot("y", 90),
    "bottom": rot("x", 180),
}


class Frame:
    """Очертанието на една снимка с неговите оси: l по дължината (fit.main_axes), s напряко."""

    def __init__(self, shot: dict):
        self.q = np.asarray(shot["outer"], float) * [1.0, -1.0]
        self.holes = [np.asarray(h, float) * [1.0, -1.0] for h in shot.get("holes", [])]
        info = shot.get("mark_info") or [{} for _ in shot.get("marks", [])]
        pairs = [(np.asarray(m, float) * [1.0, -1.0], i) for m, i in zip(shot.get("marks", []), info) if len(m) >= 4]
        self.marks = [m for m, _ in pairs]
        self.mark_info = [i for _, i in pairs]
        self.lines = shot.get("lines") or {}
        self.looks = shot.get("looks") or {}
        self.size = tuple(shot["size"])
        from .fit import main_axes

        self.l, self.s, self.phi = main_axes(self.q)

    def ext(self, e: np.ndarray) -> float:
        return float(np.ptp(self.q @ e))

    def axes(self) -> list[np.ndarray]:
        return [self.l, -self.l, self.s, -self.s]

    def normalized(self, eu: np.ndarray, ev: np.ndarray, aspect: bool = False) -> Polygon:
        """Очертанието (с отворите) в осите (eu, ev), от 0 до 1 по всяка ос (aspect: по v до ext_v/ext_u)."""
        su, sv = self.ext(eu), self.ext(ev)
        a0, b0 = float((self.q @ eu).min()), float((self.q @ ev).min())
        kv = sv / su if aspect else 1.0

        def f(p):
            return np.c_[(p @ eu - a0) / su, (p @ ev - b0) / sv * kv]

        return largest(Polygon(f(self.q), [f(h) for h in self.holes]).buffer(0))


def largest(g) -> Polygon:
    """Многоъгълникът (най-голямото парче, ако buffer(0) е разделил очертанието)."""
    if g.geom_type == "Polygon":
        return g
    return max(getattr(g, "geoms", []), key=lambda p: p.area, default=Polygon())


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def expected(name: str, top: Frame) -> tuple[str, np.ndarray, np.ndarray]:
    """Изгледът и посоките на осите му на снимката `name`, ако детайлът е обърнат, както казва приложението."""
    snap = round(top.phi / 90) * 90
    R = MOVE[name] @ rot("z", snap)
    e = {"x": R[:, 0], "y": R[:, 1], "z": R[:, 2]}  # осите на детайла след обръщането, в осите на снимката
    normal = max(e, key=lambda k: abs(e[k][2]))
    view = {"z": "top", "y": "side", "x": "end"}[normal]
    a, b = AXES[view]
    return view, e[a][:2], e[b][:2]


def snap_axes(f: Frame, eu_star: np.ndarray, ev_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Най-близките до очакваните посоки оси на очертанието (eu по eu*, ev перпендикулярна, по ev*)."""
    eu = max(f.axes(), key=lambda c: float(c @ eu_star))
    ev = max((c for c in f.axes() if abs(float(c @ eu)) < 0.5), key=lambda c: float(c @ ev_star))
    return eu, ev


def turned90(eu: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Същото огледално или не, но с разменени оси: (ev, -eu)."""
    return ev.copy(), -eu


def resting_down(poly: Polygon) -> bool:
    """Очертание (u, v), обърнато наопаки по v: детайлът стои на долната си страна, която обикновено е пълна права
    по цялата дължина. Ако горната страна е по-пълна от долната с поне 20%, нагоре е надолу."""
    x0, y0, x1, y1 = poly.bounds
    U, V = x1 - x0, y1 - y0
    tol = 0.012 * V
    pts = np.asarray(poly.exterior.coords)
    low, high = pts[pts[:, 1] <= y0 + tol], pts[pts[:, 1] >= y1 - tol]
    bottom = float(np.ptp(low[:, 0])) if len(low) > 1 else 0.0
    top = float(np.ptp(high[:, 0])) if len(high) > 1 else 0.0
    return top > 1.2 * bottom + 0.02 * U


def iou(a: Polygon, b: Polygon) -> float:
    return a.intersection(b).area / max(a.union(b).area, 1e-12)


def inside(a: Polygon, b: Polygon) -> float:
    """Колко от по-малкото очертание е в другото."""
    return a.intersection(b).area / max(min(a.area, b.area), 1e-12)


def same_side(a: Polygon, b: Polygon, limit: float) -> bool:
    """Двете очертания (нормирани до едни размери) са една страна: съвпадат или се различават само с вдлъбнатини
    от перспективата (едното е в другото)."""
    return iou(a, b) >= limit or (iou(a, b) >= CONTAIN_IOU and inside(a, b) >= CONTAIN)


def roundness(poly: Polygon) -> float:
    """Относителното отклонение от кръг на очертание, нормирано до единичен квадрат (елипса също е "кръгла")."""
    pts = np.asarray(poly.exterior.coords)[:-1]
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    r = math.sqrt(max(c + cx**2 + cy**2, 1e-12))
    return float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)) / r)


# --- Подреждането ----------------------------------------------------------------------------------------------------


def layout(shots: dict, sizes: tuple[float, float, float] | None = None, flips: dict | None = None) -> dict:
    """shots: {име на снимка: vision.segment(...)}. Връща {"orient": {снимка: {"view", "eu", "ev", "primary"}},
    "round": {"top": bool, "end": bool}, "warnings": [...], "iou": {снимка: съвпадение с основната}}.
    sizes: (L, W, H), ако вече са измерени (по-сигурна проверка на пропорциите). flips: {изглед: [по u, по v]} от
    приложението (обръщане на ръка, след автоматичното)."""
    if "top" not in shots:
        raise ValueError("no top")
    frames = {k: Frame(v) for k, v in shots.items()}
    top = frames["top"]
    orient: dict[str, list] = {"top": ["top", top.l.copy(), top.s.copy()]}
    prior = {}
    for name, f in frames.items():
        if name == "top":
            continue
        view, eu_s, ev_s = expected(name, top)
        eu, ev = snap_axes(f, eu_s, ev_s)
        prior[name] = view
        orient[name] = [view, eu, ev]
    # Отдолу е изгледът отгоре: осите по пропорциите на снимката отгоре.
    r_t = math.log(top.ext(top.l) / max(top.ext(top.s), 1e-9)) if not sizes else math.log(sizes[0] / sizes[1])
    if "bottom" in orient:
        _, eu, ev = orient["bottom"]
        f = frames["bottom"]
        alt = turned90(eu, ev)
        if abs(log_aspect(f, *alt) - r_t) + PRIOR_COST < abs(log_aspect(f, eu, ev) - r_t):
            eu, ev = alt
        orient["bottom"] = ["top", eu, ev]
    side_names = [n for n in ("front", "back", "right", "left") if n in orient]
    best = assign(frames, orient, side_names, r_t, sizes)
    for name, (view, eu, ev) in best.items():
        orient[name] = [view, eu, ev]
    if end_turned(frames, orient, sizes):
        for name, o in orient.items():
            if o[0] == "end":
                o[1], o[2] = turned90(o[1], o[2])
    warnings: list[str] = []
    quality: dict[str, float] = {}
    ok: dict[str, bool] = {}
    sure: dict[str, bool] = {}  # обръщането на снимката е ясно (несиметричен предмет)
    alts: dict[str, list] = {}  # обръщанията (по u, по v), които пасват почти еднакво добре
    ends: dict[str, float] = {}  # same_end на основната снимка отстрани (за проверка)
    for view in VIEWS:
        members = [n for n in orient if orient[n][0] == view]
        if not members:
            continue
        primary = min(members, key=lambda n: (PAIRS[view].index(n) if n in PAIRS[view] else 9, NAMES.index(n)))
        _, eu, ev = orient[primary]
        if view in ("side", "end") and resting_down(frames[primary].normalized(eu, ev, aspect=True)):
            ev = -ev
        if view == "side":  # кой край е кой: приложението казва как да се обърне детайлът, но човек може да го завърти
            ends[primary] = round(same_end(top, orient["top"][1], frames[primary], eu), 3)
            if ends[primary] < -END_EVIDENCE:
                eu = -eu
        fu, fv = (flips or {}).get(view, [False, False])
        eu, ev = (-eu if fu else eu), (-ev if fv else ev)
        orient[primary] = [view, eu, ev]
        base = frames[primary].normalized(eu, ev)
        for other in members:
            if other == primary:
                continue
            _, ou, ov = orient[other]
            cands = [(su * ou, sv * ov) for su in (1, -1) for sv in (1, -1)]
            scores = [iou(base, frames[other].normalized(a, b)) for a, b in cands]
            k = int(np.argmax(scores))
            near = [j for j, sc in enumerate(scores) if sc >= scores[k] - SURE_MARGIN]
            decided = [False, False]  # по u и по v: от линиите през цялата ширина (шев, граница капачка/тяло)
            if len(near) > 1:  # почти симетрично очертание: кое обръщане пасва на линиите вътре
                cu, cv = cands[k]
                for i in (0, 1):
                    mine = (cu, cv)[i]
                    same = line_match(frames[primary], (eu, ev)[i], frames[other], mine)
                    flip = line_match(frames[primary], (eu, ev)[i], frames[other], -mine)
                    if abs(same - flip) >= 1:
                        decided[i] = True
                        if flip > same:
                            cu, cv = (-cu, cv) if i == 0 else (cu, -cv)
                # Отгоре и отдолу симетрично (флашка): скосеният край (стъпало, по-дълго от едната страна) е от една и
                # съща страна и на двете снимки, иначе обединението го скрива.
                if not decided[1] and any(np.allclose(cands[j][0], cu) and np.allclose(cands[j][1], -cv) for j in near):
                    mine, theirs = slant_sides(frames[primary], eu, ev), slant_sides(frames[other], cu, cv)
                    common = [e for e in mine if e in theirs]
                    if common and all(mine[e] != theirs[e] for e in common):
                        cv = -cv
                        decided[1] = True
                    elif common:
                        decided[1] = True
                k = next((j for j, c in enumerate(cands) if np.allclose(c[0], cu) and np.allclose(c[1], cv)), k)
            orient[other] = [view, *cands[k]]
            quality[other] = round(float(scores[k]), 3)
            ok[other] = same_side(base, frames[other].normalized(*cands[k]), FUSE_IOU)
            ck = cands[k]
            alts[other] = [[int(round(float(a @ ck[0]))), int(round(float(b @ ck[1])))] for (a, b), sc in
                           zip(cands, scores) if sc >= scores[k] - SURE_MARGIN]  # fmt: skip
            # Ос, решена от линиите, не е под съмнение.
            alts[other] = [al for al in alts[other] if not (decided[0] and al[0] < 0) and not (decided[1] and al[1] < 0)]
            rest = max((sc for j, sc in enumerate(scores) if j != k), default=0.0)
            # Отвор само на тази снимка се добавя, само ако е ясно как е обърната (от очертанието или линиите вътре;
            # иначе може да е от грешната страна; пропуснат отвор се добавя на ръка, а грешен би развалил детайла).
            sure[other] = bool(scores[k] - rest > SURE_MARGIN) or len(alts[other]) == 1
            if not ok[other]:
                warnings.append(f"Снимката {TITLE[other]} не съвпада добре със снимката {TITLE[primary]}: ползвам "
                                f"само {TITLE[primary]}. Ако е друга страна, снимайте я отново.")  # fmt: skip
        for n in members:
            orient[n].append(n == primary)
    out = {"orient": {n: {"view": v[0], "eu": [float(x) for x in v[1]], "ev": [float(x) for x in v[2]],
                          "primary": bool(v[3])} for n, v in orient.items()},
           "iou": quality, "same": ok, "sure": sure, "alts": alts, "warnings": warnings, "ends": ends}  # fmt: skip
    out["round"] = {"top": is_round(frames, orient, "top"), "end": is_round(frames, orient, "end")}
    return out


def end_turned(frames: dict, orient: dict, sizes) -> bool:
    """Краят с почти еднакви ширина и височина (точилка 40 x 40): по пропорциите не личи коя ос на снимката е
    ширината, а приложението не знае как човекът е изправил детайла. Ако е завъртян грешно, плоските страни и
    каналът са на грешните стени. Перспективата издава коя е: каналът по горната страна вдлъбва очертанието отгоре
    в края (дължината по ширината) там, където краят е по-нисък, а по-заоблените стени скъсяват очертанието отпред.
    Затова профилът на края по ширината му върви като профила отгоре по ширината, а по височината - като профила
    отпред по височината. Завърта се само при ясна разлика и ако профилите отгоре или отпред не са просто прави."""
    ends = [n for n, o in orient.items() if o[0] == "end"]
    if not ends or "top" not in orient:
        return False
    if sizes:
        square = abs(math.log(sizes[1] / sizes[2])) < SQUARE
    else:
        square = all(abs(log_aspect(frames[n], orient[n][1], orient[n][2])) < SQUARE for n in ends)
    if not square:
        return False

    def mid(p):
        n = len(p)
        return p[int(0.12 * n):int(0.88 * n) + 1]

    def match(a, b):  # посоката по оста още не е решена (resting_down, обръщанията по-долу)
        a, b = mid(a), mid(b)
        return max(corr(a, b), corr(a, b[::-1]))

    refs = [(ext_profile(frames["top"], orient["top"][2]), 1)]  # отгоре по ширината
    side = [n for n in ("front", "back") if n in orient and orient[n][0] == "side"]
    if side:
        refs.append((ext_profile(frames[side[0]], orient[side[0]][2]), 2))  # отпред по височината
    score = 0.0
    for n in ends:
        f, eu, ev = frames[n], orient[n][1], orient[n][2]
        for ref, axis in refs:
            if float(np.std(mid(ref))) < TURN_SHAPE:
                continue
            keep, turn = (eu, ev) if axis == 1 else (ev, eu)
            score += match(ref, ext_profile(f, turn)) - match(ref, ext_profile(f, keep))
    return score / len(ends) >= TURN_EVIDENCE


def ext_profile(f: Frame, e: np.ndarray, bins: int = 48) -> np.ndarray:
    """Колко широко е очертанието напряко на e на всяко от bins места по e (от по-малкото към по-голямото по e),
    като дял от най-широкото."""
    from .fit import resample

    pts = resample(np.asarray(largest(Polygon(f.q).buffer(0)).exterior.coords), max(f.ext(e) / (bins * 6), 1e-6))
    n = np.array([-e[1], e[0]])
    t, w = pts @ e, pts @ n
    k = np.clip(((t - t.min()) / max(float(np.ptp(t)), 1e-9) * bins).astype(int), 0, bins - 1)
    out = np.array([float(np.ptp(w[k == i])) if np.any(k == i) else np.nan for i in range(bins)])
    return out / max(float(np.nanmax(out)), 1e-9)


def look_profile(f: Frame, e: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Цветът (Lab, bins x 3) и пъстротата (bins) по посоката e (vision.looks), от по-малкото към по-голямото по e."""
    key, sign = ("l", float(e @ f.l)) if abs(float(e @ f.l)) > 0.5 else ("s", float(e @ f.s))
    prof = f.looks.get(key)
    if not prof:
        return None
    lab = np.array([c if c is not None else [np.nan] * 3 for c in prof["lab"]], float)
    tex = np.array([x if x is not None else np.nan for x in prof["tex"]], float)
    return (lab, tex) if sign > 0 else (lab[::-1], tex[::-1])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 8:
        return 0.0
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / d if d > 1e-9 else 0.0


def same_end(a: Frame, ea: np.ndarray, b: Frame, eb: np.ndarray) -> float:
    """Дали посоката eb на снимката b върви като ea на снимката a (обща ос, например дължината отгоре и отпред):
    > 0 - да, < 0 - обратно, около 0 - не личи. По три белега по дължината на оста, всеки сравнен направо и обърнат:
    очертанието (стеснен или заоблен край, стъпало), пъстротата (надписи, прозрачна част) и цветът (капачката)."""
    ext_a, ext_b = ext_profile(a, ea), ext_profile(b, eb)
    score = corr(ext_a, ext_b) - corr(ext_a, ext_b[::-1])
    la, lb = look_profile(a, ea), look_profile(b, eb)
    if la is not None and lb is not None and len(la[1]) == len(lb[1]):
        score += 0.5 * (corr(la[1], lb[1]) - corr(la[1], lb[1][::-1]))
        ca, cb = la[0] - np.nanmedian(la[0], 0), lb[0] - np.nanmedian(lb[0], 0)
        same = float(np.nanmean(np.linalg.norm(ca - cb, axis=1)))
        flip = float(np.nanmean(np.linalg.norm(ca - cb[::-1], axis=1)))
        if same + flip > 1e-9:
            score += 1.5 * (flip - same) / (flip + same)
    return score


def axis_profile(f: Frame, e: np.ndarray) -> np.ndarray | None:
    """Профилът на линиите на снимката по посоката e (от по-малкото към по-голямото по e)."""
    if not f.lines:
        return None
    if abs(float(e @ f.l)) > 0.5:
        prof, sign = f.lines.get("l"), float(e @ f.l)
    else:
        prof, sign = f.lines.get("s"), float(e @ f.s)
    if not prof:
        return None
    prof = np.asarray(prof, float)
    return prof if sign > 0 else prof[::-1]


def full_lines(prof: np.ndarray | None) -> list[float]:
    """Местата (0..1) на линиите през цялата ширина (шев, граница капачка/тяло), без самите краища."""
    if prof is None:
        return []
    n = len(prof)
    out = []
    for i in range(int(0.06 * n), int(0.94 * n)):
        if prof[i] >= 0.9 and prof[i] == prof[max(0, i - 4):i + 5].max():
            if not out or i / n - out[-1] > 0.03:
                out.append(i / n)
    return out


def line_match(a: Frame, ea: np.ndarray, b: Frame, eb: np.ndarray) -> int:
    """Колко от линиите през цялата ширина на основната снимка по оста ea имат линия наблизо (до 8%) на другата
    снимка по оста eb. Печатът и отворите не пресичат цялата ширина, затова не участват."""
    la, lb = full_lines(axis_profile(a, ea)), full_lines(axis_profile(b, eb))
    return sum(1 for x in la if any(abs(x - y) <= 0.08 for y in lb))


def slant_sides(f: Frame, eu: np.ndarray, ev: np.ndarray) -> dict[int, int]:
    """По-тесните краища с ясно по-дълго стъпало от едната страна (скосеният край на флашка): {край по u: +1, ако е
    по-дълго откъм по-голямото v, иначе -1}."""
    from .fit import end_tabs, tab_lengths

    poly = f.normalized(eu, ev, aspect=True)
    if poly.is_empty:
        return {}
    U = 100.0  # мащаб, в който допуските на end_tabs (в мм) са малки спрямо очертанието
    q = np.asarray(poly.exterior.coords) * U
    V = float(np.ptp(q[:, 1]))
    out = {}
    for t in end_tabs(q, U, V):
        a, b = tab_lengths(t, U)
        if a != b:
            out[t["end"]] = 1 if b > a else -1
    return out


def log_aspect(f: Frame, eu: np.ndarray, ev: np.ndarray) -> float:
    return math.log(f.ext(eu) / max(f.ext(ev), 1e-9))


def assign(frames: dict, orient: dict, names: list[str], r_t: float, sizes) -> dict:
    """Кой изглед (отпред или отстрани) е всяка снимка встрани и коя ос е височината: подреждането, при което
    пропорциите си пасват (с мерките, ако ги има), като отклонението от посоките на приложението струва PRIOR_COST.
    Двете снимки на противоположни страни са в един изглед."""
    if not names:
        return {}
    pairs = [p for p in (("front", "back"), ("right", "left")) if any(n in names for n in p)]
    prior_view = {p: orient[next(n for n in p if n in names)][0] for p in pairs}
    top = frames.get("top")
    log_h = math.log(sizes[2] / sizes[0]) if sizes else None  # log(H/L)
    best, best_score = None, math.inf
    view_options = [prior_view[p] for p in pairs]
    alternatives = [view_options]
    if len(pairs) == 2 and view_options[0] != view_options[1]:
        alternatives.append(view_options[::-1])
    elif len(pairs) == 1 and sizes:
        alternatives.append(["end" if view_options[0] == "side" else "side"])
    for views in alternatives:
        changed_views = sum(v != prior_view[p] for v, p in zip(views, pairs))
        if len(views) == 2 and views[0] == views[1]:
            continue
        view_of = {n: views[k] for k, p in enumerate(pairs) for n in p if n in names}
        for mask in range(1 << len(names)):
            cand, preds, cost = {}, [], PRIOR_COST * changed_views
            for i, n in enumerate(names):
                view = view_of[n]
                _, eu, ev = orient[n]
                if mask >> i & 1:
                    eu, ev = turned90(eu, ev)
                    cost += PRIOR_COST
                m = log_aspect(frames[n], eu, ev)
                # log(H/L): отпред m = log(L/H); отстрани m = log(W/H) = log(L/H) - r_t
                preds.append(-m if view == "side" else -m - r_t)
                cand[n] = (view, eu, ev)
                # Мащабът: телефонът е на подобна височина за всяка снимка (рамката), затова общата с отгоре страна
                # (дължината отпред, ширината отстрани) е горе-долу със същия размер в пиксели. Подреждане, при което
                # една снимка би била няколко пъти по-отблизо, е невъзможно (например "високо" 125 мм вместо 8).
                if top is not None:
                    shared = top.ext(top.l) if view == "side" else top.ext(top.s)
                    ratio = abs(math.log(frames[n].ext(eu) / max(shared, 1e-9)))
                    cost += SCALE_COST * max(0.0, ratio - math.log(SCALE_FREE))
            ref = log_h if log_h is not None else float(np.median(preds))
            score = sum(abs(p - ref) for p in preds) + cost
            if score < best_score - 1e-9:
                best, best_score = cand, score
    return best


def is_round(frames: dict, orient: dict, view: str) -> bool:
    """Изгледът е кръг (или елипса): отгоре - изправен цилиндър, отстрани - легнал."""
    names = [n for n, o in orient.items() if o[0] == view and len(o) > 3 and o[3]]
    if not names:
        return False
    f = frames[names[0]]
    _, eu, ev = orient[names[0]][:3]
    if abs(log_aspect(f, eu, ev)) > math.log(1.07):
        return False
    poly = f.normalized(eu, ev)
    return roundness(poly) < ROUND_RMS


# --- Към мм -----------------------------------------------------------------------------------------------------------


def placement(shot: dict, o: dict, U: float, V: float, depth: dict | None = None) -> dict:
    """Изгледът на снимката в мм: очертанието и отворите (u, v от 0 до U и V) и матрицата A (2 x 3) от мм на изгледа
    към пиксели на снимката (x надясно, y надолу; без перспективата - за рисуване). depth: височините към
    телефона (heights), за поправка на перспективата."""
    f = Frame(shot)
    eu, ev = np.asarray(o["eu"], float), np.asarray(o["ev"], float)
    a, b = f.q @ eu, f.q @ ev
    a0, a1, b0, b1 = float(a.min()), float(a.max()), float(b.min()), float(b.max())
    ka, kb = (a1 - a0) / U, (b1 - b0) / V
    A = np.array([[eu[0] * ka, ev[0] * kb, eu[0] * a0 + ev[0] * b0],
                  [-eu[1] * ka, -ev[1] * kb, -(eu[1] * a0 + ev[1] * b0)]])  # fmt: skip

    def ortho(p):
        return np.c_[(p @ eu - a0) / ka, (p @ ev - b0) / kb]

    outer, holes = ortho(f.q), [ortho(h) for h in f.holes]
    marks = [ortho(m) for m in f.marks]
    ratio = ka / kb
    if depth is not None:
        outer, holes, ratio = unproject(shot, f, eu, ev, outer, holes, U, V, math.sqrt(ka * kb), depth)
    return {"outer": outer, "holes": holes, "marks": marks, "mark_info": f.mark_info, "A": A, "U": U, "V": V,
            "ratio": ratio, "ortho": ortho(f.q), "own": outer}  # fmt: skip


# --- Перспективата ---------------------------------------------------------------------------------------------------
# Телефонът е на 15-40 см над масата, затова по-високото на предмета изглежда по-голямо (и по-далеч от средата на
# кадъра) от по-ниското: краят на легнал цилиндър се издува, стъпалото от двете страни на снимката е с различен
# мащаб. Поправката: всяка точка от очертанието се връща на масата с височината си към телефона, взета от формата,
# сглобена от другите снимки (heights), и с фокусното разстояние от EXIF (иначе обичайното за телефон). Два пъти,
# защото формата се уточнява.

F35_DEFAULT = 26.0  # мм, основната камера на повечето телефони
SPIKE = 0.01  # по-тънкото от това (дял от по-малкия размер) във върнатото на масата очертание е шип, не детайл
GROW = 0.025  # раздуване на изгледите при сглобяване на формата за височините (дял от размера)
NEAR = {"top": (2, 1), "side": (1, -1), "end": (0, 1)}  # оста към телефона и от коя страна гледа изгледът (det > 0)


def hull(views: dict, L: float, W: float, H: float, cells: int = 150) -> tuple[np.ndarray, float]:
    """Формата като кубчета: сечението на изтеглените изгледи (липсващ изглед: пълен)."""
    step = max(L, W, H) / cells
    n = {"x": max(1, math.ceil(L / step)), "y": max(1, math.ceil(W / step)), "z": max(1, math.ceil(H / step))}

    def raster(view: str) -> np.ndarray:
        a, b = AXES[view]
        if view not in views:
            return np.ones((n[a], n[b]), bool)
        im = Image.new("L", (n[a], n[b]), 0)
        draw = ImageDraw.Draw(im)
        v = views[view]
        draw.polygon([(u / step, w / step) for u, w in np.asarray(v["outer"])], fill=1)
        for h in v["holes"]:
            draw.polygon([(u / step, w / step) for u, w in np.asarray(h)], fill=0)
        return np.asarray(im, bool).T

    # Изгледите са леко раздути (с GROW от размера): малка грешка в един изглед (ъгъл, отрязан от перспективата)
    # иначе би изрязала колоните на другите и би сменила височините им.
    grow = max(1, round(GROW * cells))
    T, F, E = (ndimage.binary_dilation(raster(v), iterations=grow) for v in ("top", "side", "end"))
    return T[:, :, None] & F[:, None, :] & E[None, :, :], step


def heights(occ: np.ndarray, step: float, view: str, mirrored: bool, total: float) -> tuple[np.ndarray, np.ndarray]:
    """Височините над масата (мм) на най-близката до телефона и на най-далечната точка на всяка колона на изгледа
    (u, v). Празните колони (извън формата) вземат височините на най-близката пълна: точка от очертанието на
    снимката, малко извън сглобената форма, е от нейния ръб, а не от масата."""
    axis, side = NEAR[view]
    col = np.moveaxis(occ, axis, -1)
    n = col.shape[-1]
    has = col.any(-1)
    if not has.any():
        return np.zeros(has.shape), np.zeros(has.shape)
    lo = np.argmax(col, -1) * step  # най-малката координата по оста
    hi = (n - np.argmax(col[..., ::-1], -1)) * step  # най-голямата
    if side * (-1 if mirrored else 1) > 0:  # гледаме от положителната страна на оста: близо е най-голямата координата
        near, far = hi, lo
    else:
        near, far = total - lo, total - hi
    near, far = np.clip(near, 0, total), np.clip(far, 0, total)
    # Точката от очертанието на снимката е от колона по ръба на формата (гледана от телефона): при права стена
    # колоната е цялата стена, а при заоблена повърхност (цилиндър, купол) е тънка около най-широкото място. Затова
    # височината е от най-близката колона по ръба, а не от вътрешна (там куполът е много по-висок).
    edge = has & ~ndimage.binary_erosion(has, border_value=0)
    if edge.any() and not edge.all():
        idx = tuple(ndimage.distance_transform_edt(~edge, return_distances=False, return_indices=True))
        near, far = near[idx], far[idx]
    return near, far


def normals(p: np.ndarray, hole: bool) -> np.ndarray:
    """Нормалите навън от материала (y нагоре) в точките на затворено очертание: при отвор - към отвора."""
    k = max(2, len(p) // 200)
    t = np.roll(p, -k, axis=0) - np.roll(p, k, axis=0)
    nrm = np.c_[t[:, 1], -t[:, 0]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)[:, None]
    x, y = p[:, 0], p[:, 1]
    ccw = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) > 0
    return nrm * (1 if ccw != hole else -1)


def unproject(shot, f: Frame, eu, ev, outer, holes, U, V, scale, depth) -> tuple[np.ndarray, list, float]:
    """Очертанието (и отворите) в мм, върнати на масата с височината си: точката p от снимката е на масата в
    c + (p - c) * (Zc - h) / Zc (с точност до мащаба), където c е средата на кадъра и Zc височината на телефона
    (от фокусното разстояние и мащаба на снимката). Ръбът на очертанието е от най-близката до телефона точка на
    колоната, освен където стената гледа към телефона (вътре в изрез, далечната страна на отвор): там се вижда
    стената до долу и ръбът е от най-далечната. Височините по очертанието се изглаждат (преходите стават плавни, без
    скокове), а размерите се мерят без най-крайните 0,3% от точките; накрая пак се нормира до U x V."""
    near, far, step = depth["near"], depth["far"], depth["step"]
    w, h_img = f.size
    fpx = (shot.get("f35") or F35_DEFAULT) * math.hypot(w, h_img) / 43.27
    c = np.array([w / 2, -h_img / 2])

    def height(p, uv, hole):
        iu = np.clip((uv[:, 0] / step).astype(int), 0, near.shape[0] - 1)
        iv = np.clip((uv[:, 1] / step).astype(int), 0, near.shape[1] - 1)
        k = max(5, len(p) // 120) | 1
        seen = np.einsum("ij,ij->i", normals(p, hole), c - p) > 0  # стената гледа към телефона
        seen = ndimage.median_filter(seen.astype(np.uint8), size=k, mode="wrap") > 0
        h = np.where(seen, far[iu, iv], near[iu, iv])
        return ndimage.uniform_filter1d(h, size=k, mode="wrap")

    h_out = height(f.q, outer, False)
    if not np.any(h_out > 0):
        return outer, holes, 1.0
    zc = fpx / scale + float(np.max(h_out))  # мащабът на снимката е при най-близката до телефона част

    def back(p, hh):
        hh = np.minimum(hh, 0.9 * zc)
        return c + (p - c) * ((zc - hh) / zc)[:, None]

    q = despike(back(f.q, h_out), eu, ev)
    hs = [back(hq, height(hq, huv, True)) for hq, huv in zip(f.holes, holes)]
    a, b = q @ eu, q @ ev
    a0, a1 = np.percentile(a, [0.3, 99.7])
    b0, b1 = np.percentile(b, [0.3, 99.7])

    def mm(p):
        return np.c_[np.clip((p @ eu - a0) / (a1 - a0), 0, 1) * U, np.clip((p @ ev - b0) / (b1 - b0), 0, 1) * V]

    return mm(q), [mm(x) for x in hs], float(((a1 - a0) / U) / ((b1 - b0) / V))


def despike(q: np.ndarray, eu: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """Очертанието, върнато на масата, без шиповете: стена, видяна отстрани (ивицата между близкия и далечния ръб),
    се връща на масата като линия без дебелина, която стърчи от ъгъла. Шипът пречи да се познаят ъглите и, ако е
    най-външното, разтяга мащаба (ъгълът излизаше с крак 36,5 вместо 40 мм). Махат се частите, по-тънки от SPIKE от
    по-малкия размер; острите ъгли остават остри. Само ако шиповете разтягат размера: иначе нищо не се пипа."""
    from shapely.validation import make_valid

    poly = make_valid(Polygon(q))
    a, b = q @ eu, q @ ev
    e = SPIKE * min(np.ptp(a), np.ptp(b))
    if e <= 0:
        return q
    opened = poly.buffer(-e, join_style=2).buffer(e, join_style=2)
    if opened.is_empty or opened.area < 0.9 * poly.area:
        return q  # нещо по-голямо от шип (тънък детайл): както е
    out = np.asarray(largest(opened).exterior.coords)[:-1]
    if len(out) < 8:
        return q
    if max(np.ptp(a) - np.ptp(out @ eu), np.ptp(b) - np.ptp(out @ ev)) < 0.5 * e:
        return q  # шиповете (ако има) не са най-външното: очертанието остава, както е (ъглите се познават и така)
    from .fit import resample

    return resample(np.r_[out, out[:1]], float(np.hypot(np.ptp(a), np.ptp(b))) / max(len(q), 200))


def view_sizes(view: str, L: float, W: float, H: float) -> tuple[float, float]:
    return {"top": (L, W), "side": (L, H), "end": (W, H)}[view]


def views_mm(shots: dict, lay: dict, L: float, W: float, H: float) -> tuple[dict, list[str]]:
    """Изгледите в мм за assemble, всеки по два начина: направо от снимките и с поправена перспектива (по формата
    от предишното сглобяване, три пъти). За всеки изглед се взима по-простото очертание (с по-малко ъгли):
    перспективата добавя лъжливи стъпала, които са само в единия вариант, а истинските са и в двата. Две
    противоположни снимки на един изглед се обединяват."""
    plain, warn_plain = combine(shots, lay, L, W, H, None, fuse_pairs=True)
    fixed, warn_fixed = combine(shots, lay, L, W, H, None, fuse_pairs=False)
    for k in range(3):
        fixed, warn_fixed = combine(shots, lay, L, W, H, hull(fixed, L, W, H), fuse_pairs=k == 2)
    out, warnings = {}, []
    for view in VIEWS:
        v, warn = plain.get(view), warn_plain.get(view)
        f = fixed.get(view)
        if v is not None and f is not None and complexity(f) < complexity(v):
            v, warn = f | {"perspective": True}, warn_fixed.get(view)
        if v is not None:
            out[view] = v
        if warn:
            warnings.append(warn)
    return out, warnings


def complexity(v: dict) -> int:
    """Колко ъгъла има очертанието (по-малко: по-просто); свободна форма е най-сложна."""
    from .fit import corners

    cs = corners(np.asarray(v["outer"]), v["U"], v["V"])
    return 1000 if cs is None else len(cs) + sum(1 for c in cs if c[2] > 0)


def combine(shots: dict, lay: dict, L: float, W: float, H: float, solid, fuse_pairs: bool) -> tuple[dict, dict]:
    """Изгледите от снимките (solid: формата като кубчета за височините при поправката на перспективата) и
    предупрежденията за пропорциите по изгледи."""
    out, warnings = {}, {}
    totals = {"top": H, "side": W, "end": L}
    for view in VIEWS:
        names = [n for n, o in lay["orient"].items() if o["view"] == view]
        if not names:
            continue
        U, V = view_sizes(view, L, W, H)

        def place(name, view=view, U=U, V=V):
            o = lay["orient"][name]
            depth = None
            if solid is not None:
                mirrored = o["eu"][0] * o["ev"][1] - o["eu"][1] * o["ev"][0] < 0
                near, far = heights(solid[0], solid[1], view, mirrored, totals[view])
                depth = {"near": near, "far": far, "step": solid[1]}
            return placement(shots[name], o, U, V, depth)

        def fits(pl) -> bool:
            return 1 / RATIO_DROP <= pl["ratio"] <= RATIO_DROP

        primary = next(n for n in names if lay["orient"][n]["primary"])
        p = place(primary)
        others = [n for n in names if n != primary and lay.get("same", {}).get(n, lay["iou"].get(n, 0) >= FUSE_IOU)]
        if not fits(p):  # основната снимка не пасва на мерките: другата на същия изглед, ако пасва, иначе нищо
            alt = [n for n in names if n != primary]
            q = place(alt[0]) if alt else None
            if q is None or not fits(q):
                bad = " и ".join(TITLE[n] for n in names)
                many = len(names) > 1
                warnings[view] = (f"{'Снимките' if many else 'Снимката'} {bad} не {'пасват' if many else 'пасва'} на "
                                  "мерките (снимано под ъгъл или друга страна) и не са ползвани: тази страна е "
                                  "правоъгълник. Снимайте отново отгоре, с телефона прав, за да се разчете.")  # fmt: skip
                continue
            primary, p, others = alt[0], q, []
        if not 1 / RATIO_WARN <= p["ratio"] <= RATIO_WARN:
            warnings[view] = (f"Пропорциите на снимката {TITLE[primary]} се разминават с мерките с "
                              f"{abs(1 - p['ratio']) * 100:.0f}%: снимката може да е под ъгъл или мярка да е грешна.")  # fmt: skip
        used = [primary]
        if fuse_pairs and others:
            q = place(others[0])
            pa = largest(Polygon(p["outer"]).buffer(0))
            qa = largest(Polygon(q["outer"]).buffer(0))
            if fits(q) and same_side(pa, qa, MATCH_IOU):
                fused = fuse(p, q, U, V)
                if fused is not None:
                    extra = lay.get("sure", {}).get(others[0], False)
                    holes = hole_union(p["holes"], q["holes"], extra, p["outer"], q["outer"])
                    alts = [[1, 1]] if extra else lay.get("alts", {}).get(others[0], [[1, 1]])
                    if not extra:
                        holes += confirmed(p, q, alts, U, V)
                    holes += mark_pairs(p, q, holes, alts, U, V)
                    p = p | {"outer": fused["outer"], "holes": holes}
                    used.append(others[0])
        p_holes = widen(p["holes"], p.get("marks", []), U, V)
        p_holes += dark_holes(p, p_holes, U, V)
        blind = blind_holes(lay, names, primary, place, fits, p, p_holes, U, V)
        if view == "end":
            for b in blind:
                b["face"] = end_face(shots, lay, b["face"])
        # Всяка снимка поотделно (за по-тесните краища): с поправената перспектива, ако е поправена. Иначе по-високата
        # част по средата (стената на Т-профил) изглежда по-дълга от основата и прилича на щепсел в края.
        key = "ortho" if solid is None else "own"
        each = [p[key]] + ([place(others[0])[key]] if others else [])
        out[view] = {"outer": p["outer"], "holes": p_holes, "U": U, "V": V, "from": used, "each": each,
                     "blind": blind}  # fmt: skip
    return out, warnings


def hole_union(a: list, b: list, extra: bool = True, outer_a=None, outer_b=None) -> list:
    """Отворите от две противоположни снимки: през проходен отвор с перспектива се вижда само сечението на близкия
    и далечния ръб, изместено на различни страни на двете снимки. Обединението на двете е по-близо до истинския
    отвор (средното би го смалило). Отвор само на втората снимка също остава (extra), ако е ясно как е обърната:
    тесен и дълбок отвор встрани от средата на кадъра може изобщо да не се види през детайла на другата. При
    симетричен предмет не е ясно от коя страна е, затова тогава не се добавя."""
    out = []
    used = set()
    for h in a:
        ph = largest(Polygon(h).buffer(0))
        best, k_best = None, None
        for k, g in enumerate(b):
            pg = largest(Polygon(g).buffer(0))
            if k in used or ph.area <= 0 or pg.area <= 0:
                continue
            # Един отвор: застъпват се поне наполовина или средите им са по-близо от радиуса на по-големия (от
            # двете страни перспективата го измества различно и по-малкият може да е наполовина извън другия).
            if same_hole(ph, pg, 0.5):
                best, k_best = pg, k
                break
        if best is None:
            out.append(np.asarray(h))
            continue
        used.add(k_best)
        u = largest(ph.union(best))
        if outer_a is not None and outer_b is not None:
            # Стената до ръба на детайла не е по-тънка от по-дебелата на двете снимки: на едната отворът може да
            # "изтече" към ръба (сянка, тъмен ръб), а обединението би го пробило навън.
            oa, ob = largest(Polygon(outer_a).buffer(0)), largest(Polygon(outer_b).buffer(0))
            wall = max(oa.exterior.distance(ph), ob.exterior.distance(best))
            inner = oa.intersection(ob).buffer(-wall)
            clipped = largest(u.intersection(inner)) if not inner.is_empty else u
            if clipped.area > 0.3 * u.area:
                u = clipped
        out.append(np.asarray(u.exterior.coords)[:-1])
    if extra:
        out += [np.asarray(g) for k, g in enumerate(b) if k not in used]
    return out


HOLE_DARK = 0.45  # петно, по-тъмно от това (дял от яркостта на детайла), сиво и с форма на отвор: отвор
HOLE_SAT = 0.25  # и не цветно (печатът е цветен)
HOLE_TONE = 0.9  # или цветно, но в оттенъка на детайла (отвор в жълта пластмаса е тъмножълт): глух отвор
FACE = {"top": "top", "bottom": "bottom", "front": "front", "back": "back", "right": "right", "left": "left"}
SIDE = {"top": "+", "bottom": "-", "front": "-", "back": "+", "right": "+", "left": "-"}  # design.FACES


def dark_holes(p: dict, holes: list, U: float, V: float, toned: bool = False) -> list:
    """Отвор, видян само като много тъмно, сиво петно на основната снимка (през него не се вижда масата, а и на
    другата снимка не личи): кръгъл или продълговат със заоблени краища, и не до самия ръб. Печат (знаме, лого)
    е цветен или с остри ъгли, затова не минава. toned: вместо сивите - цветните петна в оттенъка на детайла
    (дупката за молива в жълтата капачка на точилка)."""
    from .fit import plain_hole

    out = []
    have = [largest(Polygon(h).buffer(0)) for h in holes]
    edge = largest(Polygon(p["outer"]).buffer(0)).exterior
    for m, info in zip(p.get("marks", []), p.get("mark_info", [])):
        if not info or info.get("dark", 1) > HOLE_DARK:
            continue
        gray = info.get("sat", 1) <= HOLE_SAT
        if gray == toned or (toned and info.get("tone", 0) < HOLE_TONE):
            continue
        mp = largest(Polygon(m).buffer(0))
        if mp.area <= 0 or any(same_hole(mp, h, 0.3) for h in have):
            continue
        if edge.distance(mp) < MARK_WALL * max(U, V):
            continue
        if plain_hole(np.asarray(m), max(U, V))["t"] not in ("hole", "slot"):
            continue
        out.append(np.asarray(mp.exterior.coords)[:-1])
        have.append(mp)
    return out


END_COLOR = 0.7  # цветът на снимката на края е поне толкова по-близо до другия край отгоре: снимката е на него


def end_face(shots: dict, lay: dict, face: str) -> str:
    """Кой край на детайла (left: x = 0, right: x = L) е на снимката, наречена `face`: по цвета ѝ спрямо краищата
    на снимката отгоре (жълтата капачка на точилка). Човек може да изправи детайла на другия край."""
    other = {"left": "right", "right": "left"}[face]
    if face not in shots or "top" not in shots:
        return face
    top = Frame(shots["top"])
    prof = look_profile(top, np.asarray(lay["orient"]["top"]["eu"], float))
    own = Frame(shots[face]).looks.get("l") or {}
    lab = np.array([c for c in own.get("lab", []) if c is not None], float)
    if prof is None or not len(lab):
        return face
    color = np.median(lab, axis=0)
    bins = prof[0]
    k = max(2, len(bins) // 7)
    ends = {"left": np.nanmedian(bins[:k], axis=0), "right": np.nanmedian(bins[-k:], axis=0)}
    near, far = (float(np.linalg.norm(color - ends[n])) for n in (face, other))
    return other if far < END_COLOR * near else face


def blind_holes(lay: dict, names: list, primary: str, place, fits, p: dict, holes: list, U: float, V: float) -> list:
    """Глухите отвори на изгледа: [{"pts", "face"}]. Отвор в оттенъка на детайла (dark_holes с toned) на основната
    снимка и тъмен отвор само на другата снимка на изгледа (другия край на точилка - дупката за молива). Ако на
    другата снимка на същото място също има петно или отвор, отворът е проходен (с другите)."""
    rest = [n for n in names if n != primary]
    q = place(rest[0]) if rest else None
    if q is not None and not fits(q):
        q = None
    alts = lay.get("alts", {}).get(rest[0], [[1, 1]]) if rest else [[1, 1]]
    size = max(U, V)

    def moved(h, su, sv):
        h = np.asarray(h)
        return np.c_[h[:, 0] if su > 0 else U - h[:, 0], h[:, 1] if sv > 0 else V - h[:, 1]]

    def seen(h, on) -> bool:  # на същото място на другата снимка има отвор или тъмно петно
        spots = [largest(Polygon(x).buffer(0)) for x in list(on["holes"]) + list(on.get("marks", []))]
        for su, sv in alts:
            hp = largest(Polygon(moved(h, su, sv)).buffer(0))
            if any(same_hole(hp, x, 0.3) for x in spots if x.area > 0):
                return True
        return False

    out = []
    have = list(holes)
    for h in dark_holes(p, have, U, V, toned=True):
        if q is None or not seen(h, q):
            out.append({"pts": h, "face": FACE[primary]})
        else:
            holes.append(h)
        have.append(h)
    if q is None:
        return out
    for h in dark_holes(q, have, U, V) + dark_holes(q, have, U, V, toned=True):
        spots = [largest(Polygon(moved(h, su, sv)).buffer(0)).centroid for su, sv in alts]
        if max(a.distance(b) for a in spots for b in spots) > 0.05 * size:
            continue  # не е ясно от коя страна е (симетричен предмет), а отворът не е по средата
        if not seen(h, p):
            out.append({"pts": h, "face": FACE[rest[0]]})
    return out


def widen(holes: list, marks: list, U: float, V: float) -> list:
    """Дълбок отвор встрани от средата на кадъра: през него се вижда само част (масата), а до нея е тъмната му
    стена. Ръбът на отвора обхваща и двете, затова отворът е обвивката им (оценката за диаметъра беше 3,5 вместо
    5 мм). Само петно, долепено до отвора, когато обвивката не е много по-голяма от двете заедно."""
    out = []
    marks = [largest(Polygon(m).buffer(0)) for m in marks]
    for h in holes:
        hp = largest(Polygon(h).buffer(0))
        if hp.area > 0:
            for m in marks:
                if (m.area > 0 and same_hole(hp, m, 0.3) and hp.distance(m) <= 0.01 * max(U, V)
                        and m.union(hp).convex_hull.area <= 1.5 * (hp.area + m.area)):  # fmt: skip
                    hp = m.union(hp).convex_hull
            h = np.asarray(hp.exterior.coords)[:-1]
        out.append(np.asarray(h))
    return out


def same_hole(a: Polygon, b: Polygon, share: float) -> bool:
    """Едно и също място: застъпват се поне с `share` от по-малкото или средите им са по-близо от радиуса на
    по-голямото (дълбок отвор: виждащата се през него част и тъмната стена до нея са изместени, а от двете страни
    перспективата ги измества различно)."""
    if a.area <= 0 or b.area <= 0:
        return False
    close = a.centroid.distance(b.centroid) <= math.sqrt(max(a.area, b.area) / math.pi)
    return close or a.intersection(b).area >= share * min(a.area, b.area)


def mark_pairs(p: dict, q: dict, holes: list, alts: list, U: float, V: float) -> list:
    """Тъмно петно на едно и също място на двете противоположни снимки: отвор през детайла, през който от нито една
    страна не се вижда масата (тесен и дълбок, встрани от средата на кадъра). Ако не е ясно как е обърната втората
    снимка (alts), петното трябва да съвпадне само при едно от възможните обръщания."""
    out = []
    have = [largest(Polygon(h).buffer(0)) for h in holes]
    edge = largest(Polygon(p["outer"]).buffer(0)).exterior
    for m in p.get("marks", []):
        mp = largest(Polygon(m).buffer(0))
        if mp.area <= 0 or any(same_hole(mp, h, 0.3) for h in have):
            continue
        if edge.distance(mp) < MARK_WALL * max(U, V):
            continue  # до самия ръб: сянка на стена (изрез, стъпало), видяна от перспективата, а не отвор
        hits = []
        for su, sv in alts:
            for n in q.get("marks", []):
                n = np.asarray(n)
                c = np.c_[n[:, 0] if su > 0 else U - n[:, 0], n[:, 1] if sv > 0 else V - n[:, 1]]
                nq = largest(Polygon(c).buffer(0))
                if nq.area > 0 and mp.intersection(nq).area >= 0.4 * min(mp.area, nq.area):
                    hits.append(nq)
                    break
        if len(hits) == 1:  # формата от основната снимка
            out.append(np.asarray(mp.exterior.coords)[:-1])
            have.append(mp)
    return out


def confirmed(p: dict, q: dict, alts: list, U: float, V: float) -> list:
    """Отворите само на втората снимка, когато не е ясно как е обърната (симетричен предмет): отворът се взима там,
    където на основната снимка има тъмно петно (отвор, през който от тази страна не се вижда), ако такова място е
    точно едно от възможните обръщания."""
    out = []
    marks = [largest(Polygon(m).buffer(0)) for m in p.get("marks", [])]
    matched = [largest(Polygon(h).buffer(0)) for h in p["holes"]]
    for h in q["holes"]:
        h = np.asarray(h)
        hp = largest(Polygon(h).buffer(0))
        if hp.area <= 0 or any(same_hole(hp, m, 0.3) for m in matched):
            continue  # вече е от основната снимка
        hits = []
        for su, sv in alts:
            c = np.c_[h[:, 0] if su > 0 else U - h[:, 0], h[:, 1] if sv > 0 else V - h[:, 1]]
            cp = largest(Polygon(c).buffer(0))
            mk = [m for m in marks if m.area > 0 and cp.intersection(m).area > 0.3 * min(cp.area, m.area)]
            if mk:
                hits.append(mk[0])
        if len(hits) == 1:  # формата от основната снимка (тя е по-рязка и е тази, с която човекът сравнява)
            out.append(np.asarray(hits[0].exterior.coords)[:-1])
    return out


def fuse(a: dict, b: dict, U: float, V: float) -> dict | None:
    """Две очертания на един изглед от противоположни снимки -> едно (обединението им, по знаковите разстояния)."""
    h = max(U, V) / 1400
    pad = 12
    nx, ny = int(math.ceil(U / h)) + 2 * pad, int(math.ceil(V / h)) + 2 * pad

    def sdf(p):
        im = Image.new("L", (nx, ny), 0)
        draw = ImageDraw.Draw(im)
        px = lambda pts: [((u / h) + pad, (v / h) + pad) for u, v in pts]  # noqa: E731
        draw.polygon(px(p["outer"]), fill=1)
        for hole in p["holes"]:
            draw.polygon(px(hole), fill=0)
        m = np.asarray(im, bool)
        return ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)

    # Обединението: перспективата само смалява по-далечното спрямо най-близката страна (след нормирането до
    # измерения размер), така че всяка от двете снимки може да има само вдлъбнатини, които другата няма.
    s = np.minimum(sdf(a), sdf(b))
    cs = measure.find_contours(s, 0.0)
    polys = []
    for c in cs:
        if len(c) < 8:
            continue
        pts = np.c_[(c[:, 1] - pad) * h, (c[:, 0] - pad) * h]  # (ред, стълб) -> (u, v)
        poly = Polygon(pts).buffer(0)
        if poly.area > 0:
            polys.append((pts, poly))
    if not polys:
        return None
    outer, op = max(polys, key=lambda t: t[1].area)
    holes = [pts for pts, poly in polys if poly is not op and op.contains(poly.representative_point())
             and poly.area >= 0.0015 * op.area]  # fmt: skip
    outer = np.c_[np.clip(outer[:, 0], 0, U), np.clip(outer[:, 1], 0, V)]
    return {"outer": outer, "holes": holes}


def overall(lay: dict, flat: bool = False) -> list[dict]:
    """Общите размери за мерене, с коя снимка и по коя ос ("u" или "v" на изгледа) е линията. flat: плосък детайл,
    само със снимки отгоре и отдолу: третият размер е дебелината."""
    o = lay["orient"]
    rnd = lay["round"]
    primary = {o[n]["view"]: n for n in o if o[n]["primary"]}
    out = []
    if rnd["top"] and not rnd["end"]:
        out.append(question("D", "top", "u"))
    else:
        out.append(question("L", "top", "u"))
        if not rnd["end"]:
            out.append(question("W", "top", "v"))
    if rnd["end"]:
        out.append(question("D", primary.get("end"), "u"))
    elif "side" in primary:
        out.append(question("H", primary["side"], "v"))
    elif "end" in primary:
        out.append(question("H", primary["end"], "v"))
    else:
        out.append(question("H", None, "v"))
    if flat and out[-1]["id"] == "H":  # снимките отстрани на плосък детайл са в ръка: линията е върху чертежа
        out[-1] = {"id": "H", "name": "Дебелина", "ask": "Измерете дебелината на детайла.",
                   "hint": "Шублерът напряко на плоската страна, където детайлът е най-дебел.", "photo": None,
                   "along": "v"}  # fmt: skip
    return out


ASK = {
    "L": ("Дължина", "Измерете дължината по оранжевата линия.", "Най-дългата страна, от край до край."),
    "W": ("Ширина", "Измерете ширината по оранжевата линия.", "Напряко на дължината."),
    "H": ("Височина", "Измерете височината по оранжевата линия.", "Колко е висок, както лежи на първата снимка."),
    "D": ("Диаметър", "Измерете диаметъра по оранжевата линия.", "Най-дебелото място на кръглото сечение."),
}


def question(dim: str, photo: str | None, along: str) -> dict:
    name, ask, hint = ASK[dim]
    if photo is None:
        hint += " Няма снимка отстрани: мерете от масата до най-високата точка."
    return {"id": dim, "name": name, "ask": ask, "hint": hint, "photo": photo, "along": along}


def resolve(lay: dict, vals: dict) -> tuple[float, float, float]:
    """(L, W, H) от мерките на общите размери (D замества двата размера на кръглото сечение)."""
    rnd = lay["round"]
    D = vals.get("D")
    L, W, H = vals.get("L"), vals.get("W"), vals.get("H")
    if rnd["end"] and D:
        W = H = D
        if rnd["top"]:
            L = L or D
    elif rnd["top"] and D:
        L = W = D
    if not (L and W and H):
        raise ValueError("missing")
    return float(L), float(W), float(H)
