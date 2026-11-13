"""Очертание (точки в мм) -> чисти прави и заоблени ъгли, нива и отвори, както би ги начертал конструктор.

1. Опростяване (Douglas-Peucker) с допуск TOL от размера: остават дългите прави и къси парчета по заоблените ъгли.
2. Всяка дълга права става линия; почти хоризонталните и вертикалните (до SNAP_DEG) стават точно такива.
3. Ъгъл = пресечната точка на две съседни прави. Ако очертанието минава далеч от нея, ъгълът е заоблен: радиусът е
   от разстоянието до върха и ъгъла между правите (r = e / (1/sin(a/2) - 1)). Полукръгъл край (две успоредни прави)
   става два ъгъла с радиус половината разстояние между тях.
4. Цяло очертание, близко до кръг, е кръг."""

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

CORNER_DEV = 0.06  # прави и дъги, отклонени от очертанието повече от толкова от размера: свободна форма
TOL = 0.007  # допуск на опростяването, дял от по-големия размер на изгледа
LONG = 3.5  # права е по-дълга от толкова допуска
SNAP_DEG = 5.0
SHARP = 0.035  # ъгъл с радиус под този дял от размера е остър (на снимка не се различава от перспектива)
CIRCLE_RMS = 0.025
WINDOW = 0.006  # посоката се мери по прозорец ±толкова от размера
MIN_EDGE = 0.025  # най-късата права, дял от размера
LONG_EDGE = 0.1  # права, по-къса от това (дял от размера), може да е парче от дъга
MERGE_DEG = 8.0  # съседни прави с по-малка чупка (градуси) са една права
SHARP_TURN = 45.0  # завой (градуси), от който нататък и късата права е ръб, а не парче от дъга
SOFT_TURN = 35.0  # ъгъл с по-малък завой е чупка на начупена линия по крива
SOFT_MAX = 3  # толкова такива чупки: очертанието е крива (свободна форма), а не прави и дъги


def circle_fit(pts: np.ndarray) -> tuple[float, float, float, float]:
    """(cx, cy, r, относителна грешка) по метода на най-малките квадрати."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x**2 + y**2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(max(c + cx**2 + cy**2, 1e-12))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms / r


def ccw(pts: np.ndarray) -> np.ndarray:
    return pts if Polygon(pts).exterior.is_ccw else pts[::-1]


def rdp(pts: np.ndarray, tol: float) -> list[int]:
    """Индексите на опростената начупена линия (Douglas-Peucker) на отворена линия."""
    keep = [0, len(pts) - 1]
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        n = math.hypot(*d)
        rel = pts[i + 1 : j] - a
        dist = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / n if n > 1e-12 else np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep.append(m)
            stack += [(i, m), (m, j)]
    return sorted(set(keep))


def resample(closed: np.ndarray, step: float) -> np.ndarray:
    seg = np.hypot(*np.diff(closed, axis=0).T)
    t = np.r_[0, np.cumsum(seg)]
    n = max(16, int(t[-1] / step))
    ts = np.linspace(0, t[-1], n, endpoint=False)
    return np.c_[np.interp(ts, t, closed[:, 0]), np.interp(ts, t, closed[:, 1])]


AXIS_EDGES = 0.25  # прави в една посока (и напряко) поне толкова от обиколката: осите са по тях


def main_axes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Осите на очертанието q (x надясно, y нагоре): l по дължината, s напряко, и ъгълът на l в градуси (-90..90].
    По правите ръбове, ако има достатъчно (стъпала, краката на кука, страните на кутия), иначе по най-малкия описан
    правоъгълник: при кука или дъга той е завъртян, а детайлът е подравнен с правите си, както и мерките."""
    poly = Polygon(q).buffer(0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    ring = np.asarray(poly.exterior.coords)
    step = poly.length / 720
    p = resample(ring, step)
    k = 6
    n = len(p)
    ext = np.r_[p, p, p]
    fwd = ext[n + k : 2 * n + k] - ext[n - k : 2 * n - k]
    straight = np.hypot(fwd[:, 0], fwd[:, 1]) > 0.985 * 2 * k * step
    ang = np.degrees(np.arctan2(fwd[:, 1], fwd[:, 0])) % 90
    hist = np.bincount(np.round(ang[straight]).astype(int) % 90, minlength=90).astype(float)
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1) + np.roll(hist, 2) + np.roll(hist, -2)
    peak = int(np.argmax(smooth))
    phi = None
    if smooth[peak] >= AXIS_EDGES * n:
        near = straight & (np.abs((ang - peak + 45) % 90 - 45) <= 3)
        phi = peak + float(np.mean((ang[near] - peak + 45) % 90 - 45))
    else:
        rect = poly.minimum_rotated_rectangle
        rx, ry = rect.exterior.coords.xy
        edges = [(rx[i + 1] - rx[i], ry[i + 1] - ry[i]) for i in range(2)]
        e = max(edges, key=lambda v: math.hypot(*v))
        phi = math.degrees(math.atan2(e[1], e[0]))
    a = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    b = np.array([-a[1], a[0]])
    if np.ptp(ring @ b) > np.ptp(ring @ a):
        phi += 90
    phi = (phi + 90) % 180 - 90
    l = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    return l, np.array([-l[1], l[0]]), phi


def straight_runs(p: np.ndarray, k: int, max_dev_deg: float, min_len: int) -> list[tuple[int, int]]:
    """Отрязъци [i, j) от затворената линия p (равни стъпки), в които посоката (по прозорец ±k) е постоянна:
    правите. Индексите могат да минават края (по модул)."""
    n = len(p)
    ext = np.r_[p, p, p]
    fwd = ext[n + k : 2 * n + k] - ext[n - k : 2 * n - k]
    th = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    lim = math.radians(max_dev_deg)
    # Начало там, където посоката се мени най-много (сигурно не е в средата на права).
    turn = np.abs(np.r_[np.diff(th), th[0] - th[-1] + 2 * math.pi * round((th[-1] - th[0]) / (2 * math.pi))])
    start = int(np.argmax(turn)) + 1
    th = np.unwrap(np.r_[th[start:], th[:start]])
    runs = []
    i = 0
    while i < n:
        j = i + 1
        ref = th[i]
        while j < n and abs(th[j] - ref) < lim:
            j += 1
            ref = float(np.median(th[i:j])) if (j - i) % 8 == 0 else ref
        if j - i >= min_len:
            runs.append(((i + start) % n, j - i))
        i = j
    return runs


def corners(outer: np.ndarray, U: float, V: float, raw: bool = False) -> list[tuple[float, float, float]] | None:
    """Ъглите (u, v, r) на очертанието обратно на часовника, или None, ако не става с прави и дъги (свободна
    форма). Правите са дълги отрязъци с постоянна посока; между две прави е ъгъл: остър, ако очертанието минава през
    пресечната им точка, иначе заоблен с радиуса от разстоянието до нея."""
    size = max(U, V)
    tol = TOL * size
    pts = ccw(np.asarray(outer, float))
    h = size / 1200
    p = resample(np.r_[pts, pts[:1]], h)
    n = len(p)
    k = max(2, round(WINDOW * size / h))
    runs = straight_runs(p, k, 4.0, max(2 * k, round(MIN_EDGE * size / h)))
    if len(runs) < 2:
        return None
    lines = []
    for i, length in runs:
        idx = (i + np.arange(k, length - k + 1 if length > 2 * k else length)) % n
        seg = p[idx]
        c, d = fit_line(seg, seg[0], seg[-1], tol)
        lines.append((c, d, int((i + k) % n), int((i + length - k) % n)))
    # Съседни почти еднакви прави (прекъснати от шум или с лека чупка от перспективата) се сливат и се изправят
    # наново по всичките си точки.
    def same(l1, l2) -> bool:
        return (abs(cross2(l1[1], l2[1])) < math.sin(math.radians(MERGE_DEG)) and l1[1] @ l2[1] > 0
                and point_line_dist(l2[0], l1[0], l1[1]) < 1.5 * tol)  # fmt: skip

    def joined(l1, l2):
        s0, e1 = l1[2], l2[3]
        seg = p[(s0 + np.arange(((e1 - s0) % n) + 1)) % n]
        c, d = fit_line(seg, seg[0], seg[-1], 1.5 * tol)
        return (c, d, s0, e1)

    merged = []
    for ln in lines:
        if merged and same(merged[-1], ln):
            merged[-1] = joined(merged[-1], ln)
            continue
        merged.append(ln)
    if len(merged) > 2 and same(merged[-1], merged[0]):
        merged[0] = joined(merged[-1], merged[0])
        merged.pop()
    lines = smooth_out(merged, p, size)
    m = len(lines)
    if m < 2:
        return None
    out = []
    for q in range(m):
        p1, d1, _, end1 = lines[q]
        p2, d2, start2, _ = lines[(q + 1) % m]
        between = p[(end1 + np.arange(((start2 - end1) % n) + 1)) % n]
        turn = math.degrees(math.atan2(cross2(d1, d2), d1 @ d2))
        if abs(turn) > 155:  # полукръгъл край: два ъгъла по 90° с радиус половината разстояние
            gap = (p2 - p1) - ((p2 - p1) @ d1) * d1
            half = float(np.linalg.norm(gap)) / 2
            reach = float(((between - p1) @ d1).max()) if len(between) else 0.0
            c1 = p1 + d1 * reach
            out += [(float(c1[0]), float(c1[1]), half), (float((c1 + gap)[0]), float((c1 + gap)[1]), half)]
            continue
        if abs(turn) < 3:  # успоредни и разместени: късо стъпало
            q0 = between[len(between) // 2]
            c1 = p1 + ((q0 - p1) @ d1) * d1
            c2 = p2 + ((c1 - p2) @ d2) * d2
            out += [(float(c1[0]), float(c1[1]), 0.0), (float(c2[0]), float(c2[1]), 0.0)]
            continue
        c = intersect(p1, d1, p2, d2)
        if c is None:
            return None
        e = LineString(between).distance(Point(c)) if len(between) >= 2 else 0.0
        ang = math.radians(180 - abs(turn))
        # Заоблен ъгъл само при ясен завой (поне 30°) и радиус, който се побира в съседните прави: иначе е шум.
        r = e / (1 / math.sin(ang / 2) - 1) if e > 0.6 * tol and abs(turn) >= 30 else 0.0
        if r > 0.5 * size:
            r = 0.0
        out.append((float(c[0]), float(c[1]), r if raw or r >= SHARP * size else 0.0))
    if len(out) < 3 or soft_turns(out) >= SOFT_MAX:
        return None
    if not Polygon([(u, v) for u, v, _ in out]).is_valid:
        return None
    out = fitting_arcs(out)
    poly = Polygon(rounded(out)).buffer(0)
    raw = Polygon(pts).buffer(0)
    iou = poly.intersection(raw).area / max(poly.union(raw).area, 1e-9)
    if iou <= 0.95 or poly.is_empty or raw.is_empty:
        return None
    # И никъде далеч от снимката: вълниста форма с прави през вълните съвпада 95%, но се отклонява с 11% от размера
    # (истинските детайли - до 4,3%).
    return out if poly.boundary.hausdorff_distance(raw.boundary) <= CORNER_DEV * size else None


def soft_turns(cs: list[tuple[float, float, float]]) -> int:
    """Колко ъгъла завиват с по-малко от SOFT_TURN: заобленото тяло на точилка (кръг с две плоски страни)
    ставаше многоъгълник с 12 ъгъла, повечето по кривата, и въпроси за "стъпала", които не могат да се измерят."""
    n = len(cs)
    count = 0
    for k in range(n):
        a = np.subtract(cs[k][:2], cs[k - 1][:2])
        b = np.subtract(cs[(k + 1) % n][:2], cs[k][:2])
        turn = abs(math.degrees(math.atan2(cross2(a, b), float(a @ b))))
        count += turn < SOFT_TURN
    return count


def fitting_arcs(cs: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Радиусите, свити колкото двете дъги в краищата на всяка страна да се съберат на нея. Вътрешната извивка на
    кука с къса права по средата (два ъгъла от по 80° вместо полукръг) получаваше по два радиуса от 18 мм на страна
    от 9 мм: заобленото очертание се пресичаше, приликата със снимката ставаше 0 и куката - гладка форма без мерки."""
    n = len(cs)
    tan = []
    for k in range(n):
        a = np.array(cs[k - 1][:2]) - cs[k][:2]
        b = np.array(cs[(k + 1) % n][:2]) - cs[k][:2]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        ang = math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb)))) if na > 0 and nb > 0 else 0.0
        tan.append(math.tan(ang / 2) if 1e-3 < ang < math.pi - 1e-3 else 0.0)
    r = [c[2] if tan[k] > 0 else 0.0 for k, c in enumerate(cs)]
    for _ in range(4):
        for k in range(n):
            k1 = (k + 1) % n
            side = math.dist(cs[k][:2], cs[k1][:2])
            need = (r[k] / tan[k] if r[k] else 0.0) + (r[k1] / tan[k1] if r[k1] else 0.0)
            if need > side:
                s = side / need * (1 - 1e-6)
                r[k], r[k1] = r[k] * s, r[k1] * s
    return [(u, v, rr) for (u, v, _), rr in zip(cs, r)]


def rounded(cs: list[tuple[float, float, float]], steps: int = 12) -> np.ndarray:
    """Ъглите (u, v, r) -> начупена линия със заоблените ъгли като дъги."""
    n = len(cs)
    out = []
    for k in range(n):
        u, v, r = cs[k]
        a = np.array([cs[k - 1][0] - u, cs[k - 1][1] - v])
        b = np.array([cs[(k + 1) % n][0] - u, cs[(k + 1) % n][1] - v])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if r <= 0 or na < 1e-9 or nb < 1e-9:
            out.append((u, v))
            continue
        a, b = a / na, b / nb
        ang = math.acos(max(-1.0, min(1.0, float(a @ b))))
        if ang > math.pi - 1e-3 or ang < 1e-3:
            out.append((u, v))
            continue
        t = min(r / math.tan(ang / 2), na, nb)
        r = t * math.tan(ang / 2)
        c = np.array([u, v]) + (a + b) / np.linalg.norm(a + b) * r / math.sin(ang / 2)
        p1, p2 = np.array([u, v]) + a * t, np.array([u, v]) + b * t
        a1, a2 = math.atan2(*(p1 - c)[::-1]), math.atan2(*(p2 - c)[::-1])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for s_ in np.linspace(0, 1, steps):
            out.append(tuple(c + r * np.array([math.cos(a1 + da * s_), math.sin(a1 + da * s_)])))
    return np.asarray(out)


def smooth_out(lines: list, p: np.ndarray, size: float) -> list:
    """Маха правите, които са парчета от дъга: къси (под LONG_EDGE) и без ясен завой (поне SHARP_TURN) към двете
    съседни. Дъга се разпада на къси прави с малки завои между тях, а парчето до заоблен ъгъл е с малък завой поне от
    едната страна; истинският ръб (стъпало, фаска) е или дълъг, или с ясен завой и от двете страни. Между останалите прави ъгълът става заоблен с
    радиуса от дъгата (или полукръг при обръщане на 180°)."""
    n = len(p)
    m = len(lines)
    if m <= 2:
        return lines

    def length(ln) -> float:
        return ((ln[3] - ln[2]) % n) * size / 1200

    def turn(a, b) -> float:
        return abs(math.degrees(math.atan2(cross2(a[1], b[1]), a[1] @ b[1])))

    keep = []
    for k, ln in enumerate(lines):
        prev, nxt = lines[k - 1], lines[(k + 1) % m]
        sharp = min(turn(prev, ln), turn(ln, nxt)) >= SHARP_TURN
        if sharp or length(ln) >= LONG_EDGE * size:
            keep.append(ln)
    return keep if len(keep) >= 2 else lines


def symmetric_round(cs: list, U: float, V: float) -> list[tuple[float, float, float]]:
    """Ъглите със суров радиус (corners(raw=True)) -> заоблени или остри. Ъгълът е заоблен, ако радиусът му е поне
    SHARP от размера или ако огледалният му ъгъл (спрямо средата на изгледа) е заоблен и неговият е поне 40% от
    този: заоблените ъгли на детайл обикновено са по двойки, а на снимка единият може да излезе по-малък (отблясък,
    сянка, прозрачна пластмаса)."""
    size = max(U, V)
    thr = SHARP * size
    tol = 0.05 * size
    out = []
    for k, (u, v, r) in enumerate(cs):
        if r >= thr:
            out.append((u, v, r))
            continue
        mates = [r2 for j, (u2, v2, r2) in enumerate(cs) if j != k and r2 >= thr and
                 ((abs(u - u2) < tol and abs(v + v2 - V) < tol) or (abs(v - v2) < tol and abs(u + u2 - U) < tol))]  # fmt: skip
        mate = max(mates, default=0.0)
        out.append((u, v, (r + mate) / 2 if mate and r >= 0.4 * mate else 0.0))
    return out


TAB_INSET = 0.02  # край, по-тесен от тялото поне с толкова (дял от ширината, и поне 0,4 мм) от двете страни
TAB_MAX = 0.3  # и не по-дълъг от толкова от дължината (щепселът USB-A е 12 мм на флашка от 52 мм)


def end_tabs(outer: np.ndarray, U: float, V: float) -> list[dict]:
    """По-тесен край (например краят на флашка срещу капачката): в края по дължината ширината е по-малка от тази на
    тялото от двете страни. Връща [{"end": 0 или 1, "len", "lo", "hi"}] в мм: колко е дълъг и с колко е навътре
    отпред (lo) и отзад (hi). Тези стъпала са къси (под MIN_EDGE) и с меки ръбове на снимката, затова се търсят по
    ширината, а не като ъгли."""
    poly = Polygon(outer).buffer(0)
    if poly.geom_type != "Polygon":
        return []
    step = U / 400
    xs = np.arange(step / 2, U, step)
    lo, hi = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        c = poly.intersection(LineString([(x, -1), (x, V + 1)]))
        if not c.is_empty:
            lo[i], hi[i] = c.bounds[1], c.bounds[3]
    body = (xs > 0.25 * U) & (xs < 0.75 * U)
    lo_b, hi_b = np.nanmedian(lo[body]), np.nanmedian(hi[body])
    d_min = max(0.4, TAB_INSET * V)
    out = []
    for end in (0, 1):
        idx = np.arange(len(xs)) if end == 0 else np.arange(len(xs))[::-1]
        dist = xs if end == 0 else U - xs
        inside = (lo[idx] - lo_b > d_min / 2) & (hi_b - hi[idx] > d_min / 2)
        n = 0
        while n < len(idx) and dist[idx[n]] < TAB_MAX * U and (inside[n] or dist[idx[n]] < 0.015 * U):
            n += 1
        length = float(dist[idx[n]]) if n < len(idx) else 0.0
        if length < 0.04 * U or length >= TAB_MAX * U:
            continue
        core = idx[(dist[idx] > 0.25 * length) & (dist[idx] < 0.75 * length)]
        if len(core) < 3:
            continue
        ins_lo, ins_hi = lo[core] - lo_b, hi_b - hi[core]
        d_lo, d_hi = float(np.nanmedian(ins_lo)), float(np.nanmedian(ins_hi))
        # Истинският по-тесен край е еднакво тесен по цялата си дължина (после стъпало); заобленият ъгъл става все
        # по-широк към тялото: там навътре не е постоянно.
        flat = all(float(np.nanpercentile(x, 85) - np.nanpercentile(x, 15)) <= 0.45 * max(float(np.nanmedian(x)), 1e-9)
                   for x in (ins_lo, ins_hi))  # fmt: skip
        if d_lo >= d_min and d_hi >= d_min and flat:
            # Дължината е до стъпалото (там навътре е наполовина), а не докъдето тялото е пак с пълна ширина:
            # заобленият ъгъл на тялото до щепсела иначе я удължава (14 вместо 12,5 мм).
            half = (lo[idx] - lo_b > d_lo / 2) & (hi_b - hi[idx] > d_hi / 2)
            m = 0
            while m < n and (half[m] or dist[idx[m]] < 0.015 * U):
                m += 1
            if m < n and float(dist[idx[m]]) >= 0.04 * U:
                length = float(dist[idx[m]])
            # Всяка страна до своето стъпало: краят на тялото може да е скосен (флашка отпред: горната половина
            # свършва 7 мм преди върха, долната - 3 мм) и по-тесният край е по-дълъг от едната страна.
            sides = []
            for ins, d_s in ((lo[idx] - lo_b, d_lo), (hi_b - hi[idx], d_hi)):
                m = 0
                while m < len(idx) and dist[idx[m]] < TAB_MAX * U and (ins[m] > d_s / 2 or dist[idx[m]] < 0.015 * U):
                    m += 1
                own = float(dist[idx[m]]) if m < len(idx) else length
                sides.append(own if length <= own < TAB_MAX * U else length)
            out.append({"end": end, "len": length, "lo": d_lo, "hi": d_hi, "len_lo": sides[0], "len_hi": sides[1]})
    return out


STEP_ROUND = 0.9  # ъгълът на тялото до по-тесния край е заоблен най-много с толкова от стъпалото навътре
SLANT = 0.03  # стъпалата от двете страни на по-тесния край, разместени с повече от толкова от дължината, са различни


def tab_lengths(t: dict, U: float) -> tuple[float, float]:
    """Дължината на по-тесния край откъм lo и откъм hi: еднаква, освен ако стъпалата са ясно разместени."""
    a, b = t.get("len_lo", t["len"]), t.get("len_hi", t["len"])
    if abs(a - b) <= max(SLANT * U, 0.25 * min(a, b)):
        return t["len"], t["len"]
    return a, b


def apply_tabs(cs: list, tabs: list, U: float, V: float) -> list:
    """Ъглите на очертанието (обратно на часовника) с по-тесните краища: ъглите в края (и намерените от тях
    стъпала) се заменят със стъпало навътре от двете страни; радиусът им остава за ъглите на по-тесния край."""
    for t in tabs:
        len_lo, len_hi = tab_lengths(t, U)
        x_lo, x_hi = (len_lo, len_hi) if t["end"] == 0 else (U - len_lo, U - len_hi)
        reach = 1.2 * max(len_lo, len_hi)
        near = [(c[0] < reach) if t["end"] == 0 else (c[0] > U - reach) for c in cs]
        n = len(cs)
        if sum(near) < 2 or all(near):
            continue
        start = next(k for k in range(n) if near[k] and not near[k - 1])  # началото на поредицата в края
        block = []
        k = start
        while near[k % n] and len(block) < n:
            block.append(k % n)
            k += 1
        if len(block) != sum(near):
            continue  # ъглите в края не са една поредица
        r = max(cs[k][2] for k in block)
        if r > 0.55 * (V - t["lo"] - t["hi"]):
            r = 0.0  # дъга, по-широка от самия край: не е негова, а прескочено стъпало (ъгълът на тялото и щепсела)
        r = min(r, 0.45 * min(len_lo, len_hi), 0.45 * (V - t["lo"] - t["hi"]))
        lo, hi = t["lo"], V - t["hi"]
        # Ъгълът на тялото до стъпалото е заоблен като ъглите на края (плавният преход към по-тесния край), колкото
        # се събира в стъпалото.
        r_lo, r_hi = min(r, STEP_ROUND * t["lo"]), min(r, STEP_ROUND * t["hi"])
        if t["end"] == 0:  # отляво обратно на часовника: отгоре надолу
            new = [(x_hi, V, r_hi), (x_hi, hi, 0.0), (0.0, hi, r), (0.0, lo, r), (x_lo, lo, 0.0), (x_lo, 0.0, r_lo)]
        else:  # отдясно: отдолу нагоре
            new = [(x_lo, 0.0, r_lo), (x_lo, lo, 0.0), (U, lo, r), (U, hi, r), (x_hi, hi, 0.0), (x_hi, V, r_hi)]
        rest = [cs[(start + len(block) + j) % n] for j in range(n - len(block))]
        cs = new + rest
    return cs


def cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def fit_line(pts: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float):
    """Права през точките на очертанието между a и b (средната им част), изправена до хоризонтал/вертикал."""
    d = (b - a) / max(np.linalg.norm(b - a), 1e-12)
    rel = pts - a
    t = rel @ d
    off = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
    L = np.linalg.norm(b - a)
    near = pts[(t > 0.15 * L) & (t < 0.85 * L) & (off < 1.5 * tol)]
    if len(near) >= 4:
        c = near.mean(0)
        _, _, vt = np.linalg.svd(near - c)
        d2 = vt[0] if vt[0] @ d > 0 else -vt[0]
    else:
        c, d2 = (a + b) / 2, d
    ang = math.degrees(math.atan2(d2[1], d2[0]))
    for target in (0, 90, 180, -90, -180):
        if abs(ang - target) < SNAP_DEG:
            r = math.radians(target)
            d2 = np.array([round(math.cos(r)), round(math.sin(r))], float)
            break
    return np.asarray(c, float), np.asarray(d2, float)


def point_line_dist(p, c, d) -> float:
    r = np.asarray(p) - c
    return abs(r[0] * d[1] - r[1] * d[0])


def intersect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, _ = np.linalg.solve(A, p2 - p1)
    return p1 + t * d1


# --- Отвори --------------------------------------------------------------------------------------------------------


def robust_circle(pts: np.ndarray, seed: int = 0, depth: float | None = None) -> tuple[float, float, float] | None:
    """Кръгъл отвор, част от чийто ръб е скрита: кръгът, на който лежи поне 30% от очертанието по поне 150° от
    обиколката, и в който се побира целият видим отвор. От няколко такива - най-големият: при снимка с перспектива
    през проходен отвор се вижда сечението на близкия ръб (истинският размер) и по-малкия далечен; сянка в отвора
    също само смалява видимото. None: не е кръг. depth: колко е дълбок отворът (дебелината на детайла): в тънък
    детайл перспективата скрива малко от отвора, затова видимото трябва да запълва кръга повече (D-образният отвор на
    ключ, 5,5 x 10 мм в пластина от 2 мм, не е кръг с диаметър 10)."""
    n = len(pts)
    if n < 12:
        return None
    rng = np.random.default_rng(seed)
    span = float(np.max(np.ptp(pts, axis=0)))
    area = Polygon(pts).buffer(0).area
    cands = []
    for _ in range(500):
        a, b, c = pts[rng.choice(n, 3, replace=False)]
        circ = circle_3(a, b, c)
        if circ is None or not 0.25 * span < circ[2] < 1.5 * span:
            continue
        thr = 0.02 * circ[2] + 0.01
        inl = np.abs(np.hypot(pts[:, 0] - circ[0], pts[:, 1] - circ[1]) - circ[2]) < thr
        if inl.sum() >= 0.3 * n:
            cands.append((int(inl.sum()), inl))
    found = []
    for count, inl in sorted(cands, key=lambda t: -t[0])[:80]:
        cx, cy, r, _ = circle_fit(pts[inl])
        if any(math.hypot(cx - fx, cy - fy) < 0.03 * r and abs(r - fr) < 0.03 * r for fx, fy, fr in found):
            continue
        ang = np.sort(np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
        if 2 * math.pi - gaps.max() < math.radians(150):
            continue
        least = 0.45 if not depth else max(0.45, min(0.85, 1 - 1.5 * depth / (2 * r)))
        if not least * math.pi * r * r <= area <= 1.08 * math.pi * r * r:  # видимият отвор е в кръга и го запълва
            continue
        found.append((cx, cy, r))
    if not found:
        return None
    return max(found, key=lambda c: c[2])


def circle_3(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


SHAPED_IOU = 0.8  # канал или правоъгълник, който съвпада с отвора по-малко от това: отворът е по снимката
SHAPED_MIN = 3.0  # мм: по-тесен отвор е канал или правоъгълник (ъглите му не се виждат ясно на снимката)
SHARP_FILL = 0.88  # ъгъл на отвора, запълнен поне толкова (квадратче с половината от по-малката страна), е остър
ROUND_FILL = 0.75  # и най-много толкова: заоблен (полукръг: 0,785)


def hole_shape(pts: np.ndarray, size: float, depth: float | None = None) -> dict:
    """Отвор: кръг {t: hole, u, v, d}, канал {t: slot, u, v, len, w, angle} или правоъгълник {t: rect, u, v, w, h,
    r}; u, v е средата. Отвор, който не е никое от тях (D-образният отвор на ключ: два остри и два заоблени ъгъла, или
    нещо съвсем друго), е правоъгълник с "shape": формата от снимката като дял от правоъгълника (и "corners": острите
    ѝ ъгли): размерите, които се мерят, остават ширината, височината и положението. Само при отвор от поне 3 мм:
    по-малките на истинските снимки (отворът за халка на флашка, 1,5 x 7 мм) са с назъбени ъгли от пикселите.
    depth: дебелината на детайла при отвора (robust_circle)."""
    pts = np.asarray(pts, float)
    out = plain_hole(pts, size, depth)
    if out["t"] == "hole":
        return out
    poly = Polygon(pts).buffer(0)
    if poly.geom_type != "Polygon" or poly.area <= 0:
        return out
    u0, v0, u1, v1 = poly.bounds
    if min(u1 - u0, v1 - v0) < SHAPED_MIN:
        return out
    if out["t"] == "slot":
        model = slot_poly(out)
    else:
        a0, a1 = out["u"] - out["w"] / 2, out["u"] + out["w"] / 2
        b0, b1 = out["v"] - out["h"] / 2, out["v"] + out["h"] / 2
        r = min(out["r"], out["w"] / 2 - 1e-6, out["h"] / 2 - 1e-6)
        model = Polygon(rounded([(a0, b0, r), (a1, b0, r), (a1, b1, r), (a0, b1, r)]))
    iou = poly.intersection(model).area / max(poly.union(model).area, 1e-12)
    if iou >= SHAPED_IOU and not half_round(poly):
        return out
    return shaped_hole(pts) or out


def half_round(poly: Polygon) -> bool:
    """Два съседни ъгъла остри, другите два заоблени (D): нито канал, нито правоъгълник."""
    from shapely.geometry import box

    u0, v0, u1, v1 = poly.bounds
    c = min(u1 - u0, v1 - v0) / 2
    fills = [poly.intersection(box(a, b, a + c, b + c)).area / (c * c)
             for a, b in ((u0, v0), (u1 - c, v0), (u1 - c, v1 - c), (u0, v1 - c))]  # fmt: skip
    for k in range(4):
        sharp = (fills[k], fills[(k + 1) % 4])
        other = (fills[(k + 2) % 4], fills[(k + 3) % 4])
        if min(sharp) >= SHARP_FILL and max(other) <= ROUND_FILL:
            return True
    return False


def slot_poly(f: dict) -> Polygon:
    lu, lv = (f["len"], f["w"]) if f["angle"] == 0 else (f["w"], f["len"])
    r = min(lu, lv) / 2 - 1e-6
    u0, v0 = f["u"] - lu / 2, f["v"] - lv / 2
    return Polygon(rounded([(u0, v0, r), (u0 + lu, v0, r), (u0 + lu, v0 + lv, r), (u0, v0 + lv, r)]))


def shaped_hole(pts: np.ndarray) -> dict | None:
    """Отворът по снимката, вписан в правоъгълника около него (по осите)."""
    poly = Polygon(pts).buffer(0)
    if poly.geom_type != "Polygon" or poly.area <= 0:
        return None
    u0, v0, u1, v1 = poly.bounds
    w, h = u1 - u0, v1 - v0
    if min(w, h) <= 0:
        return None
    local, corners = smooth_outline(np.asarray(poly.exterior.coords) - [u0, v0], w, h)
    if len(local) < 6:
        return None
    shape = [[round(a / w, 5), round(b / h, 5)] for a, b in local]
    return {"t": "rect", "u": (u0 + u1) / 2, "v": (v0 + v1) / 2, "w": w, "h": h, "r": 0.0, "shape": shape,
            "corners": corners}


def plain_hole(pts: np.ndarray, size: float, depth: float | None = None) -> dict:
    circle = robust_circle(pts, depth=depth)
    if circle is not None:
        cx, cy, r = circle
        return {"t": "hole", "u": cx, "v": cy, "d": 2 * r}
    cx, cy, r, rel = circle_fit(pts)
    poly = Polygon(pts)
    rect = poly.minimum_rotated_rectangle
    rx, ry = rect.exterior.coords.xy
    e1 = math.hypot(rx[1] - rx[0], ry[1] - ry[0])
    e2 = math.hypot(rx[2] - rx[1], ry[2] - ry[1])
    long, short = max(e1, e2), min(e1, e2)
    if rel < 0.04 and long / short < 1.15:
        return {"t": "hole", "u": cx, "v": cy, "d": 2 * r}
    ang = math.degrees(math.atan2(ry[1] - ry[0], rx[1] - rx[0])) if e1 >= e2 else \
        math.degrees(math.atan2(ry[2] - ry[1], rx[2] - rx[1]))  # fmt: skip
    ang = (ang + 90) % 180 - 90
    c = rect.centroid
    fill = poly.area / max(rect.area, 1e-9)
    slot_fill = (math.pi * short**2 / 4 + (long - short) * short) / (long * short)
    # Изправен по осите (до 8°): ъгъл 0 или 90, иначе отворът се оставя по снимката като правоъгълник.
    axis_aligned = abs(ang) < 8 or abs(abs(ang) - 90) < 8
    along_u = abs(ang) < 45
    if long / short >= 1.3 and abs(fill - slot_fill) < abs(fill - 1.0) and axis_aligned:
        return {"t": "slot", "u": c.x, "v": c.y, "len": long, "w": short, "angle": 0 if along_u else 90}
    # Правоъгълник: ъглите като при очертанието (сянката в отвора отрязва площ, затова радиусът не е от площта).
    w, h = (long, short) if along_u else (short, long)
    cs = corners(pts, w, h) if axis_aligned else None
    rr = 0.0
    if cs is not None and len(cs) == 4:
        radii = sorted(c[2] for c in cs)
        rr = radii[1] if radii[1] > 0 else 0.0  # поне два заоблени ъгъла (сянката заобля най-много един-два)
        us, vs = [c[0] for c in cs], [c[1] for c in cs]
        w, h = max(us) - min(us), max(vs) - min(vs)
    return {"t": "rect", "u": c.x, "v": c.y, "w": w, "h": h, "r": rr if rr >= SHARP * size else 0.0}


# --- Профил на детайл от струг ---------------------------------------------------------------------------------------


def turned_profile(outer: np.ndarray, U: float, V: float) -> list[tuple[float, float]]:
    """Половината профил (x, радиус) от очертанието на детайл от струг, гледан отстрани: на всяко x радиусът е
    половината ширина. Връща ъглите на начупената линия от x=0 до x=U (без точките по оста)."""
    poly = Polygon(outer)
    xs = np.linspace(0, U, 241)
    r = []
    for x in xs:
        cut = poly.intersection(LineString([(x, -1), (x, V + 1)]))
        r.append(cut.length / 2 if not cut.is_empty else 0.0)
    r = np.asarray(r)
    r[0], r[-1] = r[1], r[-2]  # краищата: срязването точно по ръба е несигурно
    line = LineString(np.c_[xs, r]).simplify(TOL * max(U, V))
    pts = np.asarray(line.coords)
    # Почти хоризонталните части са цилиндри (един радиус), почти вертикалните са рамене (едно x).
    out = [tuple(p) for p in pts]
    for k in range(len(out) - 1):
        (x1, r1), (x2, r2) = out[k], out[k + 1]
        if abs(r2 - r1) <= math.tan(math.radians(SNAP_DEG)) * abs(x2 - x1):
            avg = (r1 + r2) / 2
            out[k], out[k + 1] = (x1, avg), (x2, avg)
    for k in range(len(out) - 1):
        (x1, r1), (x2, r2) = out[k], out[k + 1]
        if abs(x2 - x1) <= math.tan(math.radians(SNAP_DEG)) * abs(r2 - r1) + TOL * U:
            out[k + 1] = (x1, r2)
    return out


NOTCH_RECT = 0.75  # вдлъбнатина, запълваща поне толкова от описания си правоъгълник, е канал с прави стени
NOTCH_SEEN = 0.6  # на другата снимка на изгледа е достатъчно и по-неравна (сянка, перспектива по стената)
NOTCH_DEPTH = 0.02  # и поне толкова дълбока (дял от по-големия размер, и поне NOTCH_MIN_MM)
NOTCH_MIN_MM = 0.8
NOTCH_SHALLOW = 0.12  # и най-много толкова (дял от размера напряко): по-дълбока е самата форма (вътрешността на кука)
NOTCH_WIDE = 0.7  # и не по-широка от толкова от страната, на която е
NOTCH_EDGE = 0.03  # отворена към страната: ръбовете ѝ са до толкова от общия размер


def largest_poly(g) -> Polygon:
    if g.geom_type == "Polygon":
        return g
    return max(getattr(g, "geoms", []), key=lambda p: p.area, default=Polygon())


def notch_gaps(outer: np.ndarray, U: float, V: float, rect: float = NOTCH_RECT) -> list[tuple[Polygon, dict]]:
    """Плитките вдлъбнатини с прави стени по средата на страна на изгледа: (вдлъбнатината, канал)."""
    poly = largest_poly(Polygon(np.asarray(outer, float)).buffer(0))
    if poly.is_empty:
        return []
    size = max(U, V)
    tol = NOTCH_EDGE * size
    gaps = poly.convex_hull.difference(poly)
    out = []
    for g in getattr(gaps, "geoms", [gaps]):
        if g.is_empty or g.geom_type != "Polygon":
            continue
        x0, y0, x1, y1 = g.bounds
        mid_u, mid_v = x0 > tol and x1 < U - tol, y0 > tol and y1 < V - tol  # не в ъгъл (по-тесен край)
        sides = (("v", "-", y0 <= tol and mid_u, x1 - x0, y1, U, V), ("v", "+", y1 >= V - tol and mid_u, x1 - x0,
                  V - y0, U, V), ("u", "-", x0 <= tol and mid_v, y1 - y0, x1, V, U),
                 ("u", "+", x1 >= U - tol and mid_v, y1 - y0, U - x0, V, U))  # fmt: skip
        for key, sign, touches, width, depth, side, across in sides:
            if not touches or depth < max(NOTCH_MIN_MM, NOTCH_DEPTH * size) or depth > NOTCH_SHALLOW * across:
                continue
            if width > NOTCH_WIDE * side or width < 0.5 * depth or g.area < rect * (x1 - x0) * (y1 - y0):
                continue
            box = {"v-": (x0, x1, 0.0, y1), "v+": (x0, x1, y0, V), "u-": (0.0, x1, y0, y1),
                   "u+": (x0, U, y0, y1)}[key + sign]  # fmt: skip
            out.append((g, {"u0": box[0], "u1": box[1], "v0": box[2], "v1": box[3], "open": (key, sign)}))
            break
    return out


def notches(outer: np.ndarray, U: float, V: float, each: list | None = None) -> tuple[np.ndarray, list[dict]]:
    """Канали с прави стени, отворени към една от страните на изгледа (каналът по долната страна на точилка,
    видян от края; вдлъбнатината в края ѝ, видяна отгоре): очертанието без тях (запълнено до правата между ръбовете
    им) и каналите [{"u0", "u1", "v0", "v1", "open": ("u" или "v", "-" или "+")}] в мм на изгледа. Иначе каналът е
    само част от кривата или стъпало, което после се губи, без мерки за ширината и дълбочината. each: очертанията
    от отделните снимки на изгледа - каналът трябва да е на всяка (перспективата прави лъжливи вдлъбнатини само на
    едната)."""
    found = notch_gaps(outer, U, V)
    if each and len(each) > 1:
        others = [notch_gaps(e, U, V, NOTCH_SEEN) for e in each]

        def seen(c, cands):
            a = (c["u1"] - c["u0"]) * (c["v1"] - c["v0"])
            for _, o in cands:
                du = min(c["u1"], o["u1"]) - max(c["u0"], o["u0"])
                dv = min(c["v1"], o["v1"]) - max(c["v0"], o["v0"])
                b = (o["u1"] - o["u0"]) * (o["v1"] - o["v0"])
                if o["open"] == c["open"] and du > 0 and dv > 0 and du * dv >= 0.3 * min(a, b):
                    return True
            return False

        found = [(g, c) for g, c in found if all(seen(c, o) for o in others)]
    if not found:
        return np.asarray(outer, float), []
    filled = largest_poly(Polygon(np.asarray(outer, float)).buffer(0))
    for g, _ in found:
        filled = filled.union(g)
    filled = largest_poly(filled)
    return np.asarray(filled.exterior.coords)[:-1], [c for _, c in found]


ROUND_THICK = 0.75  # плосък детайл, дебел поне толкова от ширината на тялото си: кръгъл (химикалка, маркер)
ROUND_LONG = 3.0  # и поне толкова пъти по-дълъг, отколкото дебел
ROUND_BINS = 120  # точките на профила по дължината
FIN_SHARE = 0.03  # издатина встрани от кръглото тяло (щипката) - ако е поне толкова от очертанието отгоре
FIN_OUT = 0.08  # и стърчи поне толкова от дебелината


def round_body(outer: np.ndarray, U: float, V: float, H: float) -> dict | None:
    """Кръгло тяло, снимано само отгоре и отдолу (химикалка): {"axis": средата му по ширината (мм), "profile":
    [(x, радиус) в мм], "fin": издатината встрани (щипка) - лицето ѝ в мм2}, или None, ако не е такова. Тялото е
    там, където ширината е до дебелината (H, измерена), оста е по средата му, а радиусът по дължината е по-малкото
    разстояние до двата ръба (щипката е само от едната страна)."""
    poly = largest_poly(Polygon(np.asarray(outer, float)).buffer(0))
    if poly.is_empty or U < ROUND_LONG * H:
        return None
    xs = np.linspace(0, U, ROUND_BINS + 1)[1:-1]
    lo, hi = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        c = poly.intersection(LineString([(x, -1), (x, V + 1)]))
        if not c.is_empty:
            lo[i], hi[i] = c.bounds[1], c.bounds[3]
    ok = ~np.isnan(lo)
    width = hi - lo
    body = ok & (width <= 1.15 * H)  # без щипката
    if body.sum() < 0.3 * len(xs):
        return None
    if float(np.median(width[body])) < ROUND_THICK * H:
        return None  # по-тясно, отколкото е дебело: плоско е (стои на ръба си) или мерките са други
    axis = float(np.median((lo[body] + hi[body]) / 2))
    r = np.where(ok, np.minimum(axis - lo, hi - axis), 0.0)
    r = np.clip(r, 0.0, H / 2)
    from scipy import ndimage as nd

    r = nd.median_filter(r, size=3, mode="nearest")
    profile = [(float(x), float(v)) for x, v in zip(xs, r)]
    band = Polygon([(x, axis - v) for x, v in profile] + [(x, axis + v) for x, v in profile[::-1]]).buffer(0)
    out = poly.difference(band.buffer(FIN_OUT * H))  # без тънките ивици по ръба (очертанието е малко по-широко)
    parts = [g for g in getattr(out, "geoms", [out]) if g.geom_type == "Polygon" and g.area > 0]
    fin = max(parts, key=lambda g: g.area, default=None)
    if fin is None or fin.area < FIN_SHARE * poly.area:
        return {"axis": axis, "profile": profile, "fin": 0.0, "fin_x": None}
    x0, _, x1, _ = fin.bounds
    return {"axis": axis, "profile": profile, "fin": float(fin.area), "fin_x": (float(x0), float(x1))}


CURVE_STEP = 0.02  # точките на гладкото очертание: през толкова от по-големия размер (и не повече от CURVE_MAX)
CURVE_MAX = 180
CURVE_CORNER = 50.0  # завой (градуси) на късо разстояние: остър ъгъл, в който кривата се прекъсва
BOX_IOU = 0.965  # правоъгълник със заоблени ъгли, съвпадащ с очертанието поне толкова, е по-добър от свободна форма
BOX_DEV = 0.03  # и средно отклонение на ръба под толкова от по-малкия размер


def smooth_outline(outer: np.ndarray, U: float, V: float) -> tuple[list[tuple[float, float]], list[int]]:
    """Свободна форма (кука, дъга, капка): точки по очертанието на равни разстояния (u, v в мм, обратно на
    часовника) и кои от тях са остри ъгли. Детайлът минава гладко (сплайн) през точките и се чупи само в ъглите,
    затова дъгите остават дъги, а не начупени линии. Шумът от пикселите се изглажда преди това."""
    from scipy import ndimage as nd

    size = max(U, V)
    poly = Polygon(outer).buffer(0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    ring = ccw(np.asarray(poly.exterior.coords))
    step = poly.length / 2000
    p = resample(ring, step)
    sigma = max(1.0, 0.0025 * size / step)
    p = np.c_[nd.gaussian_filter1d(p[:, 0], sigma, mode="wrap"), nd.gaussian_filter1d(p[:, 1], sigma, mode="wrap")]
    n = len(p)
    k = max(2, int(0.012 * size / step))
    ext = np.r_[p, p, p]
    back, fwd = ext[n : 2 * n] - ext[n - k : 2 * n - k], ext[n + k : 2 * n + k] - ext[n : 2 * n]
    turn = np.degrees(np.abs(np.arctan2(back[:, 0] * fwd[:, 1] - back[:, 1] * fwd[:, 0],
                                        (back * fwd).sum(1))))  # fmt: skip
    corners: list[int] = []
    for i in np.argsort(-turn):
        if turn[i] < CURVE_CORNER:
            break
        if all(min(abs(int(i) - c), n - abs(int(i) - c)) > 2 * k for c in corners):
            corners.append(int(i))
    corners.sort()
    gap = max(CURVE_STEP * size, poly.length / CURVE_MAX)
    marks = corners or [0]
    pts, flags = [], []
    for a, b in zip(marks, marks[1:] + [marks[0] + n]):
        m = max(1, int(round((b - a) / (gap / step))))
        for j in range(m):
            q = p[int(round(a + (b - a) * j / m)) % n]
            pts.append((float(np.clip(q[0], 0, U)), float(np.clip(q[1], 0, V))))
            flags.append(j == 0 and bool(corners))
    return flat_sides(pts, flags, U, V)


FLAT_TOL = 0.002  # точки до толкова от общия размер от страната на изгледа лежат на нея
FLAT_LONG = 0.12  # и права е само поредица, дълга поне толкова от страната (не върхът на вълна, който я допира)


def flat_sides(pts: list, flags: list, U: float, V: float) -> tuple[list[tuple[float, float]], list[int]]:
    """Поредица от поне три точки по една страна на изгледа (плоската страна на точилка), дълга поне FLAT_LONG от
    нея, е права между двата си края, които стават ъгли: сплайн, легнал точно в равнината на другото тяло, проваляше сечението (точилка от
    74 000 мм3 ставаше 3 000)."""
    tol = FLAT_TOL * max(U, V)

    def side(q):
        for key, ok in (("u0", q[0] <= tol), ("u1", q[0] >= U - tol), ("v0", q[1] <= tol), ("v1", q[1] >= V - tol)):
            if ok:
                return key
        return None

    n = len(pts)
    lab = [side(q) for q in pts]
    start = next((i for i in range(n) if lab[i] != lab[i - 1]), None)
    if start is None:
        return pts, [i for i, f in enumerate(flags) if f]
    order = [(start + i) % n for i in range(n)]
    snap = {"u0": (0, 0.0), "u1": (0, U), "v0": (1, 0.0), "v1": (1, V)}
    out, marks = [], []
    k = 0
    while k < n:
        j = k
        while j + 1 < n and lab[order[j + 1]] == lab[order[k]]:
            j += 1
        run = order[k:j + 1]
        axis, val = snap.get(lab[order[k]], (0, 0.0))
        span = abs(pts[run[-1]][1 - axis] - pts[run[0]][1 - axis])
        if lab[order[k]] is not None and len(run) >= 3 and span >= FLAT_LONG * (V if axis == 0 else U):
            for i in (run[0], run[-1]):
                q = list(pts[i])
                q[axis] = val
                out.append(tuple(q))
                marks.append(True)
        else:
            for i in run:
                out.append(pts[i])
                marks.append(flags[i])
        k = j + 1
    return out, [i for i, f in enumerate(marks) if f]


def rounded_box(outer: np.ndarray, U: float, V: float) -> float | None:
    """Радиусът на правоъгълник U x V с четири еднакво заоблени ъгъла, който съвпада с очертанието (BOX_IOU,
    BOX_DEV), или None. Овално или с полукръгли краища сечение (флашка, дръжка) е такова."""
    poly = Polygon(outer).buffer(0)
    if poly.geom_type != "Polygon" or poly.area <= 0:
        return None
    small = min(U, V)
    best, best_r = 0.0, 0.0
    for r in np.linspace(0, 0.499 * small, 50):
        box = Polygon(rounded([(0, 0, r), (U, 0, r), (U, V, r), (0, V, r)], steps=16))
        sc = poly.intersection(box).area / max(poly.union(box).area, 1e-12)
        if sc > best:
            best, best_r = sc, float(r)
    if best < BOX_IOU:
        return None
    box = Polygon(rounded([(0, 0, best_r), (U, 0, best_r), (U, V, best_r), (0, V, best_r)], steps=16))
    edge = resample(np.asarray(poly.exterior.coords), poly.length / 400)
    dev = float(np.mean([box.exterior.distance(Point(q)) for q in edge]))
    return best_r if dev <= BOX_DEV * small else None
