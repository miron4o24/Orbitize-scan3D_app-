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
    if len(out) < 3:
        return None
    if not Polygon([(u, v) for u, v, _ in out]).is_valid:
        return None
    poly = Polygon(rounded(out)).buffer(0)
    raw = Polygon(pts).buffer(0)
    iou = poly.intersection(raw).area / max(poly.union(raw).area, 1e-9)
    return out if iou > 0.95 else None


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
TAB_MAX = 0.2  # и не по-дълъг от толкова от дължината


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
            out.append({"end": end, "len": length, "lo": d_lo, "hi": d_hi})
    return out


def apply_tabs(cs: list, tabs: list, U: float, V: float) -> list:
    """Ъглите на очертанието (обратно на часовника) с по-тесните краища: ъглите в края (и намерените от тях
    стъпала) се заменят със стъпало навътре от двете страни; радиусът им остава за ъглите на по-тесния край."""
    for t in tabs:
        x_e = t["len"] if t["end"] == 0 else U - t["len"]
        reach = 1.2 * t["len"]
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
        r = min(max(cs[k][2] for k in block), 0.45 * t["len"], 0.45 * (V - t["lo"] - t["hi"]))
        lo, hi = t["lo"], V - t["hi"]
        if t["end"] == 0:  # отляво обратно на часовника: отгоре надолу
            new = [(x_e, V, 0.0), (x_e, hi, 0.0), (0.0, hi, r), (0.0, lo, r), (x_e, lo, 0.0), (x_e, 0.0, 0.0)]
        else:  # отдясно: отдолу нагоре
            new = [(x_e, 0.0, 0.0), (x_e, lo, 0.0), (U, lo, r), (U, hi, r), (x_e, hi, 0.0), (x_e, V, 0.0)]
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


def robust_circle(pts: np.ndarray, seed: int = 0) -> tuple[float, float, float] | None:
    """Кръгъл отвор, част от чийто ръб е скрита: кръгът, на който лежи поне 30% от очертанието по поне 150° от
    обиколката, и в който се побира целият видим отвор. От няколко такива - най-големият: при снимка с перспектива
    през проходен отвор се вижда сечението на близкия ръб (истинският размер) и по-малкия далечен; сянка в отвора
    също само смалява видимото. None: не е кръг."""
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
        if not 0.45 * math.pi * r * r <= area <= 1.08 * math.pi * r * r:  # видимият отвор е в кръга и го запълва
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


def hole_shape(pts: np.ndarray, size: float) -> dict:
    """Отвор: кръг {t: hole, u, v, d}, канал {t: slot, u, v, len, w, angle} или правоъгълник {t: rect, u, v, w, h,
    r}; u, v е средата."""
    pts = np.asarray(pts, float)
    circle = robust_circle(pts)
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
