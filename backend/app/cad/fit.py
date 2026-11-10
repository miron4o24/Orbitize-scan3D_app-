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
SHARP = 0.02  # ъгъл с радиус под този дял от размера е остър (на снимка не се различава)
CIRCLE_RMS = 0.025
WINDOW = 0.006  # посоката се мери по прозорец ±толкова от размера
MIN_EDGE = 0.025  # най-късата права, дял от размера


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


def corners(outer: np.ndarray, U: float, V: float) -> list[tuple[float, float, float]] | None:
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
    # Съседни почти еднакви прави (прекъснати от шум) се сливат.
    merged = []
    for ln in lines:
        if merged and abs(cross2(merged[-1][1], ln[1])) < math.sin(math.radians(2.5)) and merged[-1][1] @ ln[1] > 0                 and point_line_dist(ln[0], merged[-1][0], merged[-1][1]) < tol:
            c0, d0, s0, _ = merged[-1]
            merged[-1] = (c0, d0, s0, ln[3])
            continue
        merged.append(ln)
    if len(merged) > 2 and abs(cross2(merged[-1][1], merged[0][1])) < math.sin(math.radians(2.5))             and merged[-1][1] @ merged[0][1] > 0 and point_line_dist(merged[0][0], merged[-1][0], merged[-1][1]) < tol:
        c0, d0, _, e0 = merged[0]
        merged[0] = (c0, d0, merged[-1][2], e0)
        merged.pop()
    lines = merged
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
        out.append((float(c[0]), float(c[1]), r if r >= SHARP * size else 0.0))
    if len(out) < 3:
        return None
    poly = Polygon([(u, v) for u, v, _ in out])
    if not poly.is_valid:
        return None
    raw = Polygon(pts).buffer(0)
    iou = poly.intersection(raw).area / max(poly.union(raw).area, 1e-9)
    return out if iou > 0.95 else None


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
    """Кръгъл отвор, част от чийто ръб е скрита (сянка в отвора, отблясък): кръгът, на който лежи поне половината
    очертание по поне половин обиколка (RANSAC и после най-малки квадрати по съгласните точки). None: не е кръг."""
    n = len(pts)
    if n < 12:
        return None
    rng = np.random.default_rng(seed)
    span = float(np.max(np.ptp(pts, axis=0)))
    best, best_in = None, None
    for _ in range(400):
        a, b, c = pts[rng.choice(n, 3, replace=False)]
        circ = circle_3(a, b, c)
        if circ is None or not 0.25 * span < circ[2] < 1.5 * span:
            continue
        thr = 0.02 * circ[2] + 0.01
        inl = np.abs(np.hypot(pts[:, 0] - circ[0], pts[:, 1] - circ[1]) - circ[2]) < thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = circ, inl
    if best is None or best_in.sum() < 0.5 * n:
        return None
    cx, cy, r, _ = circle_fit(pts[best_in])
    ang = np.sort(np.arctan2(pts[best_in, 1] - cy, pts[best_in, 0] - cx))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    if 2 * math.pi - gaps.max() < math.pi:  # покрива под половин обиколка
        return None
    # Отворът е в кръга (сянката го смалява, но не го уголемява).
    if Polygon(pts).buffer(0).area > math.pi * r * r * 1.08:
        return None
    return cx, cy, r


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
