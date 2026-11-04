"""Гъст облак от точки + маски на предмета -> чиста, затворена мрежа, готова за печат. Чист код с
numpy/trimesh/pymeshlab без зависимост от COLMAP/OpenMVS, за да може да се тества. Координати след `clean_cloud`:
+Z е нагоре, предметът стои върху z = ground_z (при намерена маса). Два източника на форма, съчетани в `poisson`:
гъстите точки, попадащи в очертанието на предмета в повечето кадри (детайл, вдлъбнати части); обвивката от
очертанията: обемът в очертанието на предмета в (почти) всеки кадър. Тя не иска шарка, затова запълва тъмни,
еднообразни или лъскави части, където гъстото съпоставяне намира малко точки."""

import json
import struct
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pymeshlab
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

from . import parallel as PAR

UP = np.array([0.0, 0.0, 1.0])
OBJECT_AGREE = 0.7  # гъста точка е върху предмета, ако е в очертанието в >= 70% от кадрите, които я показват
HULL_AGREE = 0.75  # воксел е в обвивката, ако е в очертанието в >= 75% от тях (маските не са идеални)
MIN_VIEWS = 5


class MeshError(Exception):
    """Проблем при възстановяването за потребителя."""


@dataclass
class Cloud:
    points: np.ndarray  # (n, 3)
    normals: np.ndarray  # (n, 3)
    colors: np.ndarray  # (n, 4) float 0..1
    ground_z: float | None = None
    to_up: np.ndarray | None = None  # завъртане свят -> тези координати
    weights: np.ndarray | None = None  # за точка, за Поасон (по подразбиране 1)


@dataclass
class Views:
    """Камери с точков отвор, конвенция на COLMAP: пиксел = K (R X + t), x надясно, y надолу, z напред."""

    R: np.ndarray  # (n, 3, 3)
    t: np.ndarray  # (n, 3)
    K: np.ndarray  # (n, 4) fx, fy, cx, cy
    names: list[str]

    def __len__(self):
        return len(self.names)

    @property
    def centers(self) -> np.ndarray:
        return -np.einsum("nji,nj->ni", self.R, self.t)

    @property
    def dirs(self) -> np.ndarray:
        return self.R[:, 2]  # +Z на камерата (напред) в света

    @property
    def ups(self) -> np.ndarray:
        return -self.R[:, 1]  # -Y на камерата (нагоре в изображението) в света


def silhouette_consistency(
    views: Views, masks: list[np.ndarray], scale: int, cells: int = 64, agree: float = HULL_AGREE
) -> np.ndarray:
    """За всеки кадър: припокриване (IoU) на маската на предмета с обвивката от всички кадри, проектирана обратно в
    него. При верни положения на камерите обвивката запълва всяка маска; при грешни конусите на маските едва се
    пресичат и обвивката се свива до тънка ивица (решение, което поставяше 53 кадъра "в целта", даде клин). NaN:
    няма предмет."""
    iou = np.full(len(views), np.nan)
    grid = hull_grid(views, masks, scale, cells, agree)
    if grid is None:
        return iou
    occ, step = grid
    for i, (R, t, K, m) in enumerate(zip(views.R, views.t, views.K, masks)):
        iou[i] = outline_iou(occ, step, R, t, K, m, scale)
    return iou


def hull_grid(views: Views, masks: list[np.ndarray], scale: int, cells: int = 64, agree: float = HULL_AGREE):
    """Центрове на воксели в обвивката от всички кадри, върху мрежа cells³ около мястото, където се срещат лъчите през
    центровете на маските, и размерът на воксела. None без достатъчно кадри с предмета."""
    n = len(views)
    rays, sizes = [], []
    for i, (R, t, (fx, fy, cx, cy), m) in enumerate(zip(views.R, views.t, views.K, masks)):
        ys, xs = np.nonzero(m > 127)
        if len(xs) < 10:
            continue
        u, v = (xs.mean() + 0.5) * scale, (ys.mean() + 0.5) * scale
        d = R.T @ [(u - cx) / fx, (v - cy) / fy, 1.0]
        rays.append((i, -R.T @ t, d / np.linalg.norm(d)))
        sizes.append(0.5 * max(np.ptp(xs), np.ptp(ys)) * scale / fx)  # тангенс на ъгловия полуразмер на предмета
    if len(rays) < 3:
        return None
    c = np.array([r[1] for r in rays])
    d = np.array([r[2] for r in rays])
    proj = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    try:
        center = np.linalg.solve(proj.sum(0), np.einsum("nij,nj->i", proj, c))
    except np.linalg.LinAlgError:
        return None
    radius = 1.5 * float(np.median(np.array(sizes) * np.linalg.norm(center - c, axis=1)))
    step = 2 * radius / cells
    axes = [center[k] - radius + step * (np.arange(cells) + 0.5) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    n_in, n_fg = silhouette_votes(grid, views, masks, scale)
    return grid[(n_in >= min(MIN_VIEWS, n)) & (n_fg >= agree * n_in)], step


def outline_iou(occ: np.ndarray, step: float, R, t, K, m: np.ndarray, scale: int) -> float:
    """Припокриване (IoU) на маската на кадър с воксели от обвивката `occ`, проектирани в него; NaN, ако не показва
    предмет."""
    fx, fy, cx, cy = K
    fg = m > 127
    if fg.sum() < 10:
        return np.nan
    hull = np.zeros_like(fg)
    Xc = occ @ np.asarray(R).T + t
    Xc = Xc[Xc[:, 2] > 1e-9]
    if len(Xc):
        u = np.floor((fx * Xc[:, 0] / Xc[:, 2] + cx) / scale).astype(np.int64)
        v = np.floor((fy * Xc[:, 1] / Xc[:, 2] + cy) / scale).astype(np.int64)
        ok = (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])
        hull[v[ok], u[ok]] = True
        # Центровете на вокселите оставят празнини: затварят се с размера на вокселите на екрана.
        px = step * fx / np.median(Xc[:, 2]) / scale
        if px > 1:
            hull = ndimage.binary_closing(hull, iterations=int(np.ceil(px)))
    return float((hull & fg).sum() / max(1, (hull | fg).sum()))


def subset(views: Views, keep: np.ndarray) -> Views:
    return Views(views.R[keep], views.t[keep], views.K[keep], [n for n, k in zip(views.names, keep) if k])


def consistent_frames(views: Views, masks: list[np.ndarray], scale: int, min_iou: float = 0.5) -> np.ndarray:
    """Кадри, чието положение на камерата е съгласувано с другите: маската им съвпада с обвивката от останалите кадри.
    Няколко зле разположени кадъра свиват обвивката и правят и добрите да изглеждат зле, затова най-лошите отпадат с
    по четвърт и обвивката се изгражда отново. Връща маска върху изгледите."""
    keep = np.array([(m > 127).sum() >= 10 for m in masks])
    for _ in range(8):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            return np.zeros(len(views), bool)
        iou = silhouette_consistency(subset(views, keep), [masks[i] for i in idx], scale)
        bad = np.flatnonzero(~(iou >= min_iou))  # NaN се брои за лошо
        if len(bad) == 0:
            return keep
        worst = bad[np.argsort(np.nan_to_num(iou[bad], nan=-1))][: max(1, len(idx) // 4)]
        keep[idx[worst]] = False
    # Неустановено: броят се само кадрите, съгласувани с последната обвивка (грешно решение не бива да получава оценка
    # от кадрите, останали след отпадането: модел, при който всеки кадър съвпадаше с очертанието си с 1-2%, веднъж
    # "запази" 43).
    idx = np.flatnonzero(keep)
    iou = silhouette_consistency(subset(views, keep), [masks[i] for i in idx], scale)
    keep[idx[~(iou >= min_iou)]] = False
    return keep


def silhouette_votes(X: np.ndarray, views: Views, masks: list[np.ndarray], scale: int):
    """За всяка точка: в колко кадъра е в изображението и в колко от тях е в очертанието на предмета. masks[i] е
    маската на views[i] (uint8, предмет > 127) при 1/scale от размера на кадъра."""
    n_in = np.zeros(len(X), np.int32)
    n_fg = np.zeros(len(X), np.int32)
    for s in range(0, len(X), 1_000_000):  # ограничена памет за големи облаци и мрежи
        Xs = X[s : s + 1_000_000]

        def look(frame):
            R, t, (fx, fy, cx, cy), m = frame
            Xc = PAR.turned(Xs, R) + t
            z = np.where(Xc[:, 2] > 1e-9, Xc[:, 2], np.nan)
            u = np.floor((fx * Xc[:, 0] / z + cx) / scale)
            v = np.floor((fy * Xc[:, 1] / z + cy) / scale)
            ok = (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])  # NaN (зад камерата) е False
            return ok, m[v[ok].astype(np.int64), u[ok].astype(np.int64)] > 127

        for ok, inside in PAR.each(look, zip(views.R, views.t, views.K, masks)):
            n_in[s : s + len(Xs)] += ok
            n_fg[s : s + len(Xs)][ok] += inside
    return n_in, n_fg


def rotation_to(a: np.ndarray, b: np.ndarray = UP) -> np.ndarray:
    """Матрица на завъртане, обръщаща единичния вектор a към единичния вектор b (Родригес)."""
    a = a / np.linalg.norm(a)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def statistical_inliers(points: np.ndarray, k: int = 16, n_std: float = 2.0) -> np.ndarray:
    d, _ = cKDTree(points).query(points, k=k + 1)
    mean = d[:, 1:].mean(1)
    return mean < mean.mean() + n_std * mean.std()


def find_ground(points: np.ndarray, thresh: float, max_tilt_deg: float = 20, iters: int = 400, seed: int = 0):
    """RANSAC за почти хоризонтална равнина n.p + d = 0, съдържаща >= 10% от точките, с < 5% под нея. Правилото "нищо
    отдолу" пречи плоско стъпало или платформа в предмета да се сметне за маса. Връща (n, d) или None."""
    rng = np.random.default_rng(seed)
    best, best_count = None, int(0.1 * len(points))
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < cos_tilt:
            continue
        d = -n @ p[0]
        dist = points @ n + d
        count = int((np.abs(dist) < thresh).sum())
        if count > best_count and (dist < -thresh).mean() < 0.05:
            best, best_count = (n, d), count
    return best


def estimate_up(cam_centers: np.ndarray, cam_dirs: np.ndarray, cam_ups: np.ndarray) -> np.ndarray:
    mean_up = cam_ups.mean(0)
    if np.linalg.norm(mean_up) > 0.8:  # снимки с изправен телефон (приложението): "нагоре" в изображението е обратно на гравитацията
        return mean_up
    # Наклонени камери: вместо това се използва обиколката. Камерите обикалят предмета, затова най-добре пасващата им
    # равнина е хоризонтална, а хората снимат предмети леко отгоре, така че камерите средно гледат надолу.
    n = np.linalg.svd(cam_centers - cam_centers.mean(0))[2][2]
    return n if cam_dirs.mean(0) @ n <= 0 else -n


def clean_cloud(
    points: np.ndarray,
    normals: np.ndarray | None,
    colors: np.ndarray,
    views: Views,
    masks: list[np.ndarray],
    mask_scale: int,
    up: np.ndarray | None = None,
    find_table: bool = True,
) -> Cloud:
    """Точките на предмета, изравнени по масата: всичко извън очертанието (маса, фон, носещи се точки) отпада чрез
    маските, което работи при всякакъв предмет, фон или път на камерата. up: накъде е нагоре, когато е известно
    (сензорът за гравитация на телефона); иначе се предполага от камерите."""
    n_in, n_fg = silhouette_votes(points, views, masks, mask_scale)
    on_object = (n_in >= min(MIN_VIEWS, len(views))) & (n_fg >= OBJECT_AGREE * n_in)
    if on_object.sum() < 300:
        raise MeshError(  # само on_object.sum() < 300 точки върху предмета
            "Предметът не можа да се отдели от фона. Дръжте целия предмет в кръга на екрана и го сложете на подложка "
            "в различен цвят от него."
        )

    R = rotation_to(up if up is not None else estimate_up(views.centers, views.dirs, views.ups))
    pts, cams = points @ R.T, views.centers @ R.T
    obj = pts[on_object]
    lo, hi = np.percentile(obj, 2, axis=0), np.percentile(obj, 98, axis=0)
    size = float((hi - lo).max())

    # Масата: хоризонтална равнина сред другите точки около и точно под предмета. Изравняването по нея прави плоската
    # основа наистина плоска.
    ground_z = None
    thresh = 0.015 * size
    mid = (lo + hi) / 2
    around = ~on_object & (np.linalg.norm(pts[:, :2] - mid[:2], axis=1) < 1.5 * size)
    around &= np.abs(pts[:, 2] - lo[2]) < 0.5 * size
    # find_table=False: измерването е само около предмета, затова истинската маса не е сред точките.
    plane = find_ground(pts[around], thresh) if find_table and around.sum() >= 100 else None
    if plane is not None:
        n, d = plane
        R2 = rotation_to(n)
        R, pts, cams = R2 @ R, pts @ R2.T, cams @ R2.T
        ground_z = -d  # n.p + d = 0  ->  z' = -d след обръщане на n към +Z

    keep = on_object & (pts[:, 2] > ground_z + thresh if ground_z is not None else True)
    pts, colors = pts[keep], colors[keep]
    nrm = normals[keep] @ R.T if normals is not None else estimate_normals(pts)
    inl = statistical_inliers(pts)
    pts, colors, nrm = pts[inl], colors[inl], nrm[inl]
    # Нормали без страна (оценени или от облак без такива) гледат към най-близката камера. Тези на OpenMVS вече гледат
    # към кадъра, който ги е измерил (96-99,6% към най-близката камера при реални сканирания, спрямо половината при
    # тези без страна), и се запазват: обърната към най-близката камера, вътрешната стена на чаша гледаше към камерата
    # зад нея, извън чашата (4% от точките ѝ, 14% при храм, между колоните).
    _, idx = cKDTree(cams).query(pts)
    flip = np.einsum("ij,ij->i", nrm, cams[idx] - pts) < 0
    if normals is None or flip.mean() > 0.2:
        nrm[flip] *= -1
    return Cloud(pts, nrm, colors, ground_z, R)


def _hull_fraction(views: Views, masks, scale: int, to_up: np.ndarray, axes: list[np.ndarray]) -> np.ndarray:
    """За всеки център на воксел (мрежа в системата с посока нагоре): дял от кадрите, които го показват, с него в
    очертанието."""
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) @ to_up  # обратно към света
    n_in, n_fg = silhouette_votes(grid, views, masks, scale)
    frac = np.where(n_in >= min(MIN_VIEWS, len(views)), n_fg / np.maximum(n_in, 1), 0.0)
    return frac.reshape([len(a) for a in axes])


def visual_hull(
    views: Views,
    masks,
    scale: int,
    cloud: Cloud,
    cells: int = 160,
    agree: float = HULL_AGREE,
    limit: Callable[[np.ndarray], np.ndarray] | None = None,
) -> trimesh.Trimesh | None:
    """Затворена мрежа на обема в очертанието на предмета в (почти) всички кадри, стояща върху масата. agree: делът от
    кадрите, в които вокселът трябва да е вътре. Под 1 прощава лоши маски и положения, но разширява каквото само
    няколко кадъра виждат тясно: късата страна на овал, заснет отвсякъде, излизаше с 17% по-широка при 0,75. limit: за
    световни точки в очертанията, най-големият дял на воксела им от друго доказателство (AI дълбочина: под `agree`,
    където не вижда нищо). Гладко, така че и отрязаната повърхност е гладка."""
    found = hull_volume(views, masks, scale, cloud, cells, agree)
    if found is None:
        return None
    frac, axes, step = found
    if limit is not None:
        inside, points = voxels_inside(frac, axes, agree)
        frac = limited(frac, inside, limit(points @ cloud.to_up))  # обратно към света
    return hull_mesh(frac, axes, step, agree)


def hull_volume(
    views: Views, masks, scale: int, cloud: Cloud, cells: int = 160, agree: float = HULL_AGREE
) -> tuple[np.ndarray, list[np.ndarray], float] | None:
    """Обвивката като мрежа в координатите на облака: за воксел делът от кадрите, които го показват, с него в
    очертанието. Връща (дялове, осите на мрежата, размер на клетка), None, ако нито един воксел не достига `agree`."""
    lo, hi = np.percentile(cloud.points, 1, axis=0), np.percentile(cloud.points, 99, axis=0)
    floor = floor_z(cloud)
    # Груб проход върху щедра кутия (части без гъсти точки може да стърчат), после фин около обвивката.
    pad = 0.5 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    lo[2] = floor
    for n in (48, cells):
        step = float((hi - lo).max()) / n
        axes = [lo[k] + step * (np.arange(int(np.ceil((hi[k] - lo[k]) / step))) + 0.5) for k in range(3)]
        frac = _hull_fraction(views, masks, scale, cloud.to_up, axes)
        inside = np.argwhere(frac >= agree)
        if len(inside) == 0:
            return None
        # Следващ проход: заетата кутия плюс две груби клетки; нищо под масата.
        lo = np.array([axes[k][inside[:, k].min()] for k in range(3)]) - 2 * step
        hi = np.array([axes[k][inside[:, k].max()] for k in range(3)]) + 2 * step
        lo[2] = max(lo[2], floor)
    return frac, axes, step


def voxels_inside(frac: np.ndarray, axes: list[np.ndarray], agree: float = HULL_AGREE):
    """(индекси, центрове в координатите на мрежата) на вокселите в обвивката."""
    inside = np.argwhere(frac >= agree)
    return inside, np.stack([axes[k][inside[:, k]] for k in range(3)], -1)


def limited(frac: np.ndarray, inside: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Дяловете, най-много `limit` при вокселите `inside` (вижте visual_hull)."""
    out = frac.copy()
    i, j, k = inside.T
    out[i, j, k] = np.minimum(out[i, j, k], limit)
    return out


def free_space_limit(views: Views, points: np.ndarray, normals: np.ndarray, probes: np.ndarray, margin: float,
                     agree: float = HULL_AGREE, cell: int = 4) -> np.ndarray:  # fmt: skip
    """За световни точки в обвивката (проби): най-големият им дял в нея (вижте visual_hull), от измерените точки на
    повърхността (свят, нормали към страната, от която са видени). Кадър, измерил точка от повърхността, е видял през
    всичко между него и точката: вътрешността на чаша отгоре и отворът на дръжка са празни, макар очертание да не го
    показва (обвивката ги запълва; чаша излизаше плътна). За всеки кадър обърнатите към него точки дават карта на
    дълбочината (най-близката на всеки `cell` пиксела); проба пред тази дълбочина се вижда през нея от този кадър,
    проба при нея е върху повърхността, а проба зад нея е скрита и не получава глас (така кадрите, които не виждат в
    чаша, не я пазят пълна). Гласовете се осредняват, всеки ограничен до две граници, както в depth.carve_limit;
    проби, невидени от никой кадър, запазват 1."""
    seen = np.zeros(len(probes), np.int32)
    total = np.zeros(len(probes))
    for R, t, (fx, fy, cx, cy) in zip(views.R, views.t, views.K):
        Xc = points @ R.T + t
        z = Xc[:, 2]
        ok = (z > 1e-9) & (np.einsum("ij,ij->i", normals, -R.T @ t - points) > 0)
        w, h = int(2 * cx) // cell + 1, int(2 * cy) // cell + 1  # коригирани кадри: главната точка е в средата
        u = np.floor((fx * Xc[ok, 0] / z[ok] + cx) / cell).astype(np.int64)
        v = np.floor((fy * Xc[ok, 1] / z[ok] + cy) / cell).astype(np.int64)
        inb = (u >= 0) & (v >= 0) & (u < w) & (v < h)
        depth = np.full((h, w), np.inf)
        np.minimum.at(depth, (v[inb], u[inb]), z[ok][inb])
        Pc = probes @ R.T + t
        pz = Pc[:, 2]
        idx = np.flatnonzero(pz > 1e-9)
        pu = np.floor((fx * Pc[idx, 0] / pz[idx] + cx) / cell).astype(np.int64)
        pv = np.floor((fy * Pc[idx, 1] / pz[idx] + cy) / cell).astype(np.int64)
        inb = (pu >= 0) & (pv >= 0) & (pu < w) & (pv < h)
        idx, pu, pv = idx[inb], pu[inb], pv[inb]
        far = depth[pv, pu] - pz[idx]  # колко отвъд пробата този кадър е измерил повърхността
        vote = np.isfinite(far) & (far > -margin)
        seen[idx[vote]] += 1
        total[idx[vote]] += np.clip(far[vote], -2 * margin, 2 * margin)
    mean = total / np.maximum(seen, 1)
    share = agree + (margin - mean) * (1 - agree) / (0.2 * margin)
    return np.where(seen >= 2, np.clip(share, 0, 1), 1.0)


def hull_mesh(frac: np.ndarray, axes: list[np.ndarray], step: float, agree: float = HULL_AGREE):
    """Затворената повърхност, където дяловете пресичат `agree` (най-голямата част, нормали навън), None, ако няма
    такава."""
    # Нулева граница, за да се затвори повърхността при ръбовете на кутията; дъното се затваря при масата.
    vol = np.pad(ndimage.gaussian_filter(frac, 1.0), 1)
    if vol.max() <= agree:
        return None
    verts, faces, _, _ = measure.marching_cubes(vol, agree)
    origin = np.array([a[0] for a in axes]) - step
    hull = trimesh.Trimesh(origin + step * verts, faces)
    hull = max(hull.split(only_watertight=False), key=lambda p: len(p.faces))
    trimesh.repair.fix_normals(hull)  # навън
    # Копие: след обръщане на стените навън trimesh (5.1) пази в кеша си старите нормали навътре и hull_samples ги
    # взимаше; Поасон тогава хлътваше страните на кутия и я оставяше с една трета по-плоска.
    return hull.copy()


def estimate_normals(points: np.ndarray) -> np.ndarray:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points))
    ms.compute_normal_for_point_clouds(k=12)
    return ms.current_mesh().vertex_normal_matrix().copy()


def hull_samples(hull: trimesh.Trimesh, pts: np.ndarray, nrm: np.ndarray, n: int = 60_000):
    """Точки върху повърхността на обвивката с нормали навън и две маски върху тях. near: наблизо има видяна точка
    (обвивката съвпада с данните там); over: обвивката надвишава там. Това става, където никой кадър не е гледал по
    повърхност: плосък връх, заснет само отгоре под ъгъл, става конус (от 35° отгоре висок до 0,7 от радиуса на
    върха). Където видяна повърхност, гледаща към същата страна, лежи под проба от обвивката, пробата е грешна."""
    samples, face = trimesh.sample.sample_surface_even(hull, n, seed=0)
    normals = hull.face_normals[face]
    size = hull.extents.max()
    tree = cKDTree(pts)
    # Търсене, ограничено до значимите разстояния: без граница пробите дълбоко в предмета отнемаха 70 с за 72 хиляди
    # точки и минути за половин милион от сканиране на сървъра (два пъти на сканиране).
    dist, _ = tree.query(samples, distance_upper_bound=0.03 * size, workers=-1)
    near = np.isfinite(dist)
    over = np.zeros(len(samples), bool)
    for below in np.linspace(0.03, 0.7, 23) * size:
        # Проба с ширина 5%: точките по еднообразни повърхности са редки.
        d, idx = tree.query(samples - normals * below, distance_upper_bound=0.05 * size, workers=-1)
        hit = np.flatnonzero(~near & np.isfinite(d))
        over[hit[np.einsum("ij,ij->i", nrm[idx[hit]], normals[hit]) > 0.2]] = True
    return samples, normals, near, over


def poisson(
    cloud: Cloud,
    hull: trimesh.Trimesh | None = None,
    depth: int = 8,
    target_faces: int = 200_000,
    samples_per_node: float = 1.5,
    smooth: int = 5,
    point_weight: float = 4.0,
) -> trimesh.Trimesh:
    """Затворена повърхност през точките на предмета. Където те оставят празнини (еднообразни, тъмни или невидени
    части), ги запълва повърхността на обвивката с по-малко тегло, така че истинските точки печелят, където има и
    двете. Поасон трябва да затвори обем: само върху точките предмет с празнини или отворена долна страна става
    петно, запълващо цялата кутия. Дълбочина 8 (клетки 1/256 от размера, около двойно разстоянието между гъстите
    точки): при 9 клетките съвпадаха с разстоянието и повърхността следваше шума на точките (дрелка: двойно
    по-неравна)."""
    pts, nrm = cloud.points, cloud.normals
    conf = cloud.weights if cloud.weights is not None else np.ones(len(pts))
    if hull is not None:
        samples, normals, near, over = hull_samples(hull, pts, nrm)
        gap = ~near & ~over
        pts = np.vstack([pts, samples[gap]])
        nrm = np.vstack([nrm, normals[gap]])
        conf = np.concatenate([conf, np.full(int(gap.sum()), 0.3)])
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=pts, v_normals_matrix=nrm, v_scalar_array=conf))
    ms.generate_surface_reconstruction_screened_poisson(
        depth=depth, preclean=True, confidence=True, samplespernode=samples_per_node, pointweight=point_weight
    )
    if smooth:
        ms.apply_coord_taubin_smoothing(stepsmoothnum=smooth)  # гъстите точки по тъмни или еднообразни повърхности са шумни
    if ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces, preservenormal=True, preservetopology=True
        )
    m = ms.current_mesh()
    verts, faces = m.vertex_matrix(), m.face_matrix()
    # Изрязва повърхност без опора от входни точки: през незаснет участък Поасон се разлива като лист до ръба на
    # мрежата си. make_printable затваря оставената дупка. Не близо до дъното: там затворената повърхност, измислена
    # от Поасон под невидяната долна страна, е това, което срязването при пода превръща в плоска основа.
    tree = cKDTree(pts)
    dist, _ = tree.query(verts)
    sample = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20_000), replace=False)]
    spacing = float(np.median(tree.query(sample, k=2)[0][:, 1]))  # типична празнина между входните точки
    reach = max(0.06 * float(np.ptp(pts, axis=0).max()), 4 * spacing)  # разлетите листове стигат доста над 10%
    low, high = np.percentile(pts[:, 2], [2, 98])
    supported = (dist <= reach) | (verts[:, 2] < low + 0.15 * (high - low))
    faces = faces[supported[faces].all(axis=1)]
    # Цвят от най-близката точка на предмета (областите само от обвивката взимат цвета на най-близката видяна
    # повърхност).
    _, idx = cKDTree(cloud.points).query(verts)
    colors = (np.clip(cloud.colors[idx], 0, 1) * 255).astype(np.uint8)
    out = trimesh.Trimesh(verts, faces, vertex_colors=colors)
    out.remove_unreferenced_vertices()
    return out


def clip_to_outline(
    mesh: trimesh.Trimesh, views: Views, masks: list[np.ndarray], scale: int, cloud: Cloud, agree: float = 0.6
) -> trimesh.Trimesh:
    """Премахва повърхност извън очертанието на предмета: Поасон може да се разлее като листове, където никой кадър не
    е видял предмета (например незаснет участък от кръга). Истинската повърхност никога не е извън обвивката. Всяка
    стена се проверява малко навътре в себе си (1% от размера по вътрешната ѝ нормала), така че стените точно върху
    обвивката, като страните на цилиндър, остават. make_printable затваря оставените дупки."""
    if mesh.is_empty:
        return mesh
    probe = mesh.triangles_center - mesh.face_normals * 0.01 * mesh.extents.max()
    n_in, n_fg = silhouette_votes(probe @ cloud.to_up, views, masks, scale)
    inside = (n_in < min(MIN_VIEWS, len(views))) | (n_fg >= agree * n_in)  # невидяно: няма доказателство за изрязване
    out = mesh.copy()
    out.update_faces(inside)
    out.remove_unreferenced_vertices()
    return out


def _meshlab_close(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Поправка на многообразието и запълване на дупки от MeshLab (без проверката му за самопресичане: с нея дупките
    на едно сканиране работеха над 10 минути), със запазени цветове. Части под 1% от стените (тънки ивици) отпадат."""
    ms = pymeshlab.MeshSet()
    colors = mesh.visual.vertex_colors[:, :4] / 255.0 if mesh.visual.kind == "vertex" else None
    kw = {"v_color_matrix": colors} if colors is not None else {}
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces, **kw))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=1_000_000, newfaceselected=False, selfintersection=False)
    m = ms.current_mesh()
    out_colors = (np.clip(m.vertex_color_matrix(), 0, 1) * 255).astype(np.uint8) if colors is not None else None
    out = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), vertex_colors=out_colors)
    parts = out.split(only_watertight=False)
    return trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.01 * len(out.faces)])


def close_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Затваря всяка останала дупка (изрязани невидени области, срязвания), запазвайки цветовете. Запълването на
    MeshLab може да остави няколко ръба, общи за три или повече стени, където се срещат дублирани точки (MeshLab не
    ги вижда; колонка имаше 8): стените около тях отпадат и малките дупки се запълват отново. Ако повърхността още
    е отворена, се изгражда затворена от самата нея (`reclose`, което губи фин детайл; и двете сканирания на кутии
    минаваха през него преди тази поправка)."""
    out = _meshlab_close(mesh)
    for _ in range(3):
        if out.is_watertight:
            return out
        edges, counts = np.unique(out.edges_sorted, axis=0, return_counts=True)
        bad = np.unique(edges[counts > 2])
        if not len(bad):
            trimesh.repair.fill_holes(out)
            break
        # Стените в рамките на два пръстена от точките на лошите ръбове отпадат; оставените дупки са прости контури.
        ring = np.isin(out.faces, bad).any(axis=1)
        ring = np.isin(out.faces, np.unique(out.faces[ring])).any(axis=1)
        out.update_faces(~ring)
        out.remove_unreferenced_vertices()
        out = _meshlab_close(out)
    return out if out.is_watertight else reclose(mesh)


def reclose(mesh: trimesh.Trimesh, depth: int = 9) -> trimesh.Trimesh:
    """Затворена повърхност през отворена: Поасон върху точки, взети равномерно по нея. Поасон винаги затваря обем,
    затова това приключва всяка дупка, колкото и заплетена да е; детайлът остава в рамките на клетка от мрежата (1/512
    от размера). 400 хиляди проби са на около 0,3 мм една от друга при кутия от 10 см, близо до тази клетка (200
    хиляди: 0,5 мм, повече закръгляне)."""
    samples, face = trimesh.sample.sample_surface_even(mesh, 400_000, seed=0)
    normals = mesh.face_normals[face]
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=samples, v_normals_matrix=normals))
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)
    m = ms.current_mesh()
    out = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix())
    out = max(out.split(only_watertight=False), key=lambda p: len(p.faces))
    if mesh.visual.kind == "vertex":
        _, idx = cKDTree(mesh.vertices).query(out.vertices)
        out.visual.vertex_colors = mesh.visual.vertex_colors[idx]
    return out


PAINT_REDUCE = 2  # кадрите се четат при половин размер за цветовете: около разстоянието между точките на модел с 250 хиляди триъгълника
PAINT_CELL = 3  # (смалени) пиксели: за всеки квадрат с такава ширина се виждат само най-близките до камерата точки


def paint(mesh: trimesh.Trimesh, views: Views, photos: list, to_up: np.ndarray) -> None:
    """Цветове на върховете от снимките (мрежата в координати, завъртени с to_up от световните, като на Cloud). Всеки
    връх смесва кадрите, които го виждат, претеглени към тези, гледащи го най-пряко, така че разликите в експозицията
    се смесват вместо да оставят шевове. Видян: обърнат към камерата и сред най-близките до нея точки там, където
    попада (в рамките на 1% от разстоянието), така че част, скрита зад друга, не взима нищо от този кадър. Върхове,
    невидени от никой кадър (плоската основа), взимат цвета на най-близкия видян."""
    verts, normals = mesh.vertices @ to_up, mesh.vertex_normals @ to_up  # обратно към света
    total, weight = np.zeros((len(verts), 3)), np.zeros(len(verts))
    for R, t, (fx, fy, cx, cy), p in zip(views.R, views.t, views.K, photos):
        with Image.open(p) as im:
            img = np.asarray(im.convert("RGB").reduce(PAINT_REDUCE), np.float32)
        Xc = verts @ R.T + t
        z = np.maximum(Xc[:, 2], 1e-9)
        to_cam = -R.T @ t - verts
        facing = np.einsum("ij,ij->i", normals, to_cam) / np.linalg.norm(to_cam, axis=1)
        u = (fx * Xc[:, 0] / z + cx) / PAINT_REDUCE
        v = (fy * Xc[:, 1] / z + cy) / PAINT_REDUCE
        inside = (u >= 0) & (v >= 0) & (u < img.shape[1]) & (v < img.shape[0])
        idx = np.flatnonzero((Xc[:, 2] > 1e-9) & (facing > 0.1) & inside)
        if not len(idx):
            continue
        cell = (v[idx] // PAINT_CELL).astype(np.int64) * 100_000 + (u[idx] // PAINT_CELL).astype(np.int64)
        _, inv = np.unique(cell, return_inverse=True)
        nearest = np.full(inv.max() + 1, np.inf)
        np.minimum.at(nearest, inv, z[idx])
        idx = idx[z[idx] <= nearest[inv] * 1.01]
        w = facing[idx] ** 4
        total[idx] += w[:, None] * img[v[idx].astype(int), u[idx].astype(int)]
        weight[idx] += w
    seen = weight > 0
    colors = np.full((len(verts), 3), 160.0)
    colors[seen] = total[seen] / weight[seen, None]
    if seen.any() and not seen.all():
        _, near = cKDTree(verts[seen]).query(verts[~seen])
        colors[~seen] = colors[seen][near]
    rgb = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    mesh.visual.vertex_colors = np.c_[rgb, np.full(len(verts), 255, np.uint8)]


def floor_z(cloud: Cloud) -> float:
    """Къде наистина свършва предметът отдолу: масата, иначе най-ниската наблюдавана точка."""
    return cloud.ground_z if cloud.ground_z is not None else float(cloud.points[:, 2].min())


# Телефонът показва и изнася модела с JavaScript: повторно затворено сканиране имаше 1 милион триъгълника (STL от 50
# MB), много повече, отколкото отпечатък от няколко сантиметра може да покаже.
MAX_FACES = 250_000


def decimate(mesh: trimesh.Trimesh, faces: int, keep_closed: bool = True) -> trimesh.Trimesh:
    """По-малко триъгълници при запазени форма и цветове; мрежата остава непроменена, ако това би я отворило (освен
    при изключено keep_closed: само за преглед)."""
    ms = pymeshlab.MeshSet()
    colors = mesh.visual.vertex_colors[:, :4] / 255.0 if mesh.visual.kind == "vertex" else None
    kw = {"v_color_matrix": colors} if colors is not None else {}
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces, **kw))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=faces, preservenormal=True, preservetopology=True, planarquadric=True
    )
    m = ms.current_mesh()
    out_colors = (np.clip(m.vertex_color_matrix(), 0, 1) * 255).astype(np.uint8) if colors is not None else None
    out = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), vertex_colors=out_colors)
    if keep_closed and not out.is_watertight:
        return mesh
    trimesh.repair.fix_normals(out)
    return out


# Заглаждане със запазване на особеностите (двустъпковото на MeshLab: нормалите на стените се осредняват със съседи с
# огъване под SMOOTH_ANGLE, после точките се местят да им пасват). Сканиранията излизат с вълни на редове (ивиците в
# дълбочината на OpenMVS през корекцията на app/fuse.py) и отпечатъкът показва всяка вълна. Настроено при 100 мм върху
# тиква, фигура и дрелка: един проход мести повърхността с около 0,07 мм и пази ръбовете на очите на фигура; три махат
# вълните на тиква, но омекотяват очите. Всеки модел получава LIGHT; изтеглянето за печат "smooth" в приложението иска
# повече.
SMOOTH_ANGLE = 20.0
SMOOTH_NORMALS = 5
LIGHT = 1
MAX_SMOOTH = 3


def smooth(mesh: trimesh.Trimesh, passes: int) -> trimesh.Trimesh:
    """Същите върхове и стени (със запазени цветове), преместени; плоската основа (най-ниската равнина) остава на
    мястото си."""
    if passes <= 0:
        return mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.apply_coord_two_steps_smoothing(
        stepsmoothnum=passes, normalthr=SMOOTH_ANGLE, stepnormalnum=SMOOTH_NORMALS, stepfitnum=20
    )
    v = ms.current_mesh().vertex_matrix()
    if v.shape != mesh.vertices.shape:
        return mesh
    floor = mesh.bounds[0, 2]
    base = mesh.vertices[:, 2] <= floor + 1e-6 * mesh.extents.max()
    v[base] = mesh.vertices[base]
    v[:, 2] = np.maximum(v[:, 2], floor)  # точките до основата се дърпаха под нея (0,4 мм, силно заглаждане)
    out = mesh.copy()
    out.vertices = v
    return out


def make_printable(mesh: trimesh.Trimesh, floor: float, flat_base: bool, longest_mm: float = 100.0,
                   smoothing: int = LIGHT) -> trimesh.Trimesh:  # fmt: skip
    """smoothing: проходи на smooth(); 0 за проектиран модел (CAD от app/ai.py), чиито стени вече са плоски."""
    parts = mesh.split(only_watertight=False)
    if not len(parts):
        raise MeshError("От кадрите не се получи модел. Опитайте да заснемете отново, по-бавно и с повече светлина.")
    mesh = max(parts, key=lambda p: len(p.faces))  # маха носещите се фрагменти

    zmin, zmax = mesh.bounds[:, 2]
    h = zmax - zmin
    # Поасон измисля повърхност под пода (невидяна долна страна); отрязва се.
    cut = floor + (0.02 * h if flat_base else 0.0)
    # Където отдолу нищо не е видяно, повърхността стига до мрежата на Поасон и остава отворена отдолу: срязва се
    # точно над този отвор (най-ниския), а капакът затваря мрежата. Другите отвори (изрязан незаснет участък) се
    # затварят от close_holes, вместо предметът да се среже там.
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    open_edges = edges[counts == 1]
    if len(open_edges):
        loops = trimesh.graph.connected_components(open_edges, min_len=1)
        # Само отвор, стигащ до дъното: процеп отстрани на чаша на четвърт височина я отрязваше над него.
        bottom = [loop for loop in loops if mesh.vertices[loop, 2].min() < zmin + 0.05 * h]
        if bottom:
            top = float(min(mesh.vertices[loop, 2].max() for loop in bottom))
            if top < zmin + 0.35 * h:
                cut = max(cut, top + 0.005 * h)
    if cut > zmin:
        before = mesh
        mesh = mesh.slice_plane([0, 0, cut], [0, 0, 1], cap=True)
        if before.visual.kind == "vertex":  # slice_plane маха цветовете на върховете: на най-близкия стар връх
            _, idx = cKDTree(before.vertices).query(mesh.vertices)
            mesh.visual.vertex_colors = before.visual.vertex_colors[idx]

    if not mesh.is_watertight:
        mesh = close_holes(mesh)
    # Тънки ивици, оставени от срязването или запълването; истинските части са много над 1% от стените.
    parts = mesh.split(only_watertight=False)
    mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.01 * len(mesh.faces)])
    trimesh.repair.fix_normals(mesh)
    if len(mesh.faces) > MAX_FACES:
        mesh = decimate(mesh, MAX_FACES)
    mesh = smooth(mesh, smoothing)  # тук основата е най-ниската равнина (срязана плоско, завъртането по-долу е само около z)

    # Стъпка по осите (най-малкият правоъгълник около нея), така че отчетените размери и най-дългата страна, по която
    # мащабира приложението, да са на самия предмет: кутия, завъртяна на 29°, отчиташе 100 x 100 вместо 100 x 97.
    T2, _ = trimesh.bounds.oriented_bounds_2D(mesh.vertices[:, :2])
    turn = T2[:2, :2] if np.linalg.det(T2[:2, :2]) > 0 else T2[:2, :2] @ np.diag([1.0, -1.0])  # никога огледално
    T = np.eye(4)
    T[:2, :2] = turn
    mesh.apply_transform(T)
    # Основата е центрирана в началото при z = 0, най-дълга страна = longest_mm (STL няма единици; програмите за печат
    # приемат мм).
    lo, hi = mesh.bounds
    move = trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    mesh.apply_transform(move)
    scale = trimesh.transformations.scale_matrix(longest_mm / mesh.extents.max())
    mesh.apply_transform(scale)
    mesh.metadata["to_print"] = scale @ move @ T  # от координатите на входа (app/texture.py се връща с него)
    return mesh


def revolve(profile: np.ndarray, sections: int = 96) -> trimesh.Trimesh:
    """Затворен профил (r, z), завъртян около оста Z; точките върху оста стават единични върхове (без тънки ивици)."""
    pts = np.c_[np.maximum(profile[:, 0], 0), profile[:, 1]]
    pts = pts[np.r_[True, np.abs(np.diff(pts, axis=0)).max(1) > 1e-6]]  # повтарящи се точки
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    verts, rings = [], []
    for r, z in pts:
        if r < 1e-6:
            rings.append([len(verts)] * sections)
            verts.append([0.0, 0.0, z])
        else:
            rings.append(list(range(len(verts), len(verts) + sections)))
            verts += np.c_[r * np.cos(a), r * np.sin(a), np.full(sections, z)].tolist()
    faces = []
    for i in range(len(rings)):
        p, q = rings[i], rings[(i + 1) % len(rings)]
        for k in range(sections):
            k2 = (k + 1) % sections
            faces += [[p[k], q[k], q[k2]], [p[k], q[k2], p[k2]]]
    faces = np.array([f for f in faces if len(set(f)) == 3]).reshape(-1, 3)
    m = trimesh.Trimesh(np.array(verts), faces, process=True, validate=True)
    if m.volume < 0:
        m.invert()
    return m


def solid_volume(mesh: trimesh.Trimesh) -> float:
    """Вътрешният обем; за мрежа с отвори (където обемът със знак не значи нищо) запълнените ѝ воксели."""
    if mesh.is_watertight:
        return abs(float(mesh.volume))
    pitch = float(mesh.extents.max()) / 200
    return float(mesh.voxelized(pitch).fill().volume)


def geometry(mesh: trimesh.Trimesh) -> dict:
    """От какво работи оценката за печат в приложението (src/lib/printEstimate.ts) при размера на мрежата: пластмасата
    вътре и повърхността, която следват стените."""
    return {"volume_mm3": round(solid_volume(mesh), 1), "area_mm2": round(float(mesh.area), 1)}


def report(mesh: trimesh.Trimesh) -> dict:
    return {
        "dimensions_mm": [round(float(v), 2) for v in mesh.extents],
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
    } | geometry(mesh)


MEDIA_TYPES = {
    "stl": "model/stl",
    "obj": "model/obj",
    "glb": "model/gltf-binary",
    "3mf": "model/3mf",
}


Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])  # Z нагоре -> Y нагоре в glTF


def export(mesh: trimesh.Trimesh, fmt: str, longest_mm: float | None = None, textured: list[dict] | None = None,
           unlit: bool = False) -> bytes:  # fmt: skip
    """Моделът като `fmt`. textured: частите от app/texture.load за GLB с текстура от снимки (същата геометрия).
    unlit: GLB, чиито цветове се показват както са (вижте UNLIT), когато има цветове."""
    m = mesh.copy()
    factor = longest_mm / m.extents.max() if longest_mm else 1.0
    m.apply_scale(factor)
    if fmt == "glb":
        if textured:
            to = Y_UP @ trimesh.transformations.scale_matrix(factor)
            return glb_textured([p | {"vertices": trimesh.transform_points(p["vertices"], to),
                                      "normals": p["normals"] @ Y_UP[:3, :3].T} for p in textured], unlit)  # fmt: skip
        m.apply_transform(Y_UP)
        return glb(m, unlit)
    return m.export(file_type=fmt)


# Цветовете от снимките на сканиране вече съдържат светлината при заснемане. Осветени отново в AR (оценката на ARCore
# за светлината в стаята, с отраженията), сканиранията изглеждаха намазани с олио и почерняваха при тъмна оценка.
# Неосветен, AR показва цветовете от снимките както са, каквато и да е светлината в стаята; моделът пак хвърля сянка.
UNLIT = {"KHR_materials_unlit": {}}


def glb(mesh: trimesh.Trimesh, unlit: bool = False, extra: dict[str, np.ndarray] | None = None) -> bytes:
    """Двоичен glTF за преглед: гладки нормали, цветовете на върховете и матов материал. Собственият GLB на trimesh
    нямаше материал, затова програмите ползваха стандартния изцяло метален на glTF и без какво да отразява three.js
    (прегледът на телефона) показваше модела почти черен. Цветовете бяха sRGB стойностите от снимките, докато glTF
    иска линейни, което ги показва избелели. extra: още числа за връх като атрибути с име "_NAME" (собствените в glTF;
    three.js ги чете като "_name"), например напреженията от теста за здравина (app/stress.py)."""
    blobs, views, accessors = [], [], []

    def add(data: np.ndarray, kind: str, component: int, target: int, minmax: bool = False) -> int:
        raw = data.tobytes()
        views.append({"buffer": 0, "byteOffset": sum(len(b) for b in blobs), "byteLength": len(raw), "target": target})
        blobs.append(raw + b"\0" * (-len(raw) % 4))
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": len(data), "type": kind}
        if minmax:
            acc |= {"min": data.min(0).tolist(), "max": data.max(0).tolist()}
        accessors.append(acc)
        return len(accessors) - 1

    float_, uint32, vertices, indices = 5126, 5125, 34962, 34963
    attributes = {
        "POSITION": add(mesh.vertices.astype(np.float32), "VEC3", float_, vertices, minmax=True),
        "NORMAL": add(mesh.vertex_normals.astype(np.float32), "VEC3", float_, vertices),
    }
    colored = mesh.visual.kind == "vertex"
    if colored:
        c = mesh.visual.vertex_colors[:, :3] / 255.0
        linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        attributes["COLOR_0"] = add(linear.astype(np.float32), "VEC3", float_, vertices)
    for name, values in (extra or {}).items():
        attributes[name] = add(np.asarray(values, np.float32), "SCALAR", float_, vertices)
    faces = add(mesh.faces.astype(np.uint32).reshape(-1, 1), "SCALAR", uint32, indices)
    base = [1.0, 1.0, 1.0, 1.0] if colored else [0.8, 0.8, 0.8, 1.0]
    material = {"pbrMetallicRoughness": {"baseColorFactor": base, "metallicFactor": 0.0, "roughnessFactor": 0.85}}
    unlit = unlit and colored  # неосветен сив модел би бил плосък силует
    doc = {
        "asset": {"version": "2.0", "generator": "orbitize"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": faces, "material": 0, "mode": 4}]}],
        "materials": [material | {"extensions": UNLIT} if unlit else material],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
    }
    if unlit:
        doc["extensionsUsed"] = list(UNLIT)
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binary), b"BIN\0") + binary)  # fmt: skip


def glb_textured(parts: list[dict], unlit: bool = False) -> bytes:
    """Двоичен glTF с текстура от снимки (app/texture.py): за всяко изображение на текстура по един примитив с
    позиции, нормали, координати на текстурата и матов материал, чийто основен цвят е изображението (JPEG,
    непроменен). Всяко изображение съобщава и размера си (extras), нужен на телефона, за да го предаде на
    видеокартата без декодер на браузър."""
    blobs, views, accessors, prims, images, textures, materials = [], [], [], [], [], [], []

    def view(raw: bytes, target: int | None = None) -> int:
        v = {"buffer": 0, "byteOffset": sum(len(b) for b in blobs), "byteLength": len(raw)}
        if target:
            v["target"] = target
        views.append(v)
        blobs.append(raw + b"\0" * (-len(raw) % 4))
        return len(views) - 1

    def add(data: np.ndarray, kind: str, component: int, target: int, minmax: bool = False) -> int:
        acc = {"bufferView": view(data.tobytes(), target), "componentType": component, "count": len(data), "type": kind}
        if minmax:
            acc |= {"min": data.min(0).tolist(), "max": data.max(0).tolist()}
        accessors.append(acc)
        return len(accessors) - 1

    float_, uint32, vertices, indices = 5126, 5125, 34962, 34963
    for k, p in enumerate(parts):
        uv = np.asarray(p["uv"], np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]  # координатите на текстурата в glTF започват от горната страна на изображението
        attributes = {
            "POSITION": add(np.asarray(p["vertices"], np.float32), "VEC3", float_, vertices, minmax=True),
            "NORMAL": add(np.asarray(p["normals"], np.float32), "VEC3", float_, vertices),
            "TEXCOORD_0": add(uv, "VEC2", float_, vertices),
        }
        faces = add(np.asarray(p["faces"], np.uint32).reshape(-1, 1), "SCALAR", uint32, indices)
        w, h = p["size"]
        images.append({"bufferView": view(p["jpeg"]), "mimeType": "image/jpeg", "extras": {"width": w, "height": h}})
        textures.append({"source": k, "sampler": 0})
        materials.append({"pbrMetallicRoughness": {"baseColorTexture": {"index": k}, "metallicFactor": 0.0,
                                                   "roughnessFactor": 0.85}}
                         | ({"extensions": UNLIT} if unlit else {}))  # fmt: skip
        prims.append({"attributes": attributes, "indices": faces, "material": k, "mode": 4})
    doc = {
        "asset": {"version": "2.0", "generator": "orbitize"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],  # линеен, с мипмапи, ограничен
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
    }
    if unlit:
        doc["extensionsUsed"] = list(UNLIT)
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binary), b"BIN\0") + binary)  # fmt: skip
