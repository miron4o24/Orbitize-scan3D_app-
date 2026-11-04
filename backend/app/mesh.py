"""Dense point cloud + object masks -> clean, watertight, print-ready mesh.

Pure numpy/trimesh/pymeshlab code with no COLMAP/OpenMVS dependency, so it is unit-testable.
Coordinates after `clean_cloud`: +Z is up, the object stands on z = ground_z (when a table was found).

Two sources of shape, combined in `poisson`:
- the dense points that fall inside the object's outline in most frames (detail, concave parts);
- the visual hull: the volume inside the object's outline in (almost) every frame. It needs no texture, so it
  fills in dark, plain or shiny parts where dense matching finds few points.
"""

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
OBJECT_AGREE = 0.7  # a dense point is on the object if it is inside the outline in >= 70% of the frames showing it
HULL_AGREE = 0.75  # a voxel is inside the hull if it is inside the outline in >= 75% of them (masks are imperfect)
MIN_VIEWS = 5


class MeshError(Exception):
    """User-facing reconstruction problem."""


@dataclass
class Cloud:
    points: np.ndarray  # (n, 3)
    normals: np.ndarray  # (n, 3)
    colors: np.ndarray  # (n, 4) float 0..1
    ground_z: float | None = None
    to_up: np.ndarray | None = None  # rotation world -> these coordinates
    weights: np.ndarray | None = None  # per point, for Poisson (default 1)


@dataclass
class Views:
    """Pinhole cameras, COLMAP convention: pixel = K (R X + t), x right, y down, z forward."""

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
        return self.R[:, 2]  # camera +Z (forward) in world

    @property
    def ups(self) -> np.ndarray:
        return -self.R[:, 1]  # camera -Y (image up) in world


def silhouette_consistency(
    views: Views, masks: list[np.ndarray], scale: int, cells: int = 64, agree: float = HULL_AGREE
) -> np.ndarray:
    """Per frame: overlap (IoU) of its object mask with the visual hull of all frames, projected back into it.

    With correct camera poses the hull fills every mask; with wrong ones the masks' cones barely intersect and the
    hull shrinks to a sliver (a camera solution that put 53 frames "on target" gave a wedge). NaN: no object."""
    iou = np.full(len(views), np.nan)
    grid = hull_grid(views, masks, scale, cells, agree)
    if grid is None:
        return iou
    occ, step = grid
    for i, (R, t, K, m) in enumerate(zip(views.R, views.t, views.K, masks)):
        iou[i] = outline_iou(occ, step, R, t, K, m, scale)
    return iou


def hull_grid(views: Views, masks: list[np.ndarray], scale: int, cells: int = 64, agree: float = HULL_AGREE):
    """Voxel centres inside the visual hull of all frames, on a cells³ grid around where the frames' viewing rays
    through their mask centres meet, and the voxel size. None without enough frames that show the object."""
    n = len(views)
    rays, sizes = [], []
    for i, (R, t, (fx, fy, cx, cy), m) in enumerate(zip(views.R, views.t, views.K, masks)):
        ys, xs = np.nonzero(m > 127)
        if len(xs) < 10:
            continue
        u, v = (xs.mean() + 0.5) * scale, (ys.mean() + 0.5) * scale
        d = R.T @ [(u - cx) / fx, (v - cy) / fy, 1.0]
        rays.append((i, -R.T @ t, d / np.linalg.norm(d)))
        sizes.append(0.5 * max(np.ptp(xs), np.ptp(ys)) * scale / fx)  # tan of the object's angular half size
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
    """Overlap (IoU) of a frame's object mask with hull voxels `occ` projected into it; NaN if it shows no object."""
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
        # Voxel centres leave gaps between them: close them by the voxels' size on screen.
        px = step * fx / np.median(Xc[:, 2]) / scale
        if px > 1:
            hull = ndimage.binary_closing(hull, iterations=int(np.ceil(px)))
    return float((hull & fg).sum() / max(1, (hull | fg).sum()))


def subset(views: Views, keep: np.ndarray) -> Views:
    return Views(views.R[keep], views.t[keep], views.K[keep], [n for n, k in zip(views.names, keep) if k])


def consistent_frames(views: Views, masks: list[np.ndarray], scale: int, min_iou: float = 0.5) -> np.ndarray:
    """Frames whose camera pose agrees with the others: their mask matches the hull of the remaining frames.

    A few badly placed frames shrink the hull and make good frames look bad too, so the worst are dropped a quarter
    at a time and the hull is rebuilt. Returns a mask over the views."""
    keep = np.array([(m > 127).sum() >= 10 for m in masks])
    for _ in range(8):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            return np.zeros(len(views), bool)
        iou = silhouette_consistency(subset(views, keep), [masks[i] for i in idx], scale)
        bad = np.flatnonzero(~(iou >= min_iou))  # NaN counts as bad
        if len(bad) == 0:
            return keep
        worst = bad[np.argsort(np.nan_to_num(iou[bad], nan=-1))][: max(1, len(idx) // 4)]
        keep[idx[worst]] = False
    # Not settled: only the frames that agree with the last hull count (a wrong solution must not score by the
    # frames left over after dropping: a model where every frame matched its outline by 1-2% once "kept" 43).
    idx = np.flatnonzero(keep)
    iou = silhouette_consistency(subset(views, keep), [masks[i] for i in idx], scale)
    keep[idx[~(iou >= min_iou)]] = False
    return keep


def silhouette_votes(X: np.ndarray, views: Views, masks: list[np.ndarray], scale: int):
    """Per point: in how many frames it is in the picture, and in how many of those inside the object's outline.

    masks[i] is views[i]'s mask (uint8, object > 127) at 1/scale of the frame size."""
    n_in = np.zeros(len(X), np.int32)
    n_fg = np.zeros(len(X), np.int32)
    for s in range(0, len(X), 1_000_000):  # bounded memory for big clouds and grids
        Xs = X[s : s + 1_000_000]

        def look(frame):
            R, t, (fx, fy, cx, cy), m = frame
            Xc = PAR.turned(Xs, R) + t
            z = np.where(Xc[:, 2] > 1e-9, Xc[:, 2], np.nan)
            u = np.floor((fx * Xc[:, 0] / z + cx) / scale)
            v = np.floor((fy * Xc[:, 1] / z + cy) / scale)
            ok = (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])  # NaN (behind the camera) is False
            return ok, m[v[ok].astype(np.int64), u[ok].astype(np.int64)] > 127

        for ok, inside in PAR.each(look, zip(views.R, views.t, views.K, masks)):
            n_in[s : s + len(Xs)] += ok
            n_fg[s : s + len(Xs)][ok] += inside
    return n_in, n_fg


def rotation_to(a: np.ndarray, b: np.ndarray = UP) -> np.ndarray:
    """Rotation matrix turning unit vector a onto unit vector b (Rodrigues)."""
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
    """RANSAC for a near-horizontal plane n.p + d = 0 holding >= 10% of points with < 5% below it.

    The "nothing below" rule keeps a flat step or platform inside the object from being taken for the table.
    Returns (n, d) or None."""
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
    if np.linalg.norm(mean_up) > 0.8:  # photos held upright (phone app): image "up" is gravity's opposite
        return mean_up
    # Rolled cameras: use the orbit instead. Cameras circle the object, so their best-fit plane is horizontal,
    # and people photograph objects from slightly above, so the cameras look down on average.
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
) -> Cloud:
    """The object's points, levelled on the table: everything outside the object's outline (table, background,
    floaters) is dropped by the masks, which works whatever the object, background or camera path.
    up: which way is up, when known (the phone's gravity sensor); else it is guessed from the cameras."""
    n_in, n_fg = silhouette_votes(points, views, masks, mask_scale)
    on_object = (n_in >= min(MIN_VIEWS, len(views))) & (n_fg >= OBJECT_AGREE * n_in)
    if on_object.sum() < 300:
        raise MeshError(  # only on_object.sum() < 300 points on the object
            "Предметът не можа да се отдели от фона. Дръжте целия предмет в кръга на екрана и го сложете на подложка "
            "в различен цвят от него."
        )

    R = rotation_to(up if up is not None else estimate_up(views.centers, views.dirs, views.ups))
    pts, cams = points @ R.T, views.centers @ R.T
    obj = pts[on_object]
    lo, hi = np.percentile(obj, 2, axis=0), np.percentile(obj, 98, axis=0)
    size = float((hi - lo).max())

    # The table: a level plane among the other points around and just below the object. Levelling on it makes the
    # flat base truly flat.
    ground_z = None
    thresh = 0.015 * size
    mid = (lo + hi) / 2
    around = ~on_object & (np.linalg.norm(pts[:, :2] - mid[:2], axis=1) < 1.5 * size)
    around &= np.abs(pts[:, 2] - lo[2]) < 0.5 * size
    plane = find_ground(pts[around], thresh) if around.sum() >= 100 else None
    if plane is not None:
        n, d = plane
        R2 = rotation_to(n)
        R, pts, cams = R2 @ R, pts @ R2.T, cams @ R2.T
        ground_z = -d  # n.p + d = 0  ->  z' = -d after rotating n onto +Z

    keep = on_object & (pts[:, 2] > ground_z + thresh if ground_z is not None else True)
    pts, colors = pts[keep], colors[keep]
    nrm = normals[keep] @ R.T if normals is not None else estimate_normals(pts)
    inl = statistical_inliers(pts)
    pts, colors, nrm = pts[inl], colors[inl], nrm[inl]
    # Normals without a side (estimated, or from a cloud that has none) face the nearest camera. OpenMVS's already face
    # the frame that measured them (96-99.6% their nearest camera on real scans, against half for sideless ones) and are
    # kept: turned to the nearest camera, a mug's inner wall faced the camera behind it, outside the mug (4% of its
    # points, 14% of a temple's, between its columns).
    _, idx = cKDTree(cams).query(pts)
    flip = np.einsum("ij,ij->i", nrm, cams[idx] - pts) < 0
    if normals is None or flip.mean() > 0.2:
        nrm[flip] *= -1
    return Cloud(pts, nrm, colors, ground_z, R)


def _hull_fraction(views: Views, masks, scale: int, to_up: np.ndarray, axes: list[np.ndarray]) -> np.ndarray:
    """For each voxel centre (grid in the up frame): share of the frames showing it that have it inside the outline."""
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) @ to_up  # back to world
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
    """Closed mesh of the volume inside the object's outline in (almost) all frames, standing on the table.

    agree: the share of frames that must have a voxel inside. Below 1 it forgives bad masks and poses, but widens
    whatever only a few frames see narrow: an oval's short side, filmed all around, came out 17% wide at 0.75.
    limit: for world points inside the outlines, the most their voxel's share may be, from other evidence (AI depth:
    below `agree` where it sees nothing). Smooth, so the surface it cuts is too."""
    found = hull_volume(views, masks, scale, cloud, cells, agree)
    if found is None:
        return None
    frac, axes, step = found
    if limit is not None:
        inside, points = voxels_inside(frac, axes, agree)
        frac = limited(frac, inside, limit(points @ cloud.to_up))  # back to world
    return hull_mesh(frac, axes, step, agree)


def hull_volume(
    views: Views, masks, scale: int, cloud: Cloud, cells: int = 160, agree: float = HULL_AGREE
) -> tuple[np.ndarray, list[np.ndarray], float] | None:
    """The visual hull as a grid in the cloud's coordinates: per voxel, the share of the frames showing it that have
    it inside the outline. Returns (shares, the grid's axes, cell size), None if no voxel reaches `agree`."""
    lo, hi = np.percentile(cloud.points, 1, axis=0), np.percentile(cloud.points, 99, axis=0)
    floor = floor_z(cloud)
    # Coarse pass over a generous box (parts without dense points may stick out), then a fine one around the hull.
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
        # Next pass: the occupied box plus two coarse cells; nothing below the table.
        lo = np.array([axes[k][inside[:, k].min()] for k in range(3)]) - 2 * step
        hi = np.array([axes[k][inside[:, k].max()] for k in range(3)]) + 2 * step
        lo[2] = max(lo[2], floor)
    return frac, axes, step


def voxels_inside(frac: np.ndarray, axes: list[np.ndarray], agree: float = HULL_AGREE):
    """(indices, centres in the grid's coordinates) of the voxels inside the hull."""
    inside = np.argwhere(frac >= agree)
    return inside, np.stack([axes[k][inside[:, k]] for k in range(3)], -1)


def limited(frac: np.ndarray, inside: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """The shares, at most `limit` at the voxels `inside` (see visual_hull)."""
    out = frac.copy()
    i, j, k = inside.T
    out[i, j, k] = np.minimum(out[i, j, k], limit)
    return out


def free_space_limit(views: Views, points: np.ndarray, normals: np.ndarray, probes: np.ndarray, margin: float,
                     agree: float = HULL_AGREE, cell: int = 4) -> np.ndarray:  # fmt: skip
    """For world points inside the hull (probes): the most their hull share may be (see visual_hull), from the
    measured surface points (world, normals facing the side they were seen from).

    A frame that measured a surface point saw through everything between it and the point: a cup's inside, seen
    from above, and a handle's hole are empty, though no outline shows it (the hull fills them; a mug came out solid).
    Per frame, the points facing it make a depth map (the nearest per `cell` pixels); a probe in front of that depth
    is seen through by this frame, one at it is on the surface, and one behind it is hidden from it and gets no vote
    (so the frames that cannot see into a cup do not keep it full). The votes are averaged, each clipped to two
    margins, as in depth.carve_limit; probes no frame sees keep 1."""
    seen = np.zeros(len(probes), np.int32)
    total = np.zeros(len(probes))
    for R, t, (fx, fy, cx, cy) in zip(views.R, views.t, views.K):
        Xc = points @ R.T + t
        z = Xc[:, 2]
        ok = (z > 1e-9) & (np.einsum("ij,ij->i", normals, -R.T @ t - points) > 0)
        w, h = int(2 * cx) // cell + 1, int(2 * cy) // cell + 1  # undistorted frames: principal point in the middle
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
        far = depth[pv, pu] - pz[idx]  # how far beyond the probe this frame measured the surface
        vote = np.isfinite(far) & (far > -margin)
        seen[idx[vote]] += 1
        total[idx[vote]] += np.clip(far[vote], -2 * margin, 2 * margin)
    mean = total / np.maximum(seen, 1)
    share = agree + (margin - mean) * (1 - agree) / (0.2 * margin)
    return np.where(seen >= 2, np.clip(share, 0, 1), 1.0)


def hull_mesh(frac: np.ndarray, axes: list[np.ndarray], step: float, agree: float = HULL_AGREE):
    """The closed surface where the shares cross `agree` (largest part, normals outward), None if nothing does."""
    # Zero border so the surface closes at the box edges; the bottom closes on the table.
    vol = np.pad(ndimage.gaussian_filter(frac, 1.0), 1)
    if vol.max() <= agree:
        return None
    verts, faces, _, _ = measure.marching_cubes(vol, agree)
    origin = np.array([a[0] for a in axes]) - step
    hull = trimesh.Trimesh(origin + step * verts, faces)
    hull = max(hull.split(only_watertight=False), key=lambda p: len(p.faces))
    trimesh.repair.fix_normals(hull)  # outward
    # A copy: after turning the faces outward, trimesh (5.1) keeps the old, inward face normals in its cache, and
    # hull_samples took them; Poisson then caved a box's faces in and left it a third too flat.
    return hull.copy()


def estimate_normals(points: np.ndarray) -> np.ndarray:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points))
    ms.compute_normal_for_point_clouds(k=12)
    return ms.current_mesh().vertex_normal_matrix().copy()


def hull_samples(hull: trimesh.Trimesh, pts: np.ndarray, nrm: np.ndarray, n: int = 60_000):
    """Points on the hull's surface with outward normals, and two masks over them:
    near: a seen point is close by (the hull agrees with the data there);
    over: the hull overshoots there. It does where no frame looked along a surface: a flat top filmed only from
    above at an angle becomes a cone (from 35° up, as high as 0.7x the top's radius). Where a seen surface facing
    the same side lies under a hull sample, the sample is wrong."""
    samples, face = trimesh.sample.sample_surface_even(hull, n, seed=0)
    normals = hull.face_normals[face]
    size = hull.extents.max()
    tree = cKDTree(pts)
    # Searches bounded to the distances that count: unbounded, the probes deep inside the object took 70 s on 72
    # thousand points and minutes on a server scan's half million (twice per scan).
    dist, _ = tree.query(samples, distance_upper_bound=0.03 * size, workers=-1)
    near = np.isfinite(dist)
    over = np.zeros(len(samples), bool)
    for below in np.linspace(0.03, 0.7, 23) * size:
        # Probe 5% wide: points on plain surfaces are sparse.
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
    """Watertight surface through the object's points. Where they leave gaps (plain, dark or unseen parts) the
    visual hull's surface fills in, with less weight, so real points win where both exist.

    Poisson must enclose a volume: on the points alone, an object with gaps or an open underside turns into a blob
    that fills the whole bounding box. Depth 8 (cells 1/256 of the size, about twice the dense points' spacing): at 9
    the cells matched the spacing and the surface followed the points' noise (a drill: twice as bumpy)."""
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
        ms.apply_coord_taubin_smoothing(stepsmoothnum=smooth)  # dense points on dark or plain surfaces are noisy
    if ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces, preservenormal=True, preservetopology=True
        )
    m = ms.current_mesh()
    verts, faces = m.vertex_matrix(), m.face_matrix()
    # Trim surface no input point supports: across a stretch nobody filmed, Poisson runs off as a sheet to the edge
    # of its grid. make_printable closes the hole left behind. Not near the bottom: there the closed surface Poisson
    # invents under the unseen underside is what the floor cut turns into a flat base.
    tree = cKDTree(pts)
    dist, _ = tree.query(verts)
    sample = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20_000), replace=False)]
    spacing = float(np.median(tree.query(sample, k=2)[0][:, 1]))  # typical gap between input points
    reach = max(0.06 * float(np.ptp(pts, axis=0).max()), 4 * spacing)  # run-off sheets reach well beyond 10%
    low, high = np.percentile(pts[:, 2], [2, 98])
    supported = (dist <= reach) | (verts[:, 2] < low + 0.15 * (high - low))
    faces = faces[supported[faces].all(axis=1)]
    # Colour from the nearest object point (hull-only areas take the colour of the closest seen surface).
    _, idx = cKDTree(cloud.points).query(verts)
    colors = (np.clip(cloud.colors[idx], 0, 1) * 255).astype(np.uint8)
    out = trimesh.Trimesh(verts, faces, vertex_colors=colors)
    out.remove_unreferenced_vertices()
    return out


def clip_to_outline(
    mesh: trimesh.Trimesh, views: Views, masks: list[np.ndarray], scale: int, cloud: Cloud, agree: float = 0.6
) -> trimesh.Trimesh:
    """Removes surface outside the object's outline: Poisson can run off as sheets where no frame saw the object
    (e.g. a stretch of the circle that was not filmed). The real surface is never outside the visual hull.

    Each face is tested a little inside itself (1% of the size along its inward normal), so faces lying exactly on
    the hull, as the sides of a cylinder do, stay. make_printable closes the holes left behind."""
    if mesh.is_empty:
        return mesh
    probe = mesh.triangles_center - mesh.face_normals * 0.01 * mesh.extents.max()
    n_in, n_fg = silhouette_votes(probe @ cloud.to_up, views, masks, scale)
    inside = (n_in < min(MIN_VIEWS, len(views))) | (n_fg >= agree * n_in)  # unseen: no evidence to cut
    out = mesh.copy()
    out.update_faces(inside)
    out.remove_unreferenced_vertices()
    return out


def _meshlab_close(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """MeshLab's manifold repair and hole filler (without its self-intersection check: with it, one scan's holes ran
    for over 10 minutes), keeping colours. Parts under 1% of the faces (slivers) are dropped."""
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
    """Cap every remaining hole (trimmed unseen areas, cuts), keeping colours. MeshLab's filler can leave a few edges
    shared by three or more faces where duplicate points meet (MeshLab does not see them; a speaker had 8): the faces
    around them go and the small holes left are filled again. If the surface is still open, it is rebuilt closed from
    itself (`reclose`, which loses fine detail; both box scans went through it before this repair)."""
    out = _meshlab_close(mesh)
    for _ in range(3):
        if out.is_watertight:
            return out
        edges, counts = np.unique(out.edges_sorted, axis=0, return_counts=True)
        bad = np.unique(edges[counts > 2])
        if not len(bad):
            trimesh.repair.fill_holes(out)
            break
        # Faces within two rings of the bad edges' points go; the holes left are plain loops.
        ring = np.isin(out.faces, bad).any(axis=1)
        ring = np.isin(out.faces, np.unique(out.faces[ring])).any(axis=1)
        out.update_faces(~ring)
        out.remove_unreferenced_vertices()
        out = _meshlab_close(out)
    return out if out.is_watertight else reclose(mesh)


def reclose(mesh: trimesh.Trimesh, depth: int = 9) -> trimesh.Trimesh:
    """A closed surface through an open one: Poisson on points sampled evenly over it. Poisson always encloses a
    volume, so this ends any hole, however tangled; detail stays within a grid cell (1/512 of the size). 400 thousand
    samples are about 0.3 mm apart on a 10 cm box, close to that cell (200 thousand: 0.5 mm, rounding more off)."""
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


PAINT_REDUCE = 2  # frames are read at half size for colours: about the spacing of a 250 thousand triangle model's points
PAINT_CELL = 3  # (reduced) pixels: per square this wide, only the points nearest the camera are seen


def paint(mesh: trimesh.Trimesh, views: Views, photos: list, to_up: np.ndarray) -> None:
    """Vertex colours from the photos (the mesh in coordinates turned by to_up from the world's, as a Cloud's).

    Each vertex blends the frames that see it, weighted towards those facing it squarely, so frame-to-frame changes of
    exposure blend instead of leaving seams. Seen: facing the camera and among the points nearest to it where it
    lands (within 1% of the distance), so a part hidden behind another takes nothing from that frame. Vertices no
    frame sees (the flat base) take the colour of the nearest seen one."""
    verts, normals = mesh.vertices @ to_up, mesh.vertex_normals @ to_up  # back to world
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
    """Where the object really ends at the bottom: the table, else the lowest observed point."""
    return cloud.ground_z if cloud.ground_z is not None else float(cloud.points[:, 2].min())


# The phone shows and exports the model in JavaScript: a reclosed scan had 1 million triangles (a 50 MB STL), far
# more than a print of a few centimetres can show.
MAX_FACES = 250_000


def decimate(mesh: trimesh.Trimesh, faces: int, keep_closed: bool = True) -> trimesh.Trimesh:
    """Fewer triangles, keeping the shape and colours; the mesh unchanged if that would open it (unless keep_closed is
    off: only for viewing)."""
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


# Feature-preserving smoothing (MeshLab's two-step: face normals averaged with neighbours bending less than
# SMOOTH_ANGLE, then the points moved to fit them). Scans come out rippled in rows (OpenMVS's depth streaks through
# app/fuse.py's correction) and a print shows every ripple. Tuned at 100 mm on a pumpkin, a figure and a drill: one
# pass moves the surface ~0.07 mm and keeps a figure's eye rims; three take a pumpkin's ripples off but soften the
# eyes. Every model gets LIGHT; the app's "smooth" print download asks for more.
SMOOTH_ANGLE = 20.0
SMOOTH_NORMALS = 5
LIGHT = 1
MAX_SMOOTH = 3


def smooth(mesh: trimesh.Trimesh, passes: int) -> trimesh.Trimesh:
    """Same vertices and faces (colours kept), moved; the flat base (the lowest plane) stays where it was."""
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
    v[:, 2] = np.maximum(v[:, 2], floor)  # points beside the base were pulled below it (0.4 mm, strong smoothing)
    out = mesh.copy()
    out.vertices = v
    return out


def make_printable(mesh: trimesh.Trimesh, floor: float, flat_base: bool, longest_mm: float = 100.0,
                   smoothing: int = LIGHT) -> trimesh.Trimesh:  # fmt: skip
    """smoothing: passes of smooth(); 0 for a designed model (app/ai.py's CAD), whose faces are flat already."""
    parts = mesh.split(only_watertight=False)
    if not len(parts):
        raise MeshError("От кадрите не се получи модел. Опитайте да заснемете отново, по-бавно и с повече светлина.")
    mesh = max(parts, key=lambda p: len(p.faces))  # drops floating fragments

    zmin, zmax = mesh.bounds[:, 2]
    h = zmax - zmin
    # Poisson invents surface below the floor (unseen underside); cut it off.
    cut = floor + (0.02 * h if flat_base else 0.0)
    # Where nothing was seen underneath, the surface runs into Poisson's grid and stays open at the bottom: cut just
    # above that opening (the lowest one), the cap closes the mesh. Other openings (a trimmed stretch nobody filmed)
    # are closed by close_holes instead of cutting the object there.
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    open_edges = edges[counts == 1]
    if len(open_edges):
        loops = trimesh.graph.connected_components(open_edges, min_len=1)
        # Only an opening that reaches the bottom: a gap in a mug's side, a quarter up, cut the mug off above it.
        bottom = [loop for loop in loops if mesh.vertices[loop, 2].min() < zmin + 0.05 * h]
        if bottom:
            top = float(min(mesh.vertices[loop, 2].max() for loop in bottom))
            if top < zmin + 0.35 * h:
                cut = max(cut, top + 0.005 * h)
    if cut > zmin:
        before = mesh
        mesh = mesh.slice_plane([0, 0, cut], [0, 0, 1], cap=True)
        if before.visual.kind == "vertex":  # slice_plane drops vertex colours: the nearest old vertex's
            _, idx = cKDTree(before.vertices).query(mesh.vertices)
            mesh.visual.vertex_colors = before.visual.vertex_colors[idx]

    if not mesh.is_watertight:
        mesh = close_holes(mesh)
    # Slivers left by the cut or hole filling; real parts are far bigger than 1% of the faces.
    parts = mesh.split(only_watertight=False)
    mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.01 * len(mesh.faces)])
    trimesh.repair.fix_normals(mesh)
    if len(mesh.faces) > MAX_FACES:
        mesh = decimate(mesh, MAX_FACES)
    mesh = smooth(mesh, smoothing)  # the base is the lowest plane here (cut flat, the turn below is only around z)

    # Footprint along the axes (the smallest rectangle around it), so the sizes reported and the longest side the app
    # scales by are the object's own: a box standing turned by 29° read 100 x 100 instead of 100 x 97.
    T2, _ = trimesh.bounds.oriented_bounds_2D(mesh.vertices[:, :2])
    turn = T2[:2, :2] if np.linalg.det(T2[:2, :2]) > 0 else T2[:2, :2] @ np.diag([1.0, -1.0])  # never a mirror
    T = np.eye(4)
    T[:2, :2] = turn
    mesh.apply_transform(T)
    # Base centred on the origin at z = 0, longest side = longest_mm (STL has no units; slicers assume mm).
    lo, hi = mesh.bounds
    move = trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    mesh.apply_transform(move)
    scale = trimesh.transformations.scale_matrix(longest_mm / mesh.extents.max())
    mesh.apply_transform(scale)
    mesh.metadata["to_print"] = scale @ move @ T  # from the input's coordinates (app/texture.py goes back with it)
    return mesh


def revolve(profile: np.ndarray, sections: int = 96) -> trimesh.Trimesh:
    """A closed (r, z) profile turned round the Z axis; points on the axis become single vertices (no slivers)."""
    pts = np.c_[np.maximum(profile[:, 0], 0), profile[:, 1]]
    pts = pts[np.r_[True, np.abs(np.diff(pts, axis=0)).max(1) > 1e-6]]  # repeated points
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
    """The volume inside; for a mesh with openings (where the signed volume means nothing), its filled voxels."""
    if mesh.is_watertight:
        return abs(float(mesh.volume))
    pitch = float(mesh.extents.max()) / 200
    return float(mesh.voxelized(pitch).fill().volume)


def geometry(mesh: trimesh.Trimesh) -> dict:
    """What the app's print estimate (src/lib/printEstimate.ts) works from, at the mesh's size: the plastic inside and
    the surface the walls follow."""
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


Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])  # Z-up -> glTF's Y-up


def export(mesh: trimesh.Trimesh, fmt: str, longest_mm: float | None = None, textured: list[dict] | None = None,
           unlit: bool = False) -> bytes:  # fmt: skip
    """The model as `fmt`. textured: app/texture.load's parts, for a GLB with the photo texture (same geometry).
    unlit: a GLB whose colours show as they are (see UNLIT), when it has colours."""
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


# A scan's photo colours already hold the light it was filmed in. Lit again in AR (ARCore's estimate of the room's
# light, reflections included), scans looked oiled, and went black when the estimate was dark. Unlit, the AR shows the
# photo colours as they are, whatever the room's light; the model still casts its shadow.
UNLIT = {"KHR_materials_unlit": {}}


def glb(mesh: trimesh.Trimesh, unlit: bool = False, extra: dict[str, np.ndarray] | None = None) -> bytes:
    """Binary glTF for viewing: smooth normals, the vertex colours and a matte material. trimesh's own GLB had no
    material, so viewers used glTF's default, fully metallic one, and with nothing to reflect, three.js (the phone's
    viewer) showed the model nearly black. Its colours were the photos' sRGB values where glTF wants linear ones,
    which shows them washed out. extra: more per-vertex numbers, as attributes named "_NAME" (glTF's custom ones;
    three.js reads them as "_name"), e.g. the stress test's stresses (app/stress.py)."""
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
    unlit = unlit and colored  # a grey model unlit would be a flat silhouette
    doc = {
        "asset": {"version": "2.0", "generator": "scan3d"},
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
    """Binary glTF with the photo texture (app/texture.py): per texture image one primitive with positions, normals,
    texture coordinates and a matte material whose base colour is the image (JPEG, kept as it is). Each image also
    says its size (extras), which the phone needs to hand it to the GPU without a browser's image decoder."""
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
        uv[:, 1] = 1 - uv[:, 1]  # glTF's texture coordinates start at the image's top
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
        "asset": {"version": "2.0", "generator": "scan3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],  # linear, mipmapped, clamp
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
