"""Pro's AI model from the whole video (app/ai.py, method "single"): the object's outline in every frame cut out by
AI, the depth measured between the frames and the AI's depth fused within it, the photos as its texture.

The camera solver places the frames (COLMAP); the object finder (app/segment.py) and MobileSAM prompted from 3D
(app/refine.py) cut the object out of each; the outline shape is what is inside the object in (nearly) every frame
(visual hull), on a fine grid. Outlines alone give a smooth lump (a figure's base a mound, no arms, no folds), so the
depth measured between the frames (OpenMVS) and AI depth (app/depth.py) are fused within it (app/fuse.py), as an
express scan's: a base's rim, arms, feet, folds, waves in the hair. The photo texture (app/texture.py) then shows
every painted detail as filmed. TripoSR's guess from one frame (app/single3d.py) was tried as the source of the
hollows, fitted among the cameras: on a figure it invented a second face and a hollow base, so it is only the fallback
when the frames cannot be placed.

Coordinates: world (the cameras', COLMAP) and the cloud's (world turned by to_up, z up, as app/mesh.py's Cloud)."""

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

from . import mesh as M
from . import parallel as PAR
from . import pipeline as P
from . import segment as S

# Frames are undistorted, matched and depth-fused at this size, with the fusion's cells this wide (of the size), as a
# full scan's. Against an express scan's 1280 and 0.4%, on the same frames (server, 2026-11-04): a drill's label and
# edges and a figure's face and lettering came out clearly sharper, in ~14 minutes instead of ~10.
SIDE = 1920
VOXEL = 0.0025
CELLS = 192  # the outline shape's grid: finer than a scan's 160 (when the fusion fails, it is the model's surface)
# A voxel is inside when this share of the frames showing it have it inside the object: with the outlines sharpened
# by MobileSAM, all but one in 20 (a figure, 18 frames: the shape's outlines overlap the frames' 97%, against 87% at
# a scan's forgiving 0.75, which left its thin body fat); with the object finder's masks only, a scan's 0.75.
AGREE_SHARP = 0.95
# The share used is the one of these (up to the one asked for) whose shape is truest to the outlines (overlap): a thin
# part (a drill's handle, 47 frames from the global solver) misses the outline in a few frames and goes at 0.95,
# leaving the battery as a box (job 6509: true to the outlines 76%), while a figure's thin body stays fat at 0.75.
AGREE_LEVELS = (0.95, 0.9, 0.85, 0.8, 0.75)
# The measuring sees only inside the masks, so the table is found among the solver's points, which see the room. It is
# not the table when more than this share of the object lies below it, or when the object ends this far (of its size)
# above it: a floor far below would let the outline shape run down to it.
TABLE_BELOW = 0.05
TABLE_GAP = 0.08
# Dense matching works on what the object can be in a frame, this much (of its size there) around it: its outline and
# where the outlines of the others place it, holes filled (the gap between a handle and a battery is measured: it says
# the space is empty). The fusion looks no further than 2% of the size outside the shape (fuse.OUTSIDE), and the table
# comes from the solver's points. A scan's area (segment.dense_area, nearly the whole frame) took 5 of a drill's 14
# minutes. The time goes with the pixels.
BAND = 0.06
BAND_CELLS = 96  # the outline shape's grid for it: coarse is enough


def project(X: np.ndarray, R, t, K, scale: float):
    """Pixel coordinates (u, v) of world points at 1/scale of the frame's size, and whether in front."""
    fx, fy, cx, cy = K
    Xc = PAR.turned(X, np.asarray(R)) + t
    front = Xc[:, 2] > 1e-9
    z = np.where(front, Xc[:, 2], 1.0)
    return (fx * Xc[:, 0] / z + cx) / scale, (fy * Xc[:, 1] / z + cy) / scale, front


def shown(X: np.ndarray, step: float, R, t, K, shape: tuple, scale: int) -> np.ndarray:
    """The outline of voxel centres X (world; voxels `step` wide) in a frame, at its masks' size (1/scale)."""
    u, v, front = project(X, R, t, K, scale)
    img = np.zeros(shape, bool)
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    ok = front & (ui >= 0) & (vi >= 0) & (ui < shape[1]) & (vi < shape[0])
    img[vi[ok], ui[ok]] = True
    depth = float(np.median((PAR.turned(X, np.asarray(R)) + t)[:, 2]))
    gap = int(np.ceil(step * K[0] / depth / scale)) if depth > 0 else 0  # voxel centres a pixel or more apart
    if gap:
        img = ndimage.binary_closing(img, iterations=gap, border_value=0)
    return ndimage.binary_fill_holes(img)


def overlap(occ: np.ndarray, axes, step: float, to_up: np.ndarray, views: M.Views, masks: list, scale: int) -> float:
    """Mean overlap (IoU) of the inside voxels' outlines with the frames' masks: how true the shape is to them."""
    X = np.stack([axes[k][idx] for k, idx in enumerate(np.nonzero(occ))], -1) @ to_up  # cloud -> world
    if len(X) > 300_000:
        X = X[np.random.default_rng(0).choice(len(X), 300_000, replace=False)]

    def fit(frame):
        R, t, K, m = frame
        a, b = shown(X, step, R, t, K, m.shape, scale), m > 127
        return (a & b).sum() / max(1, (a | b).sum())

    return float(np.mean(PAR.each(fit, zip(views.R, views.t, views.K, masks))))


def surface(occ: np.ndarray, axes, step: float) -> trimesh.Trimesh | None:
    """The closed surface around the inside voxels (the cloud's coordinates), smoothed by a voxel: from the voxels'
    own inside/outside, not their shares, so that a thin part (an arm, a neck) keeps its full width."""
    return M.hull_mesh(occ.astype(np.float32), axes, step, 0.5)


def on_table(job: Path, cloud: M.Cloud, ground_z: float) -> M.Cloud:
    """The measured cloud standing on the table the solver's points found (same z: the cloud was levelled on it), or
    as it is when that plane cannot be what the object stands on."""
    z = cloud.points[:, 2]
    lo, hi = np.percentile(cloud.points, 2, axis=0), np.percentile(cloud.points, 98, axis=0)
    size = float((hi - lo).max())
    below = float((z < ground_z).mean())
    gap = (float(lo[2]) - ground_z) / size
    if below > TABLE_BELOW or gap > TABLE_GAP:
        P.note(job, f"AI model: the solver's plane is not the table ({below:.0%} of the object below it, object "
               f"{gap:.1%} of its size above it)")  # fmt: skip
        return cloud
    P.note(job, f"AI model: table from the solver's points ({below:.1%} of the object below it, object {gap:.1%} of "
           "its size above it)")  # fmt: skip
    keep = z > ground_z + 0.015 * size  # as mesh.clean_cloud
    return replace(cloud, points=cloud.points[keep], normals=cloud.normals[keep], colors=cloud.colors[keep],
                   ground_z=ground_z)  # fmt: skip


def dense_areas(views: M.Views, masks: list, cloud: M.Cloud) -> list[np.ndarray] | None:
    """Per frame, at its mask's size, where dense matching should work (see BAND); None without an outline shape."""
    found = M.hull_volume(views, masks, S.MASK_SCALE, cloud, cells=BAND_CELLS)
    if found is None:
        return None
    frac, axes, step = found
    X = np.stack([axes[k][idx] for k, idx in enumerate(np.nonzero(frac >= P.LOOSE_AGREE))], -1) @ cloud.to_up

    def area(frame):
        R, t, K, m = frame
        a = shown(X, step, R, t, K, m.shape, S.MASK_SCALE) | (m > 127)
        if not a.any():
            return np.ones(m.shape, bool)
        ys, xs = np.nonzero(a)
        width = max(np.ptp(ys), np.ptp(xs)) + 1
        return ndimage.binary_fill_holes(ndimage.binary_dilation(a, iterations=max(2, int(np.ceil(BAND * width)))))

    return PAR.each(area, zip(views.R, views.t, views.K, masks))


def measured(job: Path, work: Path, dense: Path, views: M.Views, masks: list, cloud: M.Cloud, up: np.ndarray | None,
             agree: float, set_state: Callable[[str], None], sh: Callable[[str, list], None], side: int = SIDE,
             voxel: float = VOXEL, fixed: bool = False, table: bool = True, flap: float | None = None,
             wide: bool = False, geometric: bool = True, passes: int | None = None):  # fmt: skip
    """The model's surface (the cloud's coordinates) and the cloud it is in: the depth measured between the frames
    (OpenMVS, within the masks) and the AI's depth fused within the outline shape; the outline shape itself when the
    measuring or the fusion fails (dense: the undistorted frames, COLMAP's model in dense/sparse). side: the frames'
    size; voxel: the fusion's grid; fixed: the outline shape at exactly `agree`, none chosen; table: use the solver's
    table (`cloud`'s, found among the solver's points); flap: the fusion's fuse.FLAP; wide: dense matching on a scan's
    area instead of around the object; geometric: its second, checking pass; passes: the fusion's fuse.PASSES."""
    solvers = cloud.ground_z if table else None
    mask_dir = work / "dense_masks"
    mask_dir.mkdir(exist_ok=True)
    areas = None if wide else dense_areas(views, masks, cloud)
    if areas is not None:
        P.note(job, f"AI model: dense matching on {np.mean([a.mean() for a in areas]):.0%} of the frames' pixels")
    for k, (name, m) in enumerate(zip(views.names, masks)):
        with Image.open(dense / "images" / name) as im:
            size = im.size
        if areas is None:
            area = Image.fromarray(S.dense_area(m, size))
        else:
            area = Image.fromarray(areas[k].astype(np.uint8) * 255).resize(size, Image.NEAREST)
        area.save(mask_dir / f"{Path(name).stem}.mask.png")
    dense_cloud = False
    try:
        sh("dense setup", ["InterfaceCOLMAP", "-w", dense, "-i", dense, "-o", dense / "scene.mvs",
            "--image-folder", dense / "images"])  # fmt: skip
        neighbors = dense / "neighbors.txt"
        lines = P.neighbor_lines(views, P.mvs_indices(dense / "sparse" / "images.bin"))
        neighbors.write_text("\n".join(lines) + "\n")
        # As an express scan's (app/pipeline.py run), with the sharpened masks.
        sh("dense matching", ["DensifyPointCloud", "-w", dense, "-i", dense / "scene.mvs", "-o", dense / "dense.mvs",
            "--view-neighbors-file", neighbors, "--mask-path", mask_dir, "--ignore-mask-label", "0",
            "--resolution-level", "0", "--max-resolution", str(side), "--number-views", "4",
            "--geometric-iters", "1" if geometric else "0", "--estimate-roi", "0", "--crop-to-roi", "0", "--remove-dmaps", "0",
            "--max-threads", P.THREADS])  # fmt: skip
        pts, nrm, col = P.load_cloud(dense / "dense.ply")
        upward = cloud.to_up[2] if solvers is not None else up  # z as the solver's table has it
        cloud = M.clean_cloud(pts, nrm, col, views, masks, S.MASK_SCALE, upward)
        if cloud.ground_z is None and solvers is not None:
            cloud = on_table(job, cloud, solvers)
        dense_cloud = True
        P.note(job, f"AI model: {len(pts)} measured points, {len(cloud.points)} on the object, table found: "
               f"{cloud.ground_z is not None}")  # fmt: skip
    except (P.Failed, M.MeshError, OSError, ValueError) as e:  # then the outlines' shape
        P.note(job, f"AI model: no depth measured ({e}); the outlines' shape")
    set_state("meshing")
    found = M.hull_volume(views, masks, S.MASK_SCALE, cloud, cells=CELLS)
    if found is None:
        raise P.Failed("the frames' outlines give no shape")
    frac, axes, step = found
    tried = {}
    for level in [agree] if fixed else [a for a in AGREE_LEVELS if a <= agree] or [agree]:
        occ = frac >= level
        if occ.sum() >= 1000:
            tried[level] = overlap(occ, axes, step, cloud.to_up, views, masks, S.MASK_SCALE)
    if not tried:
        raise P.Failed("the frames' outlines give no shape")
    best = max(tried, key=lambda a: (round(tried[a], 3), a))  # the truest; of equals, the stricter
    agree = best
    inside = frac >= agree
    outline = surface(inside, axes, step)
    if outline is None:
        raise P.Failed("the frames' outlines give no shape")
    P.note(job, f"AI model: outline shape from {len(views)} frames, agreeing {agree:.0%}: true to them "
           f"{tried[agree]:.1%} (" + ", ".join(f"{a:.0%}: {o:.1%}" for a, o in tried.items()) + ")")  # fmt: skip
    fused = None
    if dense_cloud:
        P.carved_hull(work, dense, views, masks, cloud, sh, found)  # the AI depth (app/depth.py) for the fusion
        fused = P.fused_surface(work, dense, views, cloud, outline, sh, voxel, flap, passes or P.F.PASSES)
        for p in dense.glob("*.dmap"):
            p.unlink(missing_ok=True)
    if fused is None:
        P.note(job, "AI model: the outlines' shape (no fused depth)")
        return outline, cloud
    return M.clip_to_outline(fused, views, masks, S.MASK_SCALE, cloud), cloud
