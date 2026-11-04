"""The object's surface from the frames' depth maps fused in a volume, instead of Poisson on the dense points.

OpenMVS's depth maps are noisy (striped, a few tenths of a percent of the distance) and empty on dark or plain parts
(a drill's black handle). Its dense points inherit both: Poisson on them gave a crust of bumps, and the outline shape
(lumpy) in the gaps. A frame's AI depth (app/depth.py) is smooth and complete but off in scale and gently bent. Fitted
to the frame's own depth map (scale and offset in inverse depth, then a smooth correction field from what is left) it
keeps the measured shape with the network's clean surface, and fills the gaps. Every frame then votes where the
surface is, as a truncated signed distance averaged in a voxel grid (a TSDF), and the visual hull fills in where no
frame looked (the underside). On a drill: flat panels, step edges and a round handle instead of lumps.

A frame's mask counts only where the hull agrees (the masks' consensus), so background one mask took in stays out;
and every measured depth outside the object (table, wall) still says the space in front of it is empty, which cuts
away what the outlines could not (table under the object's edge, the gap between a handle and a battery).

Runs as its own process (pipeline step "fusing depth"):
    python fuse.py <dense dir> <inputs.npz> <crops.npz> <out.ply>
dense dir: OpenMVS's depth maps (depth*.dmap). crops.npz: app/depth.py's depth crops. inputs.npz: names (n,) frame
file names (crop i is frame i), to_up (3,3) world -> mesh coordinates, ground_z (nan: no table), hull_v, hull_f (the
visual hull from the outlines, uncarved, mesh coordinates). Writes out.ply (mesh coordinates) and out.json (frames used of how many).
"""

import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from scipy import ndimage
from skimage import measure

if __package__:
    from . import depth as D
    from . import parallel as PAR
else:  # run as a script: import the package by its path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import depth as D
    from app import parallel as PAR

# Of the object's size: the grid's cell. 0.25% (400 cells along it) keeps a 10 cm print's detail to about 0.5 mm after
# the blur below; finer only added noise on a drill.
VOXEL = 0.0025
MAX_VOXELS = 60_000_000  # about 1.5 GB at the peak; bigger grids get coarser cells
TRUNC = 6  # cells: the signed distance is cut off this far from the surface (depth noise is a few cells)
MIN_CONF = 0.5  # OpenMVS's confidence below which a measured depth does not count (its median on a drill: 0.8)
FIELD = 0.01  # of the crop's long side: how smooth the correction of the AI depth is (a drill: 10 pixels)
AI_WEIGHT = 0.5  # a vote where only the AI saw the surface (measured pixels: 1)
FREE_WEIGHT = 0.3  # a vote for empty space in front of the surface: carves, but one wrong frame cannot punch a hole
AI_FREE = 0.3  # and of that, where only the AI saw the surface (a drill's handle got torn without it)
HULL_WEIGHT = 0.3  # the visual hull's say, a fraction of one frame's
OUTSIDE = 0.02  # of the size: beyond this outside the hull nothing is fused (the table and the room)
FLOOR_BAND = 0.06  # of the size: how high above the table a sheet of it can reach (a soft cushion bulges)
FLAP = 0.0125  # of the size: sheets there thinner than twice this are table, not object (see floor_sheet)
HULL_POINTS = 1_000_000  # points inside the hull drawn into each frame to check its mask (see hull_view)
BLUR = 1.5  # cells: the averaged volume smoothed before the surface is taken (without it: crumbs and flaps)
FACES = 200_000
# A frame's say on a surface falls with how obliquely it sees it (the cosine between its ray and the surface's normal,
# never below FACING_MIN): at a grazing angle a pixel spans a long strip of surface and its depth error lies along
# the surface. 0: off.
FACING_MIN = 0.15
# A second pass leaves out a frame's surface votes that disagree with the first pass's average by more than this (of
# the truncation distance): one frame whose AI depth was fitted wrong (dark plastic, few measured pixels) otherwise
# leaves a streak or lump. 0: one pass. On a drill (80 frames), with FACING_MIN: 19% of the surface votes left out,
# the black handle's streaks mostly gone, mean bend between neighbouring faces 3.4° -> 3.1°, the dense points as close
# to the surface as before (median 0.23% of the size). Smoothing the AI depth's correction more (FIELD 0.03-0.05)
# also calmed the handle but bent the body out of shape (points 0.36-0.47% off): not done.
OUTLIER = 0.3
PASSES = 2  # voting passes, each leaving out what disagrees with the one before (1: none left out)
BLOCK = 500_000  # cells a thread works on at a time (see vote_block)


def note(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


@dataclass
class DepthMap:
    image: str  # the frame's file name
    image_size: tuple[int, int]  # (w, h)
    K: np.ndarray  # for the depth map's resolution
    R: np.ndarray  # world -> camera
    C: np.ndarray  # camera centre
    depth: np.ndarray  # (h, w), 0 = none
    conf: np.ndarray | None


HEADER = "<HBBIIIIff"  # name 'DR', flags, padding, image w, h, depth w, h, depth range
HAS_NORMAL, HAS_CONF, HAS_VIEWS = 2, 4, 8


def read_dmap(path: Path) -> DepthMap:
    """OpenMVS's raw depth map (DepthMap.cpp, ExportDepthDataRaw)."""
    b = path.read_bytes()
    magic, flags, _, iw, ih, dw, dh, _, _ = struct.unpack_from(HEADER, b, 0)
    if magic != struct.unpack("<H", b"DR")[0]:
        raise ValueError(f"{path.name}: not a depth map")
    o = struct.calcsize(HEADER)
    (n,) = struct.unpack_from("<H", b, o)
    image = Path(b[o + 2 : o + 2 + n].decode("utf-8", "replace")).name
    o += 2 + n
    (nids,) = struct.unpack_from("<I", b, o)
    o += 4 + 4 * nids
    K, R, C = (np.frombuffer(b, np.float64, k, o + off) for k, off in ((9, 0), (9, 72), (3, 144)))
    o += 168
    area = dw * dh
    depth = np.frombuffer(b, np.float32, area, o).reshape(dh, dw)
    o += 4 * area
    if flags & HAS_NORMAL:
        o += 12 * area
    conf = np.frombuffer(b, np.float32, area, o).reshape(dh, dw) if flags & HAS_CONF else None
    o += 4 * area if flags & HAS_CONF else 0
    o += 4 * area if flags & HAS_VIEWS else 0
    if o != len(b):
        raise ValueError(f"{path.name}: unexpected layout")
    return DepthMap(image, (iw, ih), K.reshape(3, 3), R.reshape(3, 3), C, depth, conf)


def at_image_size(dm: DepthMap) -> DepthMap:
    """The depth map at its frame's resolution (the crops' pixels), if OpenMVS worked at a smaller one."""
    h, w = dm.depth.shape
    iw, ih = dm.image_size
    if (w, h) == (iw, ih):
        return dm
    rows = np.minimum((np.arange(ih) * h / ih).astype(int), h - 1)
    cols = np.minimum((np.arange(iw) * w / iw).astype(int), w - 1)
    K = dm.K.copy()
    K[0] *= iw / w
    K[1] *= ih / h
    conf = dm.conf[np.ix_(rows, cols)] if dm.conf is not None else None
    return DepthMap(dm.image, dm.image_size, K, dm.R, dm.C, dm.depth[np.ix_(rows, cols)], conf)


def completed(dm: DepthMap, disp: np.ndarray, box, inner: np.ndarray, allowed: np.ndarray, core: np.ndarray):
    """The frame's depth over its crop: the AI depth fitted to the measured one, on the object's pixels: its mask's
    inside and the hull's (`core`: where the frame's mask missed a dark part, a drill's black trigger), but only where
    the hull shows (`allowed`). Returns (depth, weight, free), 0 where nothing, or None if the two do not fit
    together. free: the measured depth of the other pixels, which only shows empty space (in front of a pixel of the
    object that is left out, it is empty all the same)."""
    x0, y0, x1, y1 = box
    dep = dm.depth[y0:y1, x0:x1]
    if dep.shape != disp.shape:
        return None
    conf = dm.conf[y0:y1, x0:x1] if dm.conf is not None else np.ones_like(dep)
    measured = (dep > 0) & (conf >= MIN_CONF)
    inner = (inner | core) & allowed
    on = inner & measured
    if on.sum() < 500:
        return None
    inv = np.where(dep > 0, 1 / np.maximum(dep, 1e-12), 0)
    (a, b), err = D.fit_scale(disp[on], inv[on], np.ones(int(on.sum())))
    if a <= 0 or err > D.MAX_ERROR:
        return None
    base = a * disp + b
    # What the scale and offset leave, smoothed into a field (the network bends surfaces gently); outliers left out.
    r = np.where(on, inv - base, 0)
    med = np.median(r[on])
    mad = 1.4826 * np.median(np.abs(r[on] - med))
    good = on & (np.abs(r - med) <= 4 * mad) if mad > 0 else on
    sigma = max(3.0, FIELD * max(disp.shape))
    num = ndimage.gaussian_filter(np.where(good, r, 0).astype(np.float32), sigma)
    den = ndimage.gaussian_filter(good.astype(np.float32), sigma)
    full = base + np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    ok = inner & (full > 0)
    depth = np.where(ok, 1 / np.maximum(full, 1e-12), 0).astype(np.float32)
    weight = np.where(on, 1.0, AI_WEIGHT).astype(np.float32) * ok
    if FACING_MIN > 0:
        weight *= np.clip(facing(depth, dm.K, box), FACING_MIN, 1.0)
    free = np.where(measured & ~ok, dep, 0).astype(np.float32)
    return depth, weight, free


def facing(depth: np.ndarray, K: np.ndarray, box) -> np.ndarray:
    """Per pixel of a crop's depth: the cosine between the camera's ray and the surface's normal (1: straight on).
    Pixels without depth, or next to one, get 1 (they cast no surface vote anyway)."""
    x0, y0, x1, y1 = box
    z = depth.astype(np.float32)
    u = (np.arange(x0, x1, dtype=np.float32) - np.float32(K[0, 2])) / np.float32(K[0, 0])
    v = (np.arange(y0, y1, dtype=np.float32) - np.float32(K[1, 2])) / np.float32(K[1, 1])
    P = np.stack([u[None, :] * z, v[:, None] * z, z], -1)
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    ray = np.stack([np.broadcast_to(u[None, :], z.shape), np.broadcast_to(v[:, None], z.shape), np.ones_like(z)], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.abs((n * ray).sum(-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(ray, axis=-1))
    holes = ndimage.minimum_filter(z, size=3) <= 0
    return np.where(holes | ~np.isfinite(cos), 1.0, cos).astype(np.float32)


def hull_view(points: np.ndarray, spacing: float, dm: DepthMap, box) -> tuple[np.ndarray, np.ndarray]:
    """Where the visual hull shows in the crop (its inside's points, world, each drawn as wide as their spacing), and
    its core away from the outline (as far in as app/depth.py keeps the masks' inside: depth blurs at the outline).
    The hull is what most frames' masks agree on, so a piece of background one frame's mask took in (a chair behind a
    drill's handle, a strip of the table, a poster) is outside it, and a part one mask missed is inside."""
    x0, y0, x1, y1 = box
    out = np.zeros((y1 - y0, x1 - x0), bool)
    x = PAR.turned(points - dm.C, dm.R)
    z = x[:, 2]
    ok = z > 1e-9
    if not ok.any():
        return out, out
    u = np.round(dm.K[0, 0] * x[ok, 0] / z[ok] + dm.K[0, 2]).astype(np.int64) - x0
    v = np.round(dm.K[1, 1] * x[ok, 1] / z[ok] + dm.K[1, 2]).astype(np.int64) - y0
    inb = (u >= 0) & (v >= 0) & (u < x1 - x0) & (v < y1 - y0)
    out[v[inb], u[inb]] = True
    r = int(np.ceil(spacing * dm.K[0, 0] / np.median(z[ok]))) + 1
    shown = ndimage.maximum_filter(out, size=2 * r + 1)
    # The crop is the object's box grown by 20% on each side (app/depth.py predict).
    edge = max(2, int(0.02 * max(out.shape) / 1.4))
    return shown, ndimage.binary_erosion(shown, iterations=r + edge)


def hull_distance(hull: trimesh.Trimesh, lo: np.ndarray, shape, step: float) -> np.ndarray:
    """Signed distance (world units) of the grid's cells to the hull: negative inside."""
    vg = hull.voxelized(step).fill()
    inside = np.zeros(shape, bool)
    off = np.round((vg.transform[:3, 3] - lo) / step).astype(int)
    m = vg.matrix[: shape[0] - off[0], : shape[1] - off[1], : shape[2] - off[2]]
    inside[off[0] : off[0] + m.shape[0], off[1] : off[1] + m.shape[1], off[2] : off[2] + m.shape[2]] = m
    if not inside.any():
        raise SystemExit("the visual hull encloses nothing")
    away, within = PAR.each(lambda a: ndimage.distance_transform_edt(a).astype(np.float32), [~inside, inside])
    return (away - within) * step


def floor_sheet(solid: np.ndarray, top: int, radius: float) -> np.ndarray | None:
    """The cells of a thin sheet lying on the table and sticking out from the object (its first `top` layers), or
    None. A red box on a red cushion took in a patch of the cushion: as surface as the box, inside every mask, but
    thinner than anything the box has. Thin = gone after a morphological opening (a ball of `radius` cells); sticking
    out = more than `radius` cells from what stays (the opening also rounds the object's own edges, by less). Not when
    most of the object itself is that thin (a flat object)."""
    r = max(2.0, radius)
    sub = solid[:, :, : top + int(np.ceil(2 * r))]
    if top <= 0 or not sub.any():
        return None
    core = ndimage.distance_transform_edt(sub) > r
    if not core.any():
        return None
    opened = ndimage.distance_transform_edt(~core) <= r
    if opened.sum() < 0.5 * sub.sum():
        return None
    sheet = sub & ~opened & (ndimage.distance_transform_edt(~opened) > r + 1)
    sheet[:, :, top:] = False
    return sheet if sheet.any() else None


def vote_block(frames: list, world: np.ndarray, cells: np.ndarray, tsdf: np.ndarray, weight: np.ndarray,
               prev: np.ndarray | None, trunc: float) -> int:  # fmt: skip
    """Every frame's votes for one block of the volume's cells (world: their centres; cells: their places in tsdf and
    weight; no two blocks share a cell, so blocks go at once, and an average does not depend on the votes' order). With
    `prev` (the pass before), surface votes far from it are left out. Returns how many."""
    dropped = 0
    for (R, C, K), box, depth, wmap, fmap in frames:
        x0, y0, x1, y1 = box
        x = PAR.turned(world - C, R)
        zc = x[:, 2]
        ok = zc > 1e-9
        u = np.full(len(x), -1, np.int64)
        v = np.full(len(x), -1, np.int64)
        u[ok] = np.round(K[0, 0] * x[ok, 0] / zc[ok] + K[0, 2]).astype(np.int64) - x0
        v[ok] = np.round(K[1, 1] * x[ok, 1] / zc[ok] + K[1, 2]).astype(np.int64) - y0
        ok &= (u >= 0) & (v >= 0) & (u < x1 - x0) & (v < y1 - y0)
        idx, zc, u, v = np.flatnonzero(ok), zc[ok], u[ok], v[ok]
        surface = depth[v, u] > 0
        dep = np.where(surface, depth[v, u], fmap[v, u])
        sd = dep - zc
        # Object pixels vote near their surface and for empty space in front; other pixels only for the latter.
        ok = (dep > 0) & (sd >= np.where(surface, -trunc, trunc))
        idx, sd, surface, w = idx[ok], sd[ok], surface[ok], wmap[v[ok], u[ok]]
        val = np.minimum(1.0, sd / trunc).astype(np.float32)
        cell = cells[idx]
        if prev is not None:
            keep = ~surface | (val >= 1.0) | (np.abs(val - prev[cell]) <= OUTLIER)
            dropped += int((~keep).sum())
            cell, val, surface, w = cell[keep], val[keep], surface[keep], w[keep]
        free = val >= 1.0
        w = np.where(surface, w, 1.0)
        w = np.where(free, w * FREE_WEIGHT * np.where(w < 1.0, AI_FREE, 1.0), w).astype(np.float32)
        tsdf[cell] = (tsdf[cell] * weight[cell] + val * w) / (weight[cell] + w)
        weight[cell] += w
    return dropped


def fuse(dense: Path, inputs: Path, crops_file: Path, out: Path) -> None:
    z = np.load(inputs)
    names = [str(n) for n in z["names"]]
    to_up = z["to_up"].astype(np.float64)
    ground_z = float(z["ground_z"])
    hull = trimesh.Trimesh(z["hull_v"], z["hull_f"], process=False)
    size = float(hull.extents.max())
    step = (float(z["voxel"]) if "voxel" in z.files else VOXEL) * size  # express scans: coarser cells
    pad = max(OUTSIDE * size, TRUNC * step) + 3 * step
    lo = hull.bounds[0] - pad
    shape = np.ceil((hull.bounds[1] + pad - lo) / step).astype(int) + 1
    if np.prod(shape) > MAX_VOXELS:
        grow = (np.prod(shape) / MAX_VOXELS) ** (1 / 3)
        step *= grow
        shape = np.ceil(shape / grow).astype(int) + 1
    trunc = TRUNC * step
    note(f"grid {shape[0]}x{shape[1]}x{shape[2]}, cell {step / size:.2%} of the size")

    sdf = hull_distance(hull, lo, tuple(shape), step).ravel()
    tsdf = np.clip(sdf / trunc, -1, 1)
    weight = np.full(tsdf.shape, HULL_WEIGHT, np.float32)
    active = np.flatnonzero(sdf <= OUTSIDE * size)
    inside = np.flatnonzero(sdf < 0)
    del sdf
    pick = np.random.default_rng(0).choice(len(inside), min(len(inside), HULL_POINTS), replace=False)
    spacing = step * (len(inside) / len(pick)) ** (1 / 3)
    cells = np.stack(np.unravel_index(inside[pick], tuple(shape)), 1)
    hull_points = (cells * step + lo) @ to_up
    del inside, pick
    cells = np.stack(np.unravel_index(active, tuple(shape)), 1).astype(np.float32)
    world = (cells * np.float32(step) + lo.astype(np.float32)) @ to_up.astype(np.float32)
    del cells
    note(f"{len(active) / 1e6:.1f} million cells near the hull")

    maps = {}
    for p in dense.glob("depth*.dmap"):
        with open(p, "rb") as f:
            head = f.read(struct.calcsize(HEADER) + 2 + 1024)
        o = struct.calcsize(HEADER)
        (n,) = struct.unpack_from("<H", head, o)
        maps[Path(head[o + 2 : o + 2 + n].decode("utf-8", "replace")).name] = p
    with np.load(crops_file) as crops:
        todo = [(name, crops[f"disp_{i}"], tuple(int(v) for v in crops[f"box_{i}"]), crops[f"inner_{i}"])
                for i, name in enumerate(names) if f"disp_{i}" in crops and name in maps]  # fmt: skip

    def prepare(item):
        name, disp, box, inner = item
        dm = at_image_size(read_dmap(maps[name]))
        allowed, core = hull_view(hull_points, spacing, dm, box)
        got = completed(dm, disp, box, inner, allowed, core)
        if got is None:
            note(f"{name}: AI depth does not fit the depth map, left out")
            return None
        return ((dm.R.astype(np.float32), dm.C.astype(np.float32), dm.K), box, *got)

    # Per usable frame: (pose, box, depth, weight, free), kept for the later passes.
    frames = [f for f in PAR.each(prepare, todo) if f is not None]
    used = len(frames)
    note(f"{used} of {len(names)} frames have depth to fuse")
    if used < 3:
        raise SystemExit(f"only {used} frames' depth could be used")

    def vote(tsdf: np.ndarray, weight: np.ndarray, prev: np.ndarray | None) -> int:
        starts = range(0, len(active), BLOCK)
        return sum(PAR.each(lambda s: vote_block(frames, world[s : s + BLOCK], active[s : s + BLOCK], tsdf, weight, prev,
                                                 trunc), starts))  # fmt: skip

    passes = (int(z["passes"]) if "passes" in z.files else PASSES) if OUTLIER > 0 else 1
    empty_tsdf, empty_weight, prev = tsdf, weight, None
    for k in range(passes):
        tsdf, weight = empty_tsdf.copy(), empty_weight.copy()
        dropped = vote(tsdf, weight, prev)
        prev = tsdf
        note(f"pass {k + 1}: {dropped / 1e6:.1f} million votes left out as far from the others")
    del empty_tsdf, empty_weight, prev
    note(f"{used} of {len(names)} frames fused")

    vol = tsdf.reshape(tuple(shape))
    wv = np.minimum(weight, 50).reshape(tuple(shape))
    vol = ndimage.gaussian_filter(vol * wv, BLUR) / np.maximum(ndimage.gaussian_filter(wv, BLUR), 1e-6)
    if np.isfinite(ground_z):  # nothing under the table: a flat, closed bottom
        zs = lo[2] + np.arange(shape[2]) * step
        vol[:, :, zs < ground_z] = 1.0
        flap = float(z["flap"]) if "flap" in z.files else FLAP
        sheet = floor_sheet(vol < 0, int(np.searchsorted(zs, ground_z + FLOOR_BAND * size)), flap * size / step)
        if sheet is not None:
            vol[:, :, : sheet.shape[2]][sheet] = 1.0
            note(f"thin sheet on the table cut off: {sheet.sum() * step**3 / size**3:.2%} of the size cubed")
    vol[[0, -1]] = vol[:, [0, -1]] = vol[:, :, [0, -1]] = 1.0  # closed at the grid's border
    verts, faces, _, _ = measure.marching_cubes(vol, 0.0)
    mesh = trimesh.Trimesh(verts * step + lo, faces)
    mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
    if mesh.volume < 0:
        mesh.invert()
    note(f"surface: {len(mesh.faces)} triangles")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.apply_coord_taubin_smoothing(stepsmoothnum=5)  # the grid's stair steps
    if ms.current_mesh().face_number() > FACES:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=FACES, preservenormal=True, preservetopology=True)
    m = ms.current_mesh()
    trimesh.Trimesh(m.vertex_matrix(), m.face_matrix()).export(out)
    out.with_suffix(".json").write_text(json.dumps({"frames": used, "total": len(names)}))


if __name__ == "__main__":
    fuse(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
