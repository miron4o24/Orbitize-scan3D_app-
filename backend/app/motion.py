"""Camera poses from the phone's motion sensors: the route for objects photogrammetry cannot place.

Feature matching needs texture. On a plain cloth, frames that look down at the object see little else, and the
camera solver leaves them out or puts them in the wrong place (a speaker on red velvet: the high circle came out as
separate pieces, and the model was a plank). The phone knows how it is turned, though: the app records its
orientation while filming. The object stays in the middle of the picture as the user walks around it, so each frame's
camera sits on the ray through the object's centre in its mask, at a distance inversely proportional to how big the
object looks. An orientation error then only moves the viewpoint around the object a little, instead of shifting it
sideways. The outlines (AI masks) from all those viewpoints carve out the shape: plain, shiny or dark objects work as
well as textured ones. What the sensors do not say is fitted to the outlines: the delay between asking the camera to
record and its first frame, and per frame the exact distance, centre and compass heading. The focal length comes from
the camera solver when it placed some frames, else a typical phone value.

The frames are upright in the phone's portrait orientation (picture up = the phone's +y): the app is locked to
portrait and the camera tags its videos by the screen's orientation, not by how the phone is held. (Which way the
picture is turned cannot be told from the outlines: on a round object all four turns agreed within 4%.)

motion.json (uploaded by the app): {"samples": [[t, alpha, beta, gamma], ...]}, t in seconds since the recording was
requested, angles in radians, W3C DeviceOrientation: R = Rz(alpha) Rx(beta) Ry(gamma) turns the phone's axes (x
right, y up, z out of the screen) into the world's (x east, y north, z up).

Runs as its own process (pipeline step "shape from motion sensors"), so it can be cancelled like the other tools:
    python motion.py <job dir> <photo masks.npz> <focal px, 0 if unknown> <out.ply> <flat base 0/1>
It writes <out>.json {"score", "frames", "delay"} and, when the outlines agree (score >= USABLE), the printable mesh
as <out>.
"""

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

if __package__:
    from . import depth as D
    from . import mesh as M
    from . import segment as S
else:  # run as a script: import the package by its path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import depth as D
    from app import mesh as M
    from app import segment as S

MOTION = "motion.json"  # in the job directory
FRAME_TIMES = "frames.json"  # in the job directory: seconds into the video of each picked frame
SHIFTS = np.arange(-0.5, 2.01, 0.25)  # first video frame this long after the recording was requested, s
MIN_FRAMES = 12
# The frames must look at the object from at least this far around (degrees). With less, the outlines agree whatever
# the poses (filmed from one side, or the sensor stuck) and the shape is a guess.
MIN_AROUND = 200
YAW_STEP = np.radians(3)  # heading correction tried per frame and round
# Focal length / long image side when the camera solver did not estimate it (main phone cameras filming 16:9: ~0.75;
# COLMAP found 0.77 for one). Not fitted to the outlines: they agree better the longer it is (more distant, flatter
# views), whatever the truth.
FOCAL = 0.77
GOOD = 0.75  # median outline overlap from which the sensor poses are trusted
USABLE = 0.5  # ...and below which they are not used at all
CAM_FROM_DEVICE = np.diag([1.0, -1.0, -1.0])  # camera: x right, y down, z forward out of the back of the phone
# Share of the frames that must see a voxel inside the outline (mesh.visual_hull). The camera solver's route forgives
# more (0.75), which widened an oval filmed all around by 17%; at 0.9 by 3%.
AGREE = 0.9
DEFAULT_DELAY = 0.3  # s, when the outlines agree as well with any delay (round objects, steady walking)


def device_matrices(alpha, beta, gamma) -> np.ndarray:
    """(n, 3, 3) phone-to-world rotations from W3C angles (arrays)."""
    return Rotation.from_euler("ZXY", np.stack([alpha, beta, gamma], -1)).as_matrix()  # intrinsic Z, X', Y''


def device_rotations(samples: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Phone orientation at each time (s since the recording was requested), interpolated between samples."""
    order = np.argsort(samples[:, 0], kind="stable")
    s = samples[order]
    s = s[np.r_[True, np.diff(s[:, 0]) > 0]]
    rot = Rotation.from_matrix(device_matrices(s[:, 1], s[:, 2], s[:, 3]))
    return Slerp(s[:, 0], rot)(np.clip(times, s[0, 0], s[-1, 0])).as_matrix()


@dataclass
class Frames:
    """The frames that show the object, with where (px) and how big (px, square root of the area) it is."""

    names: list[str]
    times: np.ndarray
    masks: list[np.ndarray]
    anchors: np.ndarray  # (n, 2) mask centre in image pixels
    sizes: np.ndarray  # (n,)
    width: int
    height: int


def frames(names: list[str], times: dict, masks: dict, scale: int, size: tuple[int, int]) -> Frames:
    keep, anchors, sizes = [], [], []
    for n in names:
        fg = masks[n] > 127
        if n not in times or fg.sum() < 50:
            continue
        ys, xs = np.nonzero(fg)
        keep.append(n)
        anchors.append(((xs.mean() + 0.5) * scale, (ys.mean() + 0.5) * scale))
        sizes.append(np.sqrt(len(xs)) * scale)
    return Frames(keep, np.array([times[n] for n in keep]), [masks[n] for n in keep], np.array(anchors).reshape(-1, 2),
                  np.array(sizes), size[0], size[1])  # fmt: skip


def headings(dev: np.ndarray) -> np.ndarray:
    """Which way each frame looks, radians around the vertical, counted on across full turns (filming order)."""
    fwd = (dev @ CAM_FROM_DEVICE)[:, :, 2]
    return np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))


def steadied(fr: Frames, dev: np.ndarray) -> Frames:
    """The frames with the object's size as it would look without its shape: the mean (log) size over the frames
    filmed from within 90° around. Taken frame by frame, an oval's wide side looks closer than its narrow side, and
    the hull came out rounder than the object (1.5:1 as 1.2:1). What is left, people walking closer or farther, is
    fitted per frame (`refine`)."""
    around = headings(dev)
    near = np.abs(around[:, None] - around[None, :]) <= np.pi / 2
    return replace(fr, sizes=np.exp(near @ np.log(fr.sizes) / near.sum(1)))


def heading(yaw: np.ndarray) -> np.ndarray:
    """(n, 3, 3) turns about the vertical axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(yaw), np.ones_like(yaw)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def views(dev: np.ndarray, fr: Frames, focal: float, dist=None, offset=None, yaw=None) -> M.Views:
    """Cameras looking at the object's centre (the world origin) from its outline's centre in each frame.
    dist: per-frame factor on the distance; offset: per-frame shift of that centre, in object sizes; yaw: per-frame
    correction of the compass heading, radians."""
    n = len(fr.names)
    cx, cy = fr.width / 2, fr.height / 2
    if yaw is not None:
        dev = heading(yaw) @ dev
    C = dev @ CAM_FROM_DEVICE  # camera -> world
    a = fr.anchors + (offset if offset is not None else 0) * fr.sizes[:, None]
    ray = np.stack([(a[:, 0] - cx) / focal, (a[:, 1] - cy) / focal, np.ones(n)], -1)
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    d = (dist if dist is not None else 1.0) * focal / fr.sizes  # the object is 1 unit "big"
    centers = -d[:, None] * np.einsum("nij,nj->ni", C, ray)
    R = C.transpose(0, 2, 1)
    return M.Views(R, -np.einsum("nij,nj->ni", R, centers), np.tile([focal, focal, cx, cy], (n, 1)), list(fr.names))


def score(v: M.Views, masks: list, scale: int, cells: int = 40) -> float:
    """How well all outlines agree: median overlap of each mask with the hull of all frames."""
    iou = M.silhouette_consistency(v, masks, scale, cells, AGREE)
    return float(np.median(np.nan_to_num(iou))) if len(iou) else 0.0


def fit(fr: Frames, samples: np.ndarray, scale: int, focal: float):
    """The delay of the video after the recording was requested, by how well the outlines then agree. Walking at a
    steady pace, a wrong delay turns every viewpoint by the same angle around the object, which changes nothing; only
    changes of pace show it (and make it matter). Returns (phone rotations per frame, score, delay)."""

    def at(s: float) -> float:
        dev = device_rotations(samples, fr.times + s)
        return score(views(dev, steadied(fr, dev), focal), fr.masks, scale, 64)

    default = at(DEFAULT_DELAY)
    best, shift = max((at(s), float(s)) for s in SHIFTS)
    for s in shift + np.arange(-0.2, 0.201, 0.05):
        if (sc := at(s)) > best:
            best, shift = sc, float(s)
    if best < default + 0.002:  # no clear sign
        best, shift = default, DEFAULT_DELAY
    return device_rotations(samples, fr.times + shift), best, shift


def refine(fr: Frames, dev: np.ndarray, focal: float, scale: int, rounds: int = 2):
    """Per frame, the distance, outline centre and heading that best match the hull of all frames (people do not walk
    at a constant distance, a mask's centre is not exactly the object's, and the compass wobbles near metal).
    Returns the views."""
    n = len(fr.names)
    dist, offset, yaw = np.ones(n), np.zeros((n, 2)), np.zeros(n)
    steps = [(f, 0, 0, 0) for f in (0.9, 0.95, 1.05, 1.1)]
    steps += [(1, dx, dy, 0) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))] + [(1, 0, 0, -1), (1, 0, 0, 1)]
    for _ in range(rounds):
        v = views(dev, fr, focal, dist, offset, yaw)
        grid = M.hull_grid(v, fr.masks, scale, 64, AGREE)
        if grid is None:
            break
        occ, step = grid
        for i in range(n):
            def iou(d, o, w):
                one = views(dev[i : i + 1], _one(fr, i), focal, np.array([d]), o[None], np.array([w]))
                return M.outline_iou(occ, step, one.R[0], one.t[0], one.K[0], fr.masks[i], scale)

            best = iou(dist[i], offset[i], yaw[i])
            for f, dx, dy, dw in steps:
                cand = dist[i] * f, offset[i] + 0.03 * np.array([dx, dy]), yaw[i] + YAW_STEP * dw
                c = iou(*cand)
                if c > best:
                    best, (dist[i], offset[i], yaw[i]) = c, cand
    return views(dev, fr, focal, dist, offset, yaw)


def _one(fr: Frames, i: int) -> Frames:
    return Frames([fr.names[i]], fr.times[i : i + 1], [fr.masks[i]], fr.anchors[i : i + 1], fr.sizes[i : i + 1],
                  fr.width, fr.height)  # fmt: skip


def floor_height(v: M.Views, masks: list, scale: int, occ: np.ndarray, step: float) -> float | None:
    """The table's height: where the ray through the bottom of each outline first enters the hull. That point is on
    the object's lowest edge facing the camera, which rests on the table. (The hull itself reaches below the table
    as a wedge, as no frame sees under the object.)"""
    tree = cKDTree(occ)
    zs = []
    for R, t, (fx, fy, cx, cy), m in zip(v.R, v.t, v.K, masks):
        ys, xs = np.nonzero(m > 127)
        if len(ys) < 50:
            continue
        row = ys.max() - 1  # just inside the outline's bottom
        cols = xs[ys == row]
        if not len(cols):
            continue
        u, w = (cols.mean() + 0.5) * scale, (row + 0.5) * scale
        d = R.T @ [(u - cx) / fx, (w - cy) / fy, 1.0]
        d /= np.linalg.norm(d)
        c = -R.T @ t
        r = np.linalg.norm(c)
        pts = c + np.linspace(0.3 * r, 1.7 * r, 600)[:, None] * d
        hit = np.flatnonzero(tree.query(pts)[0] <= 0.75 * step)
        if len(hit):
            zs.append(pts[hit[0], 2])
    return float(np.median(zs)) if zs else None


def depth_carver(v: M.Views, masks: list, photos: list[Path], hull: trimesh.Trimesh):
    """What AI depth says is empty inside the outlines' hull (app/depth.py), as a limit(points) for M.visual_hull;
    None without the model or enough frames it fits. The depth is scaled to the hull's walls, which the outlines get
    right (the hull errs on tops and hollows)."""
    if not D.MODEL.exists():
        return None
    samples, face = trimesh.sample.sample_surface_even(hull, 30_000, seed=0)
    refs = samples[np.abs(hull.face_normals[face][:, 2]) < 0.5]
    fits = D.fit_frames(photos, masks, v.R, v.t, v.K, refs, np.ones(len(refs)))
    if not D.enough(fits, len(photos)):
        return None
    margin = D.CARVE_MARGIN * float(hull.extents.max())
    return lambda points: D.carve_limit(fits, points, margin, AGREE)


def model(fr: Frames, v: M.Views, scale: int, photos: list[Path] | None, flat_base: bool) -> trimesh.Trimesh:
    """The printable mesh: the hull of all outlines, standing on the table, carved by AI depth, coloured from the
    frames."""
    grid = M.hull_grid(v, fr.masks, scale, 96, AGREE)
    if grid is None:
        raise M.MeshError("The object's outline could not be followed around. Keep it inside the circle on screen.")
    occ, step = grid
    floor = floor_height(v, fr.masks, scale, occ, step)
    if floor is None:
        floor = float(occ[:, 2].min())
    cloud = M.Cloud(occ, np.zeros_like(occ), np.ones((len(occ), 4)), floor, np.eye(3))  # world z is up already
    hull = M.visual_hull(v, fr.masks, scale, cloud, cells=200, agree=AGREE)
    if hull is None:
        raise M.MeshError("The object's outlines did not agree on a shape. Film again more slowly.")
    limit = depth_carver(v, fr.masks, photos, hull) if photos else None
    if limit is not None:
        carved = M.visual_hull(v, fr.masks, scale, cloud, cells=200, agree=AGREE, limit=limit)
        share = 1 - carved.volume / hull.volume if carved is not None else 1.0
        use = share <= D.MAX_CARVE
        print(f"AI depth carves {share:.0%} of the outline shape: " + ("using it" if use else "not used"), flush=True)
        if use:
            hull = carved
    trimesh.smoothing.filter_taubin(hull, iterations=10)
    if photos:
        M.paint(hull, v, photos, np.eye(3))
    return M.make_printable(hull, floor, flat_base)


def load(job: Path) -> tuple[np.ndarray, dict] | None:
    """The phone's orientation samples and the frames' times, if the app sent them (videos only)."""
    try:
        samples = np.array(json.loads((job / MOTION).read_text())["samples"], float)
        times = json.loads((job / FRAME_TIMES).read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if samples.ndim != 2 or samples.shape[1] != 4 or len(samples) < 20 or not np.isfinite(samples).all():
        return None
    return samples, times


def gravity_up(job: Path, v: M.Views) -> tuple[np.ndarray, float] | None:
    """Which way is up in the camera solver's coordinates, from the phone's orientation log: each frame's own sense of
    gravity, turned by its solved rotation. (The camera solver does not know up; image up averages to it only over an
    even, full circle, and a box on a cushion, with no table found to level on, came out tilted 45°.) Returns (up,
    median angle between the frames' ups in degrees), or None without a log for these frames."""
    data = load(job)
    if data is None:
        return None
    samples, times = data
    known = [i for i, n in enumerate(v.names) if n in times]
    if len(known) < MIN_FRAMES:
        return None
    dev = device_rotations(samples, np.array([times[v.names[i]] for i in known]) + DEFAULT_DELAY)
    up_cam = (dev @ CAM_FROM_DEVICE)[:, 2, :]  # world +z in camera coordinates
    ups = np.einsum("nji,nj->ni", v.R[known], up_cam)  # ...in the solver's coordinates
    up = ups.sum(0)
    up /= np.linalg.norm(up)
    return up, float(np.degrees(np.median(np.arccos(np.clip(ups @ up, -1, 1)))))


def main(job: Path, masks_path: Path, focal: float, out: Path, flat_base: bool) -> dict:
    info = {"score": 0.0, "frames": 0, "around": 0, "delay": None}
    data = load(job)
    with np.load(masks_path) as z:
        masks = {n: z[n] for n in z.files}
    if data is not None and masks:
        samples, times = data
        names = sorted(masks)
        with Image.open(job / "photos" / names[0]) as im:
            size = im.size
        if not 0.5 <= focal / max(size) <= 1.3:  # unknown, or from a wrong camera solution
            focal = FOCAL * max(size)
        fr = frames(names, times, masks, S.MASK_SCALE, size)
        info["frames"] = len(fr.names)
        around = float(np.degrees(np.ptp(headings(device_rotations(samples, fr.times + DEFAULT_DELAY)))))
        info["around"] = round(around)
        print(f"{len(fr.names)} frames show the object, filmed from {around:.0f}° around it", flush=True)
        if len(fr.names) >= MIN_FRAMES and around >= MIN_AROUND:
            dev, sc, shift = fit(fr, samples, S.MASK_SCALE, focal)
            print(f"focal {focal:.0f} px: outlines agree {sc:.0%} with the video starting {shift:.2f} s after the "
                  "request", flush=True)  # fmt: skip
            fr = steadied(fr, dev)
            v = refine(fr, dev, focal, S.MASK_SCALE)
            info |= {"score": round(score(v, fr.masks, S.MASK_SCALE), 3), "delay": round(shift, 2)}
            print(f"after fitting distance, centre and heading per frame: {info['score']:.0%}", flush=True)
            if info["score"] >= USABLE:
                model(fr, v, S.MASK_SCALE, [job / "photos" / n for n in fr.names], flat_base).export(out)
    out.with_suffix(".json").write_text(json.dumps(info))
    return info


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4]), sys.argv[5] == "1")
