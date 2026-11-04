"""Photos (or video frames) -> camera poses (COLMAP) -> dense points (OpenMVS) -> printable mesh (app.mesh)."""

import functools
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import numpy as np
import pymeshlab
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from . import depth as D
from . import fuse as F
from . import mesh as M
from . import motion as MO
from . import refine as RF
from . import segment as S
from . import texture as TX
from . import video as V

# Tools are found on PATH (scripts/run-server.ps1 adds them on Windows). ffmpeg falls back to the pip-bundled build.
COLMAP = os.environ.get("COLMAP", "colmap")
FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
THREADS = str(os.cpu_count() or 2)
MAX_IMAGE = os.environ.get("MAX_IMAGE_SIZE", "1600")
MAX_POINTS = 1_500_000  # keeps cleanup/Poisson within a few GB of RAM
VIDEO = "video.mp4"  # in the job directory; MP4 or MOV content, ffmpeg goes by the content
MAX_FRAME_SIDE = 1920  # frames for dense matching and everything but the texture are scaled down to this
# Quick scans: dense matching at this size takes ~55% of the time. The fused surface's cells (0.25% of the object)
# still span more than a pixel; the photo texture keeps its full-size frames.
QUICK_FRAME_SIDE = 1440
# Express scans (Pro, about 5 minutes on the server's 12 threads): from an 80-frame scan's step times, photo matching
# grows with the frames squared, the rest with the frames; 30 frames, dense matching at 1280 px, no MobileSAM pass
# (1.5 min) and fusion cells of 0.4% (a quarter of the cells) give ~4-5 min.
EXPRESS_FRAMES = 30
EXPRESS_FRAME_SIDE = 1280
EXPRESS_VOXEL = 0.004
# One incremental model holding this share of the frames: the global solver is not run as well (it only helps when
# the incremental one splits the scan into pieces).
ONE_PIECE = 0.95
# Of the object's size: how far in front of a measured surface point a frame must see through the hull to carve it
# (dense points are a few tenths of a percent noisy).
FREE_MARGIN = 0.02
# The phone's sense of up, turned into the camera solver's coordinates frame by frame, must agree this well (median
# degrees) to be trusted: it checks the orientation log against the solved cameras.
GRAVITY_AGREE = 10.0
# Below this share of frames placed by the camera solver, the phone's motion sensors are tried (app/motion.py): a plain
# speaker on velvet had 46 of 80 frames placed, from part of the circle, and came out a wedge with flaps.
SFM_ENOUGH = 0.85
# The surface fused from the depth maps (app/fuse.py) is used when the dense points sit this close to it (median, of
# the object's size; a drill: 0.2%) and at least half the frames went into it; else Poisson on the dense points.
FUSED_OFF = 0.01


class Failed(Exception):
    """User-facing failure message."""


def note(job: Path, text: str):
    """A line in the job's log.txt next to the tools' output (GET /jobs/{id}/log)."""
    try:
        # UTF-8: Windows' default (cp1252) cannot write the Bulgarian in some lines, and the write would fail the job.
        with open(job / "log.txt", "a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n[{time.strftime('%H:%M:%S')}] {text}\n")
    except OSError:  # job deleted meanwhile
        pass


@functools.cache
def colmap_prefixes() -> tuple[str, str]:
    """COLMAP 4 renamed SiftExtraction./SiftMatching. options to FeatureExtraction./FeatureMatching."""
    r = subprocess.run([COLMAP, "feature_extractor", "-h"], capture_output=True, text=True)
    out = r.stdout + r.stderr  # COLMAP 4 prints help on stderr
    return ("FeatureExtraction", "FeatureMatching") if "FeatureExtraction.use_gpu" in out else ("SiftExtraction", "SiftMatching")


@functools.cache
def has_global_mapper() -> bool:
    """COLMAP 4 ships GLOMAP as `colmap global_mapper`; the Debian package (3.x) does not."""
    r = subprocess.run([COLMAP, "help"], capture_output=True, text=True)
    return "global_mapper" in r.stdout + r.stderr


def video_format(video: Path) -> str:
    """Codec, size and frame rate, as ffmpeg reads them: the app asks for 4K, but phones without it in the camera
    API record less (seen: 1080p), and that size limits the detail more than anything here."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True, text=True, errors="replace")
    line = next((ln.strip() for ln in r.stderr.splitlines() if "Video:" in ln), "")
    return line.split("Video:", 1)[1].strip()[:200] if line else "unknown"


# Adding a missing side: the source model's frames wait here (app/main.py start_job) until the new video's are out.
MORE_FRAMES = "more_frames"
EXTRA_FRAMES = 30  # frames taken from the new video: the old ones already cover the rest


def frames_to_take(job: Path, most: int) -> int:
    return min(most, EXTRA_FRAMES) if (job / MORE_FRAMES).is_dir() else most


def merge_more_frames(job: Path, photos: Path):
    """The source model's frames beside the new video's, solved together as one scan. The phone's orientation log
    covers only the new video, so it goes. All frames must be one size: the solver takes them as one camera."""
    more = job / MORE_FRAMES
    if not more.is_dir():
        return
    old = sorted(more.glob("*.jp*g"))
    new = sorted(photos.glob("*.jp*g"))
    if not old or not new:
        raise Failed("Кадрите на модела вече ги няма. Направете ново сканиране.")
    with Image.open(old[0]) as a, Image.open(new[0]) as b:
        if a.size != b.size:
            raise Failed("Заснемете допълнението със същия телефон и в същото положение като първото видео.")
    for f in old:
        shutil.copy(f, photos / f.name)
    (job / MO.MOTION).unlink(missing_ok=True)
    (job / MO.FRAME_TIMES).unlink(missing_ok=True)
    note(job, f"adding a side: {len(new)} new frames with the model's {len(old)}")


def extract_frames(video: Path, photos: Path, work: Path, sh: Callable[[str, list], None], most: int = V.MAX_FRAMES) -> int:
    """Writes the sharpest, evenly spread frames of the video (at most `most`) as photos/img_0001.jpg... Returns how
    many."""
    for p in photos.glob("*.jp*g"):  # a retry re-extracts from the video
        p.unlink()
    note(work.parent, f"video file: {video_format(video)}")
    # Both ffmpeg passes start with the same fps filter, so frame n means the same frame in each.
    raw = work / "thumbs.gray"
    sh("reading video", [FFMPEG, "-v", "error", "-y", "-i", video, "-an",
        "-vf", f"fps={V.CANDIDATE_FPS},scale={V.THUMB}:{V.THUMB},format=gray", "-f", "rawvideo", raw])  # fmt: skip
    thumbs = np.fromfile(raw, np.uint8)
    raw.unlink()
    scores = V.score_thumbs(thumbs)
    if len(scores) < max(V.MIN_FRAMES, V.MIN_SECONDS * V.CANDIDATE_FPS):
        raise Failed("Видеото е твърде кратко. Обиколете предмета бавно и докрай (около 30–60 секунди).")
    keep = V.pick(scores, V.frame_count(len(scores), most), V.motion(thumbs))
    del thumbs
    # When each frame was filmed, to match it with the phone's orientation (app/motion.py). The fps filter puts
    # candidate n at n / CANDIDATE_FPS s after the first frame.
    times = {f"img_{i + 1:04d}.jpg": round(k / V.CANDIDATE_FPS, 3) for i, k in enumerate(keep)}
    (work.parent / MO.FRAME_TIMES).write_text(json.dumps(times))
    kept = scores[keep]
    note(work.parent, f"video: {len(scores)} candidate frames, kept {len(keep)}; sharpness of kept frames "
         f"min {kept.min():.0f} / median {np.median(kept):.0f}, of all frames median {np.median(scores):.0f}")  # fmt: skip
    select = "+".join(f"eq(n,{i})" for i in keep)
    # Full size up to 4K: the photo texture reads small print from them (app/texture.py); camera solving and dense
    # matching scale them down themselves.
    scale = f"scale=w='trunc(iw*min(1,{TX.TEXTURE_SIDE}/max(iw,ih))/2)*2':h=-2"
    sh("picking frames", [FFMPEG, "-v", "error", "-y", "-i", video, "-an",
        "-vf", f"fps={V.CANDIDATE_FPS},select='{select}',{scale}", "-fps_mode", "vfr", "-q:v", "2",
        photos / "img_%04d.jpg"])  # fmt: skip
    return len(list(photos.glob("*.jpg")))


# COLMAP camera model id -> (number of parameters, whether it has separate fx and fy)
CAMERA_MODELS = {0: (3, False), 1: (4, True), 2: (4, False), 3: (5, False), 4: (8, True), 5: (8, True), 6: (12, True)}


def read_cameras(path: Path) -> dict[int, tuple[int, int, int, tuple[float, ...]]]:
    """(COLMAP model id, width, height, parameters) per camera id from a cameras.bin."""
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            if model not in CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model {model}")
            count, _ = CAMERA_MODELS[model]
            cams[cid] = (model, w, h, struct.unpack(f"<{count}d", f.read(8 * count)))
    return cams


def read_cameras_bin(path: Path) -> dict[int, tuple[float, float, float, float]]:
    """fx, fy, cx, cy per camera id from a COLMAP cameras.bin. Lens distortion is ignored: exact after undistortion
    (PINHOLE), close enough for judging a solution before it (SIMPLE_RADIAL, OPENCV...)."""
    return {
        cid: (p[0], p[1], p[2], p[3]) if CAMERA_MODELS[model][1] else (p[0], p[0], p[1], p[2])
        for cid, (model, _w, _h, p) in read_cameras(path).items()
    }


def distort(model: int, p: tuple[float, ...], x: np.ndarray, y: np.ndarray):
    """Pixel (u, v) in the original frame of normalized pinhole coordinates (x, y), for COLMAP's simple lens models
    (as COLMAP distorts); None for others."""
    r2 = x * x + y * y
    if model in (0, 2, 3):  # SIMPLE_PINHOLE f cx cy, SIMPLE_RADIAL f cx cy k, RADIAL f cx cy k1 k2
        f, cx, cy = p[:3]
        k1, k2 = (p[3:] + (0.0, 0.0))[:2]
        s = 1 + k1 * r2 + k2 * r2 * r2
        return f * x * s + cx, f * y * s + cy
    if model in (1, 4):  # PINHOLE fx fy cx cy, OPENCV fx fy cx cy k1 k2 p1 p2
        fx, fy, cx, cy = p[:4]
        k1, k2, p1, p2 = (p[4:] + (0.0,) * 4)[:4]
        s = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * s + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * s + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        return fx * xd + cx, fy * yd + cy
    return None


def undistort_masks(masks: dict[str, np.ndarray], distorted: Path, undistorted: Path):
    """The frames' object masks turned the way COLMAP's undistorter turned the frames (from the camera models of the
    model it was given and of its output), instead of segmenting the undistorted frames again (2.5 minutes of a
    scan). Returns ({name: mask}, {name: (width, height)}), or None for a lens model `distort` does not know."""
    src = read_cameras(distorted / "cameras.bin")
    src_cam = {name: cid for name, _, _, cid in read_images_bin(distorted / "images.bin")}
    dst = read_cameras(undistorted / "cameras.bin")
    out, sizes = {}, {}
    for name, _, _, cid in read_images_bin(undistorted / "images.bin"):
        model, w, h, (fx, fy, cx, cy) = dst[cid][0], dst[cid][1], dst[cid][2], dst[cid][3][:4]
        smodel, _, _, sp = src[src_cam[name]]
        if model != 1:  # the undistorter writes PINHOLE cameras
            return None
        cols, rows = np.meshgrid((np.arange(max(1, w // S.MASK_SCALE)) + 0.5) * S.MASK_SCALE,
                                 (np.arange(max(1, h // S.MASK_SCALE)) + 0.5) * S.MASK_SCALE)  # fmt: skip
        uv = distort(smodel, sp, (cols - cx) / fx, (rows - cy) / fy)
        if uv is None:
            return None
        u, v = uv
        m = ndimage.map_coordinates(masks[name].astype(np.float32), [v / S.MASK_SCALE - 0.5, u / S.MASK_SCALE - 0.5],
                                    order=1, cval=0.0)  # fmt: skip
        out[name], sizes[name] = np.clip(np.rint(m), 0, 255).astype(np.uint8), (w, h)
    return out, sizes


def read_images_bin(path: Path):
    """[(name, R, t, camera id)] from a COLMAP images.bin: world point X is at R X + t in the camera."""
    out = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _id, qw, qx, qy, qz, tx, ty, tz, cam = struct.unpack("<I7dI", f.read(64))
            name = bytearray()
            while (c := f.read(1)) != b"\0":
                name += c
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, 1)
            R = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            out.append((name.decode(), R, np.array([tx, ty, tz]), cam))
    return out


def read_points_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(points (n, 3), colours (n, 3) uint8) from a COLMAP points3D.bin: the solver's sparse points."""
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    pts, cols = np.empty((n, 3)), np.empty((n, 3), np.uint8)
    at = 8
    for i in range(n):
        # id (uint64), x y z (double), r g b (uint8), error (double), track length (uint64), then that many
        # (image id, point2D index) uint32 pairs
        x, y, z, r, g, b, _err, track = struct.unpack_from("<3d3BdQ", data, at + 8)
        pts[i], cols[i] = (x, y, z), (r, g, b)
        at += 8 + 43 + 8 * track
    return pts, cols


def read_views(model: Path) -> M.Views:
    cams = read_cameras_bin(model / "cameras.bin")
    imgs = read_images_bin(model / "images.bin")
    return M.Views(
        R=np.array([i[1] for i in imgs]),
        t=np.array([i[2] for i in imgs]),
        K=np.array([cams[i[3]] for i in imgs]),
        names=[i[0] for i in imgs],
    )


def mvs_indices(path: Path) -> list[int]:
    """For each image in images.bin order, its index in the OpenMVS scene (InterfaceCOLMAP adds the registered
    images in COLMAP id order)."""
    ids = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            ids.append(struct.unpack("<I", f.read(4))[0])
            f.seek(60, 1)
            while f.read(1) != b"\0":
                pass
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, 1)
    rank = {colmap_id: i for i, colmap_id in enumerate(sorted(ids))}
    return [rank[i] for i in ids]


def neighbor_lines(views: M.Views, index: list[int], best_deg=12.0, lo_deg=4.0, hi_deg=35.0, n=8) -> list[str]:
    """Each frame's partners for dense matching, as OpenMVS --view-neighbors-file lines "<id> <id> ...".

    OpenMVS picks partners from shared sparse points; on a plain table those are few and many frames got none (no
    depth at all). Frames of an orbit are good partners when they look at the object from a few to ~35° apart."""
    d = views.dirs
    angle = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
    lines = []
    for i in range(len(views)):
        ok = np.flatnonzero((angle[i] >= lo_deg) & (angle[i] <= hi_deg))
        if len(ok) < 2:  # few frames: take the closest ones that are not the same view
            ok = np.flatnonzero(angle[i] > 1.0)
        best = ok[np.argsort(np.abs(angle[i, ok] - best_deg))][:n]
        lines.append(" ".join(str(index[k]) for k in [i, *best]))
    return lines


def load_cloud(path: Path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    m = ms.current_mesh()
    pts, nrm, col = m.vertex_matrix(), m.vertex_normal_matrix(), m.vertex_color_matrix()
    if len(pts) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, nrm, col = pts[idx], nrm[idx], col[idx]
    has_normals = np.abs(nrm).sum() > 0
    return pts, (nrm if has_normals else None), col


def carved_hull(work: Path, dense: Path, views: M.Views, masks: list, cloud: M.Cloud, sh, found=None):
    """The visual hull, carved where the frames saw through it: the dense points (measured: a cup's inside, a handle's
    hole) and then AI depth (app/depth.py: dips and flat tops without dense points). The AI depth is scaled per frame
    to the known 3D points: the dense points (trusted most) and the hull where it does not overshoot them. Without the
    model, when the step fails, or when it would carve too much, the hull carved by the dense points alone. None if
    the outlines give no shape. found: M.hull_volume's result, if already at hand."""
    found = found if found is not None else M.hull_volume(views, masks, S.MASK_SCALE, cloud)
    outline = M.hull_mesh(*found) if found is not None else None
    if outline is None:
        return None
    frac, axes, step = found
    size = float(outline.extents.max())
    inside, probes = M.voxels_inside(frac, axes)
    probes = probes @ cloud.to_up  # back to world
    free = M.free_space_limit(views, cloud.points @ cloud.to_up, cloud.normals @ cloud.to_up, probes,
                              FREE_MARGIN * size)  # fmt: skip
    # Shares carved counted in voxels: a carved hull need not be closed, and then its volume means nothing.
    frac = M.limited(frac, inside, free)
    kept = int((frac >= M.HULL_AGREE).sum())
    hull = M.hull_mesh(frac, axes, step)
    if hull is None:
        hull, frac, kept = outline, found[0], len(inside)
    note(work.parent, f"dense points see through {1 - kept / len(inside):.0%} of the outline shape")
    if not D.MODEL.exists():
        note(work.parent, f"AI depth: skipped, model missing ({D.MODEL}); install-task.ps1 downloads it")
        return hull
    samples, _, _, over = M.hull_samples(hull, cloud.points, cloud.normals)
    refs = np.vstack([cloud.points, samples[~over]]) @ cloud.to_up  # back to world
    weights = np.r_[np.full(len(cloud.points), 3.0), np.ones(int((~over).sum()))]
    inputs, out = work / "depth_inputs.npz", work / "depth_carve.npz"
    np.savez(inputs, R=views.R, t=views.t, K=views.K, names=np.array(views.names), refs=refs, ref_weights=weights,
             probes=probes, margin=D.CARVE_MARGIN * size, agree=M.HULL_AGREE,
             **{f"masks_{i}": m for i, m in enumerate(masks)})  # fmt: skip
    try:
        sh("estimating depth", [sys.executable, D.__file__, dense / "images", inputs, out, work / CROPS])
        with np.load(out) as z:
            limit, fits, used = z["limit"], int(z["fits"]), bool(z["used"])
    except (Failed, OSError, ValueError, KeyError) as e:
        note(work.parent, f"AI depth: failed ({e}); going on with the outline shape")
        return hull
    ai = M.limited(frac, inside, limit)
    carved = M.hull_mesh(ai, axes, step) if used else None
    share = 1 - (ai >= M.HULL_AGREE).sum() / max(kept, 1) if carved is not None else 0.0  # beyond the dense points'
    use = carved is not None and share <= D.MAX_CARVE
    note(work.parent, f"AI depth: fits {fits} of {len(views)} frames, carves {share:.0%} more of the outline shape: "
         + ("using it" if use else "not used"))  # fmt: skip
    return carved if use else hull


CROPS = "depth_crops.npz"  # in work/: app/depth.py's depth crops, for app/fuse.py
LOOSE_AGREE = 0.5  # the looser hull for app/refine.py: half the frames' masks agree
PROMPT_MARGIN = 0.04  # of the size: how far outside the looser hull a measured point says "not object", how deep
# inside the hull one says "object" (app/refine.py)


def sharpened_masks(work: Path, dense: Path, views: M.Views, masks: list, cloud: M.Cloud, found, pts: np.ndarray, sh):
    """The masks redone by MobileSAM prompted from 3D (app/refine.py), or None: no hull, model missing, the step failed,
    or it fitted fewer than half the frames (then the first masks stay). pts: the dense points, world, background too."""
    if found is None:
        return None
    if not (RF.ENCODER.exists() and RF.DECODER.exists()):
        note(work.parent, f"sharpening the outlines: skipped, model missing ({RF.MODEL_DIR}); "
             "install-task.ps1 downloads it")  # fmt: skip
        return None
    frac, axes, step = found
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    tight = frac >= M.HULL_AGREE
    if not tight.any():
        return None
    size = float(np.ptp(grid[tight], axis=0).max())
    deep = tight & (ndimage.distance_transform_edt(tight) * step > PROMPT_MARGIN * size)
    loose = frac >= LOOSE_AGREE
    # Measured points well outside the looser hull (outside its grid too): the table, labels, the room.
    away = ndimage.distance_transform_edt(~loose) * step
    local = pts @ cloud.to_up.T
    lo = np.array([a[0] for a in axes]) - step / 2
    idx = np.floor((local - lo) / step).astype(int)
    ingrid = np.all((idx >= 0) & (idx < np.array(frac.shape)), axis=1)
    outside = np.ones(len(local), bool)
    outside[ingrid] = away[tuple(idx[ingrid].T)] > PROMPT_MARGIN * size
    outside = np.flatnonzero(outside)
    if len(outside) > 200_000:
        outside = np.random.default_rng(0).choice(outside, 200_000, replace=False)
    inputs, out = work / "refine_inputs.npz", work / "refine_masks.npz"
    np.savez(inputs, names=np.array(views.names), R=views.R, t=views.t, K=views.K, step=step,
             tight=grid[tight] @ cloud.to_up, loose=grid[loose] @ cloud.to_up, deep=grid[deep] @ cloud.to_up,
             outside=pts[outside], **{f"masks_{i}": m for i, m in enumerate(masks)})  # fmt: skip
    try:
        sh("sharpening the outlines", [sys.executable, RF.__file__, dense / "images", inputs, out, work / "sam_cache"])
        with np.load(out) as z:
            used = z["used"]
            sharp = [z[f"masks_{i}"] for i in range(len(views.names))]
    except (Failed, OSError, ValueError, KeyError) as e:
        note(work.parent, f"sharpening the outlines: failed ({e}); keeping the first masks")
        return None
    use = used.sum() >= len(views.names) / 2
    note(work.parent, f"sharpening the outlines: MobileSAM fitted {used.sum()} of {len(views.names)} frames: "
         + ("using them" if use else "keeping the first masks"))  # fmt: skip
    return sharp if use else None


def fused_surface(work: Path, dense: Path, views: M.Views, cloud: M.Cloud, hull: trimesh.Trimesh | None, sh,
                  voxel: float = F.VOXEL, flap: float | None = None, passes: int = F.PASSES):  # fmt: skip
    """The surface fused from the frames' depth maps (app/fuse.py), or None: no hull or AI depth crops, the step
    failed, or the dense points do not agree with it (then Poisson on the dense points)."""
    crops = work / CROPS
    if hull is None or not crops.exists():
        return None
    inputs, out = work / "fuse_inputs.npz", work / "fused.ply"
    np.savez(inputs, names=np.array(views.names), to_up=cloud.to_up, voxel=voxel, flap=F.FLAP if flap is None else flap,
             passes=passes,
             ground_z=np.nan if cloud.ground_z is None else cloud.ground_z, hull_v=hull.vertices, hull_f=hull.faces)  # fmt: skip
    try:
        sh("fusing depth", [sys.executable, F.__file__, dense, inputs, crops, out])
        info = json.loads(out.with_suffix(".json").read_text())
        mesh = trimesh.load(out, force="mesh")
    except (Failed, OSError, ValueError, KeyError) as e:
        note(work.parent, f"depth fusion: failed ({e}); using the dense points")
        return None
    size = float(np.ptp(cloud.points, axis=0).max())
    surface, _ = trimesh.sample.sample_surface(mesh, 400_000, seed=0)  # (a check, not the model)
    pick = np.random.default_rng(0).choice(len(cloud.points), min(len(cloud.points), 50_000), replace=False)
    off = float(np.median(cKDTree(surface).query(cloud.points[pick], workers=-1)[0])) / size
    use = info["frames"] >= len(views) / 2 and off <= FUSED_OFF
    note(work.parent, f"depth fusion: {info['frames']} of {info['total']} frames, dense points {off:.2%} of the size "
         "off its surface: " + ("using it" if use else "not used"))  # fmt: skip
    return mesh if use else None


def run(
    job: Path,
    flat_base: bool,
    set_state: Callable[[str], None],
    sh: Callable[[str, list], None],
    quick: bool = False,
    express: bool = False,
) -> dict:
    """sh(step_name, argv) runs one external tool and raises on failure. Returns the print report.
    quick: fewer frames and smaller dense frames, for about 40% of the time (the app's "Quick" scan).
    express: Pro's ~5 minute scan (EXPRESS_*)."""
    side = EXPRESS_FRAME_SIDE if express else QUICK_FRAME_SIDE if quick else MAX_FRAME_SIDE
    photos, video = job / "photos", job / VIDEO
    work = job / "work"
    shutil.rmtree(work, ignore_errors=True)
    TX.clear(job)  # a retried job's old texture
    dense = work / "dense"
    work.mkdir(parents=True)

    set_state("reconstructing")
    if video.exists():
        extract_frames(video, photos, work, sh, frames_to_take(job, EXPRESS_FRAMES if express else V.QUICK_FRAMES if quick else V.MAX_FRAMES))
        merge_more_frames(job, photos)
    n_photos = len(list(photos.glob("*.jp*g")))
    if quick or express:
        note(job, f"{'express' if express else 'quick'} scan: {n_photos} frames, dense matching at {side} px")

    best, placed, stray, all_masks, photo_masks = solve_cameras(job, work, photos, n_photos, sh)
    if placed < SFM_ENOUGH * n_photos and MO.load(job) is not None:
        focal = float(np.median(read_views(best).K[:, 0])) if placed >= 10 else 0.0
        found = by_motion(job, work, photo_masks, focal, placed, n_photos, flat_base, sh)
        if found is not None:
            mesh, used = found
            width = float(np.median([S.object_width(m) for m in all_masks.values()]))
            return finish(job, mesh) | {"photos_used": used, "photos_total": n_photos, "object_width": round(width, 3)}
    if best is None or placed < max(10, n_photos // 3):
        note(job, f"only {placed} of {n_photos} frames placed in 3D")
        raise Failed(
            "Кадрите не можаха да се съпоставят: видеото е твърде бързо или размазано, или предметът и подложката "
            "нямат достатъчно шарка. Снимайте по-бавно, дръжте предмета в кръга и го сложете на подложка с шарка в "
            "различен цвят от него."
        )
    best = drop_stray(work, best, stray, sh)

    # Dense matching gets the frames at full size: a small object is often only a few hundred pixels wide.
    sh("undistorting", [COLMAP, "image_undistorter", "--image_path", photos, "--input_path", best,
        "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(side)])  # fmt: skip
    masks_file, dense_masks = work / "masks.npz", work / "dense_masks"
    turned = undistort_masks(all_masks, best, dense / "sparse")
    if turned is None:
        sh("finding the object", [sys.executable, S.__file__, dense / "images", masks_file, dense_masks])
    else:
        warped, sizes = turned
        np.savez_compressed(masks_file, **warped)
        dense_masks.mkdir()
        for name, m in warped.items():
            Image.fromarray(S.dense_area(m, sizes[name])).save(dense_masks / f"{Path(name).stem}.mask.png")
    sh("dense setup", ["InterfaceCOLMAP", "-w", dense, "-i", dense, "-o", dense / "scene.mvs",
        "--image-folder", dense / "images"])  # fmt: skip
    neighbors = dense / "neighbors.txt"
    lines = neighbor_lines(read_views(dense / "sparse"), mvs_indices(dense / "sparse" / "images.bin"))
    neighbors.write_text("\n".join(lines) + "\n")
    # The slowest step. Only the area around the object is matched (the masks), which makes full resolution affordable:
    # at half resolution a small object got a few thousand points and its shape came from its outline alone.
    # Neighbouring video frames overlap heavily, so 4 views per depth map and one geometric pass are enough.
    # No region-of-interest crop: the masks already limit the work, and OpenMVS's own estimate once cut everything.
    # The depth maps stay for app/fuse.py (deleted after it).
    sh("dense matching", ["DensifyPointCloud", "-w", dense, "-i", dense / "scene.mvs", "-o", dense / "dense.mvs",
        "--view-neighbors-file", neighbors, "--mask-path", dense_masks, "--ignore-mask-label", "0",
        "--resolution-level", "0", "--max-resolution", str(side), "--number-views", "4",
        "--geometric-iters", "1", "--estimate-roi", "0", "--crop-to-roi", "0", "--remove-dmaps", "0",
        "--max-threads", THREADS])  # fmt: skip

    set_state("meshing")
    ply = dense / "dense.ply"
    if not ply.exists():
        raise Failed("Формата на предмета не можа да се измери. Може да е твърде лъскав, едноцветен или размазан на кадрите.")
    pts, nrm, col = load_cloud(ply)
    note(job, f"dense cloud: {len(pts)} points")
    views = read_views(dense / "sparse")
    with np.load(masks_file) as npz:
        masks = [npz[name] for name in views.names]
    width = float(np.median([S.object_width(m) for m in masks]))
    note(job, f"object width: median {width:.0%} of the frame")
    up = gravity_up(job, views)
    cloud = M.clean_cloud(pts, nrm, col, views, masks, S.MASK_SCALE, up)
    note(job, f"object cloud: {len(cloud.points)} points, table found: {cloud.ground_z is not None}")
    found = M.hull_volume(views, masks, S.MASK_SCALE, cloud)
    sharp = None if express else sharpened_masks(work, dense, views, masks, cloud, found, pts, sh)
    if sharp is not None:  # everything from here on with the better masks
        masks = sharp
        cloud = M.clean_cloud(pts, nrm, col, views, masks, S.MASK_SCALE, up)
        note(job, f"object cloud: {len(cloud.points)} points, table found: {cloud.ground_z is not None}")
        found = M.hull_volume(views, masks, S.MASK_SCALE, cloud)
    # The dense points where they are, the (carved) outline shape in their gaps. On a box this gave flat faces, sharp
    # edges and the right proportions; with AI depth points stacked in as well, the faces were rough with flaps at the
    # edges, and the box a third too flat (height 0.65 of its length; its dense points say 0.96).
    hull = carved_hull(work, dense, views, masks, cloud, sh, found)
    note(job, "visual hull: " + (f"{len(hull.faces)} faces" if hull is not None else "none, using the points only"))
    # The fusion gets the outline shape uncarved: it carves by itself, pixel by pixel. The dense points' carving cut
    # away the front of a drill's black handle (few, noisy points there), and the fusion keeps to near its hull.
    outline = M.hull_mesh(*found) if found is not None else None
    mesh = fused_surface(work, dense, views, cloud, outline, sh, EXPRESS_VOXEL if express else F.VOXEL)
    for p in dense.glob("*.dmap"):  # 40 MB each
        p.unlink(missing_ok=True)
    if mesh is None:
        mesh = M.poisson(cloud, hull)

    set_state("cleaning")
    mesh = M.clip_to_outline(mesh, views, masks, S.MASK_SCALE, cloud)
    M.paint(mesh, views, [dense / "images" / n for n in views.names], cloud.to_up)
    out = M.make_printable(mesh, M.floor_z(cloud), flat_base)
    photo_texture(job, work, photos, best, out, cloud, sh, all_masks)
    # The app suggests filming closer when the object filled little of the frames (less detail).
    return finish(job, out) | {"photos_used": placed, "photos_total": n_photos, "object_width": round(width, 3)}


def photo_texture(job: Path, work: Path, photos: Path, best: Path, out: trimesh.Trimesh, cloud: M.Cloud, sh,
                  all_masks: dict[str, np.ndarray]):  # fmt: skip
    """The printable model's photo texture (app/texture.py); on failure it keeps its vertex colours."""
    try:
        info = TX.texture_model(job, work, photos, best, out, out.metadata["to_print"], cloud.to_up, sh, COLMAP,
                                THREADS, read_views(best).names, all_masks)  # fmt: skip
        note(job, f"photo texture: from the sharpest {info['frames']} frames, {info['textures']} pixels, "
             f"{info['unseen']:.1%} of the faces seen by no frame (painted); seams levelled: {info['levelled']}")  # fmt: skip
    except (Failed, OSError, ValueError, KeyError) as e:
        TX.clear(job)
        note(job, f"photo texture: failed ({e}); the model keeps its vertex colours")


def drop_stray(work: Path, best: Path, stray: list[str], sh, name: str = "clean") -> Path:
    """The camera model without the frames placed wrongly (they would spoil the dense points and carve into the
    outline shape), in work/sparse/<name>."""
    if not stray:
        return best
    listed = work / f"{name}_dropped.txt"
    listed.write_text("\n".join(stray) + "\n")
    clean = work / "sparse" / name
    clean.mkdir()
    sh("camera solving", [COLMAP, "image_deleter", "--input_path", best, "--output_path", clean,
        "--image_names_path", listed])  # fmt: skip
    return clean


PAIR_NEAR = 5  # (paired matching) each frame with the next this many, in filming order
PAIR_EVERY = 2  # and every this-many-th frame with each other such frame (2, not 3: twice the links between a low and a high circle)


def match_pairs(names: list[str], near: int = PAIR_NEAR, every: int = PAIR_EVERY) -> list[tuple[str, str]]:
    """The frame pairs to match instead of all of them (frames in filming order): each with its next `near`, which
    chains the video, and every `every`-th frame with each other such frame, which closes the circle and joins a
    high circle to a low one. 48 frames: about 450 pairs instead of 1128 (matching ~0.1 s a pair on the server)."""
    pairs = {(i, j) for i in range(len(names)) for j in range(i + 1, min(len(names), i + near + 1))}
    spread = range(0, len(names), every)
    pairs |= {(i, j) for i in spread for j in spread if i < j}
    return [(names[i], names[j]) for i, j in sorted(pairs)]


def solve_cameras(job: Path, work: Path, photos: Path, n_photos: int, sh: Callable[[str, list], None],
                  paired: bool = False, mask_every: int = 1, features: int | None = None):  # fmt: skip
    """Where each frame was filmed from (COLMAP), and the object in each frame (AI masks, app/segment.py).
    paired: match the pairs of match_pairs, not every frame with every other (more frames in the same time).
    mask_every: cut the object out of every this-many-th frame only; the others only help place them, and count as
    stray. Returns (the best camera model's folder or None, how many of its frames with a mask agree with the
    object's outline, the frames that do not or have no mask, {frame name: mask}, the masks' file). features: the most per frame, instead of COLMAP's 8192."""
    sparse, db = work / "sparse", work / "db.db"
    sparse.mkdir(parents=True, exist_ok=True)
    ext, match = colmap_prefixes()
    sh("feature extraction", [COLMAP, "feature_extractor", "--database_path", db, "--image_path", photos,
        "--ImageReader.single_camera", "1", f"--{ext}.use_gpu", "0",
        f"--{ext}.max_image_size", MAX_IMAGE, f"--{ext}.num_threads", THREADS,
        "--SiftExtraction.peak_threshold", "0.002",
        *(["--SiftExtraction.max_num_features", str(features)] if features else [])])  # fmt: skip
    # Lower peak threshold: more features on plain surfaces (a suede cushion: 80 of 80 frames placed instead of 70).
    # Every frame against every other: closes the circle and links the low circle to the high one. No guided
    # matching: on CPU it took 12+ min for 20 frames.
    names = sorted(p.name for p in photos.glob("*.jp*g"))
    if paired:
        pair_list = work / "pairs.txt"
        pair_list.write_text("".join(f"{a} {b}\n" for a, b in match_pairs(names)))
        sh("photo matching", [COLMAP, "matches_importer", "--database_path", db, "--match_list_path", pair_list,
            "--match_type", "pairs", f"--{match}.use_gpu", "0", f"--{match}.num_threads", THREADS])  # fmt: skip
    else:
        sh("photo matching", [COLMAP, "exhaustive_matcher", "--database_path", db, f"--{match}.use_gpu", "0",
            f"--{match}.num_threads", THREADS])  # fmt: skip
    # The global mapper (COLMAP 4) solves all frames at once: on a plain cushion it placed 73 of 80 frames where the
    # incremental one split them into pieces of 44, 11 and 10. The incremental one still runs as the fallback.
    # Two solvers, as each fails on some videos: the incremental one splits a scan into pieces when the circles share
    # few features; the global one (COLMAP 4) places more frames, but on a plain cushion it once returned a model
    # whose cameras did not even face the object. Each model is scored by its frames that really aim at the object.
    (sparse / "incremental").mkdir()
    (sparse / "global").mkdir()
    # Neighbouring video frames are only a few degrees apart, so allow a smaller angle for the first pair, and
    # accept frames with fewer (but still verified) matches, as motion blur costs features.
    sh("camera solving", [COLMAP, "mapper", "--database_path", db, "--image_path", photos,
        "--output_path", sparse / "incremental", "--Mapper.num_threads", THREADS,
        "--Mapper.init_min_tri_angle", "8", "--Mapper.abs_pose_min_num_inliers", "20"])  # fmt: skip
    pieces = [read_views(d) for d in sorted((sparse / "incremental").iterdir()) if (d / "images.bin").exists()]
    one_piece = len(pieces) == 1 and len(pieces[0]) >= ONE_PIECE * n_photos
    if one_piece:
        note(job, f"camera solving: one piece with {len(pieces[0])} of {n_photos} frames; global solver not needed")
    if has_global_mapper() and not one_piece:
        try:
            # Video frames carry no focal length; the global mapper needs one estimated first.
            sh("camera solving", [COLMAP, "view_graph_calibrator", "--database_path", db])
            sh("camera solving", [COLMAP, "global_mapper", "--database_path", db, "--image_path", photos,
                "--output_path", sparse / "global"])  # fmt: skip
        except Failed:
            note(job, "global mapper failed; using the incremental one")
    if not one_piece:
        for family in ("incremental", "global"):
            merge_pieces(job, work, sparse, family, sh)

    # Judge each solution by the AI object masks: with correct poses, the object's outline shape built from all
    # frames matches every frame's mask; frames that do not match are placed wrongly and are dropped.
    photo_masks = work / "photo_masks.npz"
    masked = photos
    if mask_every > 1:
        masked = work / "masked"
        masked.mkdir(exist_ok=True)
        for name in names[::mask_every]:
            shutil.copy(photos / name, masked / name)
    sh("finding the object", [sys.executable, S.__file__, masked, photo_masks])
    with np.load(photo_masks) as npz:
        all_masks = {name: npz[name] for name in npz.files}
    models = []
    for d in sorted(p for p in sparse.rglob("*") if (p / "images.bin").exists()):
        views = read_views(d)
        has = np.array([n in all_masks for n in views.names])
        ok = np.zeros(len(views), bool)
        if has.sum() >= 3:
            ok[has] = M.consistent_frames(M.subset(views, has), [all_masks[n] for n in views.names if n in all_masks],
                                          S.MASK_SCALE)  # fmt: skip
        models.append((d, int(ok.sum()), [n for n, good in zip(views.names, ok) if not good]))
        note(job, f"camera solving: {d.parent.name} model {d.name}: {len(views)} frames, {ok.sum()} of "
             f"{int(has.sum())} with the object cut out consistent with its outline")  # fmt: skip
    best, placed, stray = max(models, key=lambda m: m[1], default=(None, 0, []))
    return best, placed, stray, all_masks, photo_masks


def refine_cameras(job: Path, work: Path, model: Path, sh: Callable[[str, list], None]) -> Path:
    """The camera model refined once more (COLMAP's bundle_adjuster); the model as it was if that fails."""
    out = work / "sparse" / "refined"
    out.mkdir(parents=True, exist_ok=True)
    try:
        sh("camera solving", [COLMAP, "bundle_adjuster", "--input_path", model, "--output_path", out])
    except Failed as e:
        note(job, f"camera solving: refining failed ({e})")
        return model
    return out


def merge_pieces(job: Path, work: Path, sparse: Path, family: str, sh: Callable[[str, list], None]):
    """One camera model from the pieces a solver split the frames into, where the pieces share frames: COLMAP's
    model_merger joins two pieces by their common frames (the biggest piece first, then each other one in turn), and
    bundle_adjuster refines the whole. A drill filmed in two circles at different distances came out as pieces of 35
    and 24 of its 48 frames, 11 of them in both (job f4e8): no piece showed it from all sides. The joined model is
    written to sparse/merged-<family>/0 and judged by the object's outlines like the solvers' own (solve_cameras), so
    it is used only when it fits them better."""
    folder = sparse / family
    pieces = sorted((d for d in folder.iterdir() if (d / "images.bin").exists()), key=lambda d: -len(read_views(d)))
    if len(pieces) < 2:
        return
    steps = work / f"merging-{family}"
    steps.mkdir(exist_ok=True)
    current, joined = pieces[0], [len(read_views(pieces[0]))]
    for k, other in enumerate(pieces[1:]):
        step = steps / str(k)
        step.mkdir(exist_ok=True)
        try:
            sh("camera solving", [COLMAP, "model_merger", "--input_path1", current, "--input_path2", other,
                "--output_path", step])  # fmt: skip
        except Failed:  # no frames in common: this piece stays apart
            continue
        if (step / "images.bin").exists() and len(read_views(step)) > len(read_views(current)):
            current = step
            joined.append(len(read_views(other)))
    if current == pieces[0]:
        note(job, f"camera solving: the {family} pieces share no frames; not joined")
        return
    out = sparse / f"merged-{family}" / "0"
    out.mkdir(parents=True, exist_ok=True)
    try:
        sh("camera solving", [COLMAP, "bundle_adjuster", "--input_path", current, "--output_path", out])
    except Failed:
        shutil.rmtree(out, ignore_errors=True)
        note(job, f"camera solving: joined the {family} pieces, but refining them failed; not used")
        return
    note(job, f"camera solving: joined {len(joined)} {family} pieces ({' + '.join(map(str, joined))} frames) into one "
         f"of {len(read_views(out))} frames")  # fmt: skip


def gravity_up(job: Path, views: M.Views) -> np.ndarray | None:
    """Up from the phone's gravity sensor (app/motion.py), if the frames agree on it; else None (guess from the
    cameras)."""
    found = MO.gravity_up(job, views)
    if found is None:
        return None
    up, spread = found
    guess = M.estimate_up(views.centers, views.dirs, views.ups)
    off = np.degrees(np.arccos(np.clip(up @ guess / np.linalg.norm(guess), -1, 1)))
    use = spread <= GRAVITY_AGREE
    note(job, f"up from the phone's gravity sensor: frames agree within {spread:.1f}°, {off:.0f}° from the cameras' "
         "guess: " + ("using it" if use else "not used"))  # fmt: skip
    return up if use else None


def finish(job: Path, mesh: trimesh.Trimesh) -> dict:
    mesh.export(job / "model.ply")
    shutil.rmtree(job / "work", ignore_errors=True)
    (job / VIDEO).unlink(missing_ok=True)  # the biggest file of a job; the extracted frames stay for debugging
    return M.report(mesh)


def by_motion(job: Path, work: Path, photo_masks: Path, focal: float, placed: int, n_photos: int, flat_base: bool,
              sh: Callable[[str, list], None]) -> tuple[trimesh.Trimesh, int] | None:  # fmt: skip
    """The shape from the phone's orientation and the outlines (app/motion.py), if the outlines agree with the sensors
    better than the camera solver did. Returns (printable mesh, frames used), or None to go on with the solver's."""
    out = work / "sensor_shape.ply"
    try:
        sh("shape from motion sensors", [sys.executable, MO.__file__, job, photo_masks, f"{focal:.1f}", out,
            "1" if flat_base else "0"])  # fmt: skip
        info = json.loads(out.with_suffix(".json").read_text())
    except (Failed, OSError, ValueError) as e:
        note(job, f"sensor poses: failed ({e}); going on with the camera solver's frames")
        return None
    score = info["score"]
    use = out.exists() and (score >= MO.GOOD or (score >= MO.USABLE and placed < n_photos / 2))
    note(job, f"sensor poses: outlines agree {score:.0%} over {info['frames']} frames filmed from {info['around']}° "
         f"around (camera solver placed {placed} of {n_photos}): " + ("using them" if use else "not used"))  # fmt: skip
    return (trimesh.load(out, force="mesh"), info["frames"]) if use else None
