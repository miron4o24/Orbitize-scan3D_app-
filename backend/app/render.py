"""A picture of the finished model for scan lists and shared links (jobs.thumbnail), drawn on the CPU: the server has
no GPU for OpenGL. The surface is sampled densely (texture colours where there is a photo texture, else the vertex
colours), lit from the camera's side and splatted into a depth buffer at twice the size, then scaled down smooth."""

import io
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

from . import texture as TX

SIDE = 480
SUPER = 2  # drawn at SUPER x SIDE, scaled down: smooth edges
POINTS_PER_PIXEL = 3  # surface samples per pixel of the drawing; the few holes left are filled
# A pixel the near surface got no sample in shows the model's far side through it, as a speck of another colour: a
# sample this far (of the drawing's side) behind its neighbours is treated as such a hole.
BEHIND = 0.02
FILL = 0.78  # share of the picture the model spans
BG_TOP, BG_BOTTOM = np.array([246, 248, 252]), np.array([228, 234, 243])
ELEVATION = np.radians(22)  # seen a little from above, from the side that shows the most (best_side)
SIDES = 8  # sides tried
SIDE_PIXELS = 160  # the small drawing each is judged on


def surface(job: Path, mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """count points on the model: positions, colours (0-255) and the normals of their faces."""
    parts = TX.load(job)
    rng = np.random.default_rng(0)
    out_p, out_c, out_n = [], [], []
    if parts:
        areas = [trimesh.Trimesh(p["vertices"], p["faces"], process=False).area for p in parts]
        for p, area in zip(parts, areas):
            m = trimesh.Trimesh(p["vertices"], p["faces"], process=False)
            n = max(1, int(count * area / sum(areas)))
            face = rng.choice(len(m.faces), n, p=m.area_faces / m.area_faces.sum())
            u, v = rng.random(n), rng.random(n)
            flip = u + v > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            w = np.stack([1 - u - v, u, v], 1)[:, :, None]
            corners = m.faces[face]
            out_p.append((m.vertices[corners] * w).sum(1))
            uv = (p["uv"][corners] * w).sum(1)
            with Image.open(io.BytesIO(p["jpeg"])) as im:
                img = np.asarray(im.convert("RGB"))
            h, wd = img.shape[:2]
            x = np.clip((uv[:, 0] * (wd - 1)).round().astype(int), 0, wd - 1)
            y = np.clip(((1 - uv[:, 1]) * (h - 1)).round().astype(int), 0, h - 1)
            out_c.append(img[y, x].astype(float))
            out_n.append(m.face_normals[face])
    else:
        pts, face = trimesh.sample.sample_surface(mesh, count, seed=0)
        out_p.append(pts)
        if mesh.visual.kind == "vertex":
            c = mesh.visual.vertex_colors[:, :3].astype(float)
            out_c.append(c[mesh.faces[face]].mean(1))
        else:
            out_c.append(np.full((len(pts), 3), 170.0))
        out_n.append(mesh.face_normals[face])
    return np.concatenate(out_p), np.concatenate(out_c), np.concatenate(out_n)


def view_turn(azimuth: float) -> np.ndarray:
    """World -> view: x right, y away from the camera, z up."""
    ca, sa, ce, se = np.cos(azimuth), np.sin(azimuth), np.cos(ELEVATION), np.sin(ELEVATION)
    turn = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    tilt = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return tilt @ turn


def best_side(pts: np.ndarray, col: np.ndarray) -> float:
    """The azimuth from which the model shows the most: the sharpest colour changes per pixel of its drawing (a face
    or a logo, rather than the plain back)."""
    pick = np.random.default_rng(0).choice(len(pts), min(len(pts), 400_000), replace=False)
    pts, grey = pts[pick], col[pick].mean(1)
    best, best_score = 0.0, -1.0
    for azimuth in 2 * np.pi * np.arange(SIDES) / SIDES:
        p = pts @ view_turn(azimuth).T
        lo, hi = p.min(0), p.max(0)
        scale = (SIDE_PIXELS - 1) / max(hi[0] - lo[0], hi[2] - lo[2], 1e-9)
        pix = ((hi[2] - p[:, 2]) * scale).astype(int) * SIDE_PIXELS + ((p[:, 0] - lo[0]) * scale).astype(int)
        order = np.lexsort((p[:, 1], pix))  # the nearest sample per pixel
        at, first = np.unique(pix[order], return_index=True)
        img = np.full(SIDE_PIXELS * SIDE_PIXELS, np.nan)
        img[at] = grey[order[first]]
        img = img.reshape(SIDE_PIXELS, SIDE_PIXELS)
        changes = np.r_[np.abs(np.diff(img, axis=0)).ravel(), np.abs(np.diff(img, axis=1)).ravel()]
        score = float(np.nanmean(changes))
        if score > best_score:
            best, best_score = float(azimuth), score
    return best


# The spinning video to share: upright for phones, one slow turn that starts on the model's best side.
SPIN_W, SPIN_H = 720, 1280
SPIN_DRAWN = 1.5  # drawn at this many times the size, scaled down: smooth edges
SPIN_FPS, SPIN_SECONDS = 20, 6
SPIN_POINTS = 3_000_000
SPIN_BG_TOP, SPIN_BG_BOTTOM = np.array([28, 28, 36]), np.array([9, 9, 11])


def spin_ground(w: int, h: int) -> np.ndarray:
    """The video's background with the name at the foot of it."""
    t = np.linspace(0, 1, h)[:, None, None]
    ground = Image.fromarray((SPIN_BG_TOP * (1 - t) + SPIN_BG_BOTTOM * t).repeat(w, 1).astype(np.uint8))
    size = round(h * 0.03)
    try:
        font = ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        font = ImageFont.load_default(size)
    ImageDraw.Draw(ground).text((w / 2, h * 0.93), "Scanly", font=font, fill=(150, 150, 160), anchor="mm")
    return np.asarray(ground)


def spin_frame(pts: np.ndarray, col: np.ndarray, nrm: np.ndarray, azimuth: float, scale: float, centre: np.ndarray,
               ground: np.ndarray) -> np.ndarray:  # fmt: skip
    h, w = ground.shape[:2]
    R = view_turn(azimuth)
    p = ((pts - centre) @ R.T) * scale
    x, y = np.round(p[:, 0] + w / 2).astype(np.int64), np.round(h * 0.46 - p[:, 2]).astype(np.int64)
    keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y, depth = x[keep], y[keep], p[keep, 1]
    n = nrm[keep] @ R.T
    far_first = np.argsort(-depth)  # each pixel keeps what is drawn last: the nearest
    x, y, depth, n = x[far_first], y[far_first], depth[far_first], n[far_first]
    seen = n * np.where(n[:, 1:2] > 0, -1.0, 1.0)
    light = np.array([-0.45, -0.75, 0.5])
    shade = 0.6 + 0.4 * np.clip(seen @ (light / np.linalg.norm(light)), 0, 1)
    rgb = np.zeros((h, w, 3), np.float32)
    nearest = np.full((h, w), np.inf, np.float32)
    rgb[y, x] = np.clip(col[keep][far_first] * shade[:, None], 0, 255)
    nearest[y, x] = depth
    mask = np.isfinite(nearest)
    mask &= nearest <= ndimage.minimum_filter(nearest, size=5) + BEHIND * h
    holes = ndimage.binary_closing(mask, iterations=2) & ~mask
    if holes.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
        rgb[holes] = rgb[iy[holes], ix[holes]]
        mask |= holes
    return np.where(mask[..., None], rgb, ground).astype(np.uint8)


def spin(job: Path, out: Path, ffmpeg: str, frames: int | None = None):
    """The model turning once, as an MP4 made with `ffmpeg`. frames: fewer, to try it out."""
    mesh = trimesh.load(job / "model.ply", force="mesh")
    pts, col, nrm = surface(job, mesh, SPIN_POINTS)
    pts, col, nrm = pts.astype(np.float32), col.astype(np.float32), nrm.astype(np.float32)
    start = best_side(pts, col)
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    w, h = round(SPIN_W * SPIN_DRAWN), round(SPIN_H * SPIN_DRAWN)
    reach = float(np.linalg.norm((pts - centre)[:, :2], axis=1).max())
    scale = min(0.42 * w / max(reach, 1e-9), 0.52 * h / max(float(hi[2] - lo[2]), 1e-9))
    ground = spin_ground(w, h)
    total = frames or SPIN_FPS * SPIN_SECONDS
    tmp = out.with_name(out.stem + ".part.mp4")
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SPIN_W}x{SPIN_H}",
         "-r", str(SPIN_FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    try:
        for i in range(total):
            frame = spin_frame(pts, col, nrm, start + 2 * np.pi * i / total, scale, centre, ground)
            small = Image.fromarray(frame).resize((SPIN_W, SPIN_H), Image.BILINEAR)
            proc.stdin.write(small.tobytes())
        proc.stdin.close()
        problem = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait(timeout=300) != 0:
            raise RuntimeError(f"ffmpeg: {problem[-300:]}")
    except BaseException:
        proc.kill()
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)


def picture(job: Path) -> Image.Image:
    mesh = trimesh.load(job / "model.ply", force="mesh")  # Z up, base on z = 0
    size = SIDE * SUPER
    pts, col, nrm = surface(job, mesh, int(POINTS_PER_PIXEL * size * size * 0.5))
    R = view_turn(best_side(pts, col))
    v = mesh.vertices @ R.T
    lo, hi = v.min(0), v.max(0)
    scale = FILL * size / max(hi[0] - lo[0], hi[2] - lo[2])
    centre = (lo + hi) / 2

    p = (pts @ R.T - centre) * scale
    n = nrm @ R.T
    x = np.round(p[:, 0] + size / 2).astype(int)
    y = np.round(size / 2 - p[:, 2]).astype(int)
    depth = p[:, 1]
    keep = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    x, y, depth, col, n = x[keep], y[keep], depth[keep], col[keep], n[keep]
    # Nearest sample per pixel: sort by pixel, then depth; the first of each pixel wins.
    pix = y * size + x
    order = np.lexsort((depth, pix))
    pix, first = np.unique(pix[order], return_index=True)
    sel = order[first]
    seen = n[sel] * np.where(n[sel][:, 1:2] > 0, -1.0, 1.0)  # turned to the camera (-y), whichever way a face points
    light = np.array([-0.45, -0.75, 0.5])  # towards the light: the camera's upper left
    shade = 0.55 + 0.45 * np.clip(seen @ (light / np.linalg.norm(light)), 0, 1)
    rgb = np.zeros((size * size, 3))
    mask = np.zeros(size * size, bool)
    rgb[pix] = np.clip(col[sel] * shade[:, None], 0, 255)
    mask[pix] = True
    rgb, mask = rgb.reshape(size, size, 3), mask.reshape(size, size)
    nearest = np.full((size, size), np.inf)
    nearest.flat[pix] = depth[sel]
    mask &= nearest <= ndimage.minimum_filter(nearest, size=5) + BEHIND * size
    # Pin holes between samples: filled from their neighbours.
    holes = ndimage.binary_closing(mask, iterations=2) & ~mask
    if holes.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
        rgb[holes] = rgb[iy[holes], ix[holes]]
        mask |= holes

    t = np.linspace(0, 1, size)[:, None, None]
    bg = (BG_TOP * (1 - t) + BG_BOTTOM * t).repeat(size, 1)
    out = Image.fromarray(bg.astype(np.uint8))
    # A soft shadow under the model.
    bx0, bx1 = (lo[0] - centre[0]) * scale + size / 2, (hi[0] - centre[0]) * scale + size / 2
    by = size / 2 - (lo[2] - centre[2]) * scale
    shadow = Image.new("L", (size, size), 0)
    ImageDraw.Draw(shadow).ellipse((bx0, by - size * 0.035, bx1, by + size * 0.035), fill=90)
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.025))
    out.paste(Image.new("RGB", (size, size), (120, 130, 150)), (0, 0), shadow)
    out.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 0), Image.fromarray((mask * 255).astype(np.uint8)))
    return out.resize((SIDE, SIDE), Image.LANCZOS)
