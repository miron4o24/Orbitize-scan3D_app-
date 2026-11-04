"""Colour changes for Pro's text edits (app/edit.py, operation "color"): "make the hair blue", "paint the base gold",
"make it red".

Which part is meant is found by AI on pictures of the model itself: it is drawn from VIEWS around it (on the CPU, as
app/render.py), CLIPSeg (CIDAS/clipseg-rd64-refined, Apache-2.0, as ONNX in tools/clipseg; scripts/models.ps1) marks
the part named in each picture, and every texel of the photo texture (and every vertex) takes the marks of the
pictures it is seen in, weighted by how straight on. CLIPSeg's marks are coarse (its picture is cut into 22 x 22
tiles), so the edge is drawn by colour: the part's own colours against the rest's, close to where CLIPSeg put it.

The new colour keeps the photo's shading and detail: in Lab, the part's lightness keeps its ups and downs around the
new colour's lightness, and its hue takes the new colour's (the hair's strands and the light on them stay).

CLIPSeg runs as its own process, so its memory is freed afterwards:
    python recolor.py <views.npz: images (V, SIDE, SIDE, 3) uint8, prompt> <out.npz: prob (V, SIDE, SIDE)>"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

MODEL_DIR = Path(os.environ.get("CLIPSEG_DIR", Path(__file__).resolve().parent.parent / "tools" / "clipseg"))
SIDE = 352  # CLIPSeg's picture size
SUPER = 2  # pictures drawn at SUPER x SIDE, then scaled down: smooth edges
MARGIN = 0.06  # share of each picture left free on every side of the model
VIEWS = [(az, 10) for az in range(0, 360, 45)] + [(az, 45) for az in range(22, 360, 90)] + [(0, 80)]
RENDER_POINTS = 2_500_000  # surface samples the pictures are drawn from
FOUND = 0.5  # CLIPSeg's mark (0-1) from which a point is taken as the part, before the colour check
SURE = 0.6  # and surely is: a piece of the part must hold such points (decide)
KEEP_PIECE = 0.05  # pieces with fewer sure points than this share of the biggest piece's are left out
STAND_NARROW = 0.5  # (stand) the figure above a stand is this much narrower
STAND_MAX = 0.4  # and a stand is at most this share of the model's height
TAP_SIDE = 512  # (tapped) picture size for MobileSAM
TAP_TILT = 35.0  # (tapped) degrees the side pictures turn away from straight on the spot
TAP_WHOLE = 0.35  # (tapped) a mask over this share of the model's outline is (most of) the whole model, not a part
ZOOM = 0.7  # a closer look when the part spans less than this share of the model
DARK = 12.0  # (decide) Lab lightness below which a colour says nothing of the part (shadow), full from DARK + 25
PLAIN = 4.0  # (decide) Lab spread over the surface below which a model is one colour
BINS = 16  # per Lab axis, for how much of each colour CLIPSeg marked
MIN_SHARE = 0.002  # less of the surface than this marked: the part was not found
BG = 242.0  # the pictures' background (light grey, as a product photo)

# Colour names, sRGB. Hair colours too, as hair is what people recolour most.
COLOURS = {
    "red": (200, 30, 35), "dark red": (130, 15, 20), "crimson": (180, 20, 50), "maroon": (110, 20, 30),
    "burgundy": (115, 20, 40), "scarlet": (230, 35, 20), "orange": (240, 120, 20), "yellow": (245, 210, 40),
    "gold": (212, 175, 55), "golden": (212, 175, 55), "lime": (140, 210, 40), "green": (40, 150, 60),
    "dark green": (20, 85, 35), "light green": (130, 210, 120), "olive": (110, 110, 40), "mint": (150, 230, 190),
    "teal": (20, 130, 130), "turquoise": (50, 200, 190), "cyan": (40, 200, 230), "aqua": (60, 210, 220),
    "light blue": (120, 180, 235), "sky blue": (120, 190, 240), "blue": (35, 80, 200), "dark blue": (20, 40, 120),
    "navy": (20, 30, 90), "navy blue": (20, 30, 90), "indigo": (75, 30, 130), "purple": (120, 50, 160),
    "violet": (140, 80, 200), "lavender": (180, 160, 225), "magenta": (210, 40, 160), "pink": (240, 130, 175),
    "hot pink": (245, 70, 150), "rose": (225, 110, 130), "brown": (110, 70, 40), "dark brown": (60, 38, 22),
    "light brown": (160, 115, 75), "chestnut": (125, 60, 35), "chocolate": (90, 50, 25), "beige": (215, 195, 160),
    "tan": (200, 160, 115), "cream": (240, 230, 200), "ivory": (245, 240, 220), "white": (245, 245, 245),
    "black": (22, 22, 24), "grey": (128, 128, 128), "gray": (128, 128, 128), "dark grey": (70, 70, 72),
    "dark gray": (70, 70, 72), "light grey": (190, 190, 192), "light gray": (190, 190, 192), "silver": (190, 192, 198),
    "copper": (184, 115, 51), "bronze": (160, 110, 50), "blonde": (225, 190, 115), "blond": (225, 190, 115),
    "platinum": (230, 225, 210), "platinum blonde": (230, 225, 210), "ginger": (190, 90, 35), "auburn": (140, 55, 35),
    "brunette": (90, 55, 35), "redhead": (175, 70, 35),
}  # fmt: skip
SHADES = {"light": 1.35, "pale": 1.45, "bright": 1.15, "dark": 0.55, "deep": 0.7}


def colour_rgb(name: str) -> tuple[int, int, int] | None:
    """A colour name ("blue", "dark green", "light blonde") or #rrggbb as sRGB; None if it is not one."""
    n = re.sub(r"\s+", " ", str(name).strip().lower().replace("colour", "").replace("color", "")).strip()
    m = re.fullmatch(r"#?([0-9a-f]{6})", n)
    if m:
        return tuple(int(m[1][i : i + 2], 16) for i in (0, 2, 4))
    if n in COLOURS:
        return COLOURS[n]
    words = n.split(" ")
    if len(words) == 2 and words[0] in SHADES and words[1] in COLOURS:
        lab = rgb_to_lab(np.array([COLOURS[words[1]]], np.float64))[0]
        lab[0] = float(np.clip(lab[0] * SHADES[words[0]] if SHADES[words[0]] < 1 else lab[0] + (100 - lab[0]) * (SHADES[words[0]] - 1), 3, 97))
        return tuple(int(v) for v in lab_to_rgb(lab[None])[0])
    return None


# --- sRGB <-> Lab (D65) ---------------------------------------------------------------------------------------------

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, np.float64)
    fy = (lab[:, 0] + 16) / 116
    f = np.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], 1)
    xyz = np.where(f**3 > 216 / 24389, f**3, (116 * f - 16) / (24389 / 27)) * _WHITE
    c = np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def recoloured(colours: np.ndarray, weight: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """colours (N, 3) uint8 turned towards `rgb` by weight (N,) 0-1: in Lab, lightness keeps its variation around
    the new colour's lightness (shading, strands, highlights); hue and saturation become the new colour's, with a
    little of the old variation."""
    out = np.asarray(colours, np.uint8).copy()
    sel = weight > 0.01
    if not sel.any():
        return out
    lab = rgb_to_lab(out[sel])
    w = weight[sel]
    mean = (lab * w[:, None]).sum(0) / w.sum()
    target = rgb_to_lab(np.array([rgb], np.float64))[0]
    # A big change of lightness squeezes the variation a little, or a dark part made white would clip to flat white.
    spread = float(np.sqrt(((lab[:, 0] - mean[0]) ** 2 * w).sum() / w.sum()))
    room = min(target[0], 100 - target[0])
    k = 1.0 if spread < 1e-6 else float(np.clip(room / (2.2 * spread), 0.35, 1.0))
    new = np.empty_like(lab)
    new[:, 0] = np.clip(target[0] + (lab[:, 0] - mean[0]) * k, 0, 100)
    new[:, 1:] = target[1:] + 0.25 * (lab[:, 1:] - mean[1:])
    mixed = lab_to_rgb(new).astype(np.float64) * w[:, None] + out[sel] * (1 - w[:, None])
    out[sel] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return out


# --- The surface's points: the texture's texels, or the vertices ------------------------------------------------------


def texels(part: dict, atlas: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """The texels of one texture image (app/texture.py load's part) that lie on the surface: their (rows, columns),
    3D points and normals. Faces no frame saw have a square of a few texels each (texture.paint_unseen): its texels
    all take the face's centre."""
    from . import texture as TX

    V, F, N = part["vertices"], part["faces"], part["normals"]
    uv = np.asarray(part["uv"], np.float64)[F]  # (faces, 3, 2), v up
    H, W = atlas.shape[:2]
    px = np.stack([uv[..., 0] * W, (1 - uv[..., 1]) * H], -1)
    e1, e2 = px[:, 1] - px[:, 0], px[:, 2] - px[:, 0]
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    flat = area < 1e-6
    owner, bary = TX.rasterize(uv, W, H, np.flatnonzero(~flat))
    yy, xx = np.nonzero(owner >= 0)
    f, b = owner[yy, xx], bary[yy, xx].astype(np.float32)
    pts = np.einsum("tk,tkc->tc", b, V[F[f]].astype(np.float32))
    nrm = np.einsum("tk,tkc->tc", b, np.asarray(N, np.float32)[F[f]])
    if flat.any():
        c = px[flat].mean(1)  # each square's centre
        steps = np.diff(np.unique(np.round(c[:, 0], 1)))
        cell = int(np.clip(np.round(steps[steps > 0.5].min()), 1, 4)) if (steps > 0.5).any() else 4
        x0 = np.round(c[:, 0] - cell / 2).astype(np.int64)
        y0 = np.round(c[:, 1] - cell / 2).astype(np.int64)
        dy, dx = np.divmod(np.arange(cell * cell), cell)
        sy, sx = (y0[:, None] + dy).reshape(-1), (x0[:, None] + dx).reshape(-1)
        ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
        which = np.repeat(np.flatnonzero(flat), cell * cell)[ok]
        yy, xx = np.r_[yy, sy[ok]], np.r_[xx, sx[ok]]
        pts = np.r_[pts, V[F[which]].mean(1).astype(np.float32)]
        nrm = np.r_[nrm, np.asarray(N, np.float32)[F[which]].mean(1)]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    return (yy, xx), pts, nrm


def recolour_atlas(atlas: np.ndarray, where: tuple[np.ndarray, np.ndarray], weight: np.ndarray,
                   rgb: tuple[int, int, int]) -> np.ndarray:  # fmt: skip
    """The texture image with its texels `where` (rows, columns) recoloured by `weight`. The texels between the
    patches (on no face) take their nearest patch texel's new colour again (as texture.level fills them): left as
    they were, a light colour showed thin dark seams on the phone, which blends neighbouring texels."""
    from . import texture as TX

    out = atlas.copy()
    yy, xx = where
    out[yy, xx] = recoloured(atlas[yy, xx], weight, rgb)
    on_face = np.zeros(atlas.shape[:2], bool)
    on_face[yy, xx] = True
    iy, ix = TX.nearest_covered(on_face)
    return np.where(on_face[..., None], out, out[iy, ix])


# --- Pictures of the model ------------------------------------------------------------------------------------------


def view_rotation(azimuth: float, elevation: float) -> np.ndarray:
    """World (z up) -> view: x right, y away from the camera, z up in the picture. The camera is `elevation` degrees
    above the model's middle, looking at it."""
    a, e = np.radians(azimuth), np.radians(elevation)
    look = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # from the camera to the model
    right = np.cross(look, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, look, np.cross(right, look)])


def draw(pts: np.ndarray, cols: np.ndarray, nrm: np.ndarray, R: np.ndarray, centre: np.ndarray,
         scale: float, side: int = SIDE) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The picture (side, side, 3) uint8 and its depth buffer (side, side; inf where empty), the nearest depth in
    each pixel. Points splatted at SUPER x side, lit from the camera's upper left, scaled down. scale: pixels per
    unit of the model in the picture."""
    size = side * SUPER
    p = (pts - centre) @ R.T * scale * SUPER
    x = np.round(p[:, 0] + size / 2).astype(np.int64)
    y = np.round(size / 2 - p[:, 2]).astype(np.int64)
    keep = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    pix, depth = (y * size + x)[keep], p[keep, 1]
    order = np.lexsort((depth, pix))
    pix, first = np.unique(pix[order], return_index=True)
    sel = np.flatnonzero(keep)[order[first]]
    n = nrm[sel] @ R.T
    n *= np.where(n[:, 1:2] > 0, -1.0, 1.0)  # turned to the camera, whichever way it points
    light = np.array([-0.45, -0.75, 0.5]) / np.linalg.norm([-0.45, -0.75, 0.5])
    shade = 0.6 + 0.4 * np.clip(n @ light, 0, 1)
    rgb = np.full((size * size, 3), BG)
    zbuf = np.full(size * size, np.inf)
    rgb[pix] = np.clip(cols[sel] * shade[:, None], 0, 255)
    zbuf[pix] = depth[order[first]] / SUPER
    rgb, zbuf = rgb.reshape(size, size, 3), zbuf.reshape(size, size)
    # Pin holes between samples: their neighbours' colour.
    from scipy import ndimage

    seen = np.isfinite(zbuf)
    holes = ndimage.binary_closing(seen, iterations=2) & ~seen
    if holes.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~seen, return_indices=True)
        rgb[holes] = rgb[iy[holes], ix[holes]]
    small = rgb.reshape(side, SUPER, side, SUPER, 3).mean((1, 3))
    near = zbuf.reshape(side, SUPER, side, SUPER).min((1, 3))
    return np.clip(np.round(small), 0, 255).astype(np.uint8), near


def seen_in(points: np.ndarray, R: np.ndarray, centre: np.ndarray, scale: float, near: np.ndarray,
            tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Which points are seen in the picture (not hidden behind nearer surface: near, its depth buffer), and their
    pixels (row, column)."""
    side = near.shape[0]
    p = (points - centre) @ R.T * scale
    x = np.floor(p[:, 0] + side / 2).astype(np.int64)
    y = np.floor(side / 2 - p[:, 2]).astype(np.int64)
    inside = (x >= 0) & (x < side) & (y >= 0) & (y < side)
    x, y = np.clip(x, 0, side - 1), np.clip(y, 0, side - 1)
    return inside & (p[:, 1] <= near[y, x] + tolerance), y, x


# --- Finding the part -------------------------------------------------------------------------------------------------

Query = tuple[np.ndarray, np.ndarray, np.ndarray]  # points, colours (0-255), normals


class NotFound(Exception):
    """The part named is not on the model (CLIPSeg marked less than MIN_SHARE of it)."""


def look(work: Path, part: str, draw_from: Query, frame: np.ndarray, margin: float, name: str,
         sh: Callable[[str, list], None]) -> list[tuple]:  # fmt: skip
    """Pictures from VIEWS, each fitted to the points `frame` (the model, or where the part was found) with `margin`
    (share of the picture) around, and CLIPSeg's marks of `part` in them: per picture (view rotation, depth buffer,
    marks 0-1, pixels per unit, centre)."""
    pts, cols, nrm = draw_from
    rotations = [view_rotation(a, e) for a, e in VIEWS]
    fits = []
    for R in rotations:
        p = frame @ R.T
        lo, hi = p.min(0), p.max(0)
        scale = (1 - 2 * margin) * SIDE / max(hi[0] - lo[0], hi[2] - lo[2], 1e-9)
        fits.append((scale, R.T @ np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])))
    drawn = [draw(pts, cols, nrm, R, centre, scale) for R, (scale, centre) in zip(rotations, fits)]
    pictures, marks = work / f"recolor_{name}.npz", work / f"recolor_{name}_marks.npz"
    np.savez_compressed(pictures, images=np.stack([d[0] for d in drawn]), prompt=np.array(prompt_for(part)))
    sh("finding the part", [sys.executable, __file__, pictures, marks])
    with np.load(marks) as z:
        prob = z["prob"].astype(np.float32)
    return [(R, d[1], pr, scale, centre) for R, d, pr, (scale, centre) in zip(rotations, drawn, prob, fits)]


# Words CLIPSeg knows better for the same part ("base": 0.13 at most on a figure's round base, "pedestal": 0.7).
SAME_PART = {"base": "pedestal", "stand": "pedestal", "plinth": "pedestal", "platform": "pedestal",
             "podium": "pedestal", "socle": "pedestal", "bottom": "pedestal"}  # fmt: skip


def prompt_for(part: str) -> str:
    words = part.strip().lower().split()
    return " ".join(SAME_PART.get(w, w) for w in words) if words else part


def gather(queries: list[Query], looks: list[tuple[list, float]], unseen: bool = True) -> list[np.ndarray]:
    """Each query point's mark: the mean over the pictures it is seen in, weighted by how straight on and by the
    look's weight; points in no picture (crevices) take their nearest seen point's."""
    from scipy.spatial import cKDTree

    out = []
    for q_pts, _, q_nrm in queries:
        total, weight = np.zeros(len(q_pts)), np.zeros(len(q_pts))
        for pictures, k in looks:
            for R, near, pr, scale, centre in pictures:
                vis, y, x = seen_in(q_pts, R, centre, scale, near, 2.5 / scale)  # depth: a few pixels' leeway
                w = k * vis * np.clip(np.abs(q_nrm @ R[1]), 0.15, 1.0)
                total += w * pr[y, x]
                weight += w
        p = np.zeros(len(q_pts))
        known = weight > 0
        p[known] = total[known] / weight[known]
        if unseen and (~known).any() and known.any():
            _, idx = cKDTree(q_pts[known]).query(q_pts[~known], workers=-1)
            p[~known] = p[known][idx]
        out.append(p)
    return out


def find(work: Path, part: str, draw_from: Query, queries: list[Query], sh: Callable[[str, list], None],
         note: Callable[[str], None]) -> list[np.ndarray]:  # fmt: skip
    """The weight (0-1) of the colour change at each query point for `part`. draw_from: the surface samples the
    pictures are drawn from; queries: per set (points, colours, normals). Raises NotFound if it is not there."""
    pts = draw_from[0]
    size = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    if set(part.lower().split()) & set(SAME_PART):
        weights = stand(queries)
        if weights is not None:
            note(f"edit: part „{part}“: the stand at the bottom, up to "
                 f"{float(np.concatenate([q[0][w > 0.5, 2] for q, w in zip(queries, weights)]).max()):.1f}")  # fmt: skip
            return weights
    work.mkdir(parents=True, exist_ok=True)
    looks = [(look(work, part, draw_from, pts, MARGIN, "views", sh), 1.0)]
    prob = gather(queries, looks)
    # A second, closer look at where it was found: CLIPSeg marks in tiles of 1/22 of its picture, so closer is finer.
    allp, allq = np.concatenate(prob), np.concatenate([q[0] for q in queries])
    there = allq[allp >= FOUND]
    zoomed = False
    if len(there) > 100 and np.linalg.norm(there.max(0) - there.min(0)) < ZOOM * size:
        looks.append((look(work, part, draw_from, there, 2 * MARGIN, "closer", sh), 2.0))
        prob = gather(queries, looks)
        zoomed = True
    weights = decide(prob, queries, size)
    share = sum(float((w > 0.5).sum()) for w in weights) / max(1, sum(len(w) for w in weights))
    note(f"edit: part „{part}“ is {share:.1%} of the surface (CLIPSeg, {len(VIEWS)} pictures"
         f"{f' and {len(VIEWS)} closer' if zoomed else ''})")  # fmt: skip
    if share < MIN_SHARE:
        raise NotFound(part)
    return weights


def looking(direction: np.ndarray) -> np.ndarray:
    """A view rotation (as view_rotation) for a camera looking along -direction (the camera on that side)."""
    look = -np.asarray(direction, np.float64) / np.linalg.norm(direction)
    right = np.cross(look, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-3:  # straight down or up: the picture's up is the model's y
        right = np.cross(look, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return np.stack([right, look, np.cross(right, look)])


def tapped(work: Path, draw_from: Query, queries: list[Query], spot: np.ndarray, normal: np.ndarray,
           sh: Callable[[str, list], None], note: Callable[[str], None]) -> list[np.ndarray]:  # fmt: skip
    """The part of the model under the spot the user tapped ("make this red"): MobileSAM (app/refine.py's model),
    told the tapped point, cuts out what is there in pictures from around the spot (straight on, and TAP_TILT to
    four sides); each point takes the share of the pictures it is seen in that cut it out, and only the piece that
    holds the spot stays. A colour rule could not do it: a dark red vest and the dark brown hair above it differ by
    less than the shading within each. Returns each point's weight 0-1."""
    from scipy import ndimage

    pts, cols, nrm = draw_from
    n = normal / np.linalg.norm(normal)
    a = np.cross(n, [0.0, 0.0, 1.0])
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-3 else np.array([1.0, 0.0, 0.0])
    b = np.cross(n, a)
    t = np.radians(TAP_TILT)
    rotations = [looking(d) for d in (n, *(n * np.cos(t) + s * np.sin(t) for s in (a, -a, b, -b)))]
    images, sils, points, pictures = [], [], [], []
    for R in rotations:
        p = pts @ R.T
        lo, hi = p.min(0), p.max(0)
        scale = (1 - 2 * MARGIN) * TAP_SIDE / max(hi[0] - lo[0], hi[2] - lo[2], 1e-9)
        centre = R.T @ np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])
        image, near = draw(pts, cols, nrm, R, centre, scale, TAP_SIDE)
        vis, y, x = seen_in(spot[None], R, centre, scale, near, 3.0 / scale)
        images.append(image)
        sils.append(np.isfinite(near))
        points.append([x[0], y[0], 1.0 if vis[0] else 0.0])
        pictures.append((R, near, centre, scale))
    work.mkdir(parents=True, exist_ok=True)
    request, answer = work / "tap_views.npz", work / "tap_marks.npz"
    np.savez_compressed(request, images=np.stack(images), silhouettes=np.stack(sils), points=np.array(points))
    sh("finding the part", [sys.executable, __file__, "tap", request, answer])
    with np.load(answer) as z:
        masks = z["masks"].astype(np.float32)
    used = [(R, near, m, scale, centre) for (R, near, centre, scale), m, p in zip(pictures, masks, points) if p[2] > 0]
    if not used:
        raise NotFound("the tapped spot")
    prob = gather(queries, [(used, 1.0)], unseen=False)  # what no picture sees is not guessed: tap there again
    # The piece that holds the spot, on a voxel grid of the surface.
    P = np.concatenate(prob)
    X = np.concatenate([q[0] for q in queries]).astype(np.float64)
    size = float(np.linalg.norm(X.max(0) - X.min(0)))
    keys = np.floor((X - X.min(0)) / (size / 256)).astype(np.int64)
    at_voxel = tuple(keys.T)
    occ = np.zeros(tuple(keys.max(0) + 1), bool)
    occ[tuple(keys[P > 0.5].T)] = True
    labels, _ = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    near_spot = np.argsort(np.linalg.norm(X - spot, axis=1))[:50]
    held = labels[at_voxel][near_spot]
    held = held[held > 0]
    if not len(held):
        raise NotFound("the tapped spot")
    kept = labels == np.bincount(held).argmax()
    flat = np.ravel_multi_index(at_voxel, occ.shape)
    hit = ((P > 0.5) & kept[at_voxel]).astype(float)
    total = ndimage.uniform_filter(np.bincount(flat, hit, occ.size).reshape(occ.shape), 3, mode="constant")
    count = ndimage.uniform_filter(np.bincount(flat, None, occ.size).astype(float).reshape(occ.shape), 3, mode="constant")
    w = np.clip((total.reshape(-1)[flat] / np.maximum(count.reshape(-1)[flat], 1e-9) - 0.25) / 0.5, 0, 1)
    note(f"edit: the tapped part is {float((w > 0.5).mean()):.1%} of the surface (MobileSAM, {len(used)} pictures)")
    out, at = [], 0
    for q in queries:
        out.append(w[at : at + len(q[0])])
        at += len(q[0])
    return out


def stand(queries: list[Query]) -> list[np.ndarray] | None:
    """The stand a figure is on, by its shape: from the bottom up to where the model's cross-section narrows to under
    STAND_NARROW of the stand's (a figure's feet on a round base). CLIPSeg marks such a stand only in patches.
    None when the model has no such step in its lower STAND_MAX (a mug, a vase)."""
    from scipy.spatial import ConvexHull, QhullError

    X = np.concatenate([q[0] for q in queries])
    z = X[:, 2]
    z0, height = float(z.min()), float(np.ptp(z))
    if height <= 0:
        return None
    step = height / 100
    rng = np.random.default_rng(0)
    areas = []
    for k in range(int(100 * STAND_MAX) + 1):
        sel = np.flatnonzero((z >= z0 + k * step) & (z < z0 + (k + 1) * step))
        if len(sel) < 10:
            areas.append(0.0)
            continue
        xy = X[rng.choice(sel, min(len(sel), 4000), replace=False), :2]
        try:
            areas.append(float(ConvexHull(xy).volume))  # in 2D: the area
        except QhullError:
            areas.append(0.0)
    areas = np.array(areas)
    ref = float(np.median(areas[1:4]))  # the stand's own (not its very bottom face)
    if ref <= 0:
        return None
    narrow = np.flatnonzero((areas < STAND_NARROW * ref) & (np.arange(len(areas)) >= 2))
    if not len(narrow):
        return None
    top = z0 + narrow[0] * step
    return [np.clip((top + step - q[0][:, 2]) / (2 * step), 0, 1) for q in queries]


def decide(prob: list[np.ndarray], queries: list[Query], size: float) -> list[np.ndarray]:
    """CLIPSeg's coarse marks made exact by colour and by being one piece:
    - colour: of each colour (Lab bins over the whole surface), the share CLIPSeg marked as the part. A forehead
      next to the hair is half marked by CLIPSeg, and ears inside the hair even more, but skin colour is rarely
      marked over the whole figure, so they keep it. Shadowed colours say little: the hair's last strands at the
      neck are as dark as the cloak below (Lab within 3), and there CLIPSeg's marks alone decide (0.65 against 0.03);
    - pieces: only the pieces (touching in 3D) that hold surely marked points stay: eyes as dark as the hair, but
      apart from it, keep their colour.
    Returns each point's weight 0-1 for the colour change: soft over about a voxel (1/256 of the model)."""
    from scipy import ndimage

    P = np.clip(np.concatenate(prob), 0.001, 0.999)
    L = rgb_to_lab(np.concatenate([np.asarray(q[1], np.float64)[:, :3] for q in queries]))
    X = np.concatenate([q[0] for q in queries])
    nb = BINS
    il = np.clip((L[:, 0] / 100 * nb).astype(int), 0, nb - 1)
    ia = np.clip(((L[:, 1] + 64) / 128 * nb).astype(int), 0, nb - 1)
    ib = np.clip(((L[:, 2] + 64) / 128 * nb).astype(int), 0, nb - 1)
    idx = (il * nb + ia) * nb + ib
    marked = ndimage.uniform_filter(np.bincount(idx, (P >= FOUND).astype(float), nb**3).reshape(nb, nb, nb), 3,
                                    mode="nearest")  # fmt: skip
    count = ndimage.uniform_filter(np.bincount(idx, None, nb**3).astype(float).reshape(nb, nb, nb), 3, mode="nearest")
    by_colour = np.clip(((marked + 0.5) / (count + 1.0)).reshape(-1)[idx], 0.01, 0.99)
    says = np.clip((L[:, 0] - DARK) / 25.0, 0, 1)  # shadows' colours say nothing; lit ones all they can
    if float(L.std(0).mean()) < PLAIN:  # one colour all over (a grey model): colour says nothing anywhere
        says[:] = 0
    final = 1 / (1 + np.exp(-(np.log(P / (1 - P)) + says * np.log(by_colour / (1 - by_colour)))))
    # Pieces, on a voxel grid of the surface.
    voxel = size / 256
    keys = np.floor((X - X.min(0)) / voxel).astype(np.int64)
    at_voxel = tuple(keys.T)
    occ = np.zeros(tuple(keys.max(0) + 1), bool)
    inside = final > 0.5
    occ[tuple(keys[inside].T)] = True
    labels, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    sure = np.bincount(labels[at_voxel][inside & (P >= SURE)], minlength=n + 1)
    sure[0] = 0
    keep = sure >= max(1, KEEP_PIECE * sure.max())
    kept = keep[labels] & occ
    # In or out per point, then softened over a voxel or so: no speckles in shadows where the marks hover near half.
    flat = np.ravel_multi_index(at_voxel, occ.shape)
    hit = (inside & kept[at_voxel]).astype(float)
    total = ndimage.uniform_filter(np.bincount(flat, hit, occ.size).reshape(occ.shape), 3, mode="constant")
    count = ndimage.uniform_filter(np.bincount(flat, None, occ.size).astype(float).reshape(occ.shape), 3, mode="constant")
    w = np.clip((total.reshape(-1)[flat] / np.maximum(count.reshape(-1)[flat], 1e-9) - 0.25) / 0.5, 0, 1)
    out, at = [], 0
    for p in prob:
        out.append(w[at : at + len(p)])
        at += len(p)
    return out


# --- CLIPSeg (own process) ------------------------------------------------------------------------------------------


def ready() -> bool:
    return (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()


def main(pictures: Path, out: Path):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    with np.load(pictures) as z:
        images, prompt = z["images"], str(z["prompt"])
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    ids = tok.encode(prompt).ids[:77]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.log_severity_level = 3  # the model's output shape note on every run
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    prob = []
    for i, img in enumerate(images):
        x = ((img / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        logits = session.run(["logits"], {"input_ids": np.array([ids], np.int64), "pixel_values": x,
                                          "attention_mask": np.ones((1, len(ids)), np.int64)})[0]  # fmt: skip
        p = 1 / (1 + np.exp(-logits.reshape(SIDE, SIDE)))
        prob.append(p.astype(np.float16))
        print(f"{i + 1}/{len(images)}: up to {p.max():.2f}", flush=True)
    np.savez_compressed(out, prob=np.stack(prob))
    print(json.dumps({"prompt": prompt, "pictures": len(images)}), flush=True)


def tap_main(request: Path, out: Path):
    """MobileSAM on the pictures around a tapped spot: per picture, of its masks for the tapped point, the one it
    rates best that holds the point and is not the whole model (over TAP_WHOLE of its outline)."""
    import onnxruntime as ort

    sam = Path(os.environ.get("SAM_MODEL_DIR", Path(__file__).resolve().parent.parent / "tools" / "sam"))
    enc = ort.InferenceSession(str(sam / "mobile_sam_image_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(sam / "sam_mask_decoder_multi.onnx"), providers=["CPUExecutionProvider"])
    with np.load(request) as z:
        images, sils, points = z["images"], z["silhouettes"], z["points"]
    masks = []
    for i, (img, sil, (x, y, seen)) in enumerate(zip(images, sils, points)):
        H, W = img.shape[:2]
        if not seen:
            masks.append(np.zeros((H, W), bool))
            continue
        s = 1024 / max(H, W)
        a = np.asarray(Image.fromarray(img).resize((round(W * s), round(H * s)), Image.BILINEAR), np.float32)
        emb = enc.run(None, {enc.get_inputs()[0].name: a})[0]
        feed = {
            "image_embeddings": emb,
            "point_coords": (np.array([[x + 0.5, y + 0.5], [0.0, 0.0]], np.float32) * s)[None],
            "point_labels": np.array([[1, -1]], np.float32),  # -1: the padding point when no box is given
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([H, W], np.float32),
        }
        got, scores = dec.run(["masks", "iou_predictions"], feed)
        best, best_score = None, -1.0
        for m, score in zip(got[0] > 0, scores[0]):
            share = (m & sil).sum() / max(1, sil.sum())
            if m[int(y), int(x)] and share < TAP_WHOLE and score > best_score:
                best, best_score = m & sil, float(score)
        masks.append(best if best is not None else np.zeros((H, W), bool))
        print(f"{i + 1}/{len(images)}: {0 if best is None else best.sum() / max(1, sil.sum()):.1%} of the model, "
              f"rated {best_score:.2f}", flush=True)
    np.savez_compressed(out, masks=np.stack(masks))


if __name__ == "__main__":
    if sys.argv[1] == "tap":
        tap_main(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
