"""AI depth for the object in each frame (Depth Anything V2 Small, ONNX on the CPU), to carve the visual hull.

The outlines give a shape that is never too small, but they cannot show a dip (a speaker's grille sunk inside its
rim) and leave a dome over a flat top seen only from above. A monocular depth network sees both, up to an unknown
scale and offset per frame, which is fitted to 3D points that are known (dense points, or the hull's walls). A point
of the hull is then empty when the frames that see it see the surface behind it, on average, by more than
CARVE_MARGIN. Averaged, the frames' small errors cancel; as points stacked from 80 frames (the way this was done
before), they made rough, striped surfaces with flaps at the edges, and a box came out a third too flat.

Runs as its own process (pipeline step "estimating depth"):
    python depth.py <images dir> <inputs.npz> <out.npz> [<crops.npz>]
inputs.npz: R (n,3,3), t (n,3), K (n,4) fx fy cx cy, names (n,), masks_<i> (uint8, any size), refs (m,3) world
points on the object's surface, ref_weights (m,), probes (k,3) world points inside the hull, margin (world units),
agree (the hull's level).
out.npz: limit (k,) the most each probe's hull share may be (see mesh.visual_hull), fits: frames fitted, used: whether
that was enough to carve (else limit is all 1).
crops.npz (optional): each frame's depth crop for app/fuse.py: disp_<i>, box_<i> (x0, y0, x1, y1), inner_<i>.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__:
    from . import parallel as PAR
else:  # run as a script (fuse.py and motion.py import this as part of the package)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parallel as PAR

MODEL = Path(
    os.environ.get(
        "DEPTH_MODEL", Path(__file__).resolve().parent.parent / "tools" / "depth" / "depth_anything_v2_small.onnx"
    )
)
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
SIZE = 518  # the network's native resolution (long side of the object crop)
MAX_SIDE = 1920  # frames are scaled down to this first (see predict)
MAX_ERROR = 0.05  # median relative depth error of a frame's fit; worse frames are left out
# Of the object's size: how far behind a point the frames must see the surface, on average, to carve it. The fits are
# a few tenths of a percent of the distance off (0.2-1% on a box).
CARVE_MARGIN = 0.03
MAX_CARVE = 0.3  # share of the hull's volume carving may remove; more means the fits went wrong
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def session():
    import onnxruntime as ort

    if not MODEL.exists():
        raise SystemExit(f"Depth model missing: {MODEL}. Run the setup again (it downloads {MODEL_URL}).")
    return ort.InferenceSession(str(MODEL), PAR.onnx_options(), providers=["CPUExecutionProvider"])


def disparity(sess, crop: Image.Image) -> np.ndarray:
    """Relative inverse depth (bigger = nearer) at the crop's size."""
    w, h = crop.size
    s = SIZE / max(w, h)
    iw, ih = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
    a = (np.asarray(crop.convert("RGB").resize((iw, ih), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
    d = sess.run(None, {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    return np.asarray(Image.fromarray(d.astype(np.float32)).resize((w, h), Image.BILINEAR))


def front_depths(Xc: np.ndarray, w: np.ndarray, fx, fy, cx, cy, box, cell=6):
    """Project camera-space points; keep per `cell`-pixel square only the nearest (the visible surface).
    Returns u, v, z, weight of the kept points inside box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    z = Xc[:, 2]
    ok = z > 1e-9
    u = fx * Xc[ok, 0] / z[ok] + cx
    v = fy * Xc[ok, 1] / z[ok] + cy
    z, w = z[ok], w[ok]
    inside = (u >= x0) & (v >= y0) & (u < x1) & (v < y1)
    u, v, z, w = u[inside], v[inside], z[inside], w[inside]
    if len(z) == 0:
        return u, v, z, w
    key = (v // cell).astype(np.int64) * 100_000 + (u // cell).astype(np.int64)
    order = np.lexsort((z, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    zmin = np.minimum.reduceat(z[order], np.flatnonzero(first))
    zmin_per_point = np.empty(len(z))
    zmin_per_point[order] = np.repeat(zmin, np.diff(np.r_[np.flatnonzero(first), len(z)]))
    near = z <= zmin_per_point * 1.01
    return u[near], v[near], z[near], w[near]


def fit_scale(d: np.ndarray, inv_z: np.ndarray, w: np.ndarray):
    """inv_z ~ a d + b, robustly (iteratively drops the worst 30%). Returns (a, b) and the median relative depth
    error. (A tilt term across the crop was tried: smoother layers, but points fitted the outlines worse.)
    The least squares written out, as this runs on many frames at once (app/parallel.py)."""
    d, inv_z, w = np.asarray(d, np.float64), np.asarray(inv_z, np.float64), np.asarray(w, np.float64)
    keep = np.ones(len(d), bool)
    coef = np.zeros(2)
    for _ in range(6):
        dk, yk, wk = d[keep], inv_z[keep], w[keep]
        s, sd, sy = wk.sum(), (wk * dk).sum(), (wk * yk).sum()
        sdd, sdy = (wk * dk * dk).sum(), (wk * dk * yk).sum()
        det = s * sdd - sd * sd
        # All the same depth: no slope to find (a <= 0 means no fit to the callers).
        if det > 1e-12 * s * sdd:
            coef = np.array([(s * sdy - sd * sy) / det, (sdd * sy - sd * sdy) / det])
        else:
            coef = np.array([0.0, sy / max(s, 1e-12)])
        r = np.abs(coef[0] * d + coef[1] - inv_z)
        keep = r <= np.percentile(r, 70)
    pred = coef[0] * d[keep] + coef[1]
    rel = np.abs(1 / np.maximum(pred, 1e-12) - 1 / inv_z[keep]) * inv_z[keep]
    return coef, float(np.median(rel)) if len(rel) else 1.0


@dataclass
class Frame:
    """One frame's object crop: what the network saw and what it returned. Arrays cover the box only."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the image
    disp: np.ndarray  # (h, w) network output
    inner: np.ndarray  # (h, w) object pixels well inside the outline
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray


def predict(sess, image: Image.Image, mask: np.ndarray, R, t, K) -> Frame | None:
    # Frames above MAX_SIDE (4K video, kept for the photo texture) are scaled down, their camera with them: the
    # network sees 518 pixels anyway, and every fitted frame's crop stays in memory.
    s = MAX_SIDE / max(image.size)
    if s < 1:
        image = image.resize((max(1, round(image.size[0] * s)), max(1, round(image.size[1] * s))), Image.BILINEAR)
        K = np.asarray(K, float) * s  # fx fy cx cy
    W, H = image.size
    m = np.asarray(Image.fromarray(mask).resize((W, H), Image.BILINEAR)) > 127
    if m.sum() < 400:
        return None
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(0.2 * max(x1 - x0, y1 - y0))
    bx0, by0, bx1, by1 = box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
    # Depth edges blur into the background: use only pixels well inside the outline.
    erode = max(2, int(0.02 * max(x1 - x0, y1 - y0)))
    inner = ndimage.binary_erosion(m[by0:by1, bx0:bx1], iterations=erode)
    return Frame(box, disparity(sess, image.crop(box)), inner, R, t, K)


def fit(f: Frame, refs: np.ndarray, ref_w: np.ndarray):
    """This frame's (a, b) from the reference points it sees. None if they do not fit."""
    fx, fy, cx, cy = f.K
    x0, y0, _, _ = f.box
    u, v, z, w = front_depths(PAR.turned(refs, f.R) + f.t, ref_w, fx, fy, cx, cy, f.box)
    r, c = v.astype(int) - y0, u.astype(int) - x0
    on = f.inner[r, c] if len(u) else np.zeros(0, bool)
    if on.sum() < 50:
        return None, f"{int(on.sum())} reference points"
    coef, err = fit_scale(f.disp[r[on], c[on]], 1 / z[on], w[on])
    if coef[0] <= 0 or err > MAX_ERROR:
        return None, f"poor fit (error {err:.1%})"
    return coef, f"error {err:.1%}"


def fit_frames(photos: list[Path], masks: list, R, t, K, refs: np.ndarray, ref_w: np.ndarray,
               seen: dict | None = None):  # fmt: skip
    """Each frame's depth crop and its scale (a, b) fitted to the known points, for the frames that fit.
    seen: if given, gets every frame's crop that showed the object, by frame index (fitted or not)."""

    def one(sess, i):
        p = photos[i]
        with Image.open(p) as im:
            f = predict(sess, im, masks[i], R[i], t[i], K[i])
        if f is None:
            print(f"{i + 1}/{len(photos)} {p.name}: no object", flush=True)
            return None, None
        coef, msg = fit(f, refs, ref_w)
        print(f"{i + 1}/{len(photos)} {p.name}: {msg}", flush=True)
        return f, coef

    fits = []
    for i, (f, coef) in enumerate(PAR.each_model(session, one, range(len(photos)))):
        if f is not None and seen is not None:
            seen[i] = f
        if coef is not None:
            fits.append((f, coef))
    print(f"depth fits {len(fits)} of {len(photos)} frames", flush=True)
    return fits


def enough(fits: list, n_frames: int) -> bool:
    """Whether the depth fits enough frames to be trusted (a real speaker: 79 of 80; plain synthetic pictures: 14)."""
    return len(fits) >= max(3, n_frames / 2)


def carve_limit(fits: list, points: np.ndarray, margin: float, agree: float) -> np.ndarray:
    """For world points inside the hull: the most their hull share may be. From how far behind each point the frames
    that see it (well inside the outline) see the surface, on average, each clipped to 2 margins so one bad frame
    cannot carve alone: `agree` (the hull's level) at the margin, solid in front of it, falling to empty beyond it.
    Frames that see the surface in front of a point vote too (unlike mesh.free_space_limit's measured points): AI depth
    is too rough to carve on the frames looking at it alone, which mangled a mug's inside. Points seen by fewer than 3
    frames keep 1. Smooth across points, so the carved surface is too (a yes/no per voxel left terraces)."""
    seen = np.zeros(len(points), np.int32)
    behind = np.zeros(len(points))
    for f, (a, b) in fits:
        fx, fy, cx, cy = f.K
        x0, y0, x1, y1 = f.box
        Xc = points @ f.R.T + f.t
        z = Xc[:, 2]
        ok = np.flatnonzero(z > 1e-9)
        c = np.floor(fx * Xc[ok, 0] / z[ok] + cx - x0).astype(np.int64)
        r = np.floor(fy * Xc[ok, 1] / z[ok] + cy - y0).astype(np.int64)
        inb = (c >= 0) & (r >= 0) & (c < x1 - x0) & (r < y1 - y0)
        ok, c, r = ok[inb], c[inb], r[inb]
        on = f.inner[r, c]
        ok, c, r = ok[on], c[on], r[on]
        inv = a * f.disp[r, c] + b
        seen[ok] += 1
        far = 1 / np.maximum(inv, 1e-9) - z[ok]  # how far beyond the point this frame sees the surface
        behind[ok] += np.where(inv > 1e-9, np.clip(far, -2 * margin, 2 * margin), 2 * margin)
    mean = behind / np.maximum(seen, 1)
    share = agree + (margin - mean) * (1 - agree) / (0.2 * margin)
    return np.where(seen >= 3, np.clip(share, 0, 1), 1.0)


def save_crops(path: Path, seen: dict) -> None:
    """The frames' depth crops, for app/fuse.py (which fits them to OpenMVS's depth maps instead)."""
    arrays = {}
    for i, f in seen.items():
        arrays |= {f"disp_{i}": f.disp.astype(np.float32), f"box_{i}": np.array(f.box), f"inner_{i}": f.inner}
    np.savez(path, **arrays)


def main(images: Path, inputs: Path, out: Path, crops: Path | None = None):
    z = np.load(inputs)
    names = [str(n) for n in z["names"]]
    masks = [z[f"masks_{i}"] for i in range(len(names))]
    seen = {} if crops is not None else None
    fits = fit_frames([images / n for n in names], masks, z["R"], z["t"], z["K"], z["refs"], z["ref_weights"],
                      seen)  # fmt: skip
    if crops is not None:
        save_crops(crops, seen)
    used = enough(fits, len(names))
    probes = z["probes"]
    limit = carve_limit(fits, probes, float(z["margin"]), float(z["agree"])) if used else np.ones(len(probes))
    np.savez(out, limit=limit.astype(np.float32), fits=len(fits), used=used)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]) if len(sys.argv) > 4 else None)
