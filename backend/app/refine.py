"""Surer object masks once the cameras are known: MobileSAM (promptable segmentation) prompted from 3D, checked
against the visual hull.

IS-Net (app/segment.py) finds the object in each frame on its own and errs in a good share of them: it takes in a
chair behind a drill's handle, a label next to a figure, a strip of the table; it leaves out a black trigger, a
figure's face, its base. 3D knows better. The visual hull (where most masks agree) shows where the object is in
every frame: its view gives MobileSAM a box, points deep inside it say "object", and measured points well outside it
(the table, labels, the room) say "not object". MobileSAM returns four candidates; the one most like the hull's view
is kept, cut to near the hull's view (a chair it took in goes) and joined with the hull's core (a part it left out
comes back). On 8 frames of a drill where IS-Net erred in 5: no chair, the whole handle, chuck and bit, the full
battery, with edges that follow the picture.

Runs as its own process (pipeline step "sharpening the outlines"):
    python refine.py <images dir> <inputs.npz> <out.npz> [<cache dir>]
inputs.npz: names (n,), R (n,3,3), t (n,3), K (n,4) fx fy cx cy (world -> camera, as mesh.Views), masks_<i> (uint8
at 1/MASK_SCALE of the frame), tight (k,3) world points inside the hull, loose (m,3) inside a looser hull (half the
frames agree), deep (j,3) well inside the hull, outside (l,3) measured world points well outside the loose hull, step
(the hull grid's cell, world units). out.npz: masks_<i> as the input's (the old mask where MobileSAM found nothing
that fits), used (n,) bool. Cache dir: what MobileSAM's slow half saw in each frame's crop, for the next round: only its
quick half depends on the prompts that change.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__:
    from . import parallel as PAR
    from . import segment as S
else:  # run as a script: import the package by its path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parallel as PAR
    from app import segment as S

MODEL_DIR = Path(os.environ.get("SAM_MODEL_DIR", Path(__file__).resolve().parent.parent / "tools" / "sam"))
ENCODER, DECODER = MODEL_DIR / "mobile_sam_image_encoder.onnx", MODEL_DIR / "sam_mask_decoder_multi.onnx"
SIZE = 1024  # MobileSAM's input: the long side
CROP = 1.5  # side of the crop given to MobileSAM, in loose-hull sizes
POSITIVE, NEGATIVE = 6, 8  # prompt points
NEAR = 0.04  # of the object's size in the frame: how far MobileSAM's mask may reach beyond the loose hull's view
CORE = 0.08  # and how far inside the hull's view the object is taken as certain
MIN_FIT = 0.5  # the least overlap with the hull's view for MobileSAM's mask to be used
SPECK = 0.02  # parts of a mask under this share of its biggest part are specks of background
SAME_CROP = 1.3  # another round's crop is the same one (its picture's meaning already known) up to this much too big


def sessions():
    import onnxruntime as ort

    if not (ENCODER.exists() and DECODER.exists()):
        raise SystemExit(f"MobileSAM model missing in {MODEL_DIR}. Run install-task.ps1 again (it downloads it).")
    options = PAR.onnx_options()
    return (ort.InferenceSession(str(ENCODER), options, providers=["CPUExecutionProvider"]),
            ort.InferenceSession(str(DECODER), options, providers=["CPUExecutionProvider"]))  # fmt: skip


def embedding(enc, crop: Image.Image) -> np.ndarray:
    """MobileSAM's slow half: what it sees in the crop, whatever is asked of it after."""
    W, H = crop.size
    s = SIZE / max(W, H)
    a = np.asarray(crop.convert("RGB").resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR), np.float32)
    return enc.run(None, {enc.get_inputs()[0].name: a})[0]


def candidates(dec, emb: np.ndarray, size: tuple[int, int], box, points, labels) -> list[np.ndarray]:
    """MobileSAM's four masks (bool, crop size) for a box (x0, y0, x1, y1) and points (label 1 object, 0 not). emb: the
    embedding of the crop (of `size`: width, height)."""
    W, H = size
    s = SIZE / max(W, H)
    pts = np.array([box[:2], box[2:], *points], np.float32).reshape(-1, 2) * s
    feed = {
        "image_embeddings": emb,
        "point_coords": pts[None],
        "point_labels": np.array([2, 3, *labels], np.float32)[None],  # 2, 3: the box's corners
        "mask_input": np.zeros((1, 1, 256, 256), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
        "orig_im_size": np.array([H, W], np.float32),
    }
    masks = dec.run(None, feed)[0]
    return [m > 0 for m in masks[0]]


def project(X: np.ndarray, R, t, K):
    """Pixel positions (n, 2) and depths of world points."""
    Xc = PAR.turned(X, R) + t
    z = Xc[:, 2]
    zs = np.where(z > 1e-9, z, np.inf)
    return np.stack([K[0] * Xc[:, 0] / zs + K[2], K[1] * Xc[:, 1] / zs + K[3]], 1), z


def spread(uv: np.ndarray, k: int) -> np.ndarray:
    """k of the points, spread out (farthest point sampling from the one nearest their middle)."""
    if len(uv) <= k:
        return uv
    pick = [int(np.argmin(np.linalg.norm(uv - uv.mean(0), axis=1)))]
    dist = np.linalg.norm(uv - uv[pick[0]], axis=1)
    for _ in range(k - 1):
        pick.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(uv - uv[pick[-1]], axis=1))
    return uv[pick]


def view(uv: np.ndarray, z: np.ndarray, shape, radius: float) -> np.ndarray:
    """Where points (pixels in the crop) cover the crop, each drawn `radius` pixels wide."""
    out = np.zeros(shape, bool)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (z > 0) & (u >= 0) & (v >= 0) & (u < shape[1]) & (v < shape[0])
    out[v[ok], u[ok]] = True
    r = int(np.ceil(radius))
    return ndimage.maximum_filter(out, size=2 * r + 1) if r > 0 else out


def tidy(m: np.ndarray) -> np.ndarray:
    """Without specks (parts under SPECK of the biggest) and with small holes filled, as segment.py's masks."""
    lab, n = ndimage.label(m)
    if n > 1:
        areas = np.bincount(lab.ravel())[1:]
        m = np.isin(lab, np.flatnonzero(areas >= SPECK * areas.max()) + 1)
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, n = ndimage.label(holes)
    if n:
        areas = np.bincount(hl.ravel())[1:]
        m |= np.isin(hl, np.flatnonzero(areas < S.SMALL_HOLE * (m.sum() + holes.sum())) + 1)
    return m


def cached_crop(saved: Path | None, bx: tuple, x0, y0, x1, y1):
    """The (crop, embedding) an earlier round left for this frame, if that crop still holds all of this round's object
    at about the same size; else None."""
    if saved is None or not saved.exists():
        return None
    with np.load(saved) as c:
        old, emb = tuple(int(v) for v in c["box"]), c["emb"]
    holds = old[0] <= max(0, x0) and old[1] <= max(0, y0) and old[2] >= min(bx[2], x1) and old[3] >= min(bx[3], y1)
    near = max(old[2] - old[0], old[3] - old[1]) <= SAME_CROP * max(bx[2] - bx[0], bx[3] - bx[1])
    return (old, emb) if holds and near else None


def refine_frame(enc, dec, image: Image.Image, R, t, K, z: dict, step: float, old: np.ndarray,
                 saved: Path | None = None) -> np.ndarray | None:  # fmt: skip
    """The frame's mask (bool, full size), or None if the hull is not in view or MobileSAM found nothing that fits.
    old: the frame's first mask (uint8, small). saved: where this frame's embedding is kept between rounds."""
    W, H = image.size
    uvl, zl = project(z["loose"], R, t, K)
    front = zl > 0
    if front.sum() < 10:
        return None
    (x0, y0), (x1, y1) = uvl[front].min(0), uvl[front].max(0)
    side = max(x1 - x0, y1 - y0)
    if side < 8 or x1 < 0 or y1 < 0 or x0 > W or y0 > H:
        return None
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, CROP * side / 2
    bx = (max(0, int(cx - half)), max(0, int(cy - half)), min(W, int(np.ceil(cx + half))), min(H, int(np.ceil(cy + half))))
    pad = 0.03 * side
    kept = cached_crop(saved, bx, x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    if kept is not None:
        bx, emb = kept
    else:
        emb = embedding(enc, image.crop(bx))
        if saved is not None:
            np.savez(saved, box=np.array(bx), emb=emb)
    off = np.array(bx[:2], float)
    shape = (bx[3] - bx[1], bx[2] - bx[0])
    box = np.clip([x0 - pad - off[0], y0 - pad - off[1], x1 + pad - off[0], y1 + pad - off[1]], 0,
                  [shape[1], shape[0], shape[1], shape[0]])  # fmt: skip

    def in_box(uv, zz):
        return (zz > 0) & (uv[:, 0] >= box[0]) & (uv[:, 1] >= box[1]) & (uv[:, 0] < box[2]) & (uv[:, 1] < box[3])

    uvd, zd = project(z["deep"], R, t, K)
    uvd -= off
    pos = spread(uvd[in_box(uvd, zd)], POSITIVE)
    uvo, zo = project(z["outside"], R, t, K)
    uvo -= off
    neg = spread(uvo[in_box(uvo, zo)], NEGATIVE)
    cands = candidates(dec, emb, (shape[1], shape[0]), box, [*pos, *neg], [1] * len(pos) + [0] * len(neg))

    zmed = float(np.median(zl[front]))
    radius = 0.5 * step * K[0] / zmed + 1  # a hull cell's half width in pixels
    uvt, zt = project(z["tight"], R, t, K)
    tight = view(uvt - off, zt, shape, radius)
    loose = view(uvl - off, zl, shape, radius)
    if not tight.any():
        return None
    fits = [(c & tight).sum() / max((c | tight).sum(), 1) for c in cands]
    k = int(np.argmax(fits))
    if fits[k] < MIN_FIT:
        return None
    obj = np.sqrt(tight.sum())
    near = ndimage.binary_dilation(loose, iterations=max(2, int(NEAR * obj)))
    # The hull's core comes back only where the first mask had it too: the hull also fills gaps between parts that
    # this frame sees through (the table between a figure's cloak and its leg), which both masks rightly leave out.
    core = ndimage.binary_erosion(tight, iterations=max(2, int(CORE * obj)))
    first = np.asarray(Image.fromarray(old).resize((W, H), Image.BILINEAR))[bx[1] : bx[3], bx[0] : bx[2]] > 127
    out = np.zeros((H, W), bool)
    out[bx[1] : bx[3], bx[0] : bx[2]] = tidy((cands[k] & near) | (core & first))
    return out


def main(images: Path, inputs: Path, out: Path, cache: Path | None = None):
    z = dict(np.load(inputs))
    names = [str(n) for n in z["names"]]
    step = float(z["step"])
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(models, i):
        name, old = names[i], z[f"masks_{i}"]
        saved = cache / f"{Path(name).stem}.npz" if cache is not None else None
        with Image.open(images / name) as im:
            m = refine_frame(*models, im, z["R"][i], z["t"][i], z["K"][i], z, step, old, saved)
        if m is None:
            print(f"{i + 1}/{len(names)} {name}: kept the first mask", flush=True)
            return old, False
        small = Image.fromarray(m.astype(np.uint8) * 255).resize((old.shape[1], old.shape[0]), Image.BILINEAR)
        print(f"{i + 1}/{len(names)} {name}: sharpened", flush=True)
        return np.asarray(small), True

    done = PAR.each_model(sessions, one, range(len(names)))
    masks = {f"masks_{i}": m for i, (m, _) in enumerate(done)}
    used = np.array([u for _, u in done], bool)
    np.savez_compressed(out, used=used, **masks)
    print(f"sharpened {used.sum()} of {len(names)} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]) if len(sys.argv) > 4 else None)
