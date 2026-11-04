"""Object masks: which pixels of each frame show the object (salient-object segmentation, IS-Net, CPU via ONNX).

Two passes per frame. On the whole frame, a small object competes with everything else that stands out: a speaker
filling 20% of the width came back joined to the desk and monitor behind it in 12 of 80 frames, which wrecked the
outline shape and the check of the camera poses. So the first pass only finds where the object is (the part nearest
the middle, where the app keeps it), and the second segments a close crop around it, where the object is the main
thing in the picture. Frames where the first pass merged it with the background take the place and size of the
object from the frames before and after.

Runs as its own process (pipeline step "finding the object"), so the model's memory is freed afterwards and the
step can be cancelled like the other tools:
    python segment.py <images dir> <out.npz> [dense masks dir]
The .npz maps each image name to a uint8 mask (0..255) at 1/MASK_SCALE of the image size.
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__:
    from . import parallel as PAR
else:  # run as a script: import the package by its path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parallel as PAR

MODEL = Path(os.environ.get("SEG_MODEL", Path(__file__).resolve().parent.parent / "tools" / "seg" / "isnet-general-use.onnx"))
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
MASK_SCALE = 4  # masks only decide inside/outside; a quarter of the frame size is plenty and keeps them small
SIZE = 1024  # the model's input size
CROP = 2.4  # side of the second pass's square crop, in object sizes
MERGED = 1.8  # a first-pass object this many times the typical size has merged with the background
SMALL_HOLE = 0.02  # holes in an object's mask below this share of it are filled


def session():
    import onnxruntime as ort

    if not MODEL.exists():
        raise SystemExit(f"Segmentation model missing: {MODEL}. Run the setup again (it downloads {MODEL_URL}).")
    return ort.InferenceSession(str(MODEL), PAR.onnx_options(), providers=["CPUExecutionProvider"])


def predict(sess, image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """The model's 0..255 object map of the image, at `size` (width, height)."""
    a = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32)
    a = a / max(float(a.max()), 1e-6) - (0.485, 0.456, 0.406)
    pred = sess.run(None, {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
    return np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize(size, Image.BILINEAR))


def mask(sess, image: Image.Image) -> np.ndarray:
    """One pass over the whole frame, at 1/MASK_SCALE of its size."""
    return predict(sess, image, (max(1, image.size[0] // MASK_SCALE), max(1, image.size[1] // MASK_SCALE)))


def central_part(m: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    """The connected object region nearest (cx, cy) (mask pixels), ignoring specks; None if there is none."""
    lab, n = ndimage.label(m > 127)
    if n == 0:
        return None
    areas = np.bincount(lab.ravel())[1:]
    big = np.flatnonzero(areas >= 0.1 * areas.max()) + 1
    ys, xs = np.nonzero(np.isin(lab, big))
    i = np.argmin((xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2)
    part = lab == lab[ys[i], xs[i]]
    # Fill small holes (specks the network missed), keep real ones, which the outline shape then shows: a ring's middle
    # was 12% of it, a mug handle's opening 7%; filled, both came out solid.
    holes = ndimage.binary_fill_holes(part) & ~part
    hl, n = ndimage.label(holes)
    if n:
        areas = np.bincount(hl.ravel())[1:]
        part |= np.isin(hl, np.flatnonzero(areas < SMALL_HOLE * (part.sum() + holes.sum())) + 1)
    return part


def locate(m: np.ndarray) -> tuple[float, float, float] | None:
    """Centre x, y and size (larger box side) of the object nearest the middle of a first-pass mask, in mask pixels."""
    part = central_part(m, m.shape[1] / 2, m.shape[0] / 2)
    if part is None:
        return None
    ys, xs = np.nonzero(part)
    return (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2, float(max(np.ptp(xs), np.ptp(ys)) + 1)


def track(found: list) -> list:
    """Per frame (in filming order) where to crop: the first pass's object, or for frames where it is missing or
    merged with the background (much bigger than usual), the place and size interpolated from the other frames."""
    sizes = [f[2] for f in found if f is not None]
    if not sizes:
        return [None] * len(found)
    typical = float(np.median(sizes))
    ok = [i for i, f in enumerate(found) if f is not None and f[2] <= MERGED * typical]
    idx = np.arange(len(found))
    cols = [np.interp(idx, ok, [found[i][k] for i in ok]) for k in range(3)]
    return [(float(cols[0][i]), float(cols[1][i]), float(cols[2][i])) for i in idx]


def crop_mask(sess, image: Image.Image, cx: float, cy: float, size: float) -> np.ndarray:
    """Second pass: the object segmented in a crop of CROP times its size around (cx, cy), as a whole-frame mask at
    1/MASK_SCALE (only the object's region; everything else 0)."""
    W, H = image.size[0] // MASK_SCALE, image.size[1] // MASK_SCALE
    half = max(16.0, CROP * size / 2)
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(W, int(np.ceil(cx + half))), min(H, int(np.ceil(cy + half)))
    out = np.zeros((H, W), np.uint8)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out
    box = (x0 * MASK_SCALE, y0 * MASK_SCALE, x1 * MASK_SCALE, y1 * MASK_SCALE)
    m = predict(sess, image.crop(box), (x1 - x0, y1 - y0))
    part = central_part(m, cx - x0, cy - y0)
    if part is not None:
        # Keep the soft edge around the part (masks are compared at 127 later, and resized).
        near = ndimage.binary_dilation(part, iterations=2)
        out[y0:y1, x0:x1] = np.where(part, np.maximum(m, 128), np.where(near, m, 0))
    return out


def object_width(m: np.ndarray) -> float:
    """Share of the frame width the object spans (0 if not found)."""
    xs = np.nonzero((m > 127).any(axis=0))[0]
    return (xs[-1] - xs[0] + 1) / m.shape[1] if len(xs) else 0.0


def dense_area(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Where dense matching should work, at full image size: the object's box grown by half its size on each side
    (a ring of table around it, for levelling, and slack for imperfect masks). Everything if no object was found."""
    out = np.zeros((size[1], size[0]), np.uint8)
    ys, xs = np.nonzero(m > 127)
    if len(xs) < 20:
        out[:] = 255
        return out
    x0, x1, y0, y1 = xs.min() * MASK_SCALE, (xs.max() + 1) * MASK_SCALE, ys.min() * MASK_SCALE, (ys.max() + 1) * MASK_SCALE
    pad = max(x1 - x0, y1 - y0) // 2
    out[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = 255
    return out


def main(images: Path, out: Path, dense_masks: Path | None = None):
    """dense_masks: also write <image stem>.mask.png there (0 = skip) for OpenMVS --mask-path."""
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if dense_masks:
        dense_masks.mkdir(parents=True, exist_ok=True)

    def first_pass(sess, f):
        with Image.open(f) as im:
            return locate(mask(sess, im))

    found = PAR.each_model(session, first_pass, files)
    crops = track(found)
    print(f"first pass: object near the middle in {sum(c is not None for c in found)} of {len(files)} frames", flush=True)

    def second_pass(sess, item):
        i, f = item
        with Image.open(f) as im:
            size = im.size
            if crops[i] is None:
                m = np.zeros((size[1] // MASK_SCALE, size[0] // MASK_SCALE), np.uint8)
            else:
                m = crop_mask(sess, im, *crops[i])
        if dense_masks:
            Image.fromarray(dense_area(m, size)).save(dense_masks / f"{f.stem}.mask.png")
        print(f"{i + 1}/{len(files)} {f.name}: object {object_width(m):.0%} of the frame width", flush=True)
        return m

    done = PAR.each_model(session, second_pass, list(enumerate(files)))
    masks = {f.name: m for f, m in zip(files, done)}
    widths = [object_width(m) for m in done]
    np.savez_compressed(out, **masks)
    print(f"object width: median {np.median(widths):.0%} of the frame width", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
