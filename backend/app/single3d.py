"""A 3D model from one frame (TripoSR, MIT: VAST-AI-Research/TripoSR, weights stabilityai/TripoSR), on the CPU: Pro's
AI model when the video's frames cannot be placed in 3D (app/ai.py). Picks the best frame by the object masks
(app/segment.py), cuts the object out onto grey as TripoSR expects, and turns its triplane into a coloured mesh by
marching cubes here (scikit-image, not torchmcubes, which needs compiling on Windows). What the frame does not show
is guessed.

Runs as its own process (step "ai modelling"), so torch's memory is freed afterwards and the step can be cancelled:
    python single3d.py <frames dir> <masks.npz> <out.ply>
Writes <out>.json with the frame used. TripoSR's code and weights are in tools/triposr (scripts/models.ps1)."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
TOOLS = Path(os.environ.get("TRIPOSR_DIR", HERE.parent / "tools" / "triposr"))
CODE = TOOLS / "code"  # the repository (tsr/)
WEIGHTS = TOOLS / "weights"  # config.yaml, model.ckpt
MASK_SCALE = 4  # app/segment.py's
FOREGROUND = 0.85  # share of the square the object fills (TripoSR's default)
RESOLUTION = 320  # marching cubes grid: finer than TripoSR's 256 for print detail
THRESHOLD = 25.0  # TripoSR's density level of the surface
CHUNK = 32768


def ready() -> bool:
    return (CODE / "tsr" / "system.py").exists() and (WEIGHTS / "model.ckpt").exists()


def sharpness(im: Image.Image) -> float:
    g = np.asarray(im.convert("L").resize((320, 320)), float)
    return float(ndimage.laplace(g).var())


# A mask that took in background (TripoSR then builds it as part of the object: a suitcase handle as "wings", a
# chair standing behind a figure's head as a black "leaf") has an unusual shape: cut to its box and scaled to a
# square, it overlaps the other frames' masks less. Only the more typical half are modelled from (the chair's frame
# 0.46 against a median 0.62, the handle's 0.46 and 0.41 against 0.73), and none much wider than the others.
TOO_WIDE = 1.3  # of the frames' median width (the handle: 78% of the frame against 37%)
SHAPE_SIDE = 64  # masks are compared at this size


def shape_of(on: np.ndarray) -> np.ndarray:
    """A mask cut to its box, centred in a square, at SHAPE_SIDE."""
    ys, xs = np.nonzero(on)
    crop = on[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = max(crop.shape)
    sq = np.zeros((side, side), np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    sq[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop * 255
    return np.asarray(Image.fromarray(sq).resize((SHAPE_SIDE, SHAPE_SIDE), Image.BILINEAR)) > 127


def best_frame(frames: list[Path], masks: dict) -> tuple[Path, np.ndarray]:
    """The frame to model from: the object's mask of a typical shape and width (see TOO_WIDE), the whole object in
    view (not cut by the frame's edge), big and sharp."""
    found = []
    for f in frames:
        m = masks.get(f.name)
        if m is None or (m > 127).sum() < 50:
            continue
        on = m > 127
        ys, xs = np.nonzero(on)
        h, w = on.shape
        cut = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2
        found.append((f, m, cut, max(np.ptp(xs), np.ptp(ys)) / min(h, w), np.ptp(xs) / w, shape_of(on)))
    if not found:
        raise SystemExit("The object was not found in any frame.")
    shapes = [x[5] for x in found]
    typical = np.array([np.mean([(a & b).sum() / max(1, (a | b).sum()) for j, b in enumerate(shapes) if j != i])
                        if len(shapes) > 1 else 1.0 for i, a in enumerate(shapes)])  # fmt: skip
    usual_shape = float(np.median(typical))
    usual_width = float(np.median([x[4] for x in found]))
    best, best_score = None, -1.0
    for (f, m, cut, size, wide, _), typ in zip(found, typical):
        with Image.open(f) as im:
            sharp = sharpness(im)
        odd = typ < usual_shape or wide > TOO_WIDE * usual_width
        score = sharp * min(size, 0.8) * (0.2 if cut else 1.0) * (0.01 if odd else 1.0)
        if score > best_score:
            best, best_score = (f, m), score
    return best


def cut_out(frame: Path, m: np.ndarray) -> Image.Image:
    """The object on grey, centred in a square it fills FOREGROUND of (TripoSR's input)."""
    with Image.open(frame) as im:
        rgb = np.asarray(im.convert("RGB"), np.float32) / 255
    alpha = np.asarray(Image.fromarray(m).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR), np.float32) / 255
    alpha = np.clip((alpha - 0.3) / 0.4, 0, 1)[..., None]  # the soft edge kept, faint background dropped
    img = rgb * alpha + 0.5 * (1 - alpha)
    ys, xs = np.nonzero(alpha[..., 0] > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) / FOREGROUND)
    out = np.full((side, side, 3), 0.5, np.float32)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    out[oy : oy + y1 - y0, ox : ox + x1 - x0] = img[y0:y1, x0:x1]
    return Image.fromarray((out * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)


def main(frames_dir: Path, masks_file: Path, out: Path):
    # TripoSR's code imports rembg and torchmcubes, neither used here: stand-ins (app/triposr_shims).
    sys.path[:0] = [str(CODE), str(HERE / "triposr_shims")]
    os.environ.setdefault("HF_HOME", str(TOOLS / "hf"))  # the image encoder's config.json, fetched once
    import torch
    from skimage import measure
    from tsr.system import TSR

    torch.set_num_threads(os.cpu_count() or 4)
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with np.load(masks_file) as z:
        masks = {k: z[k] for k in z.files}
    frame, m = best_frame(frames, masks)
    image = cut_out(frame, m)
    image.save(out.with_suffix(".input.png"))
    print(f"frame {frame.name}", flush=True)

    t = time.time()
    model = TSR.from_pretrained(str(WEIGHTS), config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(CHUNK)
    model.to("cpu")
    model.eval()
    print(f"model loaded in {time.time() - t:.0f} s", flush=True)
    t = time.time()
    with torch.no_grad():
        code = model([image], device="cpu")[0]
        r = float(model.renderer.cfg.radius)
        g = torch.linspace(-r, r, RESOLUTION)
        grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        density = model.renderer.query_triplane(model.decoder, grid, code)["density_act"]
        vol = density.reshape(RESOLUTION, RESOLUTION, RESOLUTION).numpy()
        del grid, density
        print(f"shape in {time.time() - t:.0f} s", flush=True)
        verts, faces, _, _ = measure.marching_cubes(vol, level=THRESHOLD)
        verts = verts / (RESOLUTION - 1) * 2 * r - r  # grid index -> TripoSR's coordinates (Z up)
        color = model.renderer.query_triplane(model.decoder, torch.from_numpy(verts).float(), code)["color"]
    colors = (np.clip(color.numpy(), 0, 1) * 255).astype(np.uint8)

    import trimesh

    mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors, process=True)
    trimesh.repair.fix_normals(mesh)
    mesh.export(out)
    out.with_suffix(".json").write_text(json.dumps({"frame": frame.name, "faces": int(len(mesh.faces))}))
    print(f"mesh: {len(mesh.faces)} faces, watertight {mesh.is_watertight}, in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
