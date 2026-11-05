"""3D модел от един кадър (TripoSR, MIT: VAST-AI-Research/TripoSR, тегла stabilityai/TripoSR) на процесора: AI моделът
на Pro, когато кадрите от видеото не могат да се разположат в 3D (app/ai.py). Избира най-добрия кадър по маските на
предмета (app/segment.py), изрязва предмета върху сив фон, както очаква TripoSR, и превръща триплана му в цветна
мрежа с marching cubes тук (scikit-image, не torchmcubes, който изисква компилиране на Windows). Това, което
кадърът не показва, се предполага. Работи като собствен процес (стъпка "ai modelling"), затова паметта на torch се
освобождава след това и стъпката може да се отказва: python single3d.py <папка с кадри> <masks.npz> <out.ply>.
Записва <out>.json с използвания кадър. Кодът и теглата на TripoSR са в tools/triposr (scripts/models.ps1)."""

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
CODE = TOOLS / "code"  # хранилището (tsr/)
WEIGHTS = TOOLS / "weights"  # config.yaml, model.ckpt
MASK_SCALE = 4  # от app/segment.py
FOREGROUND = 0.85  # дял от квадрата, запълнен от предмета (подразбиращото се за TripoSR)
RESOLUTION = 320  # мрежа за marching cubes: по-фина от 256 на TripoSR за детайл при печат
THRESHOLD = 25.0  # нивото на плътност на повърхността при TripoSR
CHUNK = 32768
CAM_DISTANCE, CAM_FOV = 1.9, 40.0  # изгледът, от който TripoSR приема, че е входът (tsr.system.TSR.render)


def ready() -> bool:
    return (CODE / "tsr" / "system.py").exists() and (WEIGHTS / "model.ckpt").exists()


def sharpness(im: Image.Image) -> float:
    g = np.asarray(im.convert("L").resize((320, 320)), float)
    return float(ndimage.laplace(g).var())


# Маска, включила фон (тогава TripoSR го изгражда като част от предмета: дръжка на куфар като "крила", стол зад
# главата на фигура като черно "листо"), има необичайна форма: изрязана до кутията си и мащабирана до квадрат, тя се
# припокрива по-малко с маските на другите кадри. Моделира се само от по-типичната половина (кадърът със стола 0,46
# спрямо медиана 0,62, тези с дръжката 0,46 и 0,41 спрямо 0,73) и от нито един много по-широк от другите.
TOO_WIDE = 1.3  # от медианната ширина на кадрите (дръжката: 78% от кадъра спрямо 37%)
SHAPE_SIDE = 64  # маските се сравняват при този размер


def shape_of(on: np.ndarray) -> np.ndarray:
    """Маска, изрязана до кутията си и центрирана в квадрат при SHAPE_SIDE."""
    ys, xs = np.nonzero(on)
    crop = on[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = max(crop.shape)
    sq = np.zeros((side, side), np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    sq[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop * 255
    return np.asarray(Image.fromarray(sq).resize((SHAPE_SIDE, SHAPE_SIDE), Image.BILINEAR)) > 127


def best_frame(frames: list[Path], masks: dict) -> tuple[Path, np.ndarray]:
    """Кадърът за моделиране: маска на предмета с типична форма и ширина (вижте TOO_WIDE), с целия предмет в изгледа
    (неотрязан от ръба на кадъра), голям и рязък."""
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


def cut_out(frame: Path, m: np.ndarray) -> tuple[Image.Image, tuple[int, int, int]]:
    """Предметът върху сив фон, центриран в квадрат, който запълва с дял FOREGROUND (входът на TripoSR), и къде е този
    квадрат в кадъра: (left, top, side)."""
    with Image.open(frame) as im:
        rgb = np.asarray(im.convert("RGB"), np.float32) / 255
    alpha = np.asarray(Image.fromarray(m).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR), np.float32) / 255
    alpha = np.clip((alpha - 0.3) / 0.4, 0, 1)[..., None]  # мекият ръб се пази, слабият фон отпада
    img = rgb * alpha + 0.5 * (1 - alpha)
    ys, xs = np.nonzero(alpha[..., 0] > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) / FOREGROUND)
    out = np.full((side, side, 3), 0.5, np.float32)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    out[oy : oy + y1 - y0, ox : ox + x1 - x0] = img[y0:y1, x0:x1]
    picture = Image.fromarray((out * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)
    return picture, (int(x0) - ox, int(y0) - oy, side)


def main(frames_dir: Path, masks_file: Path, out: Path, strength: float = 1.0):
    # Кодът на TripoSR импортира rembg и torchmcubes, които не се използват тук: заместители (app/triposr_shims).
    sys.path[:0] = [str(CODE), str(HERE / "triposr_shims")]
    os.environ.setdefault("HF_HOME", str(TOOLS / "hf"))  # config.json на кодировчика на изображения, изтеглен веднъж
    import torch
    from skimage import measure
    from tsr.system import TSR

    torch.set_num_threads(os.cpu_count() or 4)
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with np.load(masks_file) as z:
        masks = {k: z[k] for k in z.files}
    frame, m = best_frame(frames, masks)
    image, place = cut_out(frame, m)
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
        # Моделът от снимката: истинското ѝ очертание, детайл и цветове върху голямата форма на TripoSR
        # (app/photo3d.py). Ако това не стане, остава само моделът на TripoSR.
        sys.path.insert(0, str(HERE.parent))
        from app import photo3d

        def colour_at(points: np.ndarray) -> np.ndarray:
            return model.renderer.query_triplane(model.decoder, torch.from_numpy(points).float(), code)["color"].numpy()

        others = [(f, masks[f.name]) for f in frames if f.name in masks and f != frame]
        try:
            info = photo3d.build(frame, m, place, vol, r, THRESHOLD, CAM_DISTANCE, CAM_FOV, colour_at, out.parent, out, others,
                                 strength)  # fmt: skip
        except Exception as e:  # каквото и да е: моделът на TripoSR пак си заслужава
            print(f"two sided model failed: {e!r}", flush=True)
            info = None
        if info is not None:
            out.with_suffix(".json").write_text(json.dumps(info))
            return
        verts, faces, _, _ = measure.marching_cubes(vol, level=THRESHOLD)
        verts = verts / (RESOLUTION - 1) * 2 * r - r  # индекс в мрежата -> координатите на TripoSR (Z нагоре)
        color = model.renderer.query_triplane(model.decoder, torch.from_numpy(verts).float(), code)["color"]
    colors = (np.clip(color.numpy(), 0, 1) * 255).astype(np.uint8)

    import trimesh

    mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors, process=True)
    trimesh.repair.fix_normals(mesh)
    mesh.export(out)
    out.with_suffix(".json").write_text(json.dumps({"frame": frame.name, "faces": int(len(mesh.faces))}))
    print(f"mesh: {len(mesh.faces)} faces, watertight {mesh.is_watertight}, in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
