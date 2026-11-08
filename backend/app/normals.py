"""Посоките на повърхността на предмета във всеки кадър (MoGe-2 small, ONNX на процесора) за сливането на дълбочината
(app/fuse.py, детайлният начин). Измерената между кадрите дълбочина (OpenMVS) я има само по ръбовете на шарките (очите
на фигура, линиите на косата, надписи), а гладките части (лицето, плешива глава, гладка пластмаса) остават празни или
шумни; AI дълбочината (app/depth.py) ги запълва, но вижда целия предмет при 518 пиксела и лицето е гладко петно без
очи и нос. MoGe-2 вижда изрязването с около 3600 части и дава посоките на повърхността: косата, носа, очите и гладкото
лице, както са заснети. Сливането ги превръща в дълбочина за кадъра, която следва тях отблизо и измерената дълбочина
отдалеч. Работи като собствен процес (стъпка "estimating detail"): python normals.py <папка с изображения>
<crops.npz> <names.txt> <out.npz> [през колко кадъра]. crops.npz: изрязванията от app/depth.py (box_i, inner_i); names.txt: имената на
кадрите (кадър i е изрязване i). out.npz: normal_i (h, w, 3) float16 за изрязване i, в координатите на камерата (x
надясно, y надолу, z напред; повърхността гледа към камерата: z < 0)."""

import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__:
    from . import parallel as PAR
else:  # изпълнение като скрипт: импортира пакета по пътя му
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parallel as PAR

MODEL = Path(
    os.environ.get("NORMAL_MODEL", Path(__file__).resolve().parent.parent / "tools" / "depth" / "moge2_vits_normal.onnx")
)
PATCH = 14  # стъпката на мрежата; входът е кратен на нея
PIXELS = 705_600  # колко пиксела вижда мрежата от изрязването (квадрат 840 x 840 по площ)
MOST_UP = 2.0  # малък в кадъра предмет се увеличава най-много толкова пъти
MIN_TOKENS, MAX_TOKENS = 1200, 3600  # колко части от изрязването гледа MoGe-2
# Посоките само от всеки толкова кадър: съседните видео кадри виждат почти едно и също, а сливането ги взима от всички
# кадри, виждащи мястото (при Фродо през кадър: същият детайл за половината време, около 2 минути по-малко).
EVERY = 2


def ready() -> bool:
    return MODEL.exists()


def session():
    import onnxruntime as ort

    return ort.InferenceSession(str(MODEL), PAR.onnx_options(), providers=["CPUExecutionProvider"])


def filled_background(rgb: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Изрязването с фон от най-близкия цвят на предмета, размазан, за да не заема вниманието на мрежата."""
    iy, ix = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    soft = 0.02 * max(inside.shape)
    filled = ndimage.gaussian_filter(rgb[iy, ix], (soft, soft, 0), truncate=2.5)
    return np.where(inside[..., None], rgb, filled)


def object_normals(sess, crop: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """(h, w, 3) посоките на повърхността при размера на изрязването."""
    h, w = inside.shape
    k = min(MOST_UP, (PIXELS / (w * h)) ** 0.5)
    iw, ih = max(PATCH, round(w * k / PATCH) * PATCH), max(PATCH, round(h * k / PATCH) * PATCH)
    rgb = filled_background(crop.astype(np.float32), inside)
    small = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((iw, ih), Image.BICUBIC), np.float32) / 255
    tokens = int(np.clip(iw * ih / PATCH**2, MIN_TOKENS, MAX_TOKENS))
    out = sess.run(["normal"], {"image": small.transpose(2, 0, 1)[None], "num_tokens": np.array(tokens, np.int64)})
    n = out[0][0]
    n = np.stack([np.asarray(Image.fromarray(n[..., c]).resize((w, h), Image.BILINEAR)) for c in range(3)], -1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def main(images: Path, crops_file: Path, names_file: Path, out: Path, every: int = EVERY) -> None:
    names = names_file.read_text(encoding="utf-8").split("\n")
    with np.load(crops_file) as crops:
        todo = [(i, names[i], tuple(int(v) for v in crops[f"box_{i}"]), crops[f"inner_{i}"])
                for i in range(len(names)) if f"box_{i}" in crops]  # fmt: skip
    todo = todo[:: max(1, every)]
    started = time.time()

    def one(sess, item):
        i, name, (x0, y0, x1, y1), inner = item
        with Image.open(images / name) as im:
            crop = np.asarray(im.convert("RGB"))[y0:y1, x0:x1]
        # inner е маската, свита с 2% от размера на предмета (app/depth.py predict): отново до очертанието.
        grow = max(2, int(0.02 * max(inner.shape) / 1.4))
        inside = ndimage.binary_dilation(inner, iterations=grow) if inner.any() else np.ones(inner.shape, bool)
        n = object_normals(sess, crop, inside)
        print(f"{len(done) + 1}/{len(todo)} {name}: {time.time() - started:.0f} s", flush=True)
        done.append(i)
        return i, n.astype(np.float16)

    done: list[int] = []
    found = PAR.each_model(session, one, todo)
    np.savez(out, **{f"normal_{i}": n for i, n in found})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
         int(sys.argv[5]) if len(sys.argv) > 5 else EVERY)  # fmt: skip
