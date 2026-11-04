"""Маски на предмета: кои пиксели от всеки кадър показват предмета (сегментиране на открояващ се обект, IS-Net, на
процесора чрез ONNX). По два прохода за кадър. В целия кадър малък предмет се състезава с всичко останало, което се
откроява: колонка, заемаща 20% от ширината, се връщаше слята с бюрото и монитора зад нея в 12 от 80 кадъра, което
разваляше формата от очертанията и проверката на положенията на камерите. Затова първият проход само намира къде е
предметът (частта, най-близка до средата, където го държи приложението), а вторият сегментира близко изрязване около
него, където предметът е главното в изображението. Кадри, в които първият проход го е слял с фона, взимат мястото и
размера от предходните и следващите кадри. Работи като собствен процес (стъпка "finding the object"), затова паметта
на модела се освобождава след това и стъпката може да се отказва като другите инструменти: python segment.py <папка с
изображения> <out.npz> [папка за гъсти маски]. Файлът .npz съпоставя всяко име на изображение с маска uint8 (0..255)
при 1/MASK_SCALE от размера му."""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__:
    from . import parallel as PAR
else:  # изпълнение като скрипт: импортира пакета по пътя му
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parallel as PAR

MODEL = Path(os.environ.get("SEG_MODEL", Path(__file__).resolve().parent.parent / "tools" / "seg" / "isnet-general-use.onnx"))
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
MASK_SCALE = 4  # маските решават само вътре/вън; четвърт от размера на кадъра стига и ги държи малки
SIZE = 1024  # размерът на входа на модела
CROP = 2.4  # страна на квадратното изрязване на втория проход, в размери на предмета
MERGED = 1.8  # предмет от първия проход, толкова пъти по-голям от типичния, се е слял с фона
SMALL_HOLE = 0.02  # дупки в маската на предмет под този дял от нея се запълват


def session():
    import onnxruntime as ort

    if not MODEL.exists():
        raise SystemExit(f"Segmentation model missing: {MODEL}. Run the setup again (it downloads {MODEL_URL}).")
    return ort.InferenceSession(str(MODEL), PAR.onnx_options(), providers=["CPUExecutionProvider"])


def predict(sess, image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Картата на предмета 0..255 от модела при `size` (ширина, височина)."""
    a = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32)
    a = a / max(float(a.max()), 1e-6) - (0.485, 0.456, 0.406)
    pred = sess.run(None, {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
    return np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize(size, Image.BILINEAR))


def mask(sess, image: Image.Image) -> np.ndarray:
    """Един проход по целия кадър, при 1/MASK_SCALE от размера му."""
    return predict(sess, image, (max(1, image.size[0] // MASK_SCALE), max(1, image.size[1] // MASK_SCALE)))


def central_part(m: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    """Свързаната област на предмета, най-близка до (cx, cy) (пиксели на маската), без петънца; None, ако няма такава."""
    lab, n = ndimage.label(m > 127)
    if n == 0:
        return None
    areas = np.bincount(lab.ravel())[1:]
    big = np.flatnonzero(areas >= 0.1 * areas.max()) + 1
    ys, xs = np.nonzero(np.isin(lab, big))
    i = np.argmin((xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2)
    part = lab == lab[ys[i], xs[i]]
    # Запълват се малките дупки (петънца, пропуснати от мрежата), а истинските се пазят и формата от очертанията ги
    # показва: средата на пръстен беше 12% от него, отворът на дръжка на чаша 7%; при запълване и двете излизаха
    # плътни.
    holes = ndimage.binary_fill_holes(part) & ~part
    hl, n = ndimage.label(holes)
    if n:
        areas = np.bincount(hl.ravel())[1:]
        part |= np.isin(hl, np.flatnonzero(areas < SMALL_HOLE * (part.sum() + holes.sum())) + 1)
    return part


LOCATE_EVERY = 3  # първият проход търси предмета във всеки толкова-ти кадър
LOCATE_FROM = 16  # при по-малко кадри търси във всички


def locate(m: np.ndarray) -> tuple[float, float, float] | None:
    """Център x, y и размер (по-голямата страна на кутията) на предмета, най-близък до средата на маска от първия
    проход, в пиксели на маската."""
    part = central_part(m, m.shape[1] / 2, m.shape[0] / 2)
    if part is None:
        return None
    ys, xs = np.nonzero(part)
    return (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2, float(max(np.ptp(xs), np.ptp(ys)) + 1)


def track(found: list) -> list:
    """За всеки кадър (в реда на заснемане) къде да се изреже: предметът от първия проход или, за кадри, в които
    липсва или е слят с фона (много по-голям от обичайното), мястото и размерът, интерполирани от другите кадри."""
    sizes = [f[2] for f in found if f is not None]
    if not sizes:
        return [None] * len(found)
    typical = float(np.median(sizes))
    ok = [i for i, f in enumerate(found) if f is not None and f[2] <= MERGED * typical]
    idx = np.arange(len(found))
    cols = [np.interp(idx, ok, [found[i][k] for i in ok]) for k in range(3)]
    return [(float(cols[0][i]), float(cols[1][i]), float(cols[2][i])) for i in idx]


def crop_mask(sess, image: Image.Image, cx: float, cy: float, size: float) -> np.ndarray:
    """Втори проход: предметът, сегментиран в изрязване CROP пъти колкото размера му около (cx, cy), като маска на
    целия кадър при 1/MASK_SCALE (само областта на предмета; всичко друго е 0)."""
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
        # Запазва се мекият ръб около частта (маските се сравняват по-късно при 127 и се преоразмеряват).
        near = ndimage.binary_dilation(part, iterations=2)
        out[y0:y1, x0:x1] = np.where(part, np.maximum(m, 128), np.where(near, m, 0))
    return out


def object_width(m: np.ndarray) -> float:
    """Дял от ширината на кадъра, зает от предмета (0, ако не е намерен)."""
    xs = np.nonzero((m > 127).any(axis=0))[0]
    return (xs[-1] - xs[0] + 1) / m.shape[1] if len(xs) else 0.0


def dense_area(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Къде трябва да работи гъстото съпоставяне, при пълен размер на изображението: кутията на предмета, разширена с
    половината му размер от всяка страна (пръстен маса около него за изравняване и резерв за неточни маски). Всичко,
    ако предмет не е намерен."""
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
    """dense_masks: записва и <име на изображението>.mask.png там (0 = пропуснато) за --mask-path на OpenMVS."""
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if dense_masks:
        dense_masks.mkdir(parents=True, exist_ok=True)

    def first_pass(sess, f):
        with Image.open(f) as im:
            return locate(mask(sess, im))

    # Местоположението се променя плавно между кадрите: стига всеки трети, а track() запълва останалите.
    step = LOCATE_EVERY if len(files) >= LOCATE_FROM else 1
    looked = list(range(0, len(files), step))
    if looked[-1] != len(files) - 1:
        looked.append(len(files) - 1)
    found = [None] * len(files)
    for i, where in zip(looked, PAR.each_model(session, first_pass, [files[i] for i in looked])):
        found[i] = where
    crops = track(found)
    print(f"first pass: object near the middle in {sum(c is not None for c in found)} of {len(looked)} frames", flush=True)

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
