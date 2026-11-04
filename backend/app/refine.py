"""По-сигурни маски на предмета, след като камерите са известни: MobileSAM (сегментиране с подсказки), подсказван от
3D и проверяван спрямо обвивката от очертанията. IS-Net (app/segment.py) намира предмета във всеки кадър
самостоятелно и греши в голяма част от тях: включва стол зад дръжка на дрелка, етикет до фигура, ивица маса;
пропуска черен спусък, лице на фигура, основата ѝ. 3D знае по-добре. Обвивката (където повечето маски са съгласни)
показва къде е предметът във всеки кадър: изгледът ѝ дава на MobileSAM кутия, точки дълбоко в нея казват "предмет",
а измерени точки доста извън нея (маса, етикети, стая) казват "не е предмет". MobileSAM връща четири кандидата;
запазва се най-подобният на изгледа на обвивката, изрязан близо до него (включеният стол отпада) и съединен с
ядрото на обвивката (пропусната част се връща). При 8 кадъра от дрелка, в които IS-Net греши в 5: без стол, с
цялата дръжка, патронник и свредло, цялата батерия и ръбове, следващи изображението. Работи като собствен процес
(стъпка "sharpening the outlines"): python refine.py <папка с изображения> <inputs.npz> <out.npz> [<папка за кеш>].
inputs.npz: names (n,), R (n,3,3), t (n,3), K (n,4) fx fy cx cy (свят -> камера, като mesh.Views), masks_<i> (uint8
при 1/MASK_SCALE от кадъра), tight (k,3) световни точки в обвивката, loose (m,3) в по-свободна обвивка (половината
кадри са съгласни), deep (j,3) доста навътре в обвивката, outside (l,3) измерени световни точки доста извън
свободната обвивка, step (клетката на мрежата на обвивката, световни единици). out.npz: masks_<i> както на входа
(старата маска, където MobileSAM не намери нищо подходящо), used (n,) bool. Папка за кеш: какво е видяла бавната
половина на MobileSAM в изрязването на всеки кадър, за следващия кръг: само бързата му половина зависи от
променящите се подсказки."""

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
else:  # изпълнение като скрипт: импортира пакета по пътя му
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parallel as PAR
    from app import segment as S

MODEL_DIR = Path(os.environ.get("SAM_MODEL_DIR", Path(__file__).resolve().parent.parent / "tools" / "sam"))
ENCODER, DECODER = MODEL_DIR / "mobile_sam_image_encoder.onnx", MODEL_DIR / "sam_mask_decoder_multi.onnx"
SIZE = 1024  # входът на MobileSAM: дългата страна
CROP = 1.5  # страна на изрязването за MobileSAM, в размери на свободната обвивка
POSITIVE, NEGATIVE = 6, 8  # точки за подсказка
NEAR = 0.04  # от размера на предмета в кадъра: докъде може да стига маската на MobileSAM извън изгледа на свободната обвивка
CORE = 0.08  # и колко навътре в изгледа на обвивката предметът се приема за сигурен
MIN_FIT = 0.5  # най-малкото припокриване с изгледа на обвивката, за да се използва маската на MobileSAM
SPECK = 0.02  # части от маска под този дял от най-голямата ѝ част са петънца фон
SAME_CROP = 1.3  # изрязването от следващ кръг е същото (значението на изображението му вече е известно), ако е твърде голямо до толкова


def sessions():
    import onnxruntime as ort

    if not (ENCODER.exists() and DECODER.exists()):
        raise SystemExit(f"MobileSAM model missing in {MODEL_DIR}. Run install-task.ps1 again (it downloads it).")
    options = PAR.onnx_options()
    return (ort.InferenceSession(str(ENCODER), options, providers=["CPUExecutionProvider"]),
            ort.InferenceSession(str(DECODER), options, providers=["CPUExecutionProvider"]))  # fmt: skip


def embedding(enc, crop: Image.Image) -> np.ndarray:
    """Бавната половина на MobileSAM: какво вижда в изрязването, каквото и да го питат после."""
    W, H = crop.size
    s = SIZE / max(W, H)
    a = np.asarray(crop.convert("RGB").resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR), np.float32)
    return enc.run(None, {enc.get_inputs()[0].name: a})[0]


def candidates(dec, emb: np.ndarray, size: tuple[int, int], box, points, labels) -> list[np.ndarray]:
    """Четирите маски на MobileSAM (bool, размер на изрязването) за кутия (x0, y0, x1, y1) и точки (етикет 1 предмет,
    0 не е). emb: вграждането на изрязването (с размер `size`: ширина, височина)."""
    W, H = size
    s = SIZE / max(W, H)
    pts = np.array([box[:2], box[2:], *points], np.float32).reshape(-1, 2) * s
    feed = {
        "image_embeddings": emb,
        "point_coords": pts[None],
        "point_labels": np.array([2, 3, *labels], np.float32)[None],  # 2, 3: ъглите на кутията
        "mask_input": np.zeros((1, 1, 256, 256), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
        "orig_im_size": np.array([H, W], np.float32),
    }
    masks = dec.run(None, feed)[0]
    return [m > 0 for m in masks[0]]


def project(X: np.ndarray, R, t, K):
    """Позиции в пиксели (n, 2) и дълбочини на световни точки."""
    Xc = PAR.turned(X, R) + t
    z = Xc[:, 2]
    zs = np.where(z > 1e-9, z, np.inf)
    return np.stack([K[0] * Xc[:, 0] / zs + K[2], K[1] * Xc[:, 1] / zs + K[3]], 1), z


def spread(uv: np.ndarray, k: int) -> np.ndarray:
    """k от точките, разпръснати (подбор на най-далечна точка, започвайки от най-близката до средата им)."""
    if len(uv) <= k:
        return uv
    pick = [int(np.argmin(np.linalg.norm(uv - uv.mean(0), axis=1)))]
    dist = np.linalg.norm(uv - uv[pick[0]], axis=1)
    for _ in range(k - 1):
        pick.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(uv - uv[pick[-1]], axis=1))
    return uv[pick]


def view(uv: np.ndarray, z: np.ndarray, shape, radius: float) -> np.ndarray:
    """Къде точки (пиксели в изрязването) покриват изрязването, всяка нарисувана с ширина `radius` пиксела."""
    out = np.zeros(shape, bool)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (z > 0) & (u >= 0) & (v >= 0) & (u < shape[1]) & (v < shape[0])
    out[v[ok], u[ok]] = True
    r = int(np.ceil(radius))
    return ndimage.maximum_filter(out, size=2 * r + 1) if r > 0 else out


def tidy(m: np.ndarray) -> np.ndarray:
    """Без петънца (части под SPECK от най-голямата) и с малки запълнени дупки, като маските на segment.py."""
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
    """(Изрязването, вграждането), оставени от по-ранен кръг за този кадър, ако това изрязване още съдържа целия
    предмет от този кръг при приблизително същия размер; иначе None."""
    if saved is None or not saved.exists():
        return None
    with np.load(saved) as c:
        old, emb = tuple(int(v) for v in c["box"]), c["emb"]
    holds = old[0] <= max(0, x0) and old[1] <= max(0, y0) and old[2] >= min(bx[2], x1) and old[3] >= min(bx[3], y1)
    near = max(old[2] - old[0], old[3] - old[1]) <= SAME_CROP * max(bx[2] - bx[0], bx[3] - bx[1])
    return (old, emb) if holds and near else None


def refine_frame(enc, dec, image: Image.Image, R, t, K, z: dict, step: float, old: np.ndarray,
                 saved: Path | None = None) -> np.ndarray | None:  # fmt: skip
    """Маската на кадъра (bool, пълен размер) или None, ако обвивката не е в изгледа или MobileSAM не намери нищо
    подходящо. old: първата маска на кадъра (uint8, малка). saved: къде се пази вграждането на кадъра между кръговете."""
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
    radius = 0.5 * step * K[0] / zmed + 1  # половин ширина на клетка от обвивката в пиксели
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
    # Ядрото на обвивката се връща само където го е имало и първата маска: обвивката запълва и процепи между части,
    # през които този кадър вижда (масата между наметалото и крака на фигура), които и двете маски правилно пропускат.
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
