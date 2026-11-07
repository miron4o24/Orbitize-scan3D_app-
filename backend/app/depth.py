"""AI дълбочина за предмета във всеки кадър (Depth Anything V2 Small, ONNX на процесора) за издълбаване на обвивката
от очертанията. Очертанията дават форма, която никога не е твърде малка, но не могат да покажат вдлъбнатина
(решетка на колонка, потънала в ръба ѝ) и оставят купол над плосък връх, виждан само отгоре. Невронна мрежа за
дълбочина от един кадър вижда и двете с точност до неизвестен мащаб и отместване за всеки кадър, които се напасват
към известни 3D точки (гъсти точки или стените на обвивката). Точка от обвивката е празна, когато кадрите, които я
виждат, виждат повърхността зад нея средно с повече от CARVE_MARGIN. При осредняване малките грешки на кадрите се
компенсират; като точки, натрупани от 80 кадъра (както беше преди), те даваха груби, раирани повърхности с ръбове и
кутия излизаше с една трета по-плоска. Работи като собствен процес (стъпка "estimating depth"): python depth.py
<папка с изображения> <inputs.npz> <out.npz> [<crops.npz>]. inputs.npz: R (n,3,3), t (n,3), K (n,4) fx fy cx cy,
names (n,), masks_<i> (uint8, всякакъв размер), refs (m,3) световни точки върху повърхността на предмета,
ref_weights (m,), probes (k,3) световни точки вътре в обвивката, margin (световни единици), agree (нивото на
обвивката). out.npz: limit (k,) най-големият дял на обвивката за всяка проба (вижте mesh.hull_volume), fits:
напаснати кадри, used: дали това стига за издълбаване (иначе limit е изцяло 1). crops.npz (незадължително):
изрязването с дълбочина на всеки кадър за app/fuse.py: disp_<i>, box_<i> (x0, y0, x1, y1), inner_<i>."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

if __package__:
    from . import parallel as PAR
else:  # изпълнение като скрипт (fuse.py и motion.py го импортират като част от пакета)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parallel as PAR

MODEL = Path(
    os.environ.get(
        "DEPTH_MODEL", Path(__file__).resolve().parent.parent / "tools" / "depth" / "depth_anything_v2_small.onnx"
    )
)
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
SIZE = 518  # собствената разделителна способност на мрежата (дългата страна на изрязването на предмета)
MAX_SIDE = 1920  # кадрите първо се смаляват до това (вижте predict)
MAX_ERROR = 0.05  # медианна относителна грешка на дълбочината при напасване на кадър; по-лошите кадри отпадат
# От размера на предмета: колко зад точка кадрите трябва да виждат повърхността средно, за да я издълбаят.
# Напасванията се разминават с няколко десети от процента от разстоянието (0,2-1% при кутия).
CARVE_MARGIN = 0.03
MAX_CARVE = 0.3  # дял от обема на обвивката, който издълбаването може да премахне; повече означава грешно напасване
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def session():
    import onnxruntime as ort

    if not MODEL.exists():
        raise SystemExit(f"Depth model missing: {MODEL}. Run the setup again (it downloads {MODEL_URL}).")
    return ort.InferenceSession(str(MODEL), PAR.onnx_options(), providers=["CPUExecutionProvider"])


def disparity(sess, crop: Image.Image) -> np.ndarray:
    """Относителна обратна дълбочина (по-голямо = по-близо) при размера на изрязването."""
    w, h = crop.size
    s = SIZE / max(w, h)
    iw, ih = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
    a = (np.asarray(crop.convert("RGB").resize((iw, ih), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
    d = sess.run(None, {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    return np.asarray(Image.fromarray(d.astype(np.float32)).resize((w, h), Image.BILINEAR))


def front_depths(Xc: np.ndarray, w: np.ndarray, fx, fy, cx, cy, box, cell=6):
    """Проектира точки в пространството на камерата; за всеки квадрат от `cell` пиксела пази само най-близката
    (видимата повърхност). Връща u, v, z и теглото на запазените точки в кутията (x0, y0, x1, y1)."""
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
    """inv_z ~ a d + b, устойчиво (итеративно отпадат най-лошите 30%). Връща (a, b) и медианната относителна грешка на
    дълбочината. (Пробван беше наклон през изрязването: по-гладки слоеве, но точките пасваха по-зле на очертанията.)
    Най-малките квадрати са написани изрично, защото това работи върху много кадри наведнъж (app/parallel.py)."""
    d, inv_z, w = np.asarray(d, np.float64), np.asarray(inv_z, np.float64), np.asarray(w, np.float64)
    keep = np.ones(len(d), bool)
    coef = np.zeros(2)
    for _ in range(6):
        dk, yk, wk = d[keep], inv_z[keep], w[keep]
        s, sd, sy = wk.sum(), (wk * dk).sum(), (wk * yk).sum()
        sdd, sdy = (wk * dk * dk).sum(), (wk * dk * yk).sum()
        det = s * sdd - sd * sd
        # Всичко е с еднаква дълбочина: няма наклон за намиране (a <= 0 значи за извикващите, че няма напасване).
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
    """Изрязването на предмета от един кадър: какво вижда мрежата и какво връща. Масивите покриват само кутията."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 в изображението
    disp: np.ndarray  # (h, w) изход на мрежата
    inner: np.ndarray  # (h, w) пиксели на предмета доста навътре от очертанието
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray


def predict(sess, image: Image.Image, mask: np.ndarray, R, t, K) -> Frame | None:
    # Кадри над MAX_SIDE (4K видео, пазено за текстурата) се смаляват заедно с камерата си: мрежата така или иначе
    # вижда 518 пиксела, а изрязването на всеки напаснат кадър остава в паметта.
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
    # Ръбовете на дълбочината се размазват към фона: използват се само пиксели доста навътре от очертанието.
    erode = max(2, int(0.02 * max(x1 - x0, y1 - y0)))
    inner = ndimage.binary_erosion(m[by0:by1, bx0:bx1], iterations=erode)
    return Frame(box, disparity(sess, image.crop(box)), inner, R, t, K)


def fit(f: Frame, refs: np.ndarray, ref_w: np.ndarray):
    """(a, b) на кадъра от опорните точки, които вижда. None, ако не пасват."""
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
    """Изрязването с дълбочина на всеки кадър и мащабът му (a, b), напаснат към известните точки, за напаснатите
    кадри. seen: ако е дадено, получава изрязването на всеки кадър с предмет по индекс (напаснат или не)."""

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
    """Дали дълбочината пасва на достатъчно кадри, за да ѝ се вярва (истинска колонка: 79 от 80; едноцветни изкуствени
    изображения: 14)."""
    return len(fits) >= max(3, n_frames / 2)


def carve_limit(fits: list, points: np.ndarray, margin: float, agree: float) -> np.ndarray:
    """За световни точки в обвивката: най-големият им дял в нея. Определя се от това колко зад всяка точка кадрите,
    които я виждат (доста навътре от очертанието), виждат повърхността средно, всеки ограничен до 2 граници, за да
    не може един лош кадър да издълбае сам: `agree` (нивото на обвивката) при границата, плътно пред нея и спадане
    до празно отвъд нея. Гласуват и кадрите, виждащи повърхността пред точка (за разлика от измерените точки на
    mesh.free_space_limit): AI дълбочината е твърде груба, за да се издълбава само с гледащите кадри, което
    разваляше вътрешността на чаша. Точки, видени от под 3 кадъра, запазват 1. Гладко между точките, така че и
    издълбаната повърхност е гладка (да/не за всеки воксел оставяше тераси)."""
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
        far = 1 / np.maximum(inv, 1e-9) - z[ok]  # колко отвъд точката този кадър вижда повърхността
        behind[ok] += np.where(inv > 1e-9, np.clip(far, -2 * margin, 2 * margin), 2 * margin)
    mean = behind / np.maximum(seen, 1)
    share = agree + (margin - mean) * (1 - agree) / (0.2 * margin)
    return np.where(seen >= 3, np.clip(share, 0, 1), 1.0)


def save_crops(path: Path, seen: dict) -> None:
    """Изрязванията с дълбочина на кадрите за app/fuse.py (който ги напасва към картите на дълбочината на OpenMVS)."""
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
