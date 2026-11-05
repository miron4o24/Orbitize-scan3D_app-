"""Фините детайли на модела: копчета, ръбове, гънки, релеф. Измерването и сливането (app/fuse.py) работят с клетки от
0,25% от размера, заглаждат ги и виждат тъмните гладки повърхности слабо, затова издадените копчета на дистанционно
излизаха съвсем плоски. AI дълбочината (app/depth.py) ги вижда отлично, ако ѝ се покаже само предметът: на целия кадър
обхватът ѝ отива за разликата между предмета и стаята и повърхността му остава равна, а на предмета сам, с фон, запълнен
с цветовете му, и с повече пиксели (PIXELS срещу 518 по дългата страна) всяко копче излиза релефно, а надписите остават
плоски (проверено на дистанционно, 2026-11-05). Тази дълбочина няма мащаб, затова от нея се взема само тясната лента на
детайла (BAND), с разлика STRENGTH от размера на модела, и се нанася върху готовата форма по нормалите ѝ, осреднена от
няколко кадъра, избрани така, че да покриват всяка страна. Голямата форма остава измерената. Работи като собствен
процес (стъпка "adding detail"): python relief.py <папка с кадри> <inputs.npz> <out.ply>. inputs.npz: names, R, t, K
(n,4), masks_<i>, mask_scale, vertices, faces, colors (модела в координати за печат), to_world (4x4 към света на
камерите), strength (множител на STRENGTH), views (колко кадъра). Записва out.ply и out.json. Функциите горе се ползват
и от модела от една снимка (app/photo3d.py)."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import fft, ndimage, sparse

if __package__:
    from . import depth as D
    from . import parallel as PAR
else:  # изпълнение като скрипт: импортира пакета по пътя му
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import depth as D
    from app import parallel as PAR

PIXELS = 705_600  # колко пиксела вижда мрежата от предмета (квадрат 840 x 840 по площ)
MOST_UP = 2.0  # малък в кадъра предмет се увеличава най-много толкова пъти
BASE_SIDE = 400  # повърхността под детайла се търси при толкова точки по дългата страна
NORMAL_MODEL = Path(os.environ.get("NORMAL_MODEL", Path(__file__).resolve().parent.parent / "tools" / "depth" / "moge2_vits_normal.onnx"))
MIN_TOKENS, MAX_TOKENS = 1200, 3600  # колко части от кадъра гледа MoGe-2
MIN_FACING_Z, MOST_SLOPE = 0.2, 2.0  # при очертанието посоката е почти успоредна на кадъра и наклонът е безкраен
RAISED = 0.008  # от дължината: под това издаденото се усилва; оценка, за проверка с линийка върху истински предмет
BOOST = 10.0
FINE_QUIET = 0.05  # от издаденото: равното остава равно
PATCH = 14  # стъпката на мрежата; входът е кратен на нея
# От размера на предмета в кадъра: по-широкото вече го има във формата. По-широки части (дръжката на дрелка) не бива да се
# надуват, а издадена площ, по-широка от това, излиза само като повдигнат ръб.
BAND = 0.08
EDGE = 0.025  # от размера на предмета в кадъра: при очертанието AI дълбочината е ненадеждна, затова там детайлът намалява
STRENGTH = 0.012  # от най-дългата страна на модела: колко се издига издаденото (при дистанционно от 15 см около 1,3 мм)
MOST = 0.025  # от най-дългата страна: повече от това не се мести никое място
PEAK = 1.2  # от разликата: нищо не стърчи повече (пръст или сянка при ръба даваха парцали по страните)
EVEN_BLUR = 0.02  # от размера на предмета в кадъра
EVEN_LEVEL, EVEN_POWER, EVEN_LIMITS = 0.45, 0.8, (0.5, 6.0)
SOFT = 0.002  # от размера на предмета в кадъра: колко се заглаждат стените на детайла
QUIET = 0.25  # от издаденото: по-малките разлики са вълни от мрежата, а не детайл на предмета
FLAT = 0.03  # лента под този дял от целия обхват на дълбочината е шум върху плоска повърхност и се усилва по-малко
VIEWS = 12
FEWER_VIEWS = 6  # когато времето не стига (app/ai.py)
WORKERS = 4  # кадри наведнъж: при този размер всяка мрежа заема около 1 GB
THREADS = 2
SUBDIVIDE_TO = 700_000  # стени преди нанасянето: достатъчно гъсти за копче от няколко милиметра
FACES = 400_000  # стени след него: опростяването пази релефа и маха излишното от плоските места
MIN_FACING = 0.35  # кадър не гласува за място, което вижда по-косо от това (косинус)
HIDDEN = 0.015  # от размера: място толкова зад видимата повърхност е скрито в кадъра
ENOUGH = 0.3  # толкова тегло (около един кадър право срещу мястото) дава пълната сила на детайла


def session(threads: int = THREADS):
    """Мрежата за детайла: MoGe-2, а ако файлът ѝ липсва, AI дълбочината от преди нея."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    model = NORMAL_MODEL if NORMAL_MODEL.exists() else D.MODEL
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def gives_normals(sess) -> bool:
    return len(sess.get_inputs()) == 2  # MoGe-2 иска и брой токени


def filled_background(crop: Image.Image, inside: np.ndarray) -> np.ndarray:
    """Кадърът с фон от най-близкия цвят на предмета, размазан, за да не заема вниманието на мрежата."""
    rgb = np.asarray(crop.convert("RGB"), np.float32)
    iy, ix = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    soft = 0.02 * max(inside.shape)
    filled = ndimage.gaussian_filter(rgb[iy, ix], (soft, soft, 0), truncate=2.5)
    return np.where(inside[..., None], rgb, filled)


def input_size(w: int, h: int) -> tuple[int, int]:
    k = min(MOST_UP, (PIXELS / (w * h)) ** 0.5)
    return max(PATCH, round(w * k / PATCH) * PATCH), max(PATCH, round(h * k / PATCH) * PATCH)


def one_piece(mask: np.ndarray, hole: float = 0.005) -> np.ndarray:
    """Най-голямото парче на маска, с малките ѝ дупки запълнени (истински отвор, като този на дръжка, остава)."""
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum_labels(mask, labels, np.arange(1, n + 1))
        mask = labels == 1 + int(np.argmax(sizes))
    holes, n = ndimage.label(~mask)
    if n > 1:
        sizes = ndimage.sum_labels(~mask, holes, np.arange(1, n + 1))
        mask = mask | np.isin(holes, np.flatnonzero(sizes < hole * mask.sum()) + 1)
    return mask


def object_depth(sess, crop: Image.Image, inside: np.ndarray, shifts: tuple[int, ...] = (0,)) -> np.ndarray:
    """AI дълбочината (по-голямо: по-близо) на предмета сам, при размера на изрязването. Фонът е запълнен с най-близкия
    цвят на предмета и размазан, за да не заема обхвата на мрежата. shifts: няколко изпълнения с вход, изместен с толкова
    пиксела: осреднени, те премахват бледите ивици на стъпката на мрежата там, където няма друг кадър да ги осредни."""
    rgb = filled_background(crop, inside)
    w, h = crop.size
    iw, ih = input_size(w, h)
    small = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((iw, ih), Image.BICUBIC), np.float32) / 255
    x = ((small - D.MEAN) / D.STD).astype(np.float32).transpose(2, 0, 1)
    name = sess.get_inputs()[0].name
    total = np.zeros((ih, iw), np.float32)
    for shift in shifts:
        y = np.roll(x, (shift, shift), (1, 2)) if shift else x
        out = sess.run(None, {name: y[None]})[0][0]
        total += np.roll(out, (-shift, -shift), (0, 1)) if shift else out
    return np.asarray(Image.fromarray(total / len(shifts)).resize((w, h), Image.BICUBIC))


def object_normals(sess, crop: Image.Image, inside: np.ndarray, shifts: tuple[int, ...] = (0,)) -> np.ndarray:
    """Посоките на повърхността от MoGe-2 при размера на изрязването (ос z: от камерата към предмета)."""
    w, h = crop.size
    iw, ih = input_size(w, h)
    small = np.asarray(Image.fromarray(filled_background(crop, inside).astype(np.uint8)).resize((iw, ih), Image.BICUBIC), np.float32) / 255
    tokens = int(np.clip(iw * ih / PATCH**2, MIN_TOKENS, MAX_TOKENS))
    x = small.transpose(2, 0, 1)
    n = np.zeros((ih, iw, 3), np.float32)
    for shift in shifts:  # вход, изместен с половин стъпка на мрежата: осреднени, махат решетката ѝ там, където няма други кадри
        y = np.roll(x, (shift, shift), (1, 2)) if shift else x
        out = sess.run(["normal"], {"image": y[None], "num_tokens": np.array(tokens, np.int64)})[0][0]
        n += np.roll(out, (-shift, -shift), (0, 1)) if shift else out
    n = np.stack([np.asarray(Image.fromarray(n[..., c]).resize((w, h), Image.BILINEAR)) for c in range(3)], -1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def height_from_normals(n: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Височината към камерата (в пиксели), чиито наклони най-добре пасват на посоките върху предмета: уравнение на Поасон,
    решено с косинусово преобразуване."""
    facing = np.maximum(np.abs(n[..., 2]), MIN_FACING_Z)
    p = np.clip(-n[..., 0] / facing, -MOST_SLOPE, MOST_SLOPE) * inside
    q = np.clip(-n[..., 1] / facing, -MOST_SLOPE, MOST_SLOPE) * inside
    div = np.zeros_like(p, np.float64)
    div[:, 1:] += p[:, 1:] - p[:, :-1]
    div[1:, :] += q[1:, :] - q[:-1, :]
    h, w = div.shape
    bottom = (2 * np.cos(np.pi * np.arange(w) / w) - 2)[None, :] + (2 * np.cos(np.pi * np.arange(h) / h) - 2)[:, None]
    bottom[0, 0] = 1.0
    spectrum = fft.dctn(div, type=2, norm="ortho") / bottom
    spectrum[0, 0] = 0.0
    return fft.idctn(spectrum, type=2, norm="ortho").astype(np.float32)


def relief_from_normals(n: np.ndarray, inside: np.ndarray, width: float) -> np.ndarray | None:
    """Детайлът от посоките на повърхността, в единици STRENGTH от дължината на предмета. Истинска пропорция, а не избран
    мащаб: при дрелка и фигура издаденото излиза около 0,8%. Гладка черна пластмаса излиза с твърде малки наклони (бутоните на
    дистанционно: 0,1%, видимо по-малко от истинските), затова слабият релеф се усилва, но най-много BOOST пъти, за да не се раздуват вълни."""
    inner = ndimage.binary_erosion(inside, iterations=max(2, int(EDGE * width)))
    if inner.sum() < 100:
        return None
    high = height_from_normals(n, inside)
    band = high - base_surface(high, inside, width)
    band = band - rim_of(band, inside, int(2 * BAND * width))
    lo, mid, hi = np.percentile(band[inner], [2, 50, 98])
    raised = max(float(hi - mid), 1e-6 * width)
    gain = np.clip(RAISED * width / raised, 1.0, BOOST)
    level = np.clip((band - mid) * gain / width / STRENGTH, -MOST / STRENGTH, MOST / STRENGTH)
    level = level * level**2 / (level**2 + FINE_QUIET**2)
    level = ndimage.gaussian_filter(level, max(1.0, SOFT * width))
    fade = np.clip(ndimage.distance_transform_edt(inside) / max(1.0, EDGE * width), 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    return np.where(inside, level * fade, 0).astype(np.float32)


def image_relief(sess, crop: Image.Image, inside: np.ndarray, width: float, alone: bool = False) -> np.ndarray | None:
    """Детайлът в един кадър. alone: няма други кадри, които да проверят AI дълбочината (важи само за нея)."""
    if gives_normals(sess):
        return relief_from_normals(object_normals(sess, crop, inside, (0, PATCH // 2) if alone else (0,)), inside, width)
    if alone:
        return agreed_relief(sess, crop, inside, width, shifts=(0, PATCH // 2))
    return unit_relief(object_depth(sess, crop, inside), inside, width)


def masked_blur(a: np.ndarray, inside: np.ndarray, sigma: float) -> np.ndarray:
    """Размазване само от точките вътре: фонът не влиза в средното. Широкото размазване се прави при по-малък размер:
    резултатът така или иначе е гладък, а времето пада с квадрата на намалението."""
    h, w = a.shape
    k = max(1, int(sigma / 4))
    top, bottom = np.where(inside, a, 0).astype(np.float32), inside.astype(np.float32)
    if k > 1:
        size = (max(1, w // k), max(1, h // k))
        top = np.asarray(Image.fromarray(top).resize(size, Image.BOX))
        bottom = np.asarray(Image.fromarray(bottom).resize(size, Image.BOX))
    top = ndimage.gaussian_filter(top, sigma / k, truncate=3.0)
    bottom = ndimage.gaussian_filter(bottom, sigma / k, truncate=3.0)
    low = top / np.maximum(bottom, 1e-3)
    return np.asarray(Image.fromarray(low).resize((w, h), Image.BILINEAR)) if k > 1 else low


def base_surface(d: np.ndarray, inside: np.ndarray, width: float) -> np.ndarray:
    """Повърхността под детайла: средата между горната и долната обвивка (морфологично отваряне и затваряне). Те следват
    плавните наклони докрай, затова заобляне към очертанието не оставя улей, какъвто оставяше обикновеното размазване, а
    копче или процеп, по-тесни от BAND, остават над или под нея. Проверено върху дистанционно, дрелка и фигура."""
    h, w = d.shape
    k = max(1, int(width / BASE_SIDE))
    iy, ix = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    small = np.asarray(Image.fromarray(d[iy, ix].astype(np.float32)).resize((max(1, w // k), max(1, h // k)), Image.BOX))
    size = max(3, int(BAND * width / k)) | 1
    middle = (ndimage.grey_opening(small, size=(size, size)) + ndimage.grey_closing(small, size=(size, size))) / 2
    return np.asarray(Image.fromarray(ndimage.gaussian_filter(middle, size / 3)).resize((w, h), Image.BILINEAR))  # без тераси


def rim_of(band: np.ndarray, inside: np.ndarray, reach: int) -> np.ndarray:
    """Заобляването към очертанието: AI дълбочината "закръгля" всеки предмет там, дори когато е плосък, а това заемаше
    почти целия обхват на детайла (при дистанционно копчетата получаваха една четвърт от него). Това е обичайната
    стойност на всяко разстояние от очертанието: копчетата заемат само част от всяко такова разстояние и остават."""
    far = ndimage.distance_transform_edt(inside)
    reach = max(1, min(reach, int(far.max())))
    rows = np.minimum(far, reach).astype(np.int32)
    typical = np.nan_to_num(ndimage.median(band, rows, np.arange(1, reach + 1)))
    smooth = ndimage.gaussian_filter1d(typical, 2.0, mode="nearest")
    return np.where(inside, smooth[np.maximum(rows, 1) - 1], 0).astype(np.float32)


def unit_relief(d: np.ndarray, inside: np.ndarray, width: float) -> np.ndarray | None:
    """Тясната лента на детайла от AI дълбочината: равната повърхност е 0, а разликата между ниските и високите места
    (2-ри и 98-и процентил) е 2, а издаденото над равното е около 1. Намалява към очертанието. None, когато предметът е твърде малък за това."""
    inner = ndimage.binary_erosion(inside, iterations=max(2, int(EDGE * width)))
    if inner.sum() < 100:
        return None
    band = d - base_surface(d, inside, width)
    band = band - rim_of(band, inside, int(2 * BAND * width))
    lo, mid, hi = np.percentile(band[inner], [2, 50, 98])
    spread, whole = float(hi - lo), float(np.ptp(d[inner]))
    if spread <= 1e-6 or whole <= 1e-6:
        return None
    # Мярката е за издаденото: дълбоки тесни процепи (между копчета, при дръжка) не бива да свиват копчетата до нищо.
    unit = max(float(hi - mid), 0.5 * spread)
    gain = min(1.0, spread / (FLAT * whole)) / unit
    fade = np.clip(ndimage.distance_transform_edt(inside) / max(1.0, EDGE * width), 0, 1)
    fade = fade * fade * (3 - 2 * fade)  # плавно: иначе детайлът свършва с видима линия успоредно на очертанието
    level = np.clip((band - mid) * gain, -PEAK, PEAK)
    level = level * level**2 / (level**2 + QUIET**2)  # плавно, защото рязък праг оставя плоски петна с видими граници
    level = ndimage.gaussian_filter(level.astype(np.float32), max(1.0, SOFT * width))  # отстрани копчетата не са зъбци
    return np.where(inside, level * fade, 0).astype(np.float32)


def evened(crop: Image.Image, inside: np.ndarray, width: float) -> Image.Image:
    """Копие, в което тъмният печат е почти колкото светъл като околното: яркостта е разделена на средната си в малък
    кръг, само от точките върху предмета, а цветовете остават."""
    a = np.asarray(crop.convert("RGB"), np.float32) / 255
    wide = masked_blur(a.mean(-1), inside, EVEN_BLUR * width)
    k = np.clip((EVEN_LEVEL / np.maximum(wide, 0.03)) ** EVEN_POWER, *EVEN_LIMITS)
    return Image.fromarray((np.clip(a * k[..., None], 0, 1) * 255).astype(np.uint8))


def agreed_relief(sess, crop: Image.Image, inside: np.ndarray, width: float, shifts: tuple[int, ...] = (0,)) -> np.ndarray | None:
    """Детайлът от един кадър, когато няма други да го проверят: AI дълбочината чете тъмен надпис или лого като релеф.
    Остава само височината, за която снимката и копието ѝ с изравнена яркост са съгласни: копчетата я имат и в двете, а
    логото и вълните по равното не."""
    plain = unit_relief(object_depth(sess, crop, inside, shifts), inside, width)
    other = unit_relief(object_depth(sess, evened(crop, inside, width), inside), inside, width)
    if plain is None or other is None:
        return plain
    both = np.where(np.sign(plain) == np.sign(other), np.sign(plain) * np.sqrt(np.abs(plain * other)), 0)  # плавно към нула
    return ndimage.gaussian_filter(both.astype(np.float32), max(1.0, SOFT * width))


class Scene:
    """Кадрите и моделът: коя точка кой кадър вижда."""

    def __init__(self, z, vertices: np.ndarray, normals: np.ndarray):
        self.names = [str(n) for n in z["names"]]
        self.R, self.t, self.K = z["R"], z["t"], z["K"]
        self.scale = int(z["mask_scale"])
        self.masks = [z[f"masks_{i}"] for i in range(len(self.names))]
        self.centers = -np.einsum("nji,nj->ni", self.R, self.t)
        self.V, self.N = vertices, normals
        self.hidden = HIDDEN * float(np.ptp(vertices, axis=0).max())

    def seen(self, i: int, every: int = 1, cell: int = 4):
        """За точките (всяка `every`-та) от кадър i: кои се виждат в маската му и къде, с дълбочина и косинус. Буфер на
        дълбочината от самите точки, в квадрати от `cell` пиксела."""
        V, N = self.V[::every], self.N[::every]
        x = PAR.turned(V, self.R[i]) + self.t[i]
        z = x[:, 2]
        front = z > 1e-9
        depth = np.where(front, z, 1.0)
        fx, fy, cx, cy = self.K[i]
        u, v = fx * x[:, 0] / depth + cx, fy * x[:, 1] / depth + cy
        m = self.masks[i]
        h, w = m.shape[0] * self.scale, m.shape[1] * self.scale
        ok = front & (u >= 0) & (v >= 0) & (u < w) & (v < h)
        gx, gy = np.clip(u / cell, 0, None).astype(np.int32), np.clip(v / cell, 0, None).astype(np.int32)
        idx = np.flatnonzero(ok)
        buf = np.full((h // cell + 2, w // cell + 2), np.inf, np.float32)
        order = idx[np.argsort(-z[idx])]  # най-близките се записват последни
        buf[gy[order], gx[order]] = z[order]
        buf = ndimage.minimum_filter(buf, size=3)
        ok[idx] &= z[idx] <= buf[gy[idx], gx[idx]] + self.hidden
        idx = np.flatnonzero(ok)
        ok[idx] &= m[(v[idx] / self.scale).astype(np.int32), (u[idx] / self.scale).astype(np.int32)] > 127
        ray = V - self.centers[i]
        cos = -np.einsum("ij,ij->i", N, ray) / np.maximum(np.linalg.norm(ray, axis=1), 1e-12)
        return ok & (cos >= MIN_FACING), u, v, z, cos


def choose_views(scene: Scene, wanted: int) -> list[int]:
    """Кадрите, които заедно виждат всяко място най-добре, всяко за предпочитане повече от веднъж: всеки следващ добавя
    най-много към корена от натрупаното качество (право срещу мястото и отблизо) за всички места."""
    every = max(1, len(scene.V) // 100_000)

    def quality(i):
        good, _, _, z, cos = scene.seen(i, every, cell=16)
        return np.where(good, cos * scene.K[i][0] / np.maximum(z, 1e-9), 0).astype(np.float32)

    q = np.stack(PAR.each(quality, range(len(scene.names))))
    q /= max(float(q.max()), 1e-9)
    total = np.zeros(q.shape[1], np.float32)
    chosen: list[int] = []
    for _ in range(min(wanted, len(q))):
        gain = (np.sqrt(total[None] + q) - np.sqrt(total[None])).sum(1)
        gain[chosen] = -1
        best = int(np.argmax(gain))
        chosen.append(best)
        total += q[best]
    return chosen


def view_relief(sess, scene: Scene, images: Path, i: int):
    """Детайлът в кадър i: (x0, y0 на изрязването, карта с единична разлика) или None."""
    with Image.open(images / scene.names[i]) as im:
        photo = im.convert("RGB")
    W, H = photo.size
    alpha = Image.fromarray(scene.masks[i]).resize((W, H), Image.BICUBIC)
    inside = one_piece(np.asarray(alpha) > 127)
    ys, xs = np.nonzero(inside)
    if len(ys) < 400:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    crop, part = photo.crop((x0, y0, x1, y1)), inside[y0:y1, x0:x1]
    rel = image_relief(sess, crop, part, max(x1 - x0, y1 - y0))
    return None if rel is None else (x0, y0, rel)


def along_normals(scene: Scene, images: Path, chosen: list[int]) -> np.ndarray:
    """За всяка точка преместването по нормалата ѝ при единична разлика: от кадрите, които я виждат, по-правите с
    по-голямо тегло, без онези, които много се разминават с другите."""
    local = threading.local()

    def relief_of(i):
        if not hasattr(local, "sess"):
            local.sess = session()
        return view_relief(local.sess, scene, images, i)

    maps = PAR.each(relief_of, chosen, WORKERS)
    print(f"relief maps from {sum(m is not None for m in maps)} of {len(chosen)} frames", flush=True)
    values = np.zeros((len(chosen), len(scene.V)), np.float32)
    weights = np.zeros_like(values)

    def sample(item):
        k, i, found = item
        if found is None:
            return
        x0, y0, rel = found
        good, u, v, _, cos = scene.seen(i)
        where = np.flatnonzero(good)
        got = ndimage.map_coordinates(rel, [v[where] - y0, u[where] - x0], order=1, mode="constant", cval=0.0)
        values[k, where] = got  # всяка карта е в свой мащаб (unit_relief), затова косият кадър не бива да намалява височината още веднъж
        weights[k, where] = cos[where] ** 2

    PAR.each(sample, list(zip(range(len(chosen)), chosen, maps)))
    total = np.maximum(weights.sum(0), 1e-6)
    mean = (values * weights).sum(0) / total
    away = np.abs(values - mean)
    typical = (away * weights).sum(0) / total
    weights *= away <= 1.5 * typical + 1e-3
    total = np.maximum(weights.sum(0), 1e-6)
    return (values * weights).sum(0) / total * np.clip(total / ENOUGH, 0, 1)


def smoothed(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Едно осредняване със съседните точки: маха разликите между отделни точки, а не копчетата."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n = len(values)
    near = sparse.coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    count = np.maximum(np.asarray(near.sum(1)).ravel(), 1)
    return 0.5 * values + 0.5 * (near.dot(values)) / count


def main(images: Path, inputs: Path, out: Path):
    if __package__:
        from . import mesh as M
    else:
        from app import mesh as M
    started = time.time()
    z = np.load(inputs)
    v, f, c = z["vertices"].astype(np.float64), z["faces"].astype(np.int64), z["colors"].astype(np.float32)
    while 2 * len(f) < SUBDIVIDE_TO:  # всяко деление учетворява стените: така не излизат над 2 милиона, които бавят всичко
        v, f, attributes = trimesh.remesh.subdivide(v, f, vertex_attributes={"c": c})
        c = attributes["c"]
    normals = np.asarray(trimesh.Trimesh(v, f, process=False).vertex_normals)
    to_world = z["to_world"]
    rot = to_world[:3, :3]
    unit = rot / np.linalg.norm(rot[:, 0])
    world, world_normals = v @ rot.T + to_world[:3, 3], normals @ unit.T
    scene = Scene(z, world, world_normals)
    chosen = choose_views(scene, int(z["views"]))
    print(f"frames for the detail: {[scene.names[i] for i in chosen]}, after {time.time() - started:.0f} s", flush=True)
    moved = smoothed(along_normals(scene, images, chosen), f)
    size = float(np.ptp(v, axis=0).max())
    moved = np.clip(moved * STRENGTH * float(z["strength"]) * size, -MOST * size, MOST * size)
    floor = float(v[:, 2].min())
    base = v[:, 2] <= floor + 1e-6 * size
    moved[base] = 0
    v = v + normals * moved[:, None]
    v[:, 2] = np.maximum(v[:, 2], floor)
    lo, hi = np.percentile(moved, [2, 98])
    print(f"detail: {lo:+.2f} .. {hi:+.2f} on a model of {size:.0f}, after {time.time() - started:.0f} s", flush=True)
    colors = np.clip(np.round(c), 0, 255).astype(np.uint8)
    faces = int(z["faces_out"]) if "faces_out" in z.files else FACES
    mesh = M.decimate(trimesh.Trimesh(v, f, vertex_colors=colors, process=False), faces)
    if not mesh.is_watertight or len(mesh.faces) > 1.2 * faces:
        raise SystemExit("the detailed surface did not stay closed")
    mesh.export(out)
    out.with_suffix(".json").write_text(json.dumps({"frames": len(chosen), "low": round(float(lo), 3), "high": round(float(hi), 3),
                                                    "faces": int(len(mesh.faces)), "seconds": round(time.time() - started)}))  # fmt: skip
    print(f"detail done: {len(mesh.faces)} faces, in {time.time() - started:.0f} s (threads: {os.cpu_count()})", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
