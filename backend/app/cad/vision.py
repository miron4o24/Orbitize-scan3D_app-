"""Снимка -> очертанието на детайла в пиксели на снимката.

Снимките са направени с телефона успоредно на масата (приложението чака нивелира), затова изображението е почти
ортогонална проекция. Коя страна на детайла е на снимката, как е завъртяна и колко мм е пикселът, се решава по-късно
(photos.py), от всички снимки и от измерените общи размери. Изрязването е IS-Net (app/segment.py, без външни
услуги) на два прохода: първо къде е детайлът в целия кадър, после близко изрязване около него в пълна
разделителна способност, за точен ръб. MobileSAM (сегментиране с подсказка) дава второ мнение: когато IS-Net е
взел и ръката, която държи детайла, или повърхността под него, се взима неговото; и изрязва детайла по докосване, когато
потребителят поправя очертанието.

Отворите през детайла: мрежата понякога ги запълва, затова се търсят и по цвят: петно вътре в детайла с цвета на
фона около него, с компактна форма (виждаме масата през отвора)."""

import math
import threading

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure

from .. import segment as S

SIDE = 2400  # дългата страна, до която се смалява снимката
MARGIN = 0.3  # поле около детайла при второто изрязване, в размери на детайла
MIN_HOLE = 0.0015  # отвор под този дял от детайла е шум
COLOR_HOLE_K = 2.5  # колко близо до цвета на фона (в разсейвания на фона) е пиксел "през отвор"


def load(path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((SIDE, SIDE), Image.LANCZOS)
        return im


def part_near_centre(m: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    areas = np.bincount(lab.ravel())[1:]
    big = np.flatnonzero(areas >= 0.1 * areas.max()) + 1
    best, dist = None, np.inf
    for k in big:
        ys, xs = np.nonzero(lab == k)
        d = math.hypot(xs.mean() - cx, ys.mean() - cy)
        if d < dist:
            best, dist = k, d
    return lab == best


def cutout(sess, im: Image.Image):
    """(мека маска 0..1 и твърда маска на детайла с отворите в изрязването, отместването на изрязването)."""
    W, H = im.size
    small = S.predict(sess, im, (max(1, W // 4), max(1, H // 4))) > 127
    part = part_near_centre(small, small.shape[1] / 2, small.shape[0] / 2)
    if part is None:
        raise ValueError("no object")
    ys, xs = np.nonzero(part)
    x0, x1, y0, y1 = xs.min() * 4, (xs.max() + 1) * 4, ys.min() * 4, (ys.max() + 1) * 4
    pad = MARGIN * max(x1 - x0, y1 - y0)
    box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(W, int(x1 + pad)), min(H, int(y1 + pad)))
    crop = im.crop(box)
    soft = S.predict(sess, crop, crop.size).astype(np.float32) / 255.0
    field_, solid = finish(soft, crop)
    return field_, solid, box


def finish(soft: np.ndarray, crop: Image.Image, seed: tuple[float, float] | None = None):
    """Мека маска 0..1 в изрязването -> (поле за очертанието с точност под пиксел, твърда маска без отворите).
    seed: точка (пиксели на изрязването) върху детайла; иначе средата."""
    sx, sy = seed if seed is not None else (crop.size[0] / 2, crop.size[1] / 2)
    hard = part_near_centre(soft > 0.5, sx, sy)
    if hard is None:
        raise ValueError("no object")
    filled = ndimage.binary_fill_holes(hard)
    holes = filled & ~hard
    lab, n = ndimage.label(holes)
    area = filled.sum()
    for k in range(1, n + 1):
        if (lab == k).sum() < MIN_HOLE * area:
            holes[lab == k] = False
    holes |= color_holes(np.asarray(crop, np.float32), filled)
    solid = filled & ~holes
    if solid.sum() < 100:
        raise ValueError("no object")
    # Мекият ръб навън дава ръба с точност под пиксел; вътре в отворите и извън детайла е 0.
    near = ndimage.binary_dilation(filled, iterations=3)
    field_ = np.where(near, soft, 0.0)
    field_[holes] = np.minimum(field_[holes], 0.0)
    field_ = np.where(solid & (field_ < 0.5), 0.75, field_)  # плътното вътре остава вътре
    field_ = ndimage.gaussian_filter(np.where(holes, 0.0, field_), 0.6)
    return field_, solid


# --- MobileSAM: детайлът под точка ---------------------------------------------------------------------------------
# IS-Net намира "това, което се откроява": детайл, хванат с пръсти, излиза заедно с ръката (до ръба на кадъра), а
# понякога и с телефона или листа под него. MobileSAM (app/refine.py, вече на сървъра за сканирането) изрязва
# предмета под посочена точка: второ мнение за всяка снимка (second_opinion: точката от изрязаното на IS-Net, най-близо
# до средата на кадъра) и точките, които потребителят докосне при проверката на очертанията, когато е грешно.
_sam = None
_sam_lock = threading.Lock()


def sam():
    """(енкодер, декодер) на MobileSAM, заредени веднъж; None без файловете на модела."""
    global _sam
    from .. import refine as RF

    with _sam_lock:
        if _sam is None and RF.ENCODER.exists() and RF.DECODER.exists():
            _sam = RF.sessions()
        return _sam


def sam_candidates(models, image: Image.Image, points: list, box=None) -> list[tuple[np.ndarray, float]]:
    """Кандидатите на MobileSAM (логити с размера на изображението, оценка) за точки върху предмета (пиксели) и
    кутия около него, ако е известна."""
    from .. import refine as RF

    enc, dec = models
    emb = RF.embedding(enc, image)
    W, H = image.size
    k = RF.SIZE / max(W, H)
    coords = [tuple(p[:2]) for p in points]
    labels = [1.0 if len(p) < 3 or p[2] else 0.0 for p in points]
    if box is not None:
        coords, labels = [box[:2], box[2:]] + coords, [2.0, 3.0] + labels
    else:  # без кутия: празна точка с етикет -1 (както в примера на SAM за ONNX)
        coords, labels = coords + [(0.0, 0.0)], labels + [-1.0]
    feed = {"image_embeddings": emb, "point_coords": (np.array(coords, np.float32).reshape(-1, 2) * k)[None],
            "point_labels": np.array(labels, np.float32)[None], "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32), "orig_im_size": np.array([H, W], np.float32)}  # fmt: skip
    masks, scores = dec.run(None, feed)[:2]
    return [(masks[0][i], float(scores[0][i])) for i in range(masks.shape[1])]


BIG = 0.8  # кандидат, по-широк или по-висок от толкова от кадъра, е повърхността под детайла (телефон, лист)
SURE = 0.8  # оценката на MobileSAM (очакваното припокриване), над която кандидатът е цял предмет, не догадка


def choose(cands: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """От кандидатите под точката (маска, оценка): не тези до ръба на кадъра или почти колкото него (ръката,
    телефонът или листът под детайла), ако има други; от останалите най-големият сигурен (целият детайл, а не
    надписът, логото или шевът му, които MobileSAM също предлага), иначе най-сигурният."""
    ok = [c for c in cands if not edge_mask(c[0]) and span(c[0]) <= BIG]
    ok = ok or [c for c in cands if not edge_mask(c[0])] or cands
    sure = [c for c in ok if c[1] >= SURE]
    if sure:
        return max(sure, key=lambda c: c[0].sum())[0]
    return max(ok, key=lambda c: c[1])[0]


def span(m: np.ndarray) -> float:
    ys, xs = np.nonzero(m)
    return max((xs.max() - xs.min() + 1) / m.shape[1], (ys.max() - ys.min() + 1) / m.shape[0])


EDGE = 0.01  # кандидат на MobileSAM толкова близо до ръба на кадъра (от размера му) стига до ръба: маските му
# идват от мрежа 256 x 256 и спират на няколко пиксела от ръба (ръка, влизаща в кадъра)


def edge_mask(m: np.ndarray) -> bool:
    e = max(3, round(EDGE * max(m.shape)))
    return bool(m[:e].any() or m[-e:].any() or m[:, :e].any() or m[:, -e:].any())


def holds(m: np.ndarray, points: list) -> bool:
    """Точките върху детайла (x, y или x, y, 1) са в маската, а "не е детайлът" (x, y, 0) - извън нея."""
    h, w = m.shape
    for p in points:
        inside = bool(m[min(h - 1, max(0, int(p[1]))), min(w - 1, max(0, int(p[0])))])
        if inside != (len(p) < 3 or bool(p[2])):
            return False
    return True


def sam_cutout(models, im: Image.Image, points: list):
    """Детайлът под точките (пиксели на снимката), изрязан с MobileSAM: първо в цялата снимка (къде е), после
    отблизо около него, за точен ръб. Като cutout: (поле, твърда маска, изрязване). ValueError("no object")."""
    whole = [(lg > 0, sc) for lg, sc in sam_candidates(models, im, points)]
    whole = [(m, sc) for m, sc in whole if m.any() and holds(m, points)]
    if not whole:
        raise ValueError("no object")
    coarse = choose(whole)
    W, H = im.size
    ys, xs = np.nonzero(coarse)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = MARGIN * max(x1 - x0, y1 - y0)
    box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(W, int(x1 + pad)), min(H, int(y1 + pad)))
    crop = im.crop(box)
    local = [(p[0] - box[0], p[1] - box[1], *p[2:]) for p in points]
    near = 0.03 * max(x1 - x0, y1 - y0)
    around = (x0 - box[0] - near, y0 - box[1] - near, x1 - box[0] + near, y1 - box[1] + near)
    first = coarse[box[1]:box[3], box[0]:box[2]]
    best, fit = None, -1.0
    for lg, sc in sam_candidates(models, crop, local, around):
        m = lg > 0
        if not m.any() or not holds(m, local):
            continue
        iou = float((m & first).sum() / max(1, (m | first).sum()))
        if iou + 0.1 * sc > fit:
            best, fit = lg, iou + 0.1 * sc
    if best is None:
        raise ValueError("no object")
    soft = 1.0 / (1.0 + np.exp(-np.clip(best, -30, 30)))
    seed = next((p for p in local if len(p) < 3 or p[2]), local[0])
    field_, solid = finish(soft.astype(np.float32), crop, (seed[0], seed[1]))
    return field_, solid, box


def color_holes(rgb: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Петна вътре в детайла с цвета на фона около него: отвори, които мрежата е запълнила."""
    ring = ndimage.binary_dilation(filled, iterations=25) & ~ndimage.binary_dilation(filled, iterations=6)
    inner = ndimage.binary_erosion(filled, iterations=4)
    if ring.sum() < 200 or inner.sum() < 200:
        return np.zeros_like(filled)
    bg = rgb[ring]
    med = np.median(bg, axis=0)
    spread = np.median(np.abs(bg - med), axis=0) * 1.4826 + 4.0
    dist = np.sqrt((((rgb - med) / spread) ** 2).sum(-1))
    # Частта не бива да е като фона: тогава цветът не казва нищо (бял детайл на бял лист).
    if np.median(dist[inner]) < 2 * COLOR_HOLE_K:
        return np.zeros_like(filled)
    cand = inner & (dist < COLOR_HOLE_K)
    cand = ndimage.binary_opening(cand, iterations=2)
    lab, n = ndimage.label(cand)
    out = np.zeros_like(filled)
    area = filled.sum()
    for k in range(1, n + 1):
        blob = lab == k
        a = blob.sum()
        if a < MIN_HOLE * area:
            continue
        ys, xs = np.nonzero(blob)
        hull = Polygon(np.c_[xs, ys]).convex_hull if a > 2 else None
        if hull is None or hull.area <= 0 or a / hull.area < 0.8:
            continue
        out |= ndimage.binary_closing(blob, iterations=2)
    return out


def contours(field_: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Външното очертание и отворите (пиксели x, y на изрязването) при 0,5."""
    cs = [c[:, ::-1] for c in measure.find_contours(np.pad(field_, 1), 0.5)]
    cs = [c - 1 for c in cs if len(c) >= 8]
    polys = [(c, Polygon(c)) for c in cs]
    polys = [(c, p.buffer(0) if not p.is_valid else p) for c, p in polys]
    outer_c, outer_p = max(polys, key=lambda cp: cp[1].area)
    holes = [c for c, p in polys if c is not outer_c and outer_p.contains(p.representative_point())
             and p.area >= MIN_HOLE * outer_p.area]  # fmt: skip
    return outer_c, holes


def touches(o: np.ndarray, size, edge: float = 3) -> bool:
    return bool(o[:, 0].min() < edge or o[:, 1].min() < edge or o[:, 0].max() > size[0] - edge
                or o[:, 1].max() > size[1] - edge)  # fmt: skip


SAM_SWITCH = 0.9  # очертание на MobileSAM вътре в това на IS-Net и с по-малко припокриване от това: IS-Net е взел още
SAM_INSIDE = 0.08  # "вътре": най-много толкова от очертанието на MobileSAM е извън това на IS-Net


def second_opinion(im: Image.Image, solid: np.ndarray, off: np.ndarray, outline: np.ndarray):
    """Второ мнение от MobileSAM за очертанието на IS-Net (outline: пиксели на снимката): предметът под най-дълбоката
    точка от изрязаното, най-близо до средата на кадъра (там приложението държи детайла; най-дълбоката точка
    беше между два пръста). Взима се вместо това на IS-Net, когато то стига до
    ръба на кадъра (ръката, която държи детайла, или повърхността под него) или когато очертанието на MobileSAM е
    ясно по-малко и вътре в него (IS-Net е взел и пръстите или сянката до детайла). На истинските снимки на флашка
    (без пръсти до нея) двете съвпадат 96-99%. None: остава IS-Net; списък: остава IS-Net, а това са отворите на
    MobileSAM (пиксели на снимката) за through_holes."""
    models = sam()
    if models is None:
        return None
    depth = ndimage.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1]
    ys, xs = np.nonzero(depth >= min(4.0, float(depth.max())))  # не на самия ръб
    cx, cy = im.size[0] / 2 - off[0], im.size[1] / 2 - off[1]
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    x, y = xs[i], ys[i]
    try:
        field_, hard, box = sam_cutout(models, im, [(x + off[0], y + off[1])])
        outer, holes = contours(field_)
    except ValueError:
        return None
    off2 = np.array([box[0], box[1]], float)
    if touches(outer + off2, im.size):
        return None
    a, b = Polygon(outline).buffer(0), Polygon(outer + off2).buffer(0)
    if a.area <= 0 or b.area <= 0:
        return None
    inside = b.difference(a).area <= SAM_INSIDE * b.area
    iou = a.intersection(b).area / a.union(b).area
    if touches(outline, im.size) or (inside and iou < SAM_SWITCH):
        return field_, hard, box, outer, holes, off2
    if iou >= SAM_SAME:
        return [h + off2 for h in holes]
    return None


SAM_SAME = 0.85  # очертанията на IS-Net и MobileSAM са на един и същ предмет: отворите на MobileSAM се проверяват


def through_holes(im: Image.Image, solid: np.ndarray, off: np.ndarray, holes: list, have: list) -> list:
    """Отворите на MobileSAM (пиксели на снимката), които IS-Net е запълнил: ключ с отвор за халката, сив като дървото
    под него, при което и отворите по цвят (color_holes) не се търсят. Отворът е истински, ако е вътре в детайла на
    IS-Net, не е вече намерен и през него се вижда повърхността: средният му цвят и оттенъкът (без яркостта) са
    по-близо до фона около детайла, отколкото до детайла (надпис или сянка върху детайла са с неговия оттенък или
    тъмни). Връща ги в пиксели на изрязването на IS-Net."""
    from skimage import draw

    if not holes:
        return []
    rgb = np.asarray(im.crop((int(off[0]), int(off[1]), int(off[0]) + solid.shape[1], int(off[1]) + solid.shape[0])),
                     np.float32)  # fmt: skip
    filled = ndimage.binary_fill_holes(solid)
    ring = ndimage.binary_dilation(filled, iterations=25) & ~ndimage.binary_dilation(filled, iterations=6)
    body = ndimage.binary_erosion(solid, iterations=4)
    if ring.sum() < 200 or body.sum() < 200:
        return []
    bg, part = np.median(rgb[ring], 0), np.median(rgb[body], 0)
    spread = np.median(np.abs(rgb[ring] - bg), 0) * 1.4826 + 4.0
    found = [Polygon(h).buffer(0) for h in have]
    out = []
    for h in holes:
        local = np.asarray(h, float) - off
        poly = Polygon(local).buffer(0)
        if poly.area < MIN_HOLE * filled.sum() or any(poly.intersects(f) for f in found):
            continue
        rr, cc = draw.polygon(local[:, 1], local[:, 0], solid.shape)
        m = np.zeros(solid.shape, bool)
        m[rr, cc] = True
        if (m & filled).sum() < 0.9 * m.sum():
            continue
        core = ndimage.binary_erosion(m, iterations=3)
        px = rgb[core if core.sum() > 30 else m]
        if len(px) < 30:
            continue
        med = np.median(px, 0)

        def chroma(c):
            return c - c.mean()

        near_bg = np.linalg.norm((med - bg) / spread) < np.linalg.norm((med - part) / spread)
        hue_bg = np.linalg.norm(chroma(med) - chroma(bg)) < np.linalg.norm(chroma(med) - chroma(part))
        if near_bg and hue_bg:
            out.append(local)
    return out


def segment(sess, path, points: list | None = None) -> dict:
    """Очертанието на детайла на снимката: {"outer": [[x, y], ...], "holes": [...], "size": [w, h]} в пиксели на
    смалената снимка (load), x надясно, y надолу. points: докоснати точки върху детайла (0..1 от ширината и
    височината на снимката), когато автоматичното очертание е грешно: тогава го изрязва MobileSAM. "by": кой е
    изрязал детайла (isnet; sam - IS-Net беше взел и ръката или повърхността: second_opinion; tap - по
    докосване). ValueError: няма детайл
    ("no object"), не се събира в кадъра ("cut") или липсва MobileSAM за докосване ("no sam")."""
    im = load(path)
    by = "isnet"
    if points:
        models = sam()
        if models is None:
            raise ValueError("no sam")
        px = [(float(p[0]) * im.size[0], float(p[1]) * im.size[1], *([int(bool(p[2]))] if len(p) > 2 else []))
              for p in points]  # fmt: skip
        if not any(len(q) < 3 or q[2] for q in px):
            raise ValueError("no object")
        field_, solid, box = sam_cutout(models, im, px)
        by = "tap"
    else:
        field_, solid, box = cutout(sess, im)
    outer, holes = contours(field_)
    off = np.array([box[0], box[1]], float)
    if by == "isnet":
        alt = second_opinion(im, solid, off, outer + off)
        if isinstance(alt, tuple):
            field_, solid, box, outer, holes, off = alt
            by = "sam"
        elif alt:  # същият предмет: отворите, които вижда само MobileSAM
            holes = holes + through_holes(im, solid, off, alt, holes)
    if touches(outer + off, im.size):
        raise ValueError("cut")
    return {"outer": np.round(outer + off, 2).tolist(), "holes": [np.round(h + off, 2).tolist() for h in holes],
            "size": list(im.size), "f35": focal35(path), "lines": line_profiles(im, outer + off),
            "looks": looks(im, outer + off), "profiles": PROFILES,
            **marks_of(np.asarray(im.crop(box), np.float32), solid, off), "by": by}  # fmt: skip


def marks_of(rgb: np.ndarray, solid: np.ndarray, off: np.ndarray) -> dict:
    """Тъмните петна (dark_marks) в пиксели на снимката и колко са тъмни (спрямо детайла) и цветни."""
    found = dark_marks(rgb, solid)
    return {"marks": [np.round(m + off, 2).tolist() for m, _ in found], "mark_info": [i for _, i in found]}


LINE_BINS = 200
PROFILES = 2  # версията на lines и looks (осите на снимката): по-старите се смятат наново от снимката


def line_profiles(im: Image.Image, outer: np.ndarray) -> dict:
    """Колко от ширината на предмета пресича права линия (шев, граница капачка/тяло, ръб на отвор) на всяко място
    по дължината му и по ширината му (по LINE_BINS места от единия до другия край), по осите на най-малкия описан
    правоъгълник (fit.main_axes, както в photos.Frame). По тях се разбира как е обърната снимката на симетричен
    предмет."""
    from scipy import ndimage as nd

    from .fit import main_axes

    q = outer * [1.0, -1.0]
    l, s_, _ = main_axes(q)
    gray = nd.gaussian_filter(np.asarray(im.convert("L"), np.float32), 1.0)
    out = {}
    for key, a, b in (("l", l, s_), ("s", s_, l)):
        ta, tb = q @ a, q @ b
        a0, a1, b0, b1 = ta.min(), ta.max(), tb.min(), tb.max()
        ts = np.linspace(0, 1, LINE_BINS * 2)
        ws = np.linspace(0.12, 0.88, 50)
        tt, ww = np.meshgrid(ts, ws, indexing="ij")
        pts = (a0 + tt.ravel()[:, None] * (a1 - a0)) * a + (b0 + ww.ravel()[:, None] * (b1 - b0)) * b
        g = nd.map_coordinates(gray, [-pts[:, 1], pts[:, 0]], order=1).reshape(len(ts), len(ws))
        k = max(2, round(len(ts) * 0.008))
        diff = np.abs(g[k:] - g[:-k])
        thr = max(3 * float(np.median(diff)), 6.0)
        cover = np.r_[np.zeros(k // 2), (diff > thr).mean(1), np.zeros(k - k // 2)]
        out[key] = np.round(cover.reshape(LINE_BINS, 2).max(1), 3).tolist()
    return out


LOOK_BINS = 48


def looks(im: Image.Image, outer: np.ndarray) -> dict:
    """Как изглежда детайлът по дължината и по ширината си (LOOK_BINS места от единия до другия край, по осите на
    fit.main_axes, както в photos.Frame): средният цвят (Lab) и колко е пъстро (разсейването на
    яркостта). По тях photos.layout разбира кой край на снимката отстрани е кой край на снимката отгоре: капачката,
    прозрачната или цветната част и надписите са в единия край на всяка снимка."""
    from skimage import color
    from skimage.draw import polygon as fill

    from .fit import main_axes

    q = outer * [1.0, -1.0]
    l, s_, _ = main_axes(q)
    x0, y0 = np.floor(outer.min(0)).astype(int)
    x1, y1 = np.ceil(outer.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, im.size[0]), min(y1, im.size[1])
    rgb = np.asarray(im.crop((x0, y0, x1, y1)).convert("RGB"), np.float32) / 255
    mask = np.zeros(rgb.shape[:2], bool)
    rr, cc = fill(outer[:, 1] - y0, outer[:, 0] - x0, mask.shape)
    mask[rr, cc] = True
    small = min(np.ptp(q @ l), np.ptp(q @ s_))
    inner = ndimage.binary_erosion(mask, iterations=max(2, round(0.06 * small)))  # без сенките по ръба
    if inner.sum() < 50:
        inner = mask
    ys, xs = np.nonzero(inner)
    lab = color.rgb2lab(rgb[ys, xs])
    pts = np.c_[xs + x0, -(ys + y0)].astype(float)
    out = {}
    for key, a in (("l", l), ("s", s_)):
        ta = q @ a
        t = (pts @ a - ta.min()) / max(np.ptp(ta), 1e-9)
        k = np.clip((t * LOOK_BINS).astype(int), 0, LOOK_BINS - 1)
        lab_bins, tex = [], []
        for i in range(LOOK_BINS):
            sel = lab[k == i]
            if len(sel) < 5:
                lab_bins.append(None)
                tex.append(None)
                continue
            lab_bins.append([round(float(x), 2) for x in np.median(sel, axis=0)])
            tex.append(round(float(sel[:, 0].std()), 2))
        out[key] = {"lab": lab_bins, "tex": tex}
    return out


MARK_PLAIN_SAT = 0.25  # пиксел с по-малка наситеност е от самия (бял, сив) детайл, а не от цветен печат
MARK_PLAIN = 0.6  # ако такива са поне толкова от детайла, той е бял или сив
MARK_GRAY = 0.12  # петно, намерено само по по-слабия праг, остава, ако е толкова сиво (отворът за връзката: 0,08)
MARK_DROP = 0.3  # и петно поне толкова по-тъмно от него е тъмно петно
BODY_CHROMA = 12.0  # детайл с по-малко цвят (от 255) е бял или сив


def dark_marks(rgb: np.ndarray, solid: np.ndarray) -> list[tuple[np.ndarray, dict]]:
    """Тъмни компактни петна върху детайла (пиксели x, y на изрязването): отвор, през който от тази страна не се
    вижда (дълбок и тесен, встрани от средата на кадъра), изглежда като тъмно петно. Само потвърждават отвор от
    другата снимка (photos.hole_union), затова надписи и рисунки не пречат."""
    inner = ndimage.binary_erosion(solid, iterations=4)
    if inner.sum() < 400:
        return []
    gray = rgb.mean(-1)
    vals = gray[inner]
    med = float(np.median(vals))
    body = np.median(rgb[inner], axis=0)
    body_c = body - body.mean()
    if np.linalg.norm(body_c) < BODY_CHROMA:
        body_c = np.zeros(3)  # бял или сив детайл: оттенъкът на петното спрямо него не значи нищо
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 3.0
    cand = inner & (gray < med - max(4 * mad, 35.0))
    sat_px = (rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1.0)
    plain = inner & (sat_px < MARK_PLAIN_SAT)
    strict = cand.copy()
    if plain.sum() >= MARK_PLAIN * inner.sum():
        # Цветният печат (лога върху бяла флашка) увеличава разсейването и сивият отвор за връзката (127 при детайл
        # 204) не минаваше. Затова и петната поне MARK_DROP по-тъмни от самия бял или сив детайл (без печата), но от
        # тях остават само съвсем сивите (MARK_GRAY): отблясъците на метал са леко цветни.
        med_p = float(np.median(gray[plain]))
        cand |= inner & (gray < med_p * (1 - MARK_DROP))
        med = med_p
    cand = ndimage.binary_opening(cand, iterations=1)
    lab, n = ndimage.label(cand)
    area = solid.sum()
    out = []
    for k in range(1, n + 1):
        blob = lab == k
        a = blob.sum()
        if not MIN_HOLE * area <= a <= 0.08 * area:
            continue
        ys, xs = np.nonzero(blob)
        hull = Polygon(np.c_[xs, ys]).convex_hull
        if hull.area <= 0 or a / hull.area < 0.7:
            continue
        cs = measure.find_contours(np.pad(blob.astype(float), 1), 0.5)
        if cs:
            c = max(cs, key=len)[:, ::-1] - 1
            core = ndimage.binary_erosion(blob, iterations=1) if a > 60 else blob
            px = rgb[core if core.any() else blob]
            sat = float(np.median((px.max(1) - px.min(1)) / np.maximum(px.max(1), 1.0)))
            if (strict & blob).sum() < 0.5 * a and sat > MARK_GRAY:
                continue  # само по по-слабия праг и не сиво
            # Оттенъкът спрямо детайла (1: същият цвят, само по-тъмен - отвор в цветна пластмаса изглежда така).
            mark_c = np.median(px, axis=0) - np.median(px, axis=0).mean()
            norm = float(np.linalg.norm(mark_c) * np.linalg.norm(body_c))
            tone = float(mark_c @ body_c) / norm if norm > 1e-6 else 0.0
            out.append((c, {"dark": round(float(np.median(px.mean(1))) / max(med, 1.0), 3), "sat": round(sat, 3),
                            "tone": round(tone, 3)}))  # fmt: skip
    return out[:30]


def focal35(path) -> float | None:
    """Фокусното разстояние на камерата като за 35 мм филм (от EXIF), за поправката на перспективата."""
    try:
        with Image.open(path) as im:
            v = im.getexif().get_ifd(0x8769).get(41989)
        return float(v) if v and 8 <= float(v) <= 200 else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None
