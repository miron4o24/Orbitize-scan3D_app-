"""Снимка -> очертанието на детайла в пиксели на снимката.

Снимките са направени с телефона успоредно на масата (приложението чака нивелира), затова изображението е почти
ортогонална проекция. Коя страна на детайла е на снимката, как е завъртяна и колко мм е пикселът, се решава по-късно
(photos.py), от всички снимки и от измерените общи размери. Изрязването е IS-Net (app/segment.py, без външни
услуги) на два прохода: първо къде е детайлът в целия кадър, после близко изрязване около него в пълна
разделителна способност, за точен ръб.

Отворите през детайла: мрежата понякога ги запълва, затова се търсят и по цвят: петно вътре в детайла с цвета на
фона около него, с компактна форма (виждаме масата през отвора)."""

import math

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
    hard = part_near_centre(soft > 0.5, crop.size[0] / 2, crop.size[1] / 2)
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


def segment(sess, path) -> dict:
    """Очертанието на детайла на снимката: {"outer": [[x, y], ...], "holes": [...], "size": [w, h]} в пиксели на
    смалената снимка (load), x надясно, y надолу. ValueError: на снимката няма детайл."""
    im = load(path)
    field_, solid, box = cutout(sess, im)
    outer, holes = contours(field_)
    off = np.array([box[0], box[1]], float)
    o = outer + off
    edge = 3
    if o[:, 0].min() < edge or o[:, 1].min() < edge or o[:, 0].max() > im.size[0] - edge or             o[:, 1].max() > im.size[1] - edge:  # fmt: skip
        raise ValueError("cut")
    return {"outer": np.round(outer + off, 2).tolist(), "holes": [np.round(h + off, 2).tolist() for h in holes],
            "size": list(im.size), "f35": focal35(path), "lines": line_profiles(im, outer + off),
            **marks_of(np.asarray(im.crop(box), np.float32), solid, off)}  # fmt: skip


def marks_of(rgb: np.ndarray, solid: np.ndarray, off: np.ndarray) -> dict:
    """Тъмните петна (dark_marks) в пиксели на снимката и колко са тъмни (спрямо детайла) и цветни."""
    found = dark_marks(rgb, solid)
    return {"marks": [np.round(m + off, 2).tolist() for m, _ in found], "mark_info": [i for _, i in found]}


LINE_BINS = 200


def line_profiles(im: Image.Image, outer: np.ndarray) -> dict:
    """Колко от ширината на предмета пресича права линия (шев, граница капачка/тяло, ръб на отвор) на всяко място
    по дължината му и по ширината му (по LINE_BINS места от единия до другия край), по осите на най-малкия описан
    правоъгълник (както в photos.Frame). По тях се разбира как е обърната снимката на симетричен предмет."""
    from scipy import ndimage as nd

    q = outer * [1.0, -1.0]
    poly = Polygon(q).buffer(0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    rect = poly.minimum_rotated_rectangle
    rx, ry = rect.exterior.coords.xy
    edges = [(rx[k + 1] - rx[k], ry[k + 1] - ry[k]) for k in range(2)]
    e = max(edges, key=lambda v: math.hypot(*v))
    phi = (math.degrees(math.atan2(e[1], e[0])) + 90) % 180 - 90
    l = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    s_ = np.array([-l[1], l[0]])
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
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 3.0
    cand = inner & (gray < med - max(4 * mad, 35.0))
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
            out.append((c, {"dark": round(float(np.median(px.mean(1))) / max(med, 1.0), 3), "sat": round(sat, 3)}))
    return out[:30]


def focal35(path) -> float | None:
    """Фокусното разстояние на камерата като за 35 мм филм (от EXIF), за поправката на перспективата."""
    try:
        with Image.open(path) as im:
            v = im.getexif().get_ifd(0x8769).get(41989)
        return float(v) if v and 8 <= float(v) <= 200 else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None
