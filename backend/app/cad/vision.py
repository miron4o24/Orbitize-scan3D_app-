"""Снимка -> очертанието на детайла в мм.

Снимките са направени с телефона успоредно на масата (приложението чака нивелира), затова изображението е почти
ортогонална проекция: мащабът по двете оси идва от измерените общи размери (L и W отгоре, L и H отстрани, W и H
откъм края), а не от снимката. Изрязването е IS-Net (app/segment.py, без външни услуги) на два прохода: първо къде е
детайлът в целия кадър, после близко изрязване около него в пълна разделителна способност, за точен ръб.

Отворите през детайла: мрежата понякога ги запълва, затова се търсят и по цвят: петно вътре в детайла с цвета на
фона около него, с компактна форма (виждаме масата през отвора)."""

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from shapely import affinity
from shapely.geometry import Polygon
from skimage import measure

from .. import segment as S

SIDE = 2400  # дългата страна, до която се смалява снимката
MARGIN = 0.3  # поле около детайла при второто изрязване, в размери на детайла
MIN_HOLE = 0.0015  # отвор под този дял от детайла е шум
COLOR_HOLE_K = 2.5  # колко близо до цвета на фона (в разсейвания на фона) е пиксел "през отвор"


@dataclass
class View:
    """Очертанието на детайла в един изглед, в мм на изгледа (u надясно, v нагоре, от 0 до U и V)."""

    outer: np.ndarray  # (n, 2)
    holes: list[np.ndarray]
    U: float
    V: float
    # Обратно към снимката (за наложеното очертание): пиксел = rot(-angle) около centre на (x0 + u/sx, y1 - v/sy).
    angle: float
    centre: tuple[float, float]
    x0: float
    y1: float
    sx: float
    sy: float
    image_size: tuple[int, int]
    warnings: list[str] = field(default_factory=list)

    def to_photo(self, uv: np.ndarray) -> np.ndarray:
        """Точки в мм на изгледа -> пиксели на (смалената) снимка."""
        x = self.x0 + uv[:, 0] / self.sx
        y = self.y1 - uv[:, 1] / self.sy
        a = math.radians(self.angle)
        cx, cy = self.centre
        dx, dy = x - cx, y - cy
        return np.c_[cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a)]

    def transform(self) -> dict:
        return {"angle": self.angle, "centre": list(self.centre), "x0": self.x0, "y1": self.y1, "sx": self.sx,
                "sy": self.sy, "U": self.U, "V": self.V, "image_size": list(self.image_size)}  # fmt: skip


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


def read(sess, path, U: float, V: float, prefer_long_u: bool | None = None) -> View:
    """Изгледът от снимка при измерени размери U (хоризонтално) и V (вертикално) на изгледа, в мм."""
    im = load(path)
    field_, _solid, box = cutout(sess, im)
    outer, holes = contours(field_)
    off = np.array([box[0], box[1]], float)
    outer, holes = outer + off, [h + off for h in holes]
    poly = Polygon(outer)
    rect = poly.minimum_rotated_rectangle
    rx, ry = rect.exterior.coords.xy
    edges = [(rx[k + 1] - rx[k], ry[k + 1] - ry[k]) for k in range(2)]
    long_edge = max(edges, key=lambda e: math.hypot(*e))
    angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    angle = (angle + 90) % 180 - 90  # най-малкото завъртане до хоризонтала
    # Дългата страна на изгледа надясно, освен ако измерените размери казват, че е изправен.
    want_long_u = U >= V if prefer_long_u is None else prefer_long_u
    if abs(U - V) / max(U, V) < 0.08 and prefer_long_u is None:
        angle = (angle + 45) % 90 - 45  # почти квадратен: най-малкото завъртане, без да обръщаме
    elif not want_long_u:
        angle -= 90
    c = poly.centroid
    centre = (c.x, c.y)

    def turn(pts):
        p = affinity.rotate(Polygon(pts) if len(pts) > 2 else pts, -angle, origin=centre)
        return np.asarray(p.exterior.coords)[:-1]

    o = turn(outer)
    hs = [turn(h) for h in holes]
    x0, x1 = o[:, 0].min(), o[:, 0].max()
    y0, y1 = o[:, 1].min(), o[:, 1].max()
    sx, sy = U / (x1 - x0), V / (y1 - y0)
    warnings = []
    ratio = sx / sy
    if not 0.9 <= ratio <= 1.11:
        warnings.append("Пропорциите на снимката се разминават с мерките с "
                        f"{abs(1 - ratio) * 100:.0f}%: снимката може да е под ъгъл или мярка да е грешна.")  # fmt: skip

    def mm(p):
        return np.c_[(p[:, 0] - x0) * sx, (y1 - p[:, 1]) * sy]

    return View(mm(o), [mm(h) for h in hs], U, V, angle, centre, float(x0), float(y1), float(sx), float(sy),
                im.size, warnings)  # fmt: skip
