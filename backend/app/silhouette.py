"""Очертанието на предмета от една снимка, направена от приложението точно преди записа за AI модел. Приложението
рисува по него точките на екрана за заснемане (src/components/ScanHud.tsx): предметът, а не общ цилиндър, със
страните, които вече са заснети, в синьо. Само за екрана: моделът се прави от видеото, както преди.

Изрязването е същото като в инженерния режим и при моделите (app/segment.py, IS-Net на процесора): първи проход за
мястото на предмета (най-близкия до средата), втори в изрязване около него. Резултатът е профил по редове отдолу
нагоре: за всеки ред лявото и дясното на предмета като дял от ширината му, и съотношението височина към ширина. И
дали повърхността под предмета е без шарка (plain_surface): тогава приложението предупреждава още по време на записа."""

import io

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from . import segment as S

SIDE = 768  # смаляване преди изрязването: за екрана стига
ROWS = 32
# Повърхността под предмета е без шарка, когато около основата му (под долните 15% от него, до 60% от височината му
# надолу и 75% от ширината встрани) има под PLAIN_BELOW особености на 10 000 пиксела: локални екстремуми на разлика от
# гаусови размивания (както при SIFT) с контраст поне PLAIN_CONTRAST от яркостта. Камерата се хваща за тях (вижте
# app/pipeline.py); The Rock върху черна кутия (17.11.2026, 14 мин и двойни очи): 24-35; върху шарения поднос, Фродо,
# тиквата и колонката: 62-113, при първите кадри от видеото.
PLAIN_CONTRAST = 0.01
PLAIN_BELOW = 45.0
PLAIN_AREA = 0.02  # повърхност под толкова от снимката (предметът я запълва) не се преценява


def features(gray: np.ndarray, region: np.ndarray, contrast: float = PLAIN_CONTRAST) -> float:
    """Особеностите (екстремуми на разлика от гаусови размивания при три мащаба, над `contrast`) на 10 000 пиксела
    от `region`. gray: яркост 0..1."""
    count = 0
    for s in (1.6, 2.26, 3.2):
        d = ndimage.gaussian_filter(gray, s) - ndimage.gaussian_filter(gray, s * 1.414)
        peak = ndimage.maximum_filter(d, size=5) == d
        pit = ndimage.minimum_filter(d, size=5) == d
        count += int((((peak & (d > contrast)) | (pit & (d < -contrast))) & region).sum())
    return count / max(1, int(region.sum())) * 1e4


def plain_surface(im: Image.Image, part: np.ndarray) -> bool | None:
    """Дали повърхността около основата на предмета е без шарка (вижте PLAIN_BELOW); None, ако не се вижда достатъчно
    от нея. part: маската на предмета при каквато и да е разделителна способност."""
    W, H = im.size
    obj = np.asarray(Image.fromarray(part.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    ys, xs = np.nonzero(obj)
    if not len(xs):
        return None
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    near = ndimage.binary_dilation(obj, iterations=max(2, int(0.05 * max(w, h))))
    yy, xx = np.mgrid[0:H, 0:W]
    region = (~near & (yy >= ys.max() - 0.15 * h) & (yy <= ys.max() + 0.6 * h)
              & (xx >= xs.min() - 0.75 * w) & (xx <= xs.max() + 0.75 * w))  # fmt: skip
    if region.sum() < PLAIN_AREA * W * H:
        return None
    return features(np.asarray(im.convert("L"), np.float32) / 255, region) < PLAIN_BELOW


def profile(sess, data: bytes) -> dict:
    """{"found": False} или {"found": True, "aspect": височина/ширина, "rows": [[ляво, дясно] | None, ...] отдолу
    нагоре, 0..1 от ширината, "plain": повърхността под него е без шарка}. None в rows: на този ред предметът няма
    пиксели (например между дръжка и тяло)."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((SIDE, SIDE))
    where = S.locate(S.mask(sess, im))
    if where is None:
        return {"found": False}
    m = S.crop_mask(sess, im, *where)
    part = S.central_part(m, where[0], where[1])
    if part is None or part.sum() < 50:
        return {"found": False}
    ys, xs = np.nonzero(part)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    w, h = x1 - x0, y1 - y0
    rows = []
    for k in range(ROWS):  # отдолу нагоре
        hi = y1 - k * h / ROWS
        band = part[int(max(y0, hi - h / ROWS)) : int(np.ceil(hi)), x0:x1]
        cols = np.nonzero(band.any(axis=0))[0]
        rows.append([round(cols[0] / w, 3), round((cols[-1] + 1) / w, 3)] if len(cols) else None)
    return {"found": True, "aspect": round(h / w, 3), "rows": rows, "plain": plain_surface(im, part) is True}
