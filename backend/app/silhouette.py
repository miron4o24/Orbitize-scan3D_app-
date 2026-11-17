"""Очертанието на предмета от една снимка, направена от приложението точно преди записа за AI модел. Приложението
рисува по него точките на екрана за заснемане (src/components/ScanHud.tsx): предметът, а не общ цилиндър, със
страните, които вече са заснети, в синьо. Само за екрана: моделът се прави от видеото, както преди.

Изрязването е същото като в инженерния режим и при моделите (app/segment.py, IS-Net на процесора): първи проход за
мястото на предмета (най-близкия до средата), втори в изрязване около него. Резултатът е профил по редове отдолу
нагоре: за всеки ред лявото и дясното на предмета като дял от ширината му, и съотношението височина към ширина."""

import io

import numpy as np
from PIL import Image, ImageOps

from . import segment as S

SIDE = 768  # смаляване преди изрязването: за екрана стига
ROWS = 32


def profile(sess, data: bytes) -> dict:
    """{"found": False} или {"found": True, "aspect": височина/ширина, "rows": [[ляво, дясно] | None, ...] отдолу
    нагоре, 0..1 от ширината}. None: на този ред предметът няма пиксели (например между дръжка и тяло)."""
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
    return {"found": True, "aspect": round(h / w, 3), "rows": rows}
