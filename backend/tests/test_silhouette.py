"""Снимката преди запис за AI модел (app/silhouette.py): дали повърхността под предмета е без шарка, за която камерата
да се хване. The Rock върху черна кутия (17.11.2026) излезе за 14 мин с двойни очи; върху шарения поднос - чисто."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import segment as S
from app import silhouette as SIL

PHOTOS = Path(__file__).parent / "photos"


def surface(printed: bool) -> Image.Image:
    """Предмет (светъл правоъгълник) върху тъмна гладка повърхност или върху светла с отпечатани петна."""
    rng = np.random.default_rng(0)
    W, H = 600, 800
    if printed:
        im = Image.new("L", (W, H), 200)
        d = ImageDraw.Draw(im)
        for _ in range(900):
            x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(3, 14)
            d.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=int(rng.uniform(60, 250)))
        a = np.asarray(im, np.float32)
    else:
        a = 50 + rng.normal(0, 0.8, (H, W))  # черна кутия: само малко шум
    a[200:520, 220:380] = 210  # предметът
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")


def test_a_plain_dark_surface_is_plain_and_a_printed_one_is_not():
    part = np.zeros((200, 150), bool)
    part[50:130, 55:95] = True  # маската на предмета при 1/4 от размера
    assert SIL.plain_surface(surface(False), part) is True
    assert SIL.plain_surface(surface(True), part) is False


def test_a_surface_hidden_by_the_object_is_not_judged():
    """Предмет, който запълва кадъра: под него почти няма повърхност и не се предупреждава."""
    part = np.zeros((200, 150), bool)
    part[5:198, 5:145] = True
    assert SIL.plain_surface(surface(False), part) is None


@pytest.mark.skipif(not S.MODEL.exists(), reason="няма модела за изрязване (tools/seg)")
def test_the_black_box_under_the_rock_is_plain_and_the_pastry_plate_is_not():
    """Първите кадри на двете сканирания, изрязани около фигурката и повърхността под нея."""
    sess = S.session()
    bad = SIL.profile(sess, (PHOTOS / "surface_black_box.jpg").read_bytes())
    good = SIL.profile(sess, (PHOTOS / "surface_pastry_plate.jpg").read_bytes())
    assert bad["found"] and bad["plain"] is True
    assert good["found"] and good["plain"] is False
