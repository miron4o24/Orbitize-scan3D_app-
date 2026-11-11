"""Инженерният режим: изрязване на детайла с MobileSAM (app/cad/vision.py), когато IS-Net взима и това около него.
Снимки, нарисувани като истинските от приложението: флашка върху телефон, флашка, хваната с пръсти. Нужни са
моделите за изрязване (IS-Net и MobileSAM; на сървъра са в tools)."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import Polygon, box

from app import refine as RF
from app import segment as S
from app.cad import vision as V

if not (S.MODEL.exists() and RF.ENCODER.exists() and RF.DECODER.exists()):
    pytest.skip("няма моделите за изрязване (tools/seg, tools/sam)", allow_module_level=True)

W, H = 2400, 1800
STICK = (930, 760, 1470, 1000)  # бялата флашка (x0, y0, x1, y1)


def photo(path, phone: bool, fingers: bool):
    rng = np.random.default_rng(1)
    wood = np.clip(np.array([150, 120, 90]) + rng.normal(0, 14, (H // 8, W // 8, 1)), 0, 255).astype(np.uint8)
    im = Image.fromarray(wood).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))
    d = ImageDraw.Draw(im)
    if phone:  # тъмен телефон под флашката, почти колкото кадъра
        d.rounded_rectangle((120, 260, 2280, 1540), radius=110, fill=(28, 30, 34), outline=(150, 140, 120), width=18)
    d.rounded_rectangle((STICK[0] + 8, STICK[1] + 14, STICK[2] + 8, STICK[3] + 14), radius=40, fill=(40, 36, 32))
    d.rounded_rectangle(STICK, radius=40, fill=(236, 238, 240))
    d.rectangle((1250, 790, 1440, 970), fill=(214, 218, 224))  # капачката, малко по-сива
    d.rectangle((1000, 800, 1090, 860), fill=(30, 70, 170))  # етикет
    if fingers:  # пръсти от горния ръб на кадъра, опрени във флашката (без да я закриват)
        d.rounded_rectangle((1060, -60, 1220, 762), radius=70, fill=(222, 170, 140))
        d.rounded_rectangle((1240, -60, 1390, 762), radius=70, fill=(214, 162, 132))
    im.filter(ImageFilter.GaussianBlur(1.2)).save(path, quality=92)


def overlap(r: dict) -> float:
    got = Polygon(r["outer"]).buffer(0)
    want = box(*STICK)
    return got.intersection(want).area / got.union(want).area


@pytest.mark.parametrize("fingers", [False, True])
def test_tap_on_the_part_lying_on_a_phone(tmp_path, fingers):
    path = tmp_path / "top.jpg"
    photo(path, phone=True, fingers=fingers)
    r = V.segment(S.session(), path, [(1150 / W, 900 / H)])  # докосване върху флашката (дял от снимката)
    assert r["by"] == "tap"
    assert overlap(r) > 0.93  # флашката, без телефона и пръстите


def test_fingers_reaching_the_frame_edge_are_left_out(tmp_path):
    """Автоматично: ако IS-Net вземе и пръстите (до ръба на кадъра), MobileSAM изрязва флашката в средата."""
    path = tmp_path / "top.jpg"
    photo(path, phone=False, fingers=True)
    r = V.segment(S.session(), path)
    assert overlap(r) > 0.9
    assert min(y for _, y in r["outer"]) > 600  # не стига до горния ръб с пръстите


def test_tap_with_an_exclusion_point(tmp_path):
    """Точка "това не е детайлът" върху телефона: никога не влиза в очертанието."""
    path = tmp_path / "top.jpg"
    photo(path, phone=True, fingers=False)
    r = V.segment(S.session(), path, [(1150 / W, 900 / H), (300 / W, 400 / H, 0)])
    assert overlap(r) > 0.93
    assert not Polygon(r["outer"]).buffer(0).contains(Polygon([(290, 390), (310, 390), (310, 410)]))
