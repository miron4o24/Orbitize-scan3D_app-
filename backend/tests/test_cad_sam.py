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


def test_ports_window_and_button_on_a_face_are_found_but_not_print_or_glare():
    """Стената на powerbank: екран (тъмен правоъгълник), USB-A и USB-C (тъмни отвори с цветно езиче), кръгъл бутон
    (само ръб, тъмен като стената). Белият надпис, синьото лого и отблясъкът не са детайли."""
    from app.cad import details as DT

    rng = np.random.default_rng(2)
    wood = np.clip(np.array([150, 120, 90]) + rng.normal(0, 12, (H // 8, W // 8, 1)), 0, 255).astype(np.uint8)
    im = Image.fromarray(wood).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))
    d = ImageDraw.Draw(im)
    body = (500, 500, 1900, 1300)
    d.rounded_rectangle(body, radius=90, fill=(38, 40, 44))
    d.rounded_rectangle((560, 560, 1060, 900), radius=30, fill=(10, 10, 12))  # екранът
    d.rounded_rectangle((1150, 600, 1420, 700), radius=8, fill=(14, 14, 16))  # USB-A: тъмен отвор
    d.rectangle((1175, 615, 1395, 665), fill=(225, 90, 110))  # и розовото езиче в него
    d.rounded_rectangle((1500, 615, 1700, 685), radius=35, fill=(14, 14, 16))  # USB-C
    d.rounded_rectangle((1535, 637, 1665, 663), radius=10, fill=(220, 120, 50))  # контактите
    d.rectangle((620, 960, 780, 1060), fill=(30, 70, 170))  # синьо лого: печат, не порт
    d.ellipse((1250, 950, 1370, 1070), outline=(12, 12, 14), width=6)  # бутонът
    d.text((620, 1100), "INPUT 5V 3A  OUTPUT 9V 2A", fill=(190, 190, 190), font_size=48)  # печатът
    d.rectangle((1500, 900, 1540, 1250), fill=(235, 235, 238))  # отблясък
    im = im.filter(ImageFilter.GaussianBlur(1.0))
    outer = np.array([[500, 500], [1900, 500], [1900, 1300], [500, 1300]], float)
    found = DT.find(V.sam(), im, outer)
    kinds = sorted(f["kind"] for f in found)
    boxes = {f["kind"]: Polygon(f["pts"]).bounds for f in found}

    def near(kind, want):
        return any(f["kind"] == kind and Polygon(f["pts"]).buffer(0).intersection(box(*want)).area
                   >= 0.6 * box(*want).area for f in found)  # fmt: skip

    assert near("panel", (560, 560, 1060, 900)), boxes
    assert near("port", (1150, 600, 1420, 700)) and near("port", (1500, 615, 1700, 685)), boxes
    assert near("button", (1250, 950, 1370, 1070)), boxes
    assert all(not Polygon(f["pts"]).buffer(0).intersects(box(1500, 900, 1540, 1250)) for f in found), boxes
    assert all(not Polygon(f["pts"]).buffer(0).intersects(box(620, 1100, 1300, 1160)) for f in found), boxes
    assert all(not Polygon(f["pts"]).buffer(0).intersects(box(630, 970, 770, 1050)) for f in found), boxes
    assert len(kinds) == 4, kinds
