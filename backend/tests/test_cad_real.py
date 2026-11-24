"""Инженерният режим върху истински снимки (tests/photos, изрязани около предмета): пауърбанка, 18.11.2026 (задача
14a3b5). Нужни са моделите за изрязване (IS-Net, MobileSAM)."""

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from app import refine as RF
from app import segment as S

if not S.MODEL.exists() or not (RF.ENCODER.exists() and RF.DECODER.exists()):
    pytest.skip("няма моделите за изрязване (tools/seg, tools/sam)", allow_module_level=True)

from app.cad import details as DT  # noqa: E402
from app.cad import vision as V  # noqa: E402

PHOTOS = Path(__file__).parent / "photos"
GLARE = PHOTOS / "powerbank_front_glare.jpg"
# Където потребителят докосна върху лъжливия отвор ("Това е предметът"), като дял от снимката.
TAPS = [[0.488, 0.394], [0.516, 0.505], [0.488, 0.534]]


@pytest.fixture(scope="module")
def sess():
    return S.session()


def test_glare_on_a_shiny_side_is_not_a_hole(sess):
    """Отблясъкът по лъскавата страна е бял като листа под нея: по цвят изглежда като отвор (три петна, 4% от
    страната), но мрежата е сигурна, че там е детайлът, и ръбът на отблясъка е мек. Със или без Florence-2 и SAM 2.1
    (app/cad/locate.py): и тогава SAM 2.1 вижда отблясъка като част от детайла."""
    im = V.load(GLARE)
    field_, solid, box = V.cutout(sess, im)
    filled = ndimage.binary_fill_holes(solid)
    crop = np.asarray(im.crop(box), np.float32)
    assert V.color_holes(crop, filled).sum() > 0.01 * filled.sum()  # капанът е там
    got = V.segment(sess, GLARE)
    assert got["by"] in ("ai", "isnet") and got["holes"] == []


def test_tapping_the_object_never_leaves_a_hole_under_the_finger(sess):
    """Докосването върху лъжливия отвор ("Това е предметът") преди го изрязваше наново със същия отвор."""
    got = V.segment(sess, GLARE, TAPS)
    assert got["by"] == "tap" and got["holes"] == []
    w, h = got["size"]
    from shapely.geometry import Point, Polygon

    outer = Polygon(got["outer"])
    assert all(outer.contains(Point(x * w, y * h)) for x, y in TAPS)


def test_ports_inside_the_port_plate_are_found(sess):
    """Краят на пауърбанката: сребриста плоча с USB-A (розово езиче) и USB-C (меден). Плочата се намираше, а
    портовете в нея се губеха като "същия детайл" и като цветен печат."""
    im = V.load(PHOTOS / "powerbank_ports.jpg")
    got = V.segment(sess, PHOTOS / "powerbank_ports.jpg")
    kinds = sorted(d["kind"] for d in DT.find(V.sam(), im, np.asarray(got["outer"])))
    assert kinds == ["panel", "port", "port"]
