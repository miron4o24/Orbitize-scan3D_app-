"""Инженерният режим: докоснатото "това е детайлът" винаги влиза в очертанието (app/cad/vision.py sam_cutout), дори
когато MobileSAM не дава нито един кандидат с всички докоснати точки. Без моделите: MobileSAM е заместен с изрязване
по цвят."""

import numpy as np
from PIL import Image

from app import refine as RF
from app.cad import vision as V

DARK, LIGHT = (40, 40, 40), (225, 225, 225)  # двете половини на детайла: метал и пластмаса


def photo() -> Image.Image:
    a = np.full((300, 400, 3), 128, np.uint8)  # сива маса
    a[100:200, 100:200] = DARK
    a[100:200, 200:300] = LIGHT
    return Image.fromarray(a)


def fake_candidates(models, image, points, box=None, emb=None):
    """Като MobileSAM, който вижда двете половини като отделни предмети: за точки и върху двете - двата кандидата
    поотделно (никой не съдържа всички точки), за точка само върху едната - нея."""
    gray = np.asarray(image.convert("L"), np.float32)
    halves = {"dark": gray < 80, "light": gray > 180}
    on = set()
    for p in points:
        if len(p) < 3 or p[2]:
            v = gray[int(p[1]), int(p[0])]
            on |= {"dark"} if v < 80 else {"light"} if v > 180 else set()
    return [(np.where(halves[k], 10.0, -10.0), 0.95) for k in sorted(on)]


def test_taps_on_two_different_looking_halves_of_one_part_are_both_kept(monkeypatch):
    monkeypatch.setattr(V, "sam_candidates", fake_candidates)
    monkeypatch.setattr(RF, "embedding", lambda enc, image: None)
    taps = [(150.0, 150.0), (250.0, 150.0)]
    _, solid, box = V.sam_cutout((None, None), photo(), taps)
    for x, y in taps:  # и двете половини са в детайла
        assert solid[int(y) - box[1], int(x) - box[0]]
    assert abs(int(solid.sum()) - 200 * 100) < 0.05 * 200 * 100


def test_a_tap_on_the_table_is_still_no_part(monkeypatch):
    """Докосване само по масата (нищо от двете половини): както досега, "няма детайл"."""
    monkeypatch.setattr(V, "sam_candidates", fake_candidates)
    monkeypatch.setattr(RF, "embedding", lambda enc, image: None)
    try:
        V.sam_cutout((None, None), photo(), [(30.0, 30.0)])
    except ValueError as e:
        assert str(e) == "no object"
    else:
        raise AssertionError("a tap on the table cut out a part")
