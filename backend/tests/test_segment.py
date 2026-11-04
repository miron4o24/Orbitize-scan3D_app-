import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import segment as S


@pytest.mark.skipif(not S.MODEL.exists(), reason="segmentation model not downloaded (setup-windows.ps1 does it)")
def test_masks_find_the_object(tmp_path):
    # A red ball on a grey floor with some speckle, in the middle of the picture as the app films it.
    rng = np.random.default_rng(0)
    floor = (rng.normal(150, 12, (480, 640, 3))).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(floor)
    ImageDraw.Draw(im).ellipse([240, 160, 400, 320], fill=(200, 30, 30))
    im.save(tmp_path / "img_0001.jpg")
    S.main(tmp_path, tmp_path / "masks.npz")

    m = np.load(tmp_path / "masks.npz")["img_0001.jpg"]
    assert m.shape == (480 // S.MASK_SCALE, 640 // S.MASK_SCALE)
    s = S.MASK_SCALE
    assert (m[200 // s : 280 // s, 280 // s : 360 // s] > 127).mean() > 0.9  # the ball
    assert (m[: 100 // s] > 127).mean() < 0.05  # the floor above it


def test_masks_keep_real_holes():
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 20:80] = 255
    m[40:60, 40:60] = 0  # a ring's middle: 11% of it
    m[25:27, 25:27] = 0  # a speck the network missed
    part = S.central_part(m, 50, 50)
    assert not part[50, 50] and part[26, 26]
