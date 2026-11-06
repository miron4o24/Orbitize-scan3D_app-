import numpy as np
import pytest
from app import part


def test_calculate_dimensions_all_three_given():
    views = {
        "top": {"px": (100.0, 50.0)},
        "side": {"px": (100.0, 20.0)},
        "end": {"px": (50.0, 20.0)},
    }
    L, W, H = part.calculate_dimensions(views, [60.0, 30.0, 12.0])
    assert L == 60.0
    assert W == 30.0
    assert H == 12.0


def test_calculate_dimensions_only_length_given():
    views = {
        "top": {"px": (200.0, 100.0)},
        "side": {"px": (200.0, 40.0)},
    }
    L, W, H = part.calculate_dimensions(views, [50.0, 0.0, 0.0])
    assert L == 50.0
    assert W == 25.0
    assert H == 10.0


def test_calculate_dimensions_only_width_given():
    views = {
        "top": {"px": (300.0, 100.0)},
        "side": {"px": (300.0, 60.0)},
    }
    L, W, H = part.calculate_dimensions(views, [0.0, 20.0, 0.0])
    assert L == 60.0
    assert W == 20.0
    assert H == 12.0


def test_calculate_dimensions_length_and_height_given():
    views = {
        "top": {"px": (150.0, 75.0)},
        "side": {"px": (150.0, 30.0)},
    }
    L, W, H = part.calculate_dimensions(views, [50.0, 0.0, 10.0])
    assert L == 50.0
    assert W == 25.0
    assert H == 10.0


def test_pick_views_from_proportions():
    found = {
        "img1.jpg": {"px": (300.0, 150.0)},  # 2:1 -> top (L=60, W=30)
        "img2.jpg": {"px": (300.0, 75.0)},   # 4:1 -> side (L=60, H=15)
        "img3.jpg": {"px": (150.0, 75.0)},   # 2:1 -> end (W=30, H=15)
    }
    picked = part.pick_views(found, [60.0, 30.0, 15.0])
    assert "top" in picked
    assert "side" in picked
