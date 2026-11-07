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


def _profile(width: float, height: float, hole: bool = False):
    outer = np.array([[0, 0], [width, 0], [width, height], [0, height], [0, 0]], dtype=float)
    holes = []
    if hole:
        holes.append(np.array([[width * 0.35, height * 0.35], [width * 0.65, height * 0.35],
                               [width * 0.65, height * 0.65], [width * 0.35, height * 0.65],
                               [width * 0.35, height * 0.35]], dtype=float))
    return {"outers": [outer], "holes": holes, "box": (0.0, 0.0, width, height),
            "px": (width, height), "rgb": np.zeros((2, 2, 3), np.uint8), "rgb_scale": 1.0}


def test_build_mesh_preserves_measured_bounds_and_profile_hole():
    mesh = part.build_mesh({
        "top": _profile(60, 30, hole=True),
        "side": _profile(60, 12),
        "end": _profile(30, 12),
    }, (60.0, 30.0, 12.0))
    np.testing.assert_allclose(mesh.bounds[0], [0.0, 0.0, 0.0], atol=0.1)
    np.testing.assert_allclose(mesh.bounds[1], [60.0, 30.0, 12.0], atol=0.1)
    assert mesh.is_watertight
    assert mesh.volume < 60.0 * 30.0 * 12.0


def test_build_mesh_requires_three_profiles():
    with pytest.raises(ValueError, match="three orthogonal profiles"):
        part.build_mesh({"top": _profile(60, 30)}, (60.0, 30.0, 12.0))


def test_choose_profiles_recovers_mislabeled_physical_axes():
    named = {
        "top": _profile(300, 90), "bottom": _profile(300, 90),
        "front": _profile(90, 30), "back": _profile(90, 30),
        "left": _profile(300, 30), "right": _profile(300, 30),
    }
    _, mapping = part.choose_profiles(named, [100.0, 30.0, 10.0])
    assert mapping == {"top": "top", "side": "end", "end": "side"}


def test_choose_profiles_rejects_inconsistent_opposite_views():
    def l_shape(width: float, height: float):
        view = _profile(width, height)
        view["outers"] = [np.array([
            [0, 0], [width, 0], [width, height * 0.2], [width * 0.2, height * 0.2],
            [width * 0.2, height], [0, height], [0, 0],
        ], dtype=float)]
        return view

    named = {
        "top": _profile(250, 100), "bottom": l_shape(250, 100),
        "front": _profile(250, 25), "back": _profile(250, 25),
        "left": _profile(100, 25), "right": _profile(100, 25),
    }
    with pytest.raises(ValueError, match="opposite views disagree"):
        part.choose_profiles(named, [100.0, 40.0, 10.0])
