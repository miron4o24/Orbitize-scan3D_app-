import numpy as np
import pytest
from PIL import Image

from app import depth as D


def test_fit_scale_ignores_outliers():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.2, 1.0, 500)
    inv_z = 0.8 * d + 0.3
    inv_z[:100] += rng.uniform(0.2, 0.6, 100)  # e.g. hull samples above a flat top: wrong reference depths
    (a, b), err = D.fit_scale(d, inv_z, np.ones(500))
    assert a == pytest.approx(0.8, rel=0.02) and b == pytest.approx(0.3, abs=0.01)
    assert err < 0.01


def test_front_depths_keeps_the_visible_surface():
    # Two points on the same pixel ray, 1 and 2 away: only the near one is visible.
    Xc = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.5, 0.0, 2.0]])
    u, v, z, w = D.front_depths(Xc, np.ones(3), 100, 100, 50, 50, (0, 0, 100, 100))
    assert sorted(z) == [1.0, 2.0] and len(z) == 2  # the third point is elsewhere in the picture
    assert (u.min(), u.max()) == (50.0, 75.0)


def test_carve_limit():
    # Three frames from the origin along +z, all seeing the surface at depth 2 (1 / (1 * 0.5 + 0)).
    f = D.Frame((0, 0, 100, 100), np.full((100, 100), 0.5), np.ones((100, 100), bool), np.eye(3), np.zeros(3),
                np.array([100, 100, 50, 50.0]))  # fmt: skip
    fits = [(f, np.array([1.0, 0.0]))] * 3
    points = np.array([[0, 0, 1.0], [0, 0, 1.9], [0, 0, 2.05], [9, 0, 1.0]])
    limit = D.carve_limit(fits, points, margin=0.1, agree=0.75)
    # Well in front of the surface: empty. At the margin: the hull's level. Behind it: solid. Unseen: unchanged.
    np.testing.assert_allclose(limit, [0, 0.75, 1, 1], atol=1e-9)
    assert not D.enough(fits, 7) and D.enough(fits, 6)


def test_scan_goes_on_without_ai_depth(tmp_path, monkeypatch):
    from app import mesh as M
    from app import pipeline as P

    # The outline shape: a ball of radius 10 in a 30-cell grid.
    axes = [np.arange(30) + 0.5] * 3
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    frac = (np.linalg.norm(g - 15, axis=-1) < 10).astype(float)
    monkeypatch.setattr(M, "hull_volume", lambda *a, **k: (frac.copy(), axes, 1.0))
    work = tmp_path / "work"
    work.mkdir()
    rng = np.random.default_rng(0)
    n = rng.normal(size=(500, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    cloud = M.Cloud(15 + 10 * n, n, np.ones((500, 4)), to_up=np.eye(3))
    views = M.Views(np.eye(3)[None], np.array([[0, 0, 50.0]]), np.array([[100, 100, 50, 50.0]]), ["a.jpg"])
    masks = [np.ones((100, 100), np.uint8)]
    log = tmp_path / "log.txt"

    monkeypatch.setattr(D, "MODEL", tmp_path / "missing.onnx")
    ball = P.carved_hull(work, tmp_path, views, masks, cloud, None)
    assert ball.is_watertight and "AI depth: skipped, model missing" in log.read_text()

    def failing_sh(step, argv):
        raise P.Failed(f"Processing failed during {step}.")

    monkeypatch.setattr(D, "MODEL", tmp_path / "model.onnx")
    D.MODEL.write_bytes(b"")
    assert P.carved_hull(work, tmp_path, views, masks, cloud, failing_sh).volume == pytest.approx(ball.volume)
    assert "AI depth: failed (Processing failed during estimating depth.)" in log.read_text()

    def carve(top):  # what depth.py would write: the ball's top cap above `top` seen through
        def sh(step, argv):
            with np.load(argv[3]) as z:
                limit = np.where(z["probes"][:, 2] > top, 0.0, 1.0)
            np.savez(argv[4], limit=limit, fits=1, used=True)

        return sh

    capped = P.carved_hull(work, tmp_path, views, masks, cloud, carve(21))  # a cap 4 of 20 high: ~10% of the ball
    assert 0.85 * ball.volume < capped.volume < 0.92 * ball.volume and capped.bounds[1, 2] < 21.5
    assert log.read_text().rstrip().endswith("of the outline shape: using it")
    halved = P.carved_hull(work, tmp_path, views, masks, cloud, carve(15))  # more than MAX_CARVE: a bad fit
    assert halved.volume == pytest.approx(ball.volume)
    assert log.read_text().rstrip().endswith("of the outline shape: not used")


@pytest.mark.skipif(not D.MODEL.exists(), reason="depth model not downloaded (setup-windows.ps1 does it)")
def test_depth_model_sees_the_near_side_nearer():
    # A floor seen at an angle: the bottom of the picture is nearer than the top.
    g = np.linspace(40, 220, 300)[:, None, None] * np.ones((300, 400, 3))
    rng = np.random.default_rng(1)
    tex = (g + rng.normal(0, 25, g.shape) * np.linspace(0.3, 1.5, 300)[:, None, None]).clip(0, 255)
    d = D.disparity(D.session(), Image.fromarray(tex.astype(np.uint8)))
    assert d.shape == (300, 400)
    assert d[-50:].mean() > d[:50].mean()
