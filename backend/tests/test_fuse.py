"""Части от сливането на дълбочината (app/fuse.py)."""

import numpy as np

from app import fuse as F


def test_facing_is_the_cosine_to_the_surface():
    K = np.array([[500.0, 0, 50], [0, 500.0, 50], [0, 0, 1]])
    box = (0, 0, 100, 100)
    flat = np.full((100, 100), 2.0, np.float32)  # стена право пред камерата
    assert np.allclose(F.facing(flat, K, box)[40:60, 40:60], 1.0, atol=1e-3)
    # Същата стена, завъртяна на 60 градуса около вертикалната ос: дълбочината расте през изображението.
    u = (np.arange(100) - 50) / 500.0
    tilt = np.tan(np.radians(60))
    z = 2.0 / (1 - u * tilt)  # точки X = u z върху равнината z = 2 + X tan(60)
    turned = np.broadcast_to(z[None, :], (100, 100)).astype(np.float32)
    cos = F.facing(turned, K, box)
    centre = cos[50, 50]
    assert abs(centre - np.cos(np.radians(60))) < 0.02
    # Без дълбочина: не влияе на ъгъла.
    holes = flat.copy()
    holes[10:20, 10:20] = 0
    assert (F.facing(holes, K, box)[9:21, 9:21] == 1.0).all()


def test_smoothing_flattens_ripples_and_keeps_the_edges():
    import trimesh

    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    for _ in range(5):
        box = box.subdivide()
    rng = np.random.default_rng(0)
    noisy = trimesh.Trimesh(box.vertices + rng.normal(scale=0.004, size=box.vertices.shape), box.faces, process=False)
    smooth = F.smooth_keeping_edges(noisy, 20)

    def off(mesh):  # колко е всеки връх от повърхността на куба
        return np.abs(np.abs(mesh.vertices).max(1) - 1.0)

    assert off(smooth).mean() < 0.4 * off(noisy).mean()
    # Ръбовете не са закръглени: върховете при тях остават на куба.
    edge = np.sort(np.abs(box.vertices), axis=1)[:, 1] > 0.97
    assert off(smooth)[edge].max() < 0.02
