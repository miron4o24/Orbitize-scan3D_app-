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


def bumpy(mesh, scale, seed=0):
    import trimesh

    rng = np.random.default_rng(seed)
    moved = mesh.vertices + mesh.vertex_normals * rng.normal(scale=scale, size=(len(mesh.vertices), 1))
    return trimesh.Trimesh(moved, mesh.faces, process=False)


def test_snapping_makes_a_bumpy_can_an_exact_cylinder_with_a_flat_top():
    import trimesh

    can = trimesh.creation.cylinder(radius=1.0, height=1.6, sections=96)
    for _ in range(3):
        can = can.subdivide()
    noisy = bumpy(can, 0.008)
    out, shapes = F.snap_shapes(noisy, np.ones(len(noisy.vertices), bool))
    assert sorted(s[0] for s, _ in shapes) == ["flat", "flat", "round"]
    x, y, z = out.vertices.T
    side = np.abs(can.vertices[:, 2]) < 0.7
    assert np.abs(np.hypot(x, y)[side] - 1.0).max() < 0.004  # страната е на цилиндъра (шумът беше 0,008)
    top = (can.vertices[:, 2] > 0.79) & (np.hypot(*can.vertices[:, :2].T) < 0.9)
    assert np.ptp(z[top]) < 0.004  # капакът е плосък и напречен на оста


def test_snapping_leaves_a_ball_and_the_trusted_part_alone():
    import trimesh

    ball = bumpy(trimesh.creation.icosphere(subdivisions=5), 0.008)
    out, shapes = F.snap_shapes(ball, np.ones(len(ball.vertices), bool))
    assert shapes == [] and np.array_equal(out.vertices, ball.vertices)
    # Само върховете, където държи обвивката, се търсят: без такива нищо не се мести.
    can = trimesh.creation.cylinder(radius=1.0, height=1.6, sections=96)
    for _ in range(3):
        can = can.subdivide()
    noisy = bumpy(can, 0.008)
    out, shapes = F.snap_shapes(noisy, np.zeros(len(noisy.vertices), bool))
    assert shapes == [] and np.array_equal(out.vertices, noisy.vertices)


def test_only_an_object_whose_depth_disagrees_is_held_to_its_outlines():
    assert F.untrusted(0.15) == 1.0  # метална колонка: кадрите са съгласни за 14-18% от повърхността
    assert F.untrusted(0.67) == 0.0  # фигурка от матова пластмаса
    assert F.untrusted(0.91) == 0.0  # плетена тиква
    assert 0 < F.untrusted(0.4) < 1


def test_the_correction_reaches_far_from_the_measured_pixels():
    # Измерен е само пръстен; корекцията вътре и далеч извън него е тази на пръстена, а не нула (преди: кратер).
    yy, xx = np.mgrid[:240, :240]
    ring = np.abs(np.hypot(xx - 120, yy - 120) - 40) < 3
    field = F.spread(np.where(ring, 1.0, 0.0), ring, 4.0)
    assert abs(field[120, 120] - 1) < 0.05 and abs(field[120, 230] - 1) < 0.05 and abs(field[5, 5] - 1) < 0.05


def test_each_tone_takes_its_correction_from_measured_pixels_of_its_own_tone():
    # Тъмна коса (измерена, корекция 0) до светла кожа, от която е измерена само ивица далеч от косата (корекция 1):
    # кожата до косата взема корекцията на кожата.
    tone = np.full((200, 200), 0.1, np.float32)
    tone[:, 60:] = 0.8
    good = np.zeros((200, 200), bool)
    good[:, :60] = True
    good[:, 110:120] = True
    r = np.where(tone > 0.5, 1.0, 0.0) * good
    field = F.toned_field(r, good, tone, 4.0)
    assert field[100, 70] > 0.9 and abs(field[100, 30]) < 0.1
    assert F.spread(r, good, 4.0)[100, 70] < 0.5  # без тона: корекцията на косата


def island_scene(dent: float):
    """Наклонена стена с неизмерен кръг (остров), в който AI дълбочината е вдлъбната с `dent`."""
    yy, xx = np.mgrid[:800, :800]
    wall = 0.5 - 0.0001 * xx  # обратна дълбочина: равнина
    island = np.hypot(xx - 400, yy - 400) < 20
    full = np.where(island, 1 / (1 / wall + dent), wall)
    inner = np.zeros((800, 800), bool)
    inner[40:-40, 40:-40] = True
    return full, wall, ~island & inner, inner, island


def test_a_shallow_dent_in_an_unmeasured_island_becomes_the_surface_through_its_rim():
    full, wall, good, inner, island = island_scene(0.004)  # 2,5% от ширината на острова
    out = F.islands(full, wall * good, good, inner, 500.0)
    core = np.hypot(*(np.mgrid[:800, :800] - 400)) < 12
    assert np.abs(1 / out[core] - 1 / wall[core]).max() < 0.0008  # по наклона на стената
    away = np.hypot(*(np.mgrid[:800, :800] - 400)) > 30
    assert np.allclose(out[away], full[away], rtol=0, atol=1e-9)


def test_a_deep_hole_in_an_unmeasured_island_is_kept():
    full, wall, good, inner, _ = island_scene(0.05)  # 31% от ширината: истинска дупка
    assert np.array_equal(F.islands(full, wall * good, good, inner, 500.0), full)


def test_measured_depth_far_from_the_fitted_ai_depth_asks_to_vote_on_its_own():
    # Стена на 2 с процеп, през който се вижда нещо на 2,5: AI дълбочината е гладка през процепа.
    K = np.array([[500.0, 0, 100], [0, 500.0, 100], [0, 0, 1]])
    wall = np.broadcast_to(2.0 + 0.001 * np.arange(200, dtype=np.float32), (200, 200))  # леко наклонена
    depth = wall.copy()
    depth[:, 100:106] += 0.5
    dm = F.DepthMap("a.jpg", (200, 200), K, np.eye(3), np.zeros(3), depth, np.ones((200, 200), np.float32))
    disp = 1 / wall
    every = np.ones((200, 200), bool)
    inner = every.copy()
    inner[:, :20] = False  # ивицата покрай очертанието
    _, _, free, own = F.completed(dm, disp, (0, 0, 200, 200), inner, every, ~every, None, every, 0.1)
    assert (own[:, 100:106] == depth[:, 100:106]).all() and (own[:, 30:95] == 0).all()
    # ивицата: иска да гласува сама, а ако съседните кадри не я потвърдят, остава само глас за празно пред нея
    assert (own[:, :20] == depth[:, :20]).all() and (free[:, :20] == depth[:, :20]).all()
