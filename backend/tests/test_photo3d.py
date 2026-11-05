import numpy as np
import trimesh
from scipy import ndimage


def test_two_layer_mesh_is_closed_even_with_a_hole_and_a_corner_touch():
    from app import photo3d as P

    n = 60
    yy, xx = np.mgrid[:n, :n]
    on = ((yy - 30) ** 2 + (xx - 30) ** 2) < 25**2
    on &= ((yy - 30) ** 2 + (xx - 22) ** 2) > 5**2
    on[10, 10] = on[11, 11] = True
    q = P.no_pinches(on[:-1, :-1] & on[1:, :-1] & on[:-1, 1:] & on[1:, 1:])
    labels, count = ndimage.label(q)
    if count > 1:
        q = labels == 1 + int(np.argmax(ndimage.sum_labels(q, labels, np.arange(1, count + 1))))
    cam = P.orbit_camera(1.9, 40.0)
    rays = cam.rays(*P.grid(n))
    vertices, faces, _ = P.closed_mesh(q, cam.center + rays * 1.8, cam.center + rays * 2.0)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert mesh.is_watertight
    assert mesh.volume > 0  # стените гледат навън
    assert mesh.euler_number == 0  # пръстен: дупката остава дупка


def test_camera_projection_undoes_the_rays():
    from app import photo3d as P

    cam = P.orbit_camera(1.9, 40.0, 90.0)
    s, t = P.grid(11)
    points = cam.center + cam.rays(s, t) * 1.7
    u, v, depth = cam.pixels(points.reshape(-1, 3))
    assert np.allclose(u, s.ravel()) and np.allclose(v, t.ravel()) and np.allclose(depth, 1.7)


def test_march_finds_both_sides_of_a_ball():
    from app import photo3d as P

    g = np.linspace(-0.87, 0.87, 96)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    vol = (0.5 - np.sqrt(x**2 + y**2 + z**2)) * 100
    near, far = P.march(vol, 0.87, 0.0, P.orbit_camera(1.9, 40.0), *P.grid(65), 200)
    assert abs(near[32, 32] - 1.4) < 0.02 and abs(far[32, 32] - 2.4) < 0.02
    assert np.isnan(near[0, 0])


def test_background_stays_tells_a_still_phone_from_a_moving_one():
    from app import video as V

    rng = np.random.default_rng(0)
    room = rng.random((128, 192)).astype(np.float32) * 255
    room = ndimage.gaussian_filter(room, 1.0)
    cover = np.zeros(room.shape, bool)
    cover[:, 80:112] = True
    still = [np.roll(room, (k, -k), (0, 1)) for k in range(4)]  # само трептене на ръката
    assert V.background_stays(still, [cover] * 4) > V.STILL_SCORE
    moved = [ndimage.gaussian_filter(rng.random(room.shape).astype(np.float32) * 255, 1.0) for _ in range(4)]
    assert V.background_stays(moved, [cover] * 4) < V.STILL_SCORE


def test_relief_marks_a_raised_button_and_leaves_the_flat_part():
    from app import relief as R

    d = np.zeros((200, 200), np.float32)
    for y in range(60, 140, 32):
        for x in range(60, 140, 32):
            d[y : y + 16, x : x + 16] = 1.0  # копчета, всяко по-тясно от лентата на детайла
    inside = np.ones((200, 200), bool)
    rel = R.unit_relief(d, inside, 200.0)
    assert rel[68, 68] > 0.5  # средата на копче е високо, а не само ръбът му
    assert abs(rel[15, 15]) < 0.2  # равното далеч от копчетата остава равно


def test_masked_blur_at_low_resolution_matches_the_full_one():
    from app import relief as R

    yy, xx = np.mgrid[:240, :300].astype(np.float32)
    a = 0.02 * xx + 0.01 * yy
    inside = np.zeros(a.shape, bool)
    inside[20:220, 30:270] = True
    w = inside.astype(np.float32)
    full = ndimage.gaussian_filter(np.where(inside, a, 0).astype(np.float32), 24, truncate=3.0) / np.maximum(ndimage.gaussian_filter(w, 24, truncate=3.0), 1e-3)
    fast = R.masked_blur(a, inside, 24)
    assert fast.shape == a.shape
    assert np.abs(fast - full)[inside].max() < 0.15


def test_doubling_keeps_the_values_and_fills_between_them():
    from app import fuse as F

    a = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    b = F.doubled(a)
    assert b.shape == (5, 5, 5)
    assert np.array_equal(b[::2, ::2, ::2], a)
    assert b[1, 0, 0] == 0.5 * (a[0, 0, 0] + a[1, 0, 0])
