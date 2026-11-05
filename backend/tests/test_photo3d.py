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


def test_white_balance_reads_only_the_background(tmp_path):
    from PIL import Image

    from app import texture as TX

    # Топла светлина върху сив фон, с червен предмет в средата.
    a = np.full((120, 160, 3), [190, 150, 110], np.uint8)
    a[30:90, 50:110] = [200, 30, 30]
    Image.fromarray(a).save(tmp_path / "warm.png")
    half = np.zeros((30, 40), np.uint8)
    half[7:23, 12:28] = 255
    gain = TX.white_gain(tmp_path, ["warm.png"], {"warm.png": half})
    assert gain[0] < 1 < gain[2]
    # Предмет, запълнил кадъра: светлината не може да се отдели от цвета му.
    full = np.full((30, 40), 255, np.uint8)
    assert np.array_equal(TX.white_gain(tmp_path, ["warm.png"], {"warm.png": full}), np.ones(3))
    # Сиво си остава сиво, а алфата не се променя.
    grey = np.array([[[128, 128, 128, 200]]], np.uint8)
    assert np.array_equal(TX.balanced(grey, np.ones(3)), grey)


def test_rim_does_not_eat_the_range_of_the_buttons():
    from app import relief as R

    # Плосък предмет със заобляне към очертанието (два пъти по-силно от копчетата, както на истинско дистанционно).
    inside = np.zeros((200, 200), bool)
    inside[20:180, 20:180] = True
    d = -0.6 * np.exp(-ndimage.distance_transform_edt(inside) / 12.0)
    yy, xx = np.mgrid[:200, :200]
    spots = [(y, x) for y in (60, 80, 100, 120, 140) for x in (68, 100, 132)]
    for y, x in spots:
        d = d + 0.3 * (np.hypot(yy - y, xx - x) < 6)
    rel = R.unit_relief(d.astype(np.float32), inside, 200.0)
    assert np.mean([rel[s] for s in spots]) > 0.5  # копчетата получават обхвата
    assert abs(rel[100, 45]) < 0.2  # равното между копчетата и ръба остава равно
    assert abs(rel[30, 30]) < 0.6  # ъгълът не изскача над копчетата
    assert abs(rel[100, 26]) < 0.3  # и край очертанието няма улей, който го няма на предмета


def test_tilted_back_frame_is_straightened_to_lie_like_the_photo():
    from PIL import Image

    from app import photo3d as P3

    def bar(tilt):
        a = Image.new("L", (400, 400))
        a.paste(255, (170, 60, 230, 340))
        return np.asarray(a.rotate(tilt, Image.BICUBIC)) > 127

    front = bar(0)
    photo = Image.new("RGB", (400, 400))
    _, inside, box = P3.turned_like(photo, bar(12), P3.lengthwise(front))
    crop = inside[box[1] : box[3], box[0] : box[2]]
    assert abs(abs(P3.lengthwise(crop)) - abs(P3.lengthwise(front))) < 1.5
    assert crop.shape[1] < 80  # лежи изправен, а не по диагонал в кутията си
    # Кръгъл предмет няма посока на дължината и не се върти.
    yy, xx = np.mgrid[:200, :200]
    assert P3.lengthwise(np.hypot(yy - 100, xx - 100) < 60) is None


def test_failed_room_solve_still_tries_the_object_alone(monkeypatch, tmp_path):
    from app import ai

    tried = []

    def solve(*args, object_only=False, **kw):
        tried.append(object_only)
        if not object_only:
            raise ai.P.Failed("no initial pair")
        return "model", 30, [], {"a.jpg": None}, None

    monkeypatch.setattr(ai.P, "object_masks", lambda *a: {"a.jpg": None})
    monkeypatch.setattr(ai.P, "camera_stayed_still", lambda *a: False)
    monkeypatch.setattr(ai.P, "solve_cameras", solve)
    monkeypatch.setattr(ai.P, "note", lambda *a: None)
    monkeypatch.setattr(ai, "model_from_solved", lambda *a: a)
    got = ai.from_video(tmp_path, tmp_path, tmp_path, 30, True, None, None, {})
    assert tried == [False, True]
    assert got[6:8] == ("model", 30) and got[-1] is True  # камерите от втория опит, предметът е бил в ръка
