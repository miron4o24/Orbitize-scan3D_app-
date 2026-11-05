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


def test_surface_directions_become_real_height():
    from app import relief as R

    # A flat plate with a dome: its true height and directions, z pointing from the camera to the object.
    yy, xx = np.mgrid[:200, :200].astype(np.float32)
    dome = 8.0 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 14.0**2))
    gy, gx = np.gradient(dome)
    n = np.stack([-gx, -gy, -np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    inside = np.zeros((200, 200), bool)
    inside[20:180, 20:180] = True
    high = R.height_from_normals(n, inside)
    rise = high[100, 100] - high[100, 40]
    assert 6.0 < rise < 10.0  # a dome of 8 pixels comes out about 8 pixels, and outward
    rel = R.relief_from_normals(n, inside, 200.0)
    assert rel[100, 100] > 0.5 and abs(rel[40, 40]) < 0.2  # the dome rises, the flat part stays flat


def test_straight_sides_become_straight_and_curves_stay():
    from PIL import Image, ImageDraw

    from app import photo3d as P3

    # A tilted bar whose long sides wobble by 1 pixel, as the real masks do at this size.
    rng = np.random.default_rng(0)
    top = 60 + np.arange(0, 500, 10)
    left = [(140 + 0.1 * (y - 60) + rng.uniform(-1, 1), y) for y in top]
    right = [(220 + 0.1 * (y - 60) + rng.uniform(-1, 1), y) for y in top[::-1]]
    a = Image.new("L", (400, 600), 0)
    ImageDraw.Draw(a).polygon(left + right, fill=255)
    wobbly = np.asarray(a) > 127
    out = P3.straightened(wobbly)
    rows = np.arange(120, 500)
    edge = np.array([np.flatnonzero(out[y])[0] for y in rows], float)
    assert np.abs(edge - np.polyval(np.polyfit(rows, edge, 1), rows)).max() < 0.8  # now a straight line
    assert abs(int(out.sum()) - int(wobbly.sum())) < 0.02 * wobbly.sum()  # about the same size
    # A circle, however large, has no straight run and stays as it is.
    yy, xx = np.mgrid[:600, :600]
    circle = np.hypot(yy - 300, xx - 300) < 280
    assert np.array_equal(P3.straightened(circle), circle)


def bar_mask(width, height=600, tilt=0.0, size=800):
    from PIL import Image, ImageDraw

    a = Image.new("L", (size, size), 0)
    c = size / 2
    d = tilt * height / 2
    ImageDraw.Draw(a).polygon([(c - width / 2 - d, c - height / 2), (c + width / 2 - d, c - height / 2),
                               (c + width / 2 + d, c + height / 2), (c - width / 2 + d, c + height / 2)], fill=255)
    return np.asarray(a).copy()


def test_thickness_is_read_from_the_side_frames():
    from app import segment as S

    frames = {f"f{i}": bar_mask(w, tilt=0.1) for i, w in enumerate([137, 137, 100, 60, 41, 42, 137])}
    assert abs(S.thinnest(frames) - 41 / 600) < 0.01
    # One wrong mask, far narrower than the rest, does not count as the thickness.
    frames["bad"] = bar_mask(12)
    assert abs(S.thinnest(frames) - 41 / 600) < 0.01
    # Fingers on part of the object do not change it.
    held = bar_mask(41)
    held[500:600, 300:500] = 255
    assert abs(S.outline_ratio(held) - 41 / 600) < 0.01
    assert S.thinnest({"only": bar_mask(41)}) is None


def test_a_wedge_shape_is_much_thicker_than_the_outlines_show():
    from app import segment as S

    rng = np.random.default_rng(0)
    slab = rng.uniform([-96, -22, -6.5], [96, 22, 6.5], (20000, 3))
    assert abs(S.points_ratio(slab) - 13 / 192) < 0.01
    wedge = rng.uniform([-96, -40, -24], [96, 40, 24], (20000, 3))
    assert S.points_ratio(wedge) > 3 * 13 / 192


def test_flat_body_has_the_measured_thickness():
    from app import photo3d as P3

    n = 501
    on = np.zeros((n, n), bool)
    on[50:450, 200:290] = True
    focal = 0.5 / np.tan(np.radians(20))
    front, back = P3.flat_layers(on, 0.068, 1.9, focal)
    cell = 1.9 / (focal * (n - 1))
    assert abs((back - front)[250, 250] / (399 * cell) - 0.068) < 1e-3
    assert np.ptp(front) == 0 and np.ptp(back) == 0  # the side view is a straight line


def test_same_side_has_shared_features_and_the_other_side_has_none():
    from PIL import Image

    from app import photo3d as P3

    rng = np.random.default_rng(1)
    noise = rng.uniform(0, 1, (60, 30))
    face = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).resize((300, 600), Image.NEAREST))
    plain = np.full((600, 300), 25, np.uint8)
    on = np.ones((600, 300), bool)
    a = P3.features(Image.fromarray(face), on)
    b = P3.features(Image.fromarray(face).rotate(4, Image.BICUBIC, fillcolor=128), on)
    assert P3.shared_features(a, b) >= P3.SAME_FEATURES
    assert P3.features(Image.fromarray(plain), on) is None
    assert P3.shared_features(a, None) == 0


def test_flat_relief_keeps_buttons_and_drops_swellings_and_deep_cuts():
    from app import photo3d as P3
    from app import relief as RL

    n = 400
    yy, xx = np.mgrid[:n, :n]
    inside = np.ones((n, n), bool)
    swelling = 2.0 * np.exp(-((yy - 60.0) ** 2 + (xx - 60.0) ** 2) / (2 * 60.0**2))
    rel = swelling.copy()
    rel[200:230, 200:230] += 1.0  # a button
    rel[300:304, 40:360] -= 10.0  # a seam, far too deep
    out = P3.flat_relief(rel, inside, n)
    assert out[215, 215] - out[215, 170] > 0.6  # the button stays
    assert np.ptp(out[40:80, 40:80]) < 0.5 * np.ptp(swelling[40:80, 40:80])  # the swelling goes
    assert out.min() >= -P3.FLAT_RECESS / RL.STRENGTH - 1e-6  # the cut is limited


def test_flat_edge_leaves_a_straight_wall_and_no_ladder():
    from app import photo3d as P3

    n = 900
    on = np.asarray(bar_mask(180, height=700, tilt=0.03, size=n)) > 127
    front, back = np.full((n, n), 1.0), np.full((n, n), 1.2)
    f, b = P3.rounded_edges(front, back, on, 1 / 700, 0.015, wall=True)
    assert abs(f[450, 450] - 1.0) < 1e-6 and abs(b[450, 450] - 1.2) < 1e-6
    assert (b - f)[on].min() >= 0.2 - 2 * 0.015 - 1e-6  # the wall keeps the thickness up to the corner radius
    # Along a line parallel to the true straight edge, the height is the same all the way: no rungs.
    rows = np.arange(150.0, 750.0)
    line = 450 - 90 + 0.03 * (rows - 450) + 4
    along = ndimage.map_coordinates((b - f).astype(np.float32), [rows, line], order=1)
    assert np.ptp(along) < 0.0005


def test_flat_object_is_turned_to_lie_along_the_grid():
    from PIL import Image

    from app import photo3d as P3

    on = np.asarray(bar_mask(180, height=700, tilt=0.04, size=900)) > 127
    photo = Image.fromarray(np.where(on, 200, 30).astype(np.uint8)).convert("RGB")
    _, turned, box = P3.upright(photo, on)
    assert abs(abs(P3.lengthwise(turned)) - 90) < 0.5
    rows = np.arange(box[1] + 60, box[3] - 60)
    edge = np.array([np.flatnonzero(turned[y])[0] for y in rows])
    assert np.ptp(edge) <= 2  # the long side stays on nearly one column, so it has no long steps
    left, top, side = P3.centred_place(box)
    assert left <= box[0] and top <= box[1] and left + side >= box[2] and top + side >= box[3]


def test_other_views_of_the_same_side_cancel_a_mistake_in_one_frame():
    from PIL import Image

    from app import photo3d as P3

    rng = np.random.default_rng(3)
    noise = rng.uniform(0, 1, (120, 60))
    face = Image.fromarray((noise * 255).astype(np.uint8)).resize((300, 600), Image.NEAREST)
    on = np.ones((600, 300), bool)
    truth = np.zeros((600, 300), np.float32)
    truth[280:320, 100:200] = 1.0  # a button that the chosen frame has lost to motion blur
    chosen = truth.copy()
    chosen[280:320, 100:200] = 0.2
    views = []
    for k in range(3):
        angle = (k - 1) * 3.0
        view = face.rotate(angle, Image.BICUBIC, fillcolor=128)
        seen = np.asarray(Image.fromarray(truth).rotate(angle, Image.BILINEAR))
        views.append((view, on, seen))
    found = {id(v): s for v, _, s in views}
    out = P3.same_side_relief(chosen, face, on, [(v, i) for v, i, _ in views], lambda v, i: found[id(v)])
    assert out[300, 150] > 0.8  # the button is back
    assert abs(out[100, 150]) < 0.05  # flat stays flat
    # With no other view the frame's own detail stays as it is.
    assert P3.same_side_relief(chosen, face, on, [], lambda v, i: None) is chosen


def test_a_band_across_the_end_goes_and_a_button_there_stays():
    from app import photo3d as P3

    rel = np.zeros((1000, 230), np.float32)
    on = np.ones_like(rel, bool)
    rel[20:50, :] = 1.3  # a false ledge across the whole width
    rel[60:100, 90:140] = 1.0  # the power button
    out = P3.without_end_bands(rel, on, 1000)
    assert abs(out[35, 115]) < 0.05
    assert out[80, 115] > 0.9
    assert out[500, 115] == 0

