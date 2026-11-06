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


def side_on_picture(raised, length=1000, thick=70, finger=None):
    """A dark bar seen from the side on a light, noisy background, with parts standing out on its right. raised: [(from, to, height in pixels)]."""
    from PIL import Image

    rng = np.random.default_rng(1)
    h, w = length + 200, 500
    a = rng.normal(170, 9, (h, w, 3)).clip(0, 255)
    left = w // 2 - thick // 2
    a[100 : 100 + length, left : left + thick] = 30
    for start, stop, height in raised:
        a[100 + start : 100 + stop, left + thick : left + thick + height] = 30
    if finger is not None:  # a finger across the strip: another colour, its long sides run across
        a[100 + finger : 100 + finger + 60, left + thick :] = (205, 150, 130)
    on = np.zeros((h, w), bool)
    on[100 : 100 + length, left : left + thick] = True
    small = np.asarray(Image.fromarray(on.astype(np.uint8) * 255).resize((w // 4, h // 4), Image.NEAREST))
    return Image.fromarray(a.astype(np.uint8)), small


def test_side_view_measures_what_stands_out_and_ignores_a_finger(tmp_path):
    from app import sideview as SV

    parts = [(480, 540, 16), (200, 260, 5)]  # 3 mm and 1 mm at 192 mm long
    names = []
    masks = {}
    for k, finger in enumerate((300, 700, 800)):
        picture, small = side_on_picture(parts, finger=finger)
        name = f"img_{k:04d}.jpg"
        picture.save(tmp_path / name, quality=97)
        names.append(name)
        masks[name] = small
    got = SV.measure(tmp_path, masks, {n: 0.07 for n in names}, 0.07)
    assert got is not None
    busy, quiet = got["busy"], got["quiet"]
    at = lambda a, b: busy[int(a * SV.POINTS) : int(b * SV.POINTS)]  # noqa: E731
    assert abs(float(at(0.49, 0.53).max()) - 0.016) < 0.003  # the 3 mm part
    assert abs(float(at(0.21, 0.25).max()) - 0.005) < 0.002  # the 1 mm part
    assert float(np.median(at(0.31, 0.36))) < 0.001  # nothing where a finger crosses in one frame
    assert float(np.median(quiet)) < 0.001  # the other face is flat


def test_side_view_needs_enough_frames(tmp_path):
    from app import sideview as SV

    picture, small = side_on_picture([(480, 540, 16)])
    picture.save(tmp_path / "img_0000.jpg", quality=97)
    assert SV.measure(tmp_path, {"img_0000.jpg": small}, {"img_0000.jpg": 0.07}, 0.07) is None
    # A frame that looks at the wide face is not used for this.
    assert SV.side_on_frames({"a": 0.07, "b": 0.08, "c": 0.23, "d": 0.0}, 0.07) == ["a", "b"]


def test_measured_heights_raise_the_real_buttons_and_flatten_the_rest():
    from app import photo3d as P3
    from app import relief as RL

    rel = np.zeros((1000, 230), np.float32)
    on = np.ones_like(rel, bool)
    rel[480:540, 40:100] = 0.0065 / RL.STRENGTH  # the 3 mm button, for which the AI gave 1.2 mm
    rel[200:240, 40:100] = 0.0040 / RL.STRENGTH  # a 1 mm button, for which it gave 0.8 mm
    rel[800:860, 40:100] = 0.0030 / RL.STRENGTH  # something on the plain end that the side view does not show
    rel[600:640, 40:100] = -0.002 / RL.STRENGTH  # a groove
    measured = np.full(400, 0.001, np.float32)
    measured[192:216] = 0.0156
    measured[80:96] = 0.0052
    out = P3.with_side_heights(rel, on, measured) * RL.STRENGTH
    assert abs(float(out[510, 70]) - 0.0156) < 0.002
    assert abs(float(out[220, 70]) - 0.0052) < 0.001
    assert float(out[830, 70]) < 0.0003
    assert -0.0021 < float(out[620, 70]) <= 0  # a groove never gets deeper
    # The other face can only get flatter.
    calm = P3.with_side_heights(rel, on, measured, 1.0) * RL.STRENGTH
    assert float(calm[510, 70]) <= 0.0066


def test_side_profile_finds_its_place_and_direction_on_the_detail():
    from app import photo3d as P3
    from app import sideview as SV

    rows = np.zeros(1000, np.float32)
    rows[480:520] = 0.0065
    rows[200:240] = 0.004
    rows[700:760] = 0.004
    profile = np.zeros(SV.POINTS, np.float32)
    for start, stop, height in ((480, 520, 0.0156), (200, 240, 0.005), (700, 760, 0.005)):
        profile[start * SV.POINTS // 1000 : stop * SV.POINTS // 1000] = height
    fit, upside, shift, _ = P3.fit_side(rows, np.roll(profile[::-1], 8))
    assert fit > 0.8
    assert upside
    placed = SV.placed(np.roll(profile[::-1], 8)[::-1], shift)
    assert abs(int(np.argmax(placed)) / SV.POINTS - 0.5) < 0.03


def test_rim_goes_on_the_true_straight_line_and_the_mesh_stays_closed():
    import trimesh

    from app import photo3d as P3

    n = 900
    on = np.asarray(bar_mask(180, height=700, tilt=0.03, size=n)) > 127
    q = on[:-1, :-1] & on[1:, :-1] & on[:-1, 1:] & on[1:, 1:]
    s, t = P3.grid(n)
    s2, t2, exact = P3.rim_on_lines(s, t, on, q, 40.0)
    moved = np.hypot(s2 - s, t2 - t) * (n - 1)
    assert moved.max() < 2.0  # less than a cell or two
    assert moved[np.nonzero(ndimage_inner(q))].max() == 0  # the inside stays
    front = np.dstack([s2 * n, t2 * n, np.zeros((n, n))])
    back = np.dstack([s2 * n, t2 * n, np.full((n, n), 90.0)])
    plain = P3.closed_mesh(q, np.dstack([s * n, t * n, np.zeros((n, n))]), np.dstack([s * n, t * n, np.full((n, n), 90.0)]))
    snapped = P3.closed_mesh(q, front, back)
    assert trimesh.Trimesh(snapped[0], snapped[1], process=False).is_watertight

    def ripple(vertices):
        # The left side: how far its points stray from one line.
        v = vertices[(np.abs(vertices[:, 2]) < 1e-6) & (vertices[:, 1] > 200) & (vertices[:, 1] < 700) & (vertices[:, 0] < 450)]
        low = np.array([v[np.abs(v[:, 1] - y) < 1.0][:, 0].min() for y in np.arange(210, 690, 4.0)])
        return float(np.std(low - np.polyval(np.polyfit(np.arange(210, 690, 4.0), low, 1), np.arange(210, 690, 4.0))))

    assert ripple(snapped[0]) < 0.6 * ripple(plain[0])


def ndimage_inner(q):
    from scipy import ndimage

    return ndimage.binary_erosion(np.pad(q, ((0, 1), (0, 1))), iterations=3)


def test_rounded_edge_has_one_height_along_a_straight_rim():
    from app import photo3d as P3

    n = 900
    on = np.asarray(bar_mask(180, height=700, tilt=0.03, size=n)) > 127
    q = on[:-1, :-1] & on[1:, :-1] & on[:-1, 1:] & on[1:, 1:]
    s, t = P3.grid(n)
    _, _, exact = P3.rim_on_lines(s, t, on, q, 40.0)
    front, back = np.zeros((n, n), np.float32), np.full((n, n), 90.0, np.float32)
    rough = P3.rounded_edges(front, back, on, 1.0, 30.0, wall=True)[0]
    clean = P3.rounded_edges(front, back, on, 1.0, 30.0, wall=True, exact=exact)[0]
    rim = P3.rim_cells(q)
    rim[:300] = False
    rim[600:] = False  # the long sides only
    assert np.ptp(clean[rim]) < 0.5 * np.ptp(rough[rim])
    assert np.std(clean[rim]) < 0.01  # the wall starts at one height, so it shows no stripes


def d_section_mask(width=360, thick=110, dome=0.6, size=(480, 270), turn=0.0):
    """Маска (четвърт от кадъра) на предмет, гледан по оста: плоско лице долу и заоблен гръб горе (D)."""
    from PIL import Image, ImageDraw

    big = Image.new("L", (size[1] * 4, size[0] * 4), 0)
    cx, cy = size[1] * 2, size[0] * 2
    xs = np.linspace(-width / 2, width / 2, 200)
    flat_h = thick * (1 - dome)  # права стена до тук, после елипса
    top = cy + thick / 2 - flat_h - (thick * dome) * np.sqrt(np.clip(1 - (2 * xs / width) ** 2, 0, 1))
    points = [(cx + x, t) for x, t in zip(xs, top)] + [(cx + width / 2, cy + thick / 2), (cx - width / 2, cy + thick / 2)]
    ImageDraw.Draw(big).polygon(points, fill=255)
    big = big.rotate(turn, Image.BILINEAR)
    return np.asarray(big.resize((size[1], size[0]), Image.BILINEAR))


def bar_quarter_mask(length, width, size=(480, 270), turn=0.0):
    from PIL import Image, ImageDraw

    big = Image.new("L", (size[1] * 4, size[0] * 4), 0)
    cx, cy = size[1] * 2, size[0] * 2
    ImageDraw.Draw(big).rectangle([cx - width / 2, cy - length / 2, cx + width / 2, cy + length / 2], fill=255)
    big = big.rotate(turn, Image.BILINEAR)
    return np.asarray(big.resize((size[1], size[0]), Image.BILINEAR))


def test_cross_section_comes_from_the_frames_along_the_axis():
    from app import segment as S
    from app import sideview as SV

    masks = {}
    for k in range(4):  # лицето и гърбът: дълги 1200, широки 300
        masks[f"face_{k}"] = bar_quarter_mask(1200, 300, turn=3 * k)
    for k in range(3):  # отстрани: дебелина 100
        masks[f"side_{k}"] = bar_quarter_mask(1200, 100, turn=2 * k)
    for k in range(3):  # по оста: широко 300, дебело 100, заоблено на 60% от дебелината
        masks[f"end_{k}"] = d_section_mask(300, 100, 0.6, turn=4 * k)
    masks["end_hand"] = d_section_mask(300, 190, 0.2)  # с ръка: много по-дебело
    ratios = {n: S.outline_ratio(m) for n, m in masks.items()}
    thin = min(ratios[n] for n in masks if n.startswith("side"))
    ends = SV.end_on_frames(masks, ratios, thin)
    assert {"end_0", "end_1", "end_2"} <= set(ends) and not any(n.startswith(("face", "side")) for n in ends)
    got = SV.cross_section(masks, ratios, thin)
    assert got is not None and int(got["section_frames"]) == 3  # the frame with the hand disagrees
    flat, dome = got["section_flat"], got["section_dome"]
    assert float(flat.max()) < 0.02  # the flat face is flat
    assert abs(float(got["section_thick"]) - 1 / 3) < 0.03
    deep = float(dome[[0, -1]].mean() - dome.min())  # 60% of the thickness, a fifth of the width, at the very edge
    assert 0.08 < deep < 0.17
    assert float(dome[len(dome) // 2]) < 0.01  # the top of the rounded side


def test_thickness_along_the_length_keeps_the_slanted_end_and_drops_a_bad_frame():
    from PIL import Image, ImageDraw

    from app import segment as S
    from app import sideview as SV

    def side(slant=True, bump=False):
        big = Image.new("L", (1080, 1920), 0)
        d = ImageDraw.Draw(big)
        top = 280 if slant else 400
        d.polygon([(500, top), (580, 400), (580, 1600), (500, 1600)], fill=255)  # дебелина 80, горе скосено
        if bump:
            d.rectangle([540, 700, 580, 1100], fill=0)  # палец пред ръба: маската е непълна
        return np.asarray(big.resize((270, 480), Image.BILINEAR))

    masks = {"a": side(), "b": side(), "c": side(bump=True), "f": bar_quarter_mask(1240, 300), "g": bar_quarter_mask(1240, 300, turn=5)}
    ratios = {n: S.outline_ratio(m) for n, m in masks.items()}
    got = SV.side_thickness(masks, ratios, min(ratios["a"], ratios["b"]))
    assert got is not None and int(got["side_frames"]) == 2
    t = got["side_thickness"]
    assert float(t[:10].mean()) < 0.5 and float(t[len(t) // 2]) == 1.0  # the slanted top and the full middle
    assert float(t.max()) <= 1.0


def test_body_has_a_flat_front_and_the_measured_rounded_back():
    from app import photo3d as P3
    from app import sideview as SV

    n = 401
    on = np.zeros((n, n), bool)
    on[40:360, 160:240] = True
    u = np.linspace(SV.SECTION_EDGE, 1 - SV.SECTION_EDGE, SV.SECTION_POINTS)
    dome = 0.25 * (1 - np.sqrt(1 - (2 * u - 1) ** 2))  # in widths, 0 at the middle
    along = np.ones(SV.POINTS, np.float32)
    along[: SV.POINTS // 20] = 0.3  # a slanted top
    front, back = P3.section_body(on, (np.zeros_like(u), dome, 0.3), along, 0.07, front_round=False)
    assert np.ptp(front[on]) == 0  # the front is flat
    assert abs(float(back[200, 200]) - 0.07) < 1e-3  # full thickness in the middle
    assert float(back[200, 162]) < 0.6 * 0.07  # thinner at the long sides: the back is rounded
    assert float(back[42, 200]) < 0.4 * 0.07  # and the top is slanted
    assert (back[on] > front[on]).all()
    # The other way round: the rounded face looks at the camera.
    f2, b2 = P3.section_body(on, (np.zeros_like(u), dome, 0.3), None, 0.07, front_round=True)
    assert np.ptp(b2[on]) == 0 and float(f2[200, 162]) > float(f2[200, 200])
    # Without a cross-section: a slab of the measured thickness.
    f3, b3 = P3.section_body(on, None, None, 0.07, False)
    assert np.allclose(b3[on] - f3[on], 0.07)


def test_rounded_side_is_told_by_the_surface_directions():
    from app import photo3d as P3

    h, w = 400, 100
    inside = np.ones((h, w), bool)
    x = np.linspace(-1, 1, w)
    slope = np.broadcast_to(2 * 0.4 * x, (h, w))  # a dome: the surface turns away towards both long sides
    n = np.stack([slope, np.zeros((h, w)), -np.ones((h, w))], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    flat = np.stack([np.full((h, w), 0.1), np.zeros((h, w)), -np.ones((h, w))], -1)  # only tilted
    flat /= np.linalg.norm(flat, axis=-1, keepdims=True)
    domed, plain = P3.across_profile(n, inside), P3.across_profile(flat, inside)
    assert domed[len(domed) // 2] > 0.15 and abs(plain[len(plain) // 2]) < 0.01
    section, front_round, _ = P3.body_section({}, inside, 0.07, plain, domed)
    assert section is not None and not front_round  # the back is the rounded side
    assert P3.body_section({}, inside, 0.07, plain, plain)[0] is None  # two flat faces: a slab


def test_light_colour_comes_off_a_white_object_and_a_red_one_stays_red(tmp_path):
    from PIL import Image

    from app import texture as TX

    rng = np.random.default_rng(5)
    on = np.zeros((400, 300), bool)
    on[50:350, 75:225] = True
    shade = rng.uniform(0.3, 1.0, (400, 300))
    warm = 230 * shade[..., None] + (1 - shade[..., None]) * np.array([90.0, 60.0, 35.0])  # shadows warm, light places neutral
    white = np.where(on[..., None], warm, 30).clip(0, 255).astype(np.uint8)
    red = np.where(on[..., None], np.stack([shade * 200 + 40, shade * 30 + 10, shade * 30 + 10], -1), 30).clip(0, 255).astype(np.uint8)
    frames = []
    for k, picture in enumerate((white, white, white, white)):
        Image.fromarray(picture).save(tmp_path / f"w{k}.png")
        frames.append((tmp_path / f"w{k}.png", (on[::4, ::4] * 255).astype(np.uint8)))
    assert TX.neutral_object(frames)
    for k in range(4):
        Image.fromarray(red).save(tmp_path / f"r{k}.png")
    assert not TX.neutral_object([(tmp_path / f"r{k}.png", (on[::4, ::4] * 255).astype(np.uint8)) for k in range(4)])
    fixed = TX.light_fix(white, on, True)(white)[on].astype(float)
    spread = np.median(fixed.max(1) - fixed.min(1))
    assert spread < 6  # no colour cast left
    assert np.median(fixed.mean(1)) > np.median(white[on].astype(float).mean(1))  # and brighter
    kept = TX.light_fix(red, on, False)(red)[on].astype(float)
    assert np.median(kept[:, 0]) > 2 * np.median(kept[:, 1])  # the red object stays red


def test_back_is_the_frame_of_the_same_size_not_the_end_of_the_object(tmp_path):
    from PIL import Image

    from app import photo3d as P3

    def frame(name, picture, box):
        big = np.full((1600, 1000, 3), 40, np.uint8)
        on = np.zeros((1600, 1000), bool)
        x0, y0 = box
        big[y0 : y0 + picture.shape[0], x0 : x0 + picture.shape[1]] = picture
        on[y0 : y0 + picture.shape[0], x0 : x0 + picture.shape[1]] = True
        Image.fromarray(big).save(tmp_path / name, quality=95)
        return tmp_path / name, np.asarray(Image.fromarray(on.astype(np.uint8) * 255).resize((250, 400), Image.BILINEAR))

    def textured(seed):
        noise = np.random.default_rng(seed).uniform(0, 1, (120, 30))
        grey = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).resize((240, 960), Image.NEAREST), np.float32)
        return np.repeat((150 + 0.4 * (grey - 128))[..., None], 3, 2).clip(0, 255).astype(np.uint8)

    face = textured(1)
    chosen = frame("a_front.jpg", face, (380, 320))
    again = frame("b_front2.jpg", face, (390, 330))  # the same side
    back = frame("c_back.jpg", textured(2)[:, ::-1], (380, 320))  # the other side, mirrored
    end = frame("d_end.jpg", np.full((80, 240, 3), 170, np.uint8), (380, 760))  # the end, seen along the axis
    got = P3.opposite_frame(chosen[0], [again, back, end], chosen[1])
    assert got is not None and got[0].name == "c_back.jpg"
    assert P3.opposite_frame(chosen[0], [again, end], chosen[1]) is None


def test_a_finger_in_one_frame_is_taken_from_the_other_views():
    from PIL import Image

    from app import photo3d as P3

    rng = np.random.default_rng(3)
    noise = rng.uniform(0, 1, (120, 60))
    face = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).resize((300, 600), Image.NEAREST).convert("RGB"))
    big = np.full((800, 600, 3), 30, np.uint8)
    big[100:700, 150:450] = face
    inside = np.zeros((800, 600), bool)
    inside[100:700, 150:450] = True
    finger = big.copy()
    finger[390:450, 120:240] = (150, 110, 95)  # a finger across the left side, not far from the colour of the face
    held = inside.copy()
    held[390:450, 120:150] = True  # it sticks out of the outline
    views = []
    for k in range(3):
        angle = (k - 1) * 2.0
        v = Image.fromarray(face).rotate(angle, Image.BICUBIC, fillcolor=(30, 30, 30))
        views.append((v, np.asarray(Image.fromarray(np.full((600, 300), 255, np.uint8)).rotate(angle, Image.BILINEAR)) > 127))
    clean, new, box, better, held = P3.without_fingers(Image.fromarray(finger), held, (120, 100, 450, 700), views)
    assert better in (0, 1, 2) and not held  # every other view shows the face uncovered
    got = np.asarray(clean).astype(float)
    assert np.abs(got[400:440, 170:230] - big[400:440, 170:230]).mean() < 25  # the face under the finger is back
    assert not new[400:440, 125:145].any()  # and the outline is straight again
    assert box[0] >= 148


def test_narrow_cracks_are_filled_and_raised_parts_stay():
    from app import mesh as M

    ball = trimesh.creation.icosphere(subdivisions=6, radius=50.0)
    v = ball.vertices.copy()
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    crack = (np.abs(v[:, 0]) < 1.5) & (v[:, 2] > 20)  # a thin cut across the top, 2 mm deep
    bump = np.linalg.norm(v - np.array([0.0, 50.0, 0.0]), axis=1) < 12  # a wide raised part, 2 mm high
    v[crack] -= 2.0 * n[crack]
    v[bump] += 2.0 * n[bump]
    rough = trimesh.Trimesh(v, ball.faces, process=False)
    out, share = M.filled_pits(rough)
    r = np.linalg.norm(out.vertices, axis=1)
    assert 0 < share < 0.05
    assert np.abs(r[crack] - 50).max() < 1.0  # the crack is mostly gone
    assert np.median(r[bump]) > 51.5  # the raised part stays
    assert out.is_watertight and len(out.faces) == len(ball.faces)


def test_kind_makes_flat_and_round_only_when_sure():
    from app import kind as K

    names = [k[0] for k in K.KINDS]
    cup = np.zeros((200, 200), bool)
    yy, xx = np.mgrid[:200, :200]
    cup[40:180] = np.abs(xx[40:180] - 100) < 30 + (yy[40:180] - 40) * 0.1  # a cup, a little wider at the bottom
    p = np.full(len(K.KINDS), 0.01)
    p[names.index("a cup")] = 0.7
    got = K.decide(p / p.sum(), cup)
    assert got["group"] == "round" and got["name"] == "Чаша"
    # The same outline, but the picture could be many things: the usual shape.
    p = np.full(len(K.KINDS), 1.0)
    assert K.decide(p / p.sum(), cup)["group"] == "other"
    # A remote: flat with its usual thickness.
    p = np.full(len(K.KINDS), 0.005)
    p[names.index("a remote control")] = 0.9
    got = K.decide(p / p.sum(), cup)
    assert got["group"] == "flat" and got["thin"] == 0.09 and got["name"] == "Дистанционно"
    # "Round" with a lopsided outline (not turned around an axis): the usual shape.
    lopsided = np.zeros((200, 200), bool)
    lopsided[40:180, 60:100] = True
    lopsided[40:110, 100:190] = True
    p = np.full(len(K.KINDS), 0.01)
    p[names.index("a vase")] = 0.8
    assert K.decide(p / p.sum(), lopsided)["group"] == "other"


def test_round_body_is_the_outline_turned_and_the_handle_stays_thin():
    from app import photo3d as P3

    n = 201
    on = np.zeros((n, n), bool)
    on[40:160, 70:131] = True  # a can 61 cells wide
    on[70:110, 131:150] = True  # with a handle on the right
    half = P3.revolved_half(on)
    assert abs(float(half[100, 100]) - 30.5) < 1.5  # the middle is as deep as the can is wide
    assert float(half[100, 72]) < 0.4 * float(half[100, 100])  # thin at the outline
    assert 0 < float(half[90, 140]) < 0.2 * float(half[100, 100])  # the handle is a thin plate, not turned around
    assert float(half[20, 100]) == 0.0


def test_rolling_ball_fills_slits_and_leaves_wide_grooves():
    from app import mesh as M

    ball = trimesh.creation.icosphere(subdivisions=6, radius=50.0)
    v = ball.vertices.copy()
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    slit = (np.abs(v[:, 0]) < 0.6) & (v[:, 2] > 20)  # a slit 1 mm wide, 3 mm deep: a scanning crack
    groove = (np.abs(v[:, 1]) < 6) & (v[:, 2] < -20)  # a groove 12 mm wide, 2 mm deep: part of the shape
    v[slit] -= 3.0 * n[slit]
    v[groove] -= 2.0 * n[groove]
    out, share = M.closed_cracks(trimesh.Trimesh(v, ball.faces, process=False))
    r = np.linalg.norm(out.vertices, axis=1)
    assert 0 < share < 0.05
    assert np.median(r[slit]) > 49.0  # the slit is filled
    assert abs(np.median(r[groove]) - 48.0) < 0.2  # the groove stays
    assert out.is_watertight and len(out.faces) == len(ball.faces)
