import numpy as np
import pytest
import trimesh

from app import mesh as M

RNG = np.random.default_rng(1)


W, H, F = 160, 120, 100.0  # размер на маската и фокусно разстояние на тестовите камери (маските са в пълен размер: мащаб 1)


def scene(tilt_deg=15.0, roll_deg=0.0, n_cams=24):
    """Единично кълбо върху маса при z=0, няколко носещи се точки, камери в орбита с точни маски на предмета; целият
    свят е наклонен."""
    v = RNG.normal(size=(5000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] > -0.3]  # камерите на около 18° над центъра никога не виждат долната страна
    sphere, sphere_n = v + [0, 0, 1], v.copy()
    r, a = 2.8 * np.sqrt(RNG.random(3000)), RNG.random(3000) * 2 * np.pi
    table = np.c_[r * np.cos(a), r * np.sin(a), np.zeros(3000)]
    table_n = np.tile([0, 0, 1.0], (3000, 1))
    floaters = RNG.uniform(-2.5, 2.5, size=(30, 3)) + [0, 0, 1.5]
    pts = np.vstack([sphere, table, floaters])
    nrm = np.vstack([sphere_n, table_n, RNG.normal(size=(30, 3))])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm[RNG.random(len(nrm)) < 0.5] *= -1  # неориентирани нормали, като при истински гъст облак

    T = M.rotation_to(np.array([np.sin(np.radians(tilt_deg)), 0, np.cos(np.radians(tilt_deg))]))
    Rs, ts, masks = [], [], []
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, n_cams, endpoint=False)):
        c = np.array([6 * np.cos(ang), 6 * np.sin(ang), 3.0])
        z = [0, 0, 1.0] - c
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)  # изображението надолу
        # Случайно завъртане на камерата около оста на гледане (като при данните на Middlebury): посоките нагоре на
        # изображенията вече не съвпадат.
        roll = np.radians(RNG.uniform(-roll_deg, roll_deg))
        x, y = np.cos(roll) * x + np.sin(roll) * y, -np.sin(roll) * x + np.cos(roll) * y
        R = np.stack([x, y, z]) @ T.T  # камера от наклонения свят
        Rs.append(R)
        ts.append(-R @ (T @ c))
        # Точна маска: пиксели, чийто лъч минава на разстояние до 1 от центъра на кълбото.
        u, w = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = np.stack([(u - W / 2) / F, (w - H / 2) / F, np.ones_like(u)], -1) @ np.stack([x, y, z])
        to_c = [0, 0, 1.0] - c
        dist = np.linalg.norm(np.cross(d, to_c), axis=-1) / np.linalg.norm(d, axis=-1)
        masks.append(np.where(dist < 1, 255, 0).astype(np.uint8))
    views = M.Views(np.array(Rs), np.array(ts), np.tile([F, F, W / 2, H / 2], (n_cams, 1)), [f"{i}" for i in range(n_cams)])
    return dict(
        points=pts @ T.T,
        normals=nrm @ T.T,
        colors=np.tile([0.6, 0.5, 0.4, 1.0], (len(pts), 1)),
        views=views,
        masks=masks,
        mask_scale=1,
    )


def test_rotation_to_maps_vector_onto_up():
    for a in ([1, 0, 0], [0, 0, -1], [0.3, -0.2, 0.9]):
        a = np.array(a, float) / np.linalg.norm(a)
        np.testing.assert_allclose(M.rotation_to(a) @ a, M.UP, atol=1e-9)


def test_silhouette_votes():
    s = scene(tilt_deg=0)
    pts = np.array([[0, 0, 1.0], [2.5, 0, 0], [0, 0, 50]])  # център на кълбото, маса до него, далеч отгоре
    n_in, n_fg = M.silhouette_votes(pts, s["views"], s["masks"], 1)
    assert list(n_in[:2]) == [24, 24] and n_in[2] == 0  # далеч отгоре: извън всички изображения
    assert n_fg[0] == 24 and n_fg[1] < 8


def test_consistent_frames_drops_wrongly_placed_ones():
    s = scene()
    v, masks = s["views"], s["masks"]
    assert M.consistent_frames(v, masks, 1).all()
    assert np.nanmin(M.silhouette_consistency(v, masks, 1)) > 0.8
    # Счупено решение за камерите: една камера е завъртяна на място в обратната посока, друга е изместена настрани с
    # размера на предмета.
    c = v.centers[3]
    v.R[3] = np.diag([-1.0, 1.0, -1.0]) @ v.R[3]
    v.t[3] = -v.R[3] @ c
    v.t[9] = v.t[9] + [2.0, 0, 0]
    ok = M.consistent_frames(v, masks, 1)
    assert not ok[3] and not ok[9] and ok.sum() == len(ok) - 2


@pytest.fixture(scope="module")
def cloud():
    return M.clean_cloud(**scene())


def test_clean_cloud_levels_scene_and_removes_table(cloud):
    assert cloud.ground_z is not None
    # Масата и носещите се точки ги няма; видимото кълбо (около 3250 точки) е почти запазено.
    assert 2900 < len(cloud.points) < 3400
    assert cloud.points[:, 2].min() > cloud.ground_z
    # Нормалите са насочени навън от центъра на кълбото (който е на 1 над земята).
    c = np.array([0, 0, cloud.ground_z + 1]) + 0 * cloud.points
    xy0 = cloud.points[:, :2].mean(0)
    c[:, :2] = xy0
    outward = np.einsum("ij,ij->i", cloud.normals, cloud.points - c) > 0
    assert outward.mean() > 0.97


def test_rolled_cameras_still_level_the_scene():
    c = M.clean_cloud(**scene(roll_deg=150))
    assert c.ground_z is not None
    assert c.points[:, 2].min() > c.ground_z


def test_ground_rejects_plane_with_points_below():
    # Широк плосък ръб по средата на висока колона: много точки, но голяма част от предмета е под него.
    r, a = RNG.random(4000), RNG.random(4000) * 2 * np.pi
    ledge = np.c_[r * np.cos(a), r * np.sin(a), np.full(4000, 1.0)]
    column = np.c_[RNG.uniform(-0.3, 0.3, (6000, 2)), RNG.uniform(0, 2, 6000)]
    assert M.find_ground(np.vstack([ledge, column]), thresh=0.01) is None


def test_object_not_found_is_a_clear_error():
    s = scene()
    s["masks"] = [m * 0 for m in s["masks"]]
    with pytest.raises(M.MeshError, match="отдели от фона"):
        M.clean_cloud(**s)


def test_make_printable(cloud):
    # Кълбото на сцената (радиус 1, центърът му на 1 над масата), слято леко под масата, както от сливането.
    centre = [*cloud.points[:, :2].mean(0), cloud.ground_z + 0.95]
    mesh = trimesh.creation.icosphere(subdivisions=4).apply_translation(centre)
    # Отделен носещ се фрагмент, който трябва да отпадне.
    junk = trimesh.creation.icosphere(radius=0.1).apply_translation([3, 3, cloud.ground_z + 2])
    mesh = trimesh.util.concatenate([mesh, junk])

    out = M.make_printable(mesh, M.floor_z(cloud), flat_base=True, longest_mm=80)
    rep = M.report(out)
    assert rep["watertight"]
    assert max(rep["dimensions_mm"]) == pytest.approx(80)
    assert len(out.split(only_watertight=False)) == 1
    # Плоска основа: много върхове точно при z = 0.
    assert out.bounds[0, 2] == pytest.approx(0)
    assert (np.abs(out.vertices[:, 2]) < 1e-6).sum() > 20


def test_hole_on_the_side_is_closed():
    m = trimesh.creation.icosphere(subdivisions=3).apply_translation([0, 0, 1])
    side = np.abs(m.triangles_center[:, 0] - 1) < 0.25  # пробива дупка от страната +X, на средна височина
    m.update_faces(~side)
    assert not m.is_watertight
    out = M.make_printable(m, floor=0.0, flat_base=False)
    assert out.is_watertight


@pytest.mark.parametrize("fmt", ["stl", "obj", "glb", "3mf"])
def test_export_formats(fmt):
    m = trimesh.creation.box(extents=[10, 20, 40]).apply_translation([0, 0, 20])
    data = M.export(m, fmt, longest_mm=100)
    assert len(data) > 100
    back = trimesh.load(trimesh.util.wrap_as_stream(data), file_type=fmt, force="mesh")
    assert max(back.extents) == pytest.approx(100, rel=1e-3)
    if fmt == "glb":
        assert back.extents[1] == pytest.approx(100, rel=1e-3)  # височина по оста Y нагоре в glTF


def test_paint_takes_each_side_from_the_frame_that_sees_it(tmp_path):
    from PIL import Image

    # Кутия между две камери по оста x, гледащи към нея: единият кадър е изцяло червен, другият изцяло син.
    box = trimesh.creation.box(extents=[1, 1, 1]).subdivide().subdivide()
    R_pos = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]])  # при +x, гледа по -x
    R_neg = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.0]])  # при -x, гледа по +x
    t = np.array([[0, 0, 4.0], [0, 0, 4.0]])
    views = M.Views(np.stack([R_pos, R_neg]), t, np.tile([200, 200, 100, 100.0], (2, 1)), ["a.jpg", "b.jpg"])
    for name, rgb in (("a.jpg", (220, 30, 30)), ("b.jpg", (30, 30, 220))):
        Image.new("RGB", (200, 200), rgb).save(tmp_path / name)
    M.paint(box, views, [tmp_path / "a.jpg", tmp_path / "b.jpg"], np.eye(3))
    c = box.visual.vertex_colors[:, :3].astype(int)
    front, back = box.vertices[:, 0] > 0.49, box.vertices[:, 0] < -0.49
    assert (c[front, 0] > 200).all() and (c[back, 2] > 200).all()  # всяка страна от собствения си кадър, а не от другия


def test_glb_for_the_phone_is_matte_and_coloured():
    import json
    import struct

    m = trimesh.creation.icosphere(subdivisions=2)
    m.visual.vertex_colors = np.tile([200, 100, 50, 255], (len(m.vertices), 1)).astype(np.uint8)
    data = M.export(m, "glb", 100)
    n = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20 : 20 + n])
    prim = doc["meshes"][0]["primitives"][0]
    assert set(prim["attributes"]) == {"POSITION", "NORMAL", "COLOR_0"}
    assert doc["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 0.0  # по подразбиране glTF е метал: тъмно
    back = trimesh.load(trimesh.util.wrap_as_stream(data), file_type="glb", force="mesh")
    assert len(back.faces) == len(m.faces) and back.extents.max() == pytest.approx(100, rel=1e-5)


def test_printable_model_is_squared_up():
    # Кутия, завъртяна на 29°: собствените ѝ размери, а не тези на завъртяната стъпка (и никога огледално).
    b = trimesh.creation.box(extents=[100, 97, 92]).apply_translation([0, 0, 46])
    b.apply_transform(trimesh.transformations.rotation_matrix(np.radians(29), [0, 0, 1]))
    out = M.make_printable(b, 0.0, False)
    np.testing.assert_allclose(out.extents, [100, 97, 92], atol=0.01)
    assert out.volume == pytest.approx(100 * 97 * 92)


def test_free_space_is_what_the_frames_saw_through():
    # Три кадъра отгоре (z = 5, гледат надолу) са измерили под при z = 0; четвърти, отстрани при x = 5, гледащ по -x,
    # е измерил стена при x = 1, която скрива всичко зад нея.
    down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
    side = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]])
    R = np.stack([down, down, down, side])
    t = -np.einsum("nij,nj->ni", R, [[0, 0, 5], [0.1, 0, 5], [0, 0.1, 5], [5, 0, 0.5]])
    views = M.Views(R, t, np.tile([500, 500, 500, 500.0], (4, 1)), list("abcd"))
    g = np.linspace(-0.8, 0.8, 41)
    floor = np.c_[np.stack(np.meshgrid(g, g), -1).reshape(-1, 2), np.zeros(41 * 41)]
    wall = np.c_[np.ones(41 * 41), np.stack(np.meshgrid(g, g + 0.5), -1).reshape(-1, 2)]
    points = np.vstack([floor, wall])
    normals = np.vstack([np.tile([0, 0, 1.0], (len(floor), 1)), np.tile([1, 0, 0.0], (len(wall), 1))])
    probes = np.array([[0, 0, 0.5], [0, 0, 0.01], [0, 0, -0.3], [3, 3, 3]])
    limit = M.free_space_limit(views, points, normals, probes, margin=0.05)
    # Над пода: видяно през него отгоре (страничният кадър, който вижда само стената пред себе си, не гласува). На
    # пода: запазено. Под него или невидяно от нито един кадър: неизвестно, запазено.
    assert limit[0] < 0.5 and (limit[1:] == 1).all()


def test_printable_keeps_an_object_with_a_gap_in_its_side():
    # Процеп на четвърт височина отстрани не е отвореното дъно: предметът запазва височината си (преди се отрязваше
    # над процепа).
    m = trimesh.creation.box(extents=[1, 1, 1]).apply_translation([0, 0, 0.5]).subdivide().subdivide().subdivide()
    gap = (np.abs(m.triangles_center[:, 0] - 0.5) < 1e-6) & (np.abs(m.triangles_center[:, 2] - 0.19) < 0.04)
    m.update_faces(~gap)
    out = M.make_printable(m, 0.0, False)
    assert out.is_watertight and out.extents[2] == pytest.approx(100, rel=0.01)


def test_textured_glb_round_trip():
    """GLB с текстура от снимки се зарежда обратно с текстурата, координатите и размера на изображението, отбелязан за
    телефона."""
    import io as _io
    import json as _json

    from PIL import Image as _Image

    box = trimesh.creation.box((2.0, 1.0, 1.0))
    jpeg = _io.BytesIO()
    _Image.new("RGB", (64, 32), (200, 30, 30)).save(jpeg, "JPEG")
    part = {"vertices": box.vertices, "normals": box.vertex_normals, "uv": np.random.default_rng(0).random((8, 2)),
            "faces": box.faces, "jpeg": jpeg.getvalue(), "size": (64, 32)}  # fmt: skip
    data = M.export(box, "glb", 100.0, [part])
    doc = _json.loads(data[20 : 20 + int.from_bytes(data[12:16], "little")])
    assert doc["images"][0]["extras"] == {"width": 64, "height": 32}
    scene = trimesh.load(trimesh.util.wrap_as_stream(data), file_type="glb")
    m = list(scene.geometry.values())[0]
    assert len(m.faces) == 12
    assert m.visual.material.baseColorTexture.size == (64, 32)
    assert scene.extents.max() == pytest.approx(100.0, rel=1e-4)
