import numpy as np
import pytest
import trimesh

from app import mesh as M

RNG = np.random.default_rng(1)


W, H, F = 160, 120, 100.0  # mask size and focal length of the test cameras (masks at full size: scale 1)


def scene(tilt_deg=15.0, roll_deg=0.0, n_cams=24):
    """Unit sphere resting on a table at z=0, a few floaters, cameras orbiting with exact object masks; the whole
    world tilted."""
    v = RNG.normal(size=(5000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] > -0.3]  # cameras ~18° above the centre never see the underside
    sphere, sphere_n = v + [0, 0, 1], v.copy()
    r, a = 2.8 * np.sqrt(RNG.random(3000)), RNG.random(3000) * 2 * np.pi
    table = np.c_[r * np.cos(a), r * np.sin(a), np.zeros(3000)]
    table_n = np.tile([0, 0, 1.0], (3000, 1))
    floaters = RNG.uniform(-2.5, 2.5, size=(30, 3)) + [0, 0, 1.5]
    pts = np.vstack([sphere, table, floaters])
    nrm = np.vstack([sphere_n, table_n, RNG.normal(size=(30, 3))])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm[RNG.random(len(nrm)) < 0.5] *= -1  # unoriented normals, like a real dense cloud

    T = M.rotation_to(np.array([np.sin(np.radians(tilt_deg)), 0, np.cos(np.radians(tilt_deg))]))
    Rs, ts, masks = [], [], []
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, n_cams, endpoint=False)):
        c = np.array([6 * np.cos(ang), 6 * np.sin(ang), 3.0])
        z = [0, 0, 1.0] - c
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)  # image down
        # Random camera roll around the viewing axis (like the Middlebury datasets): image ups no longer agree.
        roll = np.radians(RNG.uniform(-roll_deg, roll_deg))
        x, y = np.cos(roll) * x + np.sin(roll) * y, -np.sin(roll) * x + np.cos(roll) * y
        R = np.stack([x, y, z]) @ T.T  # camera of the tilted world
        Rs.append(R)
        ts.append(-R @ (T @ c))
        # Exact mask: pixels whose ray passes within 1 of the sphere centre.
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
    pts = np.array([[0, 0, 1.0], [2.5, 0, 0], [0, 0, 50]])  # sphere centre, table beside it, far above
    n_in, n_fg = M.silhouette_votes(pts, s["views"], s["masks"], 1)
    assert list(n_in[:2]) == [24, 24] and n_in[2] == 0  # far above: outside every picture
    assert n_fg[0] == 24 and n_fg[1] < 8


def test_consistent_frames_drops_wrongly_placed_ones():
    s = scene()
    v, masks = s["views"], s["masks"]
    assert M.consistent_frames(v, masks, 1).all()
    assert np.nanmin(M.silhouette_consistency(v, masks, 1)) > 0.8
    # A broken camera solution: one camera turned (in place) to face away, one shifted sideways by the object's size.
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
    # Table and floaters gone; visible sphere (~3250 pts) mostly kept.
    assert 2900 < len(cloud.points) < 3400
    assert cloud.points[:, 2].min() > cloud.ground_z
    # Normals oriented outward from the sphere centre (which sits 1 above the ground).
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
    # A wide flat ledge halfway up a tall column: many points, but plenty of the object is below it.
    r, a = RNG.random(4000), RNG.random(4000) * 2 * np.pi
    ledge = np.c_[r * np.cos(a), r * np.sin(a), np.full(4000, 1.0)]
    column = np.c_[RNG.uniform(-0.3, 0.3, (6000, 2)), RNG.uniform(0, 2, 6000)]
    assert M.find_ground(np.vstack([ledge, column]), thresh=0.01) is None


def test_visual_hull_is_the_sphere_down_to_the_table():
    s = scene()
    c = M.clean_cloud(**s)
    hull = M.visual_hull(s["views"], s["masks"], 1, c, cells=64)
    assert hull.is_watertight
    lo, hi = hull.bounds
    assert lo[2] == pytest.approx(c.ground_z, abs=0.1)  # stands on the table
    assert hi[2] - c.ground_z == pytest.approx(2, abs=0.15)
    assert 1.9 < hull.extents[0] < 2.4 and 1.9 < hull.extents[1] < 2.4


def test_hull_fills_parts_without_points():
    # Dense matching found nothing on half the sphere (e.g. a plain black side): the shape must stay round.
    s = scene()
    c = M.clean_cloud(**s)
    half = c.points[:, 0] < np.median(c.points[:, 0])
    c.points, c.normals, c.colors = c.points[half], c.normals[half], c.colors[half]
    mesh = M.poisson(c, M.visual_hull(s["views"], s["masks"], 1, c, cells=64), depth=7)
    out = M.make_printable(mesh, M.floor_z(c), flat_base=True)
    x, y, z = out.extents
    assert x == pytest.approx(y, rel=0.1)  # round, not a half sphere or a blob


def test_object_not_found_is_a_clear_error():
    s = scene()
    s["masks"] = [m * 0 for m in s["masks"]]
    with pytest.raises(M.MeshError, match="отдели от фона"):
        M.clean_cloud(**s)


def test_poisson_and_make_printable(cloud):
    mesh = M.poisson(cloud, depth=7)
    # A floating fragment that must be dropped.
    junk = trimesh.creation.icosphere(radius=0.1).apply_translation([3, 3, cloud.ground_z + 2])
    mesh = trimesh.util.concatenate([mesh, junk])

    out = M.make_printable(mesh, M.floor_z(cloud), flat_base=True, longest_mm=80)
    rep = M.report(out)
    assert rep["watertight"]
    assert max(rep["dimensions_mm"]) == pytest.approx(80)
    assert len(out.split(only_watertight=False)) == 1
    # Flat base: many vertices exactly at z = 0.
    assert out.bounds[0, 2] == pytest.approx(0)
    assert (np.abs(out.vertices[:, 2]) < 1e-6).sum() > 20


def test_hole_on_the_side_is_closed():
    m = trimesh.creation.icosphere(subdivisions=3).apply_translation([0, 0, 1])
    side = np.abs(m.triangles_center[:, 0] - 1) < 0.25  # punch a hole on the +X side, mid-height
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
        assert back.extents[1] == pytest.approx(100, rel=1e-3)  # height on glTF's Y-up axis


def test_paint_takes_each_side_from_the_frame_that_sees_it(tmp_path):
    from PIL import Image

    # A box between two cameras on the x axis, looking at it: one frame all red, the other all blue.
    box = trimesh.creation.box(extents=[1, 1, 1]).subdivide().subdivide()
    R_pos = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]])  # at +x, looking along -x
    R_neg = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.0]])  # at -x, looking along +x
    t = np.array([[0, 0, 4.0], [0, 0, 4.0]])
    views = M.Views(np.stack([R_pos, R_neg]), t, np.tile([200, 200, 100, 100.0], (2, 1)), ["a.jpg", "b.jpg"])
    for name, rgb in (("a.jpg", (220, 30, 30)), ("b.jpg", (30, 30, 220))):
        Image.new("RGB", (200, 200), rgb).save(tmp_path / name)
    M.paint(box, views, [tmp_path / "a.jpg", tmp_path / "b.jpg"], np.eye(3))
    c = box.visual.vertex_colors[:, :3].astype(int)
    front, back = box.vertices[:, 0] > 0.49, box.vertices[:, 0] < -0.49
    assert (c[front, 0] > 200).all() and (c[back, 2] > 200).all()  # each side from its own frame, not the other


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
    assert doc["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 0.0  # glTF's default is metal: dark
    back = trimesh.load(trimesh.util.wrap_as_stream(data), file_type="glb", force="mesh")
    assert len(back.faces) == len(m.faces) and back.extents.max() == pytest.approx(100, rel=1e-5)


def test_printable_model_is_squared_up():
    # A box standing turned by 29°: its own sizes, not those of the turned footprint (and never mirrored).
    b = trimesh.creation.box(extents=[100, 97, 92]).apply_translation([0, 0, 46])
    b.apply_transform(trimesh.transformations.rotation_matrix(np.radians(29), [0, 0, 1]))
    out = M.make_printable(b, 0.0, False)
    np.testing.assert_allclose(out.extents, [100, 97, 92], atol=0.01)
    assert out.volume == pytest.approx(100 * 97 * 92)


def test_free_space_is_what_the_frames_saw_through():
    # Three frames from above (z = 5, looking down) measured a floor at z = 0; a fourth, from the side at x = 5
    # looking along -x, measured a wall at x = 1 that hides everything behind it.
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
    # Above the floor: seen through from above (the side frame, which only sees the wall in front, does not vote).
    # On the floor: kept. Below it, or seen by no frame: unknown, kept.
    assert limit[0] < 0.5 and (limit[1:] == 1).all()


def test_printable_keeps_an_object_with_a_gap_in_its_side():
    # A gap a quarter up the side is not the open bottom: the object keeps its height (it was cut off above the gap).
    m = trimesh.creation.box(extents=[1, 1, 1]).apply_translation([0, 0, 0.5]).subdivide().subdivide().subdivide()
    gap = (np.abs(m.triangles_center[:, 0] - 0.5) < 1e-6) & (np.abs(m.triangles_center[:, 2] - 0.19) < 0.04)
    m.update_faces(~gap)
    out = M.make_printable(m, 0.0, False)
    assert out.is_watertight and out.extents[2] == pytest.approx(100, rel=0.01)


def test_textured_glb_round_trip():
    """The photo-textured GLB loads back with its texture, coordinates and the image's size noted for the phone."""
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
