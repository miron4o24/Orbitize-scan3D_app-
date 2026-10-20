import numpy as np
import pytest
import trimesh

from app import mesh as M

RNG = np.random.default_rng(1)


def scene(tilt_deg=15.0, roll_deg=0.0):
    """Unit sphere resting on a table at z=0, a few floaters, 24 cameras orbiting; whole world tilted."""
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

    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    centers = np.c_[6 * np.cos(ang), 6 * np.sin(ang), np.full(24, 3.0)]
    dirs = [0, 0, 1.0] - centers
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    side = np.cross(dirs, [0, 0, 1.0])
    ups = np.cross(side, dirs)
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    # Random camera roll around the viewing axis (like the Middlebury datasets): ups no longer agree.
    roll = np.radians(RNG.uniform(-roll_deg, roll_deg, 24))[:, None]
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    ups = np.cos(roll) * ups + np.sin(roll) * side

    T = M.rotation_to(np.array([np.sin(np.radians(tilt_deg)), 0, np.cos(np.radians(tilt_deg))]))
    return dict(
        points=pts @ T.T,
        normals=nrm @ T.T,
        colors=np.tile([0.6, 0.5, 0.4, 1.0], (len(pts), 1)),
        cam_centers=centers @ T.T,
        cam_dirs=dirs @ T.T,
        cam_ups=ups @ T.T,
    )


def test_rotation_to_maps_vector_onto_up():
    for a in ([1, 0, 0], [0, 0, -1], [0.3, -0.2, 0.9]):
        a = np.array(a, float) / np.linalg.norm(a)
        np.testing.assert_allclose(M.rotation_to(a) @ a, M.UP, atol=1e-9)


def test_look_at_center():
    s = scene(tilt_deg=0)
    np.testing.assert_allclose(M.look_at_center(s["cam_centers"], s["cam_dirs"]), [0, 0, 1], atol=1e-9)


@pytest.fixture(scope="module")
def cloud():
    return M.clean_cloud(**scene())


def test_clean_cloud_levels_scene_and_removes_table(cloud):
    assert cloud.ground_z is not None
    # Table gone; visible sphere (~3250 pts) mostly kept.
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


def test_too_few_points_is_a_clear_error():
    s = scene()
    s["points"] = s["points"][:100]
    s["normals"] = s["normals"][:100]
    s["colors"] = s["colors"][:100]
    with pytest.raises(M.MeshError, match="Too few"):
        M.clean_cloud(**s)


@pytest.mark.parametrize("fmt", ["stl", "obj", "glb", "3mf"])
def test_export_formats(fmt):
    m = trimesh.creation.box(extents=[10, 20, 40]).apply_translation([0, 0, 20])
    data = M.export(m, fmt, longest_mm=100)
    assert len(data) > 100
    back = trimesh.load(trimesh.util.wrap_as_stream(data), file_type=fmt, force="mesh")
    assert max(back.extents) == pytest.approx(100, rel=1e-3)
    if fmt == "glb":
        assert back.extents[1] == pytest.approx(100, rel=1e-3)  # height on glTF's Y-up axis
