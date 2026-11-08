"""Части от сливането на дълбочината (app/fuse.py) и цялото сливане върху изкуствено сканиране: кълбо (по желание с
проход през него) върху маса, заснето в кръг с точни карти на дълбочината."""

import numpy as np
import pytest
import trimesh

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


W = H = 160
FOCAL = 200.0
RADIUS, CENTRE = 1.0, np.array([0.0, 0.0, 1.0])  # кълбото стои върху масата z = 0
HOLE = 0.3  # радиусът на прохода по оста x (scene(hole=True))


def look_at(c: np.ndarray, target: np.ndarray):
    """(R свят -> камера, център): камерата гледа към target, x надясно, y надолу."""
    fwd = (target - c) / np.linalg.norm(target - c)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd]), c


def trace(R, C, K, hole: bool):
    """Точната дълбочина (по оста на камерата) и маската на предмета: кълбото (без прохода) и масата."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)  # z = 1 в камерата
    d_world = d @ R  # (R^T d)
    oc = C - CENTRE
    a = (d_world**2).sum(-1)
    b = 2 * (d_world * oc).sum(-1)
    c = (oc**2).sum() - RADIUS**2
    disc = b * b - 4 * a * c
    t_near = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    t_far = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a)
    t = np.where(disc > 0, t_near, np.inf)
    if hole:  # проход по x: точките от кълбото в цилиндъра y^2 + (z-1)^2 < HOLE^2 липсват
        def in_hole(tt):
            p = C + tt[..., None] * d_world
            return (p[..., 1] ** 2 + (p[..., 2] - 1) ** 2) < HOLE**2

        # Лъч, влизащ в прохода: удря стената му (цилиндъра) или излиза от другата страна.
        dy, dz, oy, oz = d_world[..., 1], d_world[..., 2], C[1], C[2] - 1
        ca = dy * dy + dz * dz
        cb = 2 * (dy * oy + dz * oz)
        cc = oy * oy + oz * oz - HOLE**2
        cd = cb * cb - 4 * ca * cc
        t_wall = (-cb + np.sqrt(np.maximum(cd, 0))) / (2 * np.maximum(ca, 1e-12))  # излизане от цилиндъра
        wall = C + t_wall[..., None] * d_world
        wall_ok = (cd > 0) & (((wall - CENTRE) ** 2).sum(-1) < RADIUS**2) & (t_wall > 0)
        enters = (disc > 0) & in_hole(t_near)
        t = np.where(enters, np.where(wall_ok, t_wall, np.where(in_hole(t_far), np.inf, t_far)), t)
    obj = np.isfinite(t)
    t_table = np.where(d_world[..., 2] < 0, -C[2] / np.minimum(d_world[..., 2], -1e-12), np.inf)
    table = ~obj & np.isfinite(t_table)
    depth = np.where(obj, t, np.where(table, t_table, 0)).astype(np.float32)  # t по d с z = 1: дълбочина
    return depth, obj


def scene(hole=False, n=24):
    """Входът на fuse_views: обвивката е кълбото, без прохода (очертанията не го показват)."""
    K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]])
    views = []
    for i in range(n):
        az = 2 * np.pi * i / n
        el = np.radians(10 if i % 2 else 35)
        c = CENTRE + 4.0 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        R, C = look_at(c, CENTRE)
        depth, obj = trace(R, C, K, hole)
        disp = np.where(depth > 0, 1 / np.maximum(depth, 1e-6), 0).astype(np.float32)
        full, _ = trace(R, C, K, False)
        sphere = full > 0
        _, outline = trace(R, C, K, False)
        views.append(F.View(f"img_{i:04d}.jpg", K, R, C, (0, 0, W, H), depth, np.ones_like(depth), disp, obj,
                            np.ones_like(obj), outline & sphere))  # fmt: skip
    hull = trimesh.creation.icosphere(subdivisions=4, radius=RADIUS).apply_translation(CENTRE)
    z = {"names": np.array([v.name for v in views]), "to_up": np.eye(3), "ground_z": 0.0, "hull_v": hull.vertices,
         "hull_f": hull.faces, "voxel": 0.02, "flap": F.FLAP, "passes": F.PASSES}  # fmt: skip
    return z, hull, views


def test_fuses_the_sphere(tmp_path):
    z, hull, views = scene()
    out = tmp_path / "fused.ply"
    F.fuse_views(z, hull, views, out)
    mesh = trimesh.load(out, force="mesh")
    r = np.linalg.norm(mesh.vertices - CENTRE, axis=1)
    upper = mesh.vertices[:, 2] > 0.3  # над срязаното при масата дъно
    assert np.median(np.abs(r[upper] - RADIUS)) < 0.02


def with_normals(views):
    """Кадрите с точните посоки на повърхността на кълбото (в координатите на камерата), като от app/normals.py."""
    out = []
    for v in views:
        x0, y0, x1, y1 = v.box
        xs = (np.arange(x0, x1) + 0.5 - v.K[0, 2]) / v.K[0, 0]
        ys = (np.arange(y0, y1) + 0.5 - v.K[1, 2]) / v.K[1, 1]
        cam = v.dep[..., None] * np.stack(np.broadcast_arrays(xs[None, :], ys[:, None], np.ones((1, 1))), -1)
        world = cam @ v.R + v.C
        n = (world - CENTRE) @ v.R.T
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        out.append(F.View(**{**v.__dict__, "normal": n.astype(np.float16)}))
    return out


def test_fuses_the_sphere_with_surface_directions(tmp_path):
    """Детайлният начин: AI дълбочината е груба (размазана), а посоките на повърхността са точни."""
    from scipy import ndimage

    z, hull, views = scene()
    rough = [F.View(**{**v.__dict__, "disp": ndimage.gaussian_filter(v.disp, 6).astype(np.float32)}) for v in views]
    out = tmp_path / "fused.ply"
    F.fuse_views(z, hull, with_normals(rough), out)
    mesh = trimesh.load(out, force="mesh")
    r = np.linalg.norm(mesh.vertices - CENTRE, axis=1)
    upper = mesh.vertices[:, 2] > 0.3
    assert np.median(np.abs(r[upper] - RADIUS)) < 0.02


def test_keeps_the_hole_seen_through(tmp_path):
    """Проход през кълбото: маската, AI дълбочината и посоките на повърхността го запълват (виждат цялото кълбо), но
    измерената дълбочина вижда през него. Детайлният начин го оставя празен."""
    from dataclasses import replace

    from scipy import ndimage

    z, hull, views = scene(hole=True)
    z["voxel"] = 0.015
    filled = []
    for v in views:
        full, _ = trace(v.R, v.C, v.K, False)
        ai = ndimage.gaussian_filter(np.where(full > 0, 1 / np.maximum(full, 1e-6), 0), 2.0).astype(np.float32)
        nv = replace(v, disp=ai, inner=v.core)
        filled.append(replace(with_normals([replace(nv, dep=full)])[0], dep=v.dep))
    out = tmp_path / "fused.ply"
    F.fuse_views(z, hull, filled, out)
    mesh = trimesh.load(out, force="mesh")
    axis = np.c_[np.linspace(-0.85, 0.85, 35), np.zeros(35), np.ones(35)]
    assert mesh.contains(axis).mean() < 0.2


def test_rescue_depth_replaces_unfit_frames(tmp_path):
    """Кадри с шум вместо измерена дълбочина (лъскава пластмаса) се пропускат; с дълбочина от DA3 (app/rescue.py)
    влизат в сливането."""
    import json
    from dataclasses import replace

    z, hull, views = scene()
    rng = np.random.default_rng(0)
    noisy = [replace(v, dep=np.where(v.dep > 0, rng.uniform(1, 9, v.dep.shape), 0).astype(np.float32)) if i % 2 else v
             for i, v in enumerate(views)]  # fmt: skip
    assert all(F.completed(v) is None for v in noisy[1::2])
    for name, frames in (("left_out", noisy), ("rescued", [replace(v, rescue=w.dep) for v, w in zip(noisy, views)])):
        out = tmp_path / f"{name}.ply"
        F.fuse_views(z, hull, frames, out)
        used = json.loads(out.with_suffix(".json").read_text())["frames"]
        assert used == (len(views) // 2 if name == "left_out" else len(views))


@pytest.mark.parametrize("dump", [False, True])
def test_save_and_replay(tmp_path, dump):
    z, hull, views = scene()
    path = tmp_path / "debug.npz"
    F.save(path, z, views)
    z2, views2 = F.load(path)
    assert [v.name for v in views2] == [v.name for v in views]
    assert np.array_equal(views2[3].dep, views[3].dep) and np.array_equal(views2[3].core, views[3].core)
    if dump:
        F.replay(path, tmp_path / "r.ply")
        assert trimesh.load(tmp_path / "r.ply", force="mesh").is_watertight
