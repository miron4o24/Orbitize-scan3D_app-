"""Съединяването на частите на камерите през модел с научени особености (app/pipeline.py same_place) и опората на
обвивката от очертанията (app/fuse.py rim_support)."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from app import fuse as F
from app import mesh as M
from app import pipeline as P


def ring(n: int, turn: np.ndarray, scale: float, shift: np.ndarray, start: int = 0) -> M.Views:
    """n камери в кръг около началото, гледащи към него, в свят X' = scale * turn @ X + shift."""
    R, t, names = [], [], []
    for k in range(n):
        a = 2 * np.pi * k / 24
        C = np.array([3 * np.cos(a), 3 * np.sin(a), 1.0 + 0.05 * k])
        z = -C / np.linalg.norm(C)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        Rc = np.stack([x, np.cross(z, x), z]) @ turn.T
        R.append(Rc)
        t.append(-Rc @ (scale * turn @ C + shift))
        names.append(f"img_{start + k:04d}.jpg")
    return M.Views(np.array(R), np.array(t), np.tile([500.0, 500.0, 320.0, 240.0], (n, 1)), names)


def test_same_place_finds_the_similarity_despite_misplaced_frames():
    turn = Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_matrix()
    a = ring(12, np.eye(3), 1.0, np.zeros(3))
    b = ring(12, turn, 2.5, np.array([1.0, -2.0, 0.5]))
    # Седем от дванайсетте кадъра на b са на грешно място (повтаряща се шарка): на куп, с други посоки.
    wrong = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    for k in range(5, 12):
        b.R[k] = b.R[k] @ wrong
        b.t[k] = b.t[5] + 0.01 * k
    s, R, t, names = P.same_place(a, b)
    assert names == a.names[:5]
    assert abs(s - 2.5) < 1e-6 and np.allclose(R, turn, atol=1e-6) and np.allclose(t, [1.0, -2.0, 0.5], atol=1e-6)


def test_same_place_needs_enough_frames_in_the_same_place():
    a = ring(12, np.eye(3), 1.0, np.zeros(3))
    b = ring(12, np.eye(3), 1.0, np.zeros(3))
    assert P.same_place(a, ring(3, np.eye(3), 1.0, np.zeros(3))) is None  # три общи кадъра
    assert P.same_place(a, ring(6, np.eye(3), 1.0, np.zeros(3), start=20)) is None  # нито един общ
    rng = np.random.default_rng(0)
    for k in range(3, 12):  # само три кадъра остават на място
        b.R[k] = Rotation.random(random_state=rng).as_matrix()
        b.t[k] = rng.normal(size=3) * 3
    assert P.same_place(a, b) is None


def test_rim_support_marks_where_the_outline_touches_a_ball():
    ball = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    K = np.array([[800.0, 0, 400], [0, 800.0, 400], [0, 0, 1]])
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # камера на -y, гледа по +y
    C = np.array([0.0, -6.0, 0.0])
    sup = F.rim_support(ball, [(R, C, K, (0, 0, 800, 800))])
    y = ball.vertices[:, 1]
    # Очертанието на кълбо от разстояние 6 е кръгът y = -1/6; далеч от него (към и от камерата) няма опора.
    assert sup[np.abs(y + 1 / 6) < 0.05].mean() > 0.9
    assert not sup[np.abs(y + 1 / 6) > 0.35].any()


def test_rim_support_ignores_a_flat_top_seen_along_its_edge():
    can = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
    for _ in range(4):
        can = can.subdivide()
    K = np.array([[800.0, 0, 400], [0, 800.0, 400], [0, 0, 1]])
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    C = np.array([0.0, -6.0, 0.5])  # на височината на капака: той се вижда като черта
    sup = F.rim_support(can, [(R, C, K, (0, 0, 800, 800))])
    x, y, z = can.vertices.T
    top = (z > 0.499) & (np.hypot(x, y) < 0.8)
    side = (np.abs(np.abs(x) - 1) < 0.02) & (np.abs(z) < 0.4) & (np.abs(y + 1 / 6) < 0.1)
    assert top.sum() > 50 and side.sum() > 10
    assert sup[top].mean() < 0.05  # капакът лежи по целия лъч: очертанието не казва къде е
    assert sup[side].mean() > 0.9  # заоблената страна се докосва


def test_path_runs_break_where_the_camera_jumps():
    views = ring(12, np.eye(3), 1.0, np.zeros(3))  # 15 градуса между съседните кадри
    assert P.path_runs(views) == [list(range(12))]
    # Кадри 6-8 са на грешно място (гледат обратно): пътят се къса преди и след тях.
    flip = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    for k in (6, 7, 8):
        views.R[k] = views.R[k] @ flip
    assert P.path_runs(views) == [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11]]
    # Липсващи кадри: след повече от PATH_GAP не се знае дали пътят е непрекъснат.
    far = M.Views(views.R[:3], views.t[:3], views.K[:3], ["img_0001.jpg", "img_0002.jpg", "img_0020.jpg"])
    assert P.path_runs(far) == [[0, 1], [2]]


def test_frame_numbers_come_from_the_names():
    assert P.frame_numbers(["img_0012.jpg", "img_0003.jpg"]) == [12, 3]
    assert P.frame_numbers(["b.jpg", "a.jpg"]) == [1, 0]


def test_levelled_stands_a_leaning_can_upright_without_moving_it_in_the_world(tmp_path):
    lean = Rotation.from_euler("x", 6, degrees=True).as_matrix()
    can = trimesh.creation.cylinder(radius=1.0, height=1.6, sections=48)
    mesh = trimesh.Trimesh(can.vertices @ lean.T + [0.3, 0.2, 1.0], can.faces, process=False)
    mesh.metadata["shapes"] = [{"kind": "round", "up": list(lean @ [0, 0, 1.0]), "share": 0.5},
                               {"kind": "flat", "up": [1.0, 0, 0], "share": 0.9}]  # встрани: не е "нагоре"
    to_up = Rotation.from_euler("zyx", [20, 30, 40], degrees=True).as_matrix()
    cloud = M.Cloud(mesh.vertices[::7].copy(), mesh.vertex_normals[::7].copy(), np.ones((len(mesh.vertices[::7]), 4)),
                    ground_z=0.2, to_up=to_up)
    out, turned = P.levelled(tmp_path, mesh, cloud)
    assert np.ptp(mesh.vertices[:, 2]) > 1.79  # наклонена: по-висока от височината си
    assert np.ptp(out.vertices[:, 2]) == pytest.approx(1.6, abs=1e-6)  # изправена: оста е точно нагоре
    assert np.allclose(out.vertices @ turned.to_up, mesh.vertices @ to_up, atol=1e-9)  # същото място в света
    assert np.allclose(turned.points @ turned.to_up, cloud.points @ to_up, atol=1e-9)
    # Без форма почти нагоре нищо не се върти.
    mesh.metadata["shapes"] = mesh.metadata["shapes"][1:]
    same, kept = P.levelled(tmp_path, mesh, cloud)
    assert same is mesh and kept is cloud


def outline_scene():
    """24 камери в кръг около дълъг Г-образен предмет (не е симетричен: очертанията му издават грешно сложена
    камера), с маските му при размера на маските в приложението."""
    from app import segment as SG

    views = ring(24, np.eye(3), 1.0, np.zeros(3))
    rng = np.random.default_rng(0)
    box = rng.uniform([-0.7, -0.2, -0.35], [0.7, 0.2, 0.35], (60000, 3))
    X = box[~((box[:, 0] > 0) & (box[:, 2] > 0))]  # без единия горен квадрант
    masks = {}
    for n, R, t, K in zip(views.names, views.R, views.t, views.K):
        x = X @ R.T + t
        u = np.round((K[0] * x[:, 0] / x[:, 2] + K[2]) / SG.MASK_SCALE).astype(int)
        v = np.round((K[1] * x[:, 1] / x[:, 2] + K[3]) / SG.MASK_SCALE).astype(int)
        m = np.zeros((480 // SG.MASK_SCALE, 640 // SG.MASK_SCALE), bool)
        m[v, u] = True
        masks[n] = m
    return views, masks


def test_outline_join_fits_a_misplaced_piece_and_leaves_out_a_wrong_one():
    views, masks = outline_scene()
    true = {n: (R, t) for n, R, t in zip(views.names, views.R, views.t)}
    join = P.OutlineJoin({n: true[n] for n in views.names[:12]}, masks, views.K[0])
    assert join.alone > 0.8
    # Частта от другата страна, пренесена с грешка в завъртането, мястото и мащаба (като през няколко общи кадъра).
    rest = {n: true[n] for n in views.names[12:20]}
    off = P.nudged(rest, np.array([4.0, -5.0, 8.0, 10.0, -8.0, 6.0, 10.0]), join.fine.center, join.fine.size)
    ok, before, after = join.add(off, wide=True)
    assert ok and after > before + 0.05 and after > join.alone - P.JOIN_WORSE
    far = max(np.linalg.norm(-R.T @ t - (-true[n][0].T @ true[n][1])) for n, (R, t) in off.items())
    near = max(np.linalg.norm(-join.poses[n][0].T @ join.poses[n][1] - (-true[n][0].T @ true[n][1])) for n in off)
    assert near < far  # и камерите са по-близо до верните си места (само по очертания - не съвсем на тях)
    # Кадри, сложени на местата на други, на четвърт кръг от тях (грешна част): там предметът се вижда тесен, а в
    # техните очертания е широк. Толкова не се поправя с нагласяне, и не остават.
    wrong = {a: true[b] for a, b in zip(views.names[20:], views.names[14:18])}
    agreed = join.agreement()
    ok, _, _ = join.add(wrong, wide=True)
    assert not ok and set(wrong).isdisjoint(join.poses) and join.agreement() == agreed


def write_model(d, views: M.Views, focal: float = 500.0):
    """Модел на COLMAP (cameras.bin, images.bin, без точки) с камерите `views` и една камера SIMPLE_RADIAL."""
    import struct

    d.mkdir(parents=True)
    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 2, 640, 480) + struct.pack("<4d", focal, 320, 240, 0))
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(views)))
        for i, (name, R, t) in enumerate(zip(views.names, views.R, views.t)):
            x, y, z, w = Rotation.from_matrix(R).as_quat()
            f.write(struct.pack("<I7dI", i + 1, w, x, y, z, *t, 1) + name.encode() + b"\0" + struct.pack("<Q", 0))


def solved(tmp_path, wrong: range = range(0)):
    """Части на SIFT от 13 и 7 кадъра (всяка в свой мащаб и посока) и научен модел на всичките 24; кадрите `wrong`
    от втората част са на грешно място в научения (повтаряща се шарка)."""
    work, learned = tmp_path / "work", tmp_path / "work" / "learned"
    all24 = ring(24, Rotation.from_euler("xyz", [5, -10, 30], degrees=True).as_matrix(), 1.7, np.array([0.3, 0, -1]))
    flip = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    for k in wrong:
        all24.R[13 + k] = all24.R[13 + k] @ flip
        all24.t[13 + k] = all24.t[13] + 0.01 * k
    write_model(work / "sparse" / "incremental" / "0", ring(13, np.eye(3), 1.0, np.zeros(3)))
    piece = ring(20, Rotation.from_euler("z", 40, degrees=True).as_matrix(), 0.6, np.array([2.0, 1, 0]))
    write_model(work / "sparse" / "incremental" / "1", M.subset(piece, np.arange(20) >= 13), focal=470.0)
    write_model(learned / "0", all24)
    return work, learned


def test_the_learned_model_is_used_alone_when_every_sift_piece_has_its_frames_in_the_same_place_there(tmp_path):
    """The Rock върху черна кутия: частите на SIFT (13 и 7 кадъра, с различно фокусно разстояние) са изцяло на
    мястото си в научения модел на всички кадри. Съединени, кадрите на по-малката част се използваха с фокусното
    разстояние на основата (двойни очи); сам, научения модел е съгласуван."""
    work, learned = solved(tmp_path)
    assert P.confirmed(tmp_path, work, learned)
    assert P.read_views(work / "sparse" / "learned" / "0").names == P.read_views(learned / "0").names


def test_the_pieces_are_joined_when_the_learned_model_misplaces_frames_of_one(tmp_path):
    """Колонката върху подноса: 6 от 15-те кадъра на високия кръг на мястото си в научения модел. Той не се използва
    сам; частите се съединяват през него (bridge_pieces)."""
    work, learned = solved(tmp_path, wrong=range(2, 7))
    assert not P.confirmed(tmp_path, work, learned)
    assert not (work / "sparse" / "learned").exists()
