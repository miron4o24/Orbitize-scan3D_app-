"""Съединяването на частите на камерите през модел с научени особености (app/pipeline.py same_place) и опората на
обвивката от очертанията (app/fuse.py rim_support)."""

import numpy as np
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
