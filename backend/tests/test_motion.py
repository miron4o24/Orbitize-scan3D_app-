"""app/motion.py върху изкуствено сканиране: два кръга, първо ниско и после високо, с положението на телефона, както
го записва приложението (ъгли по W3C от сензора на Android, видеото започва 0,8 с след заявяването на записа)."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from app import motion as MO

W, H = 540, 960  # вертикални видео кадри
FOCAL = 0.77 * H
DELAY = 0.8
LOOP = 20.0  # секунди за кръг


def android_angles(R: np.ndarray) -> np.ndarray:
    """Какво записва приложението за завъртането R от телефона към света: DeviceMotion.rotation от expo-sensors на
    Android е (-azimuth, -pitch, roll) от SensorManager.getOrientation."""
    r = R.reshape(-1, 9)
    azimuth, pitch, roll = np.arctan2(r[:, 1], r[:, 4]), np.arcsin(-r[:, 7]), np.arctan2(-r[:, 6], r[:, 8])
    return np.stack([-azimuth, -pitch, roll], -1)


def test_device_matrices_match_android():
    R = Rotation.random(200, random_state=1).as_matrix()
    a = android_angles(R)
    np.testing.assert_allclose(MO.device_matrices(a[:, 0], a[:, 1], a[:, 2]), R, atol=1e-9)


def path(t: np.ndarray):
    """Центрове на камерите и завъртания от камерата към света във видео времена t: нисък кръг, после висок, като
    темпото, разстоянието и насочването леко се отклоняват."""
    az = 2 * np.pi * t / LOOP + 0.4 + 0.25 * np.sin(2 * np.pi * t / 8)
    rise = np.clip((t - LOOP + 1.5) / 3, 0, 1)
    el = np.radians(8 + 37 * rise * rise * (3 - 2 * rise))
    d = 3.0 * (1 + 0.08 * np.sin(2 * np.pi * t / 7))
    c = d[:, None] * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
    aim = 0.06 * np.stack([np.sin(t), np.cos(1.3 * t), 0.5 * np.sin(0.7 * t)], -1)
    fwd = (aim - c) / np.linalg.norm(aim - c, axis=1, keepdims=True)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    return c, np.stack([right, np.cross(fwd, right), fwd], -1)  # колони: камера x (надясно), y (надолу), z (напред)


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    """Папка на задача, каквато я оставя обработката: frames.json и motion.json."""
    job = tmp_path_factory.mktemp("job")
    times = np.arange(80) * 0.5
    (job / MO.FRAME_TIMES).write_text(json.dumps({f"img_{i + 1:04d}.jpg": t for i, t in enumerate(times)}))
    s = np.arange(0, 2 * LOOP + 2, 0.02)  # времена на сензора: от заявяването на записа
    _, C = path(np.clip(s - DELAY, 0, None))
    compass = Rotation.from_euler("z", 1.0).as_matrix()  # северът на сензора не е +y на изкуствения свят
    angles = android_angles(compass @ C @ MO.CAM_FROM_DEVICE)
    (job / MO.MOTION).write_text(json.dumps({"samples": np.c_[s, angles].tolist()}))
    return job


def test_gravity_up_in_the_camera_solvers_coordinates(scan):
    from app import mesh as M

    # Светът на решаващия алгоритъм за камерите е завъртян, мащабиран и изместен произволно: X' = 2.5 Q X + s.
    Q = Rotation.from_euler("xyz", [0.7, -1.1, 2.0]).as_matrix()
    times = json.loads((scan / MO.FRAME_TIMES).read_text())
    names = sorted(times)[:25]  # само част от ниския кръг: посоката нагоре на изображението тук не се осреднява до вертикала
    c, C = path(np.array([times[n] for n in names]))
    R = np.einsum("ij,njk->nik", Q, C).transpose(0, 2, 1)  # свят' -> камера
    t = -np.einsum("nij,nj->ni", R, 2.5 * c @ Q.T + [3, -1, 4])
    v = M.Views(R, t, np.tile([FOCAL, FOCAL, W / 2, H / 2], (len(names), 1)), names)
    up, spread = MO.gravity_up(scan, v)
    assert np.degrees(np.arccos(up @ Q[:, 2])) < 1.5 and spread < 1.5
