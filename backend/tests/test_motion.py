"""app/motion.py върху изкуствено сканиране: овален цилиндър, заснет в два кръга, първо ниско и после високо, с точни
очертания и положението на телефона, както го записва приложението (ъгли по W3C от сензора на Android, видеото започва
0,8 с след заявяването на записа). Овален, защото при кръгъл предмет не биха се виждали нито закъснението, нито
посоката."""

import json

import numpy as np
import pytest
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

from app import motion as MO
from app import segment as S

W, H = 540, 960  # вертикални видео кадри
FOCAL = 0.77 * H
RX, RY, HALF = 0.6, 0.4, 0.375  # цилиндърът, центриран в началото на координатите: 1,2 x 0,8 x 0,75
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


def outline(c: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Точната маска на цилиндъра (както я пази segment.py) от центъра на камерата c и завъртането от камерата към
    света C."""
    s = S.MASK_SCALE
    u, v = np.meshgrid((np.arange(W // s) + 0.5) * s, (np.arange(H // s) + 0.5) * s)
    d = np.stack([(u - W / 2) / FOCAL, (v - H / 2) / FOCAL, np.ones_like(u)], -1) @ C.T
    k = np.array([1 / RX, 1 / RY])  # сплескано, така че страната да е единичната окръжност
    cs, ds = c[:2] * k, d[..., :2] * k
    a = (ds**2).sum(-1)
    b = 2 * (cs * ds).sum(-1)
    disc = b * b - 4 * a * ((cs**2).sum() - 1)
    hit_t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
    hit = (disc >= 0) & (hit_t > 0) & (np.abs(c[2] + hit_t * d[..., 2]) <= HALF)  # страната
    for zc in (-HALF, HALF):  # основите
        tc = (zc - c[2]) / np.where(np.abs(d[..., 2]) > 1e-12, d[..., 2], 1e-12)
        hit |= (tc > 0) & (((cs + tc[..., None] * ds) ** 2).sum(-1) <= 1)
    return np.where(hit, 255, 0).astype(np.uint8)


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    """Папка на задача, каквато я оставя обработката: снимки, frames.json, motion.json и маските на снимките."""
    job = tmp_path_factory.mktemp("job")
    (job / "photos").mkdir()
    times = np.arange(80) * 0.5
    centres, rots = path(times)
    masks = {}
    for i, (c, C) in enumerate(zip(centres, rots)):
        name = f"img_{i + 1:04d}.jpg"
        masks[name] = outline(c, C)
        Image.new("RGB", (W, H), (200, 40, 40)).save(job / "photos" / name)
    np.savez_compressed(job / "masks.npz", **masks)
    (job / MO.FRAME_TIMES).write_text(json.dumps({f"img_{i + 1:04d}.jpg": t for i, t in enumerate(times)}))
    s = np.arange(0, 2 * LOOP + 2, 0.02)  # времена на сензора: от заявяването на записа
    _, C = path(np.clip(s - DELAY, 0, None))
    compass = Rotation.from_euler("z", 1.0).as_matrix()  # северът на сензора не е +y на изкуствения свят
    angles = android_angles(compass @ C @ MO.CAM_FROM_DEVICE)
    (job / MO.MOTION).write_text(json.dumps({"samples": np.c_[s, angles].tolist()}))
    return job


def test_fit(scan):
    samples, times = MO.load(scan)
    with np.load(scan / "masks.npz") as z:
        masks = {n: z[n] for n in z.files}
    fr = MO.frames(sorted(masks), times, masks, S.MASK_SCALE, (W, H))
    dev, score, delay = MO.fit(fr, samples, S.MASK_SCALE, FOCAL)
    # Закъснението се вижда слабо (и е също толкова маловажно: формата по-долу излезе еднаква от -0,2 до 1,8 с).
    assert score > 0.85 and -0.5 <= delay <= 2.0
    np.testing.assert_allclose(dev, MO.device_rotations(samples, fr.times + delay))


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


def test_main_builds_the_cylinder(scan):
    out = scan / "shape.ply"
    info = MO.main(scan, scan / "masks.npz", FOCAL, out, True)
    assert info["frames"] == 80 and info["score"] > 0.9 and json.loads(out.with_suffix(".json").read_text()) == info
    mesh = trimesh.load(out, force="mesh")
    assert mesh.is_watertight
    # Напречно: осите на овала (завъртени от компаса), от разпределението на повърхността.
    xy = mesh.vertices[:, :2] - mesh.vertices[:, :2].mean(0)
    axes = xy @ np.linalg.svd(xy, full_matrices=False)[2].T
    long, short = np.ptp(axes, axis=0)
    assert short / long == pytest.approx(RY / RX, abs=0.06)
    # Височината: без клин под масата, с малък конус отгоре (видян в най-ниската точка от 8° отгоре).
    assert mesh.extents[2] / long == pytest.approx(HALF / RX, abs=0.07)


def test_main_needs_the_phone_to_go_around(scan, tmp_path):
    """Очертанията съвпадат при всякакви положения, когато всички кадри гледат от една страна: тогава няма форма."""
    data = json.loads((scan / MO.MOTION).read_text())
    data["samples"] = [[t, 0.3, 1.2, 0.05] for t, *_ in data["samples"]]  # заседнал сензор
    job = tmp_path
    (job / MO.MOTION).write_text(json.dumps(data))
    (job / MO.FRAME_TIMES).write_text((scan / MO.FRAME_TIMES).read_text())
    (job / "photos").mkdir()
    for p in (scan / "photos").iterdir():
        (job / "photos" / p.name).write_bytes(p.read_bytes())
    info = MO.main(job, scan / "masks.npz", FOCAL, job / "m.ply", True)
    assert info["frames"] == 80 and info["around"] < 10 and info["score"] == 0 and not (job / "m.ply").exists()


def test_main_without_motion_data(scan, tmp_path):
    (tmp_path / "photos").mkdir()
    np.savez_compressed(tmp_path / "masks.npz", img_0001=np.zeros((4, 4), np.uint8))
    info = MO.main(tmp_path, tmp_path / "masks.npz", 0.0, tmp_path / "m.ply", True)
    assert info["score"] == 0 and not (tmp_path / "m.ply").exists()
