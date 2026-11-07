"""Дневникът на положението на телефона по време на заснемане: посоката нагоре за AI модела (gravity_up).
motion.json (качен от приложението): {"samples": [[t, alpha, beta, gamma], ...]}, t в секунди от заявката за запис,
ъгли в радиани, DeviceOrientation по W3C: R = Rz(alpha) Rx(beta) Ry(gamma) завърта осите на телефона (x надясно, y
нагоре, z извън екрана) към тези на света (x на изток, y на север, z нагоре)."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import mesh as M

MOTION = "motion.json"  # в папката на задачата
FRAME_TIMES = "frames.json"  # в папката на задачата: секунди от началото на видеото за всеки избран кадър
MIN_FRAMES = 12
CAM_FROM_DEVICE = np.diag([1.0, -1.0, -1.0])  # камера: x надясно, y надолу, z напред, навън от гърба на телефона
DEFAULT_DELAY = 0.3  # секунди: първият видео кадър е толкова след заявката за запис


def device_matrices(alpha, beta, gamma) -> np.ndarray:
    """(n, 3, 3) завъртания от телефона към света от ъгли по W3C (масиви)."""
    return Rotation.from_euler("ZXY", np.stack([alpha, beta, gamma], -1)).as_matrix()  # вътрешни Z, X', Y''


def device_rotations(samples: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Положението на телефона за всяко време (секунди от заявката за запис), интерполирано между пробите."""
    order = np.argsort(samples[:, 0], kind="stable")
    s = samples[order]
    s = s[np.r_[True, np.diff(s[:, 0]) > 0]]
    rot = Rotation.from_matrix(device_matrices(s[:, 1], s[:, 2], s[:, 3]))
    return Slerp(s[:, 0], rot)(np.clip(times, s[0, 0], s[-1, 0])).as_matrix()


def load(job: Path) -> tuple[np.ndarray, dict] | None:
    """Пробите за положението на телефона и времената на кадрите, ако приложението ги е изпратило (само за видеа)."""
    try:
        samples = np.array(json.loads((job / MOTION).read_text())["samples"], float)
        times = json.loads((job / FRAME_TIMES).read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if samples.ndim != 2 or samples.shape[1] != 4 or len(samples) < 20 or not np.isfinite(samples).all():
        return None
    return samples, times


def gravity_up(job: Path, v: M.Views) -> tuple[np.ndarray, float] | None:
    """Накъде е нагоре в координатите на решаващия алгоритъм, от дневника на положението на телефона: собственото
    усещане за гравитация на всеки кадър, завъртяно с решеното му завъртане. (Алгоритъмът не знае къде е нагоре;
    посоката нагоре на изображението се осреднява до него само при равномерен пълен кръг, а кутия върху възглавница
    без намерена маса за изравняване излизаше наклонена на 45°.) Връща (нагоре, медианен ъгъл между посоките на
    кадрите в градуси) или None без дневник за тези кадри."""
    data = load(job)
    if data is None:
        return None
    samples, times = data
    known = [i for i, n in enumerate(v.names) if n in times]
    if len(known) < MIN_FRAMES:
        return None
    dev = device_rotations(samples, np.array([times[v.names[i]] for i in known]) + DEFAULT_DELAY)
    up_cam = (dev @ CAM_FROM_DEVICE)[:, 2, :]  # световното +z в координатите на камерата
    ups = np.einsum("nji,nj->ni", v.R[known], up_cam)  # ...в координатите на решаващия алгоритъм
    up = ups.sum(0)
    up /= np.linalg.norm(up)
    return up, float(np.degrees(np.median(np.arccos(np.clip(ups @ up, -1, 1)))))
