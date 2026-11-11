"""Снимки за проверка на инженерния режим: детайл с известни размери, нарисуван отгоре (ортогонално, със сенки по
нормалите) върху шарена "маса" и завъртян, както би изглеждала снимка с телефона успоредно на масата."""

import math

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

VIEW_DIR = {"top": (2, False), "side": (1, True), "end": (0, True)}  # ос на гледане, обърнато ли


def table(w: int, h: int, seed: int = 0, color=(150, 112, 78)) -> Image.Image:
    rng = np.random.default_rng(seed)
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    grain = np.sin(y / 9.0 + np.sin(x / 140.0) * 3 + rng.normal(0, 0.05, (1, w))) * 14
    noise = rng.normal(0, 7, (h, w))
    base = np.array(color, float)[None, None, :] + (grain + noise)[..., None] * np.array([1.0, 0.85, 0.7])
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def photo(mesh: trimesh.Trimesh, view: str, px_per_mm: float = 14.0, angle: float = 7.0, seed: int = 0,
          size=(1600, 1200), part_color=(225, 228, 232), bg=(150, 112, 78)) -> Image.Image:  # fmt: skip
    """Изгледът `view` на детайла: отгоре (по -Z), отстрани (по +Y: X надясно, Z нагоре) или откъм левия край (по
    +X: -Y надясно, Z нагоре), с най-дългата страна около половината кадър."""
    v = mesh.vertices
    if view == "top":
        uv = np.c_[v[:, 0], v[:, 1]]
        light_n = mesh.face_normals @ np.array([0.3, -0.4, 0.86])
        depth = v[:, 2]
    elif view == "side":
        uv = np.c_[v[:, 0], v[:, 2]]
        light_n = mesh.face_normals @ np.array([0.3, -0.86, 0.4])
        depth = -v[:, 1]
    else:
        uv = np.c_[-v[:, 1], v[:, 2]]
        light_n = mesh.face_normals @ np.array([-0.86, -0.3, 0.4])
        depth = -v[:, 0]
    W, H = size
    img = table(W, H, seed, bg)
    c = (uv.max(0) + uv.min(0)) / 2
    a = math.radians(angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = (uv - c) @ rot.T * px_per_mm
    xy = np.c_[W / 2 + p[:, 0], H / 2 - p[:, 1]]
    # Сянка: очертанието, отместено и размито.
    shadow = Image.new("L", size, 0)
    sd = ImageDraw.Draw(shadow)
    for f in mesh.faces:
        sd.polygon([tuple(xy[i] + (10, 12)) for i in f], fill=120)
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    img = Image.composite(Image.new("RGB", size, (40, 30, 22)), img, shadow)
    draw = ImageDraw.Draw(img)
    order = np.argsort(depth[mesh.faces].mean(1))  # далечните първо
    for k in order:
        f = mesh.faces[k]
        s = 0.55 + 0.45 * max(0.0, float(light_n[k]))
        col = tuple(int(min(255, ch * s)) for ch in part_color)
        draw.polygon([tuple(xy[i]) for i in f], fill=col)
    return img.filter(ImageFilter.GaussianBlur(0.7))


def shot(mesh: trimesh.Trimesh, name: str, yaw: float = 7.0, spin: float = 0.0, seed: int = 0, size=(1500, 2000),
         f35: float = 26.0, fill: float = 0.35, offset=(0.0, 0.0), part_color=(225, 228, 232),
         bg=(150, 112, 78)) -> Image.Image:  # fmt: skip
    """Снимката `name` (top, front, back, right, left, bottom), както я иска приложението, с камера на телефон:
    предметът лежи с дължината под ъгъл `yaw` на снимката отгоре, обърнат е, както казва приложението
    (photos.MOVE, около хоризонталната ос, най-близка до неговата), и леко завъртян на масата (`spin`). Камерата е
    отгоре с перспектива (фокусно разстояние f35 като за 35 мм филм), толкова високо, че предметът заема около
    `fill` от ширината на кадъра (рамката в приложението е 58% от екрана, а екранът показва около 62% от ширината
    на снимката), изместен с `offset` (дял от кадъра) от средата. EXIF с фокусното разстояние: exif()."""
    from app.cad.photos import MOVE, rot

    snap = round(yaw / 90) * 90
    R = rot("z", spin) @ rot("z", yaw - snap) @ MOVE[name] @ rot("z", snap)
    v = (mesh.vertices - mesh.bounds.mean(0)) @ R.T
    v[:, 2] -= v[:, 2].min()  # лежи на масата
    W, H = size
    f = f35 * math.hypot(W, H) / 43.27
    top = float(v[:, 2].max())
    span = float(max(np.ptp(v[:, 0]), np.ptp(v[:, 1])))
    zc = top + span * f / (fill * W)  # височината на телефона: най-близкото на предмета заема `fill` от кадъра
    v[:, 0] += offset[0] * W * (zc - top) / f
    v[:, 1] += offset[1] * W * (zc - top) / f
    uv = v[:, :2] * (f / (zc - v[:, 2]))[:, None]
    normals = mesh.face_normals @ R.T
    light_n = normals @ np.array([0.3, -0.4, 0.86])
    depth = v[:, 2]
    img = table(W, H, seed, bg)
    xy = np.c_[W / 2 + uv[:, 0], H / 2 - uv[:, 1]]
    shadow = Image.new("L", size, 0)
    sd = ImageDraw.Draw(shadow)
    for fc in mesh.faces:
        sd.polygon([tuple(xy[i] + (10, 12)) for i in fc], fill=120)
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    img = Image.composite(Image.new("RGB", size, (40, 30, 22)), img, shadow)
    draw = ImageDraw.Draw(img)
    for k in np.argsort(depth[mesh.faces].mean(1)):  # далечните първо
        fc = mesh.faces[k]
        ray = np.array([0.0, 0.0, zc]) - v[fc].mean(0)  # към камерата
        if normals[k] @ ray <= 0:
            continue  # гледа встрани от камерата: не се вижда
        s = 0.55 + 0.45 * max(0.0, float(light_n[k]))
        col = tuple(int(min(255, ch * s)) for ch in part_color)
        draw.polygon([tuple(xy[i]) for i in fc], fill=col)
    return img.filter(ImageFilter.GaussianBlur(0.7))


def exif(f35: float = 26.0) -> bytes:
    """EXIF с фокусното разстояние като за 35 мм филм, както го записва телефонът."""
    e = Image.Exif()
    e.get_ifd(0x8769)[41989] = int(round(f35))
    return e.tobytes()
