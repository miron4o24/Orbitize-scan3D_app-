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
