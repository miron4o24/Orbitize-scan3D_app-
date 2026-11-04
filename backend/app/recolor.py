"""Промени на цвета за текстовите промени на Pro (app/edit.py, операция "color"): "make the hair blue", "paint the
base gold", "make it red". Коя част се има предвид намира AI върху изображения на самия модел: той се рисува от
VIEWS около него (на процесора, като app/render.py), CLIPSeg (CIDAS/clipseg-rd64-refined, Apache-2.0, като ONNX в
tools/clipseg; scripts/models.ps1) отбелязва назованата част във всяко изображение, а всеки тексел от текстурата (и
всеки връх) взима отметките от изображенията, в които се вижда, претеглени по това колко пряко. Отметките на
CLIPSeg са груби (изображението му се дели на 22 x 22 плочки), затова ръбът се определя по цвета: собствените
цветове на частта спрямо останалите, близо до мястото, посочено от CLIPSeg. Новият цвят запазва сенките и детайла
от снимката: в Lab яркостта на частта запазва колебанията си около яркостта на новия цвят, а оттенъкът става на
новия цвят (кичурите и светлината по тях остават). CLIPSeg работи като собствен процес, затова паметта му се
освобождава след това: python recolor.py <views.npz: images (V, SIDE, SIDE, 3) uint8, prompt> <out.npz: prob (V,
SIDE, SIDE)>"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

MODEL_DIR = Path(os.environ.get("CLIPSEG_DIR", Path(__file__).resolve().parent.parent / "tools" / "clipseg"))
SIDE = 352  # размерът на изображението на CLIPSeg
SUPER = 2  # изображенията се рисуват при SUPER x SIDE и се смаляват: гладки ръбове
MARGIN = 0.06  # дял от всяко изображение, оставен свободен от всяка страна на модела
VIEWS = [(az, 10) for az in range(0, 360, 45)] + [(az, 45) for az in range(22, 360, 90)] + [(0, 80)]
RENDER_POINTS = 2_500_000  # проби от повърхността, от които се рисуват изображенията
FOUND = 0.5  # отметката на CLIPSeg (0-1), от която точка се приема за част, преди проверката на цвета
SURE = 0.6  # и със сигурност е: парче от частта трябва да съдържа такива точки (decide)
KEEP_PIECE = 0.05  # парчета с по-малко сигурни точки от този дял на най-голямото отпадат
STAND_NARROW = 0.5  # (stand) фигурата над поставка е толкова по-тясна
STAND_MAX = 0.4  # и поставката е най-много този дял от височината на модела
TAP_SIDE = 512  # (докоснато) размер на изображението за MobileSAM
TAP_TILT = 35.0  # (докоснато) градуси, на които страничните изображения се завъртат спрямо прякото към мястото
TAP_WHOLE = 0.35  # (докоснато) маска над този дял от очертанието на модела е (почти) целият модел, а не част
ZOOM = 0.7  # по-близък поглед, когато частта заема по-малко от този дял от модела
DARK = 12.0  # (decide) яркост в Lab, под която цветът не казва нищо за частта (сянка), пълна от DARK + 25
PLAIN = 4.0  # (decide) разпръскване в Lab по повърхността, под което моделът е едноцветен
BINS = 16  # на ос в Lab, за това колко от всеки цвят е отбелязал CLIPSeg
MIN_SHARE = 0.002  # отбелязано е по-малко от тази повърхност: частта не е намерена
BG = 242.0  # фонът на изображенията (светлосив, като на продуктова снимка)

# Имена на цветове, sRGB. Също и цветове за коса, защото косата се прецветява най-често.
COLOURS = {
    "red": (200, 30, 35), "dark red": (130, 15, 20), "crimson": (180, 20, 50), "maroon": (110, 20, 30),
    "burgundy": (115, 20, 40), "scarlet": (230, 35, 20), "orange": (240, 120, 20), "yellow": (245, 210, 40),
    "gold": (212, 175, 55), "golden": (212, 175, 55), "lime": (140, 210, 40), "green": (40, 150, 60),
    "dark green": (20, 85, 35), "light green": (130, 210, 120), "olive": (110, 110, 40), "mint": (150, 230, 190),
    "teal": (20, 130, 130), "turquoise": (50, 200, 190), "cyan": (40, 200, 230), "aqua": (60, 210, 220),
    "light blue": (120, 180, 235), "sky blue": (120, 190, 240), "blue": (35, 80, 200), "dark blue": (20, 40, 120),
    "navy": (20, 30, 90), "navy blue": (20, 30, 90), "indigo": (75, 30, 130), "purple": (120, 50, 160),
    "violet": (140, 80, 200), "lavender": (180, 160, 225), "magenta": (210, 40, 160), "pink": (240, 130, 175),
    "hot pink": (245, 70, 150), "rose": (225, 110, 130), "brown": (110, 70, 40), "dark brown": (60, 38, 22),
    "light brown": (160, 115, 75), "chestnut": (125, 60, 35), "chocolate": (90, 50, 25), "beige": (215, 195, 160),
    "tan": (200, 160, 115), "cream": (240, 230, 200), "ivory": (245, 240, 220), "white": (245, 245, 245),
    "black": (22, 22, 24), "grey": (128, 128, 128), "gray": (128, 128, 128), "dark grey": (70, 70, 72),
    "dark gray": (70, 70, 72), "light grey": (190, 190, 192), "light gray": (190, 190, 192), "silver": (190, 192, 198),
    "copper": (184, 115, 51), "bronze": (160, 110, 50), "blonde": (225, 190, 115), "blond": (225, 190, 115),
    "platinum": (230, 225, 210), "platinum blonde": (230, 225, 210), "ginger": (190, 90, 35), "auburn": (140, 55, 35),
    "brunette": (90, 55, 35), "redhead": (175, 70, 35),
}  # fmt: skip
SHADES = {"light": 1.35, "pale": 1.45, "bright": 1.15, "dark": 0.55, "deep": 0.7}


def colour_rgb(name: str) -> tuple[int, int, int] | None:
    """Име на цвят ("blue", "dark green", "light blonde") или #rrggbb като sRGB; None, ако не е такова."""
    n = re.sub(r"\s+", " ", str(name).strip().lower().replace("colour", "").replace("color", "")).strip()
    m = re.fullmatch(r"#?([0-9a-f]{6})", n)
    if m:
        return tuple(int(m[1][i : i + 2], 16) for i in (0, 2, 4))
    if n in COLOURS:
        return COLOURS[n]
    words = n.split(" ")
    if len(words) == 2 and words[0] in SHADES and words[1] in COLOURS:
        lab = rgb_to_lab(np.array([COLOURS[words[1]]], np.float64))[0]
        lab[0] = float(np.clip(lab[0] * SHADES[words[0]] if SHADES[words[0]] < 1 else lab[0] + (100 - lab[0]) * (SHADES[words[0]] - 1), 3, 97))
        return tuple(int(v) for v in lab_to_rgb(lab[None])[0])
    return None


# --- sRGB <-> Lab (D65) ---------------------------------------------------------------------------------------------

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, np.float64)
    fy = (lab[:, 0] + 16) / 116
    f = np.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], 1)
    xyz = np.where(f**3 > 216 / 24389, f**3, (116 * f - 16) / (24389 / 27)) * _WHITE
    c = np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def recoloured(colours: np.ndarray, weight: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """Цветове (N, 3) uint8, обърнати към `rgb` с тегло (N,) 0-1: в Lab яркостта запазва колебанията си около яркостта
    на новия цвят (сенки, кичури, отблясъци); оттенъкът и наситеността стават на новия цвят, с малко от старите
    колебания."""
    out = np.asarray(colours, np.uint8).copy()
    sel = weight > 0.01
    if not sel.any():
        return out
    lab = rgb_to_lab(out[sel])
    w = weight[sel]
    mean = (lab * w[:, None]).sum(0) / w.sum()
    target = rgb_to_lab(np.array([rgb], np.float64))[0]
    # Голяма промяна на яркостта леко свива колебанията, иначе тъмна част, направена бяла, би станала чисто бяла.
    spread = float(np.sqrt(((lab[:, 0] - mean[0]) ** 2 * w).sum() / w.sum()))
    room = min(target[0], 100 - target[0])
    k = 1.0 if spread < 1e-6 else float(np.clip(room / (2.2 * spread), 0.35, 1.0))
    new = np.empty_like(lab)
    new[:, 0] = np.clip(target[0] + (lab[:, 0] - mean[0]) * k, 0, 100)
    new[:, 1:] = target[1:] + 0.25 * (lab[:, 1:] - mean[1:])
    mixed = lab_to_rgb(new).astype(np.float64) * w[:, None] + out[sel] * (1 - w[:, None])
    out[sel] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return out


# --- Точките на повърхността: текселите на текстурата или върховете
# --------------------------------------------------


def texels(part: dict, atlas: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Текселите на едно изображение на текстура (част от load в app/texture.py), които лежат върху повърхността:
    (редове, колони), 3D точки и нормали. Стени, невидени от никой кадър, имат квадрат от няколко тексела
    (texture.paint_unseen): всичките им тексели взимат центъра на стената."""
    from . import texture as TX

    V, F, N = part["vertices"], part["faces"], part["normals"]
    uv = np.asarray(part["uv"], np.float64)[F]  # (стени, 3, 2), v нагоре
    H, W = atlas.shape[:2]
    px = np.stack([uv[..., 0] * W, (1 - uv[..., 1]) * H], -1)
    e1, e2 = px[:, 1] - px[:, 0], px[:, 2] - px[:, 0]
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    flat = area < 1e-6
    owner, bary = TX.rasterize(uv, W, H, np.flatnonzero(~flat))
    yy, xx = np.nonzero(owner >= 0)
    f, b = owner[yy, xx], bary[yy, xx].astype(np.float32)
    pts = np.einsum("tk,tkc->tc", b, V[F[f]].astype(np.float32))
    nrm = np.einsum("tk,tkc->tc", b, np.asarray(N, np.float32)[F[f]])
    if flat.any():
        c = px[flat].mean(1)  # центърът на всеки квадрат
        steps = np.diff(np.unique(np.round(c[:, 0], 1)))
        cell = int(np.clip(np.round(steps[steps > 0.5].min()), 1, 4)) if (steps > 0.5).any() else 4
        x0 = np.round(c[:, 0] - cell / 2).astype(np.int64)
        y0 = np.round(c[:, 1] - cell / 2).astype(np.int64)
        dy, dx = np.divmod(np.arange(cell * cell), cell)
        sy, sx = (y0[:, None] + dy).reshape(-1), (x0[:, None] + dx).reshape(-1)
        ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
        which = np.repeat(np.flatnonzero(flat), cell * cell)[ok]
        yy, xx = np.r_[yy, sy[ok]], np.r_[xx, sx[ok]]
        pts = np.r_[pts, V[F[which]].mean(1).astype(np.float32)]
        nrm = np.r_[nrm, np.asarray(N, np.float32)[F[which]].mean(1)]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    return (yy, xx), pts, nrm


def recolour_atlas(atlas: np.ndarray, where: tuple[np.ndarray, np.ndarray], weight: np.ndarray,
                   rgb: tuple[int, int, int]) -> np.ndarray:  # fmt: skip
    """Изображението на текстурата с тексели `where` (редове, колони), прецветени с `weight`. Текселите между частите
    (извън стени) отново взимат новия цвят на най-близкия тексел от част (както ги запълва texture.level): оставени
    каквито са, при светъл цвят се виждаха тънки тъмни шевове на телефона, който смесва съседни тексели."""
    from . import texture as TX

    out = atlas.copy()
    yy, xx = where
    out[yy, xx] = recoloured(atlas[yy, xx], weight, rgb)
    on_face = np.zeros(atlas.shape[:2], bool)
    on_face[yy, xx] = True
    iy, ix = TX.nearest_covered(on_face)
    return np.where(on_face[..., None], out, out[iy, ix])


# --- Изображения на модела ------------------------------------------------------------------------------------------


def view_rotation(azimuth: float, elevation: float) -> np.ndarray:
    """Свят (z нагоре) -> изглед: x надясно, y далеч от камерата, z нагоре в изображението. Камерата е на `elevation`
    градуса над средата на модела и гледа към нея."""
    a, e = np.radians(azimuth), np.radians(elevation)
    look = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # от камерата към модела
    right = np.cross(look, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, look, np.cross(right, look)])


def draw(pts: np.ndarray, cols: np.ndarray, nrm: np.ndarray, R: np.ndarray, centre: np.ndarray,
         scale: float, side: int = SIDE) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Изображението (side, side, 3) uint8 и буферът му на дълбочина (side, side; inf при празно), с най-близката
    дълбочина за пиксел. Точките се рисуват при SUPER x side, осветени горе вляво от камерата, и се смаляват. scale:
    пиксели на единица от модела в изображението."""
    size = side * SUPER
    p = (pts - centre) @ R.T * scale * SUPER
    x = np.round(p[:, 0] + size / 2).astype(np.int64)
    y = np.round(size / 2 - p[:, 2]).astype(np.int64)
    keep = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    pix, depth = (y * size + x)[keep], p[keep, 1]
    order = np.lexsort((depth, pix))
    pix, first = np.unique(pix[order], return_index=True)
    sel = np.flatnonzero(keep)[order[first]]
    n = nrm[sel] @ R.T
    n *= np.where(n[:, 1:2] > 0, -1.0, 1.0)  # обърната към камерата, накъдето и да гледа
    light = np.array([-0.45, -0.75, 0.5]) / np.linalg.norm([-0.45, -0.75, 0.5])
    shade = 0.6 + 0.4 * np.clip(n @ light, 0, 1)
    rgb = np.full((size * size, 3), BG)
    zbuf = np.full(size * size, np.inf)
    rgb[pix] = np.clip(cols[sel] * shade[:, None], 0, 255)
    zbuf[pix] = depth[order[first]] / SUPER
    rgb, zbuf = rgb.reshape(size, size, 3), zbuf.reshape(size, size)
    # Малки дупки между пробите: цветът на съседите им.
    from scipy import ndimage

    seen = np.isfinite(zbuf)
    holes = ndimage.binary_closing(seen, iterations=2) & ~seen
    if holes.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~seen, return_indices=True)
        rgb[holes] = rgb[iy[holes], ix[holes]]
    small = rgb.reshape(side, SUPER, side, SUPER, 3).mean((1, 3))
    near = zbuf.reshape(side, SUPER, side, SUPER).min((1, 3))
    return np.clip(np.round(small), 0, 255).astype(np.uint8), near


def seen_in(points: np.ndarray, R: np.ndarray, centre: np.ndarray, scale: float, near: np.ndarray,
            tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Кои точки се виждат в изображението (не са скрити зад по-близка повърхност: near, буфера му на дълбочина) и
    пикселите им (ред, колона)."""
    side = near.shape[0]
    p = (points - centre) @ R.T * scale
    x = np.floor(p[:, 0] + side / 2).astype(np.int64)
    y = np.floor(side / 2 - p[:, 2]).astype(np.int64)
    inside = (x >= 0) & (x < side) & (y >= 0) & (y < side)
    x, y = np.clip(x, 0, side - 1), np.clip(y, 0, side - 1)
    return inside & (p[:, 1] <= near[y, x] + tolerance), y, x


# --- Намиране на частта ---------------------------------------------------------------------------------------------

Query = tuple[np.ndarray, np.ndarray, np.ndarray]  # точки, цветове (0-255), нормали


class NotFound(Exception):
    """Назованата част не е върху модела (CLIPSeg е отбелязал по-малко от MIN_SHARE от нея)."""


def look(work: Path, part: str, draw_from: Query, frame: np.ndarray, margin: float, name: str,
         sh: Callable[[str, list], None]) -> list[tuple]:  # fmt: skip
    """Изображения от VIEWS, всяко напаснато към точките `frame` (модела или където е намерена частта) с `margin` (дял
    от изображението) наоколо, и отметките на CLIPSeg за `part` в тях: за всяко изображение (завъртане на изгледа,
    буфер на дълбочина, отметки 0-1, пиксели на единица, център)."""
    pts, cols, nrm = draw_from
    rotations = [view_rotation(a, e) for a, e in VIEWS]
    fits = []
    for R in rotations:
        p = frame @ R.T
        lo, hi = p.min(0), p.max(0)
        scale = (1 - 2 * margin) * SIDE / max(hi[0] - lo[0], hi[2] - lo[2], 1e-9)
        fits.append((scale, R.T @ np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])))
    drawn = [draw(pts, cols, nrm, R, centre, scale) for R, (scale, centre) in zip(rotations, fits)]
    pictures, marks = work / f"recolor_{name}.npz", work / f"recolor_{name}_marks.npz"
    np.savez_compressed(pictures, images=np.stack([d[0] for d in drawn]), prompt=np.array(prompt_for(part)))
    sh("finding the part", [sys.executable, __file__, pictures, marks])
    with np.load(marks) as z:
        prob = z["prob"].astype(np.float32)
    return [(R, d[1], pr, scale, centre) for R, d, pr, (scale, centre) in zip(rotations, drawn, prob, fits)]


# Думи, които CLIPSeg познава по-добре за същата част ("base": най-много 0,13 върху кръгла основа на фигура,
# "pedestal": 0,7).
SAME_PART = {"base": "pedestal", "stand": "pedestal", "plinth": "pedestal", "platform": "pedestal",
             "podium": "pedestal", "socle": "pedestal", "bottom": "pedestal"}  # fmt: skip


def prompt_for(part: str) -> str:
    words = part.strip().lower().split()
    return " ".join(SAME_PART.get(w, w) for w in words) if words else part


def gather(queries: list[Query], looks: list[tuple[list, float]], unseen: bool = True) -> list[np.ndarray]:
    """Отметката на всяка точка за запитване: средното от изображенията, в които се вижда, претеглено по прякостта и
    по теглото на погледа; точки извън всички изображения (процепи) взимат тази на най-близката видяна точка."""
    from scipy.spatial import cKDTree

    out = []
    for q_pts, _, q_nrm in queries:
        total, weight = np.zeros(len(q_pts)), np.zeros(len(q_pts))
        for pictures, k in looks:
            for R, near, pr, scale, centre in pictures:
                vis, y, x = seen_in(q_pts, R, centre, scale, near, 2.5 / scale)  # дълбочина: резерв от няколко пиксела
                w = k * vis * np.clip(np.abs(q_nrm @ R[1]), 0.15, 1.0)
                total += w * pr[y, x]
                weight += w
        p = np.zeros(len(q_pts))
        known = weight > 0
        p[known] = total[known] / weight[known]
        if unseen and (~known).any() and known.any():
            _, idx = cKDTree(q_pts[known]).query(q_pts[~known], workers=-1)
            p[~known] = p[known][idx]
        out.append(p)
    return out


def find(work: Path, part: str, draw_from: Query, queries: list[Query], sh: Callable[[str, list], None],
         note: Callable[[str], None]) -> list[np.ndarray]:  # fmt: skip
    """Теглото (0-1) на промяната на цвета за всяка точка за запитване за `part`. draw_from: пробите от повърхността,
    от които се рисуват изображенията; queries: за всеки набор (точки, цветове, нормали). Хвърля NotFound, ако я
    няма."""
    pts = draw_from[0]
    size = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    if set(part.lower().split()) & set(SAME_PART):
        weights = stand(queries)
        if weights is not None:
            note(f"edit: part „{part}“: the stand at the bottom, up to "
                 f"{float(np.concatenate([q[0][w > 0.5, 2] for q, w in zip(queries, weights)]).max()):.1f}")  # fmt: skip
            return weights
    work.mkdir(parents=True, exist_ok=True)
    looks = [(look(work, part, draw_from, pts, MARGIN, "views", sh), 1.0)]
    prob = gather(queries, looks)
    # Втори, по-близък поглед там, където е намерена: CLIPSeg отбелязва на плочки от 1/22 от изображението си, затова
    # по-близо значи по-фино.
    allp, allq = np.concatenate(prob), np.concatenate([q[0] for q in queries])
    there = allq[allp >= FOUND]
    zoomed = False
    if len(there) > 100 and np.linalg.norm(there.max(0) - there.min(0)) < ZOOM * size:
        looks.append((look(work, part, draw_from, there, 2 * MARGIN, "closer", sh), 2.0))
        prob = gather(queries, looks)
        zoomed = True
    weights = decide(prob, queries, size)
    share = sum(float((w > 0.5).sum()) for w in weights) / max(1, sum(len(w) for w in weights))
    note(f"edit: part „{part}“ is {share:.1%} of the surface (CLIPSeg, {len(VIEWS)} pictures"
         f"{f' and {len(VIEWS)} closer' if zoomed else ''})")  # fmt: skip
    if share < MIN_SHARE:
        raise NotFound(part)
    return weights


def looking(direction: np.ndarray) -> np.ndarray:
    """Завъртане на изгледа (като view_rotation) за камера, гледаща по -direction (камерата е от тази страна)."""
    look = -np.asarray(direction, np.float64) / np.linalg.norm(direction)
    right = np.cross(look, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-3:  # право надолу или нагоре: посоката нагоре на изображението е y на модела
        right = np.cross(look, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return np.stack([right, look, np.cross(right, look)])


def tapped(work: Path, draw_from: Query, queries: list[Query], spot: np.ndarray, normal: np.ndarray,
           sh: Callable[[str, list], None], note: Callable[[str], None]) -> list[np.ndarray]:  # fmt: skip
    """Частта от модела под докоснатото от потребителя място ("make this red"): MobileSAM (моделът от app/refine.py),
    на който е посочена докоснатата точка, изрязва това, което е там, в изображения около мястото (пряко и с
    TAP_TILT към четири страни); всяка точка взима дела от изображенията, в които се вижда и е изрязана, а остава
    само парчето с мястото. Правило по цвят не би се справило: тъмночервена жилетка и тъмнокафявата коса над нея се
    различават по-малко от сенките във всяка. Връща теглото на всяка точка 0-1."""
    from scipy import ndimage

    pts, cols, nrm = draw_from
    n = normal / np.linalg.norm(normal)
    a = np.cross(n, [0.0, 0.0, 1.0])
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-3 else np.array([1.0, 0.0, 0.0])
    b = np.cross(n, a)
    t = np.radians(TAP_TILT)
    rotations = [looking(d) for d in (n, *(n * np.cos(t) + s * np.sin(t) for s in (a, -a, b, -b)))]
    images, sils, points, pictures = [], [], [], []
    for R in rotations:
        p = pts @ R.T
        lo, hi = p.min(0), p.max(0)
        scale = (1 - 2 * MARGIN) * TAP_SIDE / max(hi[0] - lo[0], hi[2] - lo[2], 1e-9)
        centre = R.T @ np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])
        image, near = draw(pts, cols, nrm, R, centre, scale, TAP_SIDE)
        vis, y, x = seen_in(spot[None], R, centre, scale, near, 3.0 / scale)
        images.append(image)
        sils.append(np.isfinite(near))
        points.append([x[0], y[0], 1.0 if vis[0] else 0.0])
        pictures.append((R, near, centre, scale))
    work.mkdir(parents=True, exist_ok=True)
    request, answer = work / "tap_views.npz", work / "tap_marks.npz"
    np.savez_compressed(request, images=np.stack(images), silhouettes=np.stack(sils), points=np.array(points))
    sh("finding the part", [sys.executable, __file__, "tap", request, answer])
    with np.load(answer) as z:
        masks = z["masks"].astype(np.float32)
    used = [(R, near, m, scale, centre) for (R, near, centre, scale), m, p in zip(pictures, masks, points) if p[2] > 0]
    if not used:
        raise NotFound("the tapped spot")
    prob = gather(queries, [(used, 1.0)], unseen=False)  # това, което никое изображение не вижда, не се предполага: докоснете там отново
    # Парчето, съдържащо мястото, върху мрежа от воксели на повърхността.
    P = np.concatenate(prob)
    X = np.concatenate([q[0] for q in queries]).astype(np.float64)
    size = float(np.linalg.norm(X.max(0) - X.min(0)))
    keys = np.floor((X - X.min(0)) / (size / 256)).astype(np.int64)
    at_voxel = tuple(keys.T)
    occ = np.zeros(tuple(keys.max(0) + 1), bool)
    occ[tuple(keys[P > 0.5].T)] = True
    labels, _ = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    near_spot = np.argsort(np.linalg.norm(X - spot, axis=1))[:50]
    held = labels[at_voxel][near_spot]
    held = held[held > 0]
    if not len(held):
        raise NotFound("the tapped spot")
    kept = labels == np.bincount(held).argmax()
    flat = np.ravel_multi_index(at_voxel, occ.shape)
    hit = ((P > 0.5) & kept[at_voxel]).astype(float)
    total = ndimage.uniform_filter(np.bincount(flat, hit, occ.size).reshape(occ.shape), 3, mode="constant")
    count = ndimage.uniform_filter(np.bincount(flat, None, occ.size).astype(float).reshape(occ.shape), 3, mode="constant")
    w = np.clip((total.reshape(-1)[flat] / np.maximum(count.reshape(-1)[flat], 1e-9) - 0.25) / 0.5, 0, 1)
    note(f"edit: the tapped part is {float((w > 0.5).mean()):.1%} of the surface (MobileSAM, {len(used)} pictures)")
    out, at = [], 0
    for q in queries:
        out.append(w[at : at + len(q[0])])
        at += len(q[0])
    return out


def stand(queries: list[Query]) -> list[np.ndarray] | None:
    """Поставката, върху която стои фигура, по формата ѝ: отдолу нагоре, докато напречното сечение на модела се стесни
    под STAND_NARROW от това на поставката (краката на фигура върху кръгла основа). CLIPSeg отбелязва такава поставка
    само на петна. None, когато моделът няма такова стъпало в долната си част STAND_MAX (чаша, ваза)."""
    from scipy.spatial import ConvexHull, QhullError

    X = np.concatenate([q[0] for q in queries])
    z = X[:, 2]
    z0, height = float(z.min()), float(np.ptp(z))
    if height <= 0:
        return None
    step = height / 100
    rng = np.random.default_rng(0)
    areas = []
    for k in range(int(100 * STAND_MAX) + 1):
        sel = np.flatnonzero((z >= z0 + k * step) & (z < z0 + (k + 1) * step))
        if len(sel) < 10:
            areas.append(0.0)
            continue
        xy = X[rng.choice(sel, min(len(sel), 4000), replace=False), :2]
        try:
            areas.append(float(ConvexHull(xy).volume))  # в 2D: площта
        except QhullError:
            areas.append(0.0)
    areas = np.array(areas)
    ref = float(np.median(areas[1:4]))  # собствената на поставката (без самото ѝ дъно)
    if ref <= 0:
        return None
    narrow = np.flatnonzero((areas < STAND_NARROW * ref) & (np.arange(len(areas)) >= 2))
    if not len(narrow):
        return None
    top = z0 + narrow[0] * step
    return [np.clip((top + step - q[0][:, 2]) / (2 * step), 0, 1) for q in queries]


def decide(prob: list[np.ndarray], queries: list[Query], size: float) -> list[np.ndarray]:
    """Грубите отметки на CLIPSeg, уточнени по цвят и по това дали са едно цяло. Цвят: от всеки цвят (кошове в Lab по
    цялата повърхност) делът, отбелязан от CLIPSeg като частта. Чело до косата е наполовина отбелязано, а ушите в
    косата още повече, но цветът на кожата рядко е отбелязан по цялата фигура, затова го запазват. Цветовете в сянка
    казват малко: последните кичури при врата са тъмни колкото наметалото отдолу (в рамките на 3 в Lab) и там решават
    само отметките на CLIPSeg (0,65 спрямо 0,03). Парчета: остават само парчетата (допиращи се в 3D) със сигурно
    отбелязани точки: очи, тъмни колкото косата, но отделно от нея, запазват цвета си. Връща теглото 0-1 на всяка
    точка за промяната на цвета: меко на около воксел (1/256 от модела)."""
    from scipy import ndimage

    P = np.clip(np.concatenate(prob), 0.001, 0.999)
    L = rgb_to_lab(np.concatenate([np.asarray(q[1], np.float64)[:, :3] for q in queries]))
    X = np.concatenate([q[0] for q in queries])
    nb = BINS
    il = np.clip((L[:, 0] / 100 * nb).astype(int), 0, nb - 1)
    ia = np.clip(((L[:, 1] + 64) / 128 * nb).astype(int), 0, nb - 1)
    ib = np.clip(((L[:, 2] + 64) / 128 * nb).astype(int), 0, nb - 1)
    idx = (il * nb + ia) * nb + ib
    marked = ndimage.uniform_filter(np.bincount(idx, (P >= FOUND).astype(float), nb**3).reshape(nb, nb, nb), 3,
                                    mode="nearest")  # fmt: skip
    count = ndimage.uniform_filter(np.bincount(idx, None, nb**3).astype(float).reshape(nb, nb, nb), 3, mode="nearest")
    by_colour = np.clip(((marked + 0.5) / (count + 1.0)).reshape(-1)[idx], 0.01, 0.99)
    says = np.clip((L[:, 0] - DARK) / 25.0, 0, 1)  # цветовете в сянка не казват нищо; осветените всичко, което могат
    if float(L.std(0).mean()) < PLAIN:  # един цвят навсякъде (сив модел): цветът не казва нищо никъде
        says[:] = 0
    final = 1 / (1 + np.exp(-(np.log(P / (1 - P)) + says * np.log(by_colour / (1 - by_colour)))))
    # Парчета върху мрежа от воксели на повърхността.
    voxel = size / 256
    keys = np.floor((X - X.min(0)) / voxel).astype(np.int64)
    at_voxel = tuple(keys.T)
    occ = np.zeros(tuple(keys.max(0) + 1), bool)
    inside = final > 0.5
    occ[tuple(keys[inside].T)] = True
    labels, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    sure = np.bincount(labels[at_voxel][inside & (P >= SURE)], minlength=n + 1)
    sure[0] = 0
    keep = sure >= max(1, KEEP_PIECE * sure.max())
    kept = keep[labels] & occ
    # Вътре или вън за всяка точка, после омекотено на около воксел: без петънца в сенки, където отметките са около
    # половината.
    flat = np.ravel_multi_index(at_voxel, occ.shape)
    hit = (inside & kept[at_voxel]).astype(float)
    total = ndimage.uniform_filter(np.bincount(flat, hit, occ.size).reshape(occ.shape), 3, mode="constant")
    count = ndimage.uniform_filter(np.bincount(flat, None, occ.size).astype(float).reshape(occ.shape), 3, mode="constant")
    w = np.clip((total.reshape(-1)[flat] / np.maximum(count.reshape(-1)[flat], 1e-9) - 0.25) / 0.5, 0, 1)
    out, at = [], 0
    for p in prob:
        out.append(w[at : at + len(p)])
        at += len(p)
    return out


# --- CLIPSeg (собствен процес) --------------------------------------------------------------------------------------


def ready() -> bool:
    return (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()


def main(pictures: Path, out: Path):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    with np.load(pictures) as z:
        images, prompt = z["images"], str(z["prompt"])
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    ids = tok.encode(prompt).ids[:77]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.log_severity_level = 3  # бележката за формата на изхода на модела при всяко изпълнение
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    prob = []
    for i, img in enumerate(images):
        x = ((img / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        logits = session.run(["logits"], {"input_ids": np.array([ids], np.int64), "pixel_values": x,
                                          "attention_mask": np.ones((1, len(ids)), np.int64)})[0]  # fmt: skip
        p = 1 / (1 + np.exp(-logits.reshape(SIDE, SIDE)))
        prob.append(p.astype(np.float16))
        print(f"{i + 1}/{len(images)}: up to {p.max():.2f}", flush=True)
    np.savez_compressed(out, prob=np.stack(prob))
    print(json.dumps({"prompt": prompt, "pictures": len(images)}), flush=True)


def tap_main(request: Path, out: Path):
    """MobileSAM върху изображенията около докоснатото място: за всяко изображение, от маските му за докоснатата
    точка, най-високо оценената, която съдържа точката и не е целият модел (над TAP_WHOLE от очертанието му)."""
    import onnxruntime as ort

    sam = Path(os.environ.get("SAM_MODEL_DIR", Path(__file__).resolve().parent.parent / "tools" / "sam"))
    enc = ort.InferenceSession(str(sam / "mobile_sam_image_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(sam / "sam_mask_decoder_multi.onnx"), providers=["CPUExecutionProvider"])
    with np.load(request) as z:
        images, sils, points = z["images"], z["silhouettes"], z["points"]
    masks = []
    for i, (img, sil, (x, y, seen)) in enumerate(zip(images, sils, points)):
        H, W = img.shape[:2]
        if not seen:
            masks.append(np.zeros((H, W), bool))
            continue
        s = 1024 / max(H, W)
        a = np.asarray(Image.fromarray(img).resize((round(W * s), round(H * s)), Image.BILINEAR), np.float32)
        emb = enc.run(None, {enc.get_inputs()[0].name: a})[0]
        feed = {
            "image_embeddings": emb,
            "point_coords": (np.array([[x + 0.5, y + 0.5], [0.0, 0.0]], np.float32) * s)[None],
            "point_labels": np.array([[1, -1]], np.float32),  # -1: допълващата точка, когато не е дадена кутия
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([H, W], np.float32),
        }
        got, scores = dec.run(["masks", "iou_predictions"], feed)
        best, best_score = None, -1.0
        for m, score in zip(got[0] > 0, scores[0]):
            share = (m & sil).sum() / max(1, sil.sum())
            if m[int(y), int(x)] and share < TAP_WHOLE and score > best_score:
                best, best_score = m & sil, float(score)
        masks.append(best if best is not None else np.zeros((H, W), bool))
        print(f"{i + 1}/{len(images)}: {0 if best is None else best.sum() / max(1, sil.sum()):.1%} of the model, "
              f"rated {best_score:.2f}", flush=True)
    np.savez_compressed(out, masks=np.stack(masks))


if __name__ == "__main__":
    if sys.argv[1] == "tap":
        tap_main(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
