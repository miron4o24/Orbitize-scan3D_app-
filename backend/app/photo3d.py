"""3D модел от една снимка с истинските ѝ детайли и цветове. Само TripoSR (app/single3d.py) дава правдоподобна, но
размита форма с разни цветове: при дистанционно копчетата бяха плоски, а цветовете сбъркани. Тук от TripoSR се взема само
голямата форма и задната страна, а останалото идва от самата снимка: очертанието е това на предмета в нея, предната
повърхност получава детайла от AI дълбочината (app/relief.py), а текстурата отпред е снимката. Така моделът отпред е
почти като снимката, а гърбът е предположение на AI. Когато кадрите са от видео и друг кадър показва същото очертание
огледално, но друга картина (предмет, завъртян на другата страна), гърбът е истински: този кадър дава цветовете и детайла
му. Повърхността са два слоя над пикселите, отпред и отзад, зашити по очертанието: затворена мрежа, без нужда от обем.
Функциите с torch остават в app/single3d.py; колоритът и картите се пазят в work/ до края на задачата."""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

if __package__:
    from . import relief as RL
    from . import texture as TX
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import relief as RL
    from app import texture as TX

INFO = "two_sided.json"
LAYERS = "two_sided_layers.npz"
ATLAS = "two_sided_atlas.jpg"
RAYS = 256  # лъчи на страна през обема на TripoSR
SAMPLES = 240  # точки по всеки лъч
POINTS = 220_000  # точки на слой най-много: при снимка 1080p дистанционно получава по една на пиксел
FACES = 400_000
CHART = 2048  # дългата страна на всяка половина от текстурата
SHAPE_BLUR = 0.08  # от размера: голямата форма на предната страна; по-тясното идва от AI дълбочината
BACK_BLUR = 0.012  # от размера: вълните на TripoSR по задната страна
MIN_THICK = 0.012  # от размера: слоевете никога не се докосват
MIN_IOU = 0.5  # очертание на формата, разминаващо се повече със снимката: остава само моделът на TripoSR
AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
SAME_WAY = 0.1  # друга посока се избира само ако очертанието ѝ пасва с толкова повече (симетричен предмет не я променя)
MIRROR_IOU = 0.65  # другата страна: очертанието на кадъра, обърнато огледално, пасва поне толкова (перспективата му пречи)
# и грубата картина не прилича на снимката: при дистанционно истинският гръб дава корелация 0,03, а същата страна, леко
# завъртяна, от 0,29 нагоре.
OTHER_PICTURE = 0.33  # над това кадърът показва същата страна; при дистанционно, вече с изправени кадри: истинският гръб 0,24, същата страна от 0,44 нагоре
PICTURE_SIDE = 32
# Предмет в ръка се върти около дългата си ос: обръщане отгоре надолу се приема само ако пасва ясно по-добре, защото
# симетрично очертание (дистанционно) пасва и на двете, а гърбът тогава излиза с главата надолу.
UPSIDE_DOWN = 0.08
ROUND = 0.15  # под тази издълженост (0: кръг, 1: линия) предметът няма посока на дължината
MIN_TURN = 1.0  # градуса: по-малък наклон не се поправя, за да не се размазва кадърът излишно
OUTLINE_BLUR = 0.004  # от размера на предмета в кадъра
RIM_SMOOTH = 6
ROUND_EDGE = 0.03  # от най-дългата страна: най-големият радиус на заобляне на ръбовете
COMPARE_SIDE = 96
RIM = 0.005  # от размера: толкова навътре от очертанието започват цветовете (при ръба има малко от фона зад предмета)


@dataclass
class Camera:
    """Изгледът, от който TripoSR приема, че е снимката, в координатите му (Z нагоре). s, t: пиксел като дял от страната
    на квадрата, t надолу."""

    center: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    focal: float  # в дялове от страната на квадрата

    def rays(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Посоките на лъчите с единица дълбочина по оста на камерата: точка = център + лъч * дълбочина."""
        x, y = (s - 0.5) / self.focal, (t - 0.5) / self.focal
        return self.forward + x[..., None] * self.right - y[..., None] * self.up

    def pixels(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(s, t, дълбочина) на точки."""
        off = points - self.center
        depth = np.maximum(np.einsum("ij,j->i", off, self.forward), 1e-9)
        s = 0.5 + self.focal * np.einsum("ij,j->i", off, self.right) / depth
        t = 0.5 - self.focal * np.einsum("ij,j->i", off, self.up) / depth
        return s, t, depth


def orbit_camera(distance: float, fov_deg: float, azimuth_deg: float = 0.0) -> Camera:
    """Като tsr.utils.get_spherical_cameras при нулев наклон: изгледът на входа е азимут 0."""
    a = np.radians(azimuth_deg)
    center = distance * np.array([np.cos(a), np.sin(a), 0.0])
    forward = -center / distance
    up = np.array([0.0, 0.0, 1.0])
    return Camera(center, forward, np.cross(forward, up), up, 0.5 / np.tan(np.radians(fov_deg) / 2))


def march(vol: np.ndarray, radius: float, level: float, cam: Camera, s: np.ndarray, t: np.ndarray, samples: int):
    """За всеки лъч дълбочината на първата и последната точка над нивото на обема (NaN: не го среща)."""
    rays = cam.rays(s, t).reshape(-1, 3)
    distance = float(np.linalg.norm(cam.center))
    depths = np.linspace(distance - 1.05 * radius, distance + 1.05 * radius, samples)
    near, far = np.full(len(rays), np.nan), np.full(len(rays), np.nan)
    scale = (vol.shape[0] - 1) / (2 * radius)
    for start in range(0, len(rays), 4000):
        part = rays[start : start + 4000]
        where = cam.center + part[:, None, :] * depths[None, :, None]
        idx = ((where + radius) * scale).reshape(-1, 3).T
        on = ndimage.map_coordinates(vol, idx, order=1, mode="constant", cval=0.0).reshape(len(part), samples) > level
        hit = on.any(1)
        near[start : start + 4000][hit] = depths[on.argmax(1)[hit]]
        far[start : start + 4000][hit] = depths[samples - 1 - on[:, ::-1].argmax(1)[hit]]
    return near.reshape(s.shape), far.reshape(s.shape)


def grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    s, t = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return s, t


def best_azimuth(vol, radius, level, distance, fov, mask: np.ndarray) -> tuple[float, dict]:
    """Посоката на камерата, от която очертанието на обема пасва на маската (квадратна): азимут 0, освен ако друга пасва
    ясно по-добре."""
    n = mask.shape[0]
    s, t = grid(n)
    fits = {}
    for az in AZIMUTHS:
        near, _ = march(vol, radius, level, orbit_camera(distance, fov, az), s, t, n)
        on = np.isfinite(near)
        fits[az] = float((on & mask).sum() / max(1, (on | mask).sum()))
    best = max(fits, key=fits.get)
    return (0.0 if fits[0.0] >= fits[best] - SAME_WAY else best), fits


def fill_gaps(a: np.ndarray) -> np.ndarray:
    """Лъчи без пресичане: стойността на най-близкия с такова."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    iy, ix = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[iy, ix]


def resized(a: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((n, n), Image.BILINEAR))


def no_pinches(q: np.ndarray) -> np.ndarray:
    """Клетки, допиращи се само с ъгъл, биха дали точка, през която минават две повърхности: едната отпада."""
    q = q.copy()
    for _ in range(8):
        a = q[:-1, :-1] & q[1:, 1:] & ~q[1:, :-1] & ~q[:-1, 1:]
        b = q[1:, :-1] & q[:-1, 1:] & ~q[:-1, :-1] & ~q[1:, 1:]
        if not a.any() and not b.any():
            break
        ya, xa = np.nonzero(a)
        q[ya, xa] = False
        yb, xb = np.nonzero(b)
        q[yb + 1, xb] = False
    return q


def closed_mesh(q: np.ndarray, front: np.ndarray, back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Затворена мрежа от двата слоя. q: (G-1, G-1) кои клетки на решетката са на предмета; front, back: (G, G, 3)
    точките им. Предната страна гледа към камерата. Връща (върхове, стени, индексите на решетката на предните върхове);
    задните върхове следват предните в същия ред."""
    g = q.shape[0] + 1
    used = np.zeros((g, g), bool)
    used[:-1, :-1] |= q
    used[1:, :-1] |= q
    used[:-1, 1:] |= q
    used[1:, 1:] |= q
    ids = np.full((g, g), -1, np.int64)
    n = int(used.sum())
    ids[used] = np.arange(n)
    ys, xs = np.nonzero(q)
    a, b, c, d = ids[ys, xs], ids[ys, xs + 1], ids[ys + 1, xs + 1], ids[ys + 1, xs]
    top = np.vstack([np.c_[a, d, c], np.c_[a, c, b]])  # обратно на часовника, както се вижда от камерата
    bottom = top[:, ::-1] + n
    # Ръбовете на предната страна без обратен ръб са очертанието; всеки става стена към задния слой.
    edges = np.vstack([top[:, [0, 1]], top[:, [1, 2]], top[:, [2, 0]]])
    key, reverse = edges[:, 0] * (2 * n) + edges[:, 1], edges[:, 1] * (2 * n) + edges[:, 0]
    rim = edges[~np.isin(key, reverse)]
    u, v = rim[:, 0], rim[:, 1]
    walls = np.vstack([np.c_[v, u, u + n], np.c_[v, u + n, v + n]])
    vertices = np.vstack([front[used], back[used]])
    # Ръбът по клетките на решетката е на стъпала: точките му се местят плавно към съседите си по ръба.
    on_rim = np.unique(np.r_[u, v])
    for _ in range(RIM_SMOOTH):
        for shift in (0, n):
            near, count = np.zeros_like(vertices), np.zeros(len(vertices))
            np.add.at(near, u + shift, vertices[v + shift])
            np.add.at(near, v + shift, vertices[u + shift])
            np.add.at(count, u + shift, 1)
            np.add.at(count, v + shift, 1)
            ids = on_rim + shift
            vertices[ids] = 0.5 * vertices[ids] + 0.5 * near[ids] / np.maximum(count[ids, None], 1)
    return vertices, np.vstack([top, bottom, walls]), np.flatnonzero(used.ravel())


def filled_colours(rgb: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Навсякъде извън предмета: цветът на най-близкото място върху него, за да не прониква фон по ръбовете на текстурата."""
    inner = ndimage.binary_erosion(inside, iterations=max(2, int(RIM * max(inside.shape))))
    if inner.sum() < 100:
        inner = inside
    iy, ix = ndimage.distance_transform_edt(~inner, return_distances=False, return_indices=True)
    return rgb[iy, ix]


def shape_of(on: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(on)
    crop = on[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
    return np.asarray(Image.fromarray(crop).resize((COMPARE_SIDE, COMPARE_SIDE), Image.BILINEAR)) > 127


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a - a.mean(), b - b.mean()
    return float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-9))


def coarse_picture(photo: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    return np.asarray(photo.crop(box).convert("L").resize((PICTURE_SIDE, PICTURE_SIDE), Image.BOX), np.float32)


def object_picture(frame: Path, m: np.ndarray) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    """Кадърът, маската му при неговия размер (едно парче) и кутията на предмета."""
    with Image.open(frame) as im:
        photo = im.convert("RGB")
    alpha = np.asarray(Image.fromarray(m).resize(photo.size, Image.BICUBIC), np.float32)
    # Маската е с четвърт от размера на кадъра: без това очертанието излиза на стъпала.
    inside = RL.one_piece(ndimage.gaussian_filter(alpha, OUTLINE_BLUR * max(1.0, float((alpha > 127).sum()) ** 0.5)) > 127)
    ys, xs = np.nonzero(inside)
    return photo, inside, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def rounded_edges(front: np.ndarray, back: np.ndarray, on: np.ndarray, cell: float, most: float) -> tuple[np.ndarray, np.ndarray]:
    """Двата слоя, събрани към средата си край очертанието по четвърт кръг: истинските предмети имат заоблени ръбове, а
    иначе лицето свършва с остра устна и отвесна стена. Самото очертание не се мести. cell: размер на клетка от решетката,
    most: най-големият радиус на заоблянето."""
    half = (back - front) / 2
    radius = max(1e-9, min(most, float(np.median(half[on]))) if on.any() else most)
    u = np.clip(ndimage.distance_transform_edt(on) * cell / radius, 0, 1)
    lift = 1 - np.sqrt(np.maximum(1 - (1 - u) ** 2, 0))
    return front + lift * half, back - lift * half


def lengthwise(inside: np.ndarray) -> float | None:
    """Посоката на дължината на предмета в кадъра, в градуси спрямо хоризонталата (по часовника в кадъра); None за почти
    кръгъл предмет, при който такава посока няма."""
    ys, xs = np.nonzero(inside)
    x, y = xs - xs.mean(), ys - ys.mean()
    xx, yy, xy = float((x * x).mean()), float((y * y).mean()), float((x * y).mean())
    spread = float(np.hypot(xx - yy, 2 * xy))
    if spread < ROUND * (xx + yy):
        return None
    return float(np.degrees(0.5 * np.arctan2(2 * xy, xx - yy)))


def turned_like(photo: Image.Image, inside: np.ndarray, angle: float | None) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    """Кадър на другата страна, завъртян така, че след обръщането предметът да лежи като на снимката (angle: посоката му
    там). Иначе наклонен в ръката предмет заема кутията си по диагонал и върху модела гърбът излиза като наклонена ивица."""
    own = lengthwise(inside)
    if angle is not None and own is not None:
        turn = (-angle - own + 90.0) % 180.0 - 90.0  # обръщането сменя знака на посоката, а тя е до 180 градуса
        if abs(turn) >= MIN_TURN:
            photo = photo.rotate(-turn, Image.BICUBIC, expand=True)  # PIL върти обратно на часовника
            alpha = Image.fromarray(inside.astype(np.uint8) * 255).rotate(-turn, Image.BICUBIC, expand=True)
            inside = RL.one_piece(np.asarray(alpha) > 127)
    ys, xs = np.nonzero(inside)
    return photo, inside, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def opposite_frame(frame: Path, others: list[tuple[Path, np.ndarray]], m: np.ndarray):
    """Кадър, показващ другата страна: очертанието му, обърнато огледално, съвпада с това на избрания кадър, а картината
    вътре не прилича на снимката нито обърната, нито както е (иначе е същата страна). Връща (кадър, маска, "lr" или "ud")
    или None."""
    photo, inside, box = object_picture(frame, m)
    shape = shape_of(inside[box[1] : box[3], box[0] : box[2]])
    grey = coarse_picture(photo, box)
    angle = lengthwise(inside)
    best, best_fit = None, MIRROR_IOU
    for f, mask in others:
        if f == frame or (mask > 127).sum() < 50:
            continue
        p, ins, b = turned_like(*object_picture(f, mask)[:2], angle)
        s2 = shape_of(ins[b[1] : b[3], b[0] : b[2]])
        g2 = coarse_picture(p, b)
        for flip, op, handicap in (("lr", np.fliplr, 0.0), ("ud", np.flipud, UPSIDE_DOWN)):
            fit = float((shape & op(s2)).sum() / max(1, (shape | op(s2)).sum())) - handicap
            like = max(correlation(grey, g2), correlation(grey, op(g2)))  # и обърната, и както е
            print(f"{f.name} {flip}: outline {fit:.2f}, picture {like:.2f}", flush=True)
            if fit >= best_fit and like <= OTHER_PICTURE:
                best, best_fit = (f, mask, flip), fit
    return best


def flipped(a: np.ndarray, how: str) -> np.ndarray:
    return np.fliplr(a) if how == "lr" else np.flipud(a)


def build(frame: Path, m: np.ndarray, place: tuple[int, int, int], vol: np.ndarray, radius: float, level: float,
          distance: float, fov: float, colour_at: Callable[[np.ndarray], np.ndarray], work: Path, out: Path,
          others: list[tuple[Path, np.ndarray]] = (), strength: float = 1.0) -> dict | None:  # fmt: skip
    """Записва модела в `out` и картите в `work`; връща описанието му или None, ако формата на TripoSR не пасва на
    снимката. place: (left, top, side), квадратът от кадъра, показан на TripoSR. colour_at: цветовете на TripoSR (0-1)
    за точки."""
    if __package__:
        from . import mesh as M
    else:
        from app import mesh as M
    started = time.time()
    photo, inside, box = object_picture(frame, m)
    x0, y0, x1, y1 = box
    left, top, side = place
    w0, h0 = x1 - x0, y1 - y0

    def to_square(a: np.ndarray, n: int) -> np.ndarray:
        """Масив върху изрязването на предмета, поставен в квадрата с n точки на страна."""
        k = (n - 1) / side
        wide, high = max(1, round(w0 * k)), max(1, round(h0 * k))
        out_ = np.zeros((n, n), np.float32)
        small = np.asarray(Image.fromarray(a.astype(np.float32)).resize((wide, high), Image.BILINEAR))
        gx, gy = int(round((x0 - left) * k)), int(round((y0 - top) * k))
        hh, ww = min(high, n - gy), min(wide, n - gx)
        out_[gy : gy + hh, gx : gx + ww] = small[:hh, :ww]
        return out_

    square = np.asarray(Image.fromarray(inside.astype(np.uint8) * 255).crop((left, top, left + side, top + side)), np.float32)
    small_mask = lambda n: np.asarray(Image.fromarray(square).resize((n, n), Image.BILINEAR)) > 127  # noqa: E731
    azimuth, fits = best_azimuth(vol, radius, level, distance, fov, small_mask(96))
    print(f"shape against the photo's outline, by direction: { {k: round(v, 2) for k, v in fits.items()} }, using {azimuth}", flush=True)
    if fits[azimuth] < MIN_IOU:
        return None
    cam = orbit_camera(distance, fov, azimuth)

    # Двата слоя на TripoSR през пикселите.
    s, t = grid(RAYS)
    near, far = march(vol, radius, level, cam, s, t, SAMPLES)
    near, far = fill_gaps(near), fill_gaps(far)
    print(f"layers traced, {time.time() - started:.0f} s", flush=True)

    # Толкова гъста решетка, че предметът да има около POINTS точки.
    n = int(min(side, side * (POINTS / max(1, inside.sum())) ** 0.5)) + 1
    on = RL.one_piece(small_mask(n))
    longest = 0.85 * distance / cam.focal  # предметът запълва 85% от квадрата на разстоянието на камерата
    width = 0.85 * n

    sess = RL.session(threads=max(2, (os.cpu_count() or 4) // 2))
    crop, part = photo.crop(box), inside[y0:y1, x0:x1]
    rel = RL.unit_relief(RL.object_depth(sess, crop, part, shifts=(0, RL.PATCH // 2)), part, max(w0, h0))
    print(f"AI detail, {time.time() - started:.0f} s", flush=True)
    amount = RL.STRENGTH * strength * longest
    front_t = RL.masked_blur(resized(near, n), on, SHAPE_BLUR * width)
    if rel is not None:
        front_t = front_t - to_square(rel, n) * amount
    back_t = RL.masked_blur(resized(far, n), on, BACK_BLUR * width)

    # Другата страна, ако кадрите я показват.
    back_chart, opposite = None, None
    found = opposite_frame(frame, list(others), m) if others else None
    if found is not None:
        f2, m2, how = found
        p2, ins2, b2 = turned_like(*object_picture(f2, m2)[:2], lengthwise(part))
        crop2, part2 = p2.crop(b2), ins2[b2[1] : b2[3], b2[0] : b2[2]]
        rel2 = RL.unit_relief(RL.object_depth(sess, crop2, part2, shifts=(0, RL.PATCH // 2)), part2, max(crop2.size))
        if rel2 is not None:
            lay = np.asarray(Image.fromarray(flipped(rel2, how)).resize((w0, h0), Image.BILINEAR))
            back_t = back_t + to_square(lay, n) * amount
        flat2 = filled_colours(np.asarray(crop2), part2)
        back_chart = np.asarray(Image.fromarray(flipped(flat2, how)).resize((w0, h0), Image.LANCZOS))
        opposite = f2.name
        print(f"the other side from {f2.name} ({how}), {time.time() - started:.0f} s", flush=True)
    back_t = np.maximum(back_t, front_t + MIN_THICK * longest)
    front_t, back_t = rounded_edges(front_t, back_t, on, longest / width, ROUND_EDGE * longest)

    # Мрежата.
    s, t = grid(n)
    rays = cam.rays(s, t)
    front = cam.center + rays * front_t[..., None]
    back = cam.center + rays * back_t[..., None]
    q = no_pinches(on[:-1, :-1] & on[1:, :-1] & on[:-1, 1:] & on[1:, 1:])
    labels, count = ndimage.label(q)
    if count > 1:
        q = labels == 1 + int(np.argmax(ndimage.sum_labels(q, labels, np.arange(1, count + 1))))
    vertices, faces, used = closed_mesh(q, front, back)

    # Цветовете: отпред снимката, отзад другият кадър или цветовете на TripoSR, приближени до снимката.
    rgb = np.asarray(photo.crop((left, top, left + side, top + side)).resize((n, n), Image.BILINEAR), np.float32)
    flat = filled_colours(np.asarray(crop), part)
    if back_chart is None:
        back_points = cam.center + cam.rays(*grid(RAYS)) * resized(back_t, RAYS)[..., None]
        ai = (np.clip(colour_at(back_points.reshape(-1, 3).astype(np.float32)), 0, 1) * 255).reshape(RAYS, RAYS, 3).astype(np.float32)
        small_on = small_mask(RAYS)
        real = np.asarray(photo.crop((left, top, left + side, top + side)).resize((RAYS, RAYS), Image.BILINEAR), np.float32)
        if small_on.sum() > 50:
            ai = np.clip(ai + 0.7 * (np.median(real[small_on], 0) - np.median(ai[small_on], 0)), 0, 255)
        k = (RAYS - 1) / side
        ax0, ay0 = int(round((x0 - left) * k)), int(round((y0 - top) * k))
        ax1, ay1 = max(ax0 + 1, int(round((x1 - left) * k))), max(ay0 + 1, int(round((y1 - top) * k)))
        part_ai = filled_colours(ai.astype(np.uint8)[ay0:ay1, ax0:ax1], small_on[ay0:ay1, ax0:ax1])
        back_chart = np.asarray(Image.fromarray(part_ai).resize((w0, h0), Image.BICUBIC))
    back_big = np.stack([to_square(back_chart[..., c], n) for c in range(3)], -1)  # само за цвят на върховете
    rows = used // n, used % n
    gain = TX.white_gain(frame.parent, [frame.name], {frame.name: m})
    print(f"white balance: red, green, blue x {gain.round(2).tolist()}", flush=True)
    colors = TX.balanced(np.vstack([rgb[rows], back_big[rows]]).astype(np.uint8), gain)
    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=np.c_[colors, np.full(len(colors), 255, np.uint8)], process=False)
    if mesh.volume < 0:
        mesh.invert()
    closed = bool(mesh.is_watertight)
    print(f"two layers: {len(mesh.faces)} faces, closed {closed}, {time.time() - started:.0f} s", flush=True)
    if not closed:
        return None
    if len(mesh.faces) > FACES:
        mesh = M.decimate(mesh, FACES)
    mesh.export(out)

    # Текстурата: двете половини една до друга по късата страна на предмета, така че атласът да остане почти квадратен.
    k = min(1.0, CHART / max(w0, h0))
    cw, ch = max(1, round(w0 * k)), max(1, round(h0 * k))
    sideways = ch >= cw
    atlas = Image.new("RGB", (2 * cw, ch) if sideways else (cw, 2 * ch))
    atlas.paste(Image.fromarray(flat).resize((cw, ch), Image.LANCZOS), (0, 0))
    atlas.paste(Image.fromarray(back_chart).resize((cw, ch), Image.LANCZOS), (cw, 0) if sideways else (0, ch))
    Image.fromarray(TX.balanced(np.asarray(atlas), gain)).save(work / ATLAS, quality=93)
    pick = np.linspace(0, n - 1, min(n, 512)).astype(int)
    np.savez_compressed(work / LAYERS, front=front_t[np.ix_(pick, pick)].astype(np.float32), back=back_t[np.ix_(pick, pick)].astype(np.float32))
    info = {"frame": frame.name, "opposite": opposite, "faces": int(len(mesh.faces)), "azimuth": azimuth, "distance": distance,
            "fov": fov, "fit": round(fits[azimuth], 3), "sideways": bool(sideways), "chart": [cw, ch],
            "box": [(x0 - left) / side, (y0 - top) / side, (x1 - left) / side, (y1 - top) / side]}  # fmt: skip
    (work / INFO).write_text(json.dumps(info))
    print(f"two sided model: {len(mesh.faces)} faces, in {time.time() - started:.0f} s", flush=True)
    return info


def texture_model(job: Path, work: Path, out: trimesh.Trimesh) -> dict:
    """Текстурата на готовия модел (координати за печат, с to_print от TripoSR): всяка точка се проектира в снимката, а
    всяка стена получава половината от атласа според това на кой слой е по-близо. Записва я в задачата като app/texture.py."""
    info = json.loads((work / INFO).read_text())
    cam = orbit_camera(info["distance"], info["fov"], info["azimuth"])
    points = trimesh.transform_points(out.vertices, np.linalg.inv(out.metadata["to_print"]))
    s, t, depth = cam.pixels(points)
    with np.load(work / LAYERS) as z:
        front, back = z["front"], z["back"]
    faces = np.asarray(out.faces)
    n = front.shape[0]
    fx = np.clip(np.round(s[faces].mean(1) * (n - 1)).astype(int), 0, n - 1)
    fy = np.clip(np.round(t[faces].mean(1) * (n - 1)).astype(int), 0, n - 1)
    d = depth[faces].mean(1)
    is_back = np.abs(d - back[fy, fx]) < np.abs(d - front[fy, fx])
    bx0, by0, bx1, by1 = info["box"]
    cw, ch = info["chart"]
    sideways = info["sideways"]
    total_w, total_h = (2 * cw, ch) if sideways else (cw, 2 * ch)
    # Малко навътре от ръба на половината: мипмапите на телефона не бива да смесват двете страни.
    x = np.clip((s - bx0) / (bx1 - bx0), 1.5 / cw, 1 - 1.5 / cw) * cw
    y = np.clip((t - by0) / (by1 - by0), 1.5 / ch, 1 - 1.5 / ch) * ch
    normals = np.asarray(out.vertex_normals)
    offset = 0
    parts_v, parts_n, parts_uv, parts_f = [], [], [], []
    for chart, chosen in ((0, faces[~is_back]), (1, faces[is_back])):
        if not len(chosen):
            continue
        unique, inverse = np.unique(chosen, return_inverse=True)
        u = (x[unique] + (cw if chart and sideways else 0)) / total_w
        v = (y[unique] + (ch if chart and not sideways else 0)) / total_h
        parts_v.append(out.vertices[unique])
        parts_n.append(normals[unique])
        parts_uv.append(np.c_[u, 1 - v])  # v нагоре, както в app/texture.py
        parts_f.append(inverse.reshape(-1, 3) + offset)
        offset += len(unique)
    TX.clear(job)
    np.savez_compressed(job / TX.NPZ, count=1, vertices_0=np.vstack(parts_v).astype(np.float32),
                        normals_0=np.vstack(parts_n).astype(np.float32), uv_0=np.vstack(parts_uv).astype(np.float32),
                        faces_0=np.vstack(parts_f).astype(np.uint32))  # fmt: skip
    (job / "texture_0.jpg").write_bytes((work / ATLAS).read_bytes())
    return {"front_faces": int((~is_back).sum()), "back_faces": int(is_back.sum()), "atlas": [total_w, total_h]}
