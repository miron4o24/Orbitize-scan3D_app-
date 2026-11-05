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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

if __package__:
    from . import relief as RL
    from . import sideview as SV
    from . import texture as TX
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import relief as RL
    from app import sideview as SV
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
FOREGROUND = 0.85  # дял от квадрата, запълнен от предмета (single3d.FOREGROUND)
FLAT_BODY = 0.2  # от дължината: толкова тънък предмет е плосък и тялото му идва от измерената дебелина, не от предположението на TripoSR
FLAT_EDGE = 0.015  # от дължината: радиусът на ръба на плосък предмет (дистанционно: около 3 мм)
FLAT_END = 0.08  # от дължината: край, в който лента през цялата ширина е грешка на AI (при дистанционно фалшив ръб от 3 мм)
FLAT_BACK_QUIET = 0.0016  # от дължината: под това неравностите по гърба на плосък предмет са шум, защото слабият му релеф се усилва
SAME_VIEWS = 3  # толкова още кадъра от същата страна се сравняват с избрания
SAME_VIEW_FEATURES = 30  # толкова общи особености трябват, за да се наложат два кадъра точно
SIDE_FIT = 0.3  # толкова трябва да съвпадат страничният профил и редовете на AI детайла, за да е ясно кое къде е
SIDE_SLIDE = 0.03  # от дължината: разминаване на двете след поправката на перспективата
SIDE_STRETCH = 0.02
SIDE_QUIET = (0.0016, 0.0032)  # от дължината: под първото измереното е шум (0,3 мм при 192 мм), над второто е истинско
SIDE_FLOOR = 0.0013  # от дължината: ред с по-малко AI височина от това не се усилва още: там няма какво да се повдигне
SIDE_GAIN = 6.0  # толкова най-много се усилва AI височината
SIDE_BAND = 0.008  # от дължината: измерената височина важи и за толкова встрани (1,5 мм при 192 мм)
FLAT_DRIFT = 0.06  # от дължината: по-широкото от това в детайла на плосък предмет е грешка на AI (подути ъгли), не форма
FLAT_RAISED = 0.016  # от дължината: най-издаденото на плосък предмет (дистанционно, по линийка: 3 мм от 192)
FLAT_RECESS = 0.004  # от дължината: най-дълбокото; без това процеп на капак излиза над 2,4 мм вместо тънка линия
FEATURE_SIDE = 640  # дългата страна на изрязания предмет за сравняване на особености
SAME_FEATURES = 8  # толкова общи особености и кадърът показва същата страна (дистанционно: истински гръб 0, завъртяно лице 10 до 180)
OTHER_PICTURE = 0.6  # над това кадърът показва същата страна дори без особености; грубо сравнение: истинският гръб на дистанционно 0,40
PICTURE_SIDE = 32
# Предмет в ръка се върти около дългата си ос: обръщане отгоре надолу се приема само ако пасва ясно по-добре, защото
# симетрично очертание (дистанционно) пасва и на двете, а гърбът тогава излиза с главата надолу.
UPSIDE_DOWN = 0.08
# Другата страна: истинският гръб е със същата дължина и пропорции в кадъра. Краят на бялото дистанционно, гледан по оста, пасваше
# с очертанието си (0,82, защото се мащабира до квадрат) и гърбът на модела получи картината на края с лампичката.
SAME_LENGTH = (0.7, 1.43)  # дължината на предмета в кадъра спрямо избрания (краят по оста: 0,33)
SAME_ASPECT = (0.75, 1.33)  # ширина / дължина спрямо избрания (кадър отстрани: 0,5)
PICTURE_WEIGHT = 0.5  # от двата кадъра с добро очертание по-различната фина картина (без светлината) е по-вероятно другата страна
# Пръсти върху лицето само в избрания кадър (бялото дистанционно: пръст върху +10 и 60 и издатина отстрани): другите изгледи от
# същата страна дават медианата.
COVERED = 25.0  # разлика в цвета (0-255) от медианата, без сянката на кадъра, над която мястото е закрито в този кадър
SHADE_WINDOW = 0.25  # от размера на лицето: сянката (бавната разлика между кадрите) е медианата в толкова голям прозорец
SHADE_POINTS = 160  # сянката се смята при толкова точки по дългата страна
# Друг кадър отпред, ако в избрания лицето е закрито (бялото дистанционно: пръст върху +10 в най-резкия кадър, а два други кадъра са
# чисти). Замяната от другите изгледи е само за остатъка: тя оставя двоен образ там, където налагането не е съвсем точно.
SWITCH_OUTSIDE = 0.005  # от лицето: очертанието на избрания кадър стърчи толкова извън другите (ръка през ръба) и се търси друг
SWITCH_RATIO = 0.5  # чието стърчи поне толкова пъти по-малко
FINGER_REACH = 6  # пиксела: закрито място, толкова близо до стърчащото, е пръстът, влязъл отстрани
SHARP_ENOUGH = 0.5  # и поне толкова рязък
SIZE_ENOUGH = 0.8  # и поне толкова голям в кадъра
SHARED_PART = 0.8  # от лицето: толкова трябва да покрива друг изглед, наложен върху него
KEEP_OUTLINE = 0.9  # медианното очертание замества това на кадъра само ако пази поне толкова от него
DOMED = 0.05  # от ширината: лице, издуто поне толкова по посоките на повърхността (MoGe-2), е заоблено (бялото дистанционно: гръб 0,2, лице 0,01)
THINNEST_END = 0.004  # от дължината: най-тънкото място на тялото (край, скосяване)
ROUND = 0.15  # под тази издълженост (0: кръг, 1: линия) предметът няма посока на дължината
MIN_TURN = 1.0  # градуса: по-малък наклон не се поправя, за да не се размазва кадърът излишно
STRAIGHT_TOL = 0.006  # от дължината на предмета: толкова отклонение още се смята за права
STRAIGHT_RUN = 0.2  # от дължината на предмета: само толкова дълъг участък се изправя
ROUND_SMOOTH = 0.012  # от страната на решетката: заглаждане на разстоянието до очертанието
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


def straightened(inside: np.ndarray) -> np.ndarray:
    """Маската с дългите почти прави участъци от очертанието, поставени върху истинска права: страните на дистанционно са
    прави, а AI маската леко се колебае. Закръглени места остават: кръг, колкото и голям в кадъра, няма участък от
    STRAIGHT_RUN в рамките на STRAIGHT_TOL."""
    from skimage import measure

    padded = np.pad(inside, 1).astype(np.float32)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return inside
    outline = max(contours, key=len)
    if len(outline) < 50 or not np.array_equal(outline[0], outline[-1]):
        return inside
    ys, xs = np.nonzero(inside)
    longest = float(max(np.ptp(ys), np.ptp(xs)) + 1)
    corners = measure.approximate_polygon(outline, STRAIGHT_TOL * longest)
    points = outline[:-1].copy()
    found = 0
    at = np.array([np.flatnonzero((points == c).all(1))[0] for c in corners[:-1]])
    for a, b in zip(at, np.r_[at[1:], at[0]]):
        span = np.arange(a, b + 1) if a < b else np.r_[np.arange(a, len(points)), np.arange(0, b + 1)]
        part = points[span]
        if np.linalg.norm(part[-1] - part[0]) < STRAIGHT_RUN * longest:
            continue
        keep = max(1, int(0.03 * len(span)))  # краищата на участъка остават: там започва заобленият ъгъл
        middle = span[keep:-keep]
        centre = points[middle].mean(0)
        rel = points[middle] - centre
        xx, yy, xy = float((rel[:, 1] ** 2).sum()), float((rel[:, 0] ** 2).sum()), float((rel[:, 0] * rel[:, 1]).sum())
        angle = 0.5 * np.arctan2(2 * xy, xx - yy)
        along = np.array([np.sin(angle), np.cos(angle)])  # (ред, колона)
        points[middle] = centre + np.outer(rel @ along, along)
        found += 1
    if not found:
        return inside
    drawing = Image.new("L", (padded.shape[1], padded.shape[0]), 0)
    ImageDraw.Draw(drawing).polygon([(float(c), float(r)) for r, c in points], fill=255)
    new = np.asarray(drawing)[1:-1, 1:-1] > 127
    holes = ndimage.binary_fill_holes(inside) & ~inside  # истинските отвори (дръжка) остават
    return RL.one_piece(new & ~holes)


def object_picture(frame: Path, m: np.ndarray) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    """Кадърът, маската му при неговия размер (едно парче) и кутията на предмета."""
    with Image.open(frame) as im:
        photo = im.convert("RGB")
    alpha = np.asarray(Image.fromarray(m).resize(photo.size, Image.BICUBIC), np.float32)
    # Маската е с четвърт от размера на кадъра: без това очертанието излиза на стъпала.
    inside = RL.one_piece(ndimage.gaussian_filter(alpha, OUTLINE_BLUR * max(1.0, float((alpha > 127).sum()) ** 0.5)) > 127)
    inside = straightened(inside)
    ys, xs = np.nonzero(inside)
    return photo, inside, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def rounded_edges(front: np.ndarray, back: np.ndarray, on: np.ndarray, cell: float, most: float,
                  wall: bool = False, exact: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Двата слоя, събрани към средата си край очертанието по четвърт кръг: истинските предмети имат заоблени ръбове, а
    иначе лицето свършва с остра устна и отвесна стена. Самото очертание не се мести. cell: размер на клетка от решетката,
    most: най-големият радиус на заоблянето. wall: слоевете спират на радиуса от лицето си, а между тях остава права стена, както
    при кутия със заоблени ръбове."""
    half = (back - front) / 2
    radius = max(1e-9, min(most, float(np.median(half[on]))) if on.any() else most)
    # По леко наклонен прав ръб решетката прави дълги стъпала, а всяко стъпало оставаше като резка през заобления ръб. Заглажда се
    # разстоянието със знак: при него правият ръб си остава прав, защото отвън не влизат нули, и се махат и дългите стъпала.
    signed = ndimage.distance_transform_edt(on) - ndimage.distance_transform_edt(~on)
    far = ndimage.gaussian_filter(signed.astype(np.float32), ROUND_SMOOTH * on.shape[0])
    if exact is not None:  # разстоянието до истинските прави на очертанието, където го има (rim_on_lines)
        far = np.where(np.isfinite(exact), exact, far)
    u = np.clip(far * cell / radius, 0, 1)
    lift = 1 - np.sqrt(np.maximum(1 - (1 - u) ** 2, 0))
    if wall:
        return front + lift * radius, back - lift * radius
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


def features(photo: Image.Image, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Особеностите на изрязан предмет (ORB): местата им и описанията им, само върху предмета."""
    from skimage.feature import ORB

    k = FEATURE_SIDE / max(photo.size)
    size = max(8, round(photo.width * k)), max(8, round(photo.height * k))
    grey = np.asarray(photo.convert("L").resize(size, Image.LANCZOS), np.float32) / 255
    on = np.asarray(Image.fromarray(inside).resize(size, Image.NEAREST))
    orb = ORB(n_keypoints=500, fast_threshold=0.04)
    try:
        orb.detect_and_extract(grey)
    except RuntimeError:  # едноцветна страна няма особености
        return None
    keep = on[orb.keypoints[:, 0].astype(int), orb.keypoints[:, 1].astype(int)]
    return orb.keypoints[keep], orb.descriptors[keep]


def shared_features(a: tuple[np.ndarray, np.ndarray] | None, b: tuple[np.ndarray, np.ndarray] | None) -> int:
    """Колко особености на двата кадъра са едно и също място: истинска проверка дали показват една страна, защото се търси
    обща перспектива. Върху завъртяно лице остават десетки, а срещу другата страна няма нито една."""
    return matched_features(a, b)[0]


def matched_features(a: tuple[np.ndarray, np.ndarray] | None, b: tuple[np.ndarray, np.ndarray] | None) -> tuple[int, np.ndarray | None]:
    """(колко са общите особености, перспективата 3 x 3 от точките на `a` към тези на `b`, x и y в мащаба на features)."""
    from skimage.feature import match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform

    if a is None or b is None or len(a[0]) < 8 or len(b[0]) < 8:
        return 0, None
    pairs = match_descriptors(a[1], b[1], cross_check=True, max_ratio=0.85)
    if len(pairs) < 8:
        return 0, None
    model, ok = ransac((a[0][pairs[:, 0]][:, ::-1], b[0][pairs[:, 1]][:, ::-1]), ProjectiveTransform, min_samples=4,
                       residual_threshold=6, max_trials=300, rng=np.random.default_rng(0))  # fmt: skip
    return (0, None) if ok is None or model is None else (int(ok.sum()), model.params)


def same_side_relief(rel: np.ndarray, crop: Image.Image, part: np.ndarray, views: list[tuple[Image.Image, np.ndarray]],
                     relief_of: Callable[[Image.Image, np.ndarray], np.ndarray | None]) -> np.ndarray:  # fmt: skip
    """Детайлът на избрания кадър, сравнен с другите му изгледи от същата страна: всеки се налага върху него по общите особености и
    остава медианата. Размазано или лъснало място в един кадър не минава, а истинското копче е във всички. relief_of: AI детайлът
    на един изглед; views: (изрязан предмет, очертанието му)."""
    from skimage.transform import ProjectiveTransform, warp

    maps = [rel]
    for count, view, inside, to_view in aligned_views(crop, part, views)[:SAME_VIEWS]:
        other = relief_of(view, inside)
        if other is None:
            continue
        # warp иска картата от изхода към входа: тук от пикселите на crop към тези на view.
        moved = warp(other.astype(np.float64), ProjectiveTransform(to_view), output_shape=rel.shape, order=1, cval=np.nan, preserve_range=True)
        maps.append(moved.astype(np.float32))
        print(f"detail compared with another view: {count} features in common", flush=True)
    if len(maps) == 1:
        return rel
    return np.where(part, np.nanmedian(np.stack(maps), axis=0), 0).astype(np.float32)


def upright(photo: Image.Image, inside: np.ndarray) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    """Плосък предмет, завъртян така, че дължината му да върви по решетката: по наклонен прав ръб решетката прави дълги стъпала, а те
    остават като напречни резки по стената на модела. Връща (снимка, очертание, кутия), както object_picture."""
    angle = lengthwise(inside)
    if angle is None:
        ys, xs = np.nonzero(inside)
        return photo, inside, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return turned_like(photo, inside, -90.0 if abs(angle) > 45.0 else 0.0)


def centred_place(box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """(left, top, side): квадратът около кутията, който предметът запълва с дял FOREGROUND, както single3d.cut_out."""
    x0, y0, x1, y1 = box
    side = int(max(y1 - y0, x1 - x0) / FOREGROUND)
    return x0 - (side - (x1 - x0)) // 2, y0 - (side - (y1 - y0)) // 2, side


def without_end_bands(rel: np.ndarray, inside: np.ndarray, width: float) -> np.ndarray:
    """Лентите през цялата ширина в двата края на изправен плосък предмет, махнати с медианата на всеки ред. Копче заема само част
    от реда и остава."""
    sideways = rel.shape[1] > rel.shape[0]
    a, on = (rel.T, inside.T) if sideways else (rel, inside)
    out = a.copy()
    reach = min(max(1, int(FLAT_END * width)), a.shape[0] // 2)
    for k in range(reach):
        weight = min(1.0, 2.0 * (reach - k) / reach)  # пълно в първата половина, после плавно до нула
        for row in (k, a.shape[0] - 1 - k):
            if on[row].any():
                out[row] = a[row] - weight * float(np.median(a[row][on[row]]))
    return out.T if sideways else out


def flat_relief(rel: np.ndarray, inside: np.ndarray, width: float) -> np.ndarray:
    """Детайлът на плосък предмет: без широките издутини, които са грешка на AI, и в истинските за такъв предмет граници (в
    единици relief.STRENGTH, като `rel`). Връща се до нула към очертанието."""
    out = rel - RL.masked_blur(rel, inside, FLAT_DRIFT * width)
    out = without_end_bands(out, inside, width)
    out = out - RL.rim_of(out, inside, max(1, int(FLAT_END * width)))
    out = np.clip(out, -FLAT_RECESS / RL.STRENGTH, FLAT_RAISED / RL.STRENGTH)
    fade = np.clip(ndimage.distance_transform_edt(inside) / max(1.0, RL.EDGE * width), 0, 1)
    return np.where(inside, out * fade * fade * (3 - 2 * fade), 0).astype(np.float32)


def along_length(a: np.ndarray) -> np.ndarray:
    """Масивът с редове през дължината на изправения предмет."""
    return a.T if a.shape[1] > a.shape[0] else a


def row_heights(rel: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """На всеки ред през дължината: колко стърчи детайлът, в дялове от дължината. Редовете са по дългата страна на масива."""
    a, on = along_length(rel), along_length(inside)
    out = np.zeros(a.shape[0], np.float32)
    for row in np.flatnonzero(on.any(1)):
        out[row] = np.percentile(a[row][on[row]], 97)
    return np.maximum(out, 0) * RL.STRENGTH


def fit_side(rows: np.ndarray, profile: np.ndarray) -> tuple[float, bool, float, float]:
    """(съвпадение, обърнат ли е, изместване, разтягане): как страничният профил ляга върху редовете на AI детайла."""
    grid_ = (np.arange(SV.POINTS) + 0.5) / SV.POINTS
    here = ndimage.gaussian_filter1d(np.interp(grid_, (np.arange(len(rows)) + 0.5) / len(rows), rows), 1.5)
    best = (-2.0, False, 0.0, 1.0)
    for flip in (False, True):
        fit, shift, scale = SV.best_fit(profile[::-1] if flip else profile, here, SIDE_SLIDE, SIDE_STRETCH)
        if fit > best[0]:
            best = (fit, flip, shift, scale)
    return best


def with_side_heights(rel: np.ndarray, inside: np.ndarray, measured: np.ndarray, most: float = SIDE_GAIN) -> np.ndarray:
    """Детайлът с измерените височини: всеки ред се мащабира така, че да стърчи колкото страничният профил на това място (дялове от
    дължината, по дължината на масива). AI казва къде е копчето, а страничният изглед колко е високо. Вдлъбнатите места само намаляват."""
    a = along_length(rel)
    n = a.shape[0]
    want = np.interp((np.arange(n) + 0.5) / n, (np.arange(len(measured)) + 0.5) / len(measured), measured)
    t = np.clip((want - SIDE_QUIET[0]) / (SIDE_QUIET[1] - SIDE_QUIET[0]), 0, 1)
    want = want * t * t * (3 - 2 * t)
    band = max(1, int(SIDE_BAND * n))
    want = ndimage.grey_dilation(want, size=2 * band + 1)
    have = ndimage.grey_dilation(row_heights(rel, inside), size=2 * band + 1)
    gain = ndimage.gaussian_filter1d(np.clip(want / np.maximum(have, SIDE_FLOOR), 0, most), band)[:, None]
    out = np.where(a > 0, a * gain, a * np.minimum(gain, 1)).astype(np.float32)
    return out.T if rel.shape[1] > rel.shape[0] else out


def straight_runs(on: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]] | None:
    """Точките на очертанието (ред, колона), на кой прав участък е всяка (-1: никой) и правите участъци (точка, посока). None без контур."""
    from skimage import measure

    contours = measure.find_contours(np.pad(on, 1).astype(np.float32), 0.5)
    if not contours:
        return None
    outline = max(contours, key=len)
    if len(outline) < 50 or not np.array_equal(outline[0], outline[-1]):
        return None
    ys, xs = np.nonzero(on)
    longest = float(max(np.ptp(ys), np.ptp(xs)) + 1)
    corners = measure.approximate_polygon(outline, STRAIGHT_TOL * longest)
    points = outline[:-1] - 1
    run = np.full(len(points), -1)
    lines = []
    at = np.array([np.flatnonzero((outline[:-1] == c).all(1))[0] for c in corners[:-1]])
    for a, b in zip(at, np.r_[at[1:], at[0]]):
        span = np.arange(a, b + 1) if a < b else np.r_[np.arange(a, len(points)), np.arange(0, b + 1)]
        if np.linalg.norm(points[span[-1]] - points[span[0]]) < STRAIGHT_RUN * longest:
            continue
        keep = max(1, int(0.03 * len(span)))
        middle = span[keep:-keep]
        centre = points[middle].mean(0)
        rel = points[middle] - centre
        angle = 0.5 * np.arctan2(2 * float((rel[:, 0] * rel[:, 1]).sum()), float((rel[:, 1] ** 2).sum() - (rel[:, 0] ** 2).sum()))
        run[middle] = len(lines)
        lines.append((centre, np.array([np.sin(angle), np.cos(angle)])))
    return points, run, lines


def rim_cells(q: np.ndarray) -> np.ndarray:
    """Точките на решетката (G, G) по ръба на стените q (G-1, G-1)."""
    g = q.shape[0] + 1
    count = np.zeros((g, g), np.int8)
    count[:-1, :-1] += q
    count[1:, :-1] += q
    count[:-1, 1:] += q
    count[1:, 1:] += q
    return (count > 0) & (count < 4)


def rim_on_lines(s: np.ndarray, t: np.ndarray, on: np.ndarray, q: np.ndarray, reach: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Точките на решетката s, t с ръба върху истинските прави на очертанието: решетката минава по прав ръб на стъпала, а те оставаха
    като резки по стената и като зъби по края на предмета. Движат се само точките на ръба и с по-малко от клетка. Трето: разстоянието
    до правата в клетки, на reach клетки навътре (другаде nan), за заоблянето на ръба: четвърт кръг при самия ръб е почти отвесен, така
    че разликите от една клетка, идващи от стъпалата, ставаха разлики в дебелината от милиметър."""
    from scipy.spatial import cKDTree

    exact = np.full(s.shape, np.nan, np.float32)
    found = straight_runs(on)
    if found is None or not found[2]:
        return s, t, exact
    points, run, lines = found
    rim = rim_cells(q)
    tree = cKDTree(points)
    rows, cols = np.nonzero(rim)
    if not len(rows):
        return s, t, exact
    here = np.c_[rows, cols].astype(np.float64)
    which = run[tree.query(here)[1]]
    new_s, new_t = s.copy(), t.copy()
    step = 1.0 / (s.shape[0] - 1)
    for k, (centre, along) in enumerate(lines):
        chosen = np.flatnonzero(which == k)
        if not len(chosen):
            continue
        # Двете точки на едно стъпало падат почти на едно място и стената помежду им става черта: разстоянията се изравняват.
        place = (here[chosen] - centre) @ along
        order = np.argsort(place)
        even = np.empty_like(place)
        even[order] = ndimage.gaussian_filter1d(place[order], 2.0, mode="nearest")
        moved = centre + np.outer(even, along)
        new_t[rows[chosen], cols[chosen]] = moved[:, 0] * step
        new_s[rows[chosen], cols[chosen]] = moved[:, 1] * step
    if reach > 0:
        depth = ndimage.distance_transform_edt(on)
        ys, xs = np.nonzero(on & (depth <= reach))
        near = np.c_[ys, xs].astype(np.float64)
        owner = run[tree.query(near)[1]]
        for k, (centre, along) in enumerate(lines):
            chosen = owner == k
            rel = near[chosen] - centre
            exact[ys[chosen], xs[chosen]] = np.abs(rel[:, 0] * along[1] - rel[:, 1] * along[0])
        exact[rows, cols] = 0.0  # целият ръб е на една височина, иначе там, където правата свършва, остава стъпало
    return new_s, new_t, exact


def raised_part(rel: np.ndarray, inside: np.ndarray) -> float:
    """Колко се издава детайлът над лицето: измерената дебелина вече го включва, затова тялото се намалява с толкова."""
    return max(0.0, float(np.percentile(rel[inside], 98)))


def flat_layers(on: np.ndarray, thin: float, depth: float, focal: float) -> tuple[np.ndarray, np.ndarray]:
    """Двата слоя на плосък предмет: успоредни равнини на измерената дебелина thin (от дължината му), перпендикулярни на
    камерата, с дълбочина около depth. Така страничният профил е прав и дебелината е истинската. on: (n, n) предметът; focal: в
    дялове от страната на квадрата."""
    ys, xs = np.nonzero(on)
    x, y = xs - xs.mean(), ys - ys.mean()
    angle = 0.5 * np.arctan2(2 * float((x * y).mean()), float((x * x).mean() - (y * y).mean()))
    cells = float(np.ptp(x * np.cos(angle) + y * np.sin(angle)))
    half = 0.5 * thin * cells * depth / (focal * (on.shape[0] - 1))
    return np.full(on.shape, depth - half, np.float32), np.full(on.shape, depth + half, np.float32)


def length_and_width(inside: np.ndarray) -> tuple[float, float]:
    """Дължината и ширината на очертанието по собствените му оси, в пиксели."""
    ys, xs = np.nonzero(inside)
    x, y = xs - xs.mean(), ys - ys.mean()
    a = 0.5 * np.arctan2(2 * float((x * y).mean()), float((x * x).mean() - (y * y).mean()))
    return float(np.ptp(x * np.cos(a) + y * np.sin(a))) + 1, float(np.ptp(y * np.cos(a) - x * np.sin(a))) + 1


def fine_picture(grey: np.ndarray) -> np.ndarray:
    """Грубата картина без светлината: остава напечатаното и изпъкналото. Бял гръб и бяло лице иначе си приличаха (0,80)."""
    return grey - ndimage.gaussian_filter(grey, PICTURE_SIDE / 8)


def opposite_frame(frame: Path, others: list[tuple[Path, np.ndarray]], m: np.ndarray):
    """Кадър, показващ другата страна: предметът е със същата дължина и пропорции, очертанието му, обърнато огледално, съвпада с
    това на избрания кадър, а общи особености няма (иначе е същата страна). Когато лицето има особености (други кадри от същата
    страна ги споделят), това стига; без тях картината вътре не бива да прилича на снимката. От подходящите печели този с
    най-добро очертание и най-различна фина картина. Връща (кадър, маска, "lr" или "ud") или None."""
    photo, inside, box = object_picture(frame, m)
    shape = shape_of(inside[box[1] : box[3], box[0] : box[2]])
    grey = coarse_picture(photo, box)
    fine = fine_picture(grey)
    angle = lengthwise(inside)
    own = features(photo.crop(box), inside[box[1] : box[3], box[0] : box[2]])
    length, width = length_and_width(inside)
    textured, fitting = False, []
    for f, mask in others:
        if f == frame or (mask > 127).sum() < 50:
            continue
        p, ins, b = turned_like(*object_picture(f, mask)[:2], angle)
        same = shared_features(own, features(p.crop(b), ins[b[1] : b[3], b[0] : b[2]]))
        print(f"{f.name}: {same} features in common", flush=True)
        if same >= SAME_FEATURES:
            textured = True
            continue
        l2, w2 = length_and_width(ins)
        size, proportions = l2 / length, (w2 / l2) / (width / length)
        if not (SAME_LENGTH[0] <= size <= SAME_LENGTH[1] and SAME_ASPECT[0] <= proportions <= SAME_ASPECT[1]):
            print(f"{f.name}: another view, length {size:.2f} and proportions {proportions:.2f} of this one", flush=True)
            continue
        s2 = shape_of(ins[b[1] : b[3], b[0] : b[2]])
        g2 = coarse_picture(p, b)
        for flip, op, handicap in (("lr", np.fliplr, 0.0), ("ud", np.flipud, UPSIDE_DOWN)):
            fit = float((shape & op(s2)).sum() / max(1, (shape | op(s2)).sum())) - handicap
            like = max(correlation(grey, g2), correlation(grey, op(g2)))  # и обърната, и както е
            alike = max(correlation(fine, fine_picture(g2)), correlation(fine, fine_picture(op(g2))))
            print(f"{f.name} {flip}: outline {fit:.2f}, picture {like:.2f}, fine picture {alike:.2f}", flush=True)
            if fit >= MIRROR_IOU:
                fitting.append((fit, fit - PICTURE_WEIGHT * max(alike, 0.0), like, f.name, f, mask, flip))
    allowed = [x for x in fitting if textured or x[2] <= OTHER_PICTURE]
    if not allowed:
        return None
    # С особености по лицето другата страна вече е доказана: печели най-доброто очертание (ръка върху гърба го разваля; бялото
    # дистанционно: гръб с пръст 0,89, чист 0,95, а по фината картина печелеше този с пръста). Без тях помага и картината.
    best = max(allowed, key=lambda x: (x[0] if textured else x[1], x[3]))
    return best[4], best[5], best[6]


def face_normals(sess, crop: Image.Image, inside: np.ndarray) -> np.ndarray | None:
    """Посоките на повърхността на изрязан предмет от MoGe-2 (две изпълнения, изместени с половин стъпка), ако мрежата ги дава."""
    return RL.object_normals(sess, crop, inside, (0, RL.PATCH // 2)) if RL.gives_normals(sess) else None


def across_profile(normals: np.ndarray | None, inside: np.ndarray) -> np.ndarray | None:
    """Височината на лицето през ширината му (sideview.SECTION_POINTS места, в дялове от ширината) от посоките на повърхността в
    средната половина от дължината, без наклона на целия предмет и огледално осреднена. Предметът е изправен (дължината отвесно)."""
    if normals is None:
        return None
    h = inside.shape[0]
    rows = slice(int(0.25 * h), int(0.75 * h))
    slope = -normals[rows, :, 0] / np.maximum(np.abs(normals[rows, :, 2]), 0.2)
    ok = inside[rows]
    count = ok.sum(0)
    cols = np.flatnonzero(count >= max(5, 0.3 * count.max()))
    if len(cols) < 10:
        return None
    height = np.cumsum([np.median(slope[ok[:, c], c]) for c in cols]) / len(cols)
    u = (cols - cols[0] + 0.5) / (cols[-1] - cols[0] + 1)
    out = np.interp(np.linspace(SV.SECTION_EDGE, 1 - SV.SECTION_EDGE, SV.SECTION_POINTS), u, height)
    out = out - np.linspace(out[0], out[-1], len(out))
    return 0.5 * (out + out[::-1])


def outline_aspect(on: np.ndarray) -> float:
    """Ширината на изправен предмет (медиана през средата на дължината) спрямо дължината му."""
    rows = np.flatnonzero(on.any(1))
    if len(rows) < 4:
        return 1.0
    widths = on[rows].sum(1)
    middle = widths[len(widths) // 5 : len(widths) - len(widths) // 5]
    return float(np.median(middle if len(middle) else widths)) / len(rows)


def body_section(measured: dict, on: np.ndarray, thin: float, front: np.ndarray | None,
                 back: np.ndarray | None) -> tuple[tuple[np.ndarray, np.ndarray, float] | None, bool, str]:  # fmt: skip
    """Напречното сечение на плосък предмет: (по-плоският контур, заобленият, дебелината; всичко в дялове от ширината) или None, дали
    заобленото лице е отпред и откъде е сечението. Измереното от кадрите по оста (app/sideview.py) е за предпочитане; без него
    заобленото лице дава своята форма от посоките на повърхността. Кое лице е заобленото, казва MoGe-2 (front, back: across_profile)."""
    dome_f = float(front[len(front) // 2]) if front is not None else 0.0
    dome_b = float(back[len(back) // 2]) if back is not None else None
    if "section_flat" in measured:
        flat_d, dome_d = measured["section_flat"].astype(np.float64), measured["section_dome"].astype(np.float64)
        deep = float(dome_d[[0, -1]].mean() - dome_d.min())
        front_round = dome_f > dome_b if dome_b is not None else dome_f > 0.5 * deep
        return (flat_d, dome_d, float(measured["section_thick"])), front_round, f"{int(measured['section_frames'])} frames along the axis"
    thick = thin / max(outline_aspect(on), 1e-6)
    for profile, dome, front_round in ((front, dome_f, True), (back, dome_b, False)):
        other = dome_b if front_round else dome_f
        if profile is not None and dome is not None and dome >= DOMED and (other is None or dome > other):
            depth = np.clip(profile.max() - profile, 0.0, 0.85 * thick)
            return (np.zeros_like(depth), depth, thick), front_round, "the surface directions of the rounded side"
    return None, False, "none, a flat slab"


def section_body(on: np.ndarray, section: tuple[np.ndarray, np.ndarray, float] | None, along: np.ndarray | None,
                 thin: float, front_round: bool) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Колко зад най-предната равнина са лицето и гърбът (в дялове от дължината) във всяка точка на решетката `on` (изправен
    предмет): сечението през собствената ширина на всеки ред, а заобленото му лице е ограничено от дебелината по дължината `along`
    (sideview.side_thickness, горе първо). Без сечение: правоъгълник с дебелината thin."""
    n = on.shape[0]
    seen = on.any(1)
    left = np.where(seen, on.argmax(1), 0)
    right = np.where(seen, n - 1 - on[:, ::-1].argmax(1), 0)
    u = np.clip((np.arange(n)[None, :] - left[:, None] + 0.5) / np.maximum(right - left + 1, 1)[:, None], 0.0, 1.0)
    if section is None:
        flat = np.zeros((n, n))
        dome = np.full((n, n), thin)
    else:
        flat_d, dome_d, thick = section
        places = np.linspace(SV.SECTION_EDGE, 1 - SV.SECTION_EDGE, len(flat_d))
        flat = np.interp(u, *to_the_rim(places, flat_d, thick)) / thick * thin
        dome = (1 - np.interp(u, *to_the_rim(places, dome_d, thick)) / thick) * thin
    if along is not None and seen.any():
        rows = np.flatnonzero(seen)
        v = np.clip((np.arange(n) - rows[0] + 0.5) / (rows[-1] - rows[0] + 1), 0.0, 1.0)
        dome = np.minimum(dome, thin * np.interp(v, (np.arange(len(along)) + 0.5) / len(along), along)[:, None])
    dome = np.maximum(dome, flat + THINNEST_END)
    if front_round:
        return (thin - dome).astype(np.float32), (thin - flat).astype(np.float32)
    return flat.astype(np.float32), dome.astype(np.float32)


def to_the_rim(places: np.ndarray, depth: np.ndarray, thick: float) -> tuple[np.ndarray, np.ndarray]:
    """Контурът на сечението (места през ширината, колко навътре е) продължен до самия ръб (0 и 1) по наклона на крайните места."""
    step = max(1, len(places) // 16)
    first = depth[0] + (depth[0] - depth[step]) * places[0] / (places[step] - places[0])
    last = depth[-1] + (depth[-1] - depth[-1 - step]) * (1 - places[-1]) / (places[-1] - places[-1 - step])
    return np.r_[0.0, places, 1.0], np.clip(np.r_[first, depth, last], 0.0, thick)


def length_scale(on: np.ndarray, depth: float, focal: float) -> float:
    """Колко е дължината на предмета (решетката `on`) в единиците на сцената на разстояние depth. focal: в дялове от квадрата."""
    ys, xs = np.nonzero(on)
    x, y = xs - xs.mean(), ys - ys.mean()
    angle = 0.5 * np.arctan2(2 * float((x * y).mean()), float((x * x).mean() - (y * y).mean()))
    cells = float(np.ptp(x * np.cos(angle) + y * np.sin(angle)))
    return cells * depth / (focal * (on.shape[0] - 1))


def aligned_views(crop: Image.Image, part: np.ndarray, views: list[tuple[Image.Image, np.ndarray]]) -> list[tuple[int, Image.Image, np.ndarray, np.ndarray]]:
    """Другите изгледи от същата страна, които се налагат върху този по общите си особености: (брой общи, изглед, очертание, 3 x 3 от
    пикселите на crop към тези на изгледа), най-сигурните първо."""
    own = features(crop, part)
    found = []
    for view, inside in views:
        count, model = matched_features(own, features(view, inside))
        if count >= SAME_VIEW_FEATURES and model is not None:
            here, there = FEATURE_SIDE / max(crop.size), FEATURE_SIDE / max(view.size)
            to_view = np.diag([1 / there, 1 / there, 1.0]) @ model @ np.diag([here, here, 1.0])
            found.append((count, id(view), view, inside, to_view))
    found.sort(key=lambda x: x[:2], reverse=True)
    return [(c, v, i, m) for c, _, v, i, m in found]


def crop_sharpness(picture: Image.Image) -> float:
    """Рязкостта на изрязан предмет при еднакъв размер (дълга страна 640)."""
    k = 640 / max(picture.size)
    grey = np.asarray(picture.convert("L").resize((max(8, round(picture.width * k)), max(8, round(picture.height * k)))), np.float32)
    return float(ndimage.laplace(grey).var())


def covered_part(diff: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Къде един изглед е закрит (пръст, отблясък): разликата му от медианата на изгледите, без сянката си, е голяма. Сянката (светлината
    пада другояче) е бавната част от разликата, медиана в широк прозорец, която пръст не променя: без нея пръст върху бяла пластмаса
    на топла светлина се различава по-малко от сянката, а замененото излиза на петна. Връща (закритото, сянката)."""
    step = max(1, int(max(known.shape) / SHADE_POINTS))
    small = diff[::step, ::step]
    size = max(3, int(SHADE_WINDOW * max(known.shape) / step))
    shade_small = np.stack([ndimage.median_filter(small[..., c], size=size, mode="nearest") for c in range(3)], -1)
    shade = np.stack([np.asarray(Image.fromarray(shade_small[..., c].astype(np.float32)).resize((known.shape[1], known.shape[0]), Image.BILINEAR))
                      for c in range(3)], -1)  # fmt: skip
    covered = known & (np.abs(diff - shade).max(-1) > COVERED)
    return ndimage.binary_dilation(ndimage.binary_opening(covered, iterations=2), iterations=3) & known, shade


def without_fingers(photo: Image.Image, inside: np.ndarray, box: tuple[int, int, int, int],
                    views: list[tuple[Image.Image, np.ndarray]]) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int], int | None, set[int]]:  # fmt: skip
    """Пръсти само в избрания кадър: другите изгледи на същата страна, наложени по общите особености и със светлината на този кадър.
    Ръка, която държи предмета, минава през ръба му, затова очертанието на кадъра стърчи там, където другите не стърчат (бялото
    дистанционно: 1,0% от лицето в кадъра с пръст, 0,1% в чист). Замества се само цвят, който е различен от медианата на другите и е
    свързан с такова място: различната светлина в отделните кадри не е пръст. Връща (снимката с поправените цветове, очертанието без
    стърчащото, кутията, кой от views е по-добрият кадър отпред или None, кои от views са с ръка); при по-малко от два такива изгледа
    всичко остава. По-добър е изглед, чието очертание стърчи много по-малко, достатъчно рязък и голям."""
    from skimage.transform import ProjectiveTransform, warp

    x0, y0, x1, y1 = box
    crop, part = photo.crop(box), inside[y0:y1, x0:x1]
    rgb = np.asarray(crop, np.float32)
    colours, outlines, which = [rgb], [part], [-1]
    index = {id(v): k for k, (v, _) in enumerate(views)}
    for _, view, ins, to_view in aligned_views(crop, part, views)[:SAME_VIEWS]:
        move = ProjectiveTransform(to_view)
        there = warp(np.asarray(view, np.float64), move, output_shape=part.shape, order=1, cval=np.nan, preserve_range=True)
        on = warp(ins.astype(np.float64), move, output_shape=part.shape, order=1, cval=0.0, preserve_range=True) > 0.5
        both = part & on & np.isfinite(there[..., 0])
        if both.sum() < SHARED_PART * part.sum():
            continue
        there = there * (np.median(rgb[both], 0) / np.maximum(np.median(there[both], 0), 1.0))  # светлината на този кадър
        colours.append(there.astype(np.float32))
        outlines.append(on)
        which.append(index[id(view)])
    if len(colours) < 3:
        return photo, inside, box, None, set()
    shapes = np.stack(outlines).astype(np.float32)
    # Всеки изглед се сравнява с другите: с два закрити от четири (пръст тук, ръка в друг) общата медиана се цапа.
    outside = [outlines[j] & (np.delete(shapes, j, 0).mean(0) < 0.5) for j in range(len(outlines))]
    sticking = [float(o.sum()) / max(1.0, float(part.sum())) for o in outside]
    held = {k for s, k in zip(sticking[1:], which[1:]) if s > SWITCH_OUTSIDE}
    better = None
    if sticking[0] > SWITCH_OUTSIDE:
        sharp, size = crop_sharpness(crop), max(crop.size)
        found = [(s, k) for s, k in zip(sticking[1:], which[1:])
                 if s < SWITCH_RATIO * sticking[0] and crop_sharpness(views[k][0]) >= SHARP_ENOUGH * sharp and max(views[k][0].size) >= SIZE_ENOUGH * size]  # fmt: skip
        better = min(found)[1] if found else None
    with warnings.catch_warnings():  # места, които никой друг изглед не вижда
        warnings.simplefilter("ignore", RuntimeWarning)
        middle = np.nanmedian(np.stack(colours[1:]), 0)
    known = np.isfinite(middle[..., 0]) & part
    covered, shade = covered_part(np.where(known[..., None], rgb - np.nan_to_num(middle), 0.0), known)
    labels, count = ndimage.label(covered)
    near_edge = ndimage.binary_dilation(outside[0], iterations=FINGER_REACH)
    touching = np.unique(labels[near_edge & covered])
    covered = np.isin(labels, touching[touching > 0]) & (sticking[0] > SWITCH_OUTSIDE)  # без ръка през ръба няма какво да се заменя
    clean = photo.copy()
    clean.paste(Image.fromarray(np.clip(np.where(covered[..., None], np.nan_to_num(middle) + shade, rgb), 0, 255).astype(np.uint8)), (x0, y0))
    agreed = part & ~outside[0]  # издатина на очертанието само в този кадър (пръст отстрани) отпада
    kept = float(agreed.sum()) / max(1.0, float(part.sum()))
    print(f"same side in {len(colours) - 1} more frames: outline sticks out by {sticking[0]:.2%} of the face here "
          f"(others {', '.join(f'{s:.2%}' for s in sticking[1:])}), {covered.sum() / max(1, part.sum()):.1%} of the face covered", flush=True)  # fmt: skip
    if kept < KEEP_OUTLINE:
        return clean, inside, box, better, held
    new = inside.copy()
    new[y0:y1, x0:x1] = agreed
    new = straightened(RL.one_piece(new))
    ys, xs = np.nonzero(new)
    return clean, new, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), better, held


def flipped(a: np.ndarray, how: str) -> np.ndarray:
    return np.fliplr(a) if how == "lr" else np.flipud(a)


def build(frame: Path, m: np.ndarray, place: tuple[int, int, int], vol: np.ndarray, radius: float, level: float,
          distance: float, fov: float, colour_at: Callable[[np.ndarray], np.ndarray], work: Path, out: Path,
          others: list[tuple[Path, np.ndarray]] = (), strength: float = 1.0, thin: float | None = None,
          switched: bool = False) -> dict | None:  # fmt: skip
    """Записва модела в `out` и картите в `work`; връща описанието му или None, ако формата на TripoSR не пасва на
    снимката. place: (left, top, side), квадратът от кадъра, показан на TripoSR. colour_at: цветовете на TripoSR (0-1)
    за точки. switched: кадърът вече е сменен веднъж заради пръсти (плоският предмет взима от TripoSR само дълбочината и гърба,
    така че друг кадър отпред не пречи)."""
    if __package__:
        from . import mesh as M
    else:
        from app import mesh as M
    started = time.time()
    photo, inside, box = object_picture(frame, m)
    flat = thin is not None and thin <= FLAT_BODY
    if flat:
        photo, inside, box = upright(photo, inside)
        place = centred_place(box)
    # Другите кадри, изправени като този: за същата страна (детайл, пръсти) и за другата.
    views, view_frames = [], []
    all_views, all_frames = [], []
    colour_photo = photo
    if flat and others:
        angle = lengthwise(inside[box[1] : box[3], box[0] : box[2]])
        for f, mask in others:
            if f != frame and (mask > 127).sum() >= 50:
                p, ins, b = turned_like(*object_picture(f, mask)[:2], angle)
                views.append((p.crop(b), ins[b[1] : b[3], b[0] : b[2]]))
                view_frames.append((f, mask))
        colour_photo, inside, box, better, held = without_fingers(photo, inside, box, views)
        if better is not None and not switched:
            f2, m2 = view_frames[better]
            print(f"front frame: {f2.name} instead of {frame.name}, where the face is covered", flush=True)
            rest = [(frame, m)] + [(f, mask) for f, mask in others if f != f2]
            return build(f2, m2, place, vol, radius, level, distance, fov, colour_at, work, out, rest, strength, thin, switched=True)
        place = centred_place(box)
        all_views, all_frames = list(views), list(view_frames)  # и за гърба
        views = [v for k, v in enumerate(views) if k not in held]  # ръка в изгледа би влязла и в детайла
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
    if fits[azimuth] < MIN_IOU and not flat:  # плоският предмет взима от TripoSR само цветовете на гърба, ако няма кадър
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
    normals = face_normals(sess, crop, part) if flat else None
    rel = RL.relief_from_normals(normals, part, max(w0, h0)) if normals is not None else RL.image_relief(sess, crop, part, max(w0, h0), alone=True)
    print(f"AI detail, {time.time() - started:.0f} s", flush=True)
    if rel is not None and flat and views:
        rel = same_side_relief(rel, crop, part, views, lambda v, i: RL.image_relief(sess, v, i, max(v.size)))
        print(f"AI detail from the same side, {time.time() - started:.0f} s", flush=True)
    amount = RL.STRENGTH * strength * longest
    side_file, heights, measured = work / SV.FILE, None, {}
    if flat and side_file.exists():
        with np.load(side_file) as z:
            measured = {k: z[k] for k in z.files}
    if flat and rel is not None and "busy" in measured:
        busy, quiet = measured["busy"], measured["quiet"]
        fit, upside, shift, scale = fit_side(row_heights(flat_relief(rel, part, max(w0, h0)), part), busy)
        print(f"side view against the detail: fit {fit:.2f}, turned {upside}, moved {shift:+.3f}, stretched {scale:.3f}", flush=True)
        if fit >= SIDE_FIT:
            put = lambda p: SV.placed(p[::-1] if upside else p, shift, scale)  # noqa: E731
            heights = put(busy), put(quiet)

    # Другата страна, ако кадрите я показват: преди тялото, защото заобленото лице се познава по посоките на повърхността му.
    back_side = None
    found = opposite_frame(frame, list(others), m) if others else None
    if found is not None:
        f2, m2, how = found
        p2, ins2, b2 = turned_like(*object_picture(f2, m2)[:2], lengthwise(part))
        colour2 = p2
        if flat and all_views:
            # Ръка върху гърба (бялото дистанционно: пръст в кадъра на гърба): същото като отпред, с другите кадри на тази страна.
            rest = [k for k, (f, _) in enumerate(all_frames) if f != f2]
            colour2, ins2, b2, better2, _ = without_fingers(p2, ins2, b2, [all_views[k] for k in rest])
            if better2 is not None:
                f3, m3 = all_frames[rest[better2]]
                again = opposite_frame(frame, [(f3, m3)], m)
                if again is not None:
                    f2, m2, how = again
                    print(f"back frame: {f2.name}, where the hand does not cover it", flush=True)
                    p2, ins2, b2 = turned_like(*object_picture(f2, m2)[:2], lengthwise(part))
                    rest = [k for k, (f, _) in enumerate(all_frames) if f != f2]
                    colour2, ins2, b2, _, _ = without_fingers(p2, ins2, b2, [all_views[k] for k in rest])
        crop2, part2 = p2.crop(b2), ins2[b2[1] : b2[3], b2[0] : b2[2]]
        normals2 = face_normals(sess, crop2, part2) if flat else None
        rel2 = (RL.relief_from_normals(normals2, part2, max(crop2.size)) if normals2 is not None
                else RL.image_relief(sess, crop2, part2, max(crop2.size), alone=True))  # fmt: skip
        back_side = (f2, how, colour2, ins2, crop2, part2, rel2, normals2, b2)

    shaped = False
    if flat:
        middle = np.nanmedian(resized((near + far) / 2, n)[on])
        depth = float(middle if np.isfinite(middle) else distance)
        scale = length_scale(on, depth, cam.focal)
        front_dome = across_profile(normals, part)
        back_dome = across_profile(back_side[7], back_side[5]) if back_side is not None else None
        section, front_round, made = body_section(measured, on, thin, front_dome, back_dome)
        along = measured.get("side_thickness")
        z_front, z_back = section_body(on, section, along, thin, front_round)
        front_body, back_body = depth + (z_front - 0.5 * thin) * scale, depth + (z_back - 0.5 * thin) * scale
        shaped = section is not None
        print(f"flat body, measured thickness {thin:.3f} of the length; cross-section: {made}"
              + (f", rounded side {'front' if front_round else 'back'}" if shaped else "")
              + ("" if along is None else f", thickness at the ends {float(along[:8].mean()):.2f} and {float(along[-8:].mean()):.2f}"), flush=True)  # fmt: skip
    else:
        front_body, back_body = RL.masked_blur(resized(near, n), on, SHAPE_BLUR * width), RL.masked_blur(resized(far, n), on, BACK_BLUR * width)
    front_t = front_body
    if rel is not None and flat:
        rel = flat_relief(rel, part, max(w0, h0))
        if heights is not None:  # измереното тяло е без копчетата: те стърчат над него, както в кадъра отстрани
            rel = with_side_heights(rel, part, heights[0])
            print(f"detail with the measured heights: highest {float(row_heights(rel, part).max()):.4f} of the length", flush=True)
        else:
            front_t = front_t + raised_part(rel, part) * amount
    if rel is not None:
        front_t = front_t - to_square(rel, n) * amount
    back_t = back_body

    back_chart, opposite = None, None
    if back_side is not None:
        f2, how, _, _, crop2, part2, rel2, _, _ = back_side
        if rel2 is not None and flat:
            rel2 = flat_relief(rel2, part2, max(crop2.size))
            quiet = FLAT_BACK_QUIET / RL.STRENGTH
            rel2 = np.sign(rel2) * np.maximum(np.abs(rel2) - quiet, 0)
            if heights is not None:  # другото лице: страничният изглед показва колко е равно
                rel2 = flipped(with_side_heights(flipped(rel2, how), flipped(part2, how), heights[1], 1.0), how)
            else:
                back_t = back_t - raised_part(rel2, part2) * amount
        if rel2 is not None:
            lay = np.asarray(Image.fromarray(flipped(rel2, how)).resize((w0, h0), Image.BILINEAR))
            back_t = back_t + to_square(lay, n) * amount
        opposite = f2.name
        print(f"the other side from {f2.name} ({how}), {time.time() - started:.0f} s", flush=True)
    back_t = np.maximum(back_t, front_t + MIN_THICK * longest)

    # Мрежата.
    s, t = grid(n)
    q = no_pinches(on[:-1, :-1] & on[1:, :-1] & on[:-1, 1:] & on[1:, 1:])
    labels, count = ndimage.label(q)
    if count > 1:
        q = labels == 1 + int(np.argmax(ndimage.sum_labels(q, labels, np.arange(1, count + 1))))
    exact = None
    if flat:
        s, t, exact = rim_on_lines(s, t, on, q, 2 * FLAT_EDGE * width / 0.85)
    if not shaped:  # измереното сечение вече има истинските си заоблени ръбове
        front_t, back_t = rounded_edges(front_t, back_t, on, longest / width, (FLAT_EDGE if flat else ROUND_EDGE) * longest, wall=flat, exact=exact)
    rays = cam.rays(s, t)
    front = cam.center + rays * front_t[..., None]
    back = cam.center + rays * back_t[..., None]
    vertices, faces, used = closed_mesh(q, front, back)

    # Цветовете: отпред снимката (без пръстите), отзад другият кадър или цветовете на TripoSR, приближени до снимката; всички с
    # поправената светлина (texture.light_fix).
    neutral = TX.neutral_object([(frame, m)] + [(f, mk) for f, mk in others if (mk > 127).sum() >= 50])
    gain = TX.white_gain(frame.parent, [frame.name], {frame.name: m})
    fix = TX.light_fix(np.asarray(colour_photo), inside, neutral, None if neutral else gain)
    print("light: " + ("white, grey or black object, the colour of the light removed" if neutral
                       else f"white balance: red, green, blue x {gain.round(2).tolist()}"), flush=True)  # fmt: skip
    square_rgb = lambda k: fix(np.asarray(colour_photo.crop((left, top, left + side, top + side)).resize((k, k), Image.BILINEAR)))  # noqa: E731
    rgb = square_rgb(n).astype(np.float32)
    face = filled_colours(fix(np.asarray(colour_photo.crop(box))), part)
    if back_side is not None:
        _, how, colour2, ins2, _, part2, _, _, b2 = back_side  # colour2: кадърът на гърба без ръката
        fix2 = TX.light_fix(np.asarray(colour2), ins2, neutral, None if neutral else gain)
        back_chart = np.asarray(Image.fromarray(flipped(filled_colours(fix2(np.asarray(colour2.crop(b2))), part2), how)).resize((w0, h0), Image.LANCZOS))
    if back_chart is None:
        back_points = cam.center + cam.rays(*grid(RAYS)) * resized(back_t, RAYS)[..., None]
        ai = (np.clip(colour_at(back_points.reshape(-1, 3).astype(np.float32)), 0, 1) * 255).reshape(RAYS, RAYS, 3).astype(np.float32)
        small_on = small_mask(RAYS)
        real = square_rgb(RAYS).astype(np.float32)
        if small_on.sum() > 50:
            ai = np.clip(ai + 0.7 * (np.median(real[small_on], 0) - np.median(ai[small_on], 0)), 0, 255)
        k = (RAYS - 1) / side
        ax0, ay0 = int(round((x0 - left) * k)), int(round((y0 - top) * k))
        ax1, ay1 = max(ax0 + 1, int(round((x1 - left) * k))), max(ay0 + 1, int(round((y1 - top) * k)))
        part_ai = filled_colours(ai.astype(np.uint8)[ay0:ay1, ax0:ax1], small_on[ay0:ay1, ax0:ax1])
        back_chart = np.asarray(Image.fromarray(part_ai).resize((w0, h0), Image.BICUBIC))
    back_big = np.stack([to_square(back_chart[..., c], n) for c in range(3)], -1)  # само за цвят на върховете
    rows = used // n, used % n
    colors = np.vstack([rgb[rows], back_big[rows]]).astype(np.uint8)
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
    atlas.paste(Image.fromarray(face).resize((cw, ch), Image.LANCZOS), (0, 0))
    atlas.paste(Image.fromarray(back_chart).resize((cw, ch), Image.LANCZOS), (cw, 0) if sideways else (0, ch))
    atlas.save(work / ATLAS, quality=93)
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
