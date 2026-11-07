"""Повърхността на предмета от картите на дълбочината на кадрите, слети в обем, вместо реконструкция на Поасон върху
гъстите точки. Картите на OpenMVS са шумни (раирани, няколко десети от процента от разстоянието) и празни по тъмни или
еднообразни части (черната дръжка на дрелка). Гъстите точки наследяват и двете: Поасон върху тях даваше кора от
неравности, а в празнините формата от очертанията (на бучки). AI дълбочината на кадъра (app/depth.py) е гладка и
пълна, но с грешен мащаб и леко огъната. Напасната към собствената карта на кадъра (мащаб и отместване в обратна
дълбочина, после гладко корекционно поле от остатъка), тя запазва измерената форма с чистата повърхност на мрежата и
запълва празнините. След това всеки кадър гласува къде е повърхността, като пресечено разстояние със знак, усреднено в
мрежа от воксели (TSDF), а обвивката от очертанията запълва онова, което никой кадър не е видял (долната страна). При
дрелка: плоски панели, стъпаловидни ръбове и кръгла дръжка вместо бучки. Маската на кадър се брои само където
обвивката е съгласна (консенсусът на маските), така че фон, включен от една маска, остава извън, а всяка измерена
дълбочина извън предмета (маса, стена) все така казва, че пространството пред нея е празно, което изрязва онова, което
очертанията не могат (масата под ръба на предмета, процепа между дръжка и батерия). Работи като собствен процес
(стъпка "fusing depth"): python fuse.py <папка dense> <inputs.npz> <crops.npz> <out.ply>. Папка dense: картите на
дълбочината на OpenMVS (depth*.dmap). crops.npz: изрязванията с дълбочина от app/depth.py. inputs.npz: names (n,)
имена на кадрите (изрязване i е кадър i), to_up (3,3) свят -> координати на мрежата, ground_z (nan: няма маса),
hull_v, hull_f (обвивката от очертанията, неиздълбана, в координати на мрежата). Записва out.ply (координати на
мрежата) и out.json (колко кадъра от колко са използвани)."""

import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from scipy import ndimage
from skimage import measure

if __package__:
    from . import depth as D
    from . import parallel as PAR
else:  # изпълнение като скрипт: импортира пакета по пътя му
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import depth as D
    from app import parallel as PAR

# От размера на предмета: клетката на мрежата. 0,25% (400 клетки по него) пази детайла на отпечатък от 10 см до около
# 0,5 мм след размазването по-долу; по-финото само добавяше шум при дрелка.
VOXEL = 0.0025
MAX_VOXELS = 60_000_000  # около 1,5 GB при пика; по-големите мрежи получават по-груби клетки
TRUNC = 12  # клетки: разстоянието със знак се пресича на това разстояние от повърхността (шумът на дълбочината е няколко клетки)
MIN_CONF = 0.65  # увереността на OpenMVS, под която измерена дълбочина не се брои (медианата при дрелка: 0,8)
FIELD = 0.05  # от дългата страна на изрязването: колко гладка е корекцията на AI дълбочината (при дрелка: 10 пиксела)
AI_WEIGHT = 1.2  # глас, където само AI е видял повърхността (измерени пиксели: 1)
FREE_WEIGHT = 0.3  # глас за празно пространство пред повърхността: издълбава, но един грешен кадър не може да пробие дупка
AI_FREE = 0.3  # и от него, където само AI е видял повърхността (без това дръжката на дрелка се разкъсваше)
HULL_WEIGHT = 0.3  # думата на обвивката от очертанията, част от тази на един кадър
OUTSIDE = 0.02  # от размера: отвъд това извън обвивката нищо не се слива (масата и стаята)
FLOOR_BAND = 0.06  # от размера: колко високо над масата може да стигне лист от нея (меката възглавница се издува)
FLAP = 0.0125  # от размера: по-тънките от двойно това листове там са маса, а не предмет (вижте floor_sheet)
HULL_POINTS = 1_000_000  # точки вътре в обвивката, нарисувани във всеки кадър за проверка на маската му (вижте hull_view)
BLUR = 3.5  # клетки: осредненият обем се заглажда преди извличане на повърхността (без това: трохи и ръбове)
FACES = 200_000
# Думата на кадър за повърхност намалява с това колко косо я вижда (косинусът между лъча му и нормалата на
# повърхността, никога под FACING_MIN): при плъзгащ ъгъл пиксел покрива дълга ивица повърхност и грешката на
# дълбочината му лежи по повърхността. 0: изключено.
FACING_MIN = 0.15
# Втори проход пропуска гласове за повърхност на кадър, които се разминават със средното от първия с повече от това
# (от разстоянието на пресичане): иначе един кадър с грешно напасната AI дълбочина (тъмна пластмаса, малко измерени
# пиксели) оставя ивица или бучка. 0: един проход. При дрелка (80 кадъра) с FACING_MIN: 19% от гласовете отпадат,
# ивиците по черната дръжка почти изчезват, средното огъване между съседни стени 3,4° -> 3,1°, а гъстите точки остават
# толкова близо до повърхността (медиана 0,23% от размера). По-голямото заглаждане на корекцията (FIELD 0,03-0,05)
# също успокояваше дръжката, но изкривяваше тялото (точките на 0,36-0,47%): не е направено.
OUTLIER = 0.1
PASSES = 3  # проходи за гласуване, всеки пропуска несъгласуваното с предишния (1: нищо не се пропуска)
BLOCK = 500_000  # клетки, върху които една нишка работи наведнъж (вижте vote_block)


def note(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


@dataclass
class DepthMap:
    image: str  # името на файла на кадъра
    image_size: tuple[int, int]  # (w, h)
    K: np.ndarray  # за разделителната способност на картата на дълбочината
    R: np.ndarray  # свят -> камера
    C: np.ndarray  # център на камерата
    depth: np.ndarray  # (h, w), 0 = няма
    conf: np.ndarray | None


HEADER = "<HBBIIIIff"  # име 'DR', флагове, допълване, изображение w, h, дълбочина w, h, диапазон на дълбочината
HAS_NORMAL, HAS_CONF, HAS_VIEWS = 2, 4, 8


def read_dmap(path: Path) -> DepthMap:
    """Суровата карта на дълбочината на OpenMVS (DepthMap.cpp, ExportDepthDataRaw)."""
    b = path.read_bytes()
    magic, flags, _, iw, ih, dw, dh, _, _ = struct.unpack_from(HEADER, b, 0)
    if magic != struct.unpack("<H", b"DR")[0]:
        raise ValueError(f"{path.name}: not a depth map")
    o = struct.calcsize(HEADER)
    (n,) = struct.unpack_from("<H", b, o)
    image = Path(b[o + 2 : o + 2 + n].decode("utf-8", "replace")).name
    o += 2 + n
    (nids,) = struct.unpack_from("<I", b, o)
    o += 4 + 4 * nids
    K, R, C = (np.frombuffer(b, np.float64, k, o + off) for k, off in ((9, 0), (9, 72), (3, 144)))
    o += 168
    area = dw * dh
    depth = np.frombuffer(b, np.float32, area, o).reshape(dh, dw)
    o += 4 * area
    if flags & HAS_NORMAL:
        o += 12 * area
    conf = np.frombuffer(b, np.float32, area, o).reshape(dh, dw) if flags & HAS_CONF else None
    o += 4 * area if flags & HAS_CONF else 0
    o += 4 * area if flags & HAS_VIEWS else 0
    if o != len(b):
        raise ValueError(f"{path.name}: unexpected layout")
    return DepthMap(image, (iw, ih), K.reshape(3, 3), R.reshape(3, 3), C, depth, conf)


def at_image_size(dm: DepthMap) -> DepthMap:
    """Картата на дълбочината при разделителната способност на кадъра (пикселите на изрязванията), ако OpenMVS е
    работил при по-малка."""
    h, w = dm.depth.shape
    iw, ih = dm.image_size
    if (w, h) == (iw, ih):
        return dm
    rows = np.minimum((np.arange(ih) * h / ih).astype(int), h - 1)
    cols = np.minimum((np.arange(iw) * w / iw).astype(int), w - 1)
    K = dm.K.copy()
    K[0] *= iw / w
    K[1] *= ih / h
    conf = dm.conf[np.ix_(rows, cols)] if dm.conf is not None else None
    return DepthMap(dm.image, dm.image_size, K, dm.R, dm.C, dm.depth[np.ix_(rows, cols)], conf)


def completed(dm: DepthMap, disp: np.ndarray, box, inner: np.ndarray, allowed: np.ndarray, core: np.ndarray):
    """Дълбочината на кадъра върху изрязването му: AI дълбочината, напасната към измерената, върху пикселите на
    предмета: вътрешността на маската и на обвивката (`core`: където маската е пропуснала тъмна част, черен спусък
    на дрелка), но само където обвивката се вижда (`allowed`). Връща (depth, weight, free), 0 където няма нищо, или
    None, ако двете не пасват. free: измерената дълбочина на другите пиксели, която показва само празно
    пространство (пред пропуснат пиксел от предмета пак е празно)."""
    x0, y0, x1, y1 = box
    dep = dm.depth[y0:y1, x0:x1]
    if dep.shape != disp.shape:
        return None
    conf = dm.conf[y0:y1, x0:x1] if dm.conf is not None else np.ones_like(dep)
    measured = (dep > 0) & (conf >= MIN_CONF)
    inner = (inner | core) & allowed
    on = inner & measured
    if on.sum() < 500:
        return None
    inv = np.where(dep > 0, 1 / np.maximum(dep, 1e-12), 0)
    (a, b), err = D.fit_scale(disp[on], inv[on], np.ones(int(on.sum())))
    if a <= 0 or err > D.MAX_ERROR:
        return None
    base = a * disp + b
    # Каквото оставят мащабът и отместването, заглаждано в поле (мрежата леко огъва повърхностите); силно отклонените
    # отпадат.
    r = np.where(on, inv - base, 0)
    med = np.median(r[on])
    mad = 1.4826 * np.median(np.abs(r[on] - med))
    good = on & (np.abs(r - med) <= 4 * mad) if mad > 0 else on
    sigma = max(3.0, FIELD * max(disp.shape))
    num = ndimage.gaussian_filter(np.where(good, r, 0).astype(np.float32), sigma)
    den = ndimage.gaussian_filter(good.astype(np.float32), sigma)
    full = base + np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    ok = inner & (full > 0)
    depth = np.where(ok, 1 / np.maximum(full, 1e-12), 0).astype(np.float32)
    weight = np.where(on, 1.0, AI_WEIGHT).astype(np.float32) * ok
    if FACING_MIN > 0:
        weight *= np.clip(facing(depth, dm.K, box), FACING_MIN, 1.0)
    free = np.where(measured & ~ok, dep, 0).astype(np.float32)
    return depth, weight, free


def facing(depth: np.ndarray, K: np.ndarray, box) -> np.ndarray:
    """За всеки пиксел от дълбочината на изрязване: косинусът между лъча на камерата и нормалата на повърхността (1:
    право). Пиксели без дълбочина или до такъв получават 1 (така или иначе не гласуват за повърхност)."""
    x0, y0, x1, y1 = box
    z = depth.astype(np.float32)
    u = (np.arange(x0, x1, dtype=np.float32) - np.float32(K[0, 2])) / np.float32(K[0, 0])
    v = (np.arange(y0, y1, dtype=np.float32) - np.float32(K[1, 2])) / np.float32(K[1, 1])
    P = np.stack([u[None, :] * z, v[:, None] * z, z], -1)
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    ray = np.stack([np.broadcast_to(u[None, :], z.shape), np.broadcast_to(v[:, None], z.shape), np.ones_like(z)], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.abs((n * ray).sum(-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(ray, axis=-1))
    holes = ndimage.minimum_filter(z, size=3) <= 0
    return np.where(holes | ~np.isfinite(cos), 1.0, cos).astype(np.float32)


def hull_view(points: np.ndarray, spacing: float, dm: DepthMap, box) -> tuple[np.ndarray, np.ndarray]:
    """Къде се вижда обвивката в изрязването (точките от вътрешността ѝ, световни, всяка нарисувана широка колкото
    разстоянието между тях) и ядрото ѝ далеч от очертанието (толкова навътре, колкото app/depth.py пази вътрешността
    на маските: дълбочината се размазва при очертанието). Обвивката е това, за което маските на повечето кадри са
    съгласни, така че парче фон, включено от маската на един кадър (стол зад дръжката на дрелка, ивица маса, плакат),
    е извън нея, а част, пропусната от една маска, е вътре."""
    x0, y0, x1, y1 = box
    out = np.zeros((y1 - y0, x1 - x0), bool)
    x = PAR.turned(points - dm.C, dm.R)
    z = x[:, 2]
    ok = z > 1e-9
    if not ok.any():
        return out, out
    u = np.round(dm.K[0, 0] * x[ok, 0] / z[ok] + dm.K[0, 2]).astype(np.int64) - x0
    v = np.round(dm.K[1, 1] * x[ok, 1] / z[ok] + dm.K[1, 2]).astype(np.int64) - y0
    inb = (u >= 0) & (v >= 0) & (u < x1 - x0) & (v < y1 - y0)
    out[v[inb], u[inb]] = True
    r = int(np.ceil(spacing * dm.K[0, 0] / np.median(z[ok]))) + 1
    shown = ndimage.maximum_filter(out, size=2 * r + 1)
    # Изрязването е кутията на предмета, разширена с 20% от всяка страна (app/depth.py predict).
    edge = max(2, int(0.02 * max(out.shape) / 1.4))
    return shown, ndimage.binary_erosion(shown, iterations=r + edge)


def hull_distance(hull: trimesh.Trimesh, lo: np.ndarray, shape, step: float) -> np.ndarray:
    """Разстояние със знак (световни единици) от клетките на мрежата до обвивката: отрицателно вътре."""
    vg = hull.voxelized(step).fill()
    inside = np.zeros(shape, bool)
    off = np.round((vg.transform[:3, 3] - lo) / step).astype(int)
    m = vg.matrix[: shape[0] - off[0], : shape[1] - off[1], : shape[2] - off[2]]
    inside[off[0] : off[0] + m.shape[0], off[1] : off[1] + m.shape[1], off[2] : off[2] + m.shape[2]] = m
    if not inside.any():
        raise SystemExit("the visual hull encloses nothing")
    away, within = PAR.each(lambda a: ndimage.distance_transform_edt(a).astype(np.float32), [~inside, inside])
    return (away - within) * step


def floor_sheet(solid: np.ndarray, top: int, radius: float) -> np.ndarray | None:
    """Клетките на тънък лист върху масата, стърчащ от предмета (първите му `top` слоя), или None. Червена кутия върху
    червена възглавница включваше парче от възглавницата: повърхност като на кутията, вътре във всяка маска, но
    по-тънка от всичко по кутията. Тънко = изчезва след морфологично отваряне (кълбо с радиус `radius` клетки);
    стърчащо = на повече от `radius` клетки от оставащото (отварянето закръгля и ръбовете на предмета, но по-малко).
    Не важи, когато по-голямата част от самия предмет е толкова тънка (плосък предмет)."""
    r = max(2.0, radius)
    sub = solid[:, :, : top + int(np.ceil(2 * r))]
    if top <= 0 or not sub.any():
        return None
    core = ndimage.distance_transform_edt(sub) > r
    if not core.any():
        return None
    opened = ndimage.distance_transform_edt(~core) <= r
    if opened.sum() < 0.5 * sub.sum():
        return None
    sheet = sub & ~opened & (ndimage.distance_transform_edt(~opened) > r + 1)
    sheet[:, :, top:] = False
    return sheet if sheet.any() else None


def vote_block(frames: list, world: np.ndarray, cells: np.ndarray, tsdf: np.ndarray, weight: np.ndarray,
               prev: np.ndarray | None, trunc: float) -> int:  # fmt: skip
    """Гласовете на всеки кадър за един блок от клетките на обема (world: центровете им; cells: местата им в tsdf и
    weight; два блока нямат обща клетка, затова вървят едновременно, а средното не зависи от реда на гласовете). С
    `prev` (предишния проход) гласовете за повърхност далеч от него отпадат. Връща колко."""
    dropped = 0
    for (R, C, K), box, depth, wmap, fmap in frames:
        x0, y0, x1, y1 = box
        x = PAR.turned(world - C, R)
        zc = x[:, 2]
        ok = zc > 1e-9
        u = np.full(len(x), -1, np.int64)
        v = np.full(len(x), -1, np.int64)
        u[ok] = np.round(K[0, 0] * x[ok, 0] / zc[ok] + K[0, 2]).astype(np.int64) - x0
        v[ok] = np.round(K[1, 1] * x[ok, 1] / zc[ok] + K[1, 2]).astype(np.int64) - y0
        ok &= (u >= 0) & (v >= 0) & (u < x1 - x0) & (v < y1 - y0)
        idx, zc, u, v = np.flatnonzero(ok), zc[ok], u[ok], v[ok]
        surface = depth[v, u] > 0
        dep = np.where(surface, depth[v, u], fmap[v, u])
        sd = dep - zc
        # Пикселите на предмета гласуват близо до повърхността си и за празно пространство отпред; другите пиксели
        # само за второто.
        ok = (dep > 0) & (sd >= np.where(surface, -trunc, trunc))
        idx, sd, surface, w = idx[ok], sd[ok], surface[ok], wmap[v[ok], u[ok]]
        val = np.minimum(1.0, sd / trunc).astype(np.float32)
        cell = cells[idx]
        if prev is not None:
            keep = ~surface | (val >= 1.0) | (np.abs(val - prev[cell]) <= OUTLIER)
            dropped += int((~keep).sum())
            cell, val, surface, w = cell[keep], val[keep], surface[keep], w[keep]
        free = val >= 1.0
        w = np.where(surface, w, 1.0)
        w = np.where(free, w * FREE_WEIGHT * np.where(w < 1.0, AI_FREE, 1.0), w).astype(np.float32)
        tsdf[cell] = (tsdf[cell] * weight[cell] + val * w) / (weight[cell] + w)
        weight[cell] += w
    return dropped


def fuse(dense: Path, inputs: Path, crops_file: Path, out: Path) -> None:
    z = np.load(inputs)
    names = [str(n) for n in z["names"]]
    to_up = z["to_up"].astype(np.float64)
    ground_z = float(z["ground_z"])
    hull = trimesh.Trimesh(z["hull_v"], z["hull_f"], process=False)
    size = float(hull.extents.max())
    step = (float(z["voxel"]) if "voxel" in z.files else VOXEL) * size  # експресни сканирания: по-груби клетки
    pad = max(OUTSIDE * size, TRUNC * step) + 3 * step
    lo = hull.bounds[0] - pad
    shape = np.ceil((hull.bounds[1] + pad - lo) / step).astype(int) + 1
    if np.prod(shape) > MAX_VOXELS:
        grow = (np.prod(shape) / MAX_VOXELS) ** (1 / 3)
        step *= grow
        shape = np.ceil(shape / grow).astype(int) + 1
    trunc = TRUNC * step
    note(f"grid {shape[0]}x{shape[1]}x{shape[2]}, cell {step / size:.2%} of the size")

    sdf = hull_distance(hull, lo, tuple(shape), step).ravel()
    tsdf = np.clip(sdf / trunc, -1, 1)
    weight = np.full(tsdf.shape, HULL_WEIGHT, np.float32)
    active = np.flatnonzero(sdf <= OUTSIDE * size)
    inside = np.flatnonzero(sdf < 0)
    del sdf
    pick = np.random.default_rng(0).choice(len(inside), min(len(inside), HULL_POINTS), replace=False)
    spacing = step * (len(inside) / len(pick)) ** (1 / 3)
    cells = np.stack(np.unravel_index(inside[pick], tuple(shape)), 1)
    hull_points = (cells * step + lo) @ to_up
    del inside, pick
    cells = np.stack(np.unravel_index(active, tuple(shape)), 1).astype(np.float32)
    world = (cells * np.float32(step) + lo.astype(np.float32)) @ to_up.astype(np.float32)
    del cells
    note(f"{len(active) / 1e6:.1f} million cells near the hull")

    maps = {}
    for p in dense.glob("depth*.dmap"):
        with open(p, "rb") as f:
            head = f.read(struct.calcsize(HEADER) + 2 + 1024)
        o = struct.calcsize(HEADER)
        (n,) = struct.unpack_from("<H", head, o)
        maps[Path(head[o + 2 : o + 2 + n].decode("utf-8", "replace")).name] = p
    with np.load(crops_file) as crops:
        todo = [(name, crops[f"disp_{i}"], tuple(int(v) for v in crops[f"box_{i}"]), crops[f"inner_{i}"])
                for i, name in enumerate(names) if f"disp_{i}" in crops and name in maps]  # fmt: skip

    def prepare(item):
        name, disp, box, inner = item
        dm = at_image_size(read_dmap(maps[name]))
        allowed, core = hull_view(hull_points, spacing, dm, box)
        got = completed(dm, disp, box, inner, allowed, core)
        if got is None:
            note(f"{name}: AI depth does not fit the depth map, left out")
            return None
        return ((dm.R.astype(np.float32), dm.C.astype(np.float32), dm.K), box, *got)

    # За всеки използваем кадър: (поза, кутия, дълбочина, тегло, празно), запазени за следващите проходи.
    frames = [f for f in PAR.each(prepare, todo) if f is not None]
    used = len(frames)
    note(f"{used} of {len(names)} frames have depth to fuse")
    if used < 3:
        raise SystemExit(f"only {used} frames' depth could be used")

    def vote(tsdf: np.ndarray, weight: np.ndarray, prev: np.ndarray | None) -> int:
        starts = range(0, len(active), BLOCK)
        return sum(PAR.each(lambda s: vote_block(frames, world[s : s + BLOCK], active[s : s + BLOCK], tsdf, weight, prev,
                                                 trunc), starts))  # fmt: skip

    passes = (int(z["passes"]) if "passes" in z.files else PASSES) if OUTLIER > 0 else 1
    empty_tsdf, empty_weight, prev = tsdf, weight, None
    for k in range(passes):
        tsdf, weight = empty_tsdf.copy(), empty_weight.copy()
        dropped = vote(tsdf, weight, prev)
        prev = tsdf
        note(f"pass {k + 1}: {dropped / 1e6:.1f} million votes left out as far from the others")
    del empty_tsdf, empty_weight, prev
    note(f"{used} of {len(names)} frames fused")

    vol = tsdf.reshape(tuple(shape))
    wv = np.minimum(weight, 50).reshape(tuple(shape))
    vol = ndimage.gaussian_filter(vol * wv, BLUR) / np.maximum(ndimage.gaussian_filter(wv, BLUR), 1e-6)
    if np.isfinite(ground_z):  # нищо под масата: плоско затворено дъно
        zs = lo[2] + np.arange(shape[2]) * step
        vol[:, :, zs < ground_z] = 1.0
        flap = float(z["flap"]) if "flap" in z.files else FLAP
        sheet = floor_sheet(vol < 0, int(np.searchsorted(zs, ground_z + FLOOR_BAND * size)), flap * size / step)
        if sheet is not None:
            vol[:, :, : sheet.shape[2]][sheet] = 1.0
            note(f"thin sheet on the table cut off: {sheet.sum() * step**3 / size**3:.2%} of the size cubed")
    vol[[0, -1]] = vol[:, [0, -1]] = vol[:, :, [0, -1]] = 1.0  # затворено при границата на мрежата
    verts, faces, _, _ = measure.marching_cubes(vol, 0.0)
    mesh = trimesh.Trimesh(verts * step + lo, faces)
    mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
    if mesh.volume < 0:
        mesh.invert()
    note(f"surface: {len(mesh.faces)} triangles")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.apply_coord_taubin_smoothing(stepsmoothnum=5)  # стъпалата на мрежата
    if ms.current_mesh().face_number() > FACES:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=FACES, preservenormal=True, preservetopology=True)
    m = ms.current_mesh()
    trimesh.Trimesh(m.vertex_matrix(), m.face_matrix()).export(out)
    out.with_suffix(".json").write_text(json.dumps({"frames": used, "total": len(names)}))


if __name__ == "__main__":
    fuse(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
