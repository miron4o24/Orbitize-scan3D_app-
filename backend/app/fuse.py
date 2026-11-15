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
from PIL import Image, ImageDraw
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
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
TRUNC = 6  # клетки: разстоянието със знак се пресича на това разстояние от повърхността (шумът на дълбочината е няколко клетки)
MIN_CONF = 0.5  # увереността на OpenMVS, под която измерена дълбочина не се брои (медианата при дрелка: 0,8)
FIELD = 0.01  # от дългата страна на изрязването: колко гладка е корекцията на AI дълбочината (при дрелка: 10 пиксела)
AI_WEIGHT = 0.5  # глас, където само AI е видял повърхността (измерени пиксели: 1)
FREE_WEIGHT = 0.3  # глас за празно пространство пред повърхността: издълбава, но един грешен кадър не може да пробие дупка
AI_FREE = 0.3  # и от него, където само AI е видял повърхността (без това дръжката на дрелка се разкъсваше)
HULL_WEIGHT = 0.3  # думата на обвивката от очертанията, част от тази на един кадър
# Където очертанието на кадър минава по обвивката, а погледът е по повърхността ѝ (вижте rim_support), предметът
# наистина я докосва: там думата ѝ е толкова кадъра (при RIM_FULL такива кадъра). AI дълбочината прави печат и
# боядисани петна релефни (лого по метална колонка, очите на фигурка - вдлъбнатини), а лъскав метал дава шумни карти
# на OpenMVS; обвивката там е вярна (колонка: страната е докосната от очертанията на 24 кадъра наоколо).
HULL_RIM = 30.0
# Дълбочината на кадър се брои изцяло само където поне AGREE_VIEWS от съседните кадри виждат повърхността на същото
# място (до AGREE_NEAR от пресичането; вижте agreeing); другаде гласът ѝ е UNSURE от обичайния. Лъскав метал и
# гланцова боя дават различна дълбочина от всеки кадър (отраженията се местят): там решава обвивката (HULL_RIM), а
# където кадрите са съгласни (коса, гънки, ръце) - дълбочината, и детайлът остава.
AGREE_VIEWS = 3
AGREE_NEAR = 0.5
AGREE_ASK = 6  # съседни кадри, които се питат (най-близките по посока, поне AGREE_APART градуса встрани)
AGREE_APART = 4.0
UNSURE = 0.15
# Всичко това е само за предмет, на чиято дълбочина като цяло не може да се вярва: кадрите са съгласни за под
# AGREE_SHINY от повърхността му (изцяло при AGREE_RAMP по-малко). Метална колонка: 14-18%; фигурка от матова
# пластмаса: 67%; плетена тиква: 91%. При фигурката съгласието пада и там, където малко кадри виждат (между краката,
# под брадичката), а там обвивката не е предметът: наложена, тя сливаше краката и заобляше тялото.
AGREE_SHINY = 0.5
AGREE_RAMP = 0.2
RIM_FULL = 3
RIM_ACROSS = 0.2  # синус на ъгъла между погледа и повърхността на обвивката, под който кадърът минава по нея
RIM_PX = 2.0  # пиксели от очертанието на обвивката в кадъра
# От размера: колко дълбоко по лъча на погледа може да е ивицата на обвивката при един пиксел от очертанието. Заоблена
# страна се докосва в точка (ивица до 0,2 от размера при RIM_ACROSS); плосък капак, гледан по ръба, или "шатрата" над
# него (обвивката над плосък връх, гледан само отниско) лежат по целия лъч - очертанието не казва къде е предметът.
RIM_DEEP = 0.35
RIM_NEAR = 200  # върхове наоколо, чиято опора се осреднява (малките петна без опора между фасетите на обвивката)
RIM_ENOUGH = 0.05  # осреднена опора, при която обвивката получава цялото HULL_RIM
# Дял на кадрите, видели едно място, чиято дълбочина там е за вярване: над него обвивката не се налага, под
# половината му - изцяло. Броят не стига: по лъскав метал 30% от пикселите са "съгласни" случайно (колонка), и няколко
# такива кадъра изключваха обвивката по почти цялата страна; по коса и дрехи на фигурка съгласни са 78%.
RIM_SHARE = 0.6
OUTSIDE = 0.02  # от размера: отвъд това извън обвивката нищо не се слива (масата и стаята)
FLOOR_BAND = 0.06  # от размера: колко високо над масата може да стигне лист от нея (меката възглавница се издува)
FLAP = 0.0125  # от размера: по-тънките от двойно това листове там са маса, а не предмет (вижте floor_sheet)
HULL_POINTS = 1_000_000  # точки вътре в обвивката, нарисувани във всеки кадър за проверка на маската му (вижте hull_view)
BLUR = 1.5  # клетки: осредненият обем се заглажда преди извличане на повърхността (без това: трохи и ръбове)
FACES = 200_000
# Заглаждане, което пази ръбовете (вижте smooth_keeping_edges): стена се усреднява със съседите си само доколкото
# нормалите им са близки (SMOOTH_TURN, разлика на единичните нормали: 0,35 е около 20 градуса), така че ситните
# вълни по гладка повърхност изчезват, а ръб или стъпало остават остри. 0 прохода: изключено.
SMOOTH_PASSES = 5
SMOOTH_TURN = 0.35
SMOOTH_MOVES = 30  # проходи, които местят върховете към новите нормали
# Равнини и цилиндри (вижте snap_shapes) там, където държи обвивката (дълбочината не е за вярване: лъскав метал):
# обвивката от очертанията е на фасети и леко крива, а предметът е точна форма. SNAP_TOL: от размера, колко далеч от
# формата са върховете, по които тя се намира (с нормала до SNAP_TURN от нейната); намерена, тя прибира и върховете
# до SNAP_WIDE от нея (неравностите вътре в нея и заобления ръб). Форма се приема само ако покрива толкова от цялата
# повърхност: равнина SNAP_FLAT, цилиндър SNAP_ROUND (тиква: два "цилиндъра" по 10-13% - не се приемат).
SNAP_TOL = 0.012
SNAP_WIDE = 0.03
SNAP_TURN = 25.0
SNAP_FLAT = 0.08
SNAP_ROUND = 0.2
SNAP_TRIES = 400
SNAP_HELD = 0.5  # връх е "където държи обвивката" при поне толкова от HULL_RIM там
SNAP_SPREAD = 40
SNAP_LEAN = 0.3  # наклон на нормалите по оста (по радиуса на единица път), над който не е цилиндър
SNAP_SQUARE = 8.0  # градуса между равнина и напречната на ос на цилиндър, под които става точно напречна
SNAP_FENCED = 0.8
SNAP_PATCH = 0.15
SNAP_FILL = 0.1
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
OUTLIER = 0.3
PASSES = 2  # проходи за гласуване, всеки пропуска несъгласуваното с предишния (1: нищо не се пропуска)
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


def untrusted(share: float) -> float:
    """Доколко на дълбочината на предмета не може да се вярва (0: може; 1: не), по дела от повърхността му, за която
    кадрите са съгласни (вижте AGREE_SHINY)."""
    return float(np.clip((AGREE_SHINY - share) / AGREE_RAMP, 0, 1))


def agreeing(frames: list, near: float) -> list[np.ndarray]:
    """За всеки кадър (поза, кутия, дълбочина, ...): пикселите, чиято точка от повърхността поне AGREE_VIEWS от
    съседните кадри виждат на същото място (до `near`, световни единици, по техния лъч). Закрита от друга част
    точка не се брои нито за, нито против. През пиксел, после разширено: дълбочината е гладка."""
    dirs = np.array([f[0][0][2] for f in frames])
    turn = np.degrees(np.arccos(np.clip(dirs @ dirs.T, -1, 1)))

    def one(i):
        (R, C, K), (x0, y0, x1, y1), depth = frames[i][:3]
        z = depth[::2, ::2]
        rows, cols = np.nonzero(z > 0)
        z = z[rows, cols].astype(np.float64)
        u, v = x0 + 2 * cols, y0 + 2 * rows
        X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1) @ R.astype(np.float64) + C
        seen = np.zeros(len(z), np.int32)
        others = [j for j in np.argsort(turn[i]) if j != i and turn[i, j] >= AGREE_APART][:AGREE_ASK]
        for j in others:
            (Rj, Cj, Kj), (a0, b0, a1, b1), dj = frames[j][:3]
            x = PAR.turned(X - Cj, Rj.astype(np.float64))
            ok = x[:, 2] > 1e-9
            uj = np.full(len(z), -1, np.int64)
            vj = np.full(len(z), -1, np.int64)
            uj[ok] = np.round(Kj[0, 0] * x[ok, 0] / x[ok, 2] + Kj[0, 2]).astype(np.int64) - a0
            vj[ok] = np.round(Kj[1, 1] * x[ok, 1] / x[ok, 2] + Kj[1, 2]).astype(np.int64) - b0
            ok &= (uj >= 0) & (vj >= 0) & (uj < a1 - a0) & (vj < b1 - b0)
            there = np.zeros(len(z))
            there[ok] = dj[vj[ok], uj[ok]]
            seen += ok & (there > 0) & (np.abs(there - x[:, 2]) <= near)
        out = np.zeros(depth[::2, ::2].shape, bool)
        out[rows, cols] = seen >= AGREE_VIEWS
        return np.kron(out, np.ones((2, 2), bool))[: depth.shape[0], : depth.shape[1]]

    return PAR.each(one, range(len(frames)))


def rim_support(hull_w: trimesh.Trimesh, poses: list) -> np.ndarray:
    """За всеки връх на обвивката (световни координати): в колко кадъра лежи на собственото ѝ очертание в кадъра, с
    погледа по повърхността ѝ. Там очертанието на предмета (маските) я докосва. poses: (R, C, K, кутия) на кадрите."""
    v, n = hull_w.vertices, hull_w.vertex_normals
    tri = hull_w.faces

    def one(pose):
        R, C, K, (x0, y0, x1, y1) = pose
        x = PAR.turned(v - C, R)
        z = np.maximum(x[:, 2], 1e-9)
        u = K[0, 0] * x[:, 0] / z + K[0, 2] - x0
        w = K[1, 1] * x[:, 1] / z + K[1, 2] - y0
        ray = (v - C) / np.linalg.norm(v - C, axis=1, keepdims=True)
        across = np.einsum("ij,ij->i", n, ray)
        front = across[tri].mean(1) < 0  # стените към камерата покриват очертанието на затворената обвивка
        img = Image.new("1", (x1 - x0, y1 - y0), 0)
        draw = ImageDraw.Draw(img)
        for t in tri[front]:
            draw.polygon([(u[t[0]], w[t[0]]), (u[t[1]], w[t[1]]), (u[t[2]], w[t[2]])], fill=1)
        inside = ndimage.distance_transform_edt(np.asarray(img))
        ui = np.clip(np.round(u).astype(int), 0, x1 - x0 - 1)
        wi = np.clip(np.round(w).astype(int), 0, y1 - y0 - 1)
        on = np.flatnonzero((inside[wi, ui] <= RIM_PX) & (np.abs(across) < RIM_ACROSS) & (x[:, 2] > 1e-9))
        # Колко дълбока е ивицата при всяко място от очертанието (клетки от 4 пиксела)
        cell = (wi[on] // 4) * ((x1 - x0) // 4 + 1) + ui[on] // 4
        near, far = np.full(cell.max(initial=0) + 1, np.inf), np.zeros(cell.max(initial=0) + 1)
        np.minimum.at(near, cell, z[on])
        np.maximum.at(far, cell, z[on])
        touched = np.zeros(len(v), bool)
        touched[on[far[cell] - near[cell] <= RIM_DEEP * size]] = True
        return touched

    size = float(np.ptp(v, axis=0).max())
    return np.sum(PAR.each(one, poses), axis=0)


def smooth_keeping_edges(mesh: trimesh.Trimesh, passes: int, turn: float = SMOOTH_TURN) -> trimesh.Trimesh:
    """Мрежата със загладени нормали на стените (двустранен филтър, Zheng и др. 2011: всяка нормала става средното
    на съседните по връх, претеглени по близост на центровете, по площ и по близост на нормалите) и върхове,
    преместени към тях (Sun и др. 2007)."""
    f = mesh.faces
    rows = np.repeat(np.arange(len(f)), 3)
    shape = (len(f), len(mesh.vertices))
    by_vertex = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, f.ravel())), shape=shape)
    near = (by_vertex @ by_vertex.T).tocoo()
    i, j = near.row[near.row != near.col], near.col[near.row != near.col]
    centre, area, n = mesh.triangles_center, mesh.area_faces, mesh.face_normals.copy()
    reach = float(np.linalg.norm(centre[i] - centre[j], axis=1).mean())
    close = np.exp(-np.sum((centre[i] - centre[j]) ** 2, axis=1) / (2 * reach**2)) * area[j]
    for _ in range(passes):
        w = close * np.exp(-np.sum((n[i] - n[j]) ** 2, axis=1) / (2 * turn**2))
        total = n * area[:, None]
        for k in range(3):
            total[:, k] += np.bincount(i, weights=w * n[j, k], minlength=len(n))
        n = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12)
    v = mesh.vertices.copy()
    count = np.maximum(np.bincount(f.ravel(), minlength=len(v)), 1)[:, None]
    for _ in range(SMOOTH_MOVES):
        centre = v[f].mean(1)
        move = np.zeros_like(v)
        for corner in range(3):
            step = n * np.einsum("ij,ij->i", n, centre - v[f[:, corner]])[:, None]
            for k in range(3):
                move[:, k] += np.bincount(f[:, corner], weights=step[:, k], minlength=len(v))
        v += move / count
    return trimesh.Trimesh(v, f, process=False)


def round_from(p1, n1, p2, n2):
    """Цилиндърът (ос, точка от оста, радиус) през две точки с нормалите им: оста е напречна и на двете нормали, а
    центърът е там, където те се пресичат, гледани по оста. None, ако нормалите са почти успоредни."""
    axis = np.cross(n1, n2)
    if np.linalg.norm(axis) < 0.2:
        return None
    axis /= np.linalg.norm(axis)
    e1 = n1 - (n1 @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    q1, q2 = np.array([p1 @ e1, p1 @ e2]), np.array([p2 @ e1, p2 @ e2])
    m1, m2 = np.array([n1 @ e1, n1 @ e2]), np.array([n2 @ e1, n2 @ e2])
    A = np.stack([m1, -m2], 1)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    c = q1 + np.linalg.solve(A, q2 - q1)[0] * m1
    return axis, c[0] * e1 + c[1] * e2, float(np.linalg.norm(q1 - c) + np.linalg.norm(q2 - c)) / 2


def on_shape(shape, P: np.ndarray):
    """За равнина ("flat", точка, нормала) или цилиндър ("round", ос, точка, радиус): върховете P, преместени върху
    формата, и нормалата на формата при всеки."""
    if shape[0] == "flat":
        _, c, n = shape
        return P - np.outer((P - c) @ n, n), np.broadcast_to(n, P.shape)
    _, axis, c, r = shape
    rel = P - c
    along = np.outer(rel @ axis, axis)
    out = (rel - along) / np.maximum(np.linalg.norm(rel - along, axis=1, keepdims=True), 1e-12)
    return c + along + out * r, out


def refit(kind: str, P: np.ndarray, N: np.ndarray):
    """Формата от вида, напасната по своите върхове (и нормалите им)."""
    if kind == "flat":
        c = P.mean(0)
        n = np.linalg.svd(P - c, full_matrices=False)[2][2]
        return "flat", c, n if n @ N.mean(0) > 0 else -n
    rows = np.linalg.svd(N, full_matrices=False)[2]
    axis = rows[2]
    e1 = rows[0] - (rows[0] @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    x, y = P @ e1, P @ e2
    cx, cy, k = np.linalg.lstsq(np.stack([2 * x, 2 * y, np.ones(len(x))], 1), x * x + y * y, rcond=None)[0]
    return "round", axis, cx * e1 + cy * e2, float(np.sqrt(max(k + cx * cx + cy * cy, 1e-12)))


def snap_shapes(mesh: trimesh.Trimesh, pool: np.ndarray, rng=None) -> tuple[trimesh.Trimesh, list]:
    """Мрежата с върховете от `pool` (булева маска), преместени върху равнините и цилиндрите, намерени сред тях
    (RANSAC: всяка двойка случайни върхове предлага равнина и цилиндър; печели формата с най-много площ), и
    намерените форми с дяла им от повърхността."""
    P, N = np.asarray(mesh.vertices, float), np.asarray(mesh.vertex_normals, float)
    size = float(mesh.extents.max())
    tol, wide, turn = SNAP_TOL * size, SNAP_WIDE * size, np.cos(np.radians(SNAP_TURN))
    area = np.zeros(len(P))
    for corner in range(3):
        np.add.at(area, mesh.faces[:, corner], mesh.area_faces / 3)
    total = area.sum()
    rng = rng or np.random.default_rng(0)

    def fits(shape, idx):
        at, n = on_shape(shape, P[idx])
        return (np.linalg.norm(at - P[idx], axis=1) < tol) & (np.einsum("ij,ij->i", n, N[idx]) > turn)

    free, shapes = pool.copy(), []
    for _ in range(6):
        idx = np.flatnonzero(free)
        if area[idx].sum() < SNAP_FLAT * total or len(idx) < 50:
            break
        need = {"flat": SNAP_FLAT * total, "round": SNAP_ROUND * total}
        bests = {"flat": (0.0, None, None), "round": (0.0, None, None)}  # най-добрата от всеки вид поотделно
        for _ in range(SNAP_TRIES):
            i, j = rng.choice(idx, 2, replace=False)
            found = round_from(P[i], N[i], P[j], N[j])
            for shape in [("flat", P[i], N[i])] + ([("round", *found)] if found and found[2] < 2 * size else []):
                ok = fits(shape, idx)
                covers = area[idx[ok]].sum()
                if covers > bests[shape[0]][0]:
                    bests[shape[0]] = (covers, shape, ok)
        enough = [b for kind, b in bests.items() if b[0] >= need[kind]]
        if not enough:
            break
        best = max(enough, key=lambda b: b[0])
        shape, ok = best[1], best[2]
        for _ in range(3):  # уточняване по своите върхове, докато покрива повече
            again = refit(shape[0], P[idx[ok]], N[idx[ok]])
            more = fits(again, idx)
            if area[idx[more]].sum() <= area[idx[ok]].sum():
                break
            shape, ok = again, more
        free[idx[best[2]]] = False  # и да не се приеме: тези върхове не се опитват пак
        share = area[idx[ok]].sum() / total
        if shape[0] == "round":
            # Пояс от кълбо или бъчва също е "близо до цилиндър", но нормалите му се накланят по оста: на кълбо с
            # радиус r наклонът расте с 1/r на единица път по оста, на цилиндър - с 0.
            along = (P[idx[ok]] - shape[2]) @ shape[1]
            lean = np.polyfit(along, N[idx[ok]] @ shape[1], 1)[0] * shape[3] if np.ptp(along) > 0 else 1.0
            if abs(lean) > SNAP_LEAN:
                continue
        if share >= (SNAP_FLAT if shape[0] == "flat" else SNAP_ROUND):
            shapes.append((shape, share))
            free[idx[ok]] = False
    if not shapes:
        return mesh, []
    # Равнина почти напречна на оста на цилиндър (капак, дъно) става точно напречна: поотделно напаснати, те се
    # разминават с няколко градуса и капакът излиза килнат.
    axes = [shape[1] for shape, _ in shapes if shape[0] == "round"]
    for k, (shape, share) in enumerate(shapes):
        if shape[0] == "flat":
            for axis in axes:
                if abs(shape[2] @ axis) > np.cos(np.radians(SNAP_SQUARE)):
                    shapes[k] = (("flat", shape[1], axis * np.sign(shape[2] @ axis)), share)
    # Всеки връх от pool отива върху формата, до която е близо (SNAP_WIDE) и чиято нормала му е най-близка.
    idx = np.flatnonzero(pool)
    best_fit, target = np.full(len(idx), 0.5), P[idx].copy()  # нормала поне на 60 градуса от тази на формата
    for shape, _ in shapes:
        at, n = on_shape(shape, P[idx])
        like = np.einsum("ij,ij->i", n, N[idx])
        take = (np.linalg.norm(at - P[idx], axis=1) < wide) & (like > best_fit)
        best_fit[take], target[take] = like[take], at[take]
    # Преместването се разнася и към съседните върхове, затихвайки (SNAP_SPREAD осреднявания по ръбовете на мрежата):
    # иначе между преместените и останалите остава назъбено стъпало (при колонката - под ръба на капака).
    move = np.zeros_like(P)
    move[idx] = target - P[idx]
    moved = np.zeros(len(P), bool)
    moved[idx] = np.any(move[idx] != 0, axis=1)
    e = mesh.edges_unique
    ones = np.ones(len(e) * 2, np.float32)
    link = sparse.csr_matrix((ones, (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(P), len(P)))
    # Петно от останалите върхове, оградено почти изцяло от преместени (SNAP_FENCED) и не голямо (SNAP_PATCH от
    # повърхността), е част от формите около него: върховете му отиват върху най-близката, ако е до SNAP_FILL (където
    # държи обвивката) или до SNAP_WIDE (където дълбочината е за вярване: шумът по печата върху метала е до
    # толкова, а каквото стърчи повече - копче, дръжка - остава).
    rest = np.flatnonzero(~moved)
    if len(rest):
        _, label = sparse.csgraph.connected_components(link[rest][:, rest], directed=False)
        for k in np.flatnonzero(np.bincount(label, weights=area[rest]) <= SNAP_PATCH * total):
            patch = rest[label == k]
            around = np.setdiff1d(link[patch].indices, patch)
            if len(around) == 0 or moved[around].mean() < SNAP_FENCED:
                continue
            near, goal = np.where(pool[patch], SNAP_FILL, SNAP_WIDE) * size, P[patch].copy()
            for shape, _ in shapes:
                at, _ = on_shape(shape, P[patch])
                off = np.linalg.norm(at - P[patch], axis=1)
                goal[off < near], near = at[off < near], np.minimum(near, off)
            move[patch] = goal - P[patch]
        moved[rest] = np.any(move[rest] != 0, axis=1)
    count = np.maximum(np.asarray(link.sum(1)).ravel(), 1)[:, None]
    for _ in range(SNAP_SPREAD):
        move[~moved] = ((link @ move) / count)[~moved]
    return trimesh.Trimesh(P + move, mesh.faces, process=False), shapes


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
               prev: np.ndarray | None, trunc: float, told: np.ndarray | None = None) -> int:  # fmt: skip
    """Гласовете на всеки кадър за един блок от клетките на обема (world: центровете им; cells: местата им в tsdf и
    weight; два блока нямат обща клетка, затова вървят едновременно, а средното не зависи от реда на гласовете). С
    `prev` (предишния проход) гласовете за повърхност далеч от него отпадат. Връща колко."""
    dropped = 0
    for (R, C, K), box, depth, wmap, fmap, *rest in frames:
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
        sure = rest[0][v[ok], u[ok]] if rest and told is not None else None
        val = np.minimum(1.0, sd / trunc).astype(np.float32)
        cell = cells[idx]
        if prev is not None:
            keep = ~surface | (val >= 1.0) | (np.abs(val - prev[cell]) <= OUTLIER)
            dropped += int((~keep).sum())
            cell, val, surface, w = cell[keep], val[keep], surface[keep], w[keep]
            sure = sure[keep] if sure is not None else None
        if sure is not None:  # колко кадъра са казали нещо за клетката и колко от тях с дълбочина за вярване
            told[1, cell] = np.minimum(told[1, cell], 254) + 1
            told[0, cell[sure]] = np.minimum(told[0, cell[sure]], 254) + 1
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
    rim_cells = np.flatnonzero(np.abs(sdf) <= trunc) if HULL_RIM > 0 else None
    held = None
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
    # Към всеки кадър: къде на дълбочината му може да се вярва (съгласна със съседите, или измерен фон извън предмета).
    # shiny: доколко на дълбочината на целия предмет не може да се вярва (0: може; 1: не), по дела от повърхността,
    # за която кадрите са съгласни. Само тогава несъгласната дълбочина губи глас и обвивката се налага.
    shiny = 0.0
    if AGREE_VIEWS > 0 and HULL_RIM > 0:
        sure = agreeing(frames, AGREE_NEAR * trunc)
        share = np.mean([a[f[2] > 0].mean() for a, f in zip(sure, frames)])
        shiny = untrusted(float(share))
        note(f"frames agree on {share:.0%} of the surface they see: "
             + ("the depth is trusted" if shiny == 0 else f"the outlines' shape holds where they do not ({shiny:.0%})"))
        unsure = np.float32(1 - shiny * (1 - UNSURE))
        frames = [(*f[:3], f[3] * np.where(a, np.float32(1), unsure), f[4], a | (f[2] <= 0))
                  for a, f in zip(sure, frames)]  # fmt: skip
        del sure
    else:
        frames = [(*f, np.ones(f[2].shape, bool)) for f in frames]
    if shiny == 0:
        rim_cells = None
    told = np.zeros((2, len(tsdf)), np.uint8) if rim_cells is not None else None  # за вярване, всички

    def vote(tsdf: np.ndarray, weight: np.ndarray, prev: np.ndarray | None) -> int:
        starts = range(0, len(active), BLOCK)
        return sum(PAR.each(lambda s: vote_block(frames, world[s : s + BLOCK], active[s : s + BLOCK], tsdf, weight, prev,
                                                 trunc, told), starts))  # fmt: skip

    passes = (int(z["passes"]) if "passes" in z.files else PASSES) if OUTLIER > 0 else 1
    empty_tsdf, empty_weight, prev = tsdf, weight, None
    for k in range(passes):
        tsdf, weight = empty_tsdf.copy(), empty_weight.copy()
        if told is not None:
            told[:] = 0
        dropped = vote(tsdf, weight, prev)
        prev = tsdf
        note(f"pass {k + 1}: {dropped / 1e6:.1f} million votes left out as far from the others")
    if rim_cells is not None:
        # Обвивката там, където очертанията я докосват, но само доколкото никой кадър с дълбочина за вярване не
        # казва друго за това място (told: колко такива кадъра): фигурка - косата и ръцете остават от дълбочината,
        # гланцовите очи и лъскавият метал - от обвивката.
        sup = rim_support(trimesh.Trimesh(hull.vertices @ to_up, hull.faces, process=False),
                          [(R, C, K, box) for (R, C, K), box, *_ in frames])  # fmt: skip
        at = np.stack(np.unravel_index(rim_cells, tuple(shape)), 1) * step + lo
        _, near = cKDTree(hull.vertices).query(at, workers=-1)
        full = np.minimum(sup, RIM_FULL) / RIM_FULL
        _, around = cKDTree(hull.vertices).query(hull.vertices, k=min(RIM_NEAR, len(sup)), workers=-1)
        full = np.minimum(full[around].mean(1) / RIM_ENOUGH, 1.0)
        share = told[0, rim_cells] / np.maximum(told[1, rim_cells], 1)
        doubt = np.clip((RIM_SHARE - share) / (RIM_SHARE / 2), 0, 1)
        extra = (shiny * HULL_RIM * full[near] * doubt).astype(np.float32)
        tsdf[rim_cells] = (tsdf[rim_cells] * weight[rim_cells] + empty_tsdf[rim_cells] * extra) / (
            weight[rim_cells] + extra
        )
        weight[rim_cells] += extra
        held = np.zeros(len(tsdf), bool)  # където обвивката държи (за snap_shapes)
        held[rim_cells] = extra >= SNAP_HELD * shiny * HULL_RIM
        note(f"outlines touch {np.mean(sup >= RIM_FULL):.0%} of the hull in {RIM_FULL}+ frames; the hull holds "
             f"{np.mean(extra > 0.5 * HULL_RIM):.0%} of the cells at its surface")  # fmt: skip
        del at, near, extra, doubt, told
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
    mesh = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix())
    if SMOOTH_PASSES > 0:
        mesh = smooth_keeping_edges(mesh, SMOOTH_PASSES)
    shapes = []
    if held is not None and SNAP_TOL > 0:
        at = np.clip(np.round((mesh.vertices - lo) / step).astype(int), 0, shape - 1)
        pool = ndimage.binary_dilation(held.reshape(tuple(shape)))[at[:, 0], at[:, 1], at[:, 2]]
        mesh, shapes = snap_shapes(mesh, pool)
        note(f"the hull holds {pool.mean():.0%} of the surface; exact shapes there: "
             + (", ".join(f"{s[0]} {share:.0%}" for s, share in shapes) or "none"))  # fmt: skip
    mesh.export(out)
    # Посоките на намерените форми (оста на цилиндър, нормалата на равнина), за изправяне на модела по тях.
    ups = [{"kind": s[0], "up": [float(v) for v in (s[1] if s[0] == "round" else s[2])], "share": float(share)}
           for s, share in shapes]  # fmt: skip
    out.with_suffix(".json").write_text(json.dumps({"frames": used, "total": len(names), "shapes": ups}))


if __name__ == "__main__":
    fuse(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
