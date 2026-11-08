"""Положения на камерата от сензорите за движение на телефона: пътят за предмети, които фотограметрията не може да
разположи. Съпоставянето на особености изисква шарка. Върху еднообразен плат кадрите, гледащи надолу към предмета,
виждат малко друго и решаващият алгоритъм ги пропуска или слага на грешно място (колонка върху червено кадифе:
високият кръг излезе на отделни части, а моделът беше дъска). Телефонът обаче знае как е завъртян: приложението
записва положението му по време на заснемане. Предметът остава в средата на изображението, докато потребителят го
обикаля, така че камерата на всеки кадър е на лъча през центъра на предмета в маската му, на разстояние, обратно
пропорционално на видимия му размер. Така грешка в положението само леко премества гледната точка около предмета,
вместо да я измества настрани. Очертанията (AI маски) от всички тези гледни точки издълбават формата: едноцветни,
лъскави или тъмни предмети работят толкова добре, колкото и шарените. Това, което сензорите не казват, се напасва към
очертанията: закъснението между заявката за запис и първия кадър, а за всеки кадър точното разстояние, център и посока
на компаса. Фокусното разстояние идва от решаващия алгоритъм, когато е разположил някои кадри, иначе е типична за
телефон стойност. Кадрите са изправени във вертикалната ориентация на телефона (изображението нагоре = +y на
телефона): приложението е заключено вертикално, а камерата отбелязва видеата по ориентацията на екрана, а не по това
как се държи телефонът. (Посоката на завъртане на изображението не може да се определи от очертанията: при кръгъл
предмет и четирите завъртания съвпадаха в рамките на 4%.) motion.json (качен от приложението): {"samples": [[t, alpha,
beta, gamma], ...]}, t в секунди от заявката за запис, ъгли в радиани, DeviceOrientation по W3C: R = Rz(alpha)
Rx(beta) Ry(gamma) завърта осите на телефона (x надясно, y нагоре, z извън екрана) към тези на света (x на изток, y на
север, z нагоре). Работи като собствен процес (стъпка "shape from motion sensors"), за да може да се отказва като
другите инструменти: python motion.py <папка на задачата> <маски на снимките.npz> <фокусно разстояние в px, 0 ако е
неизвестно> <out.ply> <плоска основа 0/1>. Записва <out>.json {"score", "frames", "delay"} и, когато очертанията са
съгласувани (score >= USABLE), мрежата за печат като <out>."""

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

if __package__:
    from . import depth as D
    from . import mesh as M
    from . import segment as S
else:  # изпълнение като скрипт: импортира пакета по пътя му
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import depth as D
    from app import mesh as M
    from app import segment as S

MOTION = "motion.json"  # в папката на задачата
FRAME_TIMES = "frames.json"  # в папката на задачата: секунди от началото на видеото за всеки избран кадър
SHIFTS = np.arange(-0.5, 2.01, 0.25)  # първият видео кадър е толкова след заявката за запис, в секунди
MIN_FRAMES = 12
# Кадрите трябва да гледат предмета поне от толкова градуса наоколо. При по-малко очертанията са съгласувани при
# всякакви положения (заснето от една страна или заседнал сензор) и формата е предположение.
MIN_AROUND = 200
YAW_STEP = np.radians(3)  # корекция на посоката, пробвана за всеки кадър и кръг
# Фокусно разстояние / дълга страна на изображението, когато решаващият алгоритъм не го е оценил (основни камери на
# телефони при 16:9: около 0,75; COLMAP намери 0,77 за една). Не се напасва към очертанията: те са по-съгласувани при
# по-дълго разстояние (по-далечни и плоски изгледи), независимо от истината.
FOCAL = 0.77
GOOD = 0.75  # медианното припокриване на очертанията, от което на положенията от сензорите се вярва
USABLE = 0.5  # ...и под което изобщо не се използват
CAM_FROM_DEVICE = np.diag([1.0, -1.0, -1.0])  # камера: x надясно, y надолу, z напред, навън от гърба на телефона
# Дял от кадрите, които трябва да виждат воксел в очертанието (mesh.visual_hull). Пътят с решаващия алгоритъм прощава
# повече (0,75), което разширяваше овал, заснет отвсякъде, със 17%; при 0,9 с 3%.
AGREE = 0.9
DEFAULT_DELAY = 0.3  # секунди, когато очертанията са еднакво съгласувани при всякакво закъснение (кръгли предмети, равномерно ходене)


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


@dataclass
class Frames:
    """Кадрите, показващи предмета, с местоположението (px) и размера му (px, корен квадратен от площта)."""

    names: list[str]
    times: np.ndarray
    masks: list[np.ndarray]
    anchors: np.ndarray  # (n, 2) център на маската в пиксели на изображението
    sizes: np.ndarray  # (n,)
    width: int
    height: int


def frames(names: list[str], times: dict, masks: dict, scale: int, size: tuple[int, int]) -> Frames:
    keep, anchors, sizes = [], [], []
    for n in names:
        fg = masks[n] > 127
        if n not in times or fg.sum() < 50:
            continue
        ys, xs = np.nonzero(fg)
        keep.append(n)
        anchors.append(((xs.mean() + 0.5) * scale, (ys.mean() + 0.5) * scale))
        sizes.append(np.sqrt(len(xs)) * scale)
    return Frames(keep, np.array([times[n] for n in keep]), [masks[n] for n in keep], np.array(anchors).reshape(-1, 2),
                  np.array(sizes), size[0], size[1])  # fmt: skip


def headings(dev: np.ndarray) -> np.ndarray:
    """Накъде гледа всеки кадър, в радиани около вертикалата, с продължаване през пълните обиколки (ред на заснемане)."""
    fwd = (dev @ CAM_FROM_DEVICE)[:, :, 2]
    return np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))


def steadied(fr: Frames, dev: np.ndarray) -> Frames:
    """Кадрите с размера на предмета, какъвто би изглеждал без формата си: средният (логаритмичен) размер за кадрите,
    заснети в рамките на 90° наоколо. Кадър по кадър широката страна на овал изглежда по-близо от тясната и обвивката
    излизаше по-кръгла от предмета (1,5:1 като 1,2:1). Останалото, хора, вървящи по-близо или по-далеч, се напасва за
    всеки кадър (`refine`)."""
    around = headings(dev)
    near = np.abs(around[:, None] - around[None, :]) <= np.pi / 2
    return replace(fr, sizes=np.exp(near @ np.log(fr.sizes) / near.sum(1)))


def heading(yaw: np.ndarray) -> np.ndarray:
    """(n, 3, 3) завъртания около вертикалната ос."""
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(yaw), np.ones_like(yaw)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def views(dev: np.ndarray, fr: Frames, focal: float, dist=None, offset=None, yaw=None) -> M.Views:
    """Камери, гледащи центъра на предмета (началото на световните координати), от центъра на очертанието му във всеки
    кадър. dist: множител на разстоянието за кадър; offset: изместване на този център за кадър, в размери на предмета;
    yaw: корекция на посоката на компаса за кадър, в радиани."""
    n = len(fr.names)
    cx, cy = fr.width / 2, fr.height / 2
    if yaw is not None:
        dev = heading(yaw) @ dev
    C = dev @ CAM_FROM_DEVICE  # камера -> свят
    a = fr.anchors + (offset if offset is not None else 0) * fr.sizes[:, None]
    ray = np.stack([(a[:, 0] - cx) / focal, (a[:, 1] - cy) / focal, np.ones(n)], -1)
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    d = (dist if dist is not None else 1.0) * focal / fr.sizes  # предметът е "голям" 1 единица
    centers = -d[:, None] * np.einsum("nij,nj->ni", C, ray)
    R = C.transpose(0, 2, 1)
    return M.Views(R, -np.einsum("nij,nj->ni", R, centers), np.tile([focal, focal, cx, cy], (n, 1)), list(fr.names))


def score(v: M.Views, masks: list, scale: int, cells: int = 40) -> float:
    """Колко са съгласувани всички очертания: медианно припокриване на всяка маска с обвивката от всички кадри."""
    iou = M.silhouette_consistency(v, masks, scale, cells, AGREE)
    return float(np.median(np.nan_to_num(iou))) if len(iou) else 0.0


def fit(fr: Frames, samples: np.ndarray, scale: int, focal: float):
    """Закъснението на видеото след заявката за запис, по това колко се съгласуват очертанията при него. При
    равномерно ходене грешното закъснение завърта всяка гледна точка на един и същ ъгъл около предмета, което не
    променя нищо; показват го (и го правят важно) само промените в темпото. Връща (завъртанията на телефона за
    кадър, оценка, закъснение)."""

    def at(s: float) -> float:
        dev = device_rotations(samples, fr.times + s)
        return score(views(dev, steadied(fr, dev), focal), fr.masks, scale, 64)

    default = at(DEFAULT_DELAY)
    best, shift = max((at(s), float(s)) for s in SHIFTS)
    for s in shift + np.arange(-0.2, 0.201, 0.05):
        if (sc := at(s)) > best:
            best, shift = sc, float(s)
    if best < default + 0.002:  # няма ясен знак
        best, shift = default, DEFAULT_DELAY
    return device_rotations(samples, fr.times + shift), best, shift


def refine(fr: Frames, dev: np.ndarray, focal: float, scale: int, rounds: int = 2):
    """За всеки кадър разстоянието, центърът на очертанието и посоката, които най-добре пасват на обвивката от всички
    кадри (хората не вървят на постоянно разстояние, центърът на маска не е точно този на предмета, а компасът се
    колебае близо до метал). Връща изгледите."""
    n = len(fr.names)
    dist, offset, yaw = np.ones(n), np.zeros((n, 2)), np.zeros(n)
    steps = [(f, 0, 0, 0) for f in (0.9, 0.95, 1.05, 1.1)]
    steps += [(1, dx, dy, 0) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))] + [(1, 0, 0, -1), (1, 0, 0, 1)]
    for _ in range(rounds):
        v = views(dev, fr, focal, dist, offset, yaw)
        grid = M.hull_grid(v, fr.masks, scale, 64, AGREE)
        if grid is None:
            break
        occ, step = grid
        for i in range(n):
            def iou(d, o, w):
                one = views(dev[i : i + 1], _one(fr, i), focal, np.array([d]), o[None], np.array([w]))
                return M.outline_iou(occ, step, one.R[0], one.t[0], one.K[0], fr.masks[i], scale)

            best = iou(dist[i], offset[i], yaw[i])
            for f, dx, dy, dw in steps:
                cand = dist[i] * f, offset[i] + 0.03 * np.array([dx, dy]), yaw[i] + YAW_STEP * dw
                c = iou(*cand)
                if c > best:
                    best, (dist[i], offset[i], yaw[i]) = c, cand
    return views(dev, fr, focal, dist, offset, yaw)


def _one(fr: Frames, i: int) -> Frames:
    return Frames([fr.names[i]], fr.times[i : i + 1], [fr.masks[i]], fr.anchors[i : i + 1], fr.sizes[i : i + 1],
                  fr.width, fr.height)  # fmt: skip


def floor_height(v: M.Views, masks: list, scale: int, occ: np.ndarray, step: float) -> float | None:
    """Височината на масата: където лъчът през дъното на всяко очертание за пръв път влиза в обвивката. Тази точка е
    върху най-ниския ръб на предмета към камерата, който стои върху масата. (Самата обвивка стига под масата като
    клин, защото никой кадър не вижда под предмета.)"""
    tree = cKDTree(occ)
    zs = []
    for R, t, (fx, fy, cx, cy), m in zip(v.R, v.t, v.K, masks):
        ys, xs = np.nonzero(m > 127)
        if len(ys) < 50:
            continue
        row = ys.max() - 1  # малко навътре от дъното на очертанието
        cols = xs[ys == row]
        if not len(cols):
            continue
        u, w = (cols.mean() + 0.5) * scale, (row + 0.5) * scale
        d = R.T @ [(u - cx) / fx, (w - cy) / fy, 1.0]
        d /= np.linalg.norm(d)
        c = -R.T @ t
        r = np.linalg.norm(c)
        pts = c + np.linspace(0.3 * r, 1.7 * r, 600)[:, None] * d
        hit = np.flatnonzero(tree.query(pts)[0] <= 0.75 * step)
        if len(hit):
            zs.append(pts[hit[0], 2])
    return float(np.median(zs)) if zs else None


def depth_carver(v: M.Views, masks: list, photos: list[Path], hull: trimesh.Trimesh):
    """Какво според AI дълбочината е празно в обвивката от очертанията (app/depth.py), като limit(points) за
    M.visual_hull; None без модела или без достатъчно напаснати кадри. Дълбочината се мащабира към стените на
    обвивката, които очертанията дават вярно (обвивката греши при върхове и вдлъбнатини)."""
    if not D.MODEL.exists():
        return None
    samples, face = trimesh.sample.sample_surface_even(hull, 30_000, seed=0)
    refs = samples[np.abs(hull.face_normals[face][:, 2]) < 0.5]
    fits = D.fit_frames(photos, masks, v.R, v.t, v.K, refs, np.ones(len(refs)))
    if not D.enough(fits, len(photos)):
        return None
    margin = D.CARVE_MARGIN * float(hull.extents.max())
    return lambda points: D.carve_limit(fits, points, margin, AGREE)


def model(fr: Frames, v: M.Views, scale: int, photos: list[Path] | None, flat_base: bool) -> trimesh.Trimesh:
    """Мрежата за печат: обвивката от всички очертания, стояща върху масата, издълбана с AI дълбочина и оцветена от
    кадрите."""
    grid = M.hull_grid(v, fr.masks, scale, 96, AGREE)
    if grid is None:
        raise M.MeshError("The object's outline could not be followed around. Keep it inside the circle on screen.")
    occ, step = grid
    floor = floor_height(v, fr.masks, scale, occ, step)
    if floor is None:
        floor = float(occ[:, 2].min())
    cloud = M.Cloud(occ, np.zeros_like(occ), np.ones((len(occ), 4)), floor, np.eye(3))  # световното z вече е нагоре
    hull = M.visual_hull(v, fr.masks, scale, cloud, cells=200, agree=AGREE)
    if hull is None:
        raise M.MeshError("The object's outlines did not agree on a shape. Film again more slowly.")
    limit = depth_carver(v, fr.masks, photos, hull) if photos else None
    if limit is not None:
        carved = M.visual_hull(v, fr.masks, scale, cloud, cells=200, agree=AGREE, limit=limit)
        share = 1 - carved.volume / hull.volume if carved is not None else 1.0
        use = share <= D.MAX_CARVE
        print(f"AI depth carves {share:.0%} of the outline shape: " + ("using it" if use else "not used"), flush=True)
        if use:
            hull = carved
    trimesh.smoothing.filter_taubin(hull, iterations=10)
    if photos:
        M.paint(hull, v, photos, np.eye(3))
    return M.make_printable(hull, floor, flat_base)


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


def main(job: Path, masks_path: Path, focal: float, out: Path, flat_base: bool) -> dict:
    info = {"score": 0.0, "frames": 0, "around": 0, "delay": None}
    data = load(job)
    with np.load(masks_path) as z:
        masks = {n: z[n] for n in z.files}
    if data is not None and masks:
        samples, times = data
        names = sorted(masks)
        with Image.open(job / "photos" / names[0]) as im:
            size = im.size
        if not 0.5 <= focal / max(size) <= 1.3:  # неизвестно или от грешно решение за камерите
            focal = FOCAL * max(size)
        fr = frames(names, times, masks, S.MASK_SCALE, size)
        info["frames"] = len(fr.names)
        around = float(np.degrees(np.ptp(headings(device_rotations(samples, fr.times + DEFAULT_DELAY)))))
        info["around"] = round(around)
        print(f"{len(fr.names)} frames show the object, filmed from {around:.0f}° around it", flush=True)
        if len(fr.names) >= MIN_FRAMES and around >= MIN_AROUND:
            dev, sc, shift = fit(fr, samples, S.MASK_SCALE, focal)
            print(f"focal {focal:.0f} px: outlines agree {sc:.0%} with the video starting {shift:.2f} s after the "
                  "request", flush=True)  # fmt: skip
            fr = steadied(fr, dev)
            v = refine(fr, dev, focal, S.MASK_SCALE)
            info |= {"score": round(score(v, fr.masks, S.MASK_SCALE), 3), "delay": round(shift, 2)}
            print(f"after fitting distance, centre and heading per frame: {info['score']:.0%}", flush=True)
            if info["score"] >= USABLE:
                model(fr, v, S.MASK_SCALE, [job / "photos" / n for n in fr.names], flat_base).export(out)
    out.with_suffix(".json").write_text(json.dumps(info))
    return info


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4]), sys.argv[5] == "1")
