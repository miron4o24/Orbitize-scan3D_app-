"""Снимки (или видео кадри) -> положения на камерите (COLMAP) -> гъсти точки (OpenMVS) -> мрежа за печат (app.mesh)."""

import functools
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import numpy as np
import pymeshlab
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from . import depth as D
from . import fuse as F
from . import mesh as M
from . import motion as MO
from . import refine as RF
from . import segment as S
from . import texture as TX
from . import video as V

# Инструментите се намират в PATH (scripts/run-server.ps1 ги добавя на Windows). При нужда ffmpeg идва от версията в
# пакета за pip.
COLMAP = os.environ.get("COLMAP", "colmap")
FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
THREADS = str(os.cpu_count() or 2)
MAX_IMAGE = os.environ.get("MAX_IMAGE_SIZE", "1600")
MAX_POINTS = 1_500_000  # пази почистването и Поасон в рамките на няколко GB памет
VIDEO = "video.mp4"  # в папката на задачата; съдържание MP4 или MOV, ffmpeg се води по съдържанието
MAX_FRAME_SIDE = 1920  # кадрите за гъсто съпоставяне и всичко освен текстурата се смаляват до това
# Бързи сканирания: гъстото съпоставяне при този размер отнема около 55% от времето. Клетките на слятата повърхност
# (0,25% от предмета) все още покриват повече от пиксел; текстурата пази кадрите си в пълен размер.
QUICK_FRAME_SIDE = 1440
# Експресни сканирания (Pro, около 5 минути на 12-те нишки на сървъра): по времената на стъпките при сканиране с 80
# кадъра съпоставянето на снимки расте с квадрата на кадрите, а останалото с кадрите; 30 кадъра, гъсто съпоставяне при
# 1280 px, без проход на MobileSAM (1,5 мин) и клетки за сливане от 0,4% (четвърт от клетките) дават около 4-5 мин.
EXPRESS_FRAMES = 30
EXPRESS_FRAME_SIDE = 1280
EXPRESS_VOXEL = 0.004
# Един инкрементален модел с този дял от кадрите: глобалният алгоритъм не се изпълнява допълнително (помага само
# когато инкременталният раздели сканирането на части).
ONE_PIECE = 0.95
# От размера на предмета: колко пред измерена точка от повърхността кадър трябва да вижда през обвивката, за да я
# издълбае (гъстите точки имат шум от няколко десети от процента).
FREE_MARGIN = 0.02
# Усещането на телефона за посока нагоре, завъртяно кадър по кадър в координатите на решаващия алгоритъм, трябва да е
# съгласувано толкова добре (медиана в градуси), за да му се вярва: така дневникът на положението се проверява спрямо
# решените камери.
GRAVITY_AGREE = 10.0
# Под този дял от кадрите, разположени от решаващия алгоритъм, се пробват сензорите за движение на телефона
# (app/motion.py): едноцветна колонка върху кадифе имаше 46 от 80 разположени кадъра от част от кръга и излезе клин с
# ръбове.
SFM_ENOUGH = 0.85
# Повърхността, слята от картите на дълбочината (app/fuse.py), се използва, когато гъстите точки са толкова близо до
# нея (медиана, от размера на предмета; при дрелка 0,2%) и поне половината кадри са влезли в нея; иначе Поасон върху
# гъстите точки.
FUSED_OFF = 0.01


class Failed(Exception):
    """Съобщение за неуспех за потребителя."""


OOM_WORDS = ("bad allocation", "bad_alloc", "out of memory", "memoryerror", "failed to allocate", "not enough memory")


def ran_out_of_memory(job: Path, since: int = 0) -> bool:
    """В дневника на задачата (от байт `since` нататък) има грешка от недостиг на памет."""
    try:
        with open(job / "log.txt", "rb") as f:
            f.seek(since)
            text = f.read().decode("utf-8", "replace").lower()
    except OSError:
        return False
    return any(w in text for w in OOM_WORDS)


def log_size(job: Path) -> int:
    try:
        return (job / "log.txt").stat().st_size
    except OSError:
        return 0


def note(job: Path, text: str):
    """Ред в log.txt на задачата до изхода на инструментите (GET /jobs/{id}/log)."""
    try:
        # UTF-8: подразбиращото се на Windows (cp1252) не може да запише българския текст в някои редове и записът би
        # провалил задачата.
        with open(job / "log.txt", "a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n[{time.strftime('%H:%M:%S')}] {text}\n")
    except OSError:  # задачата е изтрита междувременно
        pass


@functools.cache
def colmap_prefixes() -> tuple[str, str]:
    """COLMAP 4 преименува опциите SiftExtraction./SiftMatching. на FeatureExtraction./FeatureMatching."""
    r = subprocess.run([COLMAP, "feature_extractor", "-h"], capture_output=True, text=True)
    out = r.stdout + r.stderr  # COLMAP 4 печата помощта в stderr
    return ("FeatureExtraction", "FeatureMatching") if "FeatureExtraction.use_gpu" in out else ("SiftExtraction", "SiftMatching")


@functools.cache
def has_global_mapper() -> bool:
    """COLMAP 4 включва GLOMAP като `colmap global_mapper`; пакетът на Debian (3.x) не го включва."""
    r = subprocess.run([COLMAP, "help"], capture_output=True, text=True)
    return "global_mapper" in r.stdout + r.stderr


def video_format(video: Path) -> str:
    """Кодек, размер и кадрова честота, както ги чете ffmpeg: приложението иска 4K, но телефони без него в приложния
    интерфейс на камерата записват по-малко (наблюдавано: 1080p), а този размер ограничава детайла повече от всичко
    тук."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True, text=True, errors="replace")
    line = next((ln.strip() for ln in r.stderr.splitlines() if "Video:" in ln), "")
    return line.split("Video:", 1)[1].strip()[:200] if line else "unknown"


# Добавяне на липсваща страна: кадрите на изходния модел чакат тук (app/main.py start_job), докато излязат тези от
# новото видео.
MORE_FRAMES = "more_frames"
EXTRA_FRAMES = 30  # кадри от новото видео: старите вече покриват останалото


def frames_to_take(job: Path, most: int) -> int:
    return min(most, EXTRA_FRAMES) if (job / MORE_FRAMES).is_dir() else most


def merge_more_frames(job: Path, photos: Path):
    """Кадрите на изходния модел до тези от новото видео, решени заедно като едно сканиране. Дневникът на положението
    покрива само новото видео, затова отпада. Всички кадри трябва да са с един размер: решаващият алгоритъм ги приема
    като една камера."""
    more = job / MORE_FRAMES
    if not more.is_dir():
        return
    old = sorted(more.glob("*.jp*g"))
    new = sorted(photos.glob("*.jp*g"))
    if not old or not new:
        raise Failed("Кадрите на модела вече ги няма. Направете ново сканиране.")
    with Image.open(old[0]) as a, Image.open(new[0]) as b:
        if a.size != b.size:
            raise Failed("Заснемете допълнението със същия телефон и в същото положение като първото видео.")
    for f in old:
        shutil.copy(f, photos / f.name)
    (job / MO.MOTION).unlink(missing_ok=True)
    (job / MO.FRAME_TIMES).unlink(missing_ok=True)
    note(job, f"adding a side: {len(new)} new frames with the model's {len(old)}")


def extract_frames(video: Path, photos: Path, work: Path, sh: Callable[[str, list], None], most: int = V.MAX_FRAMES) -> int:
    """Записва най-резките, равномерно разпределени кадри от видеото (най-много `most`) като photos/img_0001.jpg...
    Връща колко са."""
    for p in photos.glob("*.jp*g"):  # повторен опит извлича отново от видеото
        p.unlink()
    note(work.parent, f"video file: {video_format(video)}")
    # И двата прохода на ffmpeg започват със същия филтър fps, така че кадър n означава един и същ кадър и в двата.
    raw = work / "thumbs.gray"
    sh("reading video", [FFMPEG, "-v", "error", "-y", "-i", video, "-an",
        "-vf", f"fps={V.CANDIDATE_FPS},scale={V.THUMB}:{V.THUMB},format=gray", "-f", "rawvideo", raw])  # fmt: skip
    thumbs = np.fromfile(raw, np.uint8)
    raw.unlink()
    scores = V.score_thumbs(thumbs)
    if len(scores) < max(V.MIN_FRAMES, V.MIN_SECONDS * V.CANDIDATE_FPS):
        raise Failed("Видеото е твърде кратко. Обиколете предмета бавно и докрай (около 30–60 секунди).")
    keep = V.pick(scores, V.frame_count(len(scores), most), V.motion(thumbs))
    del thumbs
    # Кога е заснет всеки кадър, за съпоставяне с положението на телефона (app/motion.py). Филтърът fps поставя
    # кандидат n на n / CANDIDATE_FPS с след първия кадър.
    times = {f"img_{i + 1:04d}.jpg": round(k / V.CANDIDATE_FPS, 3) for i, k in enumerate(keep)}
    (work.parent / MO.FRAME_TIMES).write_text(json.dumps(times))
    kept = scores[keep]
    note(work.parent, f"video: {len(scores)} candidate frames, kept {len(keep)}; sharpness of kept frames "
         f"min {kept.min():.0f} / median {np.median(kept):.0f}, of all frames median {np.median(scores):.0f}")  # fmt: skip
    select = "+".join(f"eq(n,{i})" for i in keep)
    # Пълен размер до 4K: текстурата чете дребен текст от тях (app/texture.py); решаването на камерите и гъстото
    # съпоставяне сами ги смаляват.
    scale = f"scale=w='trunc(iw*min(1,{TX.TEXTURE_SIDE}/max(iw,ih))/2)*2':h=-2"
    sh("picking frames", [FFMPEG, "-v", "error", "-y", "-i", video, "-an",
        "-vf", f"fps={V.CANDIDATE_FPS},select='{select}',{scale}", "-fps_mode", "vfr", "-q:v", "2",
        photos / "img_%04d.jpg"])  # fmt: skip
    return len(list(photos.glob("*.jpg")))


# идентификатор на модел на камера в COLMAP -> (брой параметри, дали има отделни fx и fy)
CAMERA_MODELS = {0: (3, False), 1: (4, True), 2: (4, False), 3: (5, False), 4: (8, True), 5: (8, True), 6: (12, True)}


def read_cameras(path: Path) -> dict[int, tuple[int, int, int, tuple[float, ...]]]:
    """(идентификатор на модел в COLMAP, ширина, височина, параметри) за всеки идентификатор на камера от cameras.bin."""
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            if model not in CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model {model}")
            count, _ = CAMERA_MODELS[model]
            cams[cid] = (model, w, h, struct.unpack(f"<{count}d", f.read(8 * count)))
    return cams


def read_cameras_bin(path: Path) -> dict[int, tuple[float, float, float, float]]:
    """fx, fy, cx, cy за всеки идентификатор на камера от cameras.bin на COLMAP. Изкривяването на обектива се
    пренебрегва: точно след корекцията (PINHOLE), достатъчно близко за оценка на решение преди нея (SIMPLE_RADIAL,
    OPENCV...)."""
    return {
        cid: (p[0], p[1], p[2], p[3]) if CAMERA_MODELS[model][1] else (p[0], p[0], p[1], p[2])
        for cid, (model, _w, _h, p) in read_cameras(path).items()
    }


def distort(model: int, p: tuple[float, ...], x: np.ndarray, y: np.ndarray):
    """Пиксел (u, v) в оригиналния кадър от нормализирани координати (x, y) за простите модели на обектив в COLMAP
    (както ги изкривява COLMAP); None за останалите."""
    r2 = x * x + y * y
    if model in (0, 2, 3):  # SIMPLE_PINHOLE f cx cy, SIMPLE_RADIAL f cx cy k, RADIAL f cx cy k1 k2
        f, cx, cy = p[:3]
        k1, k2 = (p[3:] + (0.0, 0.0))[:2]
        s = 1 + k1 * r2 + k2 * r2 * r2
        return f * x * s + cx, f * y * s + cy
    if model in (1, 4):  # PINHOLE fx fy cx cy, OPENCV fx fy cx cy k1 k2 p1 p2
        fx, fy, cx, cy = p[:4]
        k1, k2, p1, p2 = (p[4:] + (0.0,) * 4)[:4]
        s = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * s + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * s + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        return fx * xd + cx, fy * yd + cy
    return None


def undistort_masks(masks: dict[str, np.ndarray], distorted: Path, undistorted: Path):
    """Маските на предмета, завъртени така, както коригиращият инструмент на COLMAP е завъртял кадрите (от моделите на
    камерите на входа и изхода му), вместо сегментиране на коригираните кадри наново (2,5 минути от сканиране). Връща
    ({име: маска}, {име: (ширина, височина)}) или None за модел на обектив, който `distort` не познава."""
    src = read_cameras(distorted / "cameras.bin")
    src_cam = {name: cid for name, _, _, cid in read_images_bin(distorted / "images.bin")}
    dst = read_cameras(undistorted / "cameras.bin")
    out, sizes = {}, {}
    for name, _, _, cid in read_images_bin(undistorted / "images.bin"):
        model, w, h, (fx, fy, cx, cy) = dst[cid][0], dst[cid][1], dst[cid][2], dst[cid][3][:4]
        smodel, _, _, sp = src[src_cam[name]]
        if model != 1:  # коригиращият инструмент записва камери PINHOLE
            return None
        cols, rows = np.meshgrid((np.arange(max(1, w // S.MASK_SCALE)) + 0.5) * S.MASK_SCALE,
                                 (np.arange(max(1, h // S.MASK_SCALE)) + 0.5) * S.MASK_SCALE)  # fmt: skip
        uv = distort(smodel, sp, (cols - cx) / fx, (rows - cy) / fy)
        if uv is None:
            return None
        u, v = uv
        m = ndimage.map_coordinates(masks[name].astype(np.float32), [v / S.MASK_SCALE - 0.5, u / S.MASK_SCALE - 0.5],
                                    order=1, cval=0.0)  # fmt: skip
        out[name], sizes[name] = np.clip(np.rint(m), 0, 255).astype(np.uint8), (w, h)
    return out, sizes


def read_images_bin(path: Path):
    """[(име, R, t, идентификатор на камера)] от images.bin на COLMAP: световната точка X е в R X + t в камерата."""
    out = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _id, qw, qx, qy, qz, tx, ty, tz, cam = struct.unpack("<I7dI", f.read(64))
            name = bytearray()
            while (c := f.read(1)) != b"\0":
                name += c
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, 1)
            R = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            out.append((name.decode(), R, np.array([tx, ty, tz]), cam))
    return out


def read_points_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(точки (n, 3), цветове (n, 3) uint8) от points3D.bin на COLMAP: редките точки на решаващия алгоритъм."""
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    pts, cols = np.empty((n, 3)), np.empty((n, 3), np.uint8)
    at = 8
    for i in range(n):
        # идентификатор (uint64), x y z (double), r g b (uint8), грешка (double), дължина на следата (uint64), после
        # толкова двойки uint32 (идентификатор на изображение, индекс на 2D точка)
        x, y, z, r, g, b, _err, track = struct.unpack_from("<3d3BdQ", data, at + 8)
        pts[i], cols[i] = (x, y, z), (r, g, b)
        at += 8 + 43 + 8 * track
    return pts, cols


def read_views(model: Path) -> M.Views:
    cams = read_cameras_bin(model / "cameras.bin")
    imgs = read_images_bin(model / "images.bin")
    return M.Views(
        R=np.array([i[1] for i in imgs]),
        t=np.array([i[2] for i in imgs]),
        K=np.array([cams[i[3]] for i in imgs]),
        names=[i[0] for i in imgs],
    )


def mvs_indices(path: Path) -> list[int]:
    """За всяко изображение по реда в images.bin, индексът му в сцената на OpenMVS (InterfaceCOLMAP добавя
    регистрираните изображения по реда на идентификаторите в COLMAP)."""
    ids = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            ids.append(struct.unpack("<I", f.read(4))[0])
            f.seek(60, 1)
            while f.read(1) != b"\0":
                pass
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, 1)
    rank = {colmap_id: i for i, colmap_id in enumerate(sorted(ids))}
    return [rank[i] for i in ids]


def neighbor_lines(views: M.Views, index: list[int], best_deg=12.0, lo_deg=4.0, hi_deg=35.0, n=8) -> list[str]:
    """Партньорите на всеки кадър за гъсто съпоставяне, като редове за --view-neighbors-file на OpenMVS "<id> <id>
    ...". OpenMVS избира партньори по общи редки точки; при еднообразна маса те са малко и много кадри не
    получаваха нито един (без никаква дълбочина). Кадри от обиколка са добри партньори, когато гледат предмета на
    разстояние от няколко до около 35°."""
    d = views.dirs
    angle = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
    lines = []
    for i in range(len(views)):
        ok = np.flatnonzero((angle[i] >= lo_deg) & (angle[i] <= hi_deg))
        if len(ok) < 2:  # малко кадри: взимат се най-близките, които не са същият изглед
            ok = np.flatnonzero(angle[i] > 1.0)
        best = ok[np.argsort(np.abs(angle[i, ok] - best_deg))][:n]
        lines.append(" ".join(str(index[k]) for k in [i, *best]))
    return lines


def load_cloud(path: Path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    m = ms.current_mesh()
    pts, nrm, col = m.vertex_matrix(), m.vertex_normal_matrix(), m.vertex_color_matrix()
    if len(pts) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, nrm, col = pts[idx], nrm[idx], col[idx]
    has_normals = np.abs(nrm).sum() > 0
    return pts, (nrm if has_normals else None), col


def carved_hull(work: Path, dense: Path, views: M.Views, masks: list, cloud: M.Cloud, sh, found=None):
    """Обвивката от очертанията, издълбана там, където кадрите са виждали през нея: гъстите точки (измерени:
    вътрешност на чаша, отвор на дръжка) и после AI дълбочината (app/depth.py: вдлъбнатини и плоски върхове без
    гъсти точки). AI дълбочината се мащабира за всеки кадър към известните 3D точки: гъстите точки (с най-голямо
    доверие) и обвивката, където не ги надвишава. Без модела, при неуспех на стъпката или при твърде голямо
    издълбаване: обвивката, издълбана само от гъстите точки. None, ако очертанията не дават форма. found:
    резултатът от M.hull_volume, ако вече е наличен."""
    found = found if found is not None else M.hull_volume(views, masks, S.MASK_SCALE, cloud)
    outline = M.hull_mesh(*found) if found is not None else None
    if outline is None:
        return None
    frac, axes, step = found
    size = float(outline.extents.max())
    inside, probes = M.voxels_inside(frac, axes)
    probes = probes @ cloud.to_up  # обратно към света
    free = M.free_space_limit(views, cloud.points @ cloud.to_up, cloud.normals @ cloud.to_up, probes,
                              FREE_MARGIN * size)  # fmt: skip
    # Издълбаните дялове се броят във воксели: издълбаната обвивка може да не е затворена и тогава обемът ѝ не значи
    # нищо.
    frac = M.limited(frac, inside, free)
    kept = int((frac >= M.HULL_AGREE).sum())
    hull = M.hull_mesh(frac, axes, step)
    if hull is None:
        hull, frac, kept = outline, found[0], len(inside)
    note(work.parent, f"dense points see through {1 - kept / len(inside):.0%} of the outline shape")
    if not D.MODEL.exists():
        note(work.parent, f"AI depth: skipped, model missing ({D.MODEL}); install-task.ps1 downloads it")
        return hull
    samples, _, _, over = M.hull_samples(hull, cloud.points, cloud.normals)
    refs = np.vstack([cloud.points, samples[~over]]) @ cloud.to_up  # обратно към света
    weights = np.r_[np.full(len(cloud.points), 3.0), np.ones(int((~over).sum()))]
    inputs, out = work / "depth_inputs.npz", work / "depth_carve.npz"
    np.savez(inputs, R=views.R, t=views.t, K=views.K, names=np.array(views.names), refs=refs, ref_weights=weights,
             probes=probes, margin=D.CARVE_MARGIN * size, agree=M.HULL_AGREE,
             **{f"masks_{i}": m for i, m in enumerate(masks)})  # fmt: skip
    try:
        sh("estimating depth", [sys.executable, D.__file__, dense / "images", inputs, out, work / CROPS])
        with np.load(out) as z:
            limit, fits, used = z["limit"], int(z["fits"]), bool(z["used"])
    except (Failed, OSError, ValueError, KeyError) as e:
        note(work.parent, f"AI depth: failed ({e}); going on with the outline shape")
        return hull
    ai = M.limited(frac, inside, limit)
    carved = M.hull_mesh(ai, axes, step) if used else None
    share = 1 - (ai >= M.HULL_AGREE).sum() / max(kept, 1) if carved is not None else 0.0  # отвъд тези на гъстите точки
    use = carved is not None and share <= D.MAX_CARVE
    note(work.parent, f"AI depth: fits {fits} of {len(views)} frames, carves {share:.0%} more of the outline shape: "
         + ("using it" if use else "not used"))  # fmt: skip
    return carved if use else hull


CROPS = "depth_crops.npz"  # в work/: изрязванията с дълбочина от app/depth.py, за app/fuse.py
LOOSE_AGREE = 0.5  # по-свободната обвивка за app/refine.py: маските на половината кадри са съгласни
PROMPT_MARGIN = 0.04  # от размера: колко извън по-свободната обвивка измерена точка казва "не е предмет" и колко дълбоко
# в обвивката точка казва "предмет" (app/refine.py)


def sharpened_masks(work: Path, dense: Path, views: M.Views, masks: list, cloud: M.Cloud, found, pts: np.ndarray, sh):
    """Маските, направени наново от MobileSAM с подсказки от 3D (app/refine.py), или None: няма обвивка, липсва модел,
    стъпката се е провалила или е пасвала на под половината кадри (тогава остават първите маски). pts: гъстите точки,
    световни, заедно с фона."""
    if found is None:
        return None
    if not (RF.ENCODER.exists() and RF.DECODER.exists()):
        note(work.parent, f"sharpening the outlines: skipped, model missing ({RF.MODEL_DIR}); "
             "install-task.ps1 downloads it")  # fmt: skip
        return None
    frac, axes, step = found
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    tight = frac >= M.HULL_AGREE
    if not tight.any():
        return None
    size = float(np.ptp(grid[tight], axis=0).max())
    deep = tight & (ndimage.distance_transform_edt(tight) * step > PROMPT_MARGIN * size)
    loose = frac >= LOOSE_AGREE
    # Измерени точки доста извън по-свободната обвивка (и извън мрежата ѝ): масата, етикети, стаята.
    away = ndimage.distance_transform_edt(~loose) * step
    local = pts @ cloud.to_up.T
    lo = np.array([a[0] for a in axes]) - step / 2
    idx = np.floor((local - lo) / step).astype(int)
    ingrid = np.all((idx >= 0) & (idx < np.array(frac.shape)), axis=1)
    outside = np.ones(len(local), bool)
    outside[ingrid] = away[tuple(idx[ingrid].T)] > PROMPT_MARGIN * size
    outside = np.flatnonzero(outside)
    if len(outside) > 200_000:
        outside = np.random.default_rng(0).choice(outside, 200_000, replace=False)
    inputs, out = work / "refine_inputs.npz", work / "refine_masks.npz"
    np.savez(inputs, names=np.array(views.names), R=views.R, t=views.t, K=views.K, step=step,
             tight=grid[tight] @ cloud.to_up, loose=grid[loose] @ cloud.to_up, deep=grid[deep] @ cloud.to_up,
             outside=pts[outside], **{f"masks_{i}": m for i, m in enumerate(masks)})  # fmt: skip
    try:
        sh("sharpening the outlines", [sys.executable, RF.__file__, dense / "images", inputs, out, work / "sam_cache"])
        with np.load(out) as z:
            used = z["used"]
            sharp = [z[f"masks_{i}"] for i in range(len(views.names))]
    except (Failed, OSError, ValueError, KeyError) as e:
        note(work.parent, f"sharpening the outlines: failed ({e}); keeping the first masks")
        return None
    use = used.sum() >= len(views.names) / 2
    note(work.parent, f"sharpening the outlines: MobileSAM fitted {used.sum()} of {len(views.names)} frames: "
         + ("using them" if use else "keeping the first masks"))  # fmt: skip
    return sharp if use else None


def fused_surface(work: Path, dense: Path, views: M.Views, cloud: M.Cloud, hull: trimesh.Trimesh | None, sh,
                  voxel: float = F.VOXEL, flap: float | None = None, passes: int = F.PASSES):  # fmt: skip
    """Повърхността, слята от картите на дълбочината на кадрите (app/fuse.py), или None: няма обвивка или изрязвания с
    AI дълбочина, стъпката се е провалила или гъстите точки не са съгласни с нея (тогава Поасон върху гъстите
    точки)."""
    crops = work / CROPS
    if hull is None or not crops.exists():
        return None
    inputs, out = work / "fuse_inputs.npz", work / "fused.ply"
    np.savez(inputs, names=np.array(views.names), to_up=cloud.to_up, voxel=voxel, flap=F.FLAP if flap is None else flap,
             passes=passes,
             ground_z=np.nan if cloud.ground_z is None else cloud.ground_z, hull_v=hull.vertices, hull_f=hull.faces)  # fmt: skip
    try:
        sh("fusing depth", [sys.executable, F.__file__, dense, inputs, crops, out])
        info = json.loads(out.with_suffix(".json").read_text())
        mesh = trimesh.load(out, force="mesh")
    except (Failed, OSError, ValueError, KeyError) as e:
        note(work.parent, f"depth fusion: failed ({e}); using the dense points")
        return None
    size = float(np.ptp(cloud.points, axis=0).max())
    surface, _ = trimesh.sample.sample_surface(mesh, 400_000, seed=0)  # (проверка, а не моделът)
    pick = np.random.default_rng(0).choice(len(cloud.points), min(len(cloud.points), 50_000), replace=False)
    off = float(np.median(cKDTree(surface).query(cloud.points[pick], workers=-1)[0])) / size
    use = info["frames"] >= len(views) / 2 and off <= FUSED_OFF
    note(work.parent, f"depth fusion: {info['frames']} of {info['total']} frames, dense points {off:.2%} of the size "
         "off its surface: " + ("using it" if use else "not used"))  # fmt: skip
    return mesh if use else None


def run(
    job: Path,
    flat_base: bool,
    set_state: Callable[[str], None],
    sh: Callable[[str, list], None],
    quick: bool = False,
    express: bool = False,
) -> dict:
    """sh(step_name, argv) изпълнява един външен инструмент и хвърля грешка при неуспех. Връща отчета за печата.
    quick: по-малко кадри и по-малки кадри за гъсто съпоставяне, за около 40% от времето (бързото сканиране в
    приложението). express: сканирането на Pro за около 5 минути (EXPRESS_*)."""
    side = EXPRESS_FRAME_SIDE if express else QUICK_FRAME_SIDE if quick else MAX_FRAME_SIDE
    photos, video = job / "photos", job / VIDEO
    work = job / "work"
    shutil.rmtree(work, ignore_errors=True)
    TX.clear(job)  # старата текстура на повторно изпълнена задача
    dense = work / "dense"
    work.mkdir(parents=True)

    set_state("reconstructing")
    if video.exists():
        extract_frames(video, photos, work, sh, frames_to_take(job, EXPRESS_FRAMES if express else V.QUICK_FRAMES if quick else V.MAX_FRAMES))
        merge_more_frames(job, photos)
    n_photos = len(list(photos.glob("*.jp*g")))
    if quick or express:
        note(job, f"{'express' if express else 'quick'} scan: {n_photos} frames, dense matching at {side} px")

    best, placed, stray, all_masks, photo_masks = solve_cameras(job, work, photos, n_photos, sh)
    if placed < SFM_ENOUGH * n_photos and MO.load(job) is not None:
        focal = float(np.median(read_views(best).K[:, 0])) if placed >= 10 else 0.0
        found = by_motion(job, work, photo_masks, focal, placed, n_photos, flat_base, sh)
        if found is not None:
            mesh, used = found
            width = float(np.median([S.object_width(m) for m in all_masks.values()]))
            return finish(job, mesh) | {"photos_used": used, "photos_total": n_photos, "object_width": round(width, 3)}
    if best is None or placed < max(10, n_photos // 3):
        note(job, f"only {placed} of {n_photos} frames placed in 3D")
        raise Failed(
            "Кадрите не можаха да се съпоставят: видеото е твърде бързо или размазано, или предметът и подложката "
            "нямат достатъчно шарка. Снимайте по-бавно, дръжте предмета в кръга и го сложете на подложка с шарка в "
            "различен цвят от него."
        )
    best = drop_stray(work, best, stray, sh)

    # Гъстото съпоставяне получава кадрите в пълен размер: малък предмет често е широк само няколкостотин пиксела.
    sh("undistorting", [COLMAP, "image_undistorter", "--image_path", photos, "--input_path", best,
        "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(side)])  # fmt: skip
    masks_file, dense_masks = work / "masks.npz", work / "dense_masks"
    turned = undistort_masks(all_masks, best, dense / "sparse")
    if turned is None:
        sh("finding the object", [sys.executable, S.__file__, dense / "images", masks_file, dense_masks])
    else:
        warped, sizes = turned
        np.savez_compressed(masks_file, **warped)
        dense_masks.mkdir()
        for name, m in warped.items():
            Image.fromarray(S.dense_area(m, sizes[name])).save(dense_masks / f"{Path(name).stem}.mask.png")
    sh("dense setup", ["InterfaceCOLMAP", "-w", dense, "-i", dense, "-o", dense / "scene.mvs",
        "--image-folder", dense / "images"])  # fmt: skip
    neighbors = dense / "neighbors.txt"
    lines = neighbor_lines(read_views(dense / "sparse"), mvs_indices(dense / "sparse" / "images.bin"))
    neighbors.write_text("\n".join(lines) + "\n")
    # Най-бавната стъпка. Съпоставя се само областта около предмета (маските), което прави пълната разделителна
    # способност достъпна: при половината малък предмет получаваше няколко хиляди точки и формата му идваше само от
    # очертанието. Съседните видео кадри силно се припокриват, затова стигат 4 изгледа на карта и един геометричен
    # проход. Без изрязване на област на интерес: маските вече ограничават работата, а собствената оценка на OpenMVS
    # веднъж отряза всичко. Картите на дълбочината остават за app/fuse.py (изтриват се след него).
    sh("dense matching", ["DensifyPointCloud", "-w", dense, "-i", dense / "scene.mvs", "-o", dense / "dense.mvs",
        "--view-neighbors-file", neighbors, "--mask-path", dense_masks, "--ignore-mask-label", "0",
        "--resolution-level", "0", "--max-resolution", str(side), "--number-views", "4",
        "--geometric-iters", "1", "--estimate-roi", "0", "--crop-to-roi", "0", "--remove-dmaps", "0",
        "--max-threads", THREADS])  # fmt: skip

    set_state("meshing")
    ply = dense / "dense.ply"
    if not ply.exists():
        raise Failed("Формата на предмета не можа да се измери. Може да е твърде лъскав, едноцветен или размазан на кадрите.")
    pts, nrm, col = load_cloud(ply)
    note(job, f"dense cloud: {len(pts)} points")
    views = read_views(dense / "sparse")
    with np.load(masks_file) as npz:
        masks = [npz[name] for name in views.names]
    width = float(np.median([S.object_width(m) for m in masks]))
    note(job, f"object width: median {width:.0%} of the frame")
    up = gravity_up(job, views)
    cloud = M.clean_cloud(pts, nrm, col, views, masks, S.MASK_SCALE, up)
    note(job, f"object cloud: {len(cloud.points)} points, table found: {cloud.ground_z is not None}")
    found = M.hull_volume(views, masks, S.MASK_SCALE, cloud)
    sharp = None if express else sharpened_masks(work, dense, views, masks, cloud, found, pts, sh)
    if sharp is not None:  # оттук нататък всичко с по-добрите маски
        masks = sharp
        cloud = M.clean_cloud(pts, nrm, col, views, masks, S.MASK_SCALE, up)
        note(job, f"object cloud: {len(cloud.points)} points, table found: {cloud.ground_z is not None}")
        found = M.hull_volume(views, masks, S.MASK_SCALE, cloud)
    # Гъстите точки там, където ги има, а (издълбаната) форма от очертанията в празнините им. При кутия това даде
    # плоски страни, остри ръбове и верни пропорции; с добавени и точки от AI дълбочина страните бяха груби с ръбове,
    # а кутията с една трета по-плоска (височина 0,65 от дължината; гъстите ѝ точки казват 0,96).
    hull = carved_hull(work, dense, views, masks, cloud, sh, found)
    note(job, "visual hull: " + (f"{len(hull.faces)} faces" if hull is not None else "none, using the points only"))
    # Сливането получава формата от очертанията неиздълбана: издълбава сама, пиксел по пиксел. Издълбаването от
    # гъстите точки отрязваше предната част на черната дръжка на дрелка (малко шумни точки там), а сливането се държи
    # близо до обвивката си.
    outline = M.hull_mesh(*found) if found is not None else None
    mesh = fused_surface(work, dense, views, cloud, outline, sh, EXPRESS_VOXEL if express else F.VOXEL)
    for p in dense.glob("*.dmap"):  # по 40 MB
        p.unlink(missing_ok=True)
    if mesh is None:
        mesh = M.poisson(cloud, hull)

    set_state("cleaning")
    mesh = M.clip_to_outline(mesh, views, masks, S.MASK_SCALE, cloud)
    M.paint(mesh, views, [dense / "images" / n for n in views.names], cloud.to_up)
    out = M.make_printable(mesh, M.floor_z(cloud), flat_base)
    photo_texture(job, work, photos, best, out, cloud, sh, all_masks)
    # Приложението предлага заснемане отблизо, когато предметът е заемал малка част от кадрите (по-малко детайли).
    return finish(job, out) | {"photos_used": placed, "photos_total": n_photos, "object_width": round(width, 3)}


def photo_texture(job: Path, work: Path, photos: Path, best: Path, out: trimesh.Trimesh, cloud: M.Cloud, sh,
                  all_masks: dict[str, np.ndarray]):  # fmt: skip
    """Текстурата от снимки на модела за печат (app/texture.py); при неуспех той запазва цветовете на върховете си."""
    try:
        info = TX.texture_model(job, work, photos, best, out, out.metadata["to_print"], cloud.to_up, sh, COLMAP,
                                THREADS, read_views(best).names, all_masks)  # fmt: skip
        note(job, f"photo texture: from the sharpest {info['frames']} frames, {info['textures']} pixels, "
             f"{info['unseen']:.1%} of the faces seen by no frame (painted); seams levelled: {info['levelled']}")  # fmt: skip
    except (Failed, OSError, ValueError, KeyError) as e:
        TX.clear(job)
        note(job, f"photo texture: failed ({e}); the model keeps its vertex colours")


def drop_stray(work: Path, best: Path, stray: list[str], sh, name: str = "clean") -> Path:
    """Моделът на камерите без грешно разположените кадри (те биха развалили гъстите точки и биха издълбали формата от
    очертанията), в work/sparse/<name>."""
    if not stray:
        return best
    listed = work / f"{name}_dropped.txt"
    listed.write_text("\n".join(stray) + "\n")
    clean = work / "sparse" / name
    clean.mkdir()
    sh("camera solving", [COLMAP, "image_deleter", "--input_path", best, "--output_path", clean,
        "--image_names_path", listed])  # fmt: skip
    return clean


PAIR_NEAR = 5  # (съпоставяне по двойки) всеки кадър със следващите толкова, в реда на заснемане
PAIR_EVERY = 2  # и всеки толкова-ти кадър с всеки друг такъв (2, не 3: двойно повече връзки между нисък и висок кръг)


def match_pairs(names: list[str], near: int = PAIR_NEAR, every: int = PAIR_EVERY) -> list[tuple[str, str]]:
    """Двойките кадри за съпоставяне вместо всички (кадри в реда на заснемане): всеки със следващите `near`, което
    свързва видеото, и всеки `every`-ти с всеки друг такъв, което затваря кръга и свързва високия кръг с ниския. 48
    кадъра: около 450 двойки вместо 1128 (съпоставянето е около 0,1 с на двойка на сървъра)."""
    pairs = {(i, j) for i in range(len(names)) for j in range(i + 1, min(len(names), i + near + 1))}
    spread = range(0, len(names), every)
    pairs |= {(i, j) for i in spread for j in spread if i < j}
    return [(names[i], names[j]) for i, j in sorted(pairs)]


def solve_cameras(job: Path, work: Path, photos: Path, n_photos: int, sh: Callable[[str, list], None],
                  paired: bool = False, mask_every: int = 1, features: int | None = None):  # fmt: skip
    """Откъде е заснет всеки кадър (COLMAP) и предметът във всеки кадър (AI маски, app/segment.py). paired: съпоставя
    двойките от match_pairs, а не всеки кадър с всеки (повече кадри за същото време). mask_every: изрязва предмета
    само от всеки толкова-ти кадър; останалите само помагат да се разположат и се броят за отклонени. Връща (папката
    на най-добрия модел на камерите или None, колко от кадрите му с маска са съгласувани с очертанията на предмета,
    кадрите, които не са или нямат маска, {име на кадър: маска}, файла на маските). features: най-много на кадър,
    вместо 8192 на COLMAP."""
    sparse, db = work / "sparse", work / "db.db"
    sparse.mkdir(parents=True, exist_ok=True)
    ext, match = colmap_prefixes()
    sh("feature extraction", [COLMAP, "feature_extractor", "--database_path", db, "--image_path", photos,
        "--ImageReader.single_camera", "1", f"--{ext}.use_gpu", "0",
        f"--{ext}.max_image_size", MAX_IMAGE, f"--{ext}.num_threads", THREADS,
        "--SiftExtraction.peak_threshold", "0.002",
        *(["--SiftExtraction.max_num_features", str(features)] if features else [])])  # fmt: skip
    # По-нисък праг на пика: повече особености по еднообразни повърхности (велурена възглавница: 80 от 80 разположени
    # кадъра вместо 70). Всеки кадър с всеки друг: затваря кръга и свързва ниския кръг с високия. Без насочвано
    # съпоставяне: на процесор отнемаше над 12 мин за 20 кадъра.
    names = sorted(p.name for p in photos.glob("*.jp*g"))
    if paired:
        pair_list = work / "pairs.txt"
        pair_list.write_text("".join(f"{a} {b}\n" for a, b in match_pairs(names)))
        sh("photo matching", [COLMAP, "matches_importer", "--database_path", db, "--match_list_path", pair_list,
            "--match_type", "pairs", f"--{match}.use_gpu", "0", f"--{match}.num_threads", THREADS])  # fmt: skip
    else:
        sh("photo matching", [COLMAP, "exhaustive_matcher", "--database_path", db, f"--{match}.use_gpu", "0",
            f"--{match}.num_threads", THREADS])  # fmt: skip
    # Глобалният алгоритъм (COLMAP 4) решава всички кадри наведнъж: при едноцветна възглавница разположи 73 от 80
    # кадъра, докато инкременталният ги раздели на части от 44, 11 и 10. Инкременталният все още работи като резервен.
    # Два алгоритъма, защото всеки се проваля при някои видеа: инкременталният разделя сканирането на части, когато
    # кръговете имат малко общи особености; глобалният (COLMAP 4) разполага повече кадри, но при едноцветна
    # възглавница веднъж върна модел, чиито камери дори не гледаха към предмета. Всеки модел се оценява по кадрите,
    # които наистина сочат към предмета.
    (sparse / "incremental").mkdir()
    (sparse / "global").mkdir()
    # Съседните видео кадри са само на няколко градуса, затова се позволява по-малък ъгъл за първата двойка и се
    # приемат кадри с по-малко (но проверени) съвпадения, защото размазването струва особености.
    sh("camera solving", [COLMAP, "mapper", "--database_path", db, "--image_path", photos,
        "--output_path", sparse / "incremental", "--Mapper.num_threads", THREADS,
        "--Mapper.init_min_tri_angle", "8", "--Mapper.abs_pose_min_num_inliers", "20"])  # fmt: skip
    pieces = [read_views(d) for d in sorted((sparse / "incremental").iterdir()) if (d / "images.bin").exists()]
    one_piece = len(pieces) == 1 and len(pieces[0]) >= ONE_PIECE * n_photos
    if one_piece:
        note(job, f"camera solving: one piece with {len(pieces[0])} of {n_photos} frames; global solver not needed")
    if has_global_mapper() and not one_piece:
        try:
            # Видео кадрите нямат фокусно разстояние; глобалният алгоритъм изисква то първо да се оцени.
            sh("camera solving", [COLMAP, "view_graph_calibrator", "--database_path", db])
            sh("camera solving", [COLMAP, "global_mapper", "--database_path", db, "--image_path", photos,
                "--output_path", sparse / "global"])  # fmt: skip
        except Failed:
            note(job, "global mapper failed; using the incremental one")
    if not one_piece:
        for family in ("incremental", "global"):
            merge_pieces(job, work, sparse, family, sh)

    # Всяко решение се оценява по AI маските: при верни положения формата от очертанията от всички кадри съвпада с
    # маската на всеки кадър; несъвпадащите са грешно разположени и отпадат.
    photo_masks = work / "photo_masks.npz"
    masked = photos
    if mask_every > 1:
        masked = work / "masked"
        masked.mkdir(exist_ok=True)
        for name in names[::mask_every]:
            shutil.copy(photos / name, masked / name)
    sh("finding the object", [sys.executable, S.__file__, masked, photo_masks])
    with np.load(photo_masks) as npz:
        all_masks = {name: npz[name] for name in npz.files}
    models = []
    for d in sorted(p for p in sparse.rglob("*") if (p / "images.bin").exists()):
        views = read_views(d)
        has = np.array([n in all_masks for n in views.names])
        ok = np.zeros(len(views), bool)
        if has.sum() >= 3:
            ok[has] = M.consistent_frames(M.subset(views, has), [all_masks[n] for n in views.names if n in all_masks],
                                          S.MASK_SCALE)  # fmt: skip
        models.append((d, int(ok.sum()), [n for n, good in zip(views.names, ok) if not good]))
        note(job, f"camera solving: {d.parent.name} model {d.name}: {len(views)} frames, {ok.sum()} of "
             f"{int(has.sum())} with the object cut out consistent with its outline")  # fmt: skip
    best, placed, stray = max(models, key=lambda m: m[1], default=(None, 0, []))
    return best, placed, stray, all_masks, photo_masks


def refine_cameras(job: Path, work: Path, model: Path, sh: Callable[[str, list], None]) -> Path:
    """Моделът на камерите, уточнен още веднъж (bundle_adjuster на COLMAP); моделът какъвто е бил, ако това се
    провали."""
    out = work / "sparse" / "refined"
    out.mkdir(parents=True, exist_ok=True)
    try:
        sh("camera solving", [COLMAP, "bundle_adjuster", "--input_path", model, "--output_path", out])
    except Failed as e:
        note(job, f"camera solving: refining failed ({e})")
        return model
    return out


def merge_pieces(job: Path, work: Path, sparse: Path, family: str, sh: Callable[[str, list], None]):
    """Един модел на камерите от частите, на които алгоритъм е разделил кадрите, когато частите имат общи кадри:
    model_merger на COLMAP съединява две части по общите им кадри (първо най-голямата, после всяка друга поред), а
    bundle_adjuster уточнява цялото. Дрелка, заснета в два кръга на различни разстояния, излезе на части от 35 и 24 от
    48-те кадъра, като 11 бяха и в двете (задача f4e8): нито една част не я показваше отвсякъде. Съединеният модел се
    записва в sparse/merged-<family>/0 и се оценява по очертанията на предмета като собствените на алгоритмите
    (solve_cameras), така че се използва само когато им пасва по-добре."""
    folder = sparse / family
    pieces = sorted((d for d in folder.iterdir() if (d / "images.bin").exists()), key=lambda d: -len(read_views(d)))
    if len(pieces) < 2:
        return
    steps = work / f"merging-{family}"
    steps.mkdir(exist_ok=True)
    current, joined = pieces[0], [len(read_views(pieces[0]))]
    for k, other in enumerate(pieces[1:]):
        step = steps / str(k)
        step.mkdir(exist_ok=True)
        try:
            sh("camera solving", [COLMAP, "model_merger", "--input_path1", current, "--input_path2", other,
                "--output_path", step])  # fmt: skip
        except Failed:  # няма общи кадри: тази част остава отделно
            continue
        if (step / "images.bin").exists() and len(read_views(step)) > len(read_views(current)):
            current = step
            joined.append(len(read_views(other)))
    if current == pieces[0]:
        note(job, f"camera solving: the {family} pieces share no frames; not joined")
        return
    out = sparse / f"merged-{family}" / "0"
    out.mkdir(parents=True, exist_ok=True)
    try:
        sh("camera solving", [COLMAP, "bundle_adjuster", "--input_path", current, "--output_path", out])
    except Failed:
        shutil.rmtree(out, ignore_errors=True)
        note(job, f"camera solving: joined the {family} pieces, but refining them failed; not used")
        return
    note(job, f"camera solving: joined {len(joined)} {family} pieces ({' + '.join(map(str, joined))} frames) into one "
         f"of {len(read_views(out))} frames")  # fmt: skip


def gravity_up(job: Path, views: M.Views) -> np.ndarray | None:
    """Посоката нагоре от сензора за гравитация на телефона (app/motion.py), ако кадрите са съгласни; иначе None
    (предположение от камерите)."""
    found = MO.gravity_up(job, views)
    if found is None:
        return None
    up, spread = found
    guess = M.estimate_up(views.centers, views.dirs, views.ups)
    off = np.degrees(np.arccos(np.clip(up @ guess / np.linalg.norm(guess), -1, 1)))
    use = spread <= GRAVITY_AGREE
    note(job, f"up from the phone's gravity sensor: frames agree within {spread:.1f}°, {off:.0f}° from the cameras' "
         "guess: " + ("using it" if use else "not used"))  # fmt: skip
    return up if use else None


def finish(job: Path, mesh: trimesh.Trimesh) -> dict:
    mesh.export(job / "model.ply")
    shutil.rmtree(job / "work", ignore_errors=True)
    (job / VIDEO).unlink(missing_ok=True)  # най-големият файл на задачата; извлечените кадри остават за отстраняване на грешки
    return M.report(mesh)


def by_motion(job: Path, work: Path, photo_masks: Path, focal: float, placed: int, n_photos: int, flat_base: bool,
              sh: Callable[[str, list], None]) -> tuple[trimesh.Trimesh, int] | None:  # fmt: skip
    """Формата от положението на телефона и очертанията (app/motion.py), ако очертанията са съгласувани със сензорите
    по-добре, отколкото с решаващия алгоритъм. Връща (мрежа за печат, използвани кадри) или None, за да се продължи с
    алгоритъма."""
    out = work / "sensor_shape.ply"
    try:
        sh("shape from motion sensors", [sys.executable, MO.__file__, job, photo_masks, f"{focal:.1f}", out,
            "1" if flat_base else "0"])  # fmt: skip
        info = json.loads(out.with_suffix(".json").read_text())
    except (Failed, OSError, ValueError) as e:
        note(job, f"sensor poses: failed ({e}); going on with the camera solver's frames")
        return None
    score = info["score"]
    use = out.exists() and (score >= MO.GOOD or (score >= MO.USABLE and placed < n_photos / 2))
    note(job, f"sensor poses: outlines agree {score:.0%} over {info['frames']} frames filmed from {info['around']}° "
         f"around (camera solver placed {placed} of {n_photos}): " + ("using them" if use else "not used"))  # fmt: skip
    return (trimesh.load(out, force="mesh"), info["frames"]) if use else None
