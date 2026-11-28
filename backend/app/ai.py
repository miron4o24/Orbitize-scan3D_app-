"""ИИ моделът от видео (или снимки), вместо обработката на сканиране (app/pipeline.py), изцяло на сървъра, без
външни услуги: кадрите се разполагат в 3D (COLMAP), ИИ изрязва предмета от тях, а измерената между тях дълбочина и ИИ
дълбочината се сливат в тези очертания и снимките стават текстура (app/outline.py). Резервен път, когато кадрите не
могат да се разположат: TripoSR от един кадър (app/single3d.py, локален модел; гърбът е предположен)."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image, ImageOps
from scipy import ndimage

from . import mesh as M
from . import outline as OL
from . import pipeline as P
from . import segment as S
from . import single3d
from . import texture as TX

METHODS = ("single",)
CANDIDATES = 120  # кадри, прочетени от видеото за резервния път


def available() -> dict:
    """Какво може да се използва на този сървър; express: сканирането (app/pipeline.py), edit: нов модел от
    съществуващ (табелка с име, app/edit.py)."""
    return {"express": True, "edit": True, "single": single3d.ready()}


# --- Кадри ----------------------------------------------------------------------------------------------------------


def duration(video: Path) -> float:
    r = subprocess.run([P.FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True, text=True, errors="replace")
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def sharpness(path: Path) -> float:
    with Image.open(path) as im:
        g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((320, 320)), float)
    return float(ndimage.laplace(g).var())


def frames(job: Path, sh: Callable[[str, list], None]) -> list[Path]:
    """Кадрите от видеото (до CANDIDATES, равномерно разпределени) като photos/img_0001.jpg... или качените снимки."""
    photos, video = job / "photos", job / P.VIDEO
    if video.exists():
        for p in photos.glob("*.jp*g"):  # повторен опит чете видеото отново
            p.unlink()
        seconds = duration(video)
        fps = min(4.0, CANDIDATES / seconds) if seconds > 0 else 2.0
        scale = "scale=w='trunc(iw*min(1,1600/max(iw,ih))/2)*2':h=-2"
        sh("reading video", [P.FFMPEG, "-v", "error", "-y", "-i", video, "-an", "-vf", f"fps={fps:.4f},{scale}",
            "-frames:v", str(CANDIDATES), "-q:v", "3", photos / "img_%04d.jpg"])  # fmt: skip
    found = sorted(photos.glob("*.jp*g"))
    if not found:
        raise P.Failed("Във видеото няма кадри, които да се прочетат. Опитайте с друго видео.")
    return found


def spread(found: list[Path], count: int) -> list[Path]:
    """Най-рязкият кадър от всяка от `count` равни групи: предметът отвсякъде, без размазване."""
    groups = np.array_split(np.arange(len(found)), min(count, len(found)))
    return [found[max(g, key=lambda i: sharpness(found[i]))] for g in groups if len(g)]


# --- Безплатният AI модел: AI очертания и дълбочина от цялото видео (app/outline.py)
# ---------------------------------

# Кадрите, които решаващият алгоритъм за камерите разполага: два бързи кръга за 33 с дадоха 24 кадъра през 30° и не
# повече от 4 можеха да се разположат заедно; 48 са през около 15°, като при бързо сканиране (app/video.py). Всеки се
# съпоставя със съседите си и с рядък набор през видеото (pipeline.match_pairs), а предметът се изрязва от всеки:
# формата се измерва и слива от всичките 48 (от всеки втори главата и тялото на Фродо излизаха на бучки, а ръката и
# мечът едно петно: всяко място е виждано от двойно по-малко кадри и шумът на дълбочината му се осреднява двойно
# по-малко). Потребителят избра детайла пред времето (2026-11-01), а после 50 вместо 48 (2026-11-17): около 4% повече
# време; 100 би удвоило времето (над 10 минути) без да поправи тънките части, които ограничава дълбочината.
AI_FRAMES = 50
MASK_EVERY = 1
MIN_FRAMES = 10  # извлечени са по-малко кадри: TripoSR само от един кадър
# Правилно разположени кадри (съгласувани с очертанията на предмета), от които се прави формата; при по-малко: TripoSR
# от един кадър. Брой, а не дял от изрязаните кадри: при изрязване на всеки втори половината от 24 бяха 12;
# изрязването на всичките 48 го направи 24, а дрелка, чиито два кръга излязоха на две части (най-добрата част: 19
# правилно разположени), отиде при модела от един кадър, петно със залепена дръжка на куфар (задача f4e8, 2026-11-01).
MIN_PLACED = 12
SINGLE_CANDIDATES = 8  # (само един кадър) кадри, в които се търси предметът; моделира се най-добрият


def single(job: Path, work: Path, flat_base: bool, set_state: Callable[[str], None],
           sh: Callable[[str, list], None], tune: dict | None = None) -> dict:  # fmt: skip
    """AI моделът от цялото видео; TripoSR от един кадър, когато кадрите не могат да се разположат в 3D (много кратко
    или размазано видео, едноцветен предмет върху едноцветна маса)."""
    photos, video = job / "photos", job / P.VIDEO
    if video.exists():
        try:
            P.extract_frames(video, photos, work, sh, P.frames_to_take(job, AI_FRAMES))
        except P.Failed as e:  # твърде кратко за решаващия алгоритъм за камерите; TripoSR иска един кадър
            P.note(job, f"AI model: {e}")
        P.merge_more_frames(job, photos)
    found = sorted(photos.glob("*.jp*g"))
    if len(found) >= MIN_FRAMES:
        report = all_frames(job, work, photos, len(found), flat_base, set_state, sh, tune)
        if report is not None:
            return report
    if not found:
        found = frames(job, sh)
    set_state("reconstructing")
    model, info = by_single(job, work, found, sh)
    set_state("cleaning")
    out = M.make_printable(model, float(model.bounds[0, 2]), flat_base)
    return P.finish(job, out) | {"photos_used": 1, "photos_total": len(found), "ai": "single"} | info


def all_frames(job: Path, work: Path, photos: Path, n_photos: int, flat_base: bool,
               set_state: Callable[[str], None], sh: Callable[[str, list], None], tune: dict | None) -> dict | None:
    """Моделът от всички кадри (from_video) или None (тогава от един кадър). Спрял ли е от недостиг на памет (няколко
    AI модела наведнъж на сървъра), още веднъж с един модел наведнъж: по-бавно, но от всички кадри, а не от един."""
    for attempt in (0, 1):
        start = P.log_size(job)
        try:
            return from_video(job, work, photos, n_photos, flat_base, set_state, sh, tune)
        except Exception as e:  # каквото и да се е объркало, моделът от един кадър пак си заслужава
            if type(e).__name__ == "Cancelled":  # от app/jobs.py (импортирането му тук би било циклично)
                raise
            TX.clear(job)  # текстура, направена за този модел, ако се е стигнало дотам
            if attempt == 0 and P.ran_out_of_memory(job, start) and os.environ.get("PAR_MODELS") != "1":
                P.note(job, f"AI model from all the frames: out of memory ({e!r}); again with one AI model at a time")
                os.environ["PAR_MODELS"] = "1"  # за стъпките, които се пускат като отделни процеси
                shutil.rmtree(work, ignore_errors=True)
                work.mkdir(parents=True)
                continue
            P.note(job, f"AI model from all the frames: failed ({e!r}); making it from one frame")
            return None
        finally:
            if attempt == 1:
                os.environ.pop("PAR_MODELS", None)
    return None


def from_video(job: Path, work: Path, photos: Path, n_photos: int, flat_base: bool,
               set_state: Callable[[str], None], sh: Callable[[str, list], None],
               tune: dict | None = None) -> dict | None:  # fmt: skip
    """Кадрите, разположени в 3D; предметът, изрязан от тях с AI и уточнен с MobileSAM; формата вътре във всички тези
    очертания; измерената между кадрите дълбочина (OpenMVS) с AI дълбочина (app/depth.py), слети върху нея
    (app/fuse.py): релефът, който очертанията не показват (ръб на основа, ръце, гънки, коса); текстурата от снимки
    върху него (app/outline.py). None, когато твърде малко кадри могат да се разположат."""
    best, placed, stray, all_masks, _ = P.solve_cameras(job, work, photos, n_photos, sh, paired=True,
                                                        mask_every=MASK_EVERY, features=(tune or {}).get("features"))  # fmt: skip
    return model_from_solved(job, work, photos, flat_base, set_state, sh, best, placed, stray, all_masks, tune)


ROUNDS = 2  # проходи на MobileSAM, всеки с подсказки от формата на очертанията от предишния


def model_from_solved(job: Path, work: Path, photos: Path, flat_base: bool, set_state: Callable[[str], None],
                      sh: Callable[[str, list], None], best: Path | None, placed: int, stray: list[str],
                      all_masks: dict[str, np.ndarray], tune: dict | None = None) -> dict | None:  # fmt: skip
    """from_video след камерите: разположени кадри (`placed` от тях съгласувани с очертанията на търсача на предмета,
    `stray` несъгласувани) и маските на търсача. Те решават само откъде да се започне. При дрелка върху куфар той
    включваше куфара, дръжката му, стол или кутия отзад или пропускаше дръжката или патронника в 26 от 48 кадъра с
    напълно верни камери (едно цяло, задача 988c): съдени по тези маски, 26-те отпадаха, а формата се правеше от 22
    кадъра, някои от тях все още грешни, с издълбана до стъбло дръжка. Затова формата от съгласуваните кадри дава
    подсказки на MobileSAM за всеки разположен кадър (outlines_from_3d, ROUNDS пъти) и отпадат само кадри, чиито
    уточнени очертания все още се разминават с другите."""
    if best is None or placed < MIN_PLACED:
        P.note(job, f"AI model: only {placed} of the {len(all_masks)} frames with the object cut out placed in 3D; "
               "making it from one frame")  # fmt: skip
        return None
    tune = {k: v for k, v in (tune or {}).items() if v}  # Tune от app/main.py: настройките на тестова задача
    if tune:
        P.note(job, f"AI model: test settings {tune}")
    # Същите 14 минути, по-остър печат: при дрелка логото и етикетът излязоха по-ясни, с по-малко невидени стени (11%
    # -> 7%).
    if not tune.get("no_refine"):
        best = P.refine_cameras(job, work, best, sh)
    side = tune.get("side", OL.SIDE)
    solved = best
    # Един размер за всичко след това, както при експресно сканиране: картите на дълбочината, AI дълбочината и маските
    # трябва да делят пикселите си (AI дълбочина при 1920 спрямо карти при 1280 не пасваше на нито един кадър).
    dense = work / "dense"
    sh("undistorting", [P.COLMAP, "image_undistorter", "--image_path", photos, "--input_path", best,
        "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(side)])  # fmt: skip
    views = P.read_views(dense / "sparse")
    turned = P.undistort_masks(all_masks, best, dense / "sparse")
    if turned is None:
        masks_file = work / "undistorted_masks.npz"
        sh("finding the object", [sys.executable, S.__file__, dense / "images", masks_file])
        with np.load(masks_file) as z:
            masks = [z[n] for n in views.names]
    else:
        masks = [turned[0][n] for n in views.names]
    agreed = np.array([n in all_masks and n not in stray for n in views.names])

    up = P.gravity_up(job, views)
    pts, cols = P.read_points_bin(dense / "sparse" / "points3D.bin")
    first = M.subset(views, agreed)
    first_masks = [m for m, a in zip(masks, agreed) if a]
    try:
        cloud = M.clean_cloud(pts, None, np.c_[cols / 255.0, np.ones(len(cols))], first, first_masks, S.MASK_SCALE, up)
        P.note(job, f"AI model: {len(views)} frames placed, {placed} agreeing with the object finder's outlines, "
               f"{len(cloud.points)} of the solver's points on the object")  # fmt: skip
    except M.MeshError:  # твърде малко точки върху едноцветен предмет: само очертанията го разполагат
        cloud = outline_cloud(first, first_masks, up)
        if cloud is None:
            raise P.Failed("the frames' outlines give no shape")
        P.note(job, f"AI model: {len(views)} frames placed, {placed} agreeing, few points on the object")

    masks, keep, sharpened = outlines_from_3d(job, work, dense, views, masks, agreed, cloud, pts, sh)
    if keep.sum() < MIN_PLACED:
        P.note(job, f"AI model: only {keep.sum()} frames agree on the object's outline; making it from one frame")
        return None
    if not keep.all():  # кадри с очертания, които още се разминават: извън гъстото съпоставяне и формата
        dropped = [n for n, k in zip(views.names, keep) if not k]
        kept = P.drop_stray(work, dense / "sparse", dropped, sh, "outline_kept")
        shutil.rmtree(dense / "sparse")
        shutil.copytree(kept, dense / "sparse")
        views, masks = M.subset(views, keep), [m for m, k in zip(masks, keep) if k]
    images = [dense / "images" / n for n in views.names]
    agree = tune.get("agree") or (OL.AGREE_SHARP if sharpened else M.HULL_AGREE)

    surface, cloud = OL.measured(job, work, dense, views, masks, cloud, up, agree, set_state, sh, side=side,
                                 voxel=tune.get("voxel", OL.VOXEL), fixed="agree" in tune,
                                 table=not tune.get("no_table"), flap=tune.get("flap"), wide=tune.get("wide_dense", False),
                                 geometric=not tune.get("no_geo"), passes=tune.get("passes"))  # fmt: skip
    set_state("cleaning")
    M.paint(surface, views, images, cloud.to_up)
    out = M.make_printable(surface, M.floor_z(cloud), flat_base)
    used = set(views.names)
    posed, texture_masks = texture_frames(work, solved, [n for n in all_masks if n not in used], all_masks, sh)
    P.photo_texture(job, work, photos, posed, out, cloud, sh, texture_masks)
    width = float(np.median([S.object_width(m) for m in masks]))
    return P.finish(job, out) | {"photos_used": len(views), "photos_total": len(all_masks),
                                 "object_width": round(width, 3), "ai": "single", "object": "", "suitable": True,
                                 "real_mm": None}  # fmt: skip


def outlines_from_3d(job: Path, work: Path, dense: Path, views: M.Views, masks: list[np.ndarray], agreed: np.ndarray,
                     cloud: M.Cloud, pts: np.ndarray, sh) -> tuple[list[np.ndarray], np.ndarray, bool]:  # fmt: skip
    """Очертанията на предмета във всеки разположен кадър (коригирани `views`, `masks` на търсача): формата от кадрите
    със съгласувани очертания (първоначално `agreed`) дава подсказки на MobileSAM за всички
    (pipeline.sharpened_masks); кадрите с нови очертания, съгласувани с другите (mesh.consistent_frames), дават
    формата за следващия кръг. Връща (маските, кои кадри са съгласувани, дали се използват тези на MobileSAM);
    собствените на търсача, когато MobileSAM не може да се използва."""
    current, trusted, sharpened = list(masks), agreed.copy(), False
    for round_ in range(ROUNDS):
        idx = np.flatnonzero(trusted)
        found = M.hull_volume(M.subset(views, trusted), [current[i] for i in idx], S.MASK_SCALE, cloud)
        if found is None:
            break
        sharp = P.sharpened_masks(work, dense, views, current, cloud, found, pts, sh)
        if sharp is None:
            break
        keep = M.consistent_frames(views, sharp, S.MASK_SCALE)
        P.note(job, f"AI model: outlines round {round_ + 1}: {keep.sum()} of {len(views)} frames agree "
               f"(before: {trusted.sum()})")  # fmt: skip
        if keep.sum() < MIN_PLACED:  # очертанията на MobileSAM са твърде несъгласувани: остават последните
            break
        current, trusted, sharpened = sharp, keep, True
    return current, trusted, sharpened


def texture_frames(work: Path, solved: Path, stray: list[str], masks: dict[str, np.ndarray],
                   sh: Callable[[str, list], None]) -> tuple[Path, dict[str, np.ndarray]]:  # fmt: skip
    """Кадрите, от които се прави текстурата: тези с изрязан и правилно разположен предмет, а също и неизрязаните
    между тях (при MASK_EVERY > 1), чиито съседи от двете страни във видеото са правилно разположени: двойно повече
    кадри, така че всяка стена получава по-прав и по-близък изглед (Фродо: текстурата идваше от 19 от 48 кадъра, а
    6,7% от стените не се виждаха от нито един). Кадрите между тях взимат изрязването на съсед, за да се измери
    рязкостта им. Когато всеки кадър е изрязан, това са правилно разположените кадри. Връща (модела на камерите на
    тези кадри, {кадър: маска})."""
    names = P.read_views(solved).names
    placed = set(names)
    good = {n for n in names if n in masks and n not in stray}
    order = sorted(placed | set(masks))
    use, borrowed = set(good), dict(masks)
    for i, n in enumerate(order):
        if n in masks or not 0 < i < len(order) - 1:
            continue
        before, after = order[i - 1], order[i + 1]
        if before in good and after in good:
            use.add(n)
            borrowed[n] = masks[before]
    return P.drop_stray(work, solved, [n for n in names if n not in use], sh, "texture_frames"), borrowed


def outline_cloud(views: M.Views, masks: list[np.ndarray], up: np.ndarray | None) -> M.Cloud | None:
    """Заместител на точките от решаващия алгоритъм, когато малко от тях са върху предмета: вокселите на формата от
    очертанията (груби), завъртени така, че z да е нагоре. Маса не е известна; най-ниската точка е подът."""
    grid = M.hull_grid(views, masks, S.MASK_SCALE)
    if grid is None or len(grid[0]) < 10:
        return None
    to_up = M.rotation_to(up if up is not None else M.estimate_up(views.centers, views.dirs, views.ups))
    pts = grid[0] @ to_up.T
    return M.Cloud(pts, np.zeros_like(pts), np.ones((len(pts), 4)), None, to_up)


def by_single(job: Path, work: Path, found: list[Path], sh: Callable[[str, list], None]) -> tuple[trimesh.Trimesh, dict]:
    pick = work / "single"
    pick.mkdir(parents=True, exist_ok=True)
    for f in spread(found, SINGLE_CANDIDATES):
        shutil.copy(f, pick / f.name)
    masks, out = work / "single_masks.npz", work / "single.ply"
    sh("finding the object", [sys.executable, S.__file__, pick, masks])
    sh("ai modelling", [sys.executable, single3d.__file__, pick, masks, out])
    info = json.loads(out.with_suffix(".json").read_text())
    model = trimesh.load(out, force="mesh")
    P.note(job, f"single frame AI: from {info['frame']}, {info['faces']} faces")
    return model, {"object": "", "suitable": True, "real_mm": None}


# --- Задачата -------------------------------------------------------------------------------------------------------


def run(job: Path, method: str, flat_base: bool, set_state: Callable[[str], None], sh: Callable[[str, list], None],
        step: Callable[[str], None], tune: dict | None = None) -> dict:  # fmt: skip
    """step(name): какво показва приложението в момента (хвърля грешка, ако задачата е отказана). Връща отчета за
    печата."""
    if not available().get(method):
        raise P.Failed("Тази AI услуга още не е включена. Опитайте по-късно.")
    work = job / "work"
    shutil.rmtree(work, ignore_errors=True)  # от повторно изпълнена задача
    work.mkdir(parents=True)
    TX.clear(job)  # текстура от снимки от сканиране, ако задачата преди е била сканиране
    set_state("reconstructing")
    if method == "single":
        return single(job, work, flat_base, set_state, sh, tune)
    raise P.Failed("Непознат начин за модел.")
