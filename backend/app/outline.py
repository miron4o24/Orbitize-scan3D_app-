"""AI моделът на Pro от цялото видео (app/ai.py, метод "single"): очертанията на предмета, изрязани с AI във всеки
кадър, измерената между кадрите дълбочина и AI дълбочината, слети в тях, и снимките като текстура. Решаващият
алгоритъм разполага кадрите (COLMAP); търсачът на предмета (app/segment.py) и MobileSAM с подсказки от 3D
(app/refine.py) изрязват предмета от всеки; формата от очертанията е това, което е вътре в предмета в (почти) всеки
кадър (обвивка от очертанията), върху фина мрежа. Само очертанията дават гладка бучка (основата на фигура е могила,
без ръце и гънки), затова измерената между кадрите дълбочина (OpenMVS) и AI дълбочината (app/depth.py) се сливат в
нея (app/fuse.py), както при експресно сканиране: ръб на основа, ръце, крака, гънки, вълни в косата. След това
текстурата от снимки (app/texture.py) показва всеки нарисуван детайл, както е заснет. Предположението на TripoSR от
един кадър (app/single3d.py) беше пробвано като източник на вдлъбнатините, напаснато между камерите: при фигура то
измисли второ лице и куха основа, затова е само резервен вариант, когато кадрите не могат да се разположат.
Координати: световни (на камерите, COLMAP) и на облака (света, завъртян с to_up, z нагоре, като Cloud от
app/mesh.py)."""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

from . import mesh as M
from . import normals as NO
from . import parallel as PAR
from . import pipeline as P
from . import segment as S

# Кадрите се коригират, съпоставят и сливат по дълбочина при този размер, с такава ширина на клетките на сливането (от
# размера), както при пълно сканиране. Спрямо 1280 и 0,4% при експресно сканиране върху същите кадри (сървър,
# 2026-11-04): етикетът и ръбовете на дрелка и лицето и надписът на фигура излязоха видимо по-остри.
SIDE = 1920
VOXEL = 0.0025
CELLS = 192  # мрежата на формата от очертанията: по-фина от 160 при сканиране (при неуспешно сливане тя е повърхността на модела)
# Воксел е вътре, когато този дял от кадрите, които го показват, го има в предмета: с очертания, уточнени с MobileSAM,
# всички без един на 20 (фигура, 18 кадъра: очертанията на формата се припокриват 97% с тези на кадрите, спрямо 87%
# при снизходителните 0,75 на сканиране, които оставяха тънкото ѝ тяло дебело); само с маските на търсача на предмета:
# 0,75 като при сканиране.
AGREE_SHARP = 0.95
# Използва се този от дяловете (до заявения), чиято форма е най-вярна на очертанията (припокриване): тънка част
# (дръжка на дрелка, 47 кадъра от глобалния алгоритъм) се разминава с очертанието в няколко кадъра и изчезва при 0,95,
# оставяйки батерията като кутия (задача 6509: 76% вярност на очертанията), докато тънкото тяло на фигура остава
# дебело при 0,75.
AGREE_LEVELS = (0.95, 0.9, 0.85, 0.8, 0.75)
# Измерването вижда само вътре в маските, затова масата се намира сред точките на решаващия алгоритъм, които виждат
# стаята. Не е масата, когато над този дял от предмета е под нея или когато предметът свършва на толкова (от размера
# му) над нея: далечен под би оставил формата да стигне до него.
TABLE_BELOW = 0.05
TABLE_GAP = 0.08
# Гъстото съпоставяне работи там, където предметът може да бъде в кадъра, толкова (от размера му там) около него:
# очертанието му и където го поставят очертанията на другите, със запълнени дупки (процепът между дръжка и батерия се
# измерва: казва, че пространството е празно). Сливането гледа не повече от 2% от размера извън формата
# (fuse.OUTSIDE), а масата идва от точките на решаващия алгоритъм. Областта при сканиране (segment.dense_area, почти
# целият кадър) заемаше 5 от 14-те минути за дрелка. Времето върви с пикселите.
BAND = 0.06
BAND_CELLS = 96  # мрежата на формата от очертанията за това: грубата стига


def project(X: np.ndarray, R, t, K, scale: float):
    """Пикселни координати (u, v) на световни точки при 1/scale от размера на кадъра и дали са отпред."""
    fx, fy, cx, cy = K
    Xc = PAR.turned(X, np.asarray(R)) + t
    front = Xc[:, 2] > 1e-9
    z = np.where(front, Xc[:, 2], 1.0)
    return (fx * Xc[:, 0] / z + cx) / scale, (fy * Xc[:, 1] / z + cy) / scale, front


def shown(X: np.ndarray, step: float, R, t, K, shape: tuple, scale: int) -> np.ndarray:
    """Очертанието на центровете на воксели X (свят; воксели с ширина `step`) в кадър, при размера на маските му
    (1/scale)."""
    u, v, front = project(X, R, t, K, scale)
    img = np.zeros(shape, bool)
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    ok = front & (ui >= 0) & (vi >= 0) & (ui < shape[1]) & (vi < shape[0])
    img[vi[ok], ui[ok]] = True
    depth = float(np.median((PAR.turned(X, np.asarray(R)) + t)[:, 2]))
    gap = int(np.ceil(step * K[0] / depth / scale)) if depth > 0 else 0  # центровете на воксели са на пиксел или повече един от друг
    if gap:
        img = ndimage.binary_closing(img, iterations=gap, border_value=0)
    return ndimage.binary_fill_holes(img)


def overlap(occ: np.ndarray, axes, step: float, to_up: np.ndarray, views: M.Views, masks: list, scale: int) -> float:
    """Средно припокриване (IoU) на очертанията на вътрешните воксели с маските на кадрите: колко вярна им е формата."""
    X = np.stack([axes[k][idx] for k, idx in enumerate(np.nonzero(occ))], -1) @ to_up  # облак -> свят
    if len(X) > 300_000:
        X = X[np.random.default_rng(0).choice(len(X), 300_000, replace=False)]

    def fit(frame):
        R, t, K, m = frame
        a, b = shown(X, step, R, t, K, m.shape, scale), m > 127
        return (a & b).sum() / max(1, (a | b).sum())

    return float(np.mean(PAR.each(fit, zip(views.R, views.t, views.K, masks))))


def surface(occ: np.ndarray, axes, step: float) -> trimesh.Trimesh | None:
    """Затворената повърхност около вътрешните воксели (координати на облака), заглаждана с воксел: от собствените им
    вътре/вън, а не от дяловете им, така че тънка част (ръка, шия) да запази пълната си ширина."""
    return M.hull_mesh(occ.astype(np.float32), axes, step, 0.5)


def on_table(job: Path, cloud: M.Cloud, ground_z: float) -> M.Cloud:
    """Измереният облак, стоящ върху масата, намерена от точките на решаващия алгоритъм (същото z: облакът е изравнен
    по нея), или какъвто е, когато тази равнина не може да е това, върху което стои предметът."""
    z = cloud.points[:, 2]
    lo, hi = np.percentile(cloud.points, 2, axis=0), np.percentile(cloud.points, 98, axis=0)
    size = float((hi - lo).max())
    below = float((z < ground_z).mean())
    gap = (float(lo[2]) - ground_z) / size
    if below > TABLE_BELOW or gap > TABLE_GAP:
        P.note(job, f"AI model: the solver's plane is not the table ({below:.0%} of the object below it, object "
               f"{gap:.1%} of its size above it)")  # fmt: skip
        return cloud
    P.note(job, f"AI model: table from the solver's points ({below:.1%} of the object below it, object {gap:.1%} of "
           "its size above it)")  # fmt: skip
    keep = z > ground_z + 0.015 * size  # като mesh.clean_cloud
    return replace(cloud, points=cloud.points[keep], normals=cloud.normals[keep], colors=cloud.colors[keep],
                   ground_z=ground_z)  # fmt: skip


def dense_areas(views: M.Views, masks: list, cloud: M.Cloud) -> list[np.ndarray] | None:
    """За всеки кадър, при размера на маската му, къде трябва да работи гъстото съпоставяне (вижте BAND); None без
    форма от очертанията."""
    found = M.hull_volume(views, masks, S.MASK_SCALE, cloud, cells=BAND_CELLS)
    if found is None:
        return None
    frac, axes, step = found
    X = np.stack([axes[k][idx] for k, idx in enumerate(np.nonzero(frac >= P.LOOSE_AGREE))], -1) @ cloud.to_up

    def area(frame):
        R, t, K, m = frame
        a = shown(X, step, R, t, K, m.shape, S.MASK_SCALE) | (m > 127)
        if not a.any():
            return np.ones(m.shape, bool)
        ys, xs = np.nonzero(a)
        width = max(np.ptp(ys), np.ptp(xs)) + 1
        return ndimage.binary_fill_holes(ndimage.binary_dilation(a, iterations=max(2, int(np.ceil(BAND * width)))))

    return PAR.each(area, zip(views.R, views.t, views.K, masks))


def surface_directions(job: Path, work: Path, dense: Path, views: M.Views, sh, every: int | None = None) -> Path | None:
    """Посоките на повърхността във всяко изрязване (app/normals.py) или None: няма модел, няма изрязвания или
    стъпката се е провалила (тогава сливането е като преди)."""
    crops = work / P.CROPS
    if not NO.ready():
        P.note(job, f"AI model: surface directions skipped, model missing ({NO.MODEL})")
        return None
    if not crops.exists():
        return None
    names, out = work / "normals_names.txt", work / "normals.npz"
    names.write_text("\n".join(views.names), encoding="utf-8")
    try:
        sh("estimating detail", [sys.executable, NO.__file__, dense / "images", crops, names, out,
                                 str(every or NO.EVERY)])  # fmt: skip
    except P.Failed as e:
        P.note(job, f"AI model: surface directions failed ({e}); fitted AI depth only")
        return None
    return out if out.exists() else None


def measured(job: Path, work: Path, dense: Path, views: M.Views, masks: list, cloud: M.Cloud, up: np.ndarray | None,
             agree: float, set_state: Callable[[str], None], sh: Callable[[str, list], None], side: int = SIDE,
             voxel: float = VOXEL, fixed: bool = False, table: bool = True, flap: float | None = None,
             wide: bool = False, geometric: bool = True, passes: int | None = None, debug: bool = False,
             detail: bool = True, normals_every: int | None = None):  # fmt: skip
    """Повърхността на модела (координати на облака) и облакът, в който е: измерената между кадрите дълбочина
    (OpenMVS, в маските) и AI дълбочината, слети във формата от очертанията; самата форма, когато измерването или
    сливането се провали (dense: коригираните кадри, моделът на COLMAP в dense/sparse). side: размерът на кадрите;
    voxel: мрежата на сливането; fixed: формата точно при `agree`, без избор; table: използва масата на решаващия
    алгоритъм (на `cloud`, намерена сред точките му); flap: fuse.FLAP на сливането; wide: гъсто съпоставяне в
    областта на сканиране вместо около предмета; geometric: вторият му проверяващ проход; passes: fuse.PASSES на
    сливането; debug: записва входа на сливането (pipeline.FUSE_DEBUG); detail: формата отблизо от посоките на
    повърхността (app/normals.py), иначе само от напасната AI дълбочина (предишният начин)."""
    solvers = cloud.ground_z if table else None
    mask_dir = work / "dense_masks"
    mask_dir.mkdir(exist_ok=True)
    areas = None if wide else dense_areas(views, masks, cloud)
    if areas is not None:
        P.note(job, f"AI model: dense matching on {np.mean([a.mean() for a in areas]):.0%} of the frames' pixels")
    for k, (name, m) in enumerate(zip(views.names, masks)):
        with Image.open(dense / "images" / name) as im:
            size = im.size
        if areas is None:
            area = Image.fromarray(S.dense_area(m, size))
        else:
            area = Image.fromarray(areas[k].astype(np.uint8) * 255).resize(size, Image.NEAREST)
        area.save(mask_dir / f"{Path(name).stem}.mask.png")
    dense_cloud = False
    try:
        sh("dense setup", ["InterfaceCOLMAP", "-w", dense, "-i", dense, "-o", dense / "scene.mvs",
            "--image-folder", dense / "images"])  # fmt: skip
        neighbors = dense / "neighbors.txt"
        lines = P.neighbor_lines(views, P.mvs_indices(dense / "sparse" / "images.bin"))
        neighbors.write_text("\n".join(lines) + "\n")
        # Съседните видео кадри силно се припокриват: стигат 4 изгледа на карта. Картите на дълбочината остават за
        # app/fuse.py (изтриват се след него).
        sh("dense matching", ["DensifyPointCloud", "-w", dense, "-i", dense / "scene.mvs", "-o", dense / "dense.mvs",
            "--view-neighbors-file", neighbors, "--mask-path", mask_dir, "--ignore-mask-label", "0",
            "--resolution-level", "0", "--max-resolution", str(side), "--number-views", "4",
            "--geometric-iters", "1" if geometric else "0", "--estimate-roi", "0", "--crop-to-roi", "0", "--remove-dmaps", "0",
            "--max-threads", P.THREADS])  # fmt: skip
        pts, nrm, col = P.load_cloud(dense / "dense.ply")
        upward = cloud.to_up[2] if solvers is not None else up  # z, както го дава масата на решаващия алгоритъм
        cloud = M.clean_cloud(pts, nrm, col, views, masks, S.MASK_SCALE, upward)
        if cloud.ground_z is None and solvers is not None:
            cloud = on_table(job, cloud, solvers)
        dense_cloud = True
        P.note(job, f"AI model: {len(pts)} measured points, {len(cloud.points)} on the object, table found: "
               f"{cloud.ground_z is not None}")  # fmt: skip
    except (P.Failed, M.MeshError, OSError, ValueError) as e:  # после формата от очертанията
        P.note(job, f"AI model: no depth measured ({e}); the outlines' shape")
    set_state("meshing")
    found = M.hull_volume(views, masks, S.MASK_SCALE, cloud, cells=CELLS)
    if found is None:
        raise P.Failed("the frames' outlines give no shape")
    frac, axes, step = found
    tried = {}
    for level in [agree] if fixed else [a for a in AGREE_LEVELS if a <= agree] or [agree]:
        occ = frac >= level
        if occ.sum() >= 1000:
            tried[level] = overlap(occ, axes, step, cloud.to_up, views, masks, S.MASK_SCALE)
    if not tried:
        raise P.Failed("the frames' outlines give no shape")
    best = max(tried, key=lambda a: (round(tried[a], 3), a))  # най-вярната; при равни по-строгата
    agree = best
    inside = frac >= agree
    outline = surface(inside, axes, step)
    if outline is None:
        raise P.Failed("the frames' outlines give no shape")
    P.note(job, f"AI model: outline shape from {len(views)} frames, agreeing {agree:.0%}: true to them "
           f"{tried[agree]:.1%} (" + ", ".join(f"{a:.0%}: {o:.1%}" for a, o in tried.items()) + ")")  # fmt: skip
    fused = None
    if dense_cloud:
        P.carved_hull(work, dense, views, masks, cloud, sh, found)  # AI дълбочината (app/depth.py) за сливането
        normals = surface_directions(job, work, dense, views, sh, normals_every) if detail else None
        fused = P.fused_surface(work, dense, views, cloud, outline, sh, voxel, flap, passes or P.F.PASSES, debug,
                                normals)  # fmt: skip
        for p in dense.glob("*.dmap"):
            p.unlink(missing_ok=True)
    if fused is None:
        P.note(job, "AI model: the outlines' shape (no fused depth)")
        return outline, cloud
    return M.clip_to_outline(fused, views, masks, S.MASK_SCALE, cloud), cloud
