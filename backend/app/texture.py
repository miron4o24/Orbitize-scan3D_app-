"""Текстура от снимки за модела: самите кадри, нанесени върху него (OpenMVS TextureMesh), така че надписите върху
предмета, дори дребни, да се четат както в кадрите. Цветовете на върховете (mesh.paint) дават по един цвят за точка,
на няколко милиметра една от друга при отпечатък от 10 см: етикетът на батерия на дрелка беше сиво петно; с текстура
"LI-ION 4.0Ah 18V" се чете. Кадрите влизат в пълен размер (до TEXTURE_SIDE) от собствената си корекция: всичко
останало работи с кадри от pipeline.MAX_FRAME_SIDE, което стига за формата и пази гъстото съпоставяне достъпно.
Изравняването на шевовете в TextureMesh (глобално и локално) почерняше голяма част от текстурата при тази версия на
OpenMVS (дрелка: 12-19% от атласа чисто черни, на петна върху стените), затова е изключено и глобалната част прави
`level` тук: всяка част от текстурата идва от един кадър, а кадрите се различават по експозиция и отблясък, затова без
това черна дръжка излизаше на сиви и черни петна. Стени, невидени от никой кадър (в процепи), излизат в EMPTY и вместо
това се запълват с цвета на върховете си. Атласът между частите се запълва с цвета на най-близката част (`fill`):
оставен в EMPTY, той прониква в ръбовете на частите на телефона (мипмапите му), розова ивица. Пази се в папката на
задачата: textured.npz (за всяка текстура: върхове, нормали, uv, стени, в координатите за печат) и texture_<k>.jpg;
mesh.glb_textured изгражда GLB от тях."""

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

TEXTURE_SIDE = 3840  # кадри до 4K влизат в пълен размер
MAX_TEXTURE = 4096  # за всяко изображение на текстура: телефоните зареждат 4096 x 4096 (64 MB); повече се разделя на няколко
EMPTY = (255, 0, 255)  # цветът на TextureMesh за стени, невидени от никой кадър: такъв, който не се среща при реални предмети
CELL = 4  # тексели на страна от собствения квадрат с цвят на невидена стена (вижте paint_unseen)
NPZ = "textured.npz"
# level: колко гладка трябва да остане цветовата корекция на част спрямо съвпадането със съседите при шевовете
# (Waechter и съавт. 2014, "Let There Be Color!", използват 1/λ = 10).
SMOOTH = 10.0
RASTER_BATCH = 2_000_000  # кандидати за тексели на стъпка от rasterize (памет: около 50 байта на всеки)
# Кадри, по-размазани от този дял от медианната рязкост (върху предмета), остават извън текстурата: TextureMesh избира
# кадъра на всяка стена по това колко голяма и пряка е стената в него, а не колко е рязък, а едно сканиране запази
# кадри с една шеста от медианата. Отпадат най-много TEXTURE_DROP от кадрите, така че всяка страна да има кадри.
TEXTURE_SHARP = 0.5
TEXTURE_DROP = 0.3
SHARPNESS_SIDE = 1280  # кадрите се измерват при този размер
# Лека корекция на бялото: топла лампа оцветява всичко в оранжево. Цветът на светлината се оценява само от фона на кадрите
# (метод "нюанси на сивото", степен 6, в линейна светлина), а цвят на самия предмет не бива да се "поправя". Прилага се
# наполовина и в тесни граници: при дистанционно в стая с топла лампа червеното пада със 17%, а синьото расте с 25%; при
# дрелка и фигура на дневна светлина почти нищо не се променя.
WHITE_FRAMES = 8
WHITE_SIDE = 256
WHITE_POWER = 6.0
WHITE_STRENGTH = 0.5
WHITE_LIMITS = (0.8, 1.25)
WHITE_MIN_BACKGROUND = 0.3  # дял от кадъра: при по-малко фон светлината не може да се отдели от цвета на предмета


def texture_files(job: Path) -> list[Path]:
    return sorted(job.glob("texture_*.jpg"))


def clear(job: Path) -> None:
    """Премахва текстурата от предишно изпълнение (повторна задача)."""
    (job / NPZ).unlink(missing_ok=True)
    for p in texture_files(job):
        p.unlink(missing_ok=True)


def parts(loaded) -> list[trimesh.Trimesh]:
    """Мрежите с текстура от OBJ: по една за материал (изображение на текстура)."""
    if isinstance(loaded, trimesh.Scene):
        return [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    return [loaded]


def image_of(m: trimesh.Trimesh) -> Image.Image | None:
    mat = getattr(m.visual, "material", None)
    img = getattr(mat, "image", None) if mat is not None else None
    if img is None:
        img = getattr(mat, "baseColorTexture", None)
    return img


def unseen_faces(a: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Стените, чиято текстура е EMPTY (невидени от никой кадър). a: атласът (H, W, 3); uv: (стени, 3, 2), v нагоре."""
    H, W = a.shape[:2]
    centre = uv.mean(1)
    x = np.clip(np.round(centre[:, 0] * (W - 1)), 0, W - 1).astype(int)
    y = np.clip(np.round((1 - centre[:, 1]) * (H - 1)), 0, H - 1).astype(int)
    return np.abs(a[y, x].astype(int) - EMPTY).sum(1) < 120  # JPEG леко измества цвета


def rasterize(uv: np.ndarray, W: int, H: int, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Коя от `faces` покрива центъра на всеки тексел и къде в нея: (H, W) индекс на стена (-1: няма) и (H, W, 3)
    барицентрични тегла. uv: (всички стени, 3, 2), v нагоре. Стените вървят на порции с подобен размер, всяка като
    мрежа от ограничаващите им кутии."""
    px = np.stack([uv[..., 0] * W, (1 - uv[..., 1]) * H], -1)[faces]  # (n, 3, 2) пиксели, y надолу
    owner = np.full((H, W), -1, np.int32)
    bary = np.zeros((H, W, 3), np.float32)
    lo = np.floor(px.min(1) - 0.5).astype(np.int64)  # тексели, чийто център (i + 0.5) може да е вътре
    hi = np.ceil(px.max(1) - 0.5).astype(np.int64)
    size = hi - lo + 1
    order = np.argsort(size[:, 0] * size[:, 1], kind="stable")
    i = 0
    while i < len(order):
        end = min(len(order), i + max(1, RASTER_BATCH // max(1, int(size[order[i]].prod()))))
        while True:
            bw, bh = size[order[i:end]].max(0)
            if bw * bh * (end - i) <= RASTER_BATCH or end == i + 1:
                break
            end = i + (end - i) // 2
        k = order[i:end]
        i = end
        X = lo[k, 0][:, None, None] + np.arange(bw)[None, None, :]
        Y = lo[k, 1][:, None, None] + np.arange(bh)[None, :, None]
        X, Y = np.broadcast_to(X, (len(k), bh, bw)), np.broadcast_to(Y, (len(k), bh, bw))
        a, b, c = (px[k, j][:, None, None, :] for j in range(3))
        v0, v1 = b - a, c - a
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        dx, dy = X + 0.5 - a[..., 0], Y + 0.5 - a[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = (dx * v1[..., 1] - v1[..., 0] * dy) / den
            l2 = (v0[..., 0] * dy - dx * v0[..., 1]) / den
        l0 = 1 - l1 - l2
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6) & (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        f, yy, xx = np.nonzero(inside)
        owner[Y[f, yy, xx], X[f, yy, xx]] = faces[k[f]]
        bary[Y[f, yy, xx], X[f, yy, xx]] = np.stack([l0[f, yy, xx], l1[f, yy, xx], l2[f, yy, xx]], -1)
    return owner, bary


def nearest_covered(covered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """За всеки тексел най-близкият, покрит от някоя стена (ред и колона)."""
    iy, ix = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    return iy, ix


def level(a: np.ndarray, uv: np.ndarray, corner: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Глобално изравняване на шевовете: гладка цветова корекция за всяка част, така че частите да съвпадат там,
    където се срещат, а детайлът в тях да се запази. a: атлас (H, W, 3) uint8; uv: (стени, 3, 2); corner: (стени,
    3) върхът на модела при всеки ъгъл; seen: стените с истинска текстура. Връща коригирания атлас с пространството
    между частите, запълнено от най-близката част, покритите тексели (H, W) и ред за дневника."""
    H, W = a.shape[:2]
    owner, bary = rasterize(uv, W, H, np.flatnonzero(seen))
    covered = owner >= 0
    if not covered.any():
        return a, covered, "no texels covered"
    iy, ix = nearest_covered(covered)
    filled = a[iy, ix]
    # Върхове на текстурата: по един за всяка различна двойка (връх на модела, uv); връх от модела върху шев има по
    # един за всяка част около него.
    key = np.c_[corner.reshape(-1), np.round(uv.reshape(-1, 2) * 1e6)]
    _, tv = np.unique(key, axis=0, return_inverse=True)
    tv = tv.reshape(-1, 3)
    n = int(tv.max()) + 1
    # Цветът на всяка част при всеки връх: взет малко навътре в стените ѝ (към този ъгъл), върху леко размазан атлас,
    # осреднен за стените около него.
    soft = uniform_filter(filled.astype(np.float32), size=(3, 3, 1))
    w = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    pts = np.einsum("kc,fcd->fkd", w, uv[seen])
    x = np.clip((pts[..., 0] * W).astype(np.int64), 0, W - 1)
    y = np.clip(((1 - pts[..., 1]) * H).astype(np.int64), 0, H - 1)
    f = np.zeros((n, 3))
    count = np.zeros(n)
    np.add.at(f, tv[seen].reshape(-1), soft[y, x].reshape(-1, 3))
    np.add.at(count, tv[seen].reshape(-1), 1)
    known = count > 0
    f[known] /= count[known, None]
    # Шевове: съседни върхове на текстурата от един връх на модела, и двата видени.
    pairs = np.unique(np.c_[corner.reshape(-1), tv.reshape(-1)], axis=0)
    pairs = pairs[known[pairs[:, 1]]]
    same = pairs[1:, 0] == pairs[:-1, 0]
    si, sj = pairs[:-1, 1][same], pairs[1:, 1][same]
    if not len(si):
        return np.where(covered[..., None], a, filled), covered, "one patch, nothing to level"
    edges = np.unique(np.sort(np.r_[tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]], 1), axis=0)

    def incidence(i, j, m):
        r = np.arange(len(i))
        return sp.csr_matrix((np.r_[np.ones(len(i)), -np.ones(len(i))], (np.r_[r, r], np.r_[i, j])), shape=(m, n))

    S, E = incidence(si, sj, len(si)), incidence(edges[:, 0], edges[:, 1], len(edges))
    A = (S.T @ S + SMOOTH * (E.T @ E) + 1e-4 * sp.identity(n)).tocsc()
    g = splu(A).solve(-(S.T @ (f[si] - f[sj])))  # (n, 3)
    before = np.abs(f[si] - f[sj]).mean()
    after = np.abs(f[si] + g[si] - f[sj] - g[sj]).mean()

    out = a.astype(np.float32)
    flat = np.flatnonzero(covered)
    for s in range(0, len(flat), RASTER_BATCH):
        t = flat[s : s + RASTER_BATCH]
        yy, xx = np.unravel_index(t, (H, W))
        out[yy, xx] += np.einsum("tk,tkc->tc", bary[yy, xx], g[tv[owner[yy, xx]]])
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    out = out[iy, ix]  # покритите тексели пазят своя цвят; останалите взимат този на най-близкия
    note = f"{len(si)} seam points, colour step across seams {before:.1f} -> {after:.1f} of 255, mean change {np.abs(g).mean():.1f}"
    return out, covered, note


def paint_unseen(img: Image.Image, uv: np.ndarray, colors: np.ndarray, empty: np.ndarray,
                 start: int) -> tuple[Image.Image, np.ndarray, float]:  # fmt: skip
    """Стените `empty` (невидени от никой кадър) в цвета им. TextureMesh насочва всички към един тексел, затова всяка
    получава собствен квадрат с ширина CELL в редове, добавени от `start` (под последната част, защото TextureMesh
    подрежда отгоре), запълнен със средния цвят на ъглите ѝ. uv: (стени, 3, 2) за ъгъл, v нагоре; colors: (стени, 3,
    3+) за ъгъл. Връща изображението, новите uv и дела на такива стени."""
    a = np.asarray(img.convert("RGB"))
    H, W = a.shape[:2]
    n = int(empty.sum())
    if not n:
        return img, uv, 0.0
    # Добавят се още редове, ако тези под частите са твърде малко, но никога над MAX_TEXTURE (тогава квадратите са
    # по-малки).
    for cell in (CELL, 2, 1):
        rows = int(np.ceil(n / (W // cell))) * cell
        if start + rows <= MAX_TEXTURE or cell == 1:
            break
    total = max(H, start + rows)
    out = np.zeros((total, W, 3), np.uint8)
    out[:H] = a
    k = np.arange(n)
    per_row = W // cell
    cx, cy = (k % per_row) * cell, start + (k // per_row) * cell
    col = colors[empty][:, :, :3].mean(1).round().astype(np.uint8)
    for dy in range(cell):
        for dx in range(cell):
            out[cy + dy, cx + dx] = col
    uv = uv.copy()
    uv[..., 1] = 1 - (1 - uv[..., 1]) * H / total  # старото изображение сега запълва горните H от `total` реда
    uv[empty] = np.stack([(cx + cell / 2) / W, 1 - (cy + cell / 2) / total], -1)[:, None, :]
    return Image.fromarray(out), uv, n / len(uv)


def frame_sharpness(path: Path, mask: np.ndarray | None) -> float:
    """Дисперсия на лапласиана върху предмета (маската му, всякакъв размер), при кадър, мащабиран до SHARPNESS_SIDE."""
    with Image.open(path) as im:
        g = im.convert("L")
        if max(g.size) > SHARPNESS_SIDE:
            s = SHARPNESS_SIDE / max(g.size)
            g = g.resize((round(g.width * s), round(g.height * s)), Image.BILINEAR)
        a = np.asarray(g, np.float32)
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    if mask is not None:
        m = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).resize((a.shape[1], a.shape[0]))
        inside = np.asarray(m)[1:-1, 1:-1] > 127
        if inside.sum() > 500:
            return float(lap[inside].var())
    return float(lap.var())


def blurry_frames(photos: Path, names: list[str], masks: dict[str, np.ndarray]) -> list[str]:
    """Кадрите, които да отпаднат от текстурата (вижте TEXTURE_SHARP)."""
    if len(names) < 10:
        return []
    s = np.array([frame_sharpness(photos / n, masks.get(n)) for n in names])
    limit, median = int(len(names) * TEXTURE_DROP), float(np.median(s))
    return [names[i] for i in np.argsort(s)[:limit] if s[i] < TEXTURE_SHARP * median]


def white_gain(photos: Path, names: list[str], masks: dict[str, np.ndarray]) -> np.ndarray:
    """Умножителите за червено, зелено и синьо (в линейна светлина) от фона на кадрите; единици, когато фонът не стига."""
    found = []
    for name in list(names)[:: max(1, len(names) // WHITE_FRAMES)][:WHITE_FRAMES]:
        if name not in masks:
            continue
        with Image.open(photos / name) as im:
            small = im.convert("RGB")
            small.thumbnail((WHITE_SIDE, WHITE_SIDE))
        back = np.asarray(Image.fromarray(masks[name]).resize(small.size, Image.BILINEAR)) < 64
        if back.mean() < WHITE_MIN_BACKGROUND:
            continue
        linear = (np.asarray(small, np.float32)[back] / 255) ** 2.2
        light = np.mean(linear**WHITE_POWER, axis=0) ** (1 / WHITE_POWER)
        if light.min() > 1e-3:
            found.append(light / light.mean())
    if not found:
        return np.ones(3)
    return np.clip(np.median(found, axis=0) ** -WHITE_STRENGTH, *WHITE_LIMITS)


def balanced(a: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Изображение или цветове (uint8, RGB, евентуално с алфа) с умножителите на white_gain."""
    levels = (np.arange(256, dtype=np.float32) / 255) ** 2.2
    out = a.copy()
    for c in range(3):
        table = np.clip(np.round(np.clip(levels * gain[c], 0, 1) ** (1 / 2.2) * 255), 0, 255).astype(np.uint8)
        out[..., c] = table[a[..., c]]
    return out


def texture_model(job: Path, work: Path, photos: Path, best: Path, model: trimesh.Trimesh, to_print: np.ndarray,
                  to_up: np.ndarray, sh: Callable[[str, list], None], colmap: str, threads: str,
                  names: list[str] = (), masks: dict[str, np.ndarray] | None = None,
                  gain: np.ndarray | None = None) -> dict:  # fmt: skip
    """Текстурира `model` (координати за печат; to_print: 4x4 от координатите на облака към тях; to_up: завъртане свят
    -> облак) от кадрите в `photos`, разположени от модела `best` на COLMAP (кадрите му: `names`; masks: предметът
    във всеки, за измерване на рязкостта). Записва textured.npz и изображенията на текстурата в задачата; връща
    обобщение за дневника. Хвърля грешка при неуспех (извикващият запазва цветовете на върховете)."""
    tex = work / "texture"
    shutil.rmtree(tex, ignore_errors=True)
    blurry = blurry_frames(photos, list(names), masks or {})
    posed = best
    if blurry:
        (work / "blurry.txt").write_text("\n".join(blurry) + "\n")
        sharp = work / "texture_sparse"
        shutil.rmtree(sharp, ignore_errors=True)
        sharp.mkdir(parents=True)
        try:
            sh("texturing", [colmap, "image_deleter", "--input_path", best, "--output_path", sharp,
                "--image_names_path", work / "blurry.txt"])  # fmt: skip
            posed = sharp
        except Exception:  # (pipeline.Failed) тогава всички кадри, както преди
            blurry = []
    sh("texturing", [colmap, "image_undistorter", "--image_path", photos, "--input_path", posed, "--output_path", tex,
        "--output_type", "COLMAP", "--max_image_size", str(TEXTURE_SIDE)])  # fmt: skip
    sh("texturing", ["InterfaceCOLMAP", "-w", tex, "-i", tex, "-o", tex / "scene.mvs",
        "--image-folder", tex / "images"])  # fmt: skip
    # Обратно от координатите за печат към тези на камерите (свят).
    cloud = trimesh.transform_points(model.vertices, np.linalg.inv(to_print))
    trimesh.Trimesh(cloud @ to_up, model.faces, process=False).export(tex / "model.ply")
    empty = (EMPTY[0] << 16) | (EMPTY[1] << 8) | EMPTY[2]
    sh("texturing", ["TextureMesh", "-w", tex, "-i", tex / "scene.mvs", "-m", tex / "model.ply",
        "-o", tex / "textured.obj", "--export-type", "obj", "--close-holes", "0", "--decimate", "1",
        "--global-seam-leveling", "0", "--local-seam-leveling", "0", "--max-texture-size", str(MAX_TEXTURE),
        "--empty-color", str(empty), "--max-threads", threads])  # fmt: skip
    meshes = parts(trimesh.load(tex / "textured.obj", process=False))
    if not meshes:
        raise ValueError("TextureMesh wrote no mesh")
    tree = cKDTree(model.vertices)
    normals = model.vertex_normals
    colors = model.visual.vertex_colors if model.visual.kind == "vertex" else np.full((len(model.vertices), 4), 160)
    arrays, unseen, sizes, levelled = {}, [], [], []
    clear(job)
    for k, m in enumerate(meshes):
        img = image_of(m)
        if img is None or m.visual.uv is None:
            raise ValueError("TextureMesh wrote a mesh without texture")
        verts = trimesh.transform_points(m.vertices @ to_up.T, to_print)
        _, near = tree.query(verts)  # същите точки като на модела (TextureMesh запазва повърхността)
        corner = near[m.faces]  # (стени, 3): върхът на модела при всеки ъгъл
        uv = np.asarray(m.visual.uv, np.float64)[m.faces]
        a = np.asarray(img.convert("RGB"))
        empty = unseen_faces(a, uv)
        try:
            a, covered, note = level(a, uv, corner, ~empty)
        except (RuntimeError, ValueError, MemoryError) as e:  # тогава атласът, както го е направил TextureMesh
            covered, note = np.abs(a.astype(int) - EMPTY).sum(2) >= 120, f"not levelled ({e})"
        levelled.append(note)
        rows = np.flatnonzero(covered.any(1))
        start = int(rows[-1]) + 3 if len(rows) else 0
        img, uv, share = paint_unseen(Image.fromarray(a), uv, colors[corner], empty, start)
        unseen.append(share * len(m.faces))
        # По един връх за всяка различна двойка (точка, uv): TextureMesh записва по три за стена.
        corner, uv = corner.reshape(-1), uv.reshape(-1, 2)
        _, first, inverse = np.unique(np.c_[corner, np.round(uv * 1e6)], axis=0, return_index=True,
                                      return_inverse=True)  # fmt: skip
        arrays |= {
            f"vertices_{k}": model.vertices[corner[first]].astype(np.float32),
            f"normals_{k}": normals[corner[first]].astype(np.float32),
            f"uv_{k}": uv[first].astype(np.float32),
            f"faces_{k}": inverse.reshape(-1, 3).astype(np.uint32),
        }
        rgb = np.asarray(img.convert("RGB"))
        Image.fromarray(rgb if gain is None else balanced(rgb, gain)).save(job / f"texture_{k}.jpg", quality=92)
        sizes.append(img.size)
    np.savez_compressed(job / NPZ, count=len(meshes), **arrays)
    faces = sum(len(m.faces) for m in meshes)
    return {"textures": [list(s) for s in sizes], "unseen": round(sum(unseen) / max(faces, 1), 4),
            "levelled": "; ".join(levelled), "frames": f"{len(names) - len(blurry)} of {len(names)}"}  # fmt: skip


def load(job: Path) -> list[dict] | None:
    """Запазената текстура: за всяко изображение {vertices, normals, uv, faces, jpeg (байтове), size (w, h)}; None,
    ако няма."""
    path = job / NPZ
    if not path.exists():
        return None
    out = []
    with np.load(path) as z:
        for k in range(int(z["count"])):
            img = job / f"texture_{k}.jpg"
            with Image.open(img) as im:
                size = im.size
            out.append({"vertices": z[f"vertices_{k}"].astype(np.float64), "normals": z[f"normals_{k}"],
                        "uv": z[f"uv_{k}"], "faces": z[f"faces_{k}"], "jpeg": img.read_bytes(), "size": size})  # fmt: skip
    return out
