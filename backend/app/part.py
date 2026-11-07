"""Точна част (инженерен режим): модел 1:1 на малък детайл от снимки и мерки с линийка или шублер.
Изцяло локално на сървъра, без външни платени услуги и лимити.

Потребителят снима частта от основните страни (отгоре, отпред, отстрани/от края) и въвежда поне една мярка.
Липсващите мерки се изчисляват точно от измерените пиксели на очертанията (IS-Net сегментация), а 3D тялото се
изгражда чрез булево сечение на изтеглените профили (manifold3d). Така размерите са абсолютно точни за 3D печат."""

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from . import mesh as M
from . import pipeline as P
from . import segment as S
from . import texture as TX

log = logging.getLogger("uvicorn.error")

FIRST = 1024  # първи проход: намиране на предмета
CROP = 1.4  # размер на изрязването около предмета (в единици от размера му)
SMOOTH_PX = 1.2  # заглаждане на маската преди очертанието (без стъпаловидни пиксели)
SIMPLIFY = 0.0015  # допустимо отклонение при опростяване на многоъгълника
MIN_HOLE = 0.0008  # праг за площ на дупките (по-малки са шум)
DEPTH = 1.3  # коефициент на дълбочината на изтеглянето за чисто пресичане
COLOUR_SIDE = 700  # размер за текстуриране/оцветяване
OFF_VIEW = 0.35  # максимално log-отклонение за автоматично разпознаване на изглед
# The 1:1 path is intentionally independent from the smoother video pipeline.
# It keeps a fine, physical voxel pitch where the part is small, while bounding
# memory for large parts with an anisotropic grid.
DETAIL_VOXEL_MM = 0.12
GRID_MIN_AXIS = 48
GRID_MAX_AXIS = 512
GRID_MAX_VOXELS = 24_000_000
OPPOSITE_IOU_MIN = 0.55

SIDES = ("top", "bottom", "front", "back", "left", "right")
def soft_mask(sess, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    """Маската (0..1) на изрязване около предмета при пълната му разделителна способност и отместването в снимката."""
    W, H = image.size
    small_size = (FIRST, int(FIRST * H / W)) if W >= H else (int(FIRST * W / H), FIRST)
    small = S.predict(sess, image, small_size)
    m = small > 127
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("no object in the picture")
    cy, cx = (np.array(small.shape) - 1) / 2
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    centres = ndimage.center_of_mass(m, lab, range(1, n + 1))
    # Най-голямото петно, с наказание за отдалеченост от центъра
    score = [s / (1 + 3 * np.hypot((c[0] - cy) / small.shape[0], (c[1] - cx) / small.shape[1])) for s, c in zip(sizes, centres)]
    keep = lab == (int(np.argmax(score)) + 1)
    ys, xs = np.nonzero(keep)
    f = W / small.shape[1]
    x0, x1, y0, y1 = xs.min() * f, (xs.max() + 1) * f, ys.min() * f, (ys.max() + 1) * f
    side = max(x1 - x0, y1 - y0) * CROP
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(max(0, mx - side / 2)), int(max(0, my - side / 2)), int(min(W, mx + side / 2)), int(min(H, my + side / 2)))
    crop = image.crop(box)
    pred = S.predict(sess, crop, (FIRST, FIRST)).astype(np.float32) / 255.0
    full = np.asarray(Image.fromarray(pred).resize(crop.size, Image.BILINEAR), np.float32)
    full = ndimage.gaussian_filter(full, SMOOTH_PX)
    lab, n = ndimage.label(full > 0.5)
    if n > 1:
        sizes = ndimage.sum(full > 0.5, lab, range(1, n + 1))
        full = np.where(lab == (int(np.argmax(sizes)) + 1), full, np.minimum(full, 0.49))
    return full, (box[0], box[1])


def upright_angle(mask: np.ndarray) -> float:
    """Ъгълът (градуси), с който предметът става хоризонтален (минимално завъртане)."""
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) < 10:
        return 0.0
    pts = np.c_[xs, ys].astype(float)
    pts -= pts.mean(0)
    cov = np.cov(pts.T)
    if np.isnan(cov).any():
        return 0.0
    _, vecs = np.linalg.eigh(cov)
    long = vecs[:, -1]
    angle = np.degrees(np.arctan2(long[1], long[0]))
    return float((angle + 90) % 180 - 90)


def turned(a: np.ndarray, angle: float, order: int = 1) -> np.ndarray:
    if abs(angle) < 0.3:
        return a
    return ndimage.rotate(a, angle, axes=(1, 0), reshape=True, order=order, mode="constant", cval=0.0)


def outline(mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], tuple[float, float, float, float]]:
    """Външните очертания и дупките (в пиксели, y надолу) и кутията на предмета (x0, y0, x1, y1)."""
    from skimage import measure

    padded = np.pad(mask, 2)
    rings = [c[:, ::-1] - 2 for c in measure.find_contours(padded, 0.5)]
    rings = [r for r in rings if len(r) >= 8]
    if not rings:
        raise ValueError("no outline found")
    area = lambda r: 0.5 * float(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]))  # noqa: E731
    biggest = max(rings, key=lambda r: abs(area(r)))
    outer_sign = np.sign(area(biggest))
    outers = [r for r in rings if np.sign(area(r)) == outer_sign and abs(area(r)) > 0.02 * abs(area(biggest))]
    holes = [r for r in rings if np.sign(area(r)) != outer_sign and abs(area(r)) > MIN_HOLE * abs(area(biggest))]
    xs = np.concatenate([r[:, 0] for r in outers])
    ys = np.concatenate([r[:, 1] for r in outers])
    return outers, holes, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def view_data(sess, path: Path) -> dict:
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    full, (bx, by) = soft_mask(sess, image)
    angle = upright_angle(full)
    f = min(1.0, COLOUR_SIDE / max(full.shape))
    crop = image.crop((bx, by, bx + full.shape[1], by + full.shape[0]))
    small = np.asarray(crop.resize((max(1, int(crop.width * f)), max(1, int(crop.height * f))), Image.BILINEAR), np.float32)
    colour = turned(small, angle).clip(0, 255).astype(np.uint8)
    full = turned(full, angle)
    outers, holes, box = outline(full)
    return {
        "outers": outers,
        "holes": holes,
        "box": box,
        "px": (box[2] - box[0], box[3] - box[1]),
        "rgb": colour,
        "rgb_scale": colour.shape[1] / max(full.shape[1], 1),
    }


def pick_views(found: dict[str, dict], size: list[float]) -> dict[str, dict]:
    """Автоматично разпознаване на изгледи между неименувани снимки по техните пропорции."""
    L, W, H = (float(v) if v else 0.0 for v in (size + [0, 0, 0])[:3])
    ratio = {name: v["px"][0] / max(v["px"][1], 1e-6) for name, v in found.items()}
    area = {name: v["px"][0] * v["px"][1] for name, v in found.items()}
    names = list(found)
    if len(names) == 0:
        return {}

    if L and W and H:
        want = {"top": L / W, "side": L / H, "end": max(W, H) / min(W, H)}
        long = {"top": L, "side": L, "end": max(W, H)}
        best, best_score = None, np.inf
        for top in names:
            for side in names:
                if side == top and len(names) > 1:
                    continue
                other_candidates = [None] + [n for n in names if n not in (top, side)]
                for end in other_candidates:
                    trio = {"top": top, "side": side} | ({"end": end} if end else {})
                    shape = sum(abs(np.log(ratio[n] / want[v])) for v, n in trio.items())
                    scale = np.log([long[v] / max(found[n]["px"][0], 1e-6) for v, n in trio.items()])
                    score = shape + 0.5 * float(np.ptp(scale)) + (0.3 if end is None else 0.0)
                    if score < best_score:
                        best, best_score = trio, score
        if best:
            out = {"top": found[best["top"]]}
            for v in ("side", "end"):
                if v in best and abs(np.log(ratio[best[v]] / want[v])) <= OFF_VIEW:
                    out[v] = found[best[v]]
            return out

    # Без пълни 3 мерки
    left = dict(ratio)
    chosen: dict[str, dict] = {}
    side = max(left, key=left.get)
    chosen["side"] = found[side]
    left.pop(side)
    if left:
        top = max(left, key=lambda n: area[n])
        chosen["top"] = found[top]
        left.pop(top)
    if left:
        end = min(left, key=left.get)
        chosen["end"] = found[end]
    if "top" not in chosen and "side" in chosen:
        chosen["top"] = chosen.pop("side")
    return chosen


def polygon_shapely(v: dict, width: float, height: float, flip_x: bool = False):
    """Очертанието като shapely многоъгълник в мм."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    x0, y0, x1, y1 = v["box"]
    sx, sy = width / max(x1 - x0, 1e-6), height / max(y1 - y0, 1e-6)

    def to_mm(r):
        x = (r[:, 0] - x0) * sx
        if flip_x:
            x = width - x
        return np.c_[x, (y1 - r[:, 1]) * sy]

    shape = unary_union([Polygon(to_mm(r)).buffer(0) for r in v["outers"]])
    for h in v["holes"]:
        shape = shape.difference(Polygon(to_mm(h)).buffer(0))
    return shape.simplify(SIMPLIFY * max(width, height), preserve_topology=True)


def extrude_manifold(shape, depth: float):
    """manifold3d тяло от shapely многоъгълник, изтеглено по z."""
    import manifold3d as m3

    polys = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    loops = []
    for p in polys:
        loops.append(np.asarray(p.exterior.coords)[:-1])
        loops += [np.asarray(i.coords)[:-1] for i in p.interiors]
    cross = m3.CrossSection([np.ascontiguousarray(l, dtype=np.float64) for l in loops], m3.FillRule.EvenOdd)
    return m3.Manifold.extrude(cross, depth).translate([0, 0, -depth / 2])


def roundness(shape, width: float, height: float) -> float:
    from shapely.geometry import box

    s = min(width, height) / 2
    fractions = []
    for x0, y0 in ((0, 0), (width - s, 0), (0, height - s), (width - s, height - s)):
        square = box(x0, y0, x0 + s, y0 + s)
        empty = max(0.0, s * s - shape.intersection(square).area)
        fractions.append(min(1.0, float(np.sqrt(empty / (1 - np.pi / 4))) / s))
    return float(np.median(fractions))


def calculate_dimensions(views: dict, size: list[float]) -> tuple[float, float, float]:
    """Изчислява точните размери [L, W, H] в мм от дадените мерки и пропорциите на изгледите."""
    top = views.get("top")
    side = views.get("side")
    end = views.get("end")

    if not top and not side:
        raise ValueError("at least one view is required")
    if not top:
        top = side

    L, W, H = (float(v) if v and float(v) > 0 else 0.0 for v in (size + [0, 0, 0])[:3])
    tw, th = top["px"]

    if not L and W:
        L = W * tw / max(th, 1e-6)
    if not L and H and side:
        L = H * side["px"][0] / max(side["px"][1], 1e-6)
    if not L:
        L = 50.0  # резервен размер, ако потребителят не е въвел L

    if not W:
        W = L * th / max(tw, 1e-6)

    if not H:
        if side:
            H = L * side["px"][1] / max(side["px"][0], 1e-6)
        elif end:
            H = W * end["px"][1] / max(end["px"][0], 1e-6)
        else:
            H = W * 0.4  # разумна пропорция за дебелина при липса на изглед

    return float(round(L, 2)), float(round(W, 2)), float(round(H, 2))


def build_mesh(views: dict, dims: tuple[float, float, float]) -> "trimesh.Trimesh":
    """Изгражда visual hull от трите ортографски профила.

    Профилите са единствената геометрична информация, която снимките от шестте
    страни гарантират без калибрирана камера. Растеризирането на всеки профил и
    пресичането на трите екструдирани маски запазва вдлъбнатини и отвори, вместо
    да ги изглажда с евристична закръгленост. Финалният mesh се мащабира по
    отделните оси, така че външните граници съвпадат с въведените мерки.
    """
    import trimesh
    from skimage import measure
    from skimage.draw import polygon as draw_polygon

    L, W, H = dims
    top = views.get("top")
    side = views.get("side")
    end = views.get("end")
    if not top and side:
        top = side

    if not top or not side or not end:
        raise ValueError("three orthogonal profiles are required")

    def grid_shape() -> tuple[int, int, int]:
        physical = np.array([L, W, H], dtype=float)
        scale = min(
            1.0 / DETAIL_VOXEL_MM,
            GRID_MAX_AXIS / max(physical),
            (GRID_MAX_VOXELS / max(float(np.prod(physical)), 1e-6)) ** (1 / 3),
        )
        counts = np.clip(np.ceil(physical * scale).astype(int) + 1, GRID_MIN_AXIS, GRID_MAX_AXIS)
        # A very thin part can hit the per-axis minimum before its total grid
        # fits the memory budget. Only reduce axes that still have room.
        while int(np.prod(counts)) > GRID_MAX_VOXELS:
            movable = counts > GRID_MIN_AXIS
            if not movable.any():
                break
            factor = (GRID_MAX_VOXELS / float(np.prod(counts))) ** (1 / movable.sum())
            next_counts = counts.copy()
            next_counts[movable] = np.maximum(GRID_MIN_AXIS, np.floor(counts[movable] * factor).astype(int))
            if np.array_equal(next_counts, counts):
                next_counts[np.argmax(counts)] -= 1
            counts = next_counts
        return tuple(int(v) for v in counts)

    nx, ny, nz = grid_shape()

    def raster(shape, width: float, height: float, nx: int, ny: int) -> np.ndarray:
        mask = np.zeros((ny, nx), dtype=bool)
        polygons = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]

        def paint(ring, value: bool):
            points = np.asarray(ring.coords)
            xs = np.clip(points[:, 0] / width * (nx - 1), 0, nx - 1)
            ys = np.clip((height - points[:, 1]) / height * (ny - 1), 0, ny - 1)
            rr, cc = draw_polygon(ys, xs, mask.shape)
            mask[rr, cc] = value

        for poly in polygons:
            paint(poly.exterior, True)
            for hole in poly.interiors:
                paint(hole, False)
        return mask

    def profile_shape(view, width: float, height: float):
        shape = view.get("shape_mm")
        return shape if shape is not None else polygon_shapely(view, width, height)

    top_mask = raster(profile_shape(top, L, W), L, W, nx, ny)
    side_mask = raster(profile_shape(side, L, H), L, H, nx, nz)
    end_mask = raster(profile_shape(end, W, H), W, H, ny, nz)
    occupied = top_mask[None, :, :] & side_mask[:, None, :] & end_mask[:, :, None]
    if int(occupied.sum()) < 16:
        raise ValueError("the profiles do not contain a common solid")

    # A binary visual hull turns every image pixel into a stair-step. Convert
    # each orthographic silhouette to a signed distance field in millimetres,
    # then intersect the three fields. Its zero surface retains genuine small
    # profile detail while producing CAD-clean edges instead of voxel terraces.
    def sdf(mask: np.ndarray, row_mm: float, col_mm: float) -> np.ndarray:
        padded = np.pad(mask, 1)
        outside = ndimage.distance_transform_edt(~padded, sampling=(row_mm, col_mm))
        inside = ndimage.distance_transform_edt(padded, sampling=(row_mm, col_mm))
        return (outside - inside)[1:-1, 1:-1].astype(np.float32)

    dx = L / max(nx - 1, 1)
    dy = W / max(ny - 1, 1)
    dz = H / max(nz - 1, 1)
    top_sdf = sdf(top_mask, dy, dx)
    side_sdf = sdf(side_mask, dz, dx)
    end_sdf = sdf(end_mask, dz, dy)
    field = np.maximum(top_sdf[None, :, :], side_sdf[:, None, :])
    np.maximum(field, end_sdf[:, :, None], out=field)
    margin = max(dx, dy, dz) * 2
    field = np.pad(field, 1, constant_values=margin)

    vertices, faces, _, _ = measure.marching_cubes(
        field, level=0.0, spacing=(dz, dy, dx)
    )
    vertices -= np.array([dz, dy, dx])
    vertices = vertices[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    bounds = mesh.bounds
    span = np.maximum(bounds[1] - bounds[0], 1e-9)
    target = np.array([L, W, H], dtype=float)
    mesh.vertices = (mesh.vertices - bounds[0]) * (target / span)
    # Trimesh 5.1 exposes face cleanup through update_faces; the older
    # remove_degenerate_faces() method was removed in newer releases.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def paint_mesh(mesh, views: dict, dims: tuple[float, float, float]) -> None:
    """Оцветяване на върховете с цветове от снимките."""
    L, W, H = dims
    n = mesh.vertex_normals
    v = mesh.vertices
    plan = {"top": (0, 1, L, W), "side": (0, 2, L, H), "end": (1, 2, W, H)}
    choice = np.argmax(np.abs(n), axis=1)
    want = np.where(choice == 2, "top", np.where(choice == 1, "side", "end"))
    colours = np.full((len(v), 4), 215, np.uint8)
    colours[:, 3] = 255

    def paint(rows: np.ndarray, sv: dict, a: int, b: int, wa: float, wb: float) -> None:
        if not rows.any():
            return
        x0, y0, x1, y1 = sv["box"]
        k = sv["rgb_scale"]
        px = (x0 + v[rows, a] / max(wa, 1e-6) * (x1 - x0)) * k
        py = (y1 - v[rows, b] / max(wb, 1e-6) * (y1 - y0)) * k
        img = sv["rgb"]
        px = np.clip(np.round(px).astype(int), 0, img.shape[1] - 1)
        py = np.clip(np.round(py).astype(int), 0, img.shape[0] - 1)
        colours[rows, :3] = img[py, px]

    normal_axis = {"top": 2, "side": 1, "end": 0}
    for name, (a, b, wa, wb) in plan.items():
        rows = want == name
        primary = views.get(name)
        if not rows.any() or primary is None:
            continue
        facing_camera = rows & (n[:, normal_axis[name]] >= 0)
        paint(facing_camera, primary, a, b, wa, wb)
        paint(rows & ~facing_camera, primary.get("opposite_view", primary), a, b, wa, wb)

    mesh.visual.vertex_colors = colours


def choose_profiles(named_views: dict[str, dict], size: list[float]) -> tuple[dict[str, dict], dict[str, str]]:
    """Assigns the three physical profiles from all six photos by measured proportions.

    The labels are useful hints, but they are not trusted as geometry: users often
    hold a small part in a different orientation while taking the six photos.
    """
    from itertools import permutations
    from shapely import affinity
    from shapely.ops import unary_union

    measures = [float(v) for v in size[:3]]
    if not all(measures):
        raise ValueError("all three measured dimensions are required")
    groups = {
        "top": ("top", "bottom"),
        "side": ("front", "back"),
        "end": ("left", "right"),
    }
    expected = {
        "top": (measures[0] / measures[1], measures[0]),
        "side": (measures[0] / measures[2], measures[0]),
        "end": (max(measures[1], measures[2]) / min(measures[1], measures[2]), max(measures[1], measures[2])),
    }

    def score(group_name: str, role: str) -> float:
        ratio_want, long_want = expected[role]
        group = groups[group_name]
        ratios = []
        scales = []
        for side in group:
            view = named_views[side]
            px_w, px_h = view["px"]
            ratios.append(max(px_w / max(px_h, 1e-6), px_h / max(px_w, 1e-6)))
            scales.append(np.log(long_want / max(px_w, 1e-6)))
        return abs(np.log(np.median(ratios) / max(ratio_want, 1e-6))) + 0.5 * float(np.ptp(scales))

    names = tuple(groups)
    roles = ("top", "side", "end")
    assignment = min(permutations(names), key=lambda candidate: sum(score(group, role) for role, group in zip(roles, candidate)))
    profile_sizes = {
        "top": (measures[0], measures[1]),
        "side": (measures[0], measures[2]),
        "end": (measures[1], measures[2]),
    }

    def align_opposite(reference, candidate, width: float, height: float):
        """Match an opposite view's camera orientation before combining it."""
        variants = []
        for flip_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
            trial = affinity.scale(
                candidate,
                xfact=-1 if flip_x else 1,
                yfact=-1 if flip_y else 1,
                origin=(width / 2, height / 2),
            )
            union = reference.union(trial)
            score = reference.intersection(trial).area / max(union.area, 1e-9)
            variants.append((float(score), trial, flip_x, flip_y))
        return max(variants, key=lambda item: item[0])

    out = {}
    mapping = {}
    for role, group_name in zip(roles, assignment):
        candidates = [named_views[side] for side in groups[group_name]]
        width, height = profile_sizes[role]
        shapes = [polygon_shapely(candidate, width, height) for candidate in candidates]
        agreement, aligned, flip_x, flip_y = align_opposite(shapes[0], shapes[1], width, height)
        if agreement < OPPOSITE_IOU_MIN:
            raise ValueError(
                f"{role} opposite views disagree ({agreement:.0%}); keep the camera perpendicular and retake both photos"
            )
        out[role] = dict(candidates[0])
        out[role]["shape_mm"] = unary_union([shapes[0], aligned])
        out[role]["opposite_agreement"] = agreement
        opposite = dict(candidates[1])
        image = opposite["rgb"]
        if flip_y:
            image = image[::-1, :, :]
        if flip_x:
            image = image[:, ::-1, :]
        opposite["rgb"] = image
        out[role]["opposite_view"] = opposite
        mapping[role] = group_name
    return out, mapping


def run(job: Path, params: dict, set_state: Callable[[str], None], sh: Callable[[str, list], None],
        step: Callable[[str], None]) -> dict:
    """Изпълнява точна част 1:1 по качените снимки и въведените размери."""
    photos_dir = job / "photos"
    found_files = sorted(photos_dir.glob("*.jp*g"))
    if not found_files:
        raise P.Failed("Няма качени снимки на частта.")

    # Размери от потребителя: [L, W, H]
    raw_measures = params.get("measures") or params.get("size_mm") or []
    measures = [float(v) if isinstance(v, (int, float)) and float(v) > 0 else 0.0 for v in raw_measures[:3]]
    if not any(measures):
        raise P.Failed("Въведете поне една мярка на частта с линийка или шублер (мм).")
    if not all(measures):
        raise P.Failed("За инженерно копие 1:1 са нужни дължина, ширина и височина в милиметри.")

    set_state("reconstructing")
    step("finding the object")
    sess = S.session()

    # Разпределяне на снимките по страни
    named_files = {}
    unnamed_files = []
    for f in found_files:
        stem = f.stem.lower()
        matched = False
        for side in SIDES:
            if f"view_{side}" in stem or stem == side:
                named_files[side] = f
                matched = True
                break
        if not matched:
            unnamed_files.append(f)

    required = set(SIDES)
    if required - named_files.keys():
        missing = ", ".join(sorted(required - named_files.keys()))
        raise P.Failed(f"Липсват ортографски изгледи: {missing}. Качете всички шест обозначени страни.")
    parsed = {}
    for side, path in named_files.items():
        try:
            parsed[side] = view_data(sess, path)
            P.note(job, f"part: read {side} from {path.name}")
        except Exception as e:
            raise P.Failed(f"Неуспешно разпознаване на изгледа {side}. Направете снимката отново.") from e
    try:
        views, mapping = choose_profiles(parsed, measures)
    except Exception as e:
        raise P.Failed(f"Изгледите не дават надеждни три профила: {e}") from e
    P.note(job, f"part: profile assignment {mapping}")
    if unnamed_files:
        P.note(job, f"part: {len(unnamed_files)} supplementary photos retained for audit")

    step("measuring the views")
    dims = calculate_dimensions(views, measures)
    L, W, H = dims
    P.note(job, f"part: dimensions calculated {dims} mm from measures {measures}")

    set_state("meshing")
    step("building solid")
    mesh = build_mesh(views, dims)

    # Оцветяване
    set_state("cleaning")
    step("preparing for print")
    plain = bool(params.get("plain", False))
    if not plain:
        try:
            paint_mesh(mesh, views, dims)
        except Exception as e:
            P.note(job, f"part: vertex painting failed ({e}); using grey")
            mesh.visual.vertex_colors = np.tile([215, 215, 215, 255], (len(mesh.vertices), 1))
    else:
        mesh.visual.vertex_colors = np.tile([215, 215, 215, 255], (len(mesh.vertices), 1))

    # Поставяне на основата върху Z = 0
    printable = M.make_printable(mesh, float(mesh.bounds[0, 2]), flat_base=False, smoothing=0)
    TX.clear(job)

    # Завършване и запис на модела
    report = P.finish(job, printable)
    return report | {
        "photos_used": len(views),
        "photos_total": len(found_files),
        "profiles_required": ["top", "side", "end"],
        "ai": "part",
        "object": "Точна част 1:1",
        "suitable": True,
        "real_mm": round(float(max(dims)), 2),
        "size_given": True,
        "part_size_mm": [round(L, 2), round(W, 2), round(H, 2)],
        "dimensions_mm": [round(L, 2), round(W, 2), round(H, 2)],
    }
