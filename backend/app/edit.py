"""Нов модел от съществуващ: моделът застава върху поставка с надпис (табелка с име). Старият модел остава
непроменен; новият е отделна задача. Работи в реалния размер на модела (зададения в приложението), в милиметри, а
текстурата от снимки остава с модела, а табелката е отделна двуцветна част."""

import shutil
from pathlib import Path
from typing import Callable
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from . import pipeline as P
from . import texture as TX

SOURCE = "source.ply"  # в папката на новата задача: моделът, от който е създаден, копиран при създаването на задачата


class EditError(P.Failed):
    """За потребителя: заявката не може да се изпълни."""


class Model:
    """Променяната мрежа в реален размер в мм (z нагоре, стои върху z = 0, центрирана) и афинната промяна досега на
    оригиналната повърхност (за оцветяване на резултата от оригинала накрая)."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.T = np.eye(4)

    def transform(self, matrix: np.ndarray):
        self.mesh.apply_transform(matrix)
        self.T = matrix @ self.T

    def seat(self):
        lo, hi = self.mesh.bounds
        self.transform(trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]))

    def axis(self, name: str) -> int:
        """Оста на мрежата за дължината, ширината или височината от приложението."""
        if name == "height":
            return 2
        ex = self.mesh.extents
        longer = 0 if ex[0] >= ex[1] else 1
        return longer if name == "length" else 1 - longer

    def size(self, name: str) -> float:
        return float(self.mesh.extents.max()) if name == "longest" else float(self.mesh.extents[self.axis(name)])


def boolean(kind: str, a: trimesh.Trimesh, b: trimesh.Trimesh, what: str, note: Callable[[str], None]) -> trimesh.Trimesh:
    """Обединение или разлика с manifold3d. Обединение на мрежа с отвори преминава към двете припокрити (програмите за
    печат печатат припокрити части като една); разликата не може."""
    try:
        if not a.is_volume:
            raise ValueError("not a closed solid")
        out = (trimesh.boolean.union if kind == "union" else trimesh.boolean.difference)([a, b], engine="manifold")
        if out.is_empty or not len(out.faces):
            raise ValueError("empty result")
        return out
    except Exception as e:
        if kind == "union":
            note(f"edit: {what}: joined as overlapping parts ({e})")
            return trimesh.util.concatenate([a, b])
        raise EditError(f"„{what}“ не може да се направи: моделът не е затворено тяло (има отвори).")


def coloured(out: trimesh.Trimesh, original: trimesh.Trimesh, T: np.ndarray) -> trimesh.Trimesh:
    """Всеки връх взима цвета на най-близката точка от оригиналната повърхност, преместена заедно с модела."""
    if original.visual.kind != "vertex":
        return out
    _, idx = cKDTree(trimesh.transform_points(original.vertices, T)).query(out.vertices)
    out.visual.vertex_colors = original.visual.vertex_colors[idx]
    return out


TEXT_FONTS = ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")  # удебелен, с кирилица; иначе този на Pillow
MIN_TEXT = 2.0  # мм: по-малки букви не се печатат четливо
PLATE_COLOUR, LETTER_COLOUR = (31, 33, 38), (242, 242, 244)
Z_UP = np.array([0.0, 0.0, 1.0])


def text_font(px: int):
    from PIL import ImageFont

    for name in TEXT_FONTS:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default(px)


def text_outline(text: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Очертанията на буквите (затворени многоъгълници: външни и дупките на о, а, е...) в единици височина на главна
    буква, центрирани в (0, 0); и техните (ширина, височина) в тези единици."""
    from PIL import ImageDraw
    from skimage import measure

    font = text_font(160)
    l, t, r, b = font.getbbox(text)
    cap = font.getbbox("H")
    cap_h = float(cap[3] - cap[1])
    pad = 8
    img = Image.new("L", (int(r - l) + 2 * pad, int(b - t) + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=255)
    a = np.asarray(img, np.float32) / 255
    polys = []
    for c in measure.find_contours(np.pad(a, 1), 0.5):
        xy = np.c_[c[:, 1], -c[:, 0]] / cap_h
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if len(xy) >= 3:
            polys.append(xy)
    if not polys:
        raise EditError(f"Надписът „{text}“ няма букви, които мога да изпиша.")
    allp = np.concatenate(polys)
    mid = (allp.min(0) + allp.max(0)) / 2
    return [p - mid for p in polys], allp.max(0) - allp.min(0)


def text_solid(polys: list[np.ndarray], mm: float, z_lo: float, z_hi: float, drape: Callable[[np.ndarray], np.ndarray],
               fine: float | None) -> trimesh.Trimesh:  # fmt: skip
    """Буквите с височина `mm` като тяло от z_lo до z_hi (z: навън от повърхността), положени върху поставката с
    `drape`; fine: първо разделени на толкова дребно, за да следват крива."""
    import manifold3d as mf

    cs = mf.CrossSection([np.asarray(p * mm, np.float64) for p in polys], mf.FillRule.EvenOdd).simplify(0.01)
    solid = mf.Manifold.extrude(cs, z_hi - z_lo).translate((0.0, 0.0, z_lo))
    if fine:
        solid = solid.refine_to_length(fine)
    solid = solid.warp_batch(lambda P: np.asarray(drape(np.asarray(P, np.float64)), np.float64))
    mesh = solid.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def nameplate(m: trimesh.Trimesh, text: str, shape: str, front: np.ndarray,
              note: Callable[[str], None]) -> tuple[trimesh.Trimesh, np.ndarray]:  # fmt: skip
    """Поставката под модела (в координатите му, мм) с `text` в посока `front`. Връща я и кои стени са буквите."""
    n = np.array([front[0], front[1], 0.0])
    n /= max(np.linalg.norm(n), 1e-9)
    right = np.cross(Z_UP, n)
    polys, (w1, h1) = text_outline(text)
    floor, height = float(m.bounds[0, 2]), float(m.extents[2])
    bottom = m.vertices[m.vertices[:, 2] <= floor + 0.03 * height]
    a, b = bottom @ right, bottom @ n
    centre_a, centre_b = (a.min() + a.max()) / 2, (b.min() + b.max()) / 2
    margin = 0.08 * max(float(np.ptp(a)), float(np.ptp(b))) + 2.0
    stand_h = float(np.clip(0.1 * m.extents.max(), 6.0, 24.0))
    mm = min(0.6 * stand_h, 0.8 * stand_h / h1)
    overlap = min(0.4, stand_h / 4)
    zmid = floor - stand_h / 2
    middle = centre_a * right + centre_b * n + Z_UP * zmid
    relief = float(np.clip(0.15 * mm, 0.6, 1.5))
    inset = 0.8
    if shape == "round":
        radius = float(np.hypot(a - centre_a, b - centre_b).max()) + margin
        mm = min(mm, 2.4 * radius / w1)  # около 140 градуса наоколо: каквото се вижда отпред
        stand = trimesh.creation.cylinder(radius=radius, height=stand_h + overlap, sections=128)

        def drape(P: np.ndarray) -> np.ndarray:
            angle, out = P[:, 0] / radius, radius + P[:, 2]
            return middle + (out * np.sin(angle))[:, None] * right + (out * np.cos(angle))[:, None] * n + P[:, 1:2] * Z_UP

        fine = max(0.3, mm / 10)
    else:
        pad = max(2.0, 0.5 * mm)
        width = max(float(np.ptp(a)) + 2 * margin, w1 * mm + 2 * pad)
        depth = float(np.ptp(b)) + 2 * margin
        radius = depth / 2
        stand = trimesh.creation.box(extents=[width, depth, stand_h + overlap])
        turn = np.eye(4)
        turn[:3, :3] = np.c_[right, -n, Z_UP]  # кутията е една и съща и в двете посоки: само завъртане, никога огледално
        stand.apply_transform(turn)

        def drape(P: np.ndarray) -> np.ndarray:
            return middle + P[:, :1] * right + (radius + P[:, 2])[:, None] * n + P[:, 1:2] * Z_UP

        fine = None
    if mm < MIN_TEXT:
        raise EditError(f"Надписът „{text}“ не се побира (буквите стават под {MIN_TEXT:g} мм). Съкратете го.")
    stand.apply_translation(middle + Z_UP * (overlap / 2))
    letters = text_solid(polys, mm, -inset, relief, drape, fine)
    plate = trimesh.boolean.union([stand, letters], engine="manifold")
    rel = plate.triangles_center - middle
    out = np.hypot(rel @ right, rel @ n) if shape == "round" else rel @ n
    raised = out > radius + 0.02
    note(f"edit: name plate „{text}“ {mm:.1f} mm letters on a {shape} stand {stand_h:.1f} mm high, facing "
         f"{np.round(n, 2).tolist()}, {len(plate.faces)} faces")  # fmt: skip
    return plate, raised


def plate_part(plate: trimesh.Trimesh, raised: np.ndarray, k: int, job: Path) -> dict:
    """Табелката като още една част на текстурата (собствено малко двуцветно изображение), така че моделът отгоре да
    запази снимките си."""
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, :4], image[:, 4:] = PLATE_COLOUR, LETTER_COLOUR
    Image.fromarray(image).save(job / f"texture_{k}.jpg", quality=95)
    n = len(plate.faces)
    uv = np.repeat(np.where(raised, 0.75, 0.25), 3)
    return {f"vertices_{k}": plate.triangles.reshape(-1, 3).astype(np.float32),
            f"normals_{k}": np.repeat(plate.face_normals, 3, axis=0).astype(np.float32),
            f"uv_{k}": np.c_[uv, np.full(3 * n, 0.5)].astype(np.float32),
            f"faces_{k}": np.arange(3 * n).reshape(-1, 3).astype(np.uint32)}  # fmt: skip


def front_of(job: Path, mesh: trimesh.Trimesh) -> np.ndarray:
    """Накъде гледа моделът: към страната, която показва най-много (лице, лого)."""
    from . import render

    pts, col, _ = render.surface(job, mesh, 400_000)
    return render.view_turn(render.best_side(pts, col)).T @ np.array([0.0, -1.0, 0.0])


def run_nameplate(job: Path, params: dict, set_state: Callable[[str], None]) -> dict:
    wanted = params["nameplate"]
    text = " ".join(str(wanted["text"]).split())
    longest_mm = float(params.get("longest_mm") or 100.0)
    source = trimesh.load(job / SOURCE, force="mesh")
    parts = TX.load(job)
    set_state("cleaning")
    front = front_of(job, source)
    model = Model(source)
    model.transform(trimesh.transformations.scale_matrix(longest_mm / float(source.extents.max())))
    model.seat()
    plate, raised = nameplate(model.mesh, text, wanted["shape"], front, lambda t: P.note(job, t))
    original = model.mesh.copy()
    model.mesh = boolean("union", model.mesh, plate, "Табелка", lambda t: P.note(job, t))
    before = model.T.copy()
    model.seat()
    plate.apply_transform(model.T @ np.linalg.inv(before))
    trimesh.repair.fix_normals(model.mesh)
    out = coloured(model.mesh, original, model.T @ np.linalg.inv(before))
    real = float(out.extents.max())
    shrink = trimesh.transformations.scale_matrix(100.0 / real)
    out.apply_transform(shrink)
    if parts:  # снимките на модела, каквито са били, преместени с него; табелката като собствена част
        plate.apply_transform(shrink)
        to = shrink @ model.T
        arrays = {}
        for k, p in enumerate(parts):
            arrays |= {f"vertices_{k}": trimesh.transform_points(p["vertices"], to).astype(np.float32),
                       f"normals_{k}": np.asarray(p["normals"], np.float32),
                       f"uv_{k}": np.asarray(p["uv"], np.float32),
                       f"faces_{k}": np.asarray(p["faces"], np.uint32)}  # fmt: skip
        arrays |= plate_part(plate, raised, len(parts), job)
        np.savez_compressed(job / TX.NPZ, count=len(parts) + 1, **arrays)
    changes = [f"Табелка с надпис „{text}“"]
    return P.finish(job, out) | {"real_mm": round(real, 1), "edit": {"prompt": text, "changes": changes,
                                                                     "source": params.get("source")}}  # fmt: skip


def copy_source(source_job: Path, job: Path):
    """Моделът, от който започва промяната, с текстурата от снимки, ако има такава."""
    shutil.copy(source_job / "model.ply", job / SOURCE)
    if (source_job / TX.NPZ).exists():
        for f in [source_job / TX.NPZ, *TX.texture_files(source_job)]:
            shutil.copy(f, job / f.name)


def run(job: Path, params: dict, set_state: Callable[[str], None], sh: Callable[[str, list], None]) -> dict:
    if not params.get("nameplate"):
        raise EditError("Тази промяна вече не се поддържа.")
    return run_nameplate(job, params, set_state)
