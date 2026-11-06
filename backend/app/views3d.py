"""Точна част от три изгледа, както инженер копира част: отгоре, отстрани и от края. Във всяка снимка предметът се изрязва
с IS-Net върху близко изрязване около него (почти пълна разделителна способност: при снимка 4608 px около 0,05 мм на пиксел),
очертанието става многоъгълник (с дупките му: отвори през частта), мащабира се до мерките на потребителя, а тялото е общото
на трите очертания, изтеглени през частта (булево сечение с manifold3d). Така размерите идват от измерени пиксели, а не от
преценка на AI.

Оси: x дължината (отляво надясно в снимката отгоре и в тази отстрани), y ширината (нагоре в снимката отгоре, надясно в тази
от края), z височината. Снимката отгоре е отвесно надолу, отстрани е откъм потребителя, а от края е откъм дясната страна,
без частта да се мести между снимките.

Собствен процес: python views3d.py <out.ply> <мерки "L,W,H" (0: не е мерено)> top=<снимка> side=<снимка> [end=<снимка>].
Записва и <out>.json със сведенията."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

if __package__:
    from . import segment as S
else:  # като скрипт
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import segment as S

FIRST = 1024  # първият проход: къде е предметът
CROP = 1.5  # страна на изрязването около предмета, в размери на предмета
SMOOTH_PX = 1.2  # заглаждане на маската преди очертанието (пиксели на изрязването): без стъпалата на пикселите
SIMPLIFY = 0.0015  # от размера на предмета: по-малки отклонения в очертанието се махат (0,08 мм при 51 мм)
MIN_HOLE = 0.0008  # от площта на очертанието: по-малки "дупки" са шум
DEPTH = 1.2  # колко по-дълго от частта е всяко изтегляне (за да реже чисто)


def soft_mask(sess, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    """Маската (0..1) на изрязване около предмета при пълната му разделителна способност и ъгълът на изрязването в снимката."""
    W, H = image.size
    small = S.predict(sess, image, (FIRST, int(FIRST * H / W)) if W >= H else (int(FIRST * W / H), FIRST))
    m = small > 127
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("no object in the picture")
    cy, cx = (np.array(small.shape) - 1) / 2
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    centres = ndimage.center_of_mass(m, lab, range(1, n + 1))
    # Най-голямото петно, с наказание за далеч от средата (както го насочва приложението)
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
    if n > 1:  # само предметът (най-голямото петно), без дребни отражения
        sizes = ndimage.sum(full > 0.5, lab, range(1, n + 1))
        full = np.where(lab == (int(np.argmax(sizes)) + 1), full, np.minimum(full, 0.49))
    return full, (box[0], box[1])


def upright(mask: np.ndarray) -> np.ndarray:
    """Маската, завъртяна така, че дългата ос на предмета да е хоризонтална (с най-малкото завъртане)."""
    ys, xs = np.nonzero(mask > 0.5)
    pts = np.c_[xs, ys].astype(float)
    pts -= pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    long = vecs[:, -1]
    angle = np.degrees(np.arctan2(long[1], long[0]))
    angle = (angle + 90) % 180 - 90  # в (-90, 90]
    if abs(angle) < 0.3:
        return mask
    return ndimage.rotate(mask, angle, reshape=True, order=1, mode="constant", cval=0.0)


def outline(mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], tuple[float, float, float, float]]:
    """Външните очертания и дупките (в пиксели, y надолу) и кутията на предмета (x0, y0, x1, y1)."""
    from skimage import measure

    padded = np.pad(mask, 2)
    rings = [c[:, ::-1] - 2 for c in measure.find_contours(padded, 0.5)]  # (x, y)
    rings = [r for r in rings if len(r) >= 8]
    if not rings:
        raise ValueError("no outline")
    area = lambda r: 0.5 * float(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]))  # noqa: E731
    biggest = max(rings, key=lambda r: abs(area(r)))
    outer_sign = np.sign(area(biggest))
    outers = [r for r in rings if np.sign(area(r)) == outer_sign and abs(area(r)) > 0.02 * abs(area(biggest))]
    holes = [r for r in rings if np.sign(area(r)) != outer_sign and abs(area(r)) > MIN_HOLE * abs(area(biggest))]
    xs = np.concatenate([r[:, 0] for r in outers])
    ys = np.concatenate([r[:, 1] for r in outers])
    return outers, holes, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def view(sess, path: Path) -> dict:
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    full, _ = soft_mask(sess, image)
    full = upright(full)
    outers, holes, box = outline(full)
    return {"outers": outers, "holes": holes, "box": box, "px": (box[2] - box[0], box[3] - box[1])}


def polygon(v: dict, width: float, height: float, flip_x: bool = False):
    """Очертанието като shapely многоъгълник в мм: кутията на предмета става [0, width] x [0, height], y нагоре."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    x0, y0, x1, y1 = v["box"]
    sx, sy = width / (x1 - x0), height / (y1 - y0)

    def mm(r):
        x = (r[:, 0] - x0) * sx
        if flip_x:
            x = width - x
        return np.c_[x, (y1 - r[:, 1]) * sy]  # y нагоре

    shape = unary_union([Polygon(mm(r)).buffer(0) for r in v["outers"]])
    for h in v["holes"]:
        shape = shape.difference(Polygon(mm(h)).buffer(0))
    return shape.simplify(SIMPLIFY * max(width, height), preserve_topology=True)


def extrude(shape, depth: float):
    """manifold3d тяло от shapely многоъгълника, изтеглено по z от -depth/2 до depth/2."""
    import manifold3d as m3

    polys = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    loops = []
    for p in polys:
        loops.append(np.asarray(p.exterior.coords)[:-1])
        loops += [np.asarray(i.coords)[:-1] for i in p.interiors]
    cross = m3.CrossSection([np.ascontiguousarray(l, dtype=np.float64) for l in loops], m3.FillRule.EvenOdd)
    return m3.Manifold.extrude(cross, depth).translate([0, 0, -depth / 2])


def build(views: dict, size: list[float]) -> tuple:
    """Тялото от изгледите в мм и [дължина, ширина, височина]: липсващите мерки идват от пропорциите на изгледите."""
    import trimesh

    top, side, end = views["top"], views.get("side"), views.get("end")
    L, W, H = (float(v) if v else 0.0 for v in (size + [0, 0, 0])[:3])
    tw, th = top["px"]  # дължина, ширина в пиксели отгоре
    if not L and W:
        L = W * tw / th
    if not L and H and side:
        L = H * side["px"][0] / side["px"][1]
    if not L:
        raise ValueError("at least one measure is needed")
    if not W:
        W = L * th / tw
    if not H:
        if side:
            H = L * side["px"][1] / side["px"][0]
        elif end:
            H = W * end["px"][1] / end["px"][0]
        else:
            raise ValueError("the height needs a side photo or a measure")
    big = DEPTH * max(L, W, H)
    solid = extrude(polygon(top, L, W), big).translate([0, 0, H / 2])  # x, y; z през частта
    if side:
        s = extrude(polygon(side, L, H), big)  # в равнината x-z: завъртане от (x, y) към (x, z)
        s = s.rotate([90, 0, 0]).translate([0, W / 2, 0])
        solid = solid ^ s
    else:
        solid = solid ^ __import__("manifold3d").Manifold.cube([L, W, H])
    if end:
        e = extrude(polygon(end, W, H), big)  # в равнината y-z, гледано от +x
        e = e.rotate([90, 0, 0]).rotate([0, 0, 90]).translate([L / 2, 0, 0])
        solid = solid ^ e
    mesh = solid.to_mesh()
    out = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    return out, [L, W, H]


def main(out: Path, measures: str, args: list[str]) -> None:
    paths = dict(a.split("=", 1) for a in args)
    sess = S.session()
    views = {name: view(sess, Path(p)) for name, p in paths.items()}
    size = [float(v) for v in measures.split(",")]
    mesh, dims = build(views, size)
    mesh.export(out)
    info = {"size": [round(v, 2) for v in dims], "extents": np.round(mesh.extents, 2).tolist(), "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight), "views": {k: {"px": [round(p) for p in v["px"]], "holes": len(v["holes"])}
                                                               for k, v in views.items()}}  # fmt: skip
    out.with_suffix(".json").write_text(json.dumps(info))
    print("views: " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
