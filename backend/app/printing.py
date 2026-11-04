"""Помощта за печат: посоката, която иска най-малко подпори, и разрязване на твърде голям за принтера модел на части,
които се побират на плочата. Работи с модела в зададения от потребителя размер; нищо не се съхранява."""

import io
import zipfile

import numpy as np
import trimesh

from .mesh import MeshError

MAX_PARTS = 48  # над това размерът или плочата са грешка
OVERHANG = np.radians(45)  # по-стръмното от това спрямо вертикалата иска подпора (обичайната настройка на програмите за печат)
EDGE_MM = 5.0  # оставя се свободно по ръбовете на плочата: кант, собствената граница на принтера
CANDIDATES = 10  # плоски страни на модела, пробвани като дъно
KEEP_UP = 0.85  # друга посока трябва да иска под този дял от подпорите на сегашната
MIN_CONTACT = 0.02  # от повърхността: при по-малко върху плочата отпечатъкът се отлепя


def overhang_share(mesh: trimesh.Trimesh) -> float:
    """Делът от повърхността с посока надолу, по-стръмна от OVERHANG, без това, което лежи на плочата."""
    normals, areas = mesh.face_normals, mesh.area_faces
    low = mesh.triangles_center[:, 2] <= mesh.bounds[0, 2] + 0.2
    needs = (normals[:, 2] < -np.cos(OVERHANG)) & ~low
    return float(areas[needs].sum() / max(areas.sum(), 1e-9))


def contact_share(mesh: trimesh.Trimesh) -> float:
    low = (mesh.triangles_center[:, 2] <= mesh.bounds[0, 2] + 0.2) & (mesh.face_normals[:, 2] < -0.9)
    return float(mesh.area_faces[low].sum() / max(mesh.area, 1e-9))


def lying_on(normal: np.ndarray) -> np.ndarray:
    """Завъртането, което поставя страна с посока `normal` върху плочата."""
    return trimesh.geometry.align_vectors(normal, [0.0, 0.0, -1.0])


def best_way_up(mesh: trimesh.Trimesh) -> tuple[np.ndarray, float, float]:
    """(завъртането 4x4, делът на подпорите в сегашната посока, делът след завъртането). Сегашната посока се запазва,
    освен ако друга е явно по-добра: плоската основа на сканирането обикновено е правилното дъно."""
    now = overhang_share(mesh)
    best, best_share = np.eye(4), now
    hull = mesh.convex_hull
    sides = np.argsort(hull.facets_area)[::-1][:CANDIDATES]
    for side in sides:
        turn = lying_on(hull.facets_normal[side])
        turned = mesh.copy()
        turned.apply_transform(turn)
        if contact_share(turned) < MIN_CONTACT:
            continue
        share = overhang_share(turned)
        if share < best_share and share < KEEP_UP * now:
            best, best_share = turn, share
    return best, now, best_share


def seated(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


def cuts(mesh: trimesh.Trimesh, bed: tuple[float, float, float]) -> tuple[tuple[int, int, int], bool]:
    """Колко части по x, y и z и дали първо да се завърти моделът на четвърт оборот върху плочата, за най-малко части."""
    x, y, z = mesh.extents
    usable = [max(1.0, bed[0] - 2 * EDGE_MM), max(1.0, bed[1] - 2 * EDGE_MM), max(1.0, bed[2] - EDGE_MM)]

    def count(a: float, b: float) -> tuple[int, int, int]:
        return (int(np.ceil(a / usable[0])), int(np.ceil(b / usable[1])), int(np.ceil(z / usable[2])))

    straight, turned = count(x, y), count(y, x)
    if int(np.prod(turned)) < int(np.prod(straight)):
        return turned, True
    return straight, False


def plan(mesh: trimesh.Trimesh, longest_mm: float, bed: tuple[float, float, float]) -> dict:
    """Какво би съдържал архивът, за да го покаже първо приложението."""
    m = mesh.copy()
    m.apply_scale(longest_mm / float(m.extents.max()))
    turn, before, after = best_way_up(m)
    m.apply_transform(turn)
    (nx, ny, nz), _ = cuts(m, bed)
    # Мрежата може да има празни клетки или клетка с две части: истинският брой иска направени разрези.
    count = 1 if nx * ny * nz == 1 else len(parts(mesh, longest_mm, bed))
    return {"parts": count, "support_before": round(before, 3), "support_after": round(after, 3),
            "turned": not np.allclose(turn, np.eye(4)),
            "size_mm": [round(float(v), 1) for v in m.extents]}  # fmt: skip


def parts(mesh: trimesh.Trimesh, longest_mm: float, bed: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    """Моделът в най-добрата посока, разрязан по мрежа на части, които се побират на плочата, всяка центрирана върху
    нея."""
    m = mesh.copy()
    m.apply_scale(longest_mm / float(m.extents.max()))
    turn, _, _ = best_way_up(m)
    m.apply_transform(turn)
    (nx, ny, nz), quarter = cuts(m, bed)
    if quarter:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
    if nx * ny * nz > MAX_PARTS:
        raise MeshError("Моделът става на твърде много части за този принтер. Намалете размера му.")
    if nx * ny * nz == 1:
        return [seated(m)]
    if not m.is_volume:
        raise MeshError("Този модел не може да се реже на части: не е затворено тяло.")
    lo, hi = m.bounds[0] - 0.5, m.bounds[1] + 0.5  # мрежата леко извън модела: нищо не лежи върху външните ѝ стени
    size = (hi - lo) / [nx, ny, nz]
    out = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell = trimesh.creation.box(extents=size)
                cell.apply_translation(lo + (np.array([i, j, k]) + 0.5) * size)
                piece = trimesh.boolean.intersection([m, cell], engine="manifold")
                if piece.is_empty or not len(piece.faces):
                    continue
                for bit in piece.split(only_watertight=False):
                    if abs(bit.volume) > 1.0:  # мм³: трохи по ръба на разрез
                        out.append(seated(bit))
    if not out:
        raise MeshError("Моделът не можа да се среже. Опитайте с друг размер.")
    return out


def zip_of(pieces: list[trimesh.Trimesh], name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, piece in enumerate(pieces):
            z.writestr(f"{name}-{i + 1:02d}.stl" if len(pieces) > 1 else f"{name}.stl", piece.export(file_type="stl"))
    return buf.getvalue()
