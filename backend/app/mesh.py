"""Dense point cloud -> clean, watertight, print-ready mesh.

Pure numpy/trimesh/pymeshlab code with no COLMAP/OpenMVS dependency, so it is unit-testable.
Coordinates after `clean_cloud`: +Z is up, the object stands on z = ground_z (when a table was found).
"""

from dataclasses import dataclass

import numpy as np
import pymeshlab
import trimesh
from scipy.spatial import cKDTree

UP = np.array([0.0, 0.0, 1.0])


class MeshError(Exception):
    """User-facing reconstruction problem."""


@dataclass
class Cloud:
    points: np.ndarray  # (n, 3)
    normals: np.ndarray  # (n, 3)
    colors: np.ndarray  # (n, 4) float 0..1
    ground_z: float | None = None


def rotation_to(a: np.ndarray, b: np.ndarray = UP) -> np.ndarray:
    """Rotation matrix turning unit vector a onto unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def look_at_center(centers: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Point closest (least squares) to all camera optical axes: where the user aimed."""
    projs = np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]
    return np.linalg.solve(projs.sum(0), np.einsum("nij,nj->i", projs, centers))


def statistical_inliers(points: np.ndarray, k: int = 16, n_std: float = 2.0) -> np.ndarray:
    d, _ = cKDTree(points).query(points, k=k + 1)
    mean = d[:, 1:].mean(1)
    return mean < mean.mean() + n_std * mean.std()


def find_ground(points: np.ndarray, thresh: float, max_tilt_deg: float = 20, iters: int = 400, seed: int = 0):
    """RANSAC for a near-horizontal plane n.p + d = 0 holding >= 10% of points with < 5% below it.

    The "nothing below" rule keeps a flat step or platform inside the object from being taken for the table.
    Returns (n, d) or None."""
    rng = np.random.default_rng(seed)
    best, best_count = None, int(0.1 * len(points))
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < cos_tilt:
            continue
        d = -n @ p[0]
        dist = points @ n + d
        count = int((np.abs(dist) < thresh).sum())
        if count > best_count and (dist < -thresh).mean() < 0.05:
            best, best_count = (n, d), count
    return best


def estimate_up(cam_centers: np.ndarray, cam_dirs: np.ndarray, cam_ups: np.ndarray) -> np.ndarray:
    mean_up = cam_ups.mean(0)
    if np.linalg.norm(mean_up) > 0.8:  # photos held upright (phone app): image "up" is gravity's opposite
        return mean_up
    # Rolled cameras: use the orbit instead. Cameras circle the object, so their best-fit plane is horizontal,
    # and people photograph objects from slightly above, so the cameras look down on average.
    n = np.linalg.svd(cam_centers - cam_centers.mean(0))[2][2]
    return n if cam_dirs.mean(0) @ n <= 0 else -n


def clean_cloud(
    points: np.ndarray,
    normals: np.ndarray | None,
    colors: np.ndarray,
    cam_centers: np.ndarray,
    cam_dirs: np.ndarray,
    cam_ups: np.ndarray,
) -> Cloud:
    R = rotation_to(estimate_up(cam_centers, cam_dirs, cam_ups))
    center = look_at_center(cam_centers, cam_dirs)
    pts = (points - center) @ R.T
    cams = (cam_centers - center) @ R.T

    # Keep the region the cameras orbit around; drops background walls, far furniture, etc.
    radius = 0.5 * float(np.median(np.linalg.norm(cams, axis=1)))
    keep = np.linalg.norm(pts, axis=1) < radius
    pts, colors = pts[keep], colors[keep]
    nrm = normals[keep] @ R.T if normals is not None else None
    if len(pts) < 500:
        raise MeshError("Too few 3D points near the object. Keep the object centred and fill the frame more.")

    keep = statistical_inliers(pts)
    pts, colors = pts[keep], colors[keep]
    nrm = nrm[keep] if nrm is not None else None

    ground_z = None
    thresh = 0.01 * radius
    plane = find_ground(pts, thresh)
    if plane is not None:
        n, d = plane
        R2 = rotation_to(n)  # level the table exactly, so the flat base is truly flat
        pts = pts @ R2.T
        cams = cams @ R2.T
        nrm = nrm @ R2.T if nrm is not None else None
        ground_z = -d  # n.p + d = 0  ->  z' = -d after rotating n onto +Z
        above = pts[:, 2] > ground_z + thresh
        pts, colors = pts[above], colors[above]
        nrm = nrm[above] if nrm is not None else None

    if nrm is None:
        nrm = estimate_normals(pts)
    # Surfaces face the camera that saw them; flip normals pointing away from their nearest camera.
    _, idx = cKDTree(cams).query(pts)
    flip = np.einsum("ij,ij->i", nrm, cams[idx] - pts) < 0
    nrm[flip] *= -1
    return Cloud(pts, nrm, colors, ground_z)


def estimate_normals(points: np.ndarray) -> np.ndarray:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points))
    ms.compute_normal_for_point_clouds(k=12)
    return ms.current_mesh().vertex_normal_matrix().copy()


def poisson(cloud: Cloud, depth: int = 9, target_faces: int = 200_000) -> trimesh.Trimesh:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(vertex_matrix=cloud.points, v_normals_matrix=cloud.normals, v_color_matrix=cloud.colors)
    )
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)
    if ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces, preservenormal=True, preservetopology=True
        )
    m = ms.current_mesh()
    colors = (np.clip(m.vertex_color_matrix(), 0, 1) * 255).astype(np.uint8)
    return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), vertex_colors=colors)


def close_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Cap every remaining hole (e.g. where Poisson hit its grid boundary on unseen sides), keeping colours."""
    ms = pymeshlab.MeshSet()
    colors = mesh.visual.vertex_colors[:, :4] / 255.0 if mesh.visual.kind == "vertex" else None
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces, v_color_matrix=colors) if colors is not None else pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_close_holes(maxholesize=1_000_000, newfaceselected=False)
    m = ms.current_mesh()
    out_colors = (np.clip(m.vertex_color_matrix(), 0, 1) * 255).astype(np.uint8) if colors is not None else None
    return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), vertex_colors=out_colors)


def floor_z(cloud: Cloud) -> float:
    """Where the object really ends at the bottom: the table, else the lowest observed point."""
    return cloud.ground_z if cloud.ground_z is not None else float(cloud.points[:, 2].min())


def make_printable(mesh: trimesh.Trimesh, floor: float, flat_base: bool, longest_mm: float = 100.0) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if not len(parts):
        raise MeshError("Meshing produced nothing usable.")
    mesh = max(parts, key=lambda p: len(p.faces))  # drops floating fragments

    zmin, zmax = mesh.bounds[:, 2]
    h = zmax - zmin
    # Poisson invents surface below the floor (unseen underside); cut it off.
    cut = floor + (0.02 * h if flat_base else 0.0)
    # Where data was missing the surface runs into Poisson's grid boundary and stays open.
    # If those open edges are near the bottom, cut above them; the cap then closes the mesh.
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    open_z = mesh.vertices[edges[counts == 1].ravel(), 2]
    if len(open_z) and open_z.max() < zmin + 0.35 * h:
        cut = max(cut, open_z.max() + 0.005 * h)
    if cut > zmin:
        mesh = mesh.slice_plane([0, 0, cut], [0, 0, 1], cap=True)

    if not mesh.is_watertight:
        mesh = close_holes(mesh)
    # Slivers left by the cut or hole filling; real parts are far bigger than 1% of the faces.
    parts = mesh.split(only_watertight=False)
    mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.01 * len(mesh.faces)])
    trimesh.repair.fix_normals(mesh)

    # Base centred on the origin at z = 0, longest side = longest_mm (STL has no units; slicers assume mm).
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    mesh.apply_scale(longest_mm / mesh.extents.max())
    return mesh


def report(mesh: trimesh.Trimesh) -> dict:
    return {
        "dimensions_mm": [round(float(v), 2) for v in mesh.extents],
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
    }


MEDIA_TYPES = {
    "stl": "model/stl",
    "obj": "model/obj",
    "glb": "model/gltf-binary",
    "3mf": "model/3mf",
}


def export(mesh: trimesh.Trimesh, fmt: str, longest_mm: float | None = None) -> bytes:
    m = mesh.copy()
    if longest_mm:
        m.apply_scale(longest_mm / m.extents.max())
    if fmt == "glb":
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # Z-up -> glTF Y-up
    return m.export(file_type=fmt)
