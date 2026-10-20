"""Photos -> camera poses (COLMAP) -> dense points (OpenMVS) -> printable mesh (app.mesh)."""

import os
import shutil
import struct
from pathlib import Path
from typing import Callable

import numpy as np
import pymeshlab

from . import mesh as M

THREADS = str(os.cpu_count() or 2)
MAX_IMAGE = os.environ.get("MAX_IMAGE_SIZE", "1600")
MAX_POINTS = 1_500_000  # keeps cleanup/Poisson within a few GB of RAM


class Failed(Exception):
    """User-facing failure message."""


def read_images_bin(path: Path):
    """Camera centres, viewing directions and 'up' vectors from a COLMAP 3.x images.bin."""
    centers, dirs, ups = [], [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack("<I7dI", f.read(64))
            while f.read(1) != b"\0":  # image name
                pass
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, 1)
            R = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            centers.append(-R.T @ [tx, ty, tz])
            dirs.append(R[2])  # camera +Z (forward) in world
            ups.append(-R[1])  # camera -Y (image up) in world
    return np.array(centers), np.array(dirs), np.array(ups)


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


def run(
    job: Path,
    flat_base: bool,
    set_state: Callable[[str], None],
    sh: Callable[[str, list], None],
) -> dict:
    """sh(step_name, argv) runs one external tool and raises on failure. Returns the print report."""
    photos = job / "photos"
    n_photos = len(list(photos.glob("*.jp*g")))
    work = job / "work"
    shutil.rmtree(work, ignore_errors=True)
    sparse, dense, db = work / "sparse", work / "dense", work / "db.db"
    sparse.mkdir(parents=True)

    set_state("reconstructing")
    sh("feature extraction", ["colmap", "feature_extractor", "--database_path", db, "--image_path", photos,
        "--ImageReader.single_camera", "1", "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.max_image_size", MAX_IMAGE, "--SiftExtraction.num_threads", THREADS])  # fmt: skip
    sh("photo matching", ["colmap", "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", THREADS])  # fmt: skip
    sh("camera solving", ["colmap", "mapper", "--database_path", db, "--image_path", photos,
        "--output_path", sparse, "--Mapper.num_threads", THREADS])  # fmt: skip

    # The mapper can split into several partial models; keep the one that placed the most photos.
    models = [(d, len(read_images_bin(d / "images.bin")[0])) for d in sparse.iterdir() if (d / "images.bin").exists()]
    best, placed = max(models, key=lambda m: m[1], default=(None, 0))
    if best is None or placed < max(8, n_photos // 2):
        raise Failed(
            f"Only {placed} of {n_photos} photos could be placed in 3D. Not enough overlap or texture: "
            "move slower around the object, keep it in view, and avoid shiny, transparent or plain objects."
        )

    sh("undistorting", ["colmap", "image_undistorter", "--image_path", photos, "--input_path", best,
        "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", MAX_IMAGE])  # fmt: skip
    sh("dense setup", ["InterfaceCOLMAP", "-w", dense, "-i", dense, "-o", dense / "scene.mvs",
        "--image-folder", dense / "images"])  # fmt: skip
    sh("dense matching", ["DensifyPointCloud", "-w", dense, "-i", dense / "scene.mvs", "-o", dense / "dense.mvs",
        "--resolution-level", "1", "--max-threads", THREADS])  # fmt: skip

    set_state("meshing")
    ply = dense / "dense.ply"
    if not ply.exists():
        raise Failed("Dense reconstruction produced no points. The object may be too shiny, plain or blurry.")
    pts, nrm, col = load_cloud(ply)
    cloud = M.clean_cloud(pts, nrm, col, *read_images_bin(dense / "sparse" / "images.bin"))
    mesh = M.poisson(cloud)

    set_state("cleaning")
    out = M.make_printable(mesh, M.floor_z(cloud), flat_base)
    out.export(job / "model.ply")
    shutil.rmtree(work, ignore_errors=True)
    return M.report(out) | {"photos_used": placed, "photos_total": n_photos}
