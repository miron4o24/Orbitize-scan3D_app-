"""Run the pipeline on a folder of photos, keep intermediates, print diagnostics. Usage: python scripts/debug_run.py <photos> <out>"""
import shutil, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import mesh as M, pipeline as P  # noqa: E402

photos, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
if not (out / "photos").exists():
    shutil.copytree(photos, out / "photos")
dense = out / "work" / "dense"
if not (dense / "dense.ply").exists():
    real_rmtree = shutil.rmtree
    P.shutil.rmtree = lambda p, **k: None if Path(p).name == "work" and (Path(p) / "dense").exists() else real_rmtree(p, **k)
    try:
        P.run(out, True, print, lambda step, argv: subprocess.run([str(a) for a in argv], check=True, stdout=subprocess.DEVNULL))
    except Exception as e:
        print("pipeline error:", e)

pts, nrm, col = P.load_cloud(dense / "dense.ply")
centers, dirs, ups = P.read_images_bin(dense / "sparse" / "images.bin")
print("dense points", len(pts), "normals", nrm is not None, "cameras", len(centers))
print("mean up", ups.mean(0).round(3), "|mean up|", np.linalg.norm(ups.mean(0)).round(3))
c = M.look_at_center(centers, dirs)
print("look-at", c.round(3), "cam dist median", np.median(np.linalg.norm(centers - c, axis=1)).round(3))
print("points dist from look-at pct 50/90/99", np.percentile(np.linalg.norm(pts - c, axis=1), [50, 90, 99]).round(3))
cloud = M.clean_cloud(pts, nrm, col, centers, dirs, ups)
print("clean points", len(cloud.points), "ground_z", cloud.ground_z, "z range", cloud.points[:, 2].min().round(3), cloud.points[:, 2].max().round(3))
np.save(out / "clean_points.npy", cloud.points)
np.save(out / "raw_points.npy", pts)
