"""Re-run cleanup+meshing on a debug_run.py output folder. Usage: python scripts/debug_mesh.py <out>"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import mesh as M, pipeline as P  # noqa: E402

out = Path(sys.argv[1]); dense = out / "work" / "dense"
pts, nrm, col = P.load_cloud(dense / "dense.ply")
cams = P.read_images_bin(dense / "sparse" / "images.bin")
print("up", M.estimate_up(*cams).round(3))
cloud = M.clean_cloud(pts, nrm, col, *cams)
print("clean", len(cloud.points), "ground", cloud.ground_z, "z", cloud.points[:, 2].min().round(3), cloud.points[:, 2].max().round(3))
mesh = M.poisson(cloud)
print("poisson faces", len(mesh.faces), "watertight", mesh.is_watertight)
final = M.make_printable(mesh, M.floor_z(cloud), True)
print("report", M.report(final))
final.export(out / "final.stl")
