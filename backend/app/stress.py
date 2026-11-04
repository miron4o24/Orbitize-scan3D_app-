"""The stress test: will the printed object hold? A linear-elastic finite element analysis of the model.

The model (model.ply: z up, standing on z = 0) is filled with cubes, each an 8-node brick element, CELLS of them along
its longest side. The cubes of the bottom layer are held at their underside; the load depends on the case:

- top:  a weight pressing straight down on the top (the upward faces in the highest TOP_BAND of the object)
- hang: the base fastened to a ceiling, a weight hanging from the far end (the top pulled away from the base)
- side: the base held, a push sideways on the top, across the object's narrower side (its weaker way)
- drop: the object's own weight, which the app turns into a fall from a height (energy balance, lib/stress.ts)

In linear elasticity the stresses grow in proportion to the load, do not depend on the stiffness when the forces are
given (one isotropic material), and with the object's size L go as 1/LÂ² (a given weight) or as L (its own weight).
So each case is solved once per model, at REF_MM and for 1 kg (own weight: at RHO_REF), and the app scales the result
to the weight, height, material and real size the user picks, while they pick it. Results are kept next to the model.

Stresses are von Mises equivalent stresses in MPa, evaluated at the element corners and averaged per node, so a part's
surface gets its surface stress (bending peaks there)."""

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
from fastapi import HTTPException, Response
from scipy.ndimage import binary_fill_holes, label
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree

from . import mesh as M

CASES = ("top", "hang", "side", "drop")
VERSION = 1  # results of another version are made again
REF_MM = 100.0
CELLS = 64  # elements along the longest side
MAX_ELEMENTS = 70_000  # coarser past this: time and memory
NU = 0.35  # Poisson's ratio of printing plastics (PLA, PETG, ABS, resins: 0.33-0.40)
E_REF = 1e9  # Pa: the drop's sag is given at this stiffness and RHO_REF; the app scales by the material's
RHO_REF = 1000.0  # kg/mÂ³
G = 9.81
TOP_BAND = 0.08  # share of the height, at the top, that the load acts on
BASE_LAYERS = 1  # node layers above the held underside left out of the peak: the clamp's own corner is not the object's
VIEW_FACES = 60_000
RTOL = 1e-6
TIMEOUT_S = 900
USER_ERROR = 2  # the test's process: exit code for a StressError (its message on stderr)
_lock = threading.Lock()
log = logging.getLogger("uvicorn.error")


class StressError(Exception):
    pass


# --- The brick element ------------------------------------------------------------------------------------------

CORNERS = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])  # node n = a + 2b + 4c
SIGNS = 2 * CORNERS - 1


def strain_matrix(xi: np.ndarray) -> np.ndarray:
    """B (6 x 24) of a unit cube at natural coordinates xi (each -1..1): strains [xx yy zz xy yz zx] (engineering
    shears) from the 8 nodes' displacements [x y z] in node order."""
    t = 1 + SIGNS * xi
    d = np.empty((8, 3))
    d[:, 0] = SIGNS[:, 0] * t[:, 1] * t[:, 2]
    d[:, 1] = SIGNS[:, 1] * t[:, 0] * t[:, 2]
    d[:, 2] = SIGNS[:, 2] * t[:, 0] * t[:, 1]
    d /= 4  # dN/dxi = d/8, and dxi/dx = 2 on a unit cube
    B = np.zeros((6, 24))
    for n, (dx, dy, dz) in enumerate(d):
        B[:, 3 * n] = [dx, 0, 0, dy, 0, dz]
        B[:, 3 * n + 1] = [0, dy, 0, dx, dz, 0]
        B[:, 3 * n + 2] = [0, 0, dz, 0, dy, dx]
    return B


def elasticity(nu: float) -> np.ndarray:
    """Isotropic stress-strain matrix for E = 1."""
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[range(3, 6), range(3, 6)] = mu
    return D


def element_stiffness(nu: float) -> np.ndarray:
    """24 x 24 stiffness of a unit cube with E = 1 (2 x 2 x 2 Gauss points); a cube of side h with modulus E has E h
    times this."""
    D = elasticity(nu)
    K = np.zeros((24, 24))
    for s in SIGNS:
        B = strain_matrix(s / np.sqrt(3))
        K += B.T @ D @ B / 8  # the Jacobian of a unit cube: (1/2)Â³
    return K


def von_mises(s: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, yz, zx = np.moveaxis(s, -1, 0)
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + yz**2 + zx**2))


# --- Model to elements ------------------------------------------------------------------------------------------


def voxels(mesh: trimesh.Trimesh, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Cubes whose centre is inside the model: occupancy (nx, ny, nz) and the low corner of cube (0, 0, 0).

    By the centre rather than by touching the surface: that would make every part about a cube thicker, and a thin
    part's bending stress goes with the cube of its thickness. Inside: an odd number of surface crossings below the
    centre, along the vertical line through it (all columns at once: each triangle finds the column centres over it).
    A column with an odd count in all (an opening in the surface) is filled from the surface's cubes instead."""
    lo, hi = mesh.bounds
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    origin = (lo + hi) / 2 - n * pitch / 2
    # A tiny irrational nudge: a line through a vertex or along an edge would count it twice or not at all.
    cx = origin[0] + (np.arange(n[0]) + 0.5) * pitch + pitch * 1.3e-6 * np.sqrt(2)
    cy = origin[1] + (np.arange(n[1]) + 0.5) * pitch + pitch * 1.1e-6 * np.sqrt(3)
    cz = origin[2] + (np.arange(n[2]) + 0.5) * pitch

    tri = mesh.triangles
    tlo, thi = tri[:, :, :2].min(1), tri[:, :, :2].max(1)
    i0 = np.ceil((tlo[:, 0] - cx[0]) / pitch).astype(int).clip(0, n[0])
    i1 = np.floor((thi[:, 0] - cx[0]) / pitch).astype(int).clip(-1, n[0] - 1)
    j0 = np.ceil((tlo[:, 1] - cy[0]) / pitch).astype(int).clip(0, n[1])
    j1 = np.floor((thi[:, 1] - cy[0]) / pitch).astype(int).clip(-1, n[1] - 1)
    ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    t = np.repeat(np.arange(len(tri)), ni * nj)  # one (triangle, column) pair per column centre in its box
    within = np.arange(len(t)) - np.repeat(np.cumsum(ni * nj) - ni * nj, ni * nj)
    ci = i0[t] + within % np.maximum(ni[t], 1)
    cj = j0[t] + within // np.maximum(ni[t], 1)
    a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
    px, py = cx[ci], cy[cj]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(det) > 1e-18
    det = np.where(ok, det, 1.0)
    w0 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / det
    w1 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / det
    w2 = 1 - w0 - w1
    hit = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    z = (w0 * a[:, 2] + w1 * b[:, 2] + w2 * c[:, 2])[hit]
    col = (ci * n[1] + cj)[hit]

    # Each crossing turns "inside" over for every centre above it.
    flips = np.zeros((n[0] * n[1], n[2] + 1), np.int32)
    np.add.at(flips, (col, np.searchsorted(cz, z)), 1)
    occ = (np.cumsum(flips, 1)[:, :-1] % 2).astype(bool).reshape(n[0], n[1], n[2])
    odd = (np.bincount(col, minlength=n[0] * n[1]) % 2).astype(bool).reshape(n[0], n[1])
    if odd.any():
        v = mesh.voxelized(pitch)
        shell = binary_fill_holes(v.matrix)
        at = np.round((v.indices_to_points(np.argwhere(shell)) - origin) / pitch - 0.5).astype(int)
        inside = (at >= 0).all(1) & (at < n).all(1)
        filled = np.zeros_like(occ)
        filled[tuple(at[inside].T)] = True
        occ[odd] = filled[odd]
    return occ, origin


def held_part(occ: np.ndarray) -> np.ndarray:
    """The cubes joined by whole faces to the bottom layer: what is not would float free (and pieces joined only by an
    edge or a corner would turn on it)."""
    lab, _ = label(occ)  # face neighbours
    held = np.unique(lab[:, :, 0][occ[:, :, 0]])
    return np.isin(lab, held[held > 0])


class Model:
    """The elements of a filled model: cube indices, their nodes and degrees of freedom."""

    def __init__(self, keep: np.ndarray, origin: np.ndarray, pitch: float):
        self.keep, self.origin, self.pitch = keep, origin, pitch
        self.cubes = np.argwhere(keep)  # (ne, 3)
        nx, ny, nz = keep.shape
        corners = self.cubes[:, None, :] + CORNERS[None]  # (ne, 8, 3) grid nodes
        flat = (corners[..., 0] * (ny + 1) + corners[..., 1]) * (nz + 1) + corners[..., 2]
        ids, local = np.unique(flat, return_inverse=True)
        self.nodes_grid = np.stack(np.unravel_index(ids, (nx + 1, ny + 1, nz + 1)), 1)  # (nn, 3)
        self.enodes = local.reshape(-1, 8)
        self.edofs = (3 * self.enodes[:, :, None] + np.arange(3)).reshape(-1, 24)
        self.ndof = 3 * len(ids)

    @property
    def node_points(self) -> np.ndarray:
        return self.origin + self.nodes_grid * self.pitch

    def node_index(self, grid: np.ndarray) -> np.ndarray:
        """Local node numbers of grid nodes (each must belong to the model)."""
        nx, ny, nz = self.keep.shape
        key = (self.nodes_grid[:, 0] * (ny + 1) + self.nodes_grid[:, 1]) * (nz + 1) + self.nodes_grid[:, 2]
        want = (grid[:, 0] * (ny + 1) + grid[:, 1]) * (nz + 1) + grid[:, 2]
        return np.searchsorted(key, want)  # key is sorted: np.unique made it


def build(mesh: trimesh.Trimesh) -> tuple[Model, float]:
    """The model's elements at a pitch giving about CELLS along the longest side, coarser if that would be more than
    MAX_ELEMENTS; and the share of the model's cubes left out (pieces not joined to the base)."""
    pitch = mesh.extents.max() / CELLS
    for _ in range(4):
        occ, origin = voxels(mesh, pitch)
        if occ.sum() <= MAX_ELEMENTS:
            break
        pitch *= (occ.sum() / MAX_ELEMENTS) ** (1 / 3) * 1.02
    if not occ[:, :, 0].any() or occ.sum() < 8:
        raise StressError("Моделът е твърде тънък или малък за теста.")
    keep = held_part(occ)
    return Model(keep, origin, pitch), 1 - keep.sum() / occ.sum()


# --- Loads ------------------------------------------------------------------------------------------------------


def top_faces(keep: np.ndarray) -> np.ndarray:
    """Upward faces with nothing on them in the top TOP_BAND of the object: (n, 3) grid nodes of their low corner."""
    above = np.zeros_like(keep)
    above[:, :, :-1] = keep[:, :, 1:]
    faces = np.argwhere(keep & ~above)
    top = faces[:, 2].max()
    band = max(1, round(TOP_BAND * (top + 1)))
    return faces[faces[:, 2] > top - band] + [0, 0, 1]


def face_load(m: Model, faces: np.ndarray, force: np.ndarray) -> np.ndarray:
    """`force` (N, a 3-vector) shared evenly by the faces, a quarter of each face's share on each of its corners."""
    f = np.zeros(m.ndof)
    corners = (faces[:, None, :] + np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])[None]).reshape(-1, 3)
    nodes = m.node_index(corners)
    for d in range(3):
        np.add.at(f, 3 * nodes + d, force[d] / len(corners))
    return f


def weight_load(m: Model, rho: float) -> np.ndarray:
    """Each cube's own weight (density rho, kg/mÂ³), an eighth on each of its corners, downwards."""
    f = np.zeros(m.ndof)
    each = rho * (m.pitch / 1000) ** 3 * G / 8
    np.add.at(f, 3 * m.enodes.ravel() + 2, -each)
    return f


# --- Solving ----------------------------------------------------------------------------------------------------


def solve(m: Model, K: np.ndarray, f: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Displacements (m) for the nodal forces f (N), with the held degrees of freedom at 0. Conjugate gradients with the
    element matrices applied directly (never assembled: memory stays small), Jacobi preconditioned."""
    h = m.pitch / 1000
    free = np.ones(m.ndof, bool)
    free[held] = False
    flat = m.edofs.ravel()

    def apply(x):
        y = np.bincount(flat, ((x * free)[m.edofs] @ K).ravel(), minlength=m.ndof)
        return np.where(free, y, x)

    diag = np.bincount(flat, np.tile(np.diag(K), len(m.edofs)), minlength=m.ndof)
    diag = np.where(free & (diag > 0), diag, 1.0)
    A = LinearOperator((m.ndof, m.ndof), matvec=apply, dtype=float)
    P = LinearOperator((m.ndof, m.ndof), matvec=lambda r: r / diag, dtype=float)
    u, info = cg(A, f * free / (E_REF * h), rtol=RTOL, maxiter=20_000, M=P)
    if info != 0:
        raise StressError("Изчислението не успя да стигне до резултат. Опитайте с друг размер.")
    return u * free


def nodal_stress(m: Model, u: np.ndarray) -> np.ndarray:
    """Von Mises stress (MPa) per node: at each element's corners, averaged over the elements sharing a node."""
    D = elasticity(NU) * E_REF
    ue = u[m.edofs]  # (ne, 24)
    h = m.pitch / 1000
    total = np.zeros(m.ndof // 3)
    for n, s in enumerate(SIGNS):
        sigma = (ue @ strain_matrix(s.astype(float)).T / h) @ D.T  # (ne, 6), Pa
        np.add.at(total, m.enodes[:, n], von_mises(sigma))
    count = np.bincount(m.enodes.ravel(), minlength=len(total))
    return smoothed(m, total / count / 1e6)


def smoothed(m: Model, v: np.ndarray) -> np.ndarray:
    """Each node's value averaged with its (up to 6) neighbours along the grid lines. A cube model's inner corners are
    infinitely sharp, where a real object is rounded, and the stress there would grow without end as the cubes get
    smaller: a stress over about a cube's width is what a part's strength is measured against."""
    nx, ny, nz = m.keep.shape
    g = m.nodes_grid
    key = (g[:, 0] * (ny + 1) + g[:, 1]) * (nz + 1) + g[:, 2]
    total, count = v.copy(), np.ones(len(v))
    for d in np.vstack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]):
        nb = g + d
        k = (nb[:, 0] * (ny + 1) + nb[:, 1]) * (nz + 1) + nb[:, 2]
        at = np.searchsorted(key, k).clip(0, len(key) - 1)
        ok = (key[at] == k) & (nb >= 0).all(1)
        total[ok] += v[at[ok]]
        count[ok] += 1
    return total / count


# --- One case ---------------------------------------------------------------------------------------------------


def to_glb(p: np.ndarray) -> np.ndarray:
    """Model coordinates (z up) to the GLB's (y up), as mesh.Y_UP."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def analyse(mesh: trimesh.Trimesh, case: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """The case on `mesh` (at REF_MM): the summary, and the nodes' positions (mm) and stresses (MPa)."""
    started = time.time()
    m, dropped = build(mesh)
    K = element_stiffness(NU)
    grid = m.nodes_grid
    held = (3 * np.flatnonzero(grid[:, 2] == 0)[:, None] + np.arange(3)).ravel()
    summary: dict = {}
    if case == "drop":
        f = weight_load(m, RHO_REF)
        u = solve(m, K, f, held)
        weight = -f[2::3]
        # The weight's mean sag: the energy the own weight stores is (1/2) * weight * this.
        summary["sag_m"] = float((weight * -u[2::3]).sum() / weight.sum())
    else:
        faces = top_faces(m.keep)
        ext = mesh.extents
        direction = {"top": [0, 0, -1], "hang": [0, 0, 1], "side": [1, 0, 0] if ext[0] <= ext[1] else [0, 1, 0]}[case]
        u = solve(m, K, face_load(m, faces, np.array(direction, float) * G), held)  # 1 kg
        centre = m.origin + (faces + [0.5, 0.5, 0]).mean(0) * m.pitch
        summary |= {"load_at": to_glb(centre).round(2).tolist(), "load_dir": to_glb(direction).tolist()}
    vm = nodal_stress(m, u)
    counted = grid[:, 2] > BASE_LAYERS
    if not counted.any():
        counted[:] = True
    peak = int(np.flatnonzero(counted)[np.argmax(vm[counted])])
    pts = m.node_points
    volume = mesh.volume if mesh.is_watertight else m.keep.sum() * m.pitch**3
    summary |= {
        "case": case,
        "version": VERSION,
        "ref_mm": REF_MM,
        "max_mpa": float(vm[peak]),
        "weak": to_glb(pts[peak]).round(2).tolist(),
        "volume_cm3": round(float(volume) / 1000, 3),
        "elements": int(len(m.cubes)),
        "pitch_mm": round(float(m.pitch), 3),
        "dropped": round(float(dropped), 4),
        "seconds": round(time.time() - started, 1),
    }
    return summary, pts, vm


def view_values(view: trimesh.Trimesh, pts: np.ndarray, vm: np.ndarray, pitch: float) -> np.ndarray:
    """The stress at the view mesh's vertices: from the nearest nodes, weighted by nearness; -1 where no node is near
    (a thin piece the elements could not hold)."""
    d, i = cKDTree(pts).query(view.vertices, k=8, distance_upper_bound=1.5 * pitch)
    ok = np.isfinite(d)
    w = np.where(ok, 1 / (d + 0.25 * pitch), 0.0)
    vals = np.where(ok, vm[np.minimum(i, len(vm) - 1)], 0.0)
    out = (w * vals).sum(1) / np.maximum(w.sum(1), 1e-12)
    return np.where(ok.any(1), out, -1.0).astype(np.float32)


# --- Kept results -----------------------------------------------------------------------------------------------


def paths(job: Path, case: str) -> tuple[Path, Path]:
    return job / f"stress-{case}.json", job / f"stress-{case}.glb"


def make(job: Path, case: str):
    """Works the case out for the job's model and keeps the summary and the coloured-by-the-app model."""
    js, glb = paths(job, case)
    model = job / "model.ply"
    mesh = trimesh.load(model, force="mesh")
    mesh.apply_scale(REF_MM / mesh.extents.max())
    summary, pts, vm = analyse(mesh, case)
    summary["model_mtime"] = model.stat().st_mtime
    plain = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    view = M.decimate(plain, VIEW_FACES, keep_closed=False) if len(plain.faces) > VIEW_FACES else plain
    values = view_values(view, pts, vm, summary["pitch_mm"])
    view.apply_transform(M.Y_UP)
    tmp = glb.with_suffix(".tmp")
    tmp.write_bytes(M.glb(view, extra={"_STRESS": values}))
    tmp.replace(glb)
    tmp = js.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary))
    tmp.replace(js)


def result(job: Path, case: str) -> dict:
    """The case's summary, made now if missing or older than the model (re-processed).

    Made in a process of its own (`python -m app.stress <job> <case>`): this OpenBLAS crashes the whole process when
    two threads multiply matrices at once (seen with a thumbnail being drawn), and the server's threads do. A crash
    there costs this one test, not the server and the scans it is processing. One at a time: each uses a core fully."""
    if case not in CASES:
        raise StressError("Тестът не е намерен.")
    js, glb = paths(job, case)
    model = job / "model.ply"

    def fresh() -> dict | None:
        try:
            s = json.loads(js.read_text())
        except (OSError, ValueError):
            return None
        ok = s.get("version") == VERSION and s.get("model_mtime") == model.stat().st_mtime and glb.exists()
        return s if ok else None

    if (s := fresh()) is not None:
        return s
    with _lock:
        if (s := fresh()) is not None:
            return s
        try:
            run = subprocess.run([sys.executable, "-m", "app.stress", str(job), case], cwd=Path(__file__).parents[1],
                                 capture_output=True, text=True, timeout=TIMEOUT_S)  # fmt: skip
        except subprocess.TimeoutExpired:
            raise StressError("Изчислението продължи твърде дълго. Опитайте отново.")
        if run.returncode == USER_ERROR:
            raise StressError(run.stderr.strip().splitlines()[-1])
        if run.returncode != 0 or (s := fresh()) is None:
            log.error("stress test %s %s: exit %s\n%s", job.name, case, run.returncode, run.stderr[-3000:])
            raise StressError("Изчислението не успя. Опитайте отново.")
        return s


def model_glb(job: Path, case: str) -> bytes:
    result(job, case)
    return paths(job, case)[1].read_bytes()


# --- For the endpoints (app/main.py: own scans; app/community.py: posts) ----------------------------------------


def summary_response(job_id: str, case: str, user_id: str | None) -> dict:
    """The case's summary for a finished scan (made on the first request: up to a minute), and its points."""
    from . import jobs, progress  # here: the test's own process (below) needs neither

    if case not in CASES:
        raise HTTPException(404, "Тестът не е намерен.")
    try:
        summary = result(jobs.job_dir(job_id), case)
    except StressError as e:
        raise HTTPException(422, str(e))
    if user_id:
        progress.stress_done(user_id, job_id, case)
    return {k: v for k, v in summary.items() if k != "model_mtime"}


def model_response(job_id: str, case: str) -> Response:
    """The model with its stresses per vertex (the "_STRESS" attribute: MPa at the summary's reference size and load,
    -1 where a piece could not be tested), which the app colours."""
    from . import jobs

    if case not in CASES:
        raise HTTPException(404, "Тестът не е намерен.")
    try:
        data = model_glb(jobs.job_dir(job_id), case)
    except StressError as e:
        raise HTTPException(422, str(e))
    return Response(data, media_type=M.MEDIA_TYPES["glb"], headers={"Cache-Control": "private, max-age=300"})


if __name__ == "__main__":  # result() runs this: python -m app.stress <job dir> <case>
    try:
        make(Path(sys.argv[1]), sys.argv[2])
    except StressError as e:
        print(e, file=sys.stderr)
        sys.exit(USER_ERROR)
