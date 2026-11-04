"""Тестът за здравина: ще издържи ли отпечатаният предмет? Линейно-еластичен анализ с крайни елементи на модела.
Моделът (model.ply: z нагоре, стои върху z = 0) се запълва с кубове, всеки тухлен елемент с 8 възела, по CELLS на
най-дългата му страна. Кубовете от долния слой са закрепени отдолу; натоварването зависи от случая. top: тежест,
натискаща право надолу отгоре (обърнатите нагоре стени в най-високата част TOP_BAND от предмета). hang: основата е
закрепена за таван, а от далечния край виси тежест (върхът се дърпа от основата). side: основата е закрепена, а
върхът се бута настрани през по-тясната страна на предмета (по-слабата му посока). drop: собственото тегло на
предмета, което приложението превръща в падане от височина (енергиен баланс, lib/stress.ts). При линейна
еластичност напреженията растат пропорционално на товара, не зависят от твърдостта при зададени сили (един
изотропен материал), а с размера L на предмета вървят като 1/L² (зададено тегло) или като L (собствено тегло).
Затова всеки случай се решава веднъж за модел, при REF_MM и за 1 кг (собствено тегло: при RHO_REF), а приложението
мащабира резултата към избраните от потребителя тегло, височина, материал и реален размер, докато той избира.
Резултатите се пазят до модела. Напреженията са еквивалентни по фон Мизес в MPa, изчислени в ъглите на елементите и
осреднени за възел, така че повърхността на частта получава повърхностното си напрежение (огъването има пик там)."""

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
VERSION = 1  # резултати от друга версия се правят наново
REF_MM = 100.0
CELLS = 64  # елементи по най-дългата страна
MAX_ELEMENTS = 70_000  # по-грубо отвъд това: време и памет
NU = 0.35  # коефициент на Поасон за пластмаси за печат (PLA, PETG, ABS, смоли: 0,33-0,40)
E_REF = 1e9  # Pa: провисването при падане е дадено при тази твърдост и RHO_REF; приложението го мащабира по материала
RHO_REF = 1000.0  # kg/mÂ³
G = 9.81
TOP_BAND = 0.08  # дял от височината отгоре, върху който действа товарът
BASE_LAYERS = 1  # слоеве възли над закрепената долна страна, изключени от пика: ъгълът на самото закрепване не е на предмета
VIEW_FACES = 60_000
RTOL = 1e-6
TIMEOUT_S = 900
USER_ERROR = 2  # процесът на теста: изходен код за StressError (съобщението е в stderr)
_lock = threading.Lock()
log = logging.getLogger("uvicorn.error")


class StressError(Exception):
    pass


# --- Тухленият елемент ----------------------------------------------------------------------------------------------

CORNERS = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])  # възел n = a + 2b + 4c
SIGNS = 2 * CORNERS - 1


def strain_matrix(xi: np.ndarray) -> np.ndarray:
    """B (6 x 24) на единичен куб при естествени координати xi (всяка от -1 до 1): деформации [xx yy zz xy yz zx]
    (инженерни срязвания) от преместванията [x y z] на 8-те възела по реда им."""
    t = 1 + SIGNS * xi
    d = np.empty((8, 3))
    d[:, 0] = SIGNS[:, 0] * t[:, 1] * t[:, 2]
    d[:, 1] = SIGNS[:, 1] * t[:, 0] * t[:, 2]
    d[:, 2] = SIGNS[:, 2] * t[:, 0] * t[:, 1]
    d /= 4  # dN/dxi = d/8 и dxi/dx = 2 при единичен куб
    B = np.zeros((6, 24))
    for n, (dx, dy, dz) in enumerate(d):
        B[:, 3 * n] = [dx, 0, 0, dy, 0, dz]
        B[:, 3 * n + 1] = [0, dy, 0, dx, dz, 0]
        B[:, 3 * n + 2] = [0, 0, dz, 0, dy, dx]
    return B


def elasticity(nu: float) -> np.ndarray:
    """Изотропна матрица напрежение-деформация за E = 1."""
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[range(3, 6), range(3, 6)] = mu
    return D


def element_stiffness(nu: float) -> np.ndarray:
    """Твърдост 24 x 24 на единичен куб с E = 1 (2 x 2 x 2 точки на Гаус); куб със страна h и модул E има E h пъти
    това."""
    D = elasticity(nu)
    K = np.zeros((24, 24))
    for s in SIGNS:
        B = strain_matrix(s / np.sqrt(3))
        K += B.T @ D @ B / 8  # якобианът на единичен куб: (1/2)³
    return K


def von_mises(s: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, yz, zx = np.moveaxis(s, -1, 0)
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + yz**2 + zx**2))


# --- От модел към елементи ------------------------------------------------------------------------------------------


def voxels(mesh: trimesh.Trimesh, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Кубове, чийто център е в модела: заетост (nx, ny, nz) и долният ъгъл на куб (0, 0, 0). По центъра, а не по
    докосване на повърхността: иначе всяка част би станала с около куб по-дебела, а напрежението при огъване на тънка
    част върви с куба на дебелината ѝ. Вътре: нечетен брой пресичания на повърхността под центъра по вертикалата през
    него (всички колони наведнъж: всеки триъгълник намира центровете на колони над себе си). Колона с общо нечетен
    брой (отвор в повърхността) вместо това се запълва от кубовете на повърхността."""
    lo, hi = mesh.bounds
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    origin = (lo + hi) / 2 - n * pitch / 2
    # Съвсем малко ирационално побутване: линия през връх или по ръб иначе би го броила двойно или никак.
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
    t = np.repeat(np.arange(len(tri)), ni * nj)  # по една двойка (триъгълник, колона) за център на колона в кутията му
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

    # Всяко пресичане обръща "вътре" за всеки център над него.
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
    """Кубовете, свързани с цели стени към долния слой: останалите биха се носели свободно (а части, свързани само с
    ръб или ъгъл, биха се въртели около него)."""
    lab, _ = label(occ)  # съседи по стена
    held = np.unique(lab[:, :, 0][occ[:, :, 0]])
    return np.isin(lab, held[held > 0])


class Model:
    """Елементите на запълнен модел: индекси на кубовете, техните възли и степени на свобода."""

    def __init__(self, keep: np.ndarray, origin: np.ndarray, pitch: float):
        self.keep, self.origin, self.pitch = keep, origin, pitch
        self.cubes = np.argwhere(keep)  # (ne, 3)
        nx, ny, nz = keep.shape
        corners = self.cubes[:, None, :] + CORNERS[None]  # (ne, 8, 3) възли на мрежата
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
        """Местни номера на възли на мрежата (всеки трябва да принадлежи на модела)."""
        nx, ny, nz = self.keep.shape
        key = (self.nodes_grid[:, 0] * (ny + 1) + self.nodes_grid[:, 1]) * (nz + 1) + self.nodes_grid[:, 2]
        want = (grid[:, 0] * (ny + 1) + grid[:, 1]) * (nz + 1) + grid[:, 2]
        return np.searchsorted(key, want)  # ключът е сортиран: np.unique го е направил


def build(mesh: trimesh.Trimesh) -> tuple[Model, float]:
    """Елементите на модела при стъпка за около CELLS по най-дългата страна, по-груба, ако това е над MAX_ELEMENTS; и
    делът от кубовете на модела, който отпада (части, несвързани с основата)."""
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


# --- Товари ---------------------------------------------------------------------------------------------------------


def top_faces(keep: np.ndarray) -> np.ndarray:
    """Обърнати нагоре стени без нищо върху тях в най-горната част TOP_BAND от предмета: (n, 3) възли на мрежата от
    долния им ъгъл."""
    above = np.zeros_like(keep)
    above[:, :, :-1] = keep[:, :, 1:]
    faces = np.argwhere(keep & ~above)
    top = faces[:, 2].max()
    band = max(1, round(TOP_BAND * (top + 1)))
    return faces[faces[:, 2] > top - band] + [0, 0, 1]


def face_load(m: Model, faces: np.ndarray, force: np.ndarray) -> np.ndarray:
    """`force` (N, 3-вектор), разделена поравно между стените, с четвърт от дела на всяка стена върху всеки ъгъл."""
    f = np.zeros(m.ndof)
    corners = (faces[:, None, :] + np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])[None]).reshape(-1, 3)
    nodes = m.node_index(corners)
    for d in range(3):
        np.add.at(f, 3 * nodes + d, force[d] / len(corners))
    return f


def weight_load(m: Model, rho: float) -> np.ndarray:
    """Собственото тегло на всеки куб (плътност rho, кг/м³), с една осма върху всеки ъгъл, надолу."""
    f = np.zeros(m.ndof)
    each = rho * (m.pitch / 1000) ** 3 * G / 8
    np.add.at(f, 3 * m.enodes.ravel() + 2, -each)
    return f


# --- Решаване -------------------------------------------------------------------------------------------------------


def solve(m: Model, K: np.ndarray, f: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Премествания (м) за възловите сили f (N), със закрепените степени на свобода при 0. Спрегнати градиенти с
    директно прилагани елементни матрици (никога не се сглобяват: паметта остава малка), с предусловие на Якоби."""
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
    """Напрежение по фон Мизес (MPa) за възел: в ъглите на всеки елемент, осреднено за елементите с общ възел."""
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
    """Стойността на всеки възел, осреднена със (до 6) съседи по линиите на мрежата. Вътрешните ъгли на кубичен модел
    са безкрайно остри, където истинският предмет е заоблен, и напрежението там би растяло без край при по-малки
    кубове: напрежението на ширина около куб е това, с което се сравнява якостта на частта."""
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


# --- Един случай ----------------------------------------------------------------------------------------------------


def to_glb(p: np.ndarray) -> np.ndarray:
    """Координати на модела (z нагоре) към тези на GLB (y нагоре), като mesh.Y_UP."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def analyse(mesh: trimesh.Trimesh, case: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Случаят върху `mesh` (при REF_MM): обобщението и позициите (мм) и напреженията (MPa) на възлите."""
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
        # Средното провисване от теглото: енергията от собственото тегло е (1/2) * тегло * това.
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
    """Напрежението във върховете на мрежата за показване: от най-близките възли, претеглено по близост; -1, където
    няма близък възел (тънка част, която елементите не са могли да закрепят)."""
    d, i = cKDTree(pts).query(view.vertices, k=8, distance_upper_bound=1.5 * pitch)
    ok = np.isfinite(d)
    w = np.where(ok, 1 / (d + 0.25 * pitch), 0.0)
    vals = np.where(ok, vm[np.minimum(i, len(vm) - 1)], 0.0)
    out = (w * vals).sum(1) / np.maximum(w.sum(1), 1e-12)
    return np.where(ok.any(1), out, -1.0).astype(np.float32)


# --- Запазени резултати ---------------------------------------------------------------------------------------------


def paths(job: Path, case: str) -> tuple[Path, Path]:
    return job / f"stress-{case}.json", job / f"stress-{case}.glb"


def make(job: Path, case: str):
    """Изчислява случая за модела на задачата и пази обобщението и модела, оцветяван от приложението."""
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
    """Обобщението на случая, направено сега, ако липсва или е по-старо от модела (повторно обработен). Прави се в
    собствен процес (`python -m app.stress <задача> <случай>`): този OpenBLAS срива целия процес, когато две нишки
    умножават матрици едновременно (наблюдавано при рисуване на миниатюра), а нишките на сървъра го правят. Срив там
    струва само този тест, а не сървъра и обработваните сканирания. По един наведнъж: всеки заема цяло ядро."""
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


# --- За крайните точки (app/main.py: собствени сканирания; app/community.py: публикации)
# -----------------------------


def summary_response(job_id: str, case: str, user_id: str | None) -> dict:
    """Обобщението на случая за завършено сканиране (прави се при първата заявка: до минута) и точките му."""
    from . import jobs, progress  # тук: собственият процес на теста (по-долу) не се нуждае и от двете

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
    """Моделът с напреженията за всеки връх (атрибут "_STRESS": MPa при опорните размер и товар от обобщението, -1
    където част не е могла да се тества), който приложението оцветява."""
    from . import jobs

    if case not in CASES:
        raise HTTPException(404, "Тестът не е намерен.")
    try:
        data = model_glb(jobs.job_dir(job_id), case)
    except StressError as e:
        raise HTTPException(422, str(e))
    return Response(data, media_type=M.MEDIA_TYPES["glb"], headers={"Cache-Control": "private, max-age=300"})


if __name__ == "__main__":  # result() изпълнява това: python -m app.stress <папка на задачата> <случай>
    try:
        make(Path(sys.argv[1]), sys.argv[2])
    except StressError as e:
        print(e, file=sys.stderr)
        sys.exit(USER_ERROR)
