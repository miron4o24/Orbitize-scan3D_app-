"""Истинският размер на сканиран модел по мерките на потребителя (инженерен режим „Точна част“): дължина, ширина и височина
с линийка. Формата идва от AI модела на сървъра (сканирането от видео, app/ai.py). Първо се проверява дали пропорциите на
сканирането отговарят на мерките (до AGREE): тогава моделът се поставя легнал по собствените си оси (дължина по x, височина
по z, както е мерен) и всяка ос става точно мярката. Ако не отговарят (малка или тънка част: флашка от 56 x 17 x 10 мм излезе
двойно по-широка и наклонена), частта се изгражда от кадрите на същото видео: изгледите отгоре, отстрани и от края се избират
сами, очертанията им се изрязват почти пиксел по пиксел и се мащабират по мерките (app/views3d.py). Така никога не се дава
модел, чийто размер се разминава с линийката. Текстурата от снимките се премества заедно с модела."""

import json
import sys

import numpy as np
import trimesh

from . import mesh as M
from . import pipeline as P
from . import texture as TX

AGREE = 0.12  # пропорциите на сканирането може да се разминават с мерките с толкова (иначе: от изгледите)
NAMES = ("length", "width", "height")


def wanted(size_mm: list | None) -> list[float | None]:
    out = [float(v) if isinstance(v, (int, float)) and 0 < float(v) <= 2000 else None for v in (size_mm or [])[:3]]
    return out + [None] * (3 - len(out))


def lying(model: trimesh.Trimesh) -> np.ndarray:
    """Завъртане (4x4), след което моделът лежи по собствените си оси: най-дългата по x, най-късата по z, началото в средата;
    без огледално обръщане и с горната страна на сканирането нагоре, доколкото може."""
    T, ext = trimesh.bounds.oriented_bounds(model)
    order = np.argsort(ext)[::-1]  # дължина, ширина, височина
    Pm = np.zeros((3, 3))
    Pm[np.arange(3), order] = 1.0
    if np.linalg.det(Pm) < 0:
        Pm[1] *= -1
    R = np.eye(4)
    R[:3, :3] = Pm
    A = R @ T
    if (A[:3, :3] @ [0, 0, 1.0])[2] < 0:  # сканирането беше с горната страна нагоре: така да остане
        A = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]) @ A
    return A


def move_texture(job, A: np.ndarray) -> None:
    parts = TX.load(job)
    if not parts:
        return
    L = A[:3, :3]
    inv_t = np.linalg.inv(L).T
    arrays = {"count": len(parts)}
    for n, p in enumerate(parts):
        normals = p["normals"] @ inv_t.T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        arrays |= {f"vertices_{n}": trimesh.transform_points(p["vertices"], A).astype(np.float32),
                   f"normals_{n}": normals.astype(np.float32), f"uv_{n}": p["uv"], f"faces_{n}": p["faces"]}  # fmt: skip
    np.savez_compressed(job / TX.NPZ, **arrays)


def apply(job, size_mm: list | None, sh=None) -> dict:
    """Мащабира model.ply (и текстурата) по мерките или го изгражда от изгледите; връща полетата за отчета."""
    size = wanted(size_mm)
    given = [(i, v) for i, v in enumerate(size) if v]
    if not given:
        return {}
    model = trimesh.load(job / "model.ply", force="mesh", process=False)
    A = lying(model)
    flat = model.copy()
    flat.apply_transform(A)
    have = flat.extents  # дължина, ширина, височина в единиците на модела
    mm = float(np.median([v / have[i] for i, v in given]))  # мм на единица
    errors = {NAMES[i]: round(float(v / (mm * have[i]) - 1), 3) for i, v in given}
    P.note(job, f"true size: measured {size} mm, scan lying flat {np.round(have * mm, 2).tolist()} mm at the common scale, "
                f"differences {errors}")  # fmt: skip
    if max(abs(e) for e in errors.values()) > AGREE and sh is not None:
        try:
            return from_views(job, size, sh)
        except Exception as e:  # пак е по-добре от нищо: сканирането по мерките, с бележка
            if type(e).__name__ == "Cancelled":
                raise
            P.note(job, f"true size: building from the views failed ({e}); keeping the scan")
    stretch = np.ones(3)
    for i, v in given:
        stretch[i] = v / (mm * have[i])
    real = have * mm * stretch
    k = 100.0 / float(real.max())  # моделът се пази с най-дълга страна 100, а истинската е real_mm
    S = np.diag([*(mm * stretch * k), 1.0])
    lo, hi = flat.bounds
    centre = trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    T = S @ centre @ A
    model.apply_transform(T)
    model.export(job / "model.ply")
    move_texture(job, T)
    for f in job.glob("view_*.glb*"):  # лекият файл за преглед се прави отново от новия модел
        f.unlink(missing_ok=True)
    dims = [round(float(v), 2) for v in real]
    P.note(job, f"true size: from the scan, final {dims} mm")
    return M.report(model) | {"real_mm": round(float(real.max()), 2), "size_given": True, "size_mm": dims, "size_from": "scan"}


def from_views(job, size: list[float | None], sh) -> dict:
    """Частта от изгледите в кадрите на видеото (app/views3d.py), в мм по мерките."""
    from . import views3d

    frames = sorted((job / "photos").glob("*.jp*g"))
    if not frames:
        raise ValueError("no frames kept")
    work = job / "work"
    work.mkdir(exist_ok=True)
    out = work / "views.ply"
    measures = ",".join(f"{v or 0:g}" for v in size)
    sh("measuring the views", [sys.executable, views3d.__file__, out, measures, *[f"photo={f}" for f in frames]])
    info = json.loads(out.with_suffix(".json").read_text())
    part = trimesh.load(out, force="mesh", process=False)
    real = float(max(part.extents))
    printable = M.make_printable(part, float(part.bounds[0, 2]), False, smoothing=0)
    printable.export(job / "model.ply")
    TX.clear(job)  # цветовете са в самия модел (от изгледите)
    for f in job.glob("view_*.glb*"):
        f.unlink(missing_ok=True)
    P.note(job, f"true size: from the views of the video, {info}")
    return M.report(printable) | {"real_mm": round(real, 2), "size_given": True, "size_mm": info.get("size"), "size_from": "views"}
