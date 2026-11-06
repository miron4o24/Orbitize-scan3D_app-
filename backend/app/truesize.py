"""Истинският размер на сканиран модел по мерките на потребителя (инженерен режим „Точна част“): дължина, ширина и височина
с линийка. Формата идва от AI модела на сървъра (сканирането от видео, app/ai.py), а размерът по всяка ос става точно мярката:
общ мащаб от мерките и малко разтягане по всяка измерена ос (сканирането греши пропорциите с няколко процента), така че
отпечатаното копие е 1:1 по трите оси. Текстурата от снимките се премества заедно с модела.

Осите на готовия модел (app/mesh.py make_printable): z е височината (основата на масата), x и y са по най-малкия правоъгълник
около него; дължината е по-дългата от двете."""

import numpy as np
import trimesh

from . import mesh as M
from . import pipeline as P
from . import texture as TX

STRETCH = 0.2  # една ос се разтяга най-много с толкова; повече значи, че мярката е за друга страна, и не се пипа


def wanted(size_mm: list | None) -> list[float | None]:
    out = [float(v) if isinstance(v, (int, float)) and 0 < float(v) <= 2000 else None for v in (size_mm or [])[:3]]
    return out + [None] * (3 - len(out))


def axes_of(extents: np.ndarray) -> list[int]:
    """Коя ос е дължината, ширината и височината."""
    x, y, _ = extents
    return [0, 1, 2] if x >= y else [1, 0, 2]


def apply(job, size_mm: list | None) -> dict:
    """Мащабира model.ply (и текстурата) по мерките; връща полетата за отчета (празно, ако няма мерки)."""
    size = wanted(size_mm)
    given = [(i, v) for i, v in enumerate(size) if v]
    if not given:
        return {}
    model = trimesh.load(job / "model.ply", force="mesh", process=False)
    axes = axes_of(model.extents)
    have = model.extents[axes]  # дължина, ширина, височина в единиците на модела
    mm = float(np.median([v / have[i] for i, v in given]))  # мм на единица
    stretch = np.ones(3)
    skipped = []
    for i, v in given:
        f = v / (mm * have[i])
        if abs(f - 1) <= STRETCH:
            stretch[axes[i]] = f
        else:
            skipped.append(["length", "width", "height"][i])
    real = model.extents * mm * stretch  # мм по осите x, y, z
    k = 100.0 / float(real.max())  # моделът се пази с най-дълга страна 100, а истинската е real_mm
    T = np.diag([*(mm * stretch * k), 1.0])
    model.apply_transform(T)
    model.export(job / "model.ply")
    parts = TX.load(job)
    if parts:  # текстурата е в същите координати: същото преобразуване (нормалите с обратното)
        arrays = {"count": len(parts)}
        d = np.diag(T)[:3]
        for n, p in enumerate(parts):
            normals = p["normals"] / d
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            arrays |= {f"vertices_{n}": (p["vertices"] * d).astype(np.float32), f"normals_{n}": normals.astype(np.float32),
                       f"uv_{n}": p["uv"], f"faces_{n}": p["faces"]}  # fmt: skip
        np.savez_compressed(job / TX.NPZ, **arrays)
    for f in job.glob("view_*.glb*"):  # лекият файл за преглед се прави отново от новия модел
        f.unlink(missing_ok=True)
    dims = [round(float(real[a]), 2) for a in axes]
    P.note(job, f"true size: measured {size} mm, scan {np.round(have * mm, 2).tolist()} mm at the common scale, "
                f"stretch {np.round(stretch[axes], 3).tolist()}, final {dims} mm" + (f", not used: {skipped}" if skipped else ""))  # fmt: skip
    return M.report(model) | {"real_mm": round(float(real.max()), 2), "size_given": True, "size_mm": dims}
