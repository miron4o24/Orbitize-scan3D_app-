"""Инженерният режим върху папката на задачата (крайните точки са в main.py):

job/photos/top.jpg, side.jpg и по желание end.jpg  - качените снимки (както при сканиране)
job/cad/views.json  - прочетените очертания (мм) и как се връщат към снимката; пресмята се наново само при нови
                      снимки или мерки
job/cad/draft.json  - текущото описание (от него са картинките към въпросите и наложените очертания)
job/cad/out/        - последният построен детайл: part.step, part.py, part.stl, part.glb, drawing.png

Всичко става веднага (секунди), не в опашката на сканиранията: четенето на снимките е изрязване с IS-Net (по
няколко секунди на снимка), а строенето е CadQuery."""

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from . import assemble as AS
from . import design as D
from . import render as R

VIEWS = ("top", "side", "end")
REQUIRED = ("top", "side")
FILES = {"part.step": "model/step", "part.py": "text/x-python; charset=utf-8", "part.stl": "model/stl",
         "part.glb": "model/gltf-binary", "drawing.png": "image/png"}  # fmt: skip
_session = None
_session_lock = threading.Lock()
_busy = threading.BoundedSemaphore(2)  # най-много две четения на снимки наведнъж (процесорът е общ със сканиранията)
_job_locks: dict[str, threading.Lock] = {}


def lock(job: Path) -> threading.Lock:
    with _session_lock:
        return _job_locks.setdefault(job.name, threading.Lock())


def session():
    global _session
    with _session_lock:
        if _session is None:
            from .. import segment as S

            _session = S.session()
        return _session


def photo_path(job: Path, view: str) -> Path | None:
    for ext in ("jpg", "jpeg"):
        p = job / "photos" / f"{view}.{ext}"
        if p.exists():
            return p
    return None


def response(job: Path, d: dict, warnings: list[str] | None = None) -> dict:
    """Това, което приложението показва: описанието, разпознатото, въпросите и кои изгледи имат снимка."""
    out = job / "cad"
    out.mkdir(exist_ok=True)
    tmp = out / "draft.json.tmp"
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out / "draft.json")
    stamp = str(int(time.time() * 1000))
    views = [v for v in VIEWS if photo_path(job, v)]
    return {"design": d, "items": D.describe(d), "questions": D.questions(d), "notes": d.get("notes", []),
            "warnings": warnings or [], "views": views, "stamp": stamp}  # fmt: skip


def read(job: Path, L: float, W: float, H: float, flips: dict | None = None) -> dict:
    """Чете снимките (или прочетеното преди със същите мерки) и предлага описанието."""
    for v in REQUIRED:
        if not photo_path(job, v):
            raise D.DesignError("Липсва снимка " + {"top": "отгоре", "side": "отстрани"}[v] + ".")
    out = job / "cad"
    out.mkdir(exist_ok=True)
    cache = out / "views.json"
    sizes = {"top": (L, W), "side": (L, H), "end": (W, H)}
    with lock(job):
        stored = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        views, warnings = {}, []
        todo = []
        for v in VIEWS:
            path = photo_path(job, v)
            if not path:
                continue
            key = [path.stat().st_mtime, *sizes[v]]
            if v in stored and stored[v].get("key") == key:
                views[v] = stored[v]
            else:
                todo.append((v, path, key))
        if todo:
            from . import vision as VI

            with _busy:
                sess = session()
                for v, path, key in todo:
                    try:
                        got = VI.read(sess, path, *sizes[v])
                    except ValueError as e:
                        raise D.DesignError(f"На снимката {D.VIEW_NAME[v]} не се вижда детайл: снимайте отново, "
                                            "детайлът в средата, на контрастна повърхност.") from e  # fmt: skip
                    views[v] = {"key": key, "outer": got.outer.tolist(), "holes": [h.tolist() for h in got.holes],
                                "U": got.U, "V": got.V, "transform": got.transform(), "warnings": got.warnings}  # fmt: skip
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(views), encoding="utf-8")
            os.replace(tmp, cache)
        for v in views.values():
            warnings += v.get("warnings", [])
        geo = {k: {"outer": np.asarray(v["outer"]), "holes": [np.asarray(h) for h in v["holes"]], "U": v["U"],
                   "V": v["V"]} for k, v in views.items()}  # fmt: skip
        d = AS.assemble(geo, L, W, H, flips)
        d["flips"] = flips or {}
        return response(job, d, warnings)


def edit(job: Path, raw: dict, op: dict) -> dict:
    """Промяна от приложението: values, remove, add, through, flip (наново от прочетеното, мерките се губят)."""
    from .build import clean

    d = clean(raw)
    d["flips"] = sanitize_flips(raw.get("flips"))
    if "values" in op:
        vals = {}
        for k, v in (op["values"] or {}).items():
            vals[str(k)] = None if v is None else float(v)
        d = D.set_values(d, vals)
    elif "remove" in op:
        d = D.remove(d, str(op["remove"]))
    elif "add" in op:
        d = D.add_feature(d, op["add"] if isinstance(op["add"], dict) else {})
    elif "through" in op:
        fid, through = op["through"]
        d = set_through(d, str(fid), bool(through))
    elif "flip" in op:
        view, axis = op["flip"]
        if view not in ("side", "end") or axis not in (0, 1):
            raise D.DesignError("Непознато обръщане.")
        flips = d["flips"]
        cur = flips.get(view, [False, False])
        cur[axis] = not cur[axis]
        flips[view] = cur
        L, W, H = (D.value(d, "L"), D.value(d, "D"), D.value(d, "D")) if d["kind"] == "turned" else \
            (D.value(d, "L"), D.value(d, "W"), D.value(d, "H"))  # fmt: skip
        return read(job, L, W, H, flips)
    else:
        raise D.DesignError("Непозната промяна.")
    d["flips"] = sanitize_flips(raw.get("flips"))
    return response(job, d)


def sanitize_flips(f) -> dict:
    out = {}
    if isinstance(f, dict):
        for k in ("side", "end"):
            v = f.get(k)
            if isinstance(v, list) and len(v) == 2:
                out[k] = [bool(v[0]), bool(v[1])]
    return out


def set_through(d: dict, fid: str, through: bool) -> dict:
    import copy

    d = copy.deepcopy(d)
    f = next((f for f in d["features"] if f["id"] == fid), None)
    if f is None or f["t"] not in ("hole", "rect", "slot"):
        raise D.DesignError("Непознат елемент.")
    if through:
        f.pop("depth", None)
        D.unused_cleanup(d)
    elif not f.get("depth"):
        face = f.get("face") or "top"
        f["depth"] = D.add_dim(d, "t", None, "Дълбочина", f"Измерете дълбочината от {D.FACE_NAME[face]}",
                               "С дълбокомера на шублера.", show={"feature": fid})  # fmt: skip
    return d


def draft(job: Path) -> dict:
    p = job / "cad" / "draft.json"
    if not p.exists():
        raise D.DesignError("Още няма прочетени снимки.")
    return json.loads(p.read_text(encoding="utf-8"))


def question_png(job: Path, dim: str) -> bytes:
    d = draft(job)
    if dim not in d["dims"]:
        raise D.DesignError("Непознат размер.")
    return png(R.question(d, dim))


def photo_jpg(job: Path, view: str) -> bytes:
    """Снимката на изгледа с очертанието от текущото описание."""
    from PIL import Image

    from .vision import load

    d = draft(job)
    cache = job / "cad" / "views.json"
    path = photo_path(job, view)
    if not path or not cache.exists():
        raise D.DesignError("Няма такава снимка.")
    views = json.loads(cache.read_text(encoding="utf-8"))
    if view not in views:
        raise D.DesignError("Снимката още не е прочетена.")
    im = load(path)
    try:
        im = R.overlay(d, view, im, views[view]["transform"], d.get("flips") or {},
                       outer=np.asarray(views[view]["outer"]))
    except D.DesignError:
        im.thumbnail((1100, 1100), Image.LANCZOS)
    import io

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return buf.getvalue()


def png(im) -> bytes:
    import io

    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def build(job: Path, raw: dict) -> dict:
    """Строи детайла, записва файловете и модела за приложението (model.ply с най-дълга страна 100, като всеки модел;
    истинският размер е real_mm). Връща отчета на задачата."""
    import trimesh

    from .. import mesh as M
    from .build import build as make
    from .build import clean

    d = clean(raw)
    d["flips"] = sanitize_flips(raw.get("flips"))
    missing = [q for q in D.questions(d) if d["dims"][q["id"]]["v"] is None]
    if missing:
        raise D.DesignError(f"Липсва размерът „{missing[0]['name']}“.")
    with lock(job):
        work = job / "cad" / "build.tmp"
        shutil.rmtree(work, ignore_errors=True)
        report = make(d, work)
        R.drawing(d).save(work / "drawing.png", optimize=True)
        out = job / "cad" / "out"
        shutil.rmtree(out, ignore_errors=True)
        os.replace(work, out)
        response(job, d)
        model = trimesh.load(out / "part.stl", force="mesh")
        real = float(max(model.extents))
        model.apply_translation(-model.bounds.mean(0) * [1, 1, 0] - [0, 0, model.bounds[0][2]])
        model.apply_scale(100.0 / real)
        tmp = job / "model.tmp.ply"
        model.export(tmp)
        os.replace(tmp, job / "model.ply")
    return M.report(model) | {"ai": "cad", "real_mm": round(real, 3), "object": "", "suitable": True,
                              "cad": report}  # fmt: skip


def preview(job: Path) -> Path:
    """GLB на текущото описание с оценките (за проверка на посоките преди мерките). Елементи без размер (добавени
    ръчно, още неизмерени) се пропускат."""
    import copy
    import hashlib

    from .build import build as make
    from .build import clean

    d = draft(job)
    stamp = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    out = job / "cad" / "preview"
    glb = out / "part.glb"
    if glb.exists() and (out / "stamp").exists() and (out / "stamp").read_text() == stamp:
        return glb
    p = copy.deepcopy(d)
    missing = {i for i, dim in p["dims"].items() if dim["v"] is None}
    if missing:
        p["features"] = [f for f in p["features"] if not any(f.get(k) in missing for k in
                                                             ("d", "w", "h", "len", "depth", "r", "c"))]  # fmt: skip
        D.unused_cleanup(p)
    with lock(job):
        work = job / "cad" / "preview.tmp"
        shutil.rmtree(work, ignore_errors=True)
        make(clean(p), work)
        (work / "stamp").write_text(stamp)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(work, out)
    return glb
