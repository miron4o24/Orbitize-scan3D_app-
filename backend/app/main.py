import hashlib
import hmac
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import trimesh
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from . import accounts as A
from . import ai as AI
from . import ar as AR
from . import billing
from . import community
from . import edit as ED
from . import jobs
from . import mesh as M
from . import motion as MO
from . import pipeline as PI
from . import plans as PL
from . import printing as PR
from . import progress
from . import render as RE
from . import stress as ST
from . import texture as TX
from .accounts import Principal, admin_only, principal

API_TOKEN = os.environ.get("API_TOKEN", "")
if len(API_TOKEN) < 8:
    raise RuntimeError("Set API_TOKEN (at least 8 characters).")

MIN_PHOTOS, MAX_PHOTOS = 8, 120
MAX_PHOTO_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_MB", "600")) * 1024 * 1024
MAX_CHUNK_BYTES = 16 * 1024 * 1024
MAX_MOTION_SAMPLES = 60_000  # по 60 в секунда за над 15 минути
# Сървърът е публичен, затова ограничаваме какво всеки може да го накара да пази или изчислява.
MAX_JOBS = int(os.environ.get("MAX_JOBS", "30"))
MAX_QUEUED = int(os.environ.get("MAX_QUEUED", "5"))
ID_RE = re.compile(r"^[0-9a-f]{32}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.jpe?g$")  # блокира и излизане извън папката

worker = jobs.Worker()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    worker.start()
    A.db.engine()  # свързва се и създава таблиците сега, за да се вижда неправилна настройка на базата веднага в дневника
    yield


app = FastAPI(title="3D scan backend", lifespan=lifespan)
app.state.worker = worker
app.include_router(A.router)
app.include_router(community.router)
app.include_router(progress.router)
app.include_router(billing.router)


def existing(job_id: str, p: Principal) -> dict:
    """Състоянието на задачата, ако извикващият може да го вижда: администраторският ключ вижда всяка задача, профилът
    своите."""
    s = jobs.read(job_id) if ID_RE.match(job_id) else None
    if s is None or not A.can_see(p, s):
        raise HTTPException(404, "Сканирането вече не е на сървъра.")
    return s


@app.get("/health")
def health():
    return {"ok": True} | A.health_info()


@app.post("/jobs")
def create_job(p: Principal = Depends(principal)):
    if jobs.count() >= MAX_JOBS:
        raise HTTPException(503, "Услугата е натоварена в момента. Опитайте след малко.")
    if p.user_id:
        A.need_db()
    job_id = jobs.create(owner=p.user_id)
    if p.user_id:
        try:
            A.add_scan(job_id, p.user_id)
        except Exception:
            shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
            raise
    return jobs.public(job_id)


@app.get("/jobs", dependencies=[Depends(admin_only)])
def list_jobs():
    return jobs.summaries()


@app.get("/jobs/{job_id}")
def get_job(job_id: str, p: Principal = Depends(principal)):
    s = existing(job_id, p)
    # ahead: чака реда си, колко са преди него (включително обработваната задача).
    return jobs.public(job_id) | {"ahead": worker.ahead(job_id, s.get("params")) if s["state"] == "queued" else None}


@app.get("/jobs/{job_id}/log")
def get_log(job_id: str, lines: int = Query(60, ge=1, le=2000), p: Principal = Depends(principal)):
    existing(job_id, p)
    return Response(jobs.log_tail(job_id, lines), media_type="text/plain; charset=utf-8")


@app.get("/jobs/{job_id}/thumb.jpg")
def get_thumb(job_id: str, p: Principal = Depends(principal)):
    existing(job_id, p)
    path = jobs.thumbnail(job_id)
    if path is None:
        raise HTTPException(404, "Още няма снимка.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


@app.get("/jobs/{job_id}/frames.zip")
def get_frames(job_id: str, p: Principal = Depends(principal)):
    """Снимките или избраните от видеото кадри, плюс дневника на положението на телефона и времената на кадрите, ако
    ги има: за възпроизвеждане на сканиране другаде."""
    existing(job_id, p)
    d = jobs.job_dir(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:  # съхранено без компресия: JPEG файловете не се свиват повече
        for f in sorted((d / "photos").glob("*.jp*g")):
            z.write(f, f.name)
        for f in (d / MO.MOTION, d / MO.FRAME_TIMES):
            if f.exists():
                z.write(f, f.name)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="frames-{job_id[:8]}.zip"'},
    )


@app.put("/jobs/{job_id}/photos/{name}")
async def upload_photo(job_id: str, name: str, request: Request, p: Principal = Depends(principal)):
    if existing(job_id, p)["state"] not in ("uploading", "failed"):
        raise HTTPException(409, "Обработката вече е започнала.")
    if not NAME_RE.match(name):
        raise HTTPException(400, "Снимката е с неправилно име.")
    photos = jobs.job_dir(job_id) / "photos"
    if not (photos / name).exists() and len(list(photos.glob("*.jp*g"))) >= MAX_PHOTOS:
        raise HTTPException(400, f"До {MAX_PHOTOS} снимки на сканиране.")

    # Записва във временен файл и го преименува, за да не остава половин снимка при прекъснато качване.
    tmp = photos / f".{name}.part"
    size = 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_PHOTO_BYTES:
                f.close()
                tmp.unlink()
                raise HTTPException(413, "Снимката е твърде голяма.")
            f.write(chunk)
    with open(tmp, "rb") as f:
        is_jpeg = f.read(3) == b"\xff\xd8\xff"
    if not is_jpeg:
        tmp.unlink()  # след затваряне: Windows не може да изтрие отворен файл
        raise HTTPException(400, "Файлът не е снимка (JPEG).")
    os.replace(tmp, photos / name)
    return {"name": name, "bytes": size}


@app.put("/jobs/{job_id}/video")
async def upload_video_chunk(job_id: str, request: Request, offset: int = Query(ge=0), p: Principal = Depends(principal)):
    """Добавя една част при `offset`. Прекъсната по средата част отпада изцяло, така че `video_bytes` е безопасна
    точка за продължаване."""
    if existing(job_id, p)["state"] not in ("uploading", "failed"):
        raise HTTPException(409, "Обработката вече е започнала.")
    video = jobs.video_path(job_id)
    have = video.stat().st_size if video.exists() else 0
    if offset != have:
        raise HTTPException(409, f"Качването се разминава ({have}, не {offset}). Опитайте отново.")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_CHUNK_BYTES:
            raise HTTPException(413, "Частта от видеото е твърде голяма.")
    if offset + len(body) > MAX_VIDEO_BYTES:
        raise HTTPException(413, f"Видеото е твърде голямо (до {MAX_VIDEO_BYTES // 1024 // 1024} MB). Заснемете по-кратко.")
    if offset == 0 and body[4:8] != b"ftyp":
        raise HTTPException(400, "Файлът не е видео (MP4 или MOV).")
    with open(video, "ab") as f:
        f.write(body)
    return {"video_bytes": offset + len(body)}


class Motion(BaseModel):
    samples: list[tuple[float, float, float, float]] = Field(max_length=MAX_MOTION_SAMPLES)  # t, alpha, beta, gamma


@app.put("/jobs/{job_id}/motion")
def upload_motion(job_id: str, motion: Motion, p: Principal = Depends(principal)):
    """Положението на телефона по време на заснемане на видеото (app/motion.py): положенията на камерата за предмети,
    твърде едноцветни за съпоставяне на кадрите."""
    if existing(job_id, p)["state"] not in ("uploading", "failed", "done"):
        raise HTTPException(409, "Обработката вече е започнала.")
    path = jobs.job_dir(job_id) / MO.MOTION
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(motion.model_dump()))
    os.replace(tmp, path)
    return {"samples": len(motion.samples)}


class Tune(BaseModel):
    """Настройки на AI модела за сравнение върху тестовите задачи на сървъра; само с API ключа и само в тези граници
    (приложението никога не ги изпраща)."""

    side: int | None = Field(None, ge=960, le=1920)  # дългата страна на кадрите за очертанията, измерването и AI дълбочината
    voxel: float | None = Field(None, ge=0.002, le=0.006)  # от размера: мрежата на сливането
    agree: float | None = Field(None, ge=0.5, le=0.99)  # съгласието за формата от очертанията, вместо да се избира едно
    no_refine: bool = False  # без още едно уточняване на камерите преди каквото и да е измерване с тях
    no_table: bool = False  # без масата, намерена от точките на решаващия алгоритъм
    flap: float | None = Field(None, ge=0.005, le=0.05)  # от размера: fuse.FLAP, колко дебел лист върху масата може да отпадне
    wide_dense: bool = False  # гъсто съпоставяне в областта на сканиране (почти целия кадър), а не само около предмета
    no_geo: bool = False  # гъсто съпоставяне без втория, проверяващ проход
    passes: int | None = Field(None, ge=1, le=3)  # проходите за гласуване при сливането
    features: int | None = Field(None, ge=2000, le=8192)  # най-много особености на кадър за решаващия алгоритъм


class StartParams(BaseModel):
    flat_base: bool = True
    video_bytes: int | None = None  # размерът на файла в приложението, за да не се обработва никога наполовина качено видео
    quality: Literal["detailed", "quick"] = "detailed"  # quick: по-малко и по-малки кадри (pipeline.run)
    # scan: сканирането за около 35 минути (за всички). Pro: single, AI моделът (claude, photo: AI моделите, изискващи
    # платени ключове; app/ai.py). part: точна част 1:1 по снимки и мерки (app/part.py).
    method: Literal["scan", "express", "single", "claude", "photo", "part"] = "scan"
    tune: Tune | None = None
    extend: str | None = Field(None, pattern=r"^[0-9a-f]{32}$")  # това видео добавя липсваща страна към този модел
    measures: list[Annotated[float, Field(ge=0, le=2000)]] | None = None  # мерки с линийка/шублер (мм) за точна част


@app.post("/jobs/{job_id}/start")
def start_job(job_id: str, params: StartParams, p: Principal = Depends(principal)):
    # Завършено сканиране може да се обработи отново (например след обновяване на сървъра): кадрите, избрани от
    # видеото, се пазят.
    s = existing(job_id, p)
    if s["state"] not in ("uploading", "failed", "done"):
        raise HTTPException(409, "Обработката вече е започнала.")
    if (s.get("params") or {}).get("method") == "edit":
        raise HTTPException(409, "Промяна не се пуска отново: опишете я пак от оригиналния модел.")
    status = jobs.public(job_id)
    if params.video_bytes is not None and params.video_bytes != status["video_bytes"]:
        raise HTTPException(400, "Видеото не се качи докрай. Опитайте отново.")
    n = len(status["received"])
    params.quality = "detailed"
    if not p.admin:
        params.tune = None
    if params.method == "express":
        params.method = "scan"
    pro = params.method != "scan"
    trial = False
    if pro:
        if params.method not in ("single", "part") and not A.is_pro(p):
            raise HTTPException(403, "Това е само за Pro профили.")
        if params.method == "single":
            if not A.is_pro(p):
                PL.check(p, "ai", job_id)  # единственият AI модел на безплатен профил; 402, когато е използван
                trial = True
        if not AI.available().get(params.method, True):
            raise HTTPException(503, "Тази AI услуга още не е включена.")
    min_photos = 1 if params.method in (*AI.METHODS, "part") else MIN_PHOTOS
    if not status["video_bytes"] and n < min_photos:
        raise HTTPException(400, f"Нужно е видео или поне {MIN_PHOTOS} снимки.")
    if (worker.ai_q if pro else worker.q).qsize() >= MAX_QUEUED:
        raise HTTPException(503, "Твърде много сканирания чакат в момента. Опитайте по-късно.")
    if params.extend:
        if not status["video_bytes"] or params.extend == job_id:
            raise HTTPException(400, "За допълнение е нужно ново видео.")
        add_frames_of(existing(params.extend, p), params.extend, job_id)
    if trial:
        PL.record(p, "ai", job_id)
    worker.submit(job_id, params.model_dump(exclude={"extend"}))
    return jobs.public(job_id)


MORE_FRAMES_MOST = 96  # от кадрите на модела при добавяне на страна (модел с вече добавена страна има повече)


def add_frames_of(source: dict, source_id: str, job_id: str):
    """Копира кадрите на завършен модел в нова задача, където обработката ги съединява с тези от видеото ѝ."""
    if source["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    frames = sorted((jobs.job_dir(source_id) / "photos").glob("*.jp*g"))
    if len(frames) < MIN_PHOTOS:
        raise HTTPException(409, "Кадрите на този модел вече не се пазят (пазят се 7 дни). Направете ново сканиране.")
    if len(frames) > MORE_FRAMES_MOST:
        frames = [frames[i] for i in range(0, len(frames), len(frames) // MORE_FRAMES_MOST + 1)]
    more = jobs.job_dir(job_id) / PI.MORE_FRAMES
    shutil.rmtree(more, ignore_errors=True)
    more.mkdir()
    for i, f in enumerate(frames):
        shutil.copy(f, more / f"old_{i + 1:04d}.jpg")


class EditParams(BaseModel):
    prompt: str = Field(min_length=2, max_length=500)  # какво да се промени, на английски
    longest_mm: float = Field(100.0, gt=0, le=2000)  # реалният размер на модела в приложението: промяната работи с реални мм
    title: str = Field("", max_length=100)  # името на новия модел
    # Мястото, което потребителят е докоснал върху модела в приложението ("smooth here", "make this red"): дялове
    # (0-1) от ограничаващата кутия на модела в осите на GLB (app/edit.py spot_point).
    spot: list[Annotated[float, Field(ge=-0.05, le=1.05)]] | None = Field(None, min_length=3, max_length=3)


@app.post("/jobs/{job_id}/edit")
def edit_job(job_id: str, params: EditParams, p: Principal = Depends(principal)):
    """Pro: нов модел, създаден от този чрез текстова промяна (app/edit.py); този остава непроменен. Връща новата
    задача, която приложението следи като сканиране."""
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    if not A.is_pro(p):
        raise HTTPException(403, "Промените с текст са само за Pro профили.")
    if jobs.count() >= MAX_JOBS:
        raise HTTPException(503, "Услугата е натоварена в момента. Опитайте след малко.")
    if worker.ai_q.qsize() >= MAX_QUEUED:
        raise HTTPException(503, "Твърде много сканирания чакат в момента. Опитайте по-късно.")
    new = jobs.create(owner=p.user_id)
    try:
        if p.user_id:
            A.add_scan(new, p.user_id, params.title.strip())
        ED.copy_source(jobs.job_dir(job_id), jobs.job_dir(new))
    except Exception:
        shutil.rmtree(jobs.job_dir(new), ignore_errors=True)
        A.forget_scan(new)
        raise
    worker.submit(new, {"method": "edit", "source": job_id, "prompt": params.prompt.strip(),
                        "longest_mm": params.longest_mm, "spot": params.spot})  # fmt: skip
    return jobs.public(new)


class NameplateParams(BaseModel):
    text: str = Field(min_length=1, max_length=40)
    shape: Literal["round", "square"] = "square"
    longest_mm: float = Field(100.0, gt=0, le=2000)
    title: str = Field("", max_length=100)


@app.post("/jobs/{job_id}/nameplate")
def nameplate_job(job_id: str, params: NameplateParams, p: Principal = Depends(principal)):
    """Pro: нов модел, създаден от този, стоящ върху плочка с име (app/edit.py); този остава непроменен."""
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    if not A.is_pro(p):
        raise HTTPException(403, "Табелката с име е за Pro профили.")
    if not " ".join(params.text.split()):
        raise HTTPException(400, "Напишете име.")
    if jobs.count() >= MAX_JOBS:
        raise HTTPException(503, "Услугата е натоварена в момента. Опитайте след малко.")
    if worker.ai_q.qsize() >= MAX_QUEUED:
        raise HTTPException(503, "Твърде много сканирания чакат в момента. Опитайте по-късно.")
    new = jobs.create(owner=p.user_id)
    try:
        if p.user_id:
            A.add_scan(new, p.user_id, params.title.strip())
        ED.copy_source(jobs.job_dir(job_id), jobs.job_dir(new))
    except Exception:
        shutil.rmtree(jobs.job_dir(new), ignore_errors=True)
        A.forget_scan(new)
        raise
    worker.submit(new, {"method": "edit", "source": job_id, "longest_mm": params.longest_mm,
                        "nameplate": {"text": params.text, "shape": params.shape}})  # fmt: skip
    return jobs.public(new)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, p: Principal = Depends(principal)):
    existing(job_id, p)
    A.delete_job_files(job_id, worker)
    A.forget_scan(job_id)
    return {"id": job_id, "state": "cancelled", "received": [], "error": None, "report": None}


@app.get("/jobs/{job_id}/result.{fmt}")
def result(job_id: str, fmt: str, longest_mm: float | None = Query(None, gt=0, le=2000), texture: bool = False,
           smooth: int = Query(0, ge=0, le=M.MAX_SMOOTH), p: Principal = Depends(principal)):  # fmt: skip
    """texture: GLB с текстура от снимки, ако задачата има такава. Само при заявка: компилации отпреди това се
    провалят при изображения в GLB (React Native няма декодер на изображения за three.js). smooth: допълнителни
    заглаждания за печат (app/mesh.py smooth); не и за GLB с текстура, чиято повърхност е собствена."""
    if fmt not in M.MEDIA_TYPES:
        raise HTTPException(404, "Непознат формат.")
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    job = jobs.job_dir(job_id)
    mesh = trimesh.load(job / "model.ply", force="mesh")
    textured = TX.load(job) if fmt == "glb" and texture else None
    if textured is None:
        mesh = M.smooth(mesh, smooth)
    return Response(
        M.export(mesh, fmt, longest_mm, textured),
        media_type=M.MEDIA_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="scan-{job_id[:8]}.{fmt}"'},
    )


@app.get("/jobs/{job_id}/geometry")
def get_geometry(job_id: str, p: Principal = Depends(principal)):
    """Обем и повърхност при съхранения размер (най-дълга страна 100 мм) за оценката на печата в приложението, за
    сканирания отпреди отчетът да ги съдържа; оттам нататък се пазят в отчета."""
    s = existing(job_id, p)
    if s["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    report = s.get("report") or {}
    if "volume_mm3" not in report:
        mesh = trimesh.load(jobs.job_dir(job_id) / "model.ply", force="mesh")
        report = report | M.geometry(mesh)
        jobs.write(job_id, report=report)
    return {k: report[k] for k in ("dimensions_mm", "volume_mm3", "area_mm2") if k in report}


# --- Помощ за печат (app/printing.py)
# --------------------------------------------------------------------------------


class Bed(BaseModel):
    longest_mm: float = Field(100.0, gt=0, le=2000)
    bed_x: float = Field(220.0, ge=50, le=1000)
    bed_y: float = Field(220.0, ge=50, le=1000)
    bed_z: float = Field(250.0, ge=50, le=1000)

    def size(self) -> tuple[float, float, float]:
        return self.bed_x, self.bed_y, self.bed_z


def done_model(job_id: str, p: Principal) -> trimesh.Trimesh:
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    return trimesh.load(jobs.job_dir(job_id) / "model.ply", force="mesh")


@app.post("/jobs/{job_id}/print/plan")
def print_plan(job_id: str, bed: Bed, p: Principal = Depends(principal)):
    try:
        return PR.plan(done_model(job_id, p), bed.longest_mm, bed.size())
    except M.MeshError as e:
        raise HTTPException(422, str(e))


@app.get("/jobs/{job_id}/print.zip")
def print_zip(job_id: str, longest_mm: float = Query(100.0, gt=0, le=2000), bed_x: float = Query(220.0, ge=50, le=1000),
              bed_y: float = Query(220.0, ge=50, le=1000), bed_z: float = Query(250.0, ge=50, le=1000),
              p: Principal = Depends(principal)):  # fmt: skip
    """Моделът в най-добрата посока, разрязан на части, които се побират на плочата, като файлове STL."""
    try:
        pieces = PR.parts(done_model(job_id, p), longest_mm, (bed_x, bed_y, bed_z))
    except M.MeshError as e:
        raise HTTPException(422, str(e))
    return Response(PR.zip_of(pieces, f"scan-{job_id[:8]}"), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="print-{job_id[:8]}.zip"'})  # fmt: skip


# --- Видеото със завъртане (app/render.py spin)
# ----------------------------------------------------------------------

SPIN = "spin.mp4"
_spin_lock = threading.Lock()  # по едно наведнъж: заема процесорно ядро за минута-две
_spinning: set[str] = set()
_spin_failed: set[str] = set()


def make_spin(job_id: str):
    d = jobs.job_dir(job_id)
    try:
        with _spin_lock:
            RE.spin(d, d / SPIN, PI.FFMPEG)
    except Exception as e:
        PI.note(d, f"spinning video: {e!r}")
        _spin_failed.add(job_id)
    finally:
        _spinning.discard(job_id)


@app.post("/jobs/{job_id}/spin")
def start_spin(job_id: str, p: Principal = Depends(principal)):
    """Създава видеото, ако го няма; приложението пита отново до `ready`, после го изтегля."""
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    d = jobs.job_dir(job_id)
    out = d / SPIN
    if out.exists() and out.stat().st_mtime >= (d / "model.ply").stat().st_mtime:
        return {"ready": True}
    if job_id in _spin_failed:
        _spin_failed.discard(job_id)
        raise HTTPException(500, "Видеото не се получи. Опитайте отново.")
    if job_id not in _spinning:
        _spinning.add(job_id)
        threading.Thread(target=make_spin, args=(job_id,), daemon=True).start()
    return {"ready": False}


@app.get("/jobs/{job_id}/spin.mp4")
def get_spin(job_id: str, p: Principal = Depends(principal)):
    existing(job_id, p)
    out = jobs.job_dir(job_id) / SPIN
    if not out.exists():
        raise HTTPException(409, "Видеото още не е готово.")
    return Response(out.read_bytes(), media_type="video/mp4",
                    headers={"Content-Disposition": f'attachment; filename="orbitize-{job_id[:8]}.mp4"'})  # fmt: skip


# --- Тест за здравина (app/stress.py)
# --------------------------------------------------------------------------------


@app.get("/jobs/{job_id}/stress/{case}")
def get_stress(job_id: str, case: str, p: Principal = Depends(principal)):
    """Безплатни профили: ограничен брой модели (app/plans.py), брои се след първия успешен тест на модел."""
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    PL.check(p, "stress", job_id)
    out = ST.summary_response(job_id, case, p.user_id)
    PL.record(p, "stress", job_id)
    return out


@app.get("/jobs/{job_id}/stress/{case}/model.glb")
def get_stress_model(job_id: str, case: str, p: Principal = Depends(principal)):
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    PL.check(p, "stress", job_id)
    return ST.model_response(job_id, case)


# --- Добавена реалност (app/ar.py)
# -----------------------------------------------------------------------------------

AR_KEY = hmac.new(API_TOKEN.encode(), b"ar links", hashlib.sha256).digest()
AR_LINK_SECONDS = 6 * 3600


def ar_signature(job_id: str, tenths: int, expires: int) -> str:
    return hmac.new(AR_KEY, f"{job_id}/{tenths}/{expires}".encode(), hashlib.sha256).hexdigest()[:32]


class ArParams(BaseModel):
    longest_mm: float = Field(100.0, gt=0, le=2000)


@app.post("/jobs/{job_id}/ar")
def ar_link(job_id: str, params: ArParams, request: Request, p: Principal = Depends(principal)):
    """Линкове към модела в реален размер за AR: `page` (app/ar.py, отваря се в Chrome) и `url`, самият GLB. Браузърът
    ги зарежда без ключа на приложението, затова са подписани и изтичат. Файлът се създава тук, така че чака
    приложението, а не браузърът."""
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    PL.check(p, "ar", job_id)  # безплатни профили: ограничен брой сканирания (app/plans.py)
    tenths = round(params.longest_mm * 10)
    AR.cached(jobs.job_dir(job_id), tenths)
    PL.record(p, "ar", job_id)
    expires = int(time.time()) + AR_LINK_SECONDS
    base = f"{A.base_url(request)}/ar/{job_id}/{tenths}/{expires}/{ar_signature(job_id, tenths, expires)}"
    return {"page": f"{base}/view", "url": f"{base}/model.glb", "expires": expires}


def ar_checked(job_id: str, tenths: int, expires: int, sig: str):
    ok = ID_RE.match(job_id) and hmac.compare_digest(sig, ar_signature(job_id, tenths, expires))
    if not ok or expires < time.time() or (jobs.read(job_id) or {}).get("state") != "done":
        raise HTTPException(404, "Линкът е изтекъл. Отворете модела отново в приложението.")


@app.get("/ar/{job_id}/{tenths}/{expires}/{sig}/view")
def ar_page(job_id: str, tenths: int, expires: int, sig: str, title: str = Query("", max_length=100), real: bool = True):
    ar_checked(job_id, tenths, expires, sig)
    return Response(AR.page(title, tenths / 10, real), media_type="text/html; charset=utf-8",
                    headers={"Cache-Control": "no-store"})  # fmt: skip


@app.api_route("/ar/{job_id}/{tenths}/{expires}/{sig}/model.glb", methods=["GET", "HEAD"])
def ar_model(job_id: str, tenths: int, expires: int, sig: str, request: Request):
    ar_checked(job_id, tenths, expires, sig)
    path = AR.cached(jobs.job_dir(job_id), tenths)
    headers = {"Cache-Control": "private, max-age=3600", "Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", ""):
        path, headers["Content-Encoding"] = AR.zipped(path), "gzip"
    return Response(path.read_bytes(), media_type=M.MEDIA_TYPES["glb"], headers=headers)
