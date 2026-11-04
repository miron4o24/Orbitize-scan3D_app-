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
MAX_MOTION_SAMPLES = 60_000  # 60 a second for over 15 minutes
# The server is public and the app's token can be extracted, so cap what anyone can make it store or compute.
MAX_JOBS = int(os.environ.get("MAX_JOBS", "30"))
MAX_QUEUED = int(os.environ.get("MAX_QUEUED", "5"))
ID_RE = re.compile(r"^[0-9a-f]{32}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.jpe?g$")  # also blocks path traversal

worker = jobs.Worker()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    worker.start()
    A.db.engine()  # connect and make the tables now, so a wrong database setting shows in the log at once
    yield


app = FastAPI(title="3D scan backend", lifespan=lifespan)
app.state.worker = worker
app.include_router(A.router)
app.include_router(community.router)
app.include_router(progress.router)
app.include_router(billing.router)


def existing(job_id: str, p: Principal) -> dict:
    """The job's status, if this caller may see it: the admin token every job, an account its own."""
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
    # ahead: waiting for its turn, how many go first (the job being processed among them).
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
    """The photos, or the frames picked from the video, plus the phone's orientation log and the frames' times when
    there are: to reproduce a scan elsewhere."""
    existing(job_id, p)
    d = jobs.job_dir(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:  # stored, not deflated: JPEGs do not shrink further
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

    # Write to a temp file and rename, so an interrupted upload never leaves a half photo behind.
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
        tmp.unlink()  # after closing: Windows cannot delete an open file
        raise HTTPException(400, "Файлът не е снимка (JPEG).")
    os.replace(tmp, photos / name)
    return {"name": name, "bytes": size}


@app.put("/jobs/{job_id}/video")
async def upload_video_chunk(job_id: str, request: Request, offset: int = Query(ge=0), p: Principal = Depends(principal)):
    """Appends one chunk at `offset`. A chunk cut off mid-way is dropped whole, so `video_bytes` is a safe resume point."""
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
    """The phone's orientation while filming the video (app/motion.py): the camera poses for objects too plain for
    the frames to be matched."""
    if existing(job_id, p)["state"] not in ("uploading", "failed", "done"):
        raise HTTPException(409, "Обработката вече е започнала.")
    path = jobs.job_dir(job_id) / MO.MOTION
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(motion.model_dump()))
    os.replace(tmp, path)
    return {"samples": len(motion.samples)}


class Tune(BaseModel):
    """Settings of the AI model, to compare them on the server's own test jobs; only with the API token, and only
    within these limits (the app never sends them)."""

    side: int | None = Field(None, ge=960, le=1920)  # frames' long side for the outlines, measuring and AI depth
    voxel: float | None = Field(None, ge=0.002, le=0.006)  # of the size: the fusion's grid
    agree: float | None = Field(None, ge=0.5, le=0.99)  # the outline shape's agreement, instead of choosing one
    no_refine: bool = False  # without refining the cameras once more before anything is measured from them
    no_table: bool = False  # without the table the solver's points found
    flap: float | None = Field(None, ge=0.005, le=0.05)  # of the size: fuse.FLAP, how thick a sheet on the table may go
    wide_dense: bool = False  # dense matching on a scan's area (nearly the whole frame), not only around the object
    no_geo: bool = False  # dense matching without its second, checking pass
    passes: int | None = Field(None, ge=1, le=3)  # the fusion's voting passes
    features: int | None = Field(None, ge=2000, le=8192)  # the camera solver's most features per frame


class StartParams(BaseModel):
    flat_base: bool = True
    video_bytes: int | None = None  # the app's file size, so a half-uploaded video is never processed
    quality: Literal["detailed", "quick"] = "detailed"  # quick: fewer, smaller frames (pipeline.run)
    # scan: the ~35 minute scan (everyone). Pro: single, the AI model (claude, photo: the AI models that need paid
    # keys; app/ai.py). An older app's quick or express is a scan (start_job).
    method: Literal["scan", "express", "single", "claude", "photo"] = "scan"
    tune: Tune | None = None
    extend: str | None = Field(None, pattern=r"^[0-9a-f]{32}$")  # this video adds a missing side to that model


@app.post("/jobs/{job_id}/start")
def start_job(job_id: str, params: StartParams, p: Principal = Depends(principal)):
    # A finished scan can be processed again (e.g. after a server update): it keeps the frames picked from its video.
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
        if not A.is_pro(p):
            if params.method != "single":
                raise HTTPException(403, "Това е само за Pro профили.")
            PL.check(p, "ai", job_id)  # a free account's one AI model; 402 when it has been used
            trial = True
        if not AI.available()[params.method]:
            raise HTTPException(503, "Тази AI услуга още не е включена.")
    if not status["video_bytes"] and n < (1 if params.method in AI.METHODS else MIN_PHOTOS):
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


MORE_FRAMES_MOST = 96  # of a model's frames, when adding a side (a model already added to has more)


def add_frames_of(source: dict, source_id: str, job_id: str):
    """Copies a finished model's frames into a new job, where the pipeline joins them to its video's."""
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
    prompt: str = Field(min_length=2, max_length=500)  # what to change, in English
    longest_mm: float = Field(100.0, gt=0, le=2000)  # the model's real size in the app: the change works in real mm
    title: str = Field("", max_length=100)  # the new model's name
    # The spot the user tapped on the model in the app ("smooth here", "make this red"): shares (0-1) of the model's
    # bounding box in the GLB's axes (app/edit.py spot_point).
    spot: list[Annotated[float, Field(ge=-0.05, le=1.05)]] | None = Field(None, min_length=3, max_length=3)


@app.post("/jobs/{job_id}/edit")
def edit_job(job_id: str, params: EditParams, p: Principal = Depends(principal)):
    """Pro: a new model made from this one by a text change (app/edit.py); this one stays as it is. Answers the new
    job, which the app follows like a scan."""
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
    """Pro: a new model made from this one, standing on a plate with a name (app/edit.py); this one stays as it is."""
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
    """texture: a GLB with the photo texture, where the job has one. Only on request: app builds before it fail on
    images inside a GLB (React Native has no image decoder for three.js). smooth: extra smoothing passes for a print
    (app/mesh.py smooth); not on the textured GLB, whose surface is its own."""
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
    """Volume and surface at the stored size (longest side 100 mm) for the app's print estimate, for scans made before
    the report had them; kept in the report from then on."""
    s = existing(job_id, p)
    if s["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    report = s.get("report") or {}
    if "volume_mm3" not in report:
        mesh = trimesh.load(jobs.job_dir(job_id) / "model.ply", force="mesh")
        report = report | M.geometry(mesh)
        jobs.write(job_id, report=report)
    return {k: report[k] for k in ("dimensions_mm", "volume_mm3", "area_mm2") if k in report}


# --- The print helper (app/printing.py) -------------------------------------------------------------------------


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
    """The model the best way up, cut into parts that fit the bed, as STL files."""
    try:
        pieces = PR.parts(done_model(job_id, p), longest_mm, (bed_x, bed_y, bed_z))
    except M.MeshError as e:
        raise HTTPException(422, str(e))
    return Response(PR.zip_of(pieces, f"scan-{job_id[:8]}"), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="print-{job_id[:8]}.zip"'})  # fmt: skip


# --- The spinning video (app/render.py spin) --------------------------------------------------------------------

SPIN = "spin.mp4"
_spin_lock = threading.Lock()  # one at a time: it takes a processor core for a minute or two
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
    """Makes the video unless it is there; the app asks again until `ready`, then downloads it."""
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
                    headers={"Content-Disposition": f'attachment; filename="scanly-{job_id[:8]}.mp4"'})  # fmt: skip


# --- Stress test (app/stress.py) --------------------------------------------------------------------------------


@app.get("/jobs/{job_id}/stress/{case}")
def get_stress(job_id: str, case: str, p: Principal = Depends(principal)):
    """Free accounts: a limited number of models (app/plans.py), counted once the first test of a model worked."""
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


# --- Augmented reality (app/ar.py) ------------------------------------------------------------------------------

AR_KEY = hmac.new(API_TOKEN.encode(), b"ar links", hashlib.sha256).digest()
AR_LINK_SECONDS = 6 * 3600


def ar_signature(job_id: str, tenths: int, expires: int) -> str:
    return hmac.new(AR_KEY, f"{job_id}/{tenths}/{expires}".encode(), hashlib.sha256).hexdigest()[:32]


class ArParams(BaseModel):
    longest_mm: float = Field(100.0, gt=0, le=2000)


@app.post("/jobs/{job_id}/ar")
def ar_link(job_id: str, params: ArParams, request: Request, p: Principal = Depends(principal)):
    """Links to the model at its real size for AR: `page` (app/ar.py, opened in Chrome) and `url`, the GLB itself.
    The browser loads them without the app's token, so they are signed and expire. The file is made here, so the app
    waits for it rather than the browser."""
    if existing(job_id, p)["state"] != "done":
        raise HTTPException(409, "Моделът още не е готов.")
    PL.check(p, "ar", job_id)  # free accounts: a limited number of scans (app/plans.py)
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
