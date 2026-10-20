import hmac
import os
import re
import shutil
from contextlib import asynccontextmanager

import trimesh
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel

from . import jobs
from . import mesh as M

API_TOKEN = os.environ.get("API_TOKEN", "")
if len(API_TOKEN) < 8:
    raise RuntimeError("Set API_TOKEN (at least 8 characters).")

MIN_PHOTOS, MAX_PHOTOS = 8, 120
MAX_PHOTO_BYTES = 15 * 1024 * 1024
# The server is public and the app's token can be extracted, so cap what anyone can make it store or compute.
MAX_JOBS = int(os.environ.get("MAX_JOBS", "30"))
MAX_QUEUED = int(os.environ.get("MAX_QUEUED", "5"))
ID_RE = re.compile(r"^[0-9a-f]{32}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.jpe?g$")  # also blocks path traversal

worker = jobs.Worker()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    worker.start()
    yield


app = FastAPI(title="3D scan backend", lifespan=lifespan)


def auth(authorization: str = Header("")):
    if not hmac.compare_digest(authorization.encode(), f"Bearer {API_TOKEN}".encode()):
        raise HTTPException(401, "Bad or missing API token.")


def existing(job_id: str) -> dict:
    s = jobs.read(job_id) if ID_RE.match(job_id) else None
    if s is None:
        raise HTTPException(404, "No such job.")
    return s


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/jobs", dependencies=[Depends(auth)])
def create_job():
    if jobs.count() >= MAX_JOBS:
        raise HTTPException(503, "The server is full right now. Try again later.")
    return jobs.public(jobs.create())


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str):
    existing(job_id)
    return jobs.public(job_id)


@app.put("/jobs/{job_id}/photos/{name}", dependencies=[Depends(auth)])
async def upload_photo(job_id: str, name: str, request: Request):
    if existing(job_id)["state"] not in ("uploading", "failed"):
        raise HTTPException(409, "Job already started.")
    if not NAME_RE.match(name):
        raise HTTPException(400, "Photo name must look like img_0001.jpg.")
    photos = jobs.job_dir(job_id) / "photos"
    if not (photos / name).exists() and len(list(photos.glob("*.jp*g"))) >= MAX_PHOTOS:
        raise HTTPException(400, f"At most {MAX_PHOTOS} photos per scan.")

    # Write to a temp file and rename, so an interrupted upload never leaves a half photo behind.
    tmp = photos / f".{name}.part"
    size = 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_PHOTO_BYTES:
                f.close()
                tmp.unlink()
                raise HTTPException(413, "Photo too large.")
            f.write(chunk)
    with open(tmp, "rb") as f:
        if f.read(3) != b"\xff\xd8\xff":
            tmp.unlink()
            raise HTTPException(400, "Not a JPEG.")
    os.replace(tmp, photos / name)
    return {"name": name, "bytes": size}


class StartParams(BaseModel):
    flat_base: bool = True


@app.post("/jobs/{job_id}/start", dependencies=[Depends(auth)])
def start_job(job_id: str, params: StartParams):
    if existing(job_id)["state"] not in ("uploading", "failed"):
        raise HTTPException(409, "Job already started.")
    n = len(jobs.public(job_id)["received"])
    if n < MIN_PHOTOS:
        raise HTTPException(400, f"Need at least {MIN_PHOTOS} photos, got {n}.")
    if worker.q.qsize() >= MAX_QUEUED:
        raise HTTPException(503, "Too many scans are waiting. Try again later.")
    worker.submit(job_id, params.model_dump())
    return jobs.public(job_id)


@app.delete("/jobs/{job_id}", dependencies=[Depends(auth)])
def delete_job(job_id: str):
    existing(job_id)
    worker.cancel(job_id)
    shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
    return {"id": job_id, "state": "cancelled", "received": [], "error": None, "report": None}


@app.get("/jobs/{job_id}/result.{fmt}", dependencies=[Depends(auth)])
def result(job_id: str, fmt: str, longest_mm: float | None = Query(None, gt=0, le=2000)):
    if fmt not in M.MEDIA_TYPES:
        raise HTTPException(404, "Unknown format.")
    if existing(job_id)["state"] != "done":
        raise HTTPException(409, "Result not ready.")
    mesh = trimesh.load(jobs.job_dir(job_id) / "model.ply", force="mesh")
    return Response(
        M.export(mesh, fmt, longest_mm),
        media_type=M.MEDIA_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="scan-{job_id[:8]}.{fmt}"'},
    )
