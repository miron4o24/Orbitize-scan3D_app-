"""Job storage on disk + one background worker (photogrammetry needs the whole machine)."""

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import pipeline
from .mesh import MeshError

DATA = Path(os.environ.get("DATA_DIR", "data")).resolve()
STEP_TIMEOUT_S = int(os.environ.get("STEP_TIMEOUT_S", "7200"))
JOB_TTL_DAYS = float(os.environ.get("JOB_TTL_DAYS", "7"))
UPLOAD_TTL_HOURS = float(os.environ.get("UPLOAD_TTL_HOURS", "24"))  # abandoned uploads
ACTIVE = {"queued", "reconstructing", "meshing", "cleaning"}
log = logging.getLogger("uvicorn.error")


class Cancelled(Exception):
    pass


def job_dir(job_id: str) -> Path:
    return DATA / job_id


def read(job_id: str) -> dict | None:
    try:
        return json.loads((job_dir(job_id) / "status.json").read_text())
    except FileNotFoundError:
        return None


def write(job_id: str, **patch) -> dict:
    """Atomic update. Fails if the job directory was deleted, so a cancelled job cannot reappear."""
    status = (read(job_id) or {}) | patch
    tmp = job_dir(job_id) / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    os.replace(tmp, job_dir(job_id) / "status.json")
    return status


def create() -> str:
    job_id = uuid.uuid4().hex
    (job_dir(job_id) / "photos").mkdir(parents=True)
    write(job_id, id=job_id, state="uploading", created=time.time(), error=None, report=None)
    return job_id


def public(job_id: str) -> dict:
    s = read(job_id) or {}
    photos = job_dir(job_id) / "photos"
    received = sorted(p.name for p in photos.iterdir() if not p.name.startswith(".")) if photos.exists() else []
    return {k: s.get(k) for k in ("id", "state", "error", "report")} | {"received": received}


def count() -> int:
    return sum(1 for _ in DATA.iterdir())


def cleanup():
    """Delete finished jobs after JOB_TTL_DAYS and never-started uploads after UPLOAD_TTL_HOURS."""
    now = time.time()
    for d in DATA.iterdir():
        s = read(d.name)
        age = now - (s or {}).get("created", 0)
        if s is None or (s["state"] == "uploading" and age > UPLOAD_TTL_HOURS * 3600):
            shutil.rmtree(d, ignore_errors=True)
        elif s["state"] not in ACTIVE and age > JOB_TTL_DAYS * 86400:
            shutil.rmtree(d, ignore_errors=True)


class Worker:
    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.current: str | None = None
        self.proc: subprocess.Popen | None = None
        self.cancelled: set[str] = set()

    def start(self):
        DATA.mkdir(parents=True, exist_ok=True)
        cleanup()
        for d in sorted(DATA.iterdir()):
            s = read(d.name)
            if s and s["state"] in ACTIVE:  # interrupted by a restart: run again
                write(d.name, state="queued")
                self.q.put(d.name)
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, job_id: str, params: dict):
        self.cancelled.discard(job_id)
        write(job_id, state="queued", params=params, error=None, report=None)
        self.q.put(job_id)

    def cancel(self, job_id: str):
        self.cancelled.add(job_id)
        with self.lock:
            if self.current == job_id and self.proc:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def _loop(self):
        while True:
            try:
                job_id = self.q.get(timeout=3600)
            except queue.Empty:
                cleanup()
                continue
            if job_id in self.cancelled or read(job_id) is None:
                continue
            self.current = job_id
            try:
                self._process(job_id)
            except Exception:  # e.g. job deleted mid-run; never let the worker thread die
                log.exception("job %s", job_id)
            finally:
                self.current = None

    def _process(self, job_id: str):
        d = job_dir(job_id)
        params = read(job_id).get("params") or {}
        started = time.time()
        try:
            report = pipeline.run(
                d,
                bool(params.get("flat_base", True)),
                lambda state: write(job_id, state=state),
                lambda step, argv: self._sh(job_id, step, argv),
            )
            write(job_id, state="done", report=report | {"seconds": round(time.time() - started)})
        except Cancelled:
            pass
        except (pipeline.Failed, MeshError) as e:
            write(job_id, state="failed", error=str(e))
        except Exception:
            log.exception("job %s", job_id)
            write(job_id, state="failed", error="Internal error while processing. See the server log.")
        finally:
            shutil.rmtree(d / "work", ignore_errors=True)

    def _sh(self, job_id: str, step: str, argv: list):
        if job_id in self.cancelled:
            raise Cancelled
        with open(job_dir(job_id) / "log.txt", "a") as logf:
            logf.write(f"\n$ {' '.join(map(str, argv))}\n")
            logf.flush()
            with self.lock:
                self.proc = subprocess.Popen(
                    [str(a) for a in argv], stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
                )
            try:
                rc = self.proc.wait(timeout=STEP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                raise pipeline.Failed(f"Processing timed out during {step}. Try fewer photos.")
            finally:
                with self.lock:
                    self.proc = None
        if job_id in self.cancelled:
            raise Cancelled
        if rc != 0:
            # 137 = SIGKILL from outside us: almost always the out-of-memory killer.
            hint = " The server ran out of memory; try fewer photos." if rc in (-9, 137) else ""
            raise pipeline.Failed(f"Processing failed during {step}.{hint}")
