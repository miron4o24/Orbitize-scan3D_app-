"""Съхранение на задачи на диска и една фонова работна нишка (фотограметрията иска цялата машина)."""

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

from . import ai, edit, notify, part, pipeline
from .mesh import MeshError

DATA = Path(os.environ.get("DATA_DIR", "data")).resolve()
STEP_TIMEOUT_S = int(os.environ.get("STEP_TIMEOUT_S", "7200"))
JOB_TTL_DAYS = float(os.environ.get("JOB_TTL_DAYS", "7"))
UPLOAD_TTL_HOURS = float(os.environ.get("UPLOAD_TTL_HOURS", "24"))  # изоставени качвания
ACTIVE = {"queued", "reconstructing", "meshing", "cleaning"}
WINDOWS = os.name == "nt"
log = logging.getLogger("uvicorn.error")


class Cancelled(Exception):
    pass


def job_dir(job_id: str) -> Path:
    return DATA / job_id


def _retry(fn):
    """Windows отказва да замени файл, отворен от друга нишка (и да отвори файл по време на замяна); това трае
    милисекунди."""
    for attempt in range(50):
        try:
            return fn()
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.02)


def read(job_id: str) -> dict | None:
    try:
        return json.loads(_retry((job_dir(job_id) / "status.json").read_text))
    except FileNotFoundError:
        return None


def write(job_id: str, **patch) -> dict:
    """Атомарно обновяване. Проваля се, ако папката на задачата е изтрита, за да не се появи отново отказана задача."""
    status = (read(job_id) or {}) | patch
    tmp = job_dir(job_id) / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    _retry(lambda: os.replace(tmp, job_dir(job_id) / "status.json"))
    return status


def kill(proc: subprocess.Popen):
    """Linux: спира цялата група процеси на инструмента. Windows: инструментите са единични процеси."""
    if WINDOWS:
        proc.kill()
    else:
        os.killpg(proc.pid, signal.SIGKILL)


def low_priority(low: bool) -> dict:
    """Аргументи на Popen за инструмент с приоритет под нормалния (дъщерните му процеси го наследяват)."""
    if not low:
        return {}
    if WINDOWS:
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(10)}


def create(owner: str | None = None) -> str:
    """owner: идентификаторът на потребителя (app/accounts.py); None за споделения API ключ."""
    job_id = uuid.uuid4().hex
    (job_dir(job_id) / "photos").mkdir(parents=True)
    write(job_id, id=job_id, state="uploading", created=time.time(), error=None, report=None, owner=owner)
    return job_id


THUMB = "thumb.jpg"
THUMB_SIDE = 480


PREVIEW = "preview.jpg"
_drawing = threading.Lock()


def thumbnail(job_id: str) -> Path | None:
    """Квадратно изображение на предмета за списъци със сканирания и споделени линкове: след създаване на модела
    негова рисунка (render.py, прави се наново при по-нов модел); дотогава средният кадър, изрязан около центъра
    (приложението държи предмета в кръга там). Запазва се при изчистване на кадрите."""
    d = job_dir(job_id)
    model = d / "model.ply"
    if (read(job_id) or {}).get("state") == "done" and model.exists():
        out = d / PREVIEW
        with _drawing:  # списък със сканирания иска много наведнъж: по една рисунка
            if out.exists() and out.stat().st_mtime >= model.stat().st_mtime:
                return out
            try:
                from . import render

                tmp = out.with_suffix(".tmp")
                render.picture(d).save(tmp, "JPEG", quality=88)
                os.replace(tmp, out)
                return out
            except Exception as e:  # кадърът тогава
                log.warning("preview of %s: %s", job_id, e)
    out = d / THUMB
    if out.exists():
        return out
    frames = sorted((job_dir(job_id) / "photos").glob("*.jp*g"))
    if not frames:
        return None
    from PIL import Image, ImageOps

    tmp = out.with_suffix(".tmp")
    try:
        with Image.open(frames[len(frames) // 2]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            side = round(min(im.size) * 0.9)
            left, top = (im.width - side) // 2, (im.height - side) // 2
            im = im.crop((left, top, left + side, top + side)).resize((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
            im.save(tmp, "JPEG", quality=85)
    except OSError:  # нечетим кадър (UnidentifiedImageError на PIL е такъв)
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, out)
    return out


def video_path(job_id: str) -> Path:
    return job_dir(job_id) / pipeline.VIDEO


def public(job_id: str) -> dict:
    s = read(job_id) or {}
    photos = job_dir(job_id) / "photos"
    received = sorted(p.name for p in photos.iterdir() if not p.name.startswith(".")) if photos.exists() else []
    video = video_path(job_id)
    return {k: s.get(k) for k in ("id", "state", "error", "report", "created", "step", "step_started")} | {
        "received": received,
        "video_bytes": video.stat().st_size if video.exists() else 0,
    }


def log_tail(job_id: str, lines: int) -> str:
    """Последните редове от изхода на инструментите: какво показва приложението при неуспешно сканиране."""
    try:
        with open(job_dir(job_id) / "log.txt", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 256 * 1024))
            text = f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""
    # COLMAP и OpenMVS презаписват напредъка с \r; пази се само крайното състояние на всеки ред.
    rows = [r.rsplit("\r", 1)[-1] for r in text.replace("\r\n", "\n").split("\n")]
    return "\n".join([r for r in rows if r.strip()][-lines:])


def summaries() -> list[dict]:
    """Първо най-новите, без списъците със снимки: за проверка на сървъра отвън."""
    out = [s for d in DATA.iterdir() if (s := read(d.name))]
    out.sort(key=lambda s: s.get("created", 0), reverse=True)
    return [{k: s.get(k) for k in ("id", "state", "error", "step", "created", "report")} for s in out]


def count() -> int:
    """Задачи, които се броят към MAX_JOBS: всички освен завършените сканирания в профили (те имат собствено
    ограничение)."""
    n = 0
    for d in DATA.iterdir():
        s = read(d.name) or {}
        n += not s.get("owner") or s.get("state") in ACTIVE | {"uploading"}
    return n


def cleanup():
    """Нестартираните качвания изчезват след UPLOAD_TTL_HOURS. Завършените задачи след JOB_TTL_DAYS: изцяло, ако са
    без профил; в профил моделът остава (това е запазеното сканиране на потребителя) и изчезват само кадрите, които
    заемат повечето място."""
    now = time.time()
    for d in DATA.iterdir():
        s = read(d.name)
        age = now - (s or {}).get("created", 0)
        if s is None or (s["state"] == "uploading" and age > UPLOAD_TTL_HOURS * 3600):
            shutil.rmtree(d, ignore_errors=True)
        elif s["state"] not in ACTIVE and age > JOB_TTL_DAYS * 86400:
            if not s.get("owner"):
                shutil.rmtree(d, ignore_errors=True)
            elif (d / "photos").exists():
                try:
                    thumbnail(d.name)
                except OSError:
                    pass
                shutil.rmtree(d / "photos", ignore_errors=True)
                (d / "photos").mkdir(exist_ok=True)


PRO_METHODS = ("express", "edit", "part", *ai.METHODS)  # edit: текстова промяна на модел (app/edit.py)


def is_pro(params: dict | None) -> bool:
    return (params or {}).get("method") in PRO_METHODS


class Worker:
    """Две ленти с по една задача наведнъж: безплатни сканирания и тези на Pro (експресното сканиране за около 5
    минути и AI моделите: app/ai.py). Двете делят машината, затова инструментите на безплатната лента са с по-нисък
    приоритет: задача на Pro до безплатно сканиране взема процесора първа и остава в рамките на няколкото си
    минути, а безплатното сканиране просто отнема малко повече време."""

    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        self.ai_q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}  # номер на задача -> инструментът ѝ, който работи в момента
        self.cancelled: set[str] = set()
        self.running: dict[int, str] = {}  # лента (идентификатор на опашката ѝ) -> обработваната там задача

    def ahead(self, job_id: str, params: dict | None) -> int | None:
        """Колко задачи са преди тази чакаща в лентата ѝ, включително обработваната; None, ако не чака."""
        q = self.ai_q if is_pro(params) else self.q
        with q.mutex:
            waiting = list(q.queue)
        if job_id not in waiting:
            return None
        before = [j for j in waiting[: waiting.index(job_id)] if j not in self.cancelled and read(j) is not None]
        return len(before) + (1 if id(q) in self.running else 0)

    def start(self):
        DATA.mkdir(parents=True, exist_ok=True)
        cleanup()
        for d in sorted(DATA.iterdir()):
            s = read(d.name)
            if s and s["state"] in ACTIVE:  # прекъсната от рестарт: изпълнява се отново
                write(d.name, state="queued")
                self._queue(d.name, s.get("params"))
        for q in (self.q, self.ai_q):
            threading.Thread(target=self._loop, args=(q,), daemon=True).start()

    def _queue(self, job_id: str, params: dict | None):
        (self.ai_q if is_pro(params) else self.q).put(job_id)

    def submit(self, job_id: str, params: dict):
        self.cancelled.discard(job_id)
        write(job_id, state="queued", params=params, error=None, report=None, step=None, step_started=None)
        self._queue(job_id, params)

    def cancel(self, job_id: str):
        self.cancelled.add(job_id)
        with self.lock:
            if proc := self.procs.get(job_id):
                kill(proc)

    def _loop(self, q: queue.Queue):
        while True:
            try:
                job_id = q.get(timeout=3600)
            except queue.Empty:
                if q is self.q:
                    cleanup()
                continue
            if job_id in self.cancelled or read(job_id) is None:
                continue
            self.running[id(q)] = job_id
            try:
                self._process(job_id)
            except Exception:  # например изтрита по време на изпълнение задача; работната нишка никога не бива да умира
                log.exception("job %s", job_id)
            finally:
                self.running.pop(id(q), None)

    def _step(self, job_id: str, step: str):
        """Стъпка без инструмент (app/ai.py чака услуга): показва се в приложението и там се спира при отказ."""
        if job_id in self.cancelled:
            raise Cancelled
        write(job_id, step=step, step_started=time.time())

    def _process(self, job_id: str):
        d = job_dir(job_id)
        params = read(job_id).get("params") or {}
        started = time.time()
        set_state = lambda state: write(job_id, state=state, step=None, step_started=time.time())  # noqa: E731
        low = not is_pro(params)
        sh = lambda step, argv: self._sh(job_id, step, argv, low)  # noqa: E731
        method = params.get("method")
        try:
            if method == "edit":
                report = edit.run(d, params, set_state, sh)
            elif method == "part":
                report = part.run(d, params, set_state, sh, lambda step: self._step(job_id, step))
            elif method in ai.METHODS:
                report = ai.run(d, method, bool(params.get("flat_base", True)), set_state, sh,
                                lambda step: self._step(job_id, step), params.get("tune"))  # fmt: skip
            elif method == "express":
                report = pipeline.run(d, bool(params.get("flat_base", True)), set_state, sh,
                                      express=True)  # fmt: skip
            else:
                raise ValueError(f"Unsupported reconstruction method: {method!r}")
            write(job_id, state="done", step=None, report=report | {"seconds": round(time.time() - started)})
            try:
                (d / THUMB).unlink(missing_ok=True)  # повторно изпълнение може да е избрало други кадри
                thumbnail(job_id)  # изображението на модела, създадено сега, а не при първата заявка от списък
            except OSError:
                pass
        except Cancelled:
            pass
        except (pipeline.Failed, MeshError) as e:
            pipeline.note(d, f"FAILED: {e}")
            write(job_id, state="failed", error=str(e))
        except Exception as e:
            log.exception("job %s", job_id)
            pipeline.note(d, f"INTERNAL ERROR: {e!r}")
            write(job_id, state="failed", error="Възникна грешка при обработката. Опитайте отново.")
        finally:
            shutil.rmtree(d / "work", ignore_errors=True)
            shutil.rmtree(d / pipeline.MORE_FRAMES, ignore_errors=True)
            final = read(job_id)
            if final and final.get("state") in ("done", "failed") and job_id not in self.cancelled:
                notify.job_finished(job_id, final)

    def _sh(self, job_id: str, step: str, argv: list, low: bool = False):
        """low: приоритет под нормалния (безплатни сканирания, така че задачите на Pro до тях да минат първи)."""
        if job_id in self.cancelled:
            raise Cancelled
        write(job_id, step=step, step_started=time.time())
        with open(job_dir(job_id) / "log.txt", "a", encoding="utf-8", errors="replace") as logf:
            logf.write(f"\n[{time.strftime('%H:%M:%S')}] {step}\n$ {' '.join(map(str, argv))}\n")
            logf.flush()
            with self.lock:
                # cwd: OpenMVS записва собствените си файлове .log в текущата папка.
                proc = self.procs[job_id] = subprocess.Popen(
                    [str(a) for a in argv],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    cwd=job_dir(job_id),
                    start_new_session=not WINDOWS,
                    **low_priority(low),
                )
            try:
                rc = proc.wait(timeout=STEP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                kill(proc)
                logf.write(f"\ntimed out during {step}\n")
                raise pipeline.Failed("Обработката продължи твърде дълго и беше спряна. Опитайте с по-кратко видео.")
            finally:
                with self.lock:
                    self.procs.pop(job_id, None)
                # OpenMVS не печата нищо при пренасочване; вместо това записва <Tool>-<време>.log до файловете си.
                for f in sorted(job_dir(job_id).rglob("*-*.log")):
                    logf.write(f.read_text(errors="replace"))
                    f.unlink(missing_ok=True)
        if job_id in self.cancelled:
            raise Cancelled
        if rc != 0:
            # 137 = SIGKILL отвън: почти винаги при изчерпване на паметта. 0xC0000017 / 0xC0000409: "няма достатъчно
            # памет" на Windows и прекратяването, с което завършва неуспешно заделяне.
            oom = rc in (-9, 137) or (rc & 0xFFFFFFFF) in (0xC0000017, 0xC0000409)
            pipeline.note(job_dir(job_id), f"step {step} failed with exit code {rc}{' (out of memory)' if oom else ''}")
            hint = " Опитайте с по-кратко видео." if oom else " Опитайте отново."
            raise pipeline.Failed(f"Обработката спря с грешка.{hint}")
