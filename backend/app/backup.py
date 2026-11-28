"""Резервно копие на сървъра: профилите (базата данни), снимките на профилите и готовите модели. Пуска се всеки ден
от scripts/backup.ps1 (задача „scan3d-backup“); пази последните BACKUP_KEEP копия в BACKUP_DIR.

python -m app.backup            # ново копие
python -m app.backup --restore <папка или .zip>   # връща копие (сървърът трябва да е спрян)

Без временните файлове на обработката (снимки, видео, кадри, work): те изтичат така или иначе след JOB_TTL_DAYS. SQLite
се копира с вградения механизъм за резервни копия (безопасно, докато сървърът работи); MySQL - всяка таблица като
JSON редове."""

import json
import os
import shutil
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import sqlalchemy as sa

from . import accounts, db, jobs

ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", ROOT / "backups")).resolve()
BACKUP_KEEP = int(os.environ.get("BACKUP_KEEP", "7"))
SKIP_DIRS = {"photos", "work", "more_frames", "build.tmp", "preview.tmp"}
SKIP_FILES = {"video.mp4", "log.txt"}


def _job_files(d: Path):
    for p in d.rglob("*"):
        rel = p.relative_to(d)
        if p.is_file() and not (set(rel.parts[:-1]) & SKIP_DIRS) and p.name not in SKIP_FILES \
                and not p.name.endswith((".part", ".tmp")):  # fmt: skip
            yield p


def sqlite_path() -> Path | None:
    """Файлът на базата данни, ако е SQLite (по подразбиране), иначе None."""
    url = db.url()
    if url is None or sa.make_url(url).get_backend_name() != "sqlite":
        return None
    return Path(sa.make_url(url).database).resolve()


def _database(z: zipfile.ZipFile, tmp: Path):
    if db.url() is None:
        return
    src = sqlite_path()
    if src is not None:
        if src.exists():
            copy = tmp / "accounts.db"
            with sqlite3.connect(src) as a, sqlite3.connect(copy) as b:
                a.backup(b)  # последователно копие, докато сървърът пише
            z.write(copy, "accounts.db")
        return
    e = db.engine()
    if e is None:
        raise RuntimeError("the database is not reachable")
    with e.connect() as c:
        for name, table in db.meta.tables.items():
            rows = [dict(r._mapping) for r in c.execute(sa.select(table))]
            z.writestr(f"database/{name}.jsonl", "\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in rows))


def make() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(now)) + f"{int(now * 1000) % 1000:03d}"
    out = BACKUP_DIR / f"orbitize-{stamp}.zip"
    part = out.with_suffix(".zip.part")
    tmp = BACKUP_DIR / f".tmp-{stamp}"
    tmp.mkdir()
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            _database(z, tmp)
            if accounts.AVATARS.exists():
                for p in accounts.AVATARS.glob("*.jpg"):
                    z.write(p, f"avatars/{p.name}")
            if jobs.DATA.exists():
                for d in sorted(jobs.DATA.iterdir()):
                    if d.is_dir() and len(d.name) == 32:
                        for p in _job_files(d):
                            z.write(p, f"jobs/{d.name}/{p.relative_to(d).as_posix()}")
        os.replace(part, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        part.unlink(missing_ok=True)
    for old in sorted(BACKUP_DIR.glob("orbitize-*.zip"))[:-BACKUP_KEEP]:
        old.unlink()
    return out


def restore(src: Path) -> None:
    """Връща копие върху празни (или стари) папки. Сървърът трябва да е спрян. Само SQLite: при MySQL таблиците са в
    database/*.jsonl за ръчно връщане."""
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise SystemExit(f"unsafe path in backup: {name}")
            if name == "accounts.db":
                dest = sqlite_path() or db.sqlite_file()
            elif name.startswith("avatars/"):
                dest = accounts.AVATARS / Path(name).name
            elif name.startswith("jobs/"):
                dest = jobs.DATA / Path(*Path(name).parts[1:])
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as a, open(dest, "wb") as b:
                shutil.copyfileobj(a, b)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--restore":
        restore(Path(sys.argv[2]))
        print("restored", sys.argv[2])
    else:
        path = make()
        print(path, f"{path.stat().st_size / 1e6:.1f} MB")
