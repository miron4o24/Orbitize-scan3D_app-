"""Общността: завършени сканирания, които собствениците им публикуват, за да може всеки влязъл потребител да ги
разглежда, харесва и изтегля. Публикуването задава scans.published (и реалния размер от автора, scans.longest_mm);
харесванията и сигналите са в собствени таблици (app/db.py). Публикация, за която са сигнализирали HIDE_AFTER_REPORTS
души, напуска списъка за всички освен автора си, докато администраторът я прегледа (GET /admin/community/reports,
после DELETE или POST .../keep)."""

import io
import os
import re
import time
import uuid

import sqlalchemy as sa
import trimesh
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import ar as AR
from . import db, jobs, progress
from . import mesh as M
from . import plans as PL
from . import stress as ST
from . import texture as TX
from .accounts import Principal, admin_only, avatar_path, need_db, owned_row, signed_in

PAGE = 20
HIDE_AFTER_REPORTS = 3
ID_RE = re.compile(r"^[0-9a-f]{32}$")  # номера на задача и потребител

router = APIRouter()


def counted(table: sa.Table) -> sa.Subquery:
    return sa.select(table.c.job_id, sa.func.count().label("n")).group_by(table.c.job_id).subquery()


def posts(me: str) -> sa.Select:
    """Всяка публикация с автора, харесванията и сигналите и дали `me` я е харесал."""
    likes, reports = counted(db.likes), counted(db.reports)
    liked = sa.exists().where(db.likes.c.job_id == db.scans.c.job_id, db.likes.c.user_id == me)
    return (
        sa.select(
            db.scans.c.job_id, db.scans.c.title, db.scans.c.published, db.scans.c.longest_mm, db.scans.c.user_id,
            db.users.c.name, db.users.c.avatar, db.users.c.google_picture,
            sa.func.coalesce(likes.c.n, 0).label("likes"),
            sa.func.coalesce(reports.c.n, 0).label("reports"),
            liked.label("liked"),
        )  # fmt: skip
        .join(db.users, db.users.c.id == db.scans.c.user_id)
        .outerjoin(likes, likes.c.job_id == db.scans.c.job_id)
        .outerjoin(reports, reports.c.job_id == db.scans.c.job_id)
        .where(db.scans.c.published.is_not(None))
        .where(sa.or_(sa.func.coalesce(reports.c.n, 0) < HIDE_AFTER_REPORTS, db.scans.c.user_id == me))
    )


def entry(r: sa.Row, status: dict, me: str, points: dict[str, int]) -> dict:
    """points: точките на авторите (progress.points_of), за нивото им."""
    report = status.get("report") or {}
    return {
        "id": r.job_id,
        "title": r.title,
        "published": r.published,
        "longest_mm": r.longest_mm or 100.0,
        "likes": int(r.likes),
        "liked": bool(r.liked),
        "mine": r.user_id == me,
        "dimensions_mm": report.get("dimensions_mm"),
        "triangles": report.get("triangles"),
        "author": {"id": r.user_id, "name": r.name, "has_avatar": bool(r.avatar), "google_picture": r.google_picture,
                   "level": progress.level_of(points.get(r.user_id, 0))["number"]},  # fmt: skip
    }


def done(job_id: str) -> dict | None:
    s = jobs.read(job_id)
    return s if s and s.get("state") == "done" else None


def one(job_id: str, me: str) -> dict:
    row = None
    if ID_RE.match(job_id):
        with need_db().connect() as c:
            row = c.execute(posts(me).where(db.scans.c.job_id == job_id)).first()
    status = done(job_id) if row else None
    if status is None:
        raise HTTPException(404, "Моделът вече не е в общността.")
    return entry(row, status, me, progress.points_of([row.user_id]))


@router.get("/community")
def feed(sort: str = Query("new", pattern="^(new|top)$"), offset: int = Query(0, ge=0, le=100_000),
         p: Principal = Depends(signed_in)):  # fmt: skip
    q = posts(p.user_id)
    if sort == "top":
        q = q.order_by(sa.desc("likes"), db.scans.c.published.desc())
    else:
        q = q.order_by(db.scans.c.published.desc())
    with need_db().connect() as c:
        rows = c.execute(q.limit(PAGE).offset(offset)).all()
    points = progress.points_of([r.user_id for r in rows])
    out = [entry(r, s, p.user_id, points) for r in rows if (s := done(r.job_id))]
    return {"posts": out, "next": offset + PAGE if len(rows) == PAGE else None}


@router.get("/community/{job_id}")
def get_post(job_id: str, p: Principal = Depends(signed_in)):
    return one(job_id, p.user_id)


@router.get("/community/{job_id}/thumb.jpg")
def post_thumb(job_id: str, p: Principal = Depends(signed_in)):
    one(job_id, p.user_id)
    path = jobs.thumbnail(job_id)
    if path is None:
        raise HTTPException(404, "Няма снимка.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


@router.get("/community/{job_id}/model.glb")
def post_model(job_id: str, view: bool = False, p: Principal = Depends(signed_in)):
    """С текстура от снимки, като собствените модели на приложението (чете ги през lib/photoTexture.ts)."""
    one(job_id, p.user_id)
    job = jobs.job_dir(job_id)
    light = AR.view_file(job) if view else None
    if light:
        return Response(light.read_bytes(), media_type=M.MEDIA_TYPES["glb"], headers={"Cache-Control": "private, max-age=300"})
    mesh = trimesh.load(job / "model.ply", force="mesh")
    return Response(M.export(mesh, "glb", None, TX.load(job)), media_type=M.MEDIA_TYPES["glb"],
                    headers={"Cache-Control": "private, max-age=300"})  # fmt: skip


@router.get("/community/{job_id}/stress/{case}")
def post_stress(job_id: str, case: str, p: Principal = Depends(signed_in)):
    """Всеки може да направи тест за здравина на публикация (app/stress.py), както може и да я изтегли; за безплатните
    профили тя се брои към тестваните им модели (app/plans.py)."""
    one(job_id, p.user_id)
    PL.check(p, "stress", job_id)
    out = ST.summary_response(job_id, case, p.user_id)
    PL.record(p, "stress", job_id)
    return out


@router.get("/community/{job_id}/stress/{case}/model.glb")
def post_stress_model(job_id: str, case: str, p: Principal = Depends(signed_in)):
    one(job_id, p.user_id)
    PL.check(p, "stress", job_id)
    return ST.model_response(job_id, case)


@router.get("/community/authors/{user_id}/avatar.jpg")
def author_avatar(user_id: str, p: Principal = Depends(signed_in)):
    """Само снимките на хора с нещо в общността: ничии други не се показват там."""
    with need_db().connect() as c:
        posted = c.execute(sa.select(db.scans.c.job_id)
                           .where(db.scans.c.user_id == user_id, db.scans.c.published.is_not(None))).first()  # fmt: skip
    path = avatar_path(user_id) if ID_RE.match(user_id) else None
    if not posted or path is None or not path.exists():
        raise HTTPException(404, "Няма снимка.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


# --- Публикуване ----------------------------------------------------------------------------------------------------


class Publish(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    longest_mm: float = Field(100.0, gt=0, le=2000)


@router.post("/me/scans/{job_id}/publish")
def publish(job_id: str, body: Publish, p: Principal = Depends(signed_in)):
    row = owned_row(job_id, p.user_id)
    if done(job_id) is None:
        raise HTTPException(409, "Само готов модел може да се публикува.")
    title = " ".join(body.title.split())
    if not title:
        raise HTTPException(400, "Първо дайте име на модела.")
    with need_db().begin() as c:
        c.execute(sa.update(db.scans).where(db.scans.c.job_id == job_id)
                  .values(title=title, longest_mm=body.longest_mm,
                          published=row.published or int(time.time())))  # fmt: skip
    return one(job_id, p.user_id)


@router.delete("/me/scans/{job_id}/publish")
def unpublish(job_id: str, p: Principal = Depends(signed_in)):
    owned_row(job_id, p.user_id)
    with need_db().begin() as c:
        c.execute(sa.update(db.scans).where(db.scans.c.job_id == job_id).values(published=None))
        for t in (db.likes, db.reports):  # при по-късно повторно публикуване започва отначало
            c.execute(sa.delete(t).where(t.c.job_id == job_id))
    return {"published": None}


# --- Харесвания и сигнали -------------------------------------------------------------------------------------------


@router.put("/community/{job_id}/like")
def like(job_id: str, p: Principal = Depends(signed_in)):
    one(job_id, p.user_id)
    try:
        with need_db().begin() as c:
            if not c.execute(sa.select(db.likes.c.job_id)
                             .where(db.likes.c.job_id == job_id, db.likes.c.user_id == p.user_id)).first():  # fmt: skip
                c.execute(sa.insert(db.likes).values(job_id=job_id, user_id=p.user_id, created=int(time.time())))
    except sa.exc.IntegrityError:  # второ натискане, пристигнало по същото време
        pass
    post = one(job_id, p.user_id)
    return {"likes": post["likes"], "liked": post["liked"]}


@router.delete("/community/{job_id}/like")
def unlike(job_id: str, p: Principal = Depends(signed_in)):
    one(job_id, p.user_id)
    with need_db().begin() as c:
        c.execute(sa.delete(db.likes).where(db.likes.c.job_id == job_id, db.likes.c.user_id == p.user_id))
    post = one(job_id, p.user_id)
    return {"likes": post["likes"], "liked": post["liked"]}


class Report(BaseModel):
    reason: str = Field("", max_length=200)


@router.post("/community/{job_id}/report")
def report(job_id: str, body: Report, p: Principal = Depends(signed_in)):
    post = one(job_id, p.user_id)
    if post["mine"]:
        raise HTTPException(400, "Това е вашият модел.")
    try:
        with need_db().begin() as c:
            if not c.execute(sa.select(db.reports.c.job_id)
                             .where(db.reports.c.job_id == job_id, db.reports.c.user_id == p.user_id)).first():  # fmt: skip
                c.execute(sa.insert(db.reports).values(job_id=job_id, user_id=p.user_id, reason=body.reason.strip(),
                                                       created=int(time.time())))  # fmt: skip
    except sa.exc.IntegrityError:
        pass
    return {"ok": True}


# --- Снимки на отпечатани модели --- По една на човек и модел, пазят се със задачата на модела. Прекодират се при
# пристигане: снимка от телефон носи информация къде е направена. Скриват се след HIDE_AFTER_REPORTS сигнала;
# създателят ѝ и авторът на модела могат да я премахнат.

MAX_PHOTO = 12 * 1024 * 1024
PRINT_SIDE = 1600


def print_file(job_id: str, print_id: str):
    return jobs.job_dir(job_id) / "prints" / f"{print_id}.jpg"


def print_row(print_id: str) -> sa.Row:
    row = None
    if ID_RE.match(print_id):
        with need_db().connect() as c:
            row = c.execute(sa.select(db.prints).where(db.prints.c.id == print_id)).first()
    if row is None:
        raise HTTPException(404, "Снимката вече я няма.")
    return row


@router.get("/community/{job_id}/prints")
def list_prints(job_id: str, p: Principal = Depends(signed_in)):
    post = one(job_id, p.user_id)
    reported = (sa.select(db.print_reports.c.print_id, sa.func.count().label("n"))
                .group_by(db.print_reports.c.print_id).subquery())  # fmt: skip
    with need_db().connect() as c:
        rows = c.execute(sa.select(db.prints.c.id, db.prints.c.user_id, db.prints.c.created, db.users.c.name,
                                   sa.func.coalesce(reported.c.n, 0).label("reports"))
                         .join(db.users, db.users.c.id == db.prints.c.user_id)
                         .outerjoin(reported, reported.c.print_id == db.prints.c.id)
                         .where(db.prints.c.job_id == job_id)
                         .order_by(db.prints.c.created.desc())).all()  # fmt: skip
    return [{"id": r.id, "name": r.name, "created": r.created, "mine": r.user_id == p.user_id,
             "can_remove": r.user_id == p.user_id or post["mine"]}
            for r in rows if r.reports < HIDE_AFTER_REPORTS or r.user_id == p.user_id or post["mine"]]  # fmt: skip


@router.put("/community/{job_id}/prints")
async def add_print(job_id: str, request: Request, p: Principal = Depends(signed_in)):
    """Извикващият е отпечатал модела: неговата снимка на модела, на мястото на по-ранна."""
    one(job_id, p.user_id)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_PHOTO:
            raise HTTPException(413, "Снимката е твърде голяма.")
    try:
        with Image.open(io.BytesIO(bytes(body))) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((PRINT_SIDE, PRINT_SIDE), Image.LANCZOS)
            out = io.BytesIO()
            im.save(out, "JPEG", quality=85)
    except OSError:
        raise HTTPException(400, "Файлът не е снимка.")
    print_id = uuid.uuid4().hex
    path = print_file(job_id, print_id)
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(out.getvalue())
    os.replace(tmp, path)
    with need_db().begin() as c:
        old = c.execute(sa.select(db.prints.c.id).where(db.prints.c.job_id == job_id, db.prints.c.user_id == p.user_id)).all()
        c.execute(sa.delete(db.prints).where(db.prints.c.job_id == job_id, db.prints.c.user_id == p.user_id))
        c.execute(sa.insert(db.prints).values(id=print_id, job_id=job_id, user_id=p.user_id, created=int(time.time())))
    for r in old:
        print_file(job_id, r.id).unlink(missing_ok=True)
    return {"id": print_id}


@router.get("/community/prints/{print_id}.jpg")
def get_print(print_id: str, p: Principal = Depends(signed_in)):
    row = print_row(print_id)
    one(row.job_id, p.user_id)
    path = print_file(row.job_id, print_id)
    if not path.exists():
        raise HTTPException(404, "Снимката вече я няма.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.delete("/community/prints/{print_id}")
def remove_print(print_id: str, p: Principal = Depends(signed_in)):
    row = print_row(print_id)
    if row.user_id != p.user_id and not one(row.job_id, p.user_id)["mine"]:
        raise HTTPException(403, "Това не е ваша снимка.")
    with need_db().begin() as c:
        c.execute(sa.delete(db.prints).where(db.prints.c.id == print_id))
    print_file(row.job_id, print_id).unlink(missing_ok=True)
    return {"ok": True}


@router.post("/community/prints/{print_id}/report")
def report_print(print_id: str, p: Principal = Depends(signed_in)):
    row = print_row(print_id)
    if row.user_id == p.user_id:
        raise HTTPException(400, "Това е вашата снимка.")
    try:
        with need_db().begin() as c:
            if not c.execute(sa.select(db.print_reports.c.print_id).where(db.print_reports.c.print_id == print_id,
                                                                          db.print_reports.c.user_id == p.user_id)).first():  # fmt: skip
                c.execute(sa.insert(db.print_reports).values(print_id=print_id, user_id=p.user_id, created=int(time.time())))
    except sa.exc.IntegrityError:
        pass
    return {"ok": True}


# --- Преглед на сигналите от администратора -------------------------------------------------------------------------


@router.get("/admin/community/reports", dependencies=[Depends(admin_only)])
def reported():
    with need_db().connect() as c:
        rows = c.execute(sa.select(db.reports.c.job_id, db.reports.c.reason, db.reports.c.created, db.scans.c.title,
                                   db.scans.c.user_id)
                         .join(db.scans, db.scans.c.job_id == db.reports.c.job_id)
                         .order_by(db.reports.c.created.desc())).all()  # fmt: skip
    out: dict[str, dict] = {}
    for r in rows:
        post = out.setdefault(r.job_id, {"id": r.job_id, "title": r.title, "author": r.user_id, "reasons": []})
        post["reasons"].append(r.reason)
    return [post | {"reports": len(post["reasons"]), "hidden": len(post["reasons"]) >= HIDE_AFTER_REPORTS}
            for post in out.values()]  # fmt: skip


@router.post("/admin/community/{job_id}/keep", dependencies=[Depends(admin_only)])
def keep(job_id: str):
    """Сигналите са неоснователни: изчистват се и публикацията се връща в списъка."""
    with need_db().begin() as c:
        c.execute(sa.delete(db.reports).where(db.reports.c.job_id == job_id))
    return {"ok": True}


@router.delete("/admin/community/{job_id}", dependencies=[Depends(admin_only)])
def take_down(job_id: str):
    """Извън общността; самото сканиране остава на собственика си."""
    with need_db().begin() as c:
        c.execute(sa.update(db.scans).where(db.scans.c.job_id == job_id).values(published=None))
        for t in (db.likes, db.reports):
            c.execute(sa.delete(t).where(t.c.job_id == job_id))
    return {"published": None}
