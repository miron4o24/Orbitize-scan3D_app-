"""Администриране от приложението (екранът „Администриране“): преглед на сървъра, профилите и техния статус, работата
на опашката и дали пощата за кодовете за потвърждение работи. Достъп имат собствениците (accounts.OWNER_EMAILS), профили,
на които собственик е дал права на администратор (users.admin), и администраторският ключ на сървъра.

Правила: никой не променя себе си и собствениците (те се задават само на сървъра); правата на администратор и
изтриването на администратор са само за собственика; блокиран профил губи всичките си сесии и не може да влезе
(accounts.principal, login). Пощенската кутия за кодовете е само в .env на сървъра (app/mail.py)."""

import re
import shutil
import time

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from . import accounts as A
from . import community, db, jobs, mail
from .accounts import Principal, staff_only

router = APIRouter(prefix="/manage")

PAGE = 50
WEEK_S = 7 * 86400


def _me(p: Principal) -> tuple[str | None, bool]:
    """Кой е администраторът и дали е собственик (администраторският ключ се смята за собственик)."""
    if p.user_id is None:
        return None, True
    return p.user_id, A.is_owner_email(A.user_email(p.user_id))


def _user_row(c, user_id: str) -> sa.Row:
    u = c.execute(sa.select(db.users).where(db.users.c.id == user_id)).first()
    if u is None:
        raise HTTPException(404, "Няма такъв профил.")
    return u


def _scan_counts(c) -> dict[str, int]:
    rows = c.execute(sa.select(db.scans.c.user_id, sa.func.count()).group_by(db.scans.c.user_id)).all()
    return {r[0]: r[1] for r in rows}


def _user_out(u: sa.Row, scans: int) -> dict:
    owner = A.is_owner_email(u.email)
    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "created": u.created,
        "last_seen": u.last_seen,
        "scans": scans,
        "verified": not A.waiting(u),
        "blocked": bool(u.blocked),
        "admin": owner or bool(u.admin),
        "owner": owner,
        "google": not u.google_sub.startswith("email:"),
        "has_avatar": bool(u.avatar),
        "google_picture": u.google_picture,
    }


# --- Преглед ------------------------------------------------------------------------------------------------------


@router.get("/overview")
def overview(p: Principal = Depends(staff_only)):
    now = int(time.time())
    with A.need_db().connect() as c:
        users = c.execute(sa.select(db.users.c.email, db.users.c.created, db.users.c.last_seen, db.users.c.blocked,
                                    db.users.c.admin, db.users.c.verify_code)).all()  # fmt: skip
        published = c.execute(sa.select(sa.func.count()).select_from(db.scans).where(db.scans.c.published.is_not(None))).scalar()
        reported = c.execute(sa.select(sa.func.count(sa.distinct(db.reports.c.job_id)))).scalar()
    states: dict[str, int] = {}
    failed_week = 0
    for d in jobs.DATA.iterdir():
        s = jobs.read(d.name)
        if not s:
            continue
        states[s["state"]] = states.get(s["state"], 0) + 1
        failed_week += s["state"] == "failed" and now - s.get("created", 0) < WEEK_S
    disk = shutil.disk_usage(jobs.DATA)
    return {
        "users": {
            "total": len(users),
            "new_week": sum(1 for u in users if now - u.created < WEEK_S),
            "active_week": sum(1 for u in users if u.last_seen and now - u.last_seen < WEEK_S),
            "unverified": sum(1 for u in users if u.verify_code),
            "blocked": sum(1 for u in users if u.blocked),
            "admins": sum(1 for u in users if u.admin or A.is_owner_email(u.email)),
        },
        "jobs": {
            "total": sum(states.values()),
            "done": states.get("done", 0),
            "working": sum(states.get(k, 0) for k in jobs.ACTIVE - {"queued"}),
            "queued": states.get("queued", 0),
            "uploading": states.get("uploading", 0),
            "failed_week": failed_week,
        },
        "community": {"published": published, "reported": reported},
        "server": {"disk_free_gb": round(disk.free / 1e9, 1), "disk_total_gb": round(disk.total / 1e9, 1),
                   "database": db.status(), "mail": mail.configured(), "version": A.health_info()["version"]},  # fmt: skip
        "me": {"owner": _me(p)[1]},
    }


# --- Профили ------------------------------------------------------------------------------------------------------


@router.get("/users")
def users(
    q: str = Query("", max_length=100),
    show: str = Query("all", pattern="^(all|blocked|admins|unverified)$"),
    offset: int = Query(0, ge=0),
    p: Principal = Depends(staff_only),
):
    """Профилите, най-новите първи; q търси в името и имейла."""
    sel = sa.select(db.users).order_by(db.users.c.created.desc())
    if q.strip():
        like = f"%{q.strip().lower()}%"
        sel = sel.where(sa.or_(sa.func.lower(db.users.c.name).like(like), sa.func.lower(db.users.c.email).like(like)))
    if show == "blocked":
        sel = sel.where(db.users.c.blocked == sa.true())
    elif show == "unverified":
        sel = sel.where(db.users.c.verify_code.is_not(None))
    with A.need_db().connect() as c:
        rows = c.execute(sel).all()
        counts = _scan_counts(c)
    out = [_user_out(u, counts.get(u.id, 0)) for u in rows]
    if show == "admins":  # собствениците са по имейл, не в базата данни
        out = [u for u in out if u["admin"]]
    return {"total": len(out), "users": out[offset : offset + PAGE], "next": offset + PAGE if offset + PAGE < len(out) else None}


@router.get("/users/{user_id}")
def user_detail(user_id: str, p: Principal = Depends(staff_only)):
    with A.need_db().connect() as c:
        u = _user_row(c, user_id)
        rows = c.execute(sa.select(db.scans).where(db.scans.c.user_id == user_id).order_by(db.scans.c.created.desc())).all()
        sessions = c.execute(sa.select(sa.func.count()).select_from(db.sessions).where(db.sessions.c.user_id == user_id)).scalar()
    scans = []
    for r in rows:
        s = jobs.read(r.job_id)
        if s is None:
            continue
        report = s.get("report") or {}
        scans.append({"id": r.job_id, "title": r.title, "created": r.created, "state": s["state"], "error": s.get("error"),
                      "method": report.get("ai") or (s.get("params") or {}).get("method"), "published": r.published,
                      "thumb": A.scan_entry(r, s)["thumb"]})  # fmt: skip
    me, _ = _me(p)
    return _user_out(u, len(scans)) | {"sessions": sessions, "scans_list": scans, "is_me": user_id == me}


@router.get("/users/{user_id}/avatar.jpg")
def user_avatar(user_id: str, p: Principal = Depends(staff_only)):
    path = A.avatar_path(user_id) if re.fullmatch(r"[0-9a-f]{32}", user_id) else None
    if path is None or not path.exists():
        raise HTTPException(404, "Няма снимка.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


class UserPatch(BaseModel):
    blocked: bool | None = None
    admin: bool | None = None
    verified: bool | None = None  # True: потвърждава имейла ръчно (например без пощенска кутия на сървъра)
    name: str | None = Field(None, min_length=1, max_length=100)


def _guard(p: Principal, u: sa.Row, owner_only_change: bool):
    me, i_am_owner = _me(p)
    if u.id == me:
        raise HTTPException(400, "Не можете да променяте собствения си профил оттук.")
    if A.is_owner_email(u.email):
        raise HTTPException(403, "Собственикът на приложението не може да се променя оттук.")
    if (owner_only_change or u.admin) and not i_am_owner:
        raise HTTPException(403, "Само собственикът може да променя администраторите.")


@router.patch("/users/{user_id}")
def change_user(user_id: str, body: UserPatch, p: Principal = Depends(staff_only)):
    with A.need_db().begin() as c:
        u = _user_row(c, user_id)
        _guard(p, u, owner_only_change=body.admin is not None)
        values: dict = {}
        if body.blocked is not None:
            values["blocked"] = body.blocked
            if body.blocked:  # излиза от всеки телефон веднага
                c.execute(sa.delete(db.sessions).where(db.sessions.c.user_id == user_id))
        if body.admin is not None:
            values["admin"] = body.admin
        if body.verified:
            values |= {"verify_code": None, "verify_expires": None}
        if body.name is not None:
            values["name"] = " ".join(body.name.split())[:100]
        if values:
            c.execute(sa.update(db.users).where(db.users.c.id == user_id).values(**values))
    return user_detail(user_id, p)


@router.post("/users/{user_id}/signout")
def sign_out_everywhere(user_id: str, p: Principal = Depends(staff_only)):
    with A.need_db().begin() as c:
        u = _user_row(c, user_id)
        _guard(p, u, owner_only_change=False)
        n = c.execute(sa.delete(db.sessions).where(db.sessions.c.user_id == user_id)).rowcount
    return {"ended": n}


@router.delete("/users/{user_id}")
def delete_user(user_id: str, request: Request, p: Principal = Depends(staff_only)):
    with A.need_db().connect() as c:
        u = _user_row(c, user_id)
    _guard(p, u, owner_only_change=False)
    A.delete_user(user_id, request.app.state.worker)
    return {"ok": True}


# --- Работа на сървъра -------------------------------------------------------------------------------------------


@router.get("/jobs")
def recent_jobs(show: str = Query("active", pattern="^(active|failed|all)$"), p: Principal = Depends(staff_only)):
    """Последните задачи на всички: какво работи и чака, какво не успя."""
    out = []
    for s in jobs.summaries():
        if show == "active" and s["state"] not in jobs.ACTIVE | {"uploading"}:
            continue
        if show == "failed" and s["state"] != "failed":
            continue
        out.append(s)
        if len(out) >= 60:
            break
    ids = [s["id"] for s in out]
    with A.need_db().connect() as c:
        rows = c.execute(sa.select(db.scans.c.job_id, db.scans.c.title, db.users.c.id, db.users.c.name)
                         .join(db.users, db.users.c.id == db.scans.c.user_id)
                         .where(db.scans.c.job_id.in_(ids))).all() if ids else []  # fmt: skip
    who = {r.job_id: r for r in rows}
    result = []
    for s in out:
        r = who.get(s["id"])
        report = s.get("report") or {}
        full = jobs.read(s["id"]) or {}
        result.append({"id": s["id"], "state": s["state"], "step": s.get("step"), "error": s.get("error"),
                       "created": s.get("created"), "title": r.title if r else "",
                       "user": {"id": r.id, "name": r.name} if r else None,
                       "method": report.get("ai") or (full.get("params") or {}).get("method"),
                       "seconds": report.get("seconds")})  # fmt: skip
    return result


# --- Общност ------------------------------------------------------------------------------------------------------


@router.get("/reports")
def reports(p: Principal = Depends(staff_only)):
    return community.reported()


# --- Пощенска кутия (кодовете за потвърждение) ---------------------------------------------------------------------


@router.get("/mail")
def mail_status(p: Principal = Depends(staff_only)):
    """Пощата се задава само в .env на сървъра; тук се вижда дали е включена и се праща пробно писмо."""
    return mail.status()


@router.post("/mail/test")
def test_mail(p: Principal = Depends(staff_only)):
    """Пробно писмо до администратора: така се разбира, че кодовете ще стигат до хората."""
    if p.user_id is None:
        raise HTTPException(400, "Пробното писмо се изпраща до профил в приложението.")
    if not mail.configured():
        raise HTTPException(400, "Пощата още не е настроена.")
    to = A.user_email(p.user_id)
    try:
        mail.test(to)
    except Exception as e:  # noqa: BLE001 - грешна парола, мрежа, отказ на сървъра за поща
        raise HTTPException(502, f"Писмото не тръгна: {type(e).__name__}. Проверете адреса, порта и паролата.") from e
    return {"sent_to": to}
