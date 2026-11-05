"""Профили: вход с имейл и парола (или Google), профили, сканирания по профил и публични линкове за споделяне
(таблиците са в app/db.py). Приложението влиза с имейл и парола (POST /auth/register, /auth/login) и получава
собствен сесиен ключ от сървъра, който оттам нататък изпраща като "Authorization: Bearer <token>". Входът с Google
(/auth/google) остава за по-късно. Споделеният API_TOKEN продължава да работи и вижда всяка задача
(администраторският ключ)."""

import hashlib
import hmac
import html
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from . import ai as AI
from . import db, jobs, notify

API_TOKEN = os.environ.get("API_TOKEN", "")
# Собственикът на приложението: включва и изключва Pro за потребители от страница в приложението (докато Pro не стане
# платен план). Влиза с един от тези имейли (разделени със запетая), а не с API ключа.
OWNER_EMAILS = os.environ.get("OWNER_EMAILS", "")
# Клиентът за Google OAuth, с който влиза приложението (тип "Web application": аудиторията на ID токена). Разделени
# със запетая. Може да се зададе и без достъп до сървъра: PUT /admin/settings с API ключа (пази се в базата данни).
GOOGLE_SETTING = "google_client_id"
MAX_SCANS_PER_USER = int(os.environ.get("MAX_SCANS_PER_USER", "300"))
MAX_UPLOADING_PER_USER = 3  # започнати сканирания, които още не са изпратени за обработка
MAX_AVATAR_BYTES = 2 * 1024 * 1024
AVATARS = Path(os.environ.get("AVATAR_DIR", jobs.DATA.parent / "avatars")).resolve()
SESSION_TOUCH_S = 3600  # last_used се записва най-често през този интервал

router = APIRouter()


@dataclass(frozen=True)
class Principal:
    user_id: str | None  # None: споделеният API ключ
    admin: bool = False


def hashed(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def need_db() -> sa.Engine:
    e = db.engine()
    if e is None:
        raise HTTPException(503, "Профилите още не са включени." if not db.configured() else
                            "Профилите временно не са достъпни. Опитайте след малко.")  # fmt: skip
    return e


def principal(authorization: str = Header("")) -> Principal:
    token = authorization.removeprefix("Bearer ").strip()
    if API_TOKEN and hmac.compare_digest(authorization.encode(), f"Bearer {API_TOKEN}".encode()):
        return Principal(None, admin=True)
    if token and db.configured():
        e = db.engine()
        if e is None:  # не "не сте влезли": приложението би извело потребителя от профила
            raise HTTPException(503, "Профилите временно не са достъпни. Опитайте след малко.")
        now = int(time.time())
        with e.begin() as c:
            row = c.execute(sa.select(db.sessions).where(db.sessions.c.token_hash == hashed(token))).first()
            if row:
                if now - row.last_used > SESSION_TOUCH_S:
                    c.execute(sa.update(db.sessions).where(db.sessions.c.token_hash == row.token_hash)
                              .values(last_used=now))  # fmt: skip
                return Principal(row.user_id)
    raise HTTPException(401, "Влезте отново в профила си.")


def admin_only(p: Principal = Depends(principal)) -> Principal:
    if not p.admin:
        raise HTTPException(403, "Само за администратора.")
    return p


def signed_in(p: Principal = Depends(principal)) -> Principal:
    if not p.user_id:
        raise HTTPException(403, "Първо влезте в профила си.")
    return p


def is_owner_email(email: str) -> bool:
    return email.strip().lower() in {e.strip().lower() for e in OWNER_EMAILS.split(",") if e.strip()}


def user_email_pro(user_id: str) -> tuple[str, bool]:
    with need_db().connect() as c:
        u = c.execute(sa.select(db.users.c.email, db.users.c.pro, db.users.c.pro_until).where(db.users.c.id == user_id)).first()
    if u is None:
        raise HTTPException(401, "Този профил вече не съществува.")
    return u.email, bool(u.pro) or (u.pro_until or 0) > time.time()


def is_pro(p: Principal) -> bool:
    """API ключът (администраторът) се брои за Pro, а собственикът винаги е с Pro."""
    if p.admin:
        return True
    if not p.user_id:
        return False
    email, pro = user_email_pro(p.user_id)
    return pro or is_owner_email(email)


def owner_only(p: Principal = Depends(principal)) -> Principal:
    if not p.admin and not (p.user_id and is_owner_email(user_email_pro(p.user_id)[0])):
        raise HTTPException(403, "Само за собственика на приложението.")
    return p


def can_see(p: Principal, status: dict) -> bool:
    return p.admin or (p.user_id is not None and status.get("owner") == p.user_id)


# --- Вход с Google --------------------------------------------------------------------------------------------------


def google_client_ids() -> list[str]:
    ids = os.environ.get("GOOGLE_CLIENT_ID", "") + "," + (db.get_setting(GOOGLE_SETTING) or "")
    return [i.strip() for i in ids.split(",") if i.strip()]


def verify_google(id_token: str, audiences: list[str]) -> dict:
    """Данните от токена, когато Google го е подписал за някой от нашите клиенти; иначе хвърля ValueError."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    claims = google_id_token.verify_oauth2_token(id_token, google_requests.Request(), clock_skew_in_seconds=10)
    if claims.get("aud") not in audiences:
        raise ValueError("token is for another app")
    if not claims.get("email_verified"):
        raise ValueError("the Google account's email is not verified")
    return claims


class GoogleSignIn(BaseModel):
    id_token: str = Field(max_length=8192)


@router.post("/auth/google")
def sign_in_google(body: GoogleSignIn):
    e = need_db()
    audiences = google_client_ids()
    if not audiences:
        raise HTTPException(503, "Вход с Google още не е включен.")
    try:
        claims = verify_google(body.id_token, audiences)
    except ValueError:  # причината (изтекъл, грешна аудитория) не е нещо, което потребителят може да поправи
        raise HTTPException(401, "Входът с Google не успя. Опитайте отново.")
    now = int(time.time())
    token = secrets.token_urlsafe(32)
    with e.begin() as c:
        user = c.execute(sa.select(db.users).where(db.users.c.google_sub == claims["sub"])).first()
        if user is None:  # профил с имейл и парола с този адрес, потвърден от Google: същият човек
            user = c.execute(sa.select(db.users).where(db.users.c.google_sub == email_key(claims["email"]))).first()
        if user is None:
            user_id = secrets.token_hex(16)
            name = (claims.get("name") or claims["email"].split("@")[0])[:100]
            c.execute(sa.insert(db.users).values(id=user_id, google_sub=claims["sub"], email=claims["email"],
                                                 name=name, google_picture=claims.get("picture"), avatar=False,
                                                 created=now))  # fmt: skip
        else:
            user_id = user.id
            c.execute(sa.update(db.users).where(db.users.c.id == user_id)
                      .values(email=claims["email"], google_picture=claims.get("picture")))  # fmt: skip
        c.execute(sa.insert(db.sessions).values(token_hash=hashed(token), user_id=user_id, created=now, last_used=now))
    return {"token": token, "user": profile(user_id)}


# --- Имейл и парола --- Без проверка на имейла (избор на потребителя): профилът е имейл и парола. Паролите се пазят
# като хешове със scrypt.

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_MIN = 6
MAX_FAILED = 10  # грешни пароли за имейл в рамките на FAILED_WINDOW_S, след което има пауза
FAILED_WINDOW_S = 15 * 60
_failed: dict[str, list[float]] = {}


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def password_matches(password: str, stored: str | None) -> bool:
    try:
        kind, salt, h = (stored or "").split("$")
    except ValueError:
        return False
    test = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1, dklen=32)
    return kind == "scrypt" and hmac.compare_digest(test.hex(), h)


def email_key(email: str) -> str:
    """Ключът на профила в users.google_sub (уникален), така че имейли, различаващи се само по регистъра на буквите,
    са един профил."""
    return "email:" + email.strip().lower()


def new_session(c, user_id: str) -> str:
    token, now = secrets.token_urlsafe(32), int(time.time())
    c.execute(sa.insert(db.sessions).values(token_hash=hashed(token), user_id=user_id, created=now, last_used=now))
    return token


class Register(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=200)
    name: str = Field("", max_length=100)


class Login(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=200)


@router.post("/auth/register")
def register(body: Register):
    email = body.email.strip()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Въведете валиден имейл.")
    if len(body.password) < PASSWORD_MIN:
        raise HTTPException(400, f"Паролата трябва да е поне {PASSWORD_MIN} знака.")
    name = " ".join(body.name.split()) or email.split("@")[0]
    with need_db().begin() as c:
        if c.execute(sa.select(db.users.c.id).where(db.users.c.google_sub == email_key(email))).first():
            raise HTTPException(409, "Вече има профил с този имейл. Влезте в него.")
        if c.execute(sa.select(db.users.c.id).where(sa.func.lower(db.users.c.email) == email.lower())).first():
            raise HTTPException(409, "Този имейл има профил през Google. Използвайте „Продължи с Google“.")
        user_id = secrets.token_hex(16)
        c.execute(sa.insert(db.users).values(id=user_id, google_sub=email_key(email), email=email, name=name[:100],
                                             avatar=False, created=int(time.time()),
                                             password_hash=hash_password(body.password)))  # fmt: skip
        token = new_session(c, user_id)
    return {"token": token, "user": profile(user_id)}


@router.post("/auth/login")
def login(body: Login):
    key = email_key(body.email)
    now = time.time()
    recent = [t for t in _failed.get(key, []) if now - t < FAILED_WINDOW_S]
    if len(recent) >= MAX_FAILED:
        raise HTTPException(429, "Твърде много грешни пароли. Изчакайте 15 минути и опитайте пак.")
    with need_db().begin() as c:
        user = c.execute(sa.select(db.users).where(db.users.c.google_sub == key)).first()
        if user is None or not password_matches(body.password, user.password_hash):
            _failed[key] = recent + [now]
            raise HTTPException(401, "Грешен имейл или парола.")
        _failed.pop(key, None)
        token = new_session(c, user.id)
    return {"token": token, "user": profile(user.id)}


@router.post("/auth/logout")
def sign_out(authorization: str = Header(""), p: Principal = Depends(signed_in)):
    token = authorization.removeprefix("Bearer ").strip()
    with need_db().begin() as c:
        c.execute(sa.delete(db.sessions).where(db.sessions.c.token_hash == hashed(token)))
    return {"ok": True}


# --- Профил ---------------------------------------------------------------------------------------------------------


def avatar_path(user_id: str) -> Path:
    return AVATARS / f"{user_id}.jpg"


def owned_jobs(user_id: str) -> list[sa.Row]:
    with need_db().connect() as c:
        rows = c.execute(sa.select(db.scans).where(db.scans.c.user_id == user_id)
                         .order_by(db.scans.c.created.desc())).all()  # fmt: skip
    return rows


def profile(user_id: str) -> dict:
    from . import billing, plans  # те импортират този модул

    billing.refresh(user_id)  # изтекъл абонамент може да е подновен
    with need_db().connect() as c:
        u = c.execute(sa.select(db.users).where(db.users.c.id == user_id)).first()
    if u is None:
        raise HTTPException(401, "Този профил вече не съществува.")
    rows = owned_jobs(user_id)
    statuses = [jobs.read(r.job_id) or {} for r in rows]
    owner = is_owner_email(u.email)
    paid = (u.pro_until or 0) > time.time()
    pro = owner or bool(u.pro) or paid
    done = [s for s in statuses if s.get("state") == "done"]
    return {
        "id": u.id,
        "email": u.email,
        "name": u.name,
        "google_picture": u.google_picture,
        "has_avatar": bool(u.avatar),
        "created": u.created,
        # pro: може да създава AI модели (app/ai.py), "ai" казва за кои сървърът има ключове; owner: страницата за
        # Pro.
        "pro": pro,
        "pro_until": u.pro_until if paid else None,  # чрез плащане: до кога (подновява се сам, ако не е отменен)
        "billing": billing.configured(),  # Pro може да се купи в приложението
        "pro_product": billing.PRODUCT,
        "owner": owner,
        "ai": AI.available(),
        "limits": plans.summary(user_id, pro),  # тестовете за здравина, AR и пробният AI модел на безплатния план (app/plans.py)
        "stats": {
            "scans": sum(1 for s in statuses if s),
            "done": len(done),
            "seconds": sum((s.get("report") or {}).get("seconds") or 0 for s in done),
            "shared": sum(1 for r in rows if r.share),
            "published": sum(1 for r in rows if r.published),
        },
    }


@router.get("/me")
def get_me(p: Principal = Depends(signed_in)):
    return profile(p.user_id)


class ProfilePatch(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class PushToken(BaseModel):
    token: str = Field(min_length=10, max_length=200)


@router.put("/me/push-token")
def put_push_token(body: PushToken, p: Principal = Depends(signed_in)):
    """Този телефон иска да знае кога са готови моделите на профила (app/notify.py)."""
    if not notify.TOKEN_RE.match(body.token):
        raise HTTPException(400, "Неправилен адрес за известия.")
    need_db()
    notify.save(p.user_id, body.token)
    return {"ok": True}


@router.post("/me/push-token/forget")
def forget_push_token(body: PushToken, p: Principal = Depends(signed_in)):
    """Изход на този телефон: спира да получава известия за този профил."""
    need_db()
    notify.forget(body.token, p.user_id)
    return {"ok": True}


@router.patch("/me")
def patch_me(body: ProfilePatch, p: Principal = Depends(signed_in)):
    name = " ".join(body.name.split())
    if not name:
        raise HTTPException(400, "Името не може да е празно.")
    with need_db().begin() as c:
        c.execute(sa.update(db.users).where(db.users.c.id == p.user_id).values(name=name))
    return profile(p.user_id)


@router.put("/me/avatar")
async def put_avatar(request: Request, p: Principal = Depends(signed_in)):
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_AVATAR_BYTES:
            raise HTTPException(413, "Снимката е твърде голяма (до 2 MB).")
    if body[:3] != b"\xff\xd8\xff":
        raise HTTPException(400, "Файлът не е снимка (JPEG).")
    AVATARS.mkdir(parents=True, exist_ok=True)
    tmp = avatar_path(p.user_id).with_suffix(".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, avatar_path(p.user_id))
    with need_db().begin() as c:
        c.execute(sa.update(db.users).where(db.users.c.id == p.user_id).values(avatar=True))
    return profile(p.user_id)


@router.delete("/me/avatar")
def delete_avatar(p: Principal = Depends(signed_in)):
    avatar_path(p.user_id).unlink(missing_ok=True)
    with need_db().begin() as c:
        c.execute(sa.update(db.users).where(db.users.c.id == p.user_id).values(avatar=False))
    return profile(p.user_id)


@router.get("/me/avatar.jpg")
def get_avatar(p: Principal = Depends(signed_in)):
    path = avatar_path(p.user_id)
    if not path.exists():
        raise HTTPException(404, "Няма собствена снимка.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})


def delete_job_files(job_id: str, worker):
    worker.cancel(job_id)
    shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)


@router.delete("/me")
def delete_me(request: Request, p: Principal = Depends(signed_in)):
    """Изтрива профила с всичките му сканирания завинаги (магазините за приложения изискват това да е възможно в
    приложението)."""
    for r in owned_jobs(p.user_id):
        delete_job_files(r.job_id, request.app.state.worker)
    avatar_path(p.user_id).unlink(missing_ok=True)
    with need_db().begin() as c:
        mine = sa.select(db.scans.c.job_id).where(db.scans.c.user_id == p.user_id)
        for t in (db.likes, db.reports):  # върху публикациите на профила (за бази данни без каскадно изтриване)
            c.execute(sa.delete(t).where(t.c.job_id.in_(mine)))
        for t in (db.likes, db.reports, db.scans, db.sessions, db.xp_events):
            c.execute(sa.delete(t).where(t.c.user_id == p.user_id))
        c.execute(sa.delete(db.users).where(db.users.c.id == p.user_id))
    return {"ok": True}


# --- Сканирания на профил -------------------------------------------------------------------------------------------


def scan_entry(row: sa.Row, status: dict) -> dict:
    d = jobs.job_dir(row.job_id)
    return {k: status.get(k) for k in ("state", "error", "report", "step")} | {
        "id": row.job_id,
        "title": row.title,
        "created": row.created,
        "share": row.share,
        "published": row.published,
        "thumb": (d / jobs.THUMB).exists() or any((d / "photos").glob("*.jp*g")),
    }


@router.get("/me/scans")
def my_scans(p: Principal = Depends(signed_in)):
    out, gone = [], []
    for r in owned_jobs(p.user_id):
        s = jobs.read(r.job_id)
        if s is None:  # файловете му са премахнати (изоставено качване е изтекло)
            gone.append(r.job_id)
        else:
            out.append(scan_entry(r, s))
    if gone:
        with need_db().begin() as c:
            for t in (db.likes, db.reports, db.scans):
                c.execute(sa.delete(t).where(t.c.job_id.in_(gone)))
    return out


def add_scan(job_id: str, user_id: str, title: str = ""):
    """Вика се, когато профил създава задача; отказва над ограниченията на профила."""
    rows = owned_jobs(user_id)
    if len(rows) >= MAX_SCANS_PER_USER:
        raise HTTPException(400, f"Профилът ви има {len(rows)} сканирания, колкото най-много побира. Изтрийте някои първо.")
    uploading = sum(1 for r in rows if (jobs.read(r.job_id) or {}).get("state") == "uploading")
    if uploading >= MAX_UPLOADING_PER_USER:
        raise HTTPException(429, "Няколко сканирания още се качват. Довършете или изтрийте тях първо.")
    with need_db().begin() as c:
        c.execute(sa.insert(db.scans).values(job_id=job_id, user_id=user_id, title=title[:100], created=int(time.time())))


def forget_scan(job_id: str):
    if db.engine() is not None:
        with db.engine().begin() as c:
            for t in (db.likes, db.reports, db.scans):
                c.execute(sa.delete(t).where(t.c.job_id == job_id))


def owned_row(job_id: str, user_id: str) -> sa.Row:
    with need_db().connect() as c:
        row = c.execute(sa.select(db.scans).where(db.scans.c.job_id == job_id, db.scans.c.user_id == user_id)).first()
    if row is None or jobs.read(job_id) is None:
        raise HTTPException(404, "Сканирането не е намерено.")
    return row


class ScanPatch(BaseModel):
    title: str = Field(max_length=100)


@router.patch("/me/scans/{job_id}")
def rename_scan(job_id: str, body: ScanPatch, p: Principal = Depends(signed_in)):
    owned_row(job_id, p.user_id)
    with need_db().begin() as c:
        c.execute(sa.update(db.scans).where(db.scans.c.job_id == job_id).values(title=" ".join(body.title.split())))
    return scan_entry(owned_row(job_id, p.user_id), jobs.read(job_id))


class Claim(BaseModel):
    job_ids: list[str] = Field(max_length=200)


@router.post("/me/scans/claim")
def claim_scans(body: Claim, p: Principal = Depends(signed_in)):
    """Сканиранията, направени на този телефон преди влизане (със споделения ключ), се присъединяват към профила. Само
    задачи без собственик; номерата на задачите са случайни 128-битови стойности, известни само на телефона, който ги
    е създал."""
    claimed = []
    for job_id in dict.fromkeys(body.job_ids):
        s = jobs.read(job_id) if len(job_id) == 32 and job_id.isalnum() else None
        if s is None or s.get("owner"):
            continue
        jobs.write(job_id, owner=p.user_id)
        with need_db().begin() as c:
            if not c.execute(sa.select(db.scans.c.job_id).where(db.scans.c.job_id == job_id)).first():
                c.execute(sa.insert(db.scans).values(job_id=job_id, user_id=p.user_id, title="",
                                                     created=int(s.get("created") or time.time())))  # fmt: skip
        claimed.append(job_id)
    return {"claimed": claimed}


# --- Публични линкове -----------------------------------------------------------------------------------------------


@router.post("/me/scans/{job_id}/share")
def share_scan(job_id: str, p: Principal = Depends(signed_in)):
    row = owned_row(job_id, p.user_id)
    if jobs.read(job_id).get("state") != "done":
        raise HTTPException(409, "Може да се сподели само готов модел.")
    if row.share:
        return {"share": row.share}
    slug = secrets.token_urlsafe(9)  # 72 случайни бита: линковете не могат да се познаят
    with need_db().begin() as c:
        c.execute(sa.update(db.scans).where(db.scans.c.job_id == job_id).values(share=slug))
    return {"share": slug}


@router.delete("/me/scans/{job_id}/share")
def unshare_scan(job_id: str, p: Principal = Depends(signed_in)):
    owned_row(job_id, p.user_id)
    with need_db().begin() as c:
        c.execute(sa.update(db.scans).where(db.scans.c.job_id == job_id).values(share=None))
    return {"share": None}


def shared(slug: str) -> sa.Row:
    e = db.engine() if len(slug) <= 32 else None
    row = None
    if e is not None:
        with e.connect() as c:
            row = c.execute(sa.select(db.scans, db.users.c.name.label("owner_name"))
                            .join(db.users, db.users.c.id == db.scans.c.user_id)
                            .where(db.scans.c.share == slug)).first()  # fmt: skip
    if row is None or (jobs.read(row.job_id) or {}).get("state") != "done":
        raise HTTPException(404, "Този линк вече не е споделен.")
    return row


SHARE_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Orbitize</title>
<meta property="og:title" content="{title}">
<meta property="og:description" content="A 3D scan by {owner}. Open it to turn it around.">
<meta property="og:image" content="{base}/s/{slug}/thumb.jpg">
<style>
  :root {{ color-scheme: light; --bg: #f5f7fb; --card: #fff; --text: #0f172a; --muted: #64748b; --accent: #2563eb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 16px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg);
         color: var(--text); }}
  main {{ max-width: 960px; margin: 0 auto; padding: 16px; }}
  .view {{ position: relative; height: min(70vh, 640px); background: var(--card); border-radius: 20px;
          box-shadow: 0 1px 3px rgba(15,23,42,.08), 0 8px 24px rgba(15,23,42,.06); overflow: hidden; }}
  canvas {{ display: block; width: 100%; height: 100%; touch-action: none; }}
  #status {{ position: absolute; inset: 0; display: grid; place-items: center; color: var(--muted); }}
  h1 {{ font-size: 22px; margin: 18px 4px 4px; }}
  p {{ margin: 4px; color: var(--muted); }}
  .hint {{ font-size: 14px; }}
</style></head>
<body><main>
  <div class="view"><canvas id="c"></canvas><div id="status">Loading the model…</div></div>
  <h1>{title}</h1>
  <p>Scanned by {owner} · {size}</p>
  <p class="hint">Drag to turn it, pinch or scroll to zoom.</p>
</main>
<script type="importmap">
{{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js",
              "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const canvas = document.getElementById('c'), status = document.getElementById('status');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color('#ffffff');
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
scene.add(new THREE.HemisphereLight(0xffffff, 0xdde3ee, 2.2));
const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(3, 5, 4); scene.add(key);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
function resize() {{
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}}
addEventListener('resize', resize); resize();
new GLTFLoader().load('/s/{slug}/model.glb', (gltf) => {{
  const model = gltf.scene, box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
  model.position.sub(centre); scene.add(model);
  const r = size.length() / 2;
  camera.position.set(r * 1.4, r * 0.9, r * 1.9); camera.near = r / 100; camera.far = r * 20;
  camera.updateProjectionMatrix(); controls.update();
  status.remove();
}}, undefined, () => {{ status.textContent = 'The model could not be loaded.'; }});
renderer.setAnimationLoop(() => {{ controls.update(); renderer.render(scene, camera); }});
</script></body></html>"""


def base_url(request: Request) -> str:
    """Адресът, използван от посетителя. Зад Tailscale Funnel заявката пристига по обикновен HTTP на localhost, но
    публичният адрес е HTTPS."""
    if pub := os.environ.get("PUBLIC_URL"):
        return pub.rstrip("/")
    host = request.headers.get("host", "")
    if host.endswith(".ts.net"):
        return f"https://{host}"
    return str(request.base_url).rstrip("/")


@router.get("/s/{slug}")
def share_page(slug: str, request: Request):
    row = shared(slug)
    report = (jobs.read(row.job_id) or {}).get("report") or {}
    dims = report.get("dimensions_mm")
    size = "proportions " + " : ".join(f"{d / max(dims):.2f}" for d in dims) if dims else ""
    page = SHARE_PAGE.format(
        title=html.escape(row.title or "3D scan"),
        owner=html.escape(row.owner_name),
        size=html.escape(size),
        slug=html.escape(slug),
        base=html.escape(base_url(request)),
    )
    return Response(page, media_type="text/html; charset=utf-8")


@router.get("/s/{slug}/model.glb")
def share_model(slug: str):
    from . import mesh as M
    from . import texture as TX
    import trimesh

    row = shared(slug)
    job = jobs.job_dir(row.job_id)
    mesh = trimesh.load(job / "model.ply", force="mesh")
    return Response(M.export(mesh, "glb", None, TX.load(job)), media_type=M.MEDIA_TYPES["glb"],
                    headers={"Cache-Control": "public, max-age=300"})  # fmt: skip


@router.get("/s/{slug}/thumb.jpg")
def share_thumb(slug: str):
    path = jobs.thumbnail(shared(slug).job_id)
    if path is None:
        raise HTTPException(404, "Няма снимка.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=300"})


# --- Настройки на сървъра (администратор)
# ----------------------------------------------------------------------------


class Settings(BaseModel):
    google_client_id: str | None = Field(None, max_length=1000)


@router.get("/admin/settings", dependencies=[Depends(admin_only)])
def get_settings():
    return {"database": db.status(), "google_client_id": db.get_setting(GOOGLE_SETTING) if db.engine() else None,
            "google_client_id_env": os.environ.get("GOOGLE_CLIENT_ID") or None}  # fmt: skip


@router.put("/admin/settings", dependencies=[Depends(admin_only)])
def put_settings(body: Settings):
    need_db()
    if body.google_client_id is not None:
        db.set_setting(GOOGLE_SETTING, body.google_client_id.strip())
    return get_settings()


# --- Страницата на собственика за Pro (екран "Pro" в приложението)
# ---------------------------------------------------


@router.get("/owner/users", dependencies=[Depends(owner_only)])
def owner_users(q: str = Query("", max_length=100), limit: int = Query(100, ge=1, le=500)):
    """Първо най-новите профили; q: част от името или имейла."""
    sel = sa.select(db.users.c.id, db.users.c.email, db.users.c.name, db.users.c.created, db.users.c.pro)
    if q.strip():
        like = f"%{q.strip().lower()}%"
        sel = sel.where(sa.or_(sa.func.lower(db.users.c.email).like(like), sa.func.lower(db.users.c.name).like(like)))
    with need_db().connect() as c:
        rows = c.execute(sel.order_by(db.users.c.created.desc()).limit(limit)).all()
    return [{"id": r.id, "email": r.email, "name": r.name, "created": r.created, "pro": bool(r.pro),
             "owner": is_owner_email(r.email)} for r in rows]  # fmt: skip


class ProSwitch(BaseModel):
    pro: bool


@router.put("/owner/users/{user_id}/pro", dependencies=[Depends(owner_only)])
def owner_set_pro(user_id: str, body: ProSwitch):
    with need_db().begin() as c:
        if not c.execute(sa.update(db.users).where(db.users.c.id == user_id).values(pro=body.pro)).rowcount:
            raise HTTPException(404, "Профилът не е намерен.")
    return {"id": user_id, "pro": body.pro}


# Ключовете на AI услугите са на приложението, а не на потребителите: задават се само с API ключа (никога от
# приложението), пазят се в таблицата с настройки на базата данни и никога не се връщат. ANTHROPIC_API_KEY /
# MESHY_API_KEY от средата са с предимство пред тях.
class AiKeys(BaseModel):
    """Ключ за запазване; "" премахва запазения (незададено поле го оставя както е)."""

    anthropic: str | None = Field(None, max_length=500)
    meshy: str | None = Field(None, max_length=500)


@router.get("/admin/ai", dependencies=[Depends(admin_only)])
def admin_ai():
    return AI.key_status()


@router.put("/admin/ai", dependencies=[Depends(admin_only)])
def admin_set_ai(body: AiKeys):
    need_db()
    for service in ("anthropic", "meshy"):
        value = getattr(body, service)
        if value is not None:
            AI.set_key(service, value.strip())
    return AI.key_status()


def health_info() -> dict:
    """Идентификаторът на уеб клиента е публичен (има го във всяка компилация с вход); приложението го чете оттук,
    затова създаването на клиента в Google не изисква нова компилация."""
    ok = db.engine() is not None
    ids = google_client_ids() if ok else []
    # accounts: входът с имейл и парола работи (нужна е само базата данни). version: коя версия работи на сървъра
    # (проверява се отдалечено след инсталиране). stress, progress, pro: тестът за здравина (app/stress.py), точките
    # (app/progress.py) и AI моделите на Pro (app/ai.py) са налични, затова приложението ги показва.
    return {"accounts": ok, "database": db.status(), "google_client_id": ids[0] if ids else None,
            "version": "2026-11-05-detail", "stress": True, "progress": True, "pro": True}  # fmt: skip
