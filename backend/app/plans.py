"""Free and Pro (accounts.is_pro). Free accounts test FREE_LIMITS["stress"] models for strength, see FREE_LIMITS["ar"]
scans in AR and try FREE_LIMITS["ai"] AI model; Pro has no limits. A model used again counts once (db.plan_uses:
account, kind, job). An AI model that failed does not count: the trial is for a model, not for an attempt.
The API token (the admin; guests on servers without accounts) has no limits.

check() turns a request away early, before any work; record() counts it once the work succeeded, and checks again
under a lock, so two requests at once cannot both take the last free use."""

import threading
import time

import sqlalchemy as sa
from fastapi import HTTPException

from . import db, jobs
from .accounts import Principal, is_pro, need_db

FREE_LIMITS = {"stress": 5, "ar": 3, "ai": 1}
FULL = {
    "stress": "Безплатният план включва тест за здравина на 5 модела и те са използвани. С Pro тестовете са неограничени.",
    "ar": "Безплатният план включва AR за 3 сканирания и те са използвани. С Pro AR е неограничен.",
    "ai": "Безплатният опит с AI модел е използван. С Pro можете да правите AI модели без ограничение.",
}
_lock = threading.Lock()


def _used(c, user_id: str, kind: str) -> list[str]:
    rows = c.execute(sa.select(db.plan_uses.c.ref).where(db.plan_uses.c.user_id == user_id, db.plan_uses.c.kind == kind)
                     .order_by(db.plan_uses.c.created))  # fmt: skip
    refs = [r.ref for r in rows]
    return [r for r in refs if (jobs.read(r) or {}).get("state") != "failed"] if kind == "ai" else refs


def _limited(p: Principal) -> bool:
    return bool(p.user_id) and not p.admin and not is_pro(p)


def check(p: Principal, kind: str, ref: str):
    """Raises 402 when a free account has used its limit and `ref` is not one of the uses."""
    if not _limited(p):
        return
    with need_db().connect() as c:
        used = _used(c, p.user_id, kind)
    if ref not in used and len(used) >= FREE_LIMITS[kind]:
        raise HTTPException(402, FULL[kind])


def record(p: Principal, kind: str, ref: str):
    """Counts the use (once per ref). Pro's uses are counted too, as a record; only free accounts are held to it."""
    if not p.user_id or p.admin:
        return
    limited = _limited(p)
    with _lock, need_db().begin() as c:
        used = _used(c, p.user_id, kind)
        if ref in used:
            return
        if limited and len(used) >= FREE_LIMITS[kind]:
            raise HTTPException(402, FULL[kind])
        c.execute(sa.insert(db.plan_uses).values(user_id=p.user_id, kind=kind, ref=ref, created=int(time.time())))


def summary(user_id: str, pro: bool) -> dict:
    """For the profile: per kind, how many used and the limit (None: unlimited), and which models (free only: the app
    shows whether a model is one of them)."""
    with need_db().connect() as c:
        out = {}
        for kind, limit in FREE_LIMITS.items():
            used = _used(c, user_id, kind)
            out[kind] = {"used": len(used), "limit": None if pro else limit, "refs": [] if pro else used}
    return out
