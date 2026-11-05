"""Безплатен план и Pro (accounts.is_pro). AI моделите са за всички профили без ограничение. Безплатните профили
тестват за здравина FREE_LIMITS["stress"] модела и виждат в AR FREE_LIMITS["ar"] сканирания; Pro няма ограничения. Модел,
използван отново, се брои веднъж (db.plan_uses: профил, вид, задача). API ключът (администраторът; гостите на сървъри без
профили) няма ограничения. check() отказва
заявка рано, преди всякаква работа; record() я брои след успеха на работата и проверява отново под заключване, така
че две едновременни заявки да не вземат последното безплатно използване."""

import threading
import time

import sqlalchemy as sa
from fastapi import HTTPException

from . import db
from .accounts import Principal, is_pro, need_db

FREE_LIMITS = {"stress": 5, "ar": 3}
FULL = {
    "stress": "Безплатният план включва тест за здравина на 5 модела и те са използвани. С Pro тестовете са неограничени.",
    "ar": "Безплатният план включва AR за 3 сканирания и те са използвани. С Pro AR е неограничен.",
}
_lock = threading.Lock()


def _used(c, user_id: str, kind: str) -> list[str]:
    rows = c.execute(sa.select(db.plan_uses.c.ref).where(db.plan_uses.c.user_id == user_id, db.plan_uses.c.kind == kind)
                     .order_by(db.plan_uses.c.created))  # fmt: skip
    return [r.ref for r in rows]


def _limited(p: Principal) -> bool:
    return bool(p.user_id) and not p.admin and not is_pro(p)


def check(p: Principal, kind: str, ref: str):
    """Хвърля 402, когато безплатен профил е изчерпал ограничението си и `ref` не е сред използваните."""
    if not _limited(p):
        return
    with need_db().connect() as c:
        used = _used(c, p.user_id, kind)
    if ref not in used and len(used) >= FREE_LIMITS[kind]:
        raise HTTPException(402, FULL[kind])


def record(p: Principal, kind: str, ref: str):
    """Брои използването (веднъж за ref). Използванията на Pro също се броят като запис; ограничават се само
    безплатните профили."""
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
    """За профила: за всеки вид колко са използвани и какво е ограничението (None: без ограничение) и кои модели (само
    при безплатен: приложението показва дали модел е сред тях)."""
    with need_db().connect() as c:
        out = {}
        for kind, limit in FREE_LIMITS.items():
            used = _used(c, user_id, kind)
            out[kind] = {"used": len(used), "limit": None if pro else limit, "refs": [] if pro else used}
    return out
