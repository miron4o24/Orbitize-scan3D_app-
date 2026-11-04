"""Известява телефона, че моделът е готов, чрез безплатната услуга за известия на Expo. Приложението регистрира ключа
си за известия, когато потребителят ги разреши (PUT /me/push-token); ключът принадлежи на профила, който последно е
влязъл на този телефон. Проблем тук никога не засяга задачата: моделът е готов, независимо дали телефонът е
разбрал."""

import logging
import re
import time

import requests
import sqlalchemy as sa

from . import db

log = logging.getLogger("uvicorn.error")

EXPO_PUSH = "https://exp.host/--/api/v2/push/send"
TOKEN_RE = re.compile(r"^Expo(nent)?PushToken\[[^\]\s]{1,180}\]$")
CHANNEL = "scans"  # каналът за Android на приложението (lib/push.ts)


def save(user_id: str, token: str):
    with db.engine().begin() as c:
        c.execute(sa.delete(db.push_tokens).where(db.push_tokens.c.token == token))
        c.execute(sa.insert(db.push_tokens).values(token=token, user_id=user_id, created=int(time.time())))


def forget(token: str, user_id: str | None = None):
    q = sa.delete(db.push_tokens).where(db.push_tokens.c.token == token)
    if user_id:
        q = q.where(db.push_tokens.c.user_id == user_id)
    with db.engine().begin() as c:
        c.execute(q)


def send(user_id: str, title: str, body: str, data: dict):
    e = db.engine()
    if e is None:
        return
    with e.connect() as c:
        tokens = [r.token for r in c.execute(sa.select(db.push_tokens.c.token).where(db.push_tokens.c.user_id == user_id))]
    if not tokens:
        return
    messages = [{"to": t, "title": title, "body": body, "data": data, "sound": "default", "channelId": CHANNEL}
                for t in tokens]  # fmt: skip
    answer = requests.post(EXPO_PUSH, json=messages, timeout=15)
    answer.raise_for_status()
    for token, ticket in zip(tokens, answer.json().get("data") or []):
        if (ticket.get("details") or {}).get("error") == "DeviceNotRegistered":  # приложението е премахнато от този телефон
            forget(token)


def job_finished(job_id: str, status: dict):
    """Задача е завършила: успешно или неуспешно. Задачите без профил (тестовете на администратора) не известяват
    никого."""
    owner = status.get("owner")
    if not owner:
        return
    try:
        with db.engine().connect() as c:
            title = c.execute(sa.select(db.scans.c.title).where(db.scans.c.job_id == job_id)).scalar() or ""
        name = f"„{title}“" if title else "Моделът ви"
        if status.get("state") == "done":
            send(owner, "Моделът е готов", f"{name} може да се отвори.", {"jobId": job_id})
        else:
            send(owner, "Моделът не успя", "Отворете приложението, за да опитате отново.", {"jobId": job_id})
    except Exception as e:
        log.warning("push for %s: %s", job_id, e)
