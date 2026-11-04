"""Telling the phone that a model is ready, through Expo's free push service. The app registers its push token when the
user allows notifications (PUT /me/push-token); a token belongs to the account last signed in on that phone. Trouble
here never touches the job: a model is ready whether or not the phone heard of it."""

import logging
import re
import time

import requests
import sqlalchemy as sa

from . import db

log = logging.getLogger("uvicorn.error")

EXPO_PUSH = "https://exp.host/--/api/v2/push/send"
TOKEN_RE = re.compile(r"^Expo(nent)?PushToken\[[^\]\s]{1,180}\]$")
CHANNEL = "scans"  # the app's Android channel (lib/push.ts)


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
        if (ticket.get("details") or {}).get("error") == "DeviceNotRegistered":  # the app was removed from that phone
            forget(token)


def job_finished(job_id: str, status: dict):
    """A job came to its end: done or failed. Jobs without an account (the admin's tests) tell no one."""
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
