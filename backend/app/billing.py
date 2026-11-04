"""Pro като абонамент в Google Play. Телефонът купува; сървърът пита Google дали покупката е истинска и докога важи и
го пази в users.pro_until. Нищо, казано от телефона за покупка, не се приема на доверие. Изключено, докато сървърът
няма файл с ключ на служебен профил в Google: GOOGLE_PLAY_KEY_FILE в .env (никога в кода или приложението).
Подновявания: когато pro_until е минало, следващото четене на профила пита Google отново (най-много веднъж на
RECHECK_S)."""

import hashlib
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import db
from .accounts import Principal, need_db, profile, signed_in

log = logging.getLogger("uvicorn.error")
router = APIRouter()

PACKAGE = os.environ.get("ANDROID_PACKAGE", "com.miron4o.scan3d")
PRODUCT = os.environ.get("PRO_PRODUCT_ID", "orbitize_pro_monthly")
API = "https://androidpublisher.googleapis.com/androidpublisher/v3/applications"
SCOPE = "https://www.googleapis.com/auth/androidpublisher"
# CANCELED: подновяването е изключено, но абонаментът важи до края си.
VALID = {"SUBSCRIPTION_STATE_ACTIVE", "SUBSCRIPTION_STATE_IN_GRACE_PERIOD", "SUBSCRIPTION_STATE_CANCELED"}
RECHECK_S = 3600

_checked: dict[str, float] = {}
_lock = threading.Lock()


def key_file() -> Path | None:
    path = os.environ.get("GOOGLE_PLAY_KEY_FILE", "")
    return Path(path) if path and Path(path).exists() else None


def configured() -> bool:
    return key_file() is not None


def access_token() -> str:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(str(key_file()), scopes=[SCOPE])
    creds.refresh(Request())
    return creds.token


def expires_at(answer: dict) -> int:
    """Докога според Google важи абонаментът за Pro в отговора; 0, ако не важи."""
    if answer.get("subscriptionState") not in VALID:
        return 0
    best = 0
    for item in answer.get("lineItems") or []:
        if item.get("productId") == PRODUCT and item.get("expiryTime"):
            when = datetime.fromisoformat(item["expiryTime"].replace("Z", "+00:00"))
            best = max(best, int(when.timestamp()))
    return best


def ask_google(token: str) -> int:
    answer = requests.get(f"{API}/{PACKAGE}/purchases/subscriptionsv2/tokens/{token}", timeout=20,
                          headers={"Authorization": f"Bearer {access_token()}"})  # fmt: skip
    if answer.status_code in (400, 404, 410):
        return 0
    answer.raise_for_status()
    return expires_at(answer.json())


def refresh(user_id: str):
    """Изтекъл абонамент може да е подновен: питаме Google отново. Без съобщение при какъвто и да е проблем; Pro
    просто остава изключен до следващия опит."""
    if not configured():
        return
    now = time.time()
    with _lock:
        if now - _checked.get(user_id, 0) < RECHECK_S:
            return
        _checked[user_id] = now
    try:
        with need_db().connect() as c:
            u = c.execute(sa.select(db.users.c.pro_until, db.users.c.pro_token).where(db.users.c.id == user_id)).first()
        if u is None or not u.pro_token or (u.pro_until or 0) > now:
            return
        until = ask_google(u.pro_token)
        with need_db().begin() as c:
            c.execute(sa.update(db.users).where(db.users.c.id == user_id).values(pro_until=until))
    except Exception as e:
        log.warning("billing: renewal check for %s: %s", user_id, e)


class Purchase(BaseModel):
    purchase_token: str = Field(min_length=10, max_length=512, pattern=r"^[A-Za-z0-9._:\-]+$")


@router.post("/billing/google")
def confirm(body: Purchase, p: Principal = Depends(signed_in)):
    """Телефонът е купил Pro: след проверка при Google този профил е с Pro до датата, дадена от Google. Една покупка
    важи за един профил."""
    if not configured():
        raise HTTPException(503, "Плащанията още не са включени.")
    token_id = hashlib.sha256(body.purchase_token.encode()).hexdigest()
    with need_db().connect() as c:
        owner = c.execute(sa.select(db.purchases.c.user_id).where(db.purchases.c.token_hash == token_id)).scalar()
    if owner and owner != p.user_id:
        raise HTTPException(409, "Тази покупка вече е към друг профил.")
    try:
        until = ask_google(body.purchase_token)
    except Exception as e:
        log.warning("billing: confirm: %s", e)
        raise HTTPException(503, "Покупката не можа да се провери. Опитайте след малко.")
    if until <= time.time():
        raise HTTPException(402, "Google не потвърди покупката.")
    with need_db().begin() as c:
        if not owner:
            c.execute(sa.insert(db.purchases).values(token_hash=token_id, user_id=p.user_id, created=int(time.time())))
        c.execute(sa.update(db.users).where(db.users.c.id == p.user_id)
                  .values(pro_until=until, pro_token=body.purchase_token))  # fmt: skip
    return profile(p.user_id)
