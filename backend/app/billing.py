"""Pro as a Google Play subscription. The phone buys; the server asks Google whether the purchase is real and until when
it runs, and keeps that as users.pro_until. Nothing the phone says about a purchase is believed.

Off until the server has a Google service account's key file: GOOGLE_PLAY_KEY_FILE in .env (never in the code or the
app). Renewals: when pro_until has passed, the next profile read asks Google again (at most once per RECHECK_S)."""

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
PRODUCT = os.environ.get("PRO_PRODUCT_ID", "scanly_pro_monthly")
API = "https://androidpublisher.googleapis.com/androidpublisher/v3/applications"
SCOPE = "https://www.googleapis.com/auth/androidpublisher"
# CANCELED: renewal was switched off, and it still runs until its end.
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
    """Until when Google says the Pro subscription in its answer runs; 0 when it does not."""
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
    """A subscription that has run out may have renewed: ask Google again. Quiet on any trouble; Pro just stays off
    until the next try."""
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
    """The phone bought Pro: checked with Google, then this account is Pro until the date Google gives. One purchase
    serves one account."""
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
