"""Имейли от сървъра (засега само кодът за потвърждение на нов профил, app/accounts.py), през SMTP пощенска кутия.
Настройва се с променливи на средата: SMTP_HOST, SMTP_PORT (587: STARTTLS, 465: SSL), SMTP_USER, SMTP_PASSWORD и
SMTP_FROM (по подразбиране SMTP_USER), или от собственика в приложението (администриране, app/admin.py): тогава се пази
в таблицата с настройки на базата данни и паролата никога не се връща. Средата е с предимство. С Gmail:
smtp.gmail.com, 587, адресът и "парола за приложения" (Google профил -> Сигурност -> Потвърждаване в две стъпки ->
Пароли за приложения). Без пощенска кутия имейли не се изпращат и новите профили не се потвърждават."""

import json
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

from . import db

log = logging.getLogger(__name__)

SETTING = "smtp"  # JSON с host, port, user, password, sender в таблицата с настройки


def settings() -> dict:
    """Пощенската кутия: от средата, ако е зададена там, иначе от базата данни ({} без настройка)."""
    if os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASSWORD"):
        return {"host": os.environ["SMTP_HOST"], "port": int(os.environ.get("SMTP_PORT", "587")),
                "user": os.environ["SMTP_USER"], "password": os.environ["SMTP_PASSWORD"],
                "sender": os.environ.get("SMTP_FROM") or "", "source": "server"}  # fmt: skip
    try:
        saved = json.loads(db.get_setting(SETTING) or "{}")
    except Exception:  # noqa: BLE001 - повредена стойност или недостъпна база данни: без поща
        return {}
    if not (saved.get("host") and saved.get("user") and saved.get("password")):
        return {}
    return saved | {"port": int(saved.get("port") or 587), "source": "app"}


def save(host: str, port: int, user: str, password: str | None, sender: str) -> None:
    """Записва пощенската кутия от приложението; password None запазва досегашната парола, host "" я изтрива."""
    if not host:
        db.set_setting(SETTING, "{}")
        return
    old = {}
    try:
        old = json.loads(db.get_setting(SETTING) or "{}")
    except ValueError:
        pass
    db.set_setting(SETTING, json.dumps({"host": host, "port": port, "user": user, "sender": sender,
                                        "password": old.get("password", "") if password is None else password}))


def status() -> dict:
    """За екрана на администратора: без паролата."""
    s = settings()
    return {"configured": bool(s), "source": s.get("source"), "host": s.get("host"), "port": s.get("port"),
            "user": s.get("user"), "sender": s.get("sender") or None}  # fmt: skip


def configured() -> bool:
    return bool(settings())


def send(to: str, subject: str, text: str, html: str | None = None) -> None:
    """Изпраща писмо; хвърля OSError/smtplib.SMTPException при неуспех."""
    cfg = settings()
    if not cfg:
        raise OSError("no mailbox configured")
    host, user, port = cfg["host"], cfg["user"], cfg["port"]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.get("sender") or f"Orbitize <{user}>"
    msg["To"] = to
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as s:
            s.login(user, cfg["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls(context=context)
            s.login(user, cfg["password"])
            s.send_message(msg)
    log.info("mail: sent %r to %s", subject, to)


def verification_code(to: str, code: str, minutes: int) -> None:
    text = (
        f"Здравейте,\n\nкодът за потвърждение на профила ви в Orbitize е: {code}\n\n"
        f"Въведете го в приложението. Важи {minutes} минути.\n\n"
        "Ако не сте създавали профил, просто пренебрегнете това писмо."
    )
    html = (
        "<div style=\"font-family:Arial,sans-serif;color:#0b1116\">"
        "<p>Здравейте,</p><p>кодът за потвърждение на профила ви в <b>Orbitize</b> е:</p>"
        f"<p style=\"font-family:monospace;font-size:30px;letter-spacing:6px;font-weight:bold\">{code}</p>"
        f"<p>Въведете го в приложението. Важи {minutes} минути.</p>"
        "<p style=\"color:#566370;font-size:13px\">Ако не сте създавали профил, просто пренебрегнете това писмо.</p></div>"
    )
    send(to, f"Код за Orbitize: {code}", text, html)


def test(to: str) -> None:
    send(to, "Пробно писмо от Orbitize",
         "Пощенската кутия на сървъра работи: новите профили ще получават код за потвърждение.",
         "<p style=\"font-family:Arial,sans-serif\">Пощенската кутия на сървъра работи: новите профили ще получават "
         "<b>код за потвърждение</b>.</p>")
