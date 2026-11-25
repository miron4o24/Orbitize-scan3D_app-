"""Имейли от сървъра (засега само кодът за потвърждение на нов профил, app/accounts.py), през SMTP пощенска кутия.
Настройва се само в .env на сървъра (никога в кода и никога от приложението): SMTP_HOST, SMTP_PORT (587: STARTTLS,
465: SSL), SMTP_USER, SMTP_PASSWORD и SMTP_FROM (по подразбиране SMTP_USER). С Gmail: smtp.gmail.com, 587, адресът и
"парола за приложения" (Google профил -> Сигурност -> Потвърждаване в две стъпки -> Пароли за приложения). Без пощенска
кутия няма регистрация: всеки нов профил влиза едва след кода от писмото."""

import logging
import os
import re
import smtplib
from pathlib import Path
import ssl
from email.message import EmailMessage


log = logging.getLogger(__name__)

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _from_file() -> dict:
    """SMTP_* направо от backend/.env, при всяко писмо: промяна във файла важи веднага, дори ако сървърът е пуснат
    преди нея или от скрипт, който не е заредил .env."""
    out = {}
    try:
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^\s*(SMTP_[A-Z]+)\s*=\s*(.*?)\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    except OSError:
        pass
    return out


def settings() -> dict:
    """Пощенската кутия ({} без нея): от средата на процеса, а липсващото от backend/.env."""
    env = {k: v for k, v in _from_file().items() if v} | {k: v for k, v in os.environ.items() if k.startswith("SMTP_") and v}
    if not (env.get("SMTP_HOST") and env.get("SMTP_USER") and env.get("SMTP_PASSWORD")):
        return {}
    return {"host": env["SMTP_HOST"], "port": int(env.get("SMTP_PORT") or "587"), "user": env["SMTP_USER"],
            "password": env["SMTP_PASSWORD"].replace(" ", ""), "sender": env.get("SMTP_FROM") or ""}  # fmt: skip


def status() -> dict:
    """За екрана на администратора: включена ли е пощата и от кой адрес, без паролата."""
    s = settings()
    return {"configured": bool(s), "user": s.get("user")}


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
