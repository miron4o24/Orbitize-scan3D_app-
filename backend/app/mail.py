"""Имейли от сървъра (засега само кодът за потвърждение на нов профил, app/accounts.py), през SMTP пощенска кутия.
Настройва се с променливи на средата: SMTP_HOST, SMTP_PORT (587: STARTTLS, 465: SSL), SMTP_USER, SMTP_PASSWORD и
SMTP_FROM (по подразбиране SMTP_USER). С Gmail: smtp.gmail.com, 587, адресът и "парола за приложения" (Google профил
-> Сигурност -> Потвърждаване в две стъпки -> Пароли за приложения). Без SMTP_HOST имейли не се изпращат."""

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger(__name__)


def configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASSWORD"))


def send(to: str, subject: str, text: str, html: str | None = None) -> None:
    """Изпраща писмо; хвърля OSError/smtplib.SMTPException при неуспех."""
    host, user = os.environ["SMTP_HOST"], os.environ["SMTP_USER"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM") or f"Orbitize <{user}>"
    msg["To"] = to
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as s:
            s.login(user, os.environ["SMTP_PASSWORD"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls(context=context)
            s.login(user, os.environ["SMTP_PASSWORD"])
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
