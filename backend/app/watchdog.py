"""Проверка на сървъра отвън (задача „scan3d-watch“ на всеки 5 минути, scripts/watch.ps1): отговаря ли /health на
сървъра локално и на публичния адрес (PUBLIC_URL в .env, например адресът на Tailscale Funnel) и има ли място на
диска. При промяна (нещо се развали или се оправи) праща едно писмо на собственика (OWNER_EMAILS) през същата поща
като кодовете. Не променя нищо по сървъра. python -m app.watchdog"""

import json
import os
import shutil
import time
from pathlib import Path

import requests

from . import jobs, mail

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "logs" / "watchdog.json"
MIN_FREE_GB = float(os.environ.get("WATCH_MIN_FREE_GB", "10"))


def problems() -> list[str]:
    out = []
    port = os.environ.get("PORT", "8010")
    targets = [("локално", f"http://127.0.0.1:{port}/health")]
    if public := os.environ.get("PUBLIC_URL", "").strip().rstrip("/"):
        targets.append(("публичния адрес", f"{public}/health"))
    for where, url in targets:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200 or not r.json().get("ok"):
                out.append(f"Сървърът не отговаря нормално на {where} ({url}): HTTP {r.status_code}.")
            elif r.json().get("database") not in (None, "ok", "off"):
                out.append(f"Базата данни не работи ({url}).")
        except (requests.RequestException, ValueError) as e:
            out.append(f"Сървърът не отговаря на {where} ({url}): {type(e).__name__}.")
    disk = jobs.DATA if jobs.DATA.exists() else ROOT
    free = shutil.disk_usage(disk).free / 1e9
    if free < MIN_FREE_GB:
        out.append(f"На диска остават {free:.1f} GB (под {MIN_FREE_GB:g} GB).")
    return out


def check() -> list[str]:
    now = problems()
    try:
        before = json.loads(STATE.read_text(encoding="utf-8")).get("problems", [])
    except (OSError, ValueError):
        before = []
    if now != before:
        to = [e.strip() for e in os.environ.get("OWNER_EMAILS", "").split(",") if e.strip()]
        if now:
            subject, text = "Orbitize: проблем със сървъра", "Проверката в " + time.strftime("%H:%M") + " намери:\n\n" + "\n".join(f"- {p}" for p in now)
        else:
            subject, text = "Orbitize: сървърът отново работи", "Проверката в " + time.strftime("%H:%M") + " не намери проблеми."
        for addr in to:
            try:
                mail.send(addr, subject, text)
            except Exception as e:  # без поща: поне в дневника
                print(f"watchdog: mail to {addr} failed: {e}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"checked": int(time.time()), "problems": now}, ensure_ascii=False), encoding="utf-8")
    return now


if __name__ == "__main__":
    found = check()
    print("\n".join(found) if found else "ok")
