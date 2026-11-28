"""Проверката на сървъра (app/watchdog.py): едно писмо, когато нещо се развали, и едно, когато се оправи."""

import requests


class Health:
    status_code = 200

    def json(self):
        return {"ok": True, "database": "ok"}


def test_one_mail_when_down_and_one_when_back(tmp_path, monkeypatch):
    from app import mail, watchdog

    sent, up = [], {"v": False}
    monkeypatch.setattr(watchdog, "STATE", tmp_path / "watchdog.json")
    monkeypatch.setenv("OWNER_EMAILS", "owner@example.com")
    monkeypatch.setenv("PUBLIC_URL", "https://example.ts.net/")
    monkeypatch.setattr(mail, "send", lambda to, subject, text: sent.append((to, subject, text)))

    def get(url, timeout):
        if not up["v"]:
            raise requests.ConnectionError("down")
        return Health()

    monkeypatch.setattr(watchdog.requests, "get", get)
    monkeypatch.setattr(watchdog, "MIN_FREE_GB", 0)
    assert len(watchdog.check()) == 2 and len(sent) == 1 and "проблем" in sent[0][1]
    assert "https://example.ts.net/health" in sent[0][2]
    watchdog.check()
    assert len(sent) == 1  # същият проблем: без ново писмо
    up["v"] = True
    assert watchdog.check() == [] and len(sent) == 2 and "отново работи" in sent[1][1]
    monkeypatch.setattr(watchdog, "MIN_FREE_GB", 10**9)  # дискът "свършва"
    assert "диска" in watchdog.check()[0] and len(sent) == 3
