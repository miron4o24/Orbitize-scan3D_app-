"""Администриране от приложението (app/admin.py): кой има достъп, статус на профилите, задачите, пощенската кутия."""

from tests.test_accounts import ADMIN, CLIENT_ID, client, finished_scan, sign_in  # noqa: F401 (client: fixture)


def setup(client, monkeypatch):
    from app import accounts

    monkeypatch.setattr(accounts, "OWNER_EMAILS", "boss@example.com")
    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    boss, _ = sign_in(client, "g-boss", "boss@example.com")
    ana, ana_user = sign_in(client, "g-ana", "ana@example.com")
    return boss, ana, ana_user


def test_only_admins_see_the_admin_screens(client, monkeypatch):
    boss, ana, _ = setup(client, monkeypatch)
    assert client.get("/manage/overview", headers=ana).status_code == 403
    assert client.get("/manage/users", headers=ana).status_code == 403
    r = client.get("/manage/overview", headers=boss)
    assert r.status_code == 200, r.text
    o = r.json()
    assert o["users"]["total"] == 2 and o["users"]["admins"] == 1 and o["me"]["owner"] is True
    assert client.get("/me", headers=boss).json()["admin"] is True
    assert client.get("/me", headers=ana).json()["admin"] is False
    assert client.get("/manage/overview", headers=ADMIN).status_code == 200  # ключът на сървъра също


def test_block_user_signs_them_out_and_stops_sign_in(client, monkeypatch):
    boss, ana, ana_user = setup(client, monkeypatch)
    users = client.get("/manage/users", params={"q": "ana"}, headers=boss).json()["users"]
    assert [u["email"] for u in users] == ["ana@example.com"]
    r = client.patch(f"/manage/users/{ana_user['id']}", json={"blocked": True}, headers=boss)
    assert r.status_code == 200 and r.json()["blocked"] is True and r.json()["sessions"] == 0
    assert client.get("/me", headers=ana).status_code == 401  # сесията е изтрита
    assert client.post("/auth/google", json={"id_token": "g-ana:ana@example.com"}).status_code == 403
    assert client.get("/manage/users", params={"show": "blocked"}, headers=boss).json()["total"] == 1
    client.patch(f"/manage/users/{ana_user['id']}", json={"blocked": False}, headers=boss)
    assert client.post("/auth/google", json={"id_token": "g-ana:ana@example.com"}).status_code == 200


def test_admin_rights_belong_to_the_owner(client, monkeypatch):
    boss, ana, ana_user = setup(client, monkeypatch)
    cid, cid_user = sign_in(client, "g-cid", "cid@example.com")
    boss_id = client.get("/me", headers=boss).json()["id"]
    assert client.patch(f"/manage/users/{ana_user['id']}", json={"admin": True}, headers=boss).json()["admin"] is True
    # Ана е администратор: управлява обикновени профили, но не администраторите, собственика и себе си.
    assert client.get("/manage/overview", headers=ana).json()["me"]["owner"] is False
    assert client.patch(f"/manage/users/{cid_user['id']}", json={"blocked": True}, headers=ana).status_code == 200
    assert client.patch(f"/manage/users/{cid_user['id']}", json={"admin": True}, headers=ana).status_code == 403
    assert client.patch(f"/manage/users/{boss_id}", json={"blocked": True}, headers=ana).status_code == 403
    assert client.patch(f"/manage/users/{ana_user['id']}", json={"blocked": True}, headers=ana).status_code == 400
    assert client.delete(f"/manage/users/{boss_id}", headers=ana).status_code == 403


def test_admin_sees_any_scan_and_can_delete_an_account(client, monkeypatch):
    boss, ana, ana_user = setup(client, monkeypatch)
    job = finished_scan(client, ana)
    assert client.get(f"/jobs/{job}/thumb.jpg", headers=boss).status_code == 200
    detail = client.get(f"/manage/users/{ana_user['id']}", headers=boss).json()
    assert [s["id"] for s in detail["scans_list"]] == [job] and detail["scans_list"][0]["thumb"] is True
    assert client.get("/manage/jobs", params={"show": "all"}, headers=boss).json()[0]["user"]["id"] == ana_user["id"]
    assert client.delete(f"/manage/users/{ana_user['id']}", headers=boss).status_code == 200
    assert client.get("/manage/users", headers=boss).json()["total"] == 1
    assert client.get(f"/jobs/{job}", headers=boss).status_code == 404


def test_manual_verification_and_mail_only_from_the_server(client, monkeypatch):
    from app import mail

    boss, ana, _ = setup(client, monkeypatch)
    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    assert client.get("/manage/mail", headers=boss).json() == {"configured": False, "user": None}
    # Пощата не се задава от приложението: само от .env на сървъра.
    assert client.put("/manage/mail", json={"host": "smtp.gmail.com"}, headers=boss).status_code == 405
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_USER", "x@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    status = client.get("/manage/mail", headers=boss).json()
    assert status == {"configured": True, "user": "x@gmail.com"}  # без паролата
    sent = []
    monkeypatch.setattr(mail, "verification_code", lambda to, code, minutes: sent.append(code))
    r = client.post("/auth/register", json={"email": "dan@example.com", "password": "secret12"})
    assert r.json() == {"verify": True, "email": "dan@example.com"} and len(sent) == 1
    dan = client.get("/manage/users", params={"show": "unverified"}, headers=boss).json()["users"]
    assert [u["email"] for u in dan] == ["dan@example.com"] and dan[0]["verified"] is False
    assert client.patch(f"/manage/users/{dan[0]['id']}", json={"verified": True}, headers=boss).json()["verified"] is True
    assert "token" in client.post("/auth/login", json={"email": "dan@example.com", "password": "secret12"}).json()
    monkeypatch.setattr(mail, "test", lambda to: sent.append(to))
    assert client.post("/manage/mail/test", headers=boss).json() == {"sent_to": "boss@example.com"}


def test_old_scans_keep_their_picture_after_the_frames_are_gone(client, monkeypatch):
    """Кадрите изчезват след JOB_TTL_DAYS; рисунката на модела остава и списъкът трябва да я показва."""
    import time

    from app import jobs

    _, ana, _ = setup(client, monkeypatch)
    job = finished_scan(client, ana)
    jobs.write(job, created=time.time() - 30 * 86400)
    jobs.cleanup()
    (jobs.job_dir(job) / jobs.THUMB).unlink(missing_ok=True)
    entry = next(s for s in client.get("/me/scans", headers=ana).json() if s["id"] == job)
    assert entry["thumb"] is True
    assert client.get(f"/jobs/{job}/thumb.jpg", headers=ana).status_code == 200
