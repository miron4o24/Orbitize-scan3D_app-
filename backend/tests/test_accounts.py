"""Профили върху SQLite вместо MySQL (същите таблици на SQLAlchemy), с подправена проверка на ключа от Google."""

import io
import os
import time

import pytest
import trimesh
from PIL import Image

os.environ["API_TOKEN"] = "test-token-123"
ADMIN = {"Authorization": "Bearer test-token-123"}
CLIENT_ID = "1234-web.apps.googleusercontent.com"


def jpeg(color=(200, 40, 40), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import accounts, db, jobs, main, pipeline

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'accounts.db').as_posix()}")
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.setenv("GOOGLE_SIGNIN", "1")  # входът с Google е изключен по подразбиране; тестовете го ползват за профили
    from app import mail

    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_PORT", "SMTP_FROM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(mail, "ENV_FILE", tmp_path / "no.env")
    db.reset()
    accounts.reset_limits()
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(jobs, "DATA", tmp_path / "jobs")
    monkeypatch.setattr(accounts, "AVATARS", tmp_path / "avatars")

    def fake_verify(token, audiences):
        sub, email = token.split(":")  # подправеният "ID токен" е просто sub:email
        if CLIENT_ID not in audiences:
            raise ValueError("token is for another app")
        return {"sub": sub, "email": email, "email_verified": True, "name": f"User {sub}", "picture": "https://p/x.jpg"}

    monkeypatch.setattr(accounts, "verify_google", fake_verify)

    def fake_run(job, flat_base, set_state, sh, quick=False, express=False):
        set_state("reconstructing")
        trimesh.creation.box(extents=[1, 2, 4]).apply_translation([0, 0, 2]).export(job / "model.ply")
        return {"dimensions_mm": [25, 50, 100], "triangles": 12, "watertight": True}

    monkeypatch.setattr(pipeline, "run", fake_run)
    with TestClient(main.app) as c:
        yield c
    db.reset()


def sign_in(client, sub="g1", email="a@example.com"):
    r = client.post("/auth/google", json={"id_token": f"{sub}:{email}"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}, r.json()["user"]


def finished_scan(client, h, frames=8):
    job_id = client.post("/jobs", headers=h).json()["id"]
    for i in range(frames):
        client.put(f"/jobs/{job_id}/photos/img_{i:04d}.jpg", content=jpeg(), headers=h)
    # Стандартното сканиране е само за API ключа на сървъра: потребителят не може да го пусне.
    if h is not ADMIN:
        r = client.post(f"/jobs/{job_id}/start", json={}, headers=h)
        assert r.status_code == 400 and "Стандартното сканиране" in r.json()["detail"]
    assert client.post(f"/jobs/{job_id}/start", json={}, headers=ADMIN).status_code == 200
    end = time.time() + 5
    while client.get(f"/jobs/{job_id}", headers=h).json()["state"] != "done":
        assert time.time() < end
        time.sleep(0.05)
    return job_id


def test_google_client_id_set_remotely_by_admin(client):
    # accounts: входът с имейл работи (базата данни е достатъчна); за Google първо е нужен идентификатор на клиента.
    assert client.get("/health").json() == {"ok": True, "accounts": True, "database": "ok", "google_client_id": None,
                                         "version": "2026-11-25-mail", "stress": True, "progress": True, "pro": True,
                                         "mail": False}
    assert client.post("/auth/google", json={"id_token": "g1:a@example.com"}).status_code == 503
    assert client.put("/admin/settings", json={"google_client_id": CLIENT_ID}).status_code == 401
    r = client.put("/admin/settings", json={"google_client_id": f" {CLIENT_ID} "}, headers=ADMIN)
    assert r.json()["google_client_id"] == CLIENT_ID
    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)  # същата стойност отново
    assert client.get("/health").json()["accounts"] is True
    assert client.get("/health").json()["google_client_id"] == CLIENT_ID
    h, user = sign_in(client)
    assert user["email"] == "a@example.com" and user["name"] == "User g1" and user["stats"]["scans"] == 0
    assert client.get("/admin/settings", headers=h).status_code == 403


def test_profile_scans_and_privacy(client):
    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    h, user = sign_in(client)
    h_again, again = sign_in(client)  # втори телефон: същият профил, собствена сесия
    assert again["id"] == user["id"] and h_again != h
    other, _ = sign_in(client, "g2", "b@example.com")

    job_id = finished_scan(client, h)
    assert [s["id"] for s in client.get("/me/scans", headers=h_again).json()] == [job_id]
    assert client.get("/me/scans", headers=other).json() == []
    for url in (f"/jobs/{job_id}", f"/jobs/{job_id}/result.glb", f"/jobs/{job_id}/thumb.jpg", f"/jobs/{job_id}/log"):
        assert client.get(url, headers=other).status_code == 404, url
        assert client.get(url, headers=h).status_code == 200, url
        assert client.get(url, headers=ADMIN).status_code == 200, url  # администраторският ключ вижда всяка задача
    assert client.delete(f"/jobs/{job_id}", headers=other).status_code == 404
    assert client.get("/jobs", headers=h).status_code == 403  # списъкът на всички задачи е само за администратор

    assert client.patch(f"/me/scans/{job_id}", json={"title": "  Drill   one "}, headers=h).json()["title"] == "Drill one"
    me = client.patch("/me", json={"name": " Miro "}, headers=h).json()
    assert me["name"] == "Miro" and me["stats"] == {"scans": 1, "done": 1, "seconds": me["stats"]["seconds"], "shared": 0,
                                              "published": 0}  # fmt: skip
    assert client.patch("/me", json={"name": "   "}, headers=h).status_code == 400

    assert client.put("/me/avatar", content=b"not a jpeg", headers=h).status_code == 400
    assert client.put("/me/avatar", content=jpeg(), headers=h).json()["has_avatar"] is True
    assert client.get("/me/avatar.jpg", headers=h).content[:3] == b"\xff\xd8\xff"  # записана наново (без EXIF)
    assert client.get("/me/avatar.jpg", headers=other).status_code == 404
    assert client.delete("/me/avatar", headers=h).json()["has_avatar"] is False

    assert client.post("/auth/logout", headers=h).status_code == 200
    assert client.get("/me", headers=h).status_code == 401
    assert client.get("/me", headers=h_again).status_code == 200  # другият телефон остава влязъл


def test_share_link(client):
    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    h, _ = sign_in(client)
    job_id = finished_scan(client, h)
    client.patch(f"/me/scans/{job_id}", json={"title": "<b>Drill</b>"}, headers=h)
    slug = client.post(f"/me/scans/{job_id}/share", headers=h).json()["share"]
    assert client.post(f"/me/scans/{job_id}/share", headers=h).json()["share"] == slug  # същият линк отново

    page = client.get(f"/s/{slug}", headers={"host": "myserver.tail1234.ts.net"})
    assert page.status_code == 200 and "&lt;b&gt;Drill&lt;/b&gt;" in page.text and "<b>Drill" not in page.text
    assert f"https://myserver.tail1234.ts.net/s/{slug}/thumb.jpg" in page.text  # прегледите на линкове се нуждаят от целия адрес
    assert f"'/s/{slug}/model.glb'" in page.text
    glb = client.get(f"/s/{slug}/model.glb")  # без ключ: публично
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert client.get(f"/s/{slug}/thumb.jpg").content[:3] == b"\xff\xd8\xff"
    assert client.get("/me", headers=h).json()["stats"]["shared"] == 1

    client.delete(f"/me/scans/{job_id}/share", headers=h)
    assert client.get(f"/s/{slug}").status_code == 404
    assert client.get(f"/s/{slug}/model.glb").status_code == 404
    assert client.get("/s/nope").status_code == 404


def test_claim_old_scans_and_delete_account(client):
    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    old = finished_scan(client, ADMIN)  # създадено преди профилите, със споделения ключ
    h, _ = sign_in(client)
    other, _ = sign_in(client, "g2", "b@example.com")
    assert client.post("/me/scans/claim", json={"job_ids": [old, "0" * 32, "../x"]}, headers=h).json() == {
        "claimed": [old]
    }
    assert client.post("/me/scans/claim", json={"job_ids": [old]}, headers=other).json() == {"claimed": []}
    assert [s["id"] for s in client.get("/me/scans", headers=h).json()] == [old]

    from app import jobs

    assert client.delete("/me", headers=h).status_code == 200
    assert jobs.read(old) is None  # сканиранията на профила се изтриват с него
    assert client.get("/me", headers=h).status_code == 401
    assert client.get("/me", headers=other).status_code == 200


def test_cleanup_keeps_account_models_but_frees_frames(client):
    from app import jobs

    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    h, _ = sign_in(client)
    mine = finished_scan(client, h)
    anon = finished_scan(client, ADMIN)
    for j in (mine, anon):
        jobs.write(j, created=time.time() - 30 * 86400)
    jobs.cleanup()
    assert jobs.read(anon) is None
    assert client.get(f"/jobs/{mine}/result.glb", headers=h).status_code == 200
    assert not any((jobs.job_dir(mine) / "photos").iterdir())
    assert client.get(f"/jobs/{mine}/thumb.jpg", headers=h).status_code == 200  # създадено преди кадрите да отпаднат


def test_account_limits(client, monkeypatch):
    from app import accounts, main

    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    h, _ = sign_in(client)
    for _ in range(3):
        assert client.post("/jobs", headers=h).status_code == 200
    assert client.post("/jobs", headers=h).status_code == 429  # твърде много незавършени качвания
    monkeypatch.setattr(accounts, "MAX_SCANS_PER_USER", 3)
    assert client.post("/jobs", headers=h).status_code == 400
    monkeypatch.setattr(main, "MAX_JOBS", 3)
    assert client.post("/jobs", headers=ADMIN).status_code == 503  # качванията се броят към ограничението на сървъра


def test_every_account_can_use_everything(client, monkeypatch):
    """Pro и безплатният план са премахнати: обикновен профил прави ИИ модели колкото иска, без Pro."""
    from app import ai

    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    h, user = sign_in(client)
    assert user["pro"] is True and "limits" not in user and "billing" not in user and not user["owner"]
    monkeypatch.setattr(ai, "available", lambda: {"single": True})

    def fake_ai(job, method, flat_base, set_state, sh, step, tune=None):
        set_state("reconstructing")
        trimesh.creation.box(extents=[1, 2, 4]).apply_translation([0, 0, 2]).export(job / "model.ply")
        return {"dimensions_mm": [25, 50, 100], "triangles": 12, "watertight": True, "ai": method}

    monkeypatch.setattr(ai, "run", fake_ai)
    made = []
    for _ in range(3):  # безплатният план имаше един AI модел
        job_id = client.post("/jobs", headers=h).json()["id"]
        client.put(f"/jobs/{job_id}/photos/img_0000.jpg", content=jpeg(), headers=h)
        r = client.post(f"/jobs/{job_id}/start", json={"method": "single"}, headers=h)
        assert r.status_code == 200, r.text
        made.append(job_id)
    end = time.time() + 10
    while any(client.get(f"/jobs/{j}", headers=h).json()["state"] != "done" for j in made):
        assert time.time() < end
        time.sleep(0.05)
    for job_id in made:  # добавената реалност (AR) е махната
        assert client.post(f"/jobs/{job_id}/ar", json={"longest_mm": 100}, headers=h).status_code in (404, 405)
    assert client.get("/owner/users", headers=h).status_code == 404  # страницата за Pro я няма


def test_accounts_turned_off(client, monkeypatch):
    from app import db

    monkeypatch.setenv("DATABASE_URL", "off")
    db.reset()
    assert client.get("/health").json() == {"ok": True, "accounts": False, "database": "off", "google_client_id": None,
                                         "version": "2026-11-25-mail", "stress": True, "progress": True, "pro": True,
                                         "mail": False}
    assert client.post("/auth/google", json={"id_token": "g1:a@example.com"}).status_code == 503
    assert client.post("/jobs", headers=ADMIN).status_code == 200  # споделеният ключ работи както преди
    assert client.get("/me", headers={"Authorization": "Bearer whatever"}).status_code == 401


def test_new_email_account_is_confirmed_with_a_code_from_the_mail(client, monkeypatch):
    """Регистрация с имейл и парола, когато сървърът има пощенска кутия: без вход до кода от писмото."""
    from app import accounts, mail

    sent = []
    monkeypatch.setattr(mail, "configured", lambda: True)
    monkeypatch.setattr(mail, "verification_code", lambda to, code, minutes: sent.append((to, code)))
    r = client.post("/auth/register", json={"email": "ana@example.com", "password": "secret12", "name": "Ана"})
    assert r.status_code == 200 and r.json() == {"verify": True, "email": "ana@example.com"} and "token" not in r.json()
    assert len(sent) == 1 and sent[0][0] == "ana@example.com" and len(sent[0][1]) == 6
    # Вход преди потвърждението: пак кодът (писмото не се праща отново веднага), без сесия.
    r = client.post("/auth/login", json={"email": "ana@example.com", "password": "secret12"})
    assert r.json() == {"verify": True, "email": "ana@example.com"} and len(sent) == 1
    wrong = "000000" if sent[0][1] != "000000" else "111111"
    r = client.post("/auth/verify", json={"email": "ana@example.com", "code": wrong})
    assert r.status_code == 400
    # Ново писмо: не по-рано от RESEND_S.
    assert client.post("/auth/resend", json={"email": "ana@example.com"}).status_code == 429
    r = client.post("/auth/verify", json={"email": "ana@example.com", "code": f" {sent[0][1][:3]} {sent[0][1][3:]} "})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    assert client.get("/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    # Вече потвърден: входът дава сесия, а кодът не важи втори път.
    r = client.post("/auth/login", json={"email": "ana@example.com", "password": "secret12"})
    assert "token" in r.json()
    assert client.post("/auth/verify", json={"email": "ana@example.com", "code": sent[0][1]}).status_code == 400
    assert accounts.CODE_TRIES >= 3


def test_the_code_stops_working_after_too_many_wrong_tries(client, monkeypatch):
    from app import accounts, mail

    sent = []
    monkeypatch.setattr(mail, "configured", lambda: True)
    monkeypatch.setattr(mail, "verification_code", lambda to, code, minutes: sent.append(code))
    client.post("/auth/register", json={"email": "bob@example.com", "password": "secret12"})
    wrong = "000000" if sent[0] != "000000" else "111111"
    for _ in range(accounts.CODE_TRIES):
        assert client.post("/auth/verify", json={"email": "bob@example.com", "code": wrong}).status_code == 400
    assert client.post("/auth/verify", json={"email": "bob@example.com", "code": sent[0]}).status_code == 429


def test_without_a_mailbox_there_is_no_registration(client, monkeypatch):
    """Сървър без SMTP: регистрацията се отказва и профил не остава (иначе някой би влязъл без потвърден имейл)."""
    from app import mail

    monkeypatch.setattr(mail, "configured", lambda: False)
    r = client.post("/auth/register", json={"email": "cid@example.com", "password": "secret12"})
    assert r.status_code == 503 and "token" not in r.json()
    assert client.post("/auth/login", json={"email": "cid@example.com", "password": "secret12"}).status_code == 401


def test_a_failed_mail_leaves_no_account(client, monkeypatch):
    from app import mail

    def broken(to, code, minutes):
        raise OSError("smtp down")

    monkeypatch.setattr(mail, "configured", lambda: True)
    monkeypatch.setattr(mail, "verification_code", broken)
    assert client.post("/auth/register", json={"email": "eva@example.com", "password": "secret12"}).status_code == 503
    assert client.post("/auth/login", json={"email": "eva@example.com", "password": "secret12"}).status_code == 401


def test_registering_again_before_the_code_replaces_the_unconfirmed_account(client, monkeypatch):
    """Грешна парола или изгубено писмо не заключват имейла: нова регистрация презаписва непотвърдения профил."""
    from app import accounts, mail

    sent = []
    monkeypatch.setattr(mail, "configured", lambda: True)
    monkeypatch.setattr(mail, "verification_code", lambda to, code, minutes: sent.append(code))
    client.post("/auth/register", json={"email": "fil@example.com", "password": "first123"})
    assert client.post("/auth/register", json={"email": "fil@example.com", "password": "second12"}).status_code == 429
    monkeypatch.setattr(accounts, "RESEND_S", 0)
    r = client.post("/auth/register", json={"email": "fil@example.com", "password": "second12", "name": "Фил"})
    assert r.json() == {"verify": True, "email": "fil@example.com"} and len(sent) == 2
    assert client.post("/auth/verify", json={"email": "fil@example.com", "code": sent[0]}).status_code == 400
    r = client.post("/auth/verify", json={"email": "fil@example.com", "code": sent[1]})
    assert r.json()["user"]["name"] == "Фил"
    assert client.post("/auth/login", json={"email": "fil@example.com", "password": "first123"}).status_code == 401
    assert "token" in client.post("/auth/login", json={"email": "fil@example.com", "password": "second12"}).json()
    r = client.post("/auth/register", json={"email": "fil@example.com", "password": "third123"})
    assert r.status_code == 409  # потвърден профил не се презаписва


def test_an_old_google_account_gets_a_password_through_the_mail_code(client, monkeypatch):
    """Входът с Google го няма: стар профил от Google се регистрира със същия имейл и пази сканиранията си."""
    from app import mail

    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    _, user = sign_in(client, "g9", "gina@example.com")
    old_id = user["id"]
    sent = []
    monkeypatch.setattr(mail, "configured", lambda: True)
    monkeypatch.setattr(mail, "verification_code", lambda to, code, minutes: sent.append(code))
    assert client.post("/auth/login", json={"email": "gina@example.com", "password": "secret12"}).status_code == 401
    r = client.post("/auth/register", json={"email": "Gina@example.com", "password": "secret12"})
    assert r.json()["verify"] is True and len(sent) == 1
    assert client.post("/auth/login", json={"email": "gina@example.com", "password": "secret12"}).json()["verify"] is True
    r = client.post("/auth/verify", json={"email": "gina@example.com", "code": sent[0]})
    assert r.json()["user"]["id"] == old_id
    assert client.post("/auth/login", json={"email": "gina@example.com", "password": "secret12"}).json()["user"]["id"] == old_id
    assert client.post("/auth/register", json={"email": "gina@example.com", "password": "other123"}).status_code == 409



def test_the_mailbox_is_read_from_the_env_file_even_if_the_server_started_without_it(client, monkeypatch, tmp_path):
    """Сървър, пуснат преди редовете SMTP_* да са в .env (или от скрипт без .env): кодът пак тръгва."""
    from app import mail

    env = tmp_path / "server.env"
    env.write_text("\ufeffPORT=8010\nSMTP_HOST=smtp.gmail.com\nSMTP_PORT=587\nSMTP_USER=x@gmail.com\n"
                   "SMTP_PASSWORD=abcd efgh ijkl mnop\nSMTP_FROM=Orbitize <x@gmail.com>\n", encoding="utf-8")
    monkeypatch.setattr(mail, "ENV_FILE", env)
    cfg = mail.settings()
    assert cfg["host"] == "smtp.gmail.com" and cfg["port"] == 587 and cfg["password"] == "abcdefghijklmnop"
    assert cfg["sender"] == "Orbitize <x@gmail.com>"
    assert client.get("/health").json()["mail"] is True
    sent = []
    monkeypatch.setattr(mail, "verification_code", lambda to, code, minutes: sent.append(code))
    r = client.post("/auth/register", json={"email": "hal@example.com", "password": "secret12"})
    assert r.json() == {"verify": True, "email": "hal@example.com"} and len(sent) == 1


def test_nameplate_under_a_model_still_works_and_text_changes_are_gone(client):
    """Табелката с име остава (app/edit.py); текстовите промени и добавената реалност са махнати."""
    client.put("/admin/settings", json={"google_client_id": CLIENT_ID}, headers=ADMIN)
    h, _ = sign_in(client)
    source = finished_scan(client, h)
    assert client.post(f"/jobs/{source}/edit", json={"prompt": "make it red"}, headers=h).status_code in (404, 405)
    r = client.post(f"/jobs/{source}/nameplate", json={"text": "Иван", "shape": "round", "longest_mm": 80}, headers=h)
    assert r.status_code == 200, r.text
    new = r.json()["id"]
    end = time.time() + 120
    while (job := client.get(f"/jobs/{new}", headers=h).json())["state"] not in ("done", "failed"):
        assert time.time() < end
        time.sleep(0.2)
    assert job["state"] == "done", job
    assert job["report"]["edit"]["changes"] == ["Табелка с надпис „Иван“"]
