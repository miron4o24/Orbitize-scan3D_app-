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
    db.reset()
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(jobs, "DATA", tmp_path / "jobs")
    monkeypatch.setattr(main, "CLASSIC_SCAN", True)  # потокът на задачите се проверява с подправения pipeline.run
    monkeypatch.setattr(accounts, "AVATARS", tmp_path / "avatars")

    def fake_verify(token, audiences):
        sub, email = token.split(":")  # подправеният "ID токен" е просто sub:email
        if CLIENT_ID not in audiences:
            raise ValueError("token is for another app")
        return {"sub": sub, "email": email, "email_verified": True, "name": f"User {sub}", "picture": "https://p/x.jpg"}

    monkeypatch.setattr(accounts, "verify_google", fake_verify)

    def fake_run(job, flat_base, set_state, sh, quick=False):
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
    assert client.post(f"/jobs/{job_id}/start", json={}, headers=h).status_code == 200
    end = time.time() + 5
    while client.get(f"/jobs/{job_id}", headers=h).json()["state"] != "done":
        assert time.time() < end
        time.sleep(0.05)
    return job_id


def test_google_client_id_set_remotely_by_admin(client):
    # accounts: входът с имейл работи (базата данни е достатъчна); за Google първо е нужен идентификатор на клиента.
    assert client.get("/health").json() == {"ok": True, "accounts": True, "database": "ok", "google_client_id": None,
                                         "version": "2026-11-05-detail", "stress": True, "progress": True, "pro": True}
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
    assert client.get("/me/avatar.jpg", headers=h).content == jpeg()
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


def test_accounts_turned_off(client, monkeypatch):
    from app import db

    monkeypatch.setenv("DATABASE_URL", "off")
    db.reset()
    assert client.get("/health").json() == {"ok": True, "accounts": False, "database": "off", "google_client_id": None,
                                         "version": "2026-11-05-detail", "stress": True, "progress": True, "pro": True}
    assert client.post("/auth/google", json={"id_token": "g1:a@example.com"}).status_code == 503
    assert client.post("/jobs", headers=ADMIN).status_code == 200  # споделеният ключ работи както преди
    assert client.get("/me", headers={"Authorization": "Bearer whatever"}).status_code == 401
