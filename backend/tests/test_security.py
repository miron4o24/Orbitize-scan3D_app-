"""Сигурност: ограничения на опитите, изтичане на сесиите, правило за паролата, заглавки, скрито описание на API-то,
изчистени снимки на профила."""

import io
import time

import sqlalchemy as sa
from PIL import Image

from tests.test_accounts import CLIENT_ID, client, jpeg, sign_in  # noqa: F401 (client: fixture)


def mailbox(monkeypatch):
    from app import accounts, mail

    sent = []
    monkeypatch.setattr(mail, "configured", lambda: True)
    monkeypatch.setattr(mail, "verification_code", lambda to, code, minutes: sent.append((to, code)))
    return accounts, sent


def test_password_needs_8_chars_a_letter_and_a_digit(client, monkeypatch):
    mailbox(monkeypatch)
    for weak in ("abc12", "abcdefgh", "12345678"):
        r = client.post("/auth/register", json={"email": "w@example.com", "password": weak})
        assert r.status_code == 400 and "поне 8 знака" in r.json()["detail"]
    assert client.post("/auth/register", json={"email": "w@example.com", "password": "abcdefg1"}).status_code == 200


def test_google_sign_in_is_off_unless_enabled(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_SIGNIN")
    assert client.post("/auth/google", json={"id_token": "g1:a@example.com"}).status_code == 404
    assert client.get("/health").json()["google_client_id"] is None


def test_login_limited_per_ip_across_many_emails(client, monkeypatch):
    from app import accounts

    monkeypatch.setattr(accounts, "client_ip", lambda request: "203.0.113.7")
    codes = [client.post("/auth/login", json={"email": f"u{i}@example.com", "password": "wrong123"}).status_code
             for i in range(31)]  # fmt: skip
    assert codes[:30] == [401] * 30 and codes[30] == 429
    monkeypatch.setattr(accounts, "client_ip", lambda request: "198.51.100.1")  # друг адрес не е засегнат
    assert client.post("/auth/login", json={"email": "u0@example.com", "password": "wrong123"}).status_code == 401


def test_code_mails_are_limited(client, monkeypatch):
    accounts, sent = mailbox(monkeypatch)
    monkeypatch.setattr(accounts, "RESEND_S", 0)
    monkeypatch.setattr(accounts, "client_ip", lambda request: "203.0.113.8")
    got = [client.post("/auth/register", json={"email": f"v{i}@example.com", "password": "abcdefg1"}).status_code
           for i in range(6)]  # fmt: skip
    assert got == [200] * 5 + [429] and len(sent) == 5  # 5 писма на час от един адрес
    monkeypatch.setattr(accounts, "client_ip", lambda request: None)
    got = [client.post("/auth/resend", json={"email": "v0@example.com"}).status_code for _ in range(5)]
    assert got == [200] * 4 + [429]  # 5 писма на час до един имейл (едно вече тръгна при регистрацията)


def test_wrong_codes_limited_per_ip(client, monkeypatch):
    accounts, sent = mailbox(monkeypatch)
    monkeypatch.setattr(accounts, "RESEND_S", 0)
    monkeypatch.setattr(accounts, "client_ip", lambda request: "203.0.113.9")
    for i in range(4):
        client.post("/auth/register", json={"email": f"c{i}@example.com", "password": "abcdefg1"})
    tries = [client.post("/auth/verify", json={"email": f"c{i % 4}@example.com", "code": "000000"}).status_code
             for i in range(21)]  # fmt: skip
    assert tries[-1] == 429 and 429 not in tries[:20]


def test_old_and_idle_sessions_expire(client, monkeypatch):
    from app import accounts, db

    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    h, _ = sign_in(client)
    assert client.get("/me", headers=h).status_code == 200
    with db.engine().begin() as c:
        c.execute(sa.update(db.sessions).values(last_used=int(time.time()) - accounts.SESSION_IDLE_S - 10))
    assert client.get("/me", headers=h).status_code == 401
    h, _ = sign_in(client)
    with db.engine().begin() as c:
        c.execute(sa.update(db.sessions).values(created=int(time.time()) - accounts.SESSION_MAX_S - 10))
    assert client.get("/me", headers=h).status_code == 401
    with db.engine().connect() as c:
        assert c.execute(sa.select(sa.func.count()).select_from(db.sessions)).scalar() == 0


def test_security_headers_and_hidden_api_docs(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["content-security-policy"].startswith("default-src 'none'")
    assert r.json()["database"] == "ok"
    for path in ("/docs", "/openapi.json", "/redoc"):
        assert client.get(path).status_code == 404


def test_avatar_is_reencoded_without_exif(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    h, _ = sign_in(client)
    im = Image.new("RGB", (900, 600), (10, 200, 30))
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"  # Make
    exif[0x8825] = {2: (42.0, 41.0, 0.0)}  # GPSInfo: latitude
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    raw = buf.getvalue() + b"<script>alert(1)</script>"
    assert b"PhoneMaker" in raw
    assert client.put("/me/avatar", content=raw, headers=h).status_code == 200
    got = client.get("/me/avatar.jpg", headers=h).content
    assert b"PhoneMaker" not in got and b"<script>" not in got
    with Image.open(io.BytesIO(got)) as out:
        assert max(out.size) == 512 and not out.getexif()
    assert client.put("/me/avatar", content=b"\xff\xd8\xff" + b"0" * 100, headers=h).status_code == 400
    assert client.put("/me/avatar", content=jpeg(), headers=h).status_code == 200


def test_share_page_scripts_need_the_nonce(client, monkeypatch):
    import re

    from tests.test_accounts import finished_scan

    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    h, _ = sign_in(client)
    job_id = finished_scan(client, h)
    client.patch(f"/me/scans/{job_id}", json={"title": "</title><script>alert(1)</script>"}, headers=h)
    slug = client.post(f"/me/scans/{job_id}/share", headers=h).json()["share"]
    r = client.get(f"/s/{slug}")
    nonce = re.search(r"'nonce-([^']+)'", r.headers["content-security-policy"]).group(1)
    assert "'unsafe-inline'" not in r.headers["content-security-policy"].split("style-src")[0]
    assert r.text.count(f'nonce="{nonce}"') == 2 and r.text.count("<script") == 2
    assert "<script>alert(1)" not in r.text and "&lt;script&gt;" in r.text
    assert nonce not in client.get(f"/s/{slug}").text  # нов при всяко отваряне


def test_server_script_sets_a_new_password_and_signs_out(client, monkeypatch):
    import pytest

    from app import set_password as SP

    accounts, sent = mailbox(monkeypatch)
    client.post("/auth/register", json={"email": "t@example.com", "password": "abcdefg1"})
    client.post("/auth/verify", json={"email": "t@example.com", "code": sent[-1][1]})
    token = client.post("/auth/login", json={"email": "t@example.com", "password": "abcdefg1"}).json()["token"]
    with pytest.raises(SystemExit):
        SP.set_password("t@example.com", "123456789")  # същото правило като при регистрация
    SP.set_password(" T@example.com ", "Orbitize2026")
    assert client.get("/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert client.post("/auth/login", json={"email": "t@example.com", "password": "abcdefg1"}).status_code == 401
    assert client.post("/auth/login", json={"email": "t@example.com", "password": "Orbitize2026"}).status_code == 200


def test_cad_design_numbers_and_names_are_strict():
    import math as _m

    import pytest

    from app.cad import build as B
    from app.cad import design as D

    base = {"kind": "block", "dims": {k: {"v": 10.0} for k in ("L", "W", "H")}}
    for bad in (float("nan"), float("inf"), 1e9, -1e9, "1e400", True, [1]):
        with pytest.raises(D.DesignError):
            B.clean(base | {"dims": base["dims"] | {"L": {"v": bad}}})
    assert B.clean(base)["dims"]["L"]["v"] == 10.0 and not _m.isnan(B.clean(base)["dims"]["W"]["v"])
    with pytest.raises(D.DesignError):  # нов ред в края на име щеше да мине с "$"
        B.clean(base | {"dims": base["dims"] | {"x\n": {"v": 1.0}}})
    assert not D.DIM_ID.match("L\n") and D.DIM_ID.match("L")


def test_cad_build_runs_in_a_separate_process_with_a_time_limit(tmp_path, monkeypatch):
    import pytest

    from app.cad import design as D
    from app.cad import service as SV

    with pytest.raises(D.DesignError, match="Липсват общите размери"):  # грешката от отделния процес стига до човека
        SV.isolated({"kind": "block"}, tmp_path / "a")
    monkeypatch.setattr(SV, "BUILD_TIMEOUT_S", 0.01)
    with pytest.raises(D.DesignError, match="твърде сложен"):
        SV.isolated({"kind": "block"}, tmp_path / "b")


def test_heavy_requests_and_registrations_are_limited(client, monkeypatch):
    from app import accounts

    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setattr(accounts, "_heavy", accounts.Limit(2, 600))
    h, _ = sign_in(client)
    got = [client.post("/outline", content=b"not a photo", headers=h).status_code for _ in range(3)]
    assert got == [400, 400, 429]
    monkeypatch.setattr(accounts, "client_ip", lambda request: "203.0.113.20")
    got = [client.post("/auth/register", json={"email": "bad", "password": "abcdefg1"}).status_code for _ in range(31)]
    assert got[:30] == [400] * 30 and got[30] == 429


def test_users_do_not_see_server_paths_in_the_log(client, monkeypatch):
    from app import jobs
    from tests.test_accounts import ADMIN

    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    h, _ = sign_in(client)
    job = client.post("/jobs", headers=h).json()["id"]
    (jobs.job_dir(job) / "log.txt").write_text("colmap D:\\Orbitize\\backend\\data\\jobs\\x\\db.db /home/u/tools/colmap\n")
    mine = client.get(f"/jobs/{job}/log", headers=h).text
    assert "Orbitize" not in mine and "/home" not in mine and "…/db.db" in mine and "…/colmap" in mine
    assert "D:\\Orbitize" in client.get(f"/jobs/{job}/log", headers=ADMIN).text
