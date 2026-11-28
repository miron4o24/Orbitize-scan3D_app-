"""Резервното копие (app/backup.py): профилите и готовите модели влизат, временните файлове не; връщането работи."""

import time
import zipfile

from tests.test_accounts import CLIENT_ID, client, finished_scan, sign_in  # noqa: F401 (client: fixture)


def test_backup_has_accounts_and_models_but_not_raw_uploads_and_restores(client, tmp_path, monkeypatch):
    import sqlalchemy as sa

    from app import accounts, backup, db, jobs

    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setattr(backup, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(backup, "BACKUP_KEEP", 2)
    h, user = sign_in(client)
    job = finished_scan(client, h)
    d = jobs.job_dir(job)
    (d / "photos" / "img_0000.jpg").write_bytes(b"x")
    (d / "video.mp4").write_bytes(b"v" * 1000)
    (d / "work").mkdir(exist_ok=True)
    (d / "work" / "big.bin").write_bytes(b"w")
    accounts.AVATARS.mkdir(parents=True, exist_ok=True)
    (accounts.AVATARS / f"{user['id']}.jpg").write_bytes(b"jpg")

    paths = []
    for _ in range(3):
        paths.append(backup.make())
        time.sleep(0.01)
    assert sorted((tmp_path / "backups").glob("*.zip")) == paths[-2:]  # пазят се последните BACKUP_KEEP
    names = zipfile.ZipFile(paths[-1]).namelist()
    assert "accounts.db" in names and f"avatars/{user['id']}.jpg" in names and f"jobs/{job}/model.ply" in names
    assert not any("/photos/" in n or n.endswith("video.mp4") or "/work/" in n for n in names)

    # Всичко изгубено: връщането от копието дава профила и модела обратно.
    db.reset()
    backup.sqlite_path().unlink()
    import shutil

    shutil.rmtree(d)
    backup.restore(paths[-1])
    assert (jobs.job_dir(job) / "model.ply").exists()
    with db.engine().connect() as c:
        assert c.execute(sa.select(db.users.c.email).where(db.users.c.id == user["id"])).scalar() == user["email"]
