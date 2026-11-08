import io
import os
import struct
import time
import zipfile

import numpy as np
import pytest
import trimesh

os.environ["API_TOKEN"] = "test-token-123"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import ai, jobs, main

    monkeypatch.setattr(jobs, "DATA", tmp_path)

    def fake_run(job, method, flat_base, set_state, sh, step, tune=None, classic=False):
        set_state("reconstructing")
        trimesh.creation.box(extents=[1, 2, 4]).apply_translation([0, 0, 2]).export(job / "model.ply")
        return {"dimensions_mm": [25, 50, 100], "triangles": 12, "watertight": True}

    monkeypatch.setattr(ai, "run", fake_run)
    monkeypatch.setattr(ai, "available", lambda: {"edit": True, "single": True, "claude": False, "photo": False})
    with TestClient(main.app, headers={"Authorization": "Bearer test-token-123"}) as c:
        yield c


JPEG = b"\xff\xd8\xff\xe0" + b"0" * 100


def wait_state(client, job_id, want, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/jobs/{job_id}").json()
        if s["state"] == want:
            return s
        time.sleep(0.05)
    raise AssertionError(f"state stayed {s['state']}")


def test_auth_required(client):
    assert client.get("/health").status_code == 200
    assert client.post("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_upload_validation_and_resume(client):
    job = client.post("/jobs").json()
    url = f"/jobs/{job['id']}/photos"
    assert client.put(f"{url}/img_0001.jpg", content=JPEG).status_code == 200
    assert client.put(f"{url}/img_0001.jpg", content=JPEG).status_code == 200  # повторното качване е безопасно
    assert client.put(f"{url}/notes.txt", content=JPEG).status_code == 400
    assert client.put(f"{url}/..%2Fstatus.json", content=JPEG).status_code in (400, 404)
    assert client.put(f"{url}/img_0002.jpg", content=b"not a jpeg").status_code == 400
    assert client.get(f"/jobs/{job['id']}").json()["received"] == ["img_0001.jpg"]
    assert client.get("/jobs/../etc").status_code == 404


def test_full_lifecycle(client):
    job_id = client.post("/jobs").json()["id"]
    assert client.post(f"/jobs/{job_id}/start", json={}).status_code == 400  # твърде малко снимки
    for i in range(8):
        client.put(f"/jobs/{job_id}/photos/img_{i:04d}.jpg", content=JPEG)
    assert client.post(f"/jobs/{job_id}/start", json={"flat_base": False}).status_code == 200
    done = wait_state(client, job_id, "done")
    assert done["report"]["watertight"] is True

    r = client.get(f"/jobs/{job_id}/result.stl", params={"longest_mm": 80})
    assert r.status_code == 200 and r.headers["content-type"] == "model/stl"
    m = trimesh.load(trimesh.util.wrap_as_stream(r.content), file_type="stl")
    assert max(m.extents) == pytest.approx(80)
    assert client.get(f"/jobs/{job_id}/result.exe").status_code == 404

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{job_id}/frames.zip").content))
    assert len(z.namelist()) == 8 and z.read("img_0000.jpg") == JPEG
    # Завършено сканиране може да се обработи отново от кадрите му.
    assert client.post(f"/jobs/{job_id}/start", json={}).status_code == 200
    wait_state(client, job_id, "done")

    assert client.delete(f"/jobs/{job_id}").json()["state"] == "cancelled"
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_failed_job_log_and_list(client, monkeypatch):
    import sys

    from app import ai, pipeline

    def failing_run(job, method, flat_base, set_state, sh, step, tune=None, classic=False):
        set_state("reconstructing")
        sh("camera solving", [sys.executable, "-c", "print('early' + chr(13) + 'final')"])
        raise pipeline.Failed("Only 3 of 90 frames could be placed in 3D.")

    monkeypatch.setattr(ai, "run", failing_run)
    job_id = client.post("/jobs").json()["id"]
    for i in range(8):
        client.put(f"/jobs/{job_id}/photos/img_{i:04d}.jpg", content=JPEG)
    client.post(f"/jobs/{job_id}/start", json={})
    failed = wait_state(client, job_id, "failed")
    assert failed["step"] == "camera solving" and failed["step_started"] > 0

    log = client.get(f"/jobs/{job_id}/log").text
    assert "camera solving" in log and "\nfinal\n" in log and "\nearly" not in log  # презаписванията на напредъка с \r отпадат
    assert log.rstrip().endswith("FAILED: Only 3 of 90 frames could be placed in 3D.")
    assert client.get(f"/jobs/{job_id}/log", params={"lines": 1}).text.startswith("[")

    listed = client.get("/jobs").json()
    assert [j["id"] for j in listed] == [job_id] and listed[0]["state"] == "failed"
    assert client.get("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401


MP4 = b"\0\0\0\x18ftypmp42" + b"1" * 100


def test_video_chunked_upload(client):
    job_id = client.post("/jobs").json()["id"]
    url = f"/jobs/{job_id}/video"
    assert client.put(url, params={"offset": 0}, content=b"not a video at all").status_code == 400
    assert client.put(url, params={"offset": 0}, content=MP4[:50]).json() == {"video_bytes": 50}
    assert client.put(url, params={"offset": 0}, content=MP4[50:]).status_code == 409  # грешна точка за продължаване
    assert client.get(f"/jobs/{job_id}").json()["video_bytes"] == 50
    assert client.post(f"/jobs/{job_id}/start", json={"video_bytes": len(MP4)}).status_code == 400  # незавършено
    assert client.put(url, params={"offset": 50}, content=MP4[50:]).json() == {"video_bytes": len(MP4)}
    assert client.post(f"/jobs/{job_id}/start", json={"video_bytes": len(MP4)}).status_code == 200
    wait_state(client, job_id, "done")
    assert client.put(url, params={"offset": len(MP4)}, content=b"x").status_code == 409  # вече е стартирано


def test_motion_upload(client):
    import json

    from app import jobs, motion

    job_id = client.post("/jobs").json()["id"]
    url = f"/jobs/{job_id}/motion"
    samples = [[0.02 * i, 0.1, 1.2, -0.05] for i in range(100)]
    assert client.put(url, json={"samples": samples}).json() == {"samples": 100}
    assert json.loads((jobs.job_dir(job_id) / motion.MOTION).read_text()) == {"samples": samples}
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{job_id}/frames.zip").content))
    assert z.namelist() == [motion.MOTION]  # с кадрите, за да се възпроизведе сканиране
    assert client.put(url, json={"samples": [[1, 2, 3]]}).status_code == 422  # не t, alpha, beta, gamma
    assert client.put(url, json={"samples": [[0, 0, 0, 0]] * 60_001}).status_code == 422
    client.put(f"/jobs/{job_id}/video", params={"offset": 0}, content=MP4)
    client.post(f"/jobs/{job_id}/start", json={"video_bytes": len(MP4)})
    wait_state(client, job_id, "done")
    assert client.put(url, json={"samples": samples}).status_code == 200  # завършено сканиране може да се направи отново
    assert client.put("/jobs/" + "0" * 32 + "/motion", json={"samples": samples}).status_code == 404


def test_video_size_limit(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "MAX_VIDEO_BYTES", 64)
    job_id = client.post("/jobs").json()["id"]
    assert client.put(f"/jobs/{job_id}/video", params={"offset": 0}, content=MP4).status_code == 413
    assert client.get(f"/jobs/{job_id}").json()["video_bytes"] == 0


def test_limits(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "MAX_JOBS", 2)
    assert client.post("/jobs").status_code == 200
    assert client.post("/jobs").status_code == 200
    assert client.post("/jobs").status_code == 503


def test_finished_test_copies_do_not_fill_the_limit(client, monkeypatch):
    from app import jobs, main

    monkeypatch.setattr(main, "MAX_JOBS", 2)
    for _ in range(3):
        jobs.write(jobs.create(test=True), state="done")
    assert client.post("/jobs").status_code == 200
    jobs.create(test=True)  # still uploading: counts
    assert client.post("/jobs").status_code == 503


def test_cleanup_removes_abandoned_uploads(client):
    from app import jobs

    job_id = client.post("/jobs").json()["id"]
    jobs.write(job_id, created=time.time() - 2 * 86400)
    jobs.cleanup()
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_neighbor_lines_pair_frames_a_few_degrees_apart():
    from app import mesh as M
    from app.pipeline import neighbor_lines

    # 24 кадъра в кръг през 15°, насочени към центъра.
    ang = np.radians(np.arange(24) * 15)
    z = -np.c_[np.cos(ang), np.sin(ang), np.zeros(24)]
    x = np.cross(z, [0, 0, 1.0])
    R = np.stack([x, np.cross(z, x), z], axis=1)
    views = M.Views(R, np.zeros((24, 3)), np.ones((24, 4)), [str(i) for i in range(24)])
    lines = neighbor_lines(views, [100 + i for i in range(24)])
    assert lines[0].split()[0] == "100"
    assert set(lines[0].split()[1:]) == {"101", "123", "102", "122"}  # на 15° и 30°; 45° е твърде далеч


def test_read_views(tmp_path):
    from app.pipeline import read_views

    # Камера в (0, 0, -5), гледаща по +Z, без завъртане: t = -R C = (0, 0, 5).
    with open(tmp_path / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I7dI", 1, 1, 0, 0, 0, 0, 0, 5, 7))
        f.write(b"img_0001.jpg\0")
        f.write(struct.pack("<Q", 2) + b"\0" * 48)
    with open(tmp_path / "cameras.bin", "wb") as f:  # PINHOLE, както го записва image_undistorter
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 7, 1, 1600, 1200) + struct.pack("<4d", 1300, 1310, 800, 600))
    views = read_views(tmp_path)
    assert views.names == ["img_0001.jpg"]
    np.testing.assert_allclose(views.K, [[1300, 1310, 800, 600]])
    np.testing.assert_allclose(views.centers, [[0, 0, -5]])
    np.testing.assert_allclose(views.dirs, [[0, 0, 1]])
    np.testing.assert_allclose(views.ups, [[0, -1, 0]])
