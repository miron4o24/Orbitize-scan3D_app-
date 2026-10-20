import os
import struct
import time

import numpy as np
import pytest
import trimesh

os.environ["API_TOKEN"] = "test-token-123"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import jobs, main, pipeline

    monkeypatch.setattr(jobs, "DATA", tmp_path)

    def fake_run(job, flat_base, set_state, sh):
        set_state("reconstructing")
        trimesh.creation.box(extents=[1, 2, 4]).apply_translation([0, 0, 2]).export(job / "model.ply")
        return {"dimensions_mm": [25, 50, 100], "triangles": 12, "watertight": True}

    monkeypatch.setattr(pipeline, "run", fake_run)
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
    assert client.put(f"{url}/img_0001.jpg", content=JPEG).status_code == 200  # re-upload is idempotent
    assert client.put(f"{url}/notes.txt", content=JPEG).status_code == 400
    assert client.put(f"{url}/..%2Fstatus.json", content=JPEG).status_code in (400, 404)
    assert client.put(f"{url}/img_0002.jpg", content=b"not a jpeg").status_code == 400
    assert client.get(f"/jobs/{job['id']}").json()["received"] == ["img_0001.jpg"]
    assert client.get("/jobs/../etc").status_code == 404


def test_full_lifecycle(client):
    job_id = client.post("/jobs").json()["id"]
    assert client.post(f"/jobs/{job_id}/start", json={}).status_code == 400  # too few photos
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

    assert client.delete(f"/jobs/{job_id}").json()["state"] == "cancelled"
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_limits(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "MAX_JOBS", 2)
    assert client.post("/jobs").status_code == 200
    assert client.post("/jobs").status_code == 200
    assert client.post("/jobs").status_code == 503


def test_cleanup_removes_abandoned_uploads(client):
    from app import jobs

    job_id = client.post("/jobs").json()["id"]
    jobs.write(job_id, created=time.time() - 2 * 86400)
    jobs.cleanup()
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_read_images_bin(tmp_path):
    from app.pipeline import read_images_bin

    # Camera at (0, 0, -5) looking along +Z, identity rotation: t = -R C = (0, 0, 5).
    path = tmp_path / "images.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I7dI", 1, 1, 0, 0, 0, 0, 0, 5, 1))
        f.write(b"img_0001.jpg\0")
        f.write(struct.pack("<Q", 2) + b"\0" * 48)
    centers, dirs, ups = read_images_bin(path)
    np.testing.assert_allclose(centers, [[0, 0, -5]])
    np.testing.assert_allclose(dirs, [[0, 0, 1]])
    np.testing.assert_allclose(ups, [[0, -1, 0]])
