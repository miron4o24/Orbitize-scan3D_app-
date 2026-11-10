"""Инженерният режим през API: снимки -> прочетено описание -> мерки -> построен детайл и файловете. Изрязването на
снимките (IS-Net) се заменя с известни очертания, за да е бързо и без модела; самото четене е в
test_cad_read.py."""

import io
import math
import os

import numpy as np
import pytest

pytest.importorskip("cadquery")
os.environ["API_TOKEN"] = "test-token-123"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import jobs, main
    from app.cad import service, vision

    monkeypatch.setattr(jobs, "DATA", tmp_path)
    monkeypatch.setattr(service, "session", lambda: None)

    def fake_read(sess, path, U, V, prefer_long_u=None):
        """Плоча с два ъгъла R5 вдясно и отвор Ø6 на 15 мм от левия край (отгоре); отстрани правоъгълник."""
        if path.stem == "top":
            t = np.linspace(-math.pi / 2, math.pi / 2, 40)
            outer = np.r_[[[0, 0]], np.c_[U - 5 + 5 * np.cos(t[:20]), 5 + 5 * np.sin(t[:20])],
                          np.c_[U - 5 + 5 * np.cos(t[20:]), V - 5 + 5 * np.sin(t[20:])], [[0, V]]]  # fmt: skip
            outer = densify(outer)
            a = np.linspace(0, 2 * math.pi, 60, endpoint=False)
            holes = [np.c_[15 + 3 * np.cos(a), V / 2 + 3 * np.sin(a)]]
        else:
            outer, holes = densify(np.array([[0, 0], [U, 0], [U, V], [0, V]], float)), []
        return vision.View(outer, holes, U, V, 0.0, (400.0, 300.0), 100.0, 500.0, U / 600, V / 300, (800, 600))

    monkeypatch.setattr(vision, "read", fake_read)
    with TestClient(main.app, headers={"Authorization": "Bearer test-token-123"}) as c:
        yield c


def densify(poly: np.ndarray, step: float = 0.1) -> np.ndarray:
    out = []
    for a, b in zip(poly, np.r_[poly[1:], poly[:1]]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(n)]
    return np.asarray(out)


def jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (800, 600), (150, 110, 80)).save(buf, "JPEG")
    return buf.getvalue()


def test_read_measure_build_and_download(client):
    job = client.post("/jobs").json()["id"]
    for v in ("top", "side"):
        assert client.put(f"/jobs/{job}/photos/{v}.jpg", content=jpeg()).status_code == 200
    assert client.post(f"/jobs/{job}/cad/read", json={"L": 60, "W": 30, "H": 8}).status_code == 200
    got = client.post(f"/jobs/{job}/cad/read", json={"L": 60, "W": 30, "H": 8}).json()
    kinds = [i["kind"] for i in got["items"]]
    assert "hole" in kinds and got["design"]["kind"] == "block"
    asked = {q["id"]: q for q in got["questions"]}
    hole = next(f for f in got["design"]["features"] if f["t"] == "hole")
    assert asked[hole["d"]]["est"] == pytest.approx(6, abs=0.05)
    # Картинките към въпросите и снимката с очертанието.
    for q in got["questions"]:
        r = client.get(f"/jobs/{job}/cad/question/{q['id']}.png")
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get(f"/jobs/{job}/cad/photo/top.jpg").content[:3] == b"\xff\xd8\xff"
    # Мерките: отворът 6,2 с шублера, другото по оценката.
    values = {q["id"]: q["est"] for q in got["questions"] if q["est"] is not None}
    values[hole["d"]] = 6.2
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"values": values}}).json()
    assert got["design"]["dims"][hole["d"]]["src"] == "measured"
    # Без размер: обяснение, а не грешка на сървъра.
    bad = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"values": {"L": -1}}})
    assert bad.status_code == 400 and "между" in bad.json()["detail"]
    done = client.post(f"/jobs/{job}/cad/build", json={"design": got["design"]}).json()
    assert done["state"] == "done"
    cad = done["report"]["cad"]
    assert cad["size_mm"] == [60.0, 30.0, 8.0]
    for name in ("part.step", "part.py", "part.stl", "part.glb", "drawing.png"):
        r = client.get(f"/jobs/{job}/cad/file/{name}")
        assert r.status_code == 200 and len(r.content) > 100
    assert b"ISO-10303" in client.get(f"/jobs/{job}/cad/file/part.step").content[:200]
    assert client.get(f"/jobs/{job}/cad/file/../status.json").status_code == 404
    assert client.get(f"/jobs/{job}/result.glb").status_code == 200  # прегледът в приложението


def test_add_remove_and_through(client):
    job = client.post("/jobs").json()["id"]
    for v in ("top", "side"):
        client.put(f"/jobs/{job}/photos/{v}.jpg", content=jpeg())
    got = client.post(f"/jobs/{job}/cad/read", json={"L": 60, "W": 30, "H": 8}).json()
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {
        "add": {"t": "rect", "face": "front", "through": False, "centered": [True, False]}}}).json()  # fmt: skip
    rect = next(f for f in got["design"]["features"] if f["t"] == "rect")
    assert rect["axis"] == "y" and rect.get("depth")
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"through": [rect["id"], True]}}).json()
    assert not next(f for f in got["design"]["features"] if f["id"] == rect["id"]).get("depth")
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"remove": rect["id"]}}).json()
    assert all(f["t"] != "rect" for f in got["design"]["features"])
    assert all(not i.startswith("w") for i in got["design"]["dims"])  # размерите му също
    # Без снимка отстрани: ясно съобщение.
    other = client.post("/jobs").json()["id"]
    client.put(f"/jobs/{other}/photos/top.jpg", content=jpeg())
    r = client.post(f"/jobs/{other}/cad/read", json={"L": 60, "W": 30, "H": 8})
    assert r.status_code == 400 and "отстрани" in r.json()["detail"]
