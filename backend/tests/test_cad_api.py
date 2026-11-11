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

    def fake_segment(sess, path):
        """Плоча 60 x 30 x 8 мм (10 пиксела на мм): отгоре два ъгъла R5 вдясно и отвор Ø6 на 15 мм от левия край,
        отпред правоъгълник. Както я дава vision.segment: пиксели, y надолу."""
        U, V = (60.0, 30.0) if path.stem == "top" else (60.0, 8.0)
        if path.stem == "top":
            t = np.linspace(-math.pi / 2, math.pi / 2, 40)
            outer = np.r_[[[0, 0]], np.c_[U - 5 + 5 * np.cos(t[:20]), 5 + 5 * np.sin(t[:20])],
                          np.c_[U - 5 + 5 * np.cos(t[20:]), V - 5 + 5 * np.sin(t[20:])], [[0, V]]]  # fmt: skip
            outer = densify(outer)
            a = np.linspace(0, 2 * math.pi, 60, endpoint=False)
            holes = [np.c_[15 + 3 * np.cos(a), V / 2 + 3 * np.sin(a)]]
        else:
            outer, holes = densify(np.array([[0, 0], [U, 0], [U, V], [0, V]], float)), []

        def px(p):
            return (np.c_[100 + p[:, 0] * 10, 500 - p[:, 1] * 10]).tolist()

        return {"outer": px(outer), "holes": [px(h) for h in holes], "size": [800, 600]}

    monkeypatch.setattr(vision, "segment", fake_segment)
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
    for v in ("top", "front"):
        assert client.put(f"/jobs/{job}/photos/{v}.jpg", content=jpeg()).status_code == 200
    assert client.post(f"/jobs/{job}/cad/prepare").json()["ready"] == ["front", "top"]
    # Без мерки: кои общи размери и на коя снимка, всеки с картинка.
    plan = client.post(f"/jobs/{job}/cad/read", json={}).json()
    assert [(q["id"], q["photo"]) for q in plan["overall"]] == [("L", "top"), ("W", "top"), ("H", "front")]
    for q in plan["overall"]:
        r = client.get(f"/jobs/{job}/cad/question/{q['id']}.png")
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    got = client.post(f"/jobs/{job}/cad/read", json={"L": 60, "W": 30, "H": 8}).json()
    kinds = [i["kind"] for i in got["items"]]
    assert "hole" in kinds and got["design"]["kind"] == "block"
    asked = {q["id"]: q for q in got["questions"]}
    hole = next(f for f in got["design"]["features"] if f["t"] == "hole")
    assert asked[hole["d"]]["est"] == pytest.approx(6, abs=0.05)
    assert got["views"] == ["top", "front"]
    # Картинките към въпросите (върху снимката) и снимките с очертанието.
    for q in got["questions"]:
        r = client.get(f"/jobs/{job}/cad/question/{q['id']}.png")
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    for v in ("top", "front"):
        assert client.get(f"/jobs/{job}/cad/photo/{v}.jpg").content[:3] == b"\xff\xd8\xff"
    # Мерките: отворът 6,2 с шублера, другото по оценката.
    values = {q["id"]: q["est"] for q in got["questions"] if q["est"] is not None}
    values[hole["d"]] = 6.2
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"values": values}}).json()
    assert got["design"]["dims"][hole["d"]]["src"] == "measured"
    # Без размер: обяснение, а не грешка на сървъра.
    bad = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"values": {"L": -1}}})
    assert bad.status_code == 400 and "между" in bad.json()["detail"]
    # Обръщане на снимката отпред (по името на снимката): описанието се сглобява наново.
    flipped = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"flip": ["front", 1]}}).json()
    assert flipped["design"]["flips"] == {"side": [False, True]}
    done = client.post(f"/jobs/{job}/cad/build", json={"design": got["design"]}).json()
    assert done["state"] == "done"
    cad = done["report"]["cad"]
    assert cad["size_mm"] == [60.0, 30.0, 8.0]
    for name in ("part.step", "part.py", "part.stl", "part.glb", "drawing.png"):
        r = client.get(f"/jobs/{job}/cad/file/{name}")
        assert r.status_code == 200 and len(r.content) > 100
    assert b"ISO-10303" in client.get(f"/jobs/{job}/cad/file/part.step").content[:200]
    assert client.get(f"/jobs/{job}/cad/file/../status.json").status_code == 404
    glb = client.get(f"/jobs/{job}/result.glb")  # прегледът в приложението
    assert glb.status_code == 200
    import json
    import struct

    n = struct.unpack("<I", glb.content[12:16])[0]
    attrs = json.loads(glb.content[20:20 + n])["meshes"][0]["primitives"][0]["attributes"]
    # Остри ръбове и цвят: с общи нормали през ръбовете (както при сканиране) детайлът изглеждаше размазан.
    assert {"NORMAL", "COLOR_0"} <= set(attrs)
    import numpy as np
    import trimesh

    shown = trimesh.load(io.BytesIO(glb.content), file_type="glb", force="mesh", process=False)
    corner = np.isclose(np.abs(shown.vertex_normals).max(1), 1.0, atol=1e-3).mean()
    assert corner > 0.5  # по равните стени нормалите са точно перпендикулярни, не наклонени към ръбовете
    # Детайл, построен преди display.ply: острите ръбове се разделят при изпращане.
    from app import jobs

    (jobs.job_dir(job) / "cad" / "out" / "display.ply").unlink()
    old = client.get(f"/jobs/{job}/result.glb")
    assert old.status_code == 200
    shown = trimesh.load(io.BytesIO(old.content), file_type="glb", force="mesh", process=False)
    assert np.isclose(np.abs(shown.vertex_normals).max(1), 1.0, atol=1e-3).mean() > 0.5


def test_add_remove_and_through(client):
    job = client.post("/jobs").json()["id"]
    for v in ("top", "front"):
        client.put(f"/jobs/{job}/photos/{v}.jpg", content=jpeg())
    got = client.post(f"/jobs/{job}/cad/read", json={"L": 60, "W": 30, "H": 8}).json()
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {
        "add": {"t": "rect", "face": "front", "through": False, "centered": [True, False]}}}).json()  # fmt: skip
    rect = next(f for f in got["design"]["features"] if f["t"] == "rect")
    assert rect["axis"] == "y" and rect.get("depth")
    op = {"through": [rect["id"], True]}
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": op}).json()
    assert not next(f for f in got["design"]["features"] if f["id"] == rect["id"]).get("depth")
    got = client.post(f"/jobs/{job}/cad/edit", json={"design": got["design"], "op": {"remove": rect["id"]}}).json()
    assert all(f["t"] != "rect" for f in got["design"]["features"])
    assert all(not i.startswith("w") for i in got["design"]["dims"])  # размерите му също
    # Само снимка отгоре: става (плоча), височината без снимка е по схема.
    flat = client.post("/jobs").json()["id"]
    client.put(f"/jobs/{flat}/photos/top.jpg", content=jpeg())
    plan = client.post(f"/jobs/{flat}/cad/read", json={}).json()
    assert [(q["id"], q["photo"]) for q in plan["overall"]] == [("L", "top"), ("W", "top"), ("H", None)]
    assert client.get(f"/jobs/{flat}/cad/question/H.png").status_code == 200
    # Без снимка отгоре: ясно съобщение.
    other = client.post("/jobs").json()["id"]
    client.put(f"/jobs/{other}/photos/front.jpg", content=jpeg())
    r = client.post(f"/jobs/{other}/cad/read", json={})
    assert r.status_code == 400 and "отгоре" in r.json()["detail"]
