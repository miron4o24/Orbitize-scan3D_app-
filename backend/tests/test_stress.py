"""The stress test (app/stress.py): the element, beam formulas on a column, and the endpoints."""

import json

import numpy as np
import trimesh

# First: it sets the API token the app reads when it is imported.
from tests.test_accounts import client, finished_scan, sign_in  # noqa: F401, I001 (client: fixture)
from tests.test_community import setup_two
from app import stress as S

COLUMN = trimesh.creation.box(extents=[10, 10, 100]).apply_translation([0, 0, 50])  # mm, standing on z = 0


def test_brick_element_is_symmetric_with_only_rigid_motions_free():
    K = S.element_stiffness(S.NU)
    assert np.allclose(K, K.T)
    assert (np.abs(np.linalg.eigvalsh(K)) < 1e-9).sum() == 6  # 3 shifts and 3 turns cost no energy


def test_cubes_by_their_centres():
    occ, origin = S.voxels(COLUMN, 1.25)
    assert occ.shape == (8, 8, 80) and occ.all()  # 10 mm is exactly 8 cubes of 1.25: none more, none fewer
    ring = trimesh.creation.annulus(r_min=20, r_max=30, height=10).apply_translation([0, 0, 5])
    occ, origin = S.voxels(ring, 2.0)
    centre = np.array(occ.shape) // 2
    assert not occ[centre[0], centre[1], :].any() and occ[:, centre[1], :].any()  # the hole stays a hole


def test_column_matches_beam_formulas(monkeypatch):
    monkeypatch.setattr(S, "CELLS", 80)  # 1.25 mm cubes: the column is exactly 8 across
    F, b, H = S.G, 0.01, 0.1  # 1 kg; m
    top, _, _ = S.analyse(COLUMN, "top")
    assert abs(top["max_mpa"] / (F / b**2 / 1e6) - 1) < 0.1  # straight compression: F / area
    side, _, _ = S.analyse(COLUMN, "side")
    lever = H - (S.BASE_LAYERS + 1) * side["pitch_mm"] / 1000
    assert abs(side["max_mpa"] / (F * lever / (b**3 / 6) / 1e6) - 1) < 0.15  # bending: M / section modulus
    assert side["weak"][1] < 10  # at the foot (GLB y is up)
    drop, _, _ = S.analyse(COLUMN, "drop")
    assert abs(drop["sag_m"] / (S.RHO_REF * S.G * H**2 / (3 * S.E_REF)) - 1) < 0.05


def test_stress_endpoints_and_their_points(client):
    a, _, b, _ = setup_two(client)
    job = finished_scan(client, a)
    r = client.get(f"/jobs/{job}/stress/side", headers=a)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["case"] == "side" and s["max_mpa"] > 0 and len(s["weak"]) == 3 and "model_mtime" not in s
    glb = client.get(f"/jobs/{job}/stress/side/model.glb", headers=a).content
    assert glb[:4] == b"glTF"
    doc = json.loads(glb[20 : 20 + int.from_bytes(glb[12:16], "little")])
    assert "_STRESS" in doc["meshes"][0]["primitives"][0]["attributes"]
    assert client.get(f"/jobs/{job}/stress/bend", headers=a).status_code == 404
    assert client.get(f"/jobs/{job}/stress/top", headers=b).status_code == 404  # not Bob's, not posted

    before = client.get("/me/progress", headers=a).json()["points"]
    client.get(f"/jobs/{job}/stress/side", headers=a)  # again: no more points
    assert client.get("/me/progress", headers=a).json()["points"] == before

    client.post(f"/me/scans/{job}/publish", json={"title": "Кула"}, headers=a)
    assert client.get(f"/community/{job}/stress/top", headers=b).status_code == 200  # anyone may test a post
    assert client.get(f"/community/{job}/stress/top/model.glb", headers=b).content[:4] == b"glTF"
    kinds = [e["kind"] for e in client.get("/me/progress", headers=b).json()["recent"]]
    assert kinds == ["stress"]
