"""Инженерният режим: локалният CAD модел от елементи (app/cad_local.py) и проверката на код от Claude
(app/cad_check.py). Нужен е CadQuery."""

import json

import numpy as np
import pytest
import trimesh

cq = pytest.importorskip("cadquery")

from app import cad_check, cad_local  # noqa: E402


def bracket() -> trimesh.Trimesh:
    """Плоча 60 x 30 x 5 с два отвора Ø6, издатина Ø16 до 20 мм с отвор Ø8 и напречен отвор Ø5, като сканиране:
    в друг мащаб, завъртяна и с плавен шум по повърхността."""
    b = (cq.Workplane("XY").box(60, 30, 5, centered=(True, True, False))
         .faces(">Z").workplane().pushPoints([(-22, 0), (22, 0)]).hole(6.0))  # fmt: skip
    b = b.union(cq.Workplane("XY").workplane(offset=5).circle(8).extrude(15))
    b = b.cut(cq.Workplane("XY").circle(4).extrude(20))
    b = b.cut(cq.Workplane("YZ").workplane(offset=-30).center(0, 14).circle(2.5).extrude(60))
    v, f = b.val().tessellate(0.02, 0.2)
    m = trimesh.Trimesh(np.array([p.toTuple() for p in v]), np.array(f))
    m = m.subdivide_to_size(1.0)
    m.merge_vertices()
    x = m.vertices
    m.vertices = x + m.vertex_normals * (0.06 * (np.sin(x[:, 0] / 3) + np.sin(x[:, 1] / 2.5 + 1) + np.sin(x[:, 2] / 2 + 2)))[:, None]
    m.apply_transform(trimesh.transformations.rotation_matrix(0.35, [0, 0, 1]))
    return m.apply_scale(0.0137)


def test_bracket_becomes_features(tmp_path):
    bracket().export(tmp_path / "scan.stl")
    meas = {"length": 60, "width": 30, "height": 20, "features": [{"kind": "hole", "value": 6}, {"kind": "hole", "value": 8},
            {"kind": "hole", "value": 5}, {"kind": "thickness", "value": 5}]}  # fmt: skip
    rep = cad_local.run(tmp_path / "scan.stl", meas, tmp_path / "out")
    assert rep["size_mm"] == [60.0, 30.0, 20.0]
    els = " | ".join(rep["elements"])
    assert "Блок 60.0 × 30.0 × 5.0" in els
    assert els.count("Отвор Ø6.0") == 2 and "Отвор Ø8.0" in els and "Отвор Ø5.0" in els
    assert sum(e.startswith("Цилиндър") for e in rep["elements"]) == 1
    assert rep["deviation_mm"]["mean"] < 0.3
    step = (tmp_path / "out" / "part.step").read_text(errors="replace")
    assert step.startswith("ISO-10303-21") and "CYLINDRICAL_SURFACE" in step
    # Кодът е самостоятелен: изпълнен отново дава същото тяло.
    shape = cad_check.run_code((tmp_path / "out" / "part.py").read_text(encoding="utf-8"))
    bb = shape.BoundingBox()
    assert abs(bb.xlen - 60) < 0.01 and abs(bb.zlen - 20) < 0.01


@pytest.mark.parametrize("code", ["import os\nresult = os.getcwd()", "result = open('x').read()",
                                  "import cadquery as cq\nresult = cq.Workplane().__class__",
                                  "from subprocess import run\nresult = run"])  # fmt: skip
def test_unsafe_code_is_refused(tmp_path, code):
    (tmp_path / "part.py").write_text(code)
    trimesh.creation.box(extents=[10, 10, 10]).export(tmp_path / "scan.stl")
    rep = cad_check.check(tmp_path / "part.py", tmp_path / "scan.stl", {"length": 10}, tmp_path / "out")
    assert not rep["ok"] and "not allowed" in rep["error"]


def test_claude_code_check(tmp_path):
    (tmp_path / "part.py").write_text("import cadquery as cq\nL = 10\nresult = cq.Workplane().box(L, 10, 10).translate((0, 0, 5))\n")
    trimesh.creation.box(extents=[10, 10, 10]).apply_translation([0, 0, 5]).export(tmp_path / "scan.stl")
    rep = cad_check.check(tmp_path / "part.py", tmp_path / "scan.stl", {"length": 10, "height": 10}, tmp_path / "out")
    assert rep["ok"] and rep["off_from_measured_mm"] == {"length": 0.0, "height": 0.0}
    assert rep["deviation_from_scan_mm"]["mean"] < 0.05
    assert (tmp_path / "out" / "compare.png").exists() and (tmp_path / "out" / "part.step").exists()
    assert json.loads((tmp_path / "out" / "check.json").read_text())["ok"]


def test_plan_env_never_passes_a_paid_key(monkeypatch):
    from app import cad_claude

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-paid")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "plan")
    env = cad_claude.plan_env()
    assert "ANTHROPIC_API_KEY" not in env and env["CLAUDE_CODE_OAUTH_TOKEN"] == "plan"
