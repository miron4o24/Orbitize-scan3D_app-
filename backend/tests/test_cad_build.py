"""Инженерният режим: от описание до твърдо тяло (app/cad/build.py). Размерите и обемът трябва да са точно тези от
описанието (до 0,001 мм и 0,01%), защото моделът се строи само от измерените числа. Нужен е CadQuery."""

import math

import pytest

pytest.importorskip("cadquery")

from app.cad import build as B  # noqa: E402
from app.cad import design as D  # noqa: E402


def measured(d: dict) -> dict:
    """Всички размери "измерени" със стойността си (както след въпросите)."""
    return D.set_values(d, {i: dim["v"] for i, dim in d["dims"].items()})


def test_plate_with_rounded_corners_hole_and_pocket(tmp_path):
    d = D.new("block", 60, 30, 10)
    r = D.add_dim(d, "r", 5, "Радиус на ъглите", "Радиус")
    d["views"]["top"]["pts"] = [[u, v, r] for u, v, _ in d["views"]["top"]["pts"]]
    d = D.add_feature(d, {"t": "hole", "face": "top", "through": True, "centered": [True, True]})
    d = D.add_feature(d, {"t": "rect", "face": "top", "through": False, "centered": [False, True]})
    hole, pocket = d["features"]
    vals = {hole["d"]: 6.0, pocket["w"]: 10.0, pocket["h"]: 8.0, pocket["depth"]: 3.0,
            d["levels"][pocket["u"]["lvl"]]["dim"]: 5.0}  # fmt: skip
    d = measured(D.set_values(d, vals))
    rep = B.build(d, tmp_path)
    assert rep["size_mm"] == [60.0, 30.0, 10.0]
    expect = (60 * 30 - (4 - math.pi) * 25) * 10 - math.pi * 9 * 10 - 10 * 8 * 3
    assert rep["volume_mm3"] == pytest.approx(expect, rel=1e-4)
    for f in ("part.py", "part.step", "part.stl", "part.glb", "design.json"):
        assert (tmp_path / f).stat().st_size > 0
    # Кодът е самостоятелен: изпълнен отново дава същото тяло.
    shape = B.solid((tmp_path / "part.py").read_text(encoding="utf-8"))
    assert shape.Volume() == pytest.approx(expect, rel=1e-4)


def test_step_from_the_side_view(tmp_path):
    d = D.new("block", 60, 30, 10)
    x = D.level_for(d, "x", 40, "Стъпало", "стъпалото", view="side")
    z = D.level_for(d, "z", 6, "Стъпало", "стъпалото", view="side")
    d["views"]["side"]["pts"] = [["x0", "z0", None], ["xL", "z0", None], ["xL", z, None], [x, z, None],
                                 [x, "zH", None], ["x0", "zH", None]]  # fmt: skip
    # 40 от 60 и 6 от 10: мерят се от по-близкия край (десния и горния): стъпалото е 20 дълго и 4 надолу.
    assert d["levels"][x]["from"] == "end" and d["levels"][z]["from"] == "end"
    d = measured(D.set_values(d, {d["levels"][x]["dim"]: 20.0, d["levels"][z]["dim"]: 4.0}))
    rep = B.build(d, tmp_path)
    assert rep["size_mm"] == [60.0, 30.0, 10.0]
    assert rep["volume_mm3"] == pytest.approx(60 * 30 * 10 - 20 * 30 * 4, rel=1e-4)


def test_turned_shaft_with_bore(tmp_path):
    d = D.new("turned", 50, 20, 20)
    x = D.level_for(d, "x", 30, "Рамо", "рамото", view="profile")
    d2 = D.add_dim(d, "d", 12, "Диаметър 2", "Диаметър", kind="diameter")
    d["levels"]["Lr1"] = {"axis": "r", "dim": d2, "from": "diameter"}
    d["views"]["profile"]["pts"] = [["x0", "r0", None], ["xL", "r0", None], ["xL", "Lr1", None], [x, "Lr1", None],
                                    [x, "rD", None], ["x0", "rD", None]]  # fmt: skip
    d = D.add_feature(d, {"t": "hole", "face": "left", "through": True})
    d = measured(D.set_values(d, {d["features"][0]["d"]: 6.0}))
    rep = B.build(d, tmp_path)
    assert rep["size_mm"] == pytest.approx([50, 20, 20], abs=1e-3)
    expect = math.pi * (100 * 30 + 36 * 20) - math.pi * 9 * 50
    assert rep["volume_mm3"] == pytest.approx(expect, rel=1e-3)


def test_fillet_too_big_is_explained(tmp_path):
    d = D.add_feature(D.new("block", 20, 10, 4), {"t": "fillet", "edges": "top"})
    d = measured(D.set_values(d, {d["features"][0]["r"]: 6.0}))
    with pytest.raises(D.DesignError, match="Заоблянето"):
        B.build(d, tmp_path)


@pytest.mark.parametrize("bad", [
    {"kind": "block", "dims": {"cq": {"v": 1}}},
    {"kind": "block", "dims": {"L; import os": {"v": 1}}},
    {"kind": "evil"},
])  # fmt: skip
def test_unknown_or_unsafe_designs_are_refused(bad):
    with pytest.raises(D.DesignError):
        B.clean(bad)


def test_names_cannot_break_out_of_comments(tmp_path):
    d = measured(D.new("block", 10, 10, 10))
    d["dims"]["L"]["name"] = "x\nimport os\nos.system('echo hacked')"
    text = B.code(B.clean(d))
    assert "\nimport os" not in text and "\nos.system" not in text


def test_two_parts_shown_in_their_own_colours(tmp_path, monkeypatch):
    """Капачка и тяло (граница по диагонал): прегледът (part.glb, с височината нагоре като всеки glTF) и картинката в
    историята показват двете части в два ясно различни цвята, без смесване там, където се допират."""
    import json
    import shutil
    import struct

    import numpy as np
    import trimesh

    from app import render

    d = measured(D.add_split(D.new("block", 50, 20, 8), 35.0, 31.0))
    out = tmp_path / "cad" / "out"
    assert B.build(d, out)["solids"] == 2
    shown = trimesh.load(out / "part.glb", force="mesh", process=False)
    assert np.allclose(shown.extents, [50, 8, 20], atol=0.01)  # височината по Y: прегледът е изправен като историята
    raw = (out / "part.glb").read_bytes()  # цветовете на върховете (COLOR_0) от самия файл
    n = struct.unpack("<I", raw[12:16])[0]
    gltf = json.loads(raw[20:20 + n])
    acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]]
    start = gltf["bufferViews"][acc["bufferView"]].get("byteOffset", 0) + acc.get("byteOffset", 0)
    k = {"VEC3": 3, "VEC4": 4}[acc["type"]]
    colors = np.frombuffer(raw[20 + n + 8:], np.float32, acc["count"] * k, start).reshape(-1, k)
    assert len(np.unique(colors.round(4), axis=0)) == 2
    shutil.copy(out / "display.ply", tmp_path / "model.ply")  # моделът в историята (service.build)
    seen = []
    sample = render.surface

    def spy(job, mesh, count):
        pts, col, nrm = sample(job, mesh, count)
        seen.append(col)
        return pts, col, nrm

    monkeypatch.setattr(render, "surface", spy)
    render.picture(tmp_path)
    colours = {tuple(c) for c in np.unique(seen[0].round(), axis=0)}
    assert colours == {tuple(float(v) for v in c[:3]) for c in B.PART_COLORS[:2]}
