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


def hook_outline():
    """Кука като истинската закачалка (54 x 41, рамо 5 мм): дълъг крак, полукръг и къс крак със заоблен край."""
    from shapely.geometry import LineString

    path = [(2.5, 0.0), (2.5, 28.0)] + [(20.5 - 18 * math.cos(t), 28 + 18 * math.sin(t))
                                        for t in [i * math.pi / 40 for i in range(41)]] + [(38.5, 16.0)]  # fmt: skip
    poly = LineString(path).buffer(2.5, cap_style="flat", quad_segs=16)
    return poly


def test_hook_follows_the_photo_as_a_flat_part(tmp_path):
    """Кука (54 x 41): от прави и дъги или гладка крива, но по очертанието на снимката и без въпроси за "точки от
    формата". Плосък детайл: само очертанието отгоре, една дебелина и заобляне на ръбовете отгоре и отдолу."""
    import numpy as np

    from app.cad import assemble as A

    poly = hook_outline()
    x0, y0, x1, y1 = poly.bounds
    U, V, H = x1 - x0, y1 - y0, 4.0
    outer = np.asarray(poly.exterior.coords)[:-1] - [x0, y0]
    d = A.assemble({"top": {"outer": outer, "holes": [], "U": U, "V": V}}, U, V, H, flat=True)
    assert not any("формата" in q["name"] for q in D.questions(d))
    rep = B.build(measured(d), tmp_path / "sharp")
    assert rep["volume_mm3"] == pytest.approx(poly.area * H, rel=0.02)
    d = D.add_feature(d, {"t": "fillet", "edges": "flat"})
    d = D.set_values(d, {d["features"][-1]["r"]: 1.5})
    rounded = B.build(measured(d), tmp_path / "round")
    assert 0.85 * poly.area * H < rounded["volume_mm3"] < 0.97 * poly.area * H
    assert rounded["size_mm"] == pytest.approx([U, V, H], abs=0.05)


def test_free_shape_is_a_smooth_curve_not_chopped_lines(tmp_path):
    """Вълниста форма, която не е от прави и дъги: гладка крива (сплайн) през точките от снимката, а не начупена
    линия, и точките не се питат."""
    import numpy as np
    from shapely.geometry import Polygon

    from app.cad import assemble as A

    t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    r = 20 + 3 * np.sin(5 * t) + 1.5 * np.cos(3 * t)
    poly = Polygon(np.c_[r * np.cos(t), r * np.sin(t)])
    x0, y0, x1, y1 = poly.bounds
    U, V = x1 - x0, y1 - y0
    outer = np.asarray(poly.exterior.coords)[:-1] - [x0, y0]
    d = A.assemble({"top": {"outer": outer, "holes": [], "U": U, "V": V}}, U, V, 5.0, flat=True)
    top = d["views"]["top"]
    assert top.get("curve") and len(top["pts"]) >= 40 and not top["corners"]
    assert all(q["id"] in ("L", "W", "H") for q in D.questions(d))
    rep = B.build(measured(d), tmp_path)
    assert rep["volume_mm3"] == pytest.approx(poly.area * 5.0, rel=0.01)
    assert "spline" in (tmp_path / "part.py").read_text(encoding="utf-8")


def test_zero_means_no_step_no_hole_and_a_sharp_corner(tmp_path):
    """0 за стъпало, отвор, заобляне или радиус: просто го няма (кутия), а не грешка."""
    d = D.new("block", 60, 30, 10)
    x = D.level_for(d, "x", 40, "Стъпало", "стъпалото", view="side")
    z = D.level_for(d, "z", 6, "Стъпало", "стъпалото", view="side")
    r = D.add_dim(d, "r", 2, "Радиус на ъгъла", "Радиус")
    d["views"]["side"]["pts"] = [["x0", "z0", None], ["xL", "z0", None], ["xL", z, r], [x, z, None],
                                 [x, "zH", None], ["x0", "zH", None]]  # fmt: skip
    d = D.add_feature(d, {"t": "hole", "face": "top", "through": True, "centered": [True, True]})
    d = D.add_feature(d, {"t": "fillet", "edges": "vertical"})
    hole, fillet = d["features"]
    d = measured(D.set_values(d, {d["levels"][x]["dim"]: 0.0, d["levels"][z]["dim"]: 0.0, r: 0.0, hole["d"]: 0.0,
                                  fillet["r"]: 0.0}))  # fmt: skip
    rep = B.build(d, tmp_path)
    assert rep["volume_mm3"] == pytest.approx(60 * 30 * 10, rel=1e-4)
    with pytest.raises(D.DesignError):  # общите размери не може да са 0
        D.set_values(d, {"L": 0.0})


def test_oval_end_is_a_rounded_rectangle(tmp_path):
    """Сечение като на флашка (21 x 9 с полукръгли страни), което разпознаването на прави не хваща: правоъгълник с
    четири заоблени ъгъла (един радиус за мерене), а не начупена форма."""
    from shapely.geometry import LineString

    from app.cad import assemble as A
    import numpy as np

    poly = LineString([(4.5, 4.5), (16.5, 4.5)]).buffer(4.5, quad_segs=24)
    outer = np.asarray(poly.exterior.coords)[:-1]
    d = A.assemble({"end": {"outer": outer, "holes": [], "U": 21.0, "V": 9.0}}, 50, 21, 9)
    end = d["views"]["end"]
    assert not end.get("free") and len(end["pts"]) == 4
    radii = {row[2] for row in end["pts"]}
    assert len(radii) == 1 and None not in radii
    assert D.value(d, radii.pop()) == pytest.approx(4.5, abs=0.25)
    rep = B.build(measured(d), tmp_path)
    assert rep["volume_mm3"] == pytest.approx(poly.area * 50, rel=0.01)


def test_radius_that_does_not_fit_is_shrunk_not_the_whole_part_boxed(tmp_path):
    """Оценка за радиус 3,6 мм на стена от 3,5 мм (ъгъл-скоба отпред): само радиусът се свива да се събере, а
    очертанието остава. Преди целият детайл ставаше кутия."""
    from app.cad import assemble as A

    d = D.new("block", 40, 30, 25)
    x = D.level_for(d, "x", 3.5, "Стъпало", "стъпалото", view="side")
    z = D.level_for(d, "z", 3.0, "Стъпало", "стъпалото", view="side")
    r = D.add_dim(d, "r", 3.6, "Радиус", "Радиус", kind="radius")
    d["views"]["side"]["pts"] = [["x0", "z0", None], ["xL", "z0", None], ["xL", z, None], [x, z, None],
                                 [x, "zH", r], ["x0", "zH", None]]  # fmt: skip
    A.repair(d)
    D.check(d)
    assert len(d["views"]["side"]["pts"]) == 6 and not d["notes"]
    assert D.value(d, r) < 3.5
    rep = B.build(measured(d), tmp_path)
    assert rep["volume_mm3"] < 0.4 * 40 * 30 * 25  # ъгъл, а не кутия


def test_usb_plug_is_a_narrow_end_up_to_its_step():
    """Флашка 52 x 21: тяло 39,5 мм със заоблени ъгли R3 и щепсел USB-A 12,5 x 12 (23% от дължината). По-тесният край
    се намира и когато е дълъг, а дължината му е до стъпалото, не докъдето тялото е пак с пълна ширина (заобленият
    ъгъл я удължаваше до 14 мм). Дъга, по-широка от самия щепсел, не става заобляне на върха му."""
    import numpy as np
    from shapely.geometry import box

    from app.cad import fit

    body = box(0, 0, 39.5, 21).buffer(-3).buffer(3, quad_segs=16)
    plug = box(39.4, 4.5, 52, 16.5)
    outer = np.asarray(body.union(plug).exterior.coords)[:-1]
    tabs = fit.end_tabs(outer, 52, 21)
    assert len(tabs) == 1 and tabs[0]["end"] == 1
    assert tabs[0]["len"] == pytest.approx(12.5, abs=0.6)
    assert tabs[0]["lo"] == pytest.approx(4.5, abs=0.3) and tabs[0]["hi"] == pytest.approx(4.5, abs=0.3)
    # Ъглите, в които щепселът и ъгълът на тялото са станали една дъга R10,6: върхът на щепсела е остър.
    cs = [(0.0, 21.0, 2.9), (0.0, 0.0, 2.9), (39.9, 0.0, 0.0), (39.9, 4.5, 0.0), (52.0, 4.5, 0.0), (52.0, 21.0, 10.6)]
    fixed = fit.apply_tabs(cs, tabs, 52, 21)
    tip = [c for c in fixed if c[0] == 52]
    assert len(tip) == 2 and all(c[2] == 0 for c in tip)


def test_slanted_end_of_a_usb_stick_has_a_longer_step_on_one_side():
    """Флашка 52 x 9 отпред: краят срещу капачката е скосен - долната половина на тялото свършва 7 мм преди върха,
    горната 3 мм, а по-тясното парче по средата стърчи от него. Стъпалото е там, където е на всяка страна, а не еднакво
    (преди и двете бяха 3,9 мм). Почти еднаквите стъпала от двете страни остават едно."""
    import numpy as np
    from shapely.geometry import Polygon, box

    from app.cad import fit

    body = Polygon([(0, 0), (45, 0), (49, 9), (0, 9)])
    tab = box(40, 1.1, 52, 7.9)
    outer = np.asarray(body.union(tab).exterior.coords)[:-1]
    tabs = fit.end_tabs(outer, 52, 9)
    assert len(tabs) == 1 and tabs[0]["end"] == 1
    lo, hi = fit.tab_lengths(tabs[0], 52)
    assert lo == pytest.approx(6.5, abs=0.6) and hi == pytest.approx(3.4, abs=0.6)
    cs = [(0.0, 0.0, 0.0), (45.0, 0.0, 0.0), (49.0, 9.0, 0.0), (0.0, 9.0, 0.0)]
    fixed = fit.apply_tabs(cs, tabs, 52, 9)
    assert sorted({round(c[0], 1) for c in fixed if 0 < c[0] < 52}) == [round(52 - lo, 1), round(52 - hi, 1)]
    even = fit.end_tabs(np.asarray(box(0, 0, 48, 9).union(box(47, 1.1, 52, 7.9)).exterior.coords)[:-1], 52, 9)
    a, b = fit.tab_lengths(even[0], 52)
    assert a == b
