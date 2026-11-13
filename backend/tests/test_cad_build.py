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


def hook_design() -> tuple[dict, dict]:
    """Куката от телефона (54 x 41 x 4): вътрешната извивка R14,45 (полукръг), външната R20,36, връх R2,94, оценки."""
    d = D.new("block", 54, 41, 4)
    d["flat"] = True
    x1 = D.level_for(d, "x", 5.23, "Стъпало", "стъпалото", view="top")
    y1 = D.level_for(d, "y", 6.06, "Стъпало", "стъпалото", view="top")
    y2 = D.level_for(d, "y", 41 - 6.04, "Стъпало", "стъпалото", view="top")
    x2 = D.level_for(d, "x", 54 - 24.01, "Стъпало", "стъпалото", view="top")
    r = {k: D.add_dim(d, "r", est, "Радиус на заоблен ъгъл", "Радиус", kind="radius", optional=True)
         for k, est in (("in", 14.45), ("tip", 2.94), ("out", 20.36))}  # fmt: skip
    d["views"]["top"]["pts"] = [[x1, y1, r["in"]], [x1, y2, r["in"]], [x2, y2, r["tip"]], [x2, "yW", r["tip"]],
                                ["x0", "yW", r["out"]], ["x0", "y0", r["out"]], ["xL", "y0", None], ["xL", y1, None]]
    d = D.add_feature(d, {"t": "fillet", "edges": "flat"})
    steps = {d["levels"][lv]["dim"]: d["dims"][d["levels"][lv]["dim"]]["est"] for lv in (x1, y1, y2, x2)}
    return d, {"in": r["in"], "tip": r["tip"], "out": r["out"], "edge": d["features"][-1]["r"], **steps}


def test_hook_radius_left_to_the_photo_follows_the_measurements(tmp_path):
    """"Не мога да го измеря" за радиус на куката, а другите мерки малко по-различни от снимката: оценката се свива,
    колкото да се събере, вместо "радиусът е твърде голям" (и тогава дори 0,1 мм за друг радиус не помагаше).
    Вътрешен радиус 0,1 мм: външният (оценка) се свива, за да остане стена, на която става заоблянето на ръбовете."""
    d, k = hook_design()
    steps = {i: v for i, v in k.items() if i not in ("in", "tip", "out", "edge")}
    wider = D.set_values(d, {**{i: 7.0 for i in steps if d["dims"][i]["est"] < 7}, k["edge"]: 1,
                             k["in"]: None, k["tip"]: None, k["out"]: None})  # fmt: skip
    rep = B.build(wider, tmp_path / "a")
    assert rep["volume_mm3"] > 0 and any("14,45" in n for n in rep["adjusted"])
    for inner, edge in ((0.1, 1.9), (0.0, 1.0), (2.0, 1.5)):
        sharp = D.set_values(d, {**steps, k["edge"]: edge, k["in"]: inner, k["tip"]: None, k["out"]: None})
        rep = B.build(sharp, tmp_path / f"b{inner}")
        assert rep["volume_mm3"] > 0 and any("20,36" in n for n in rep["adjusted"])


def test_measured_radius_too_big_says_how_big_it_can_be(tmp_path):
    """Измерен радиус, много по-голям от мястото си: грешката казва кой и колко най-много става; до 5% по-голям
    (полукръгъл край, мерен отделно от ширината) се свива без грешка."""
    d, k = hook_design()
    steps = {i: (7.0 if d["dims"][i]["est"] < 7 else v) for i, v in k.items() if i not in ("in", "tip", "out", "edge")}
    with pytest.raises(D.DesignError, match="най-много 13,49"):
        B.build(D.set_values(d, {**steps, k["edge"]: 1, k["in"]: 16, k["tip"]: None, k["out"]: None}), tmp_path / "a")
    rep = B.build(D.set_values(d, {**steps, k["edge"]: 1, k["in"]: 14, k["tip"]: None, k["out"]: None}), tmp_path / "b")
    assert any("измерено 14 мм" in n for n in rep["adjusted"])


def test_edge_rounding_the_kernel_cannot_make_is_reduced_not_a_dead_end(tmp_path, monkeypatch):
    """Заобляне на ръбовете, което OpenCascade не може (близо до половината дебелина при остър вътрешен ъгъл):
    по-малко, с бележка, вместо "намалете го" на последния въпрос без изход."""
    d, k = hook_design()
    real = B.solid
    calls = []

    def picky(text):
        calls.append(text)
        if len(calls) == 1:
            raise ValueError("BRep_API: command not done")
        return real(text)

    monkeypatch.setattr(B, "solid", picky)
    rep = B.build(D.set_values(d, {k["edge"]: 1.8, k["in"]: None, k["tip"]: None, k["out"]: None}), tmp_path)
    assert len(calls) == 2 and rep["volume_mm3"] > 0
    assert any("1,8 мм не става" in n and "1,35 мм" in n for n in rep["adjusted"])


def test_semicircle_end_arcs_that_just_touch_keep_lines_and_arcs():
    """Краят на кука е полукръг: двата радиуса запълват точно страната (4,2 + 4,2 на 8,4 мм). Шумът от закръгляването
    правеше заобленото очертание да се пресича, приликата със снимката ставаше 0 и куката - гладка форма без мерки."""
    from shapely.geometry import Polygon

    from app.cad import fit

    cs = [(0.0, 0.0, 0.0), (40.0, 0.0, 4.2000001), (40.0, 8.4, 4.2000001), (0.0, 8.4, 0.0)]
    assert Polygon(fit.rounded(fit.fitting_arcs(cs))).is_valid


def test_key_hole_shaped_like_a_d_follows_the_photo(tmp_path):
    """Отворът за халката на ключ (D: права отляво, дъга отдясно, 5,5 x 10 мм) в пластина от 2 мм не е кръг с
    диаметър 10, нито канал: формата е по снимката, а се мерят ширината, височината и положението."""
    import numpy as np
    from shapely.geometry import Point, box

    from app.cad import fit

    d_shape = box(0, 0, 3, 10).union(Point(3, 5).buffer(2.5, 64).intersection(box(3, 0, 6, 10)))
    d_shape = d_shape.union(box(0, 0, 3.0, 10)).buffer(0)
    pts = np.asarray(d_shape.exterior.coords)[:-1] + [40, 6]
    shape = fit.hole_shape(pts, 54, depth=2.0)
    assert shape["t"] == "rect" and shape.get("shape")
    assert shape["w"] == pytest.approx(5.5, abs=0.1) and shape["h"] == pytest.approx(10, abs=0.1)
    circle = np.c_[np.cos(np.linspace(0, 6.28, 80)), np.sin(np.linspace(0, 6.28, 80))] * 4 + [20, 10]
    assert fit.hole_shape(circle, 54, depth=2.0)["t"] == "hole"
    slot = fit.slot_poly({"u": 20, "v": 10, "len": 12, "w": 4, "angle": 0})
    assert fit.hole_shape(np.asarray(slot.exterior.coords)[:-1], 54, depth=2.0)["t"] == "slot"

    d = D.new("block", 54, 23, 2)
    d["flat"] = True
    f = {"t": "rect", "axis": "z", "side": "+", "w": D.add_dim(d, "w", shape["w"], "Ширина", "Ширина"),
         "h": D.add_dim(d, "h", shape["h"], "Височина", "Височина"), "u": {"lvl": "xc", "anchor": "center"},
         "v": {"lvl": "yc", "anchor": "center"}, "id": "rect1", "shape": shape["shape"], "corners": shape["corners"]}
    d["features"].append(f)
    rep = B.build(measured(d), tmp_path)
    assert rep["volume_mm3"] == pytest.approx((54 * 23 - d_shape.area) * 2, rel=0.01)
    assert "Отвор по формата от снимката" in " ".join(rep["elements"])


def sharpener_end():
    """Краят на точилка 40 x 40: кръг с четири плоски страни и канал 14 x 2,7 мм по средата на долната."""
    from shapely.geometry import Point, box

    body = Point(20, 20).buffer(23, quad_segs=64).intersection(box(0, 0, 40, 40))
    return body, body.difference(box(13, -1, 27, 2.7))


def test_channel_in_the_end_view_is_measured_and_the_pencil_hole_is_blind(tmp_path):
    """Каналът по долната страна на точилка (вдлъбнатина в очертанието отстрани) е канал по дължината с мерки за
    ширината и дълбочината, а не част от кривата; дупката за молива, видяна само на левия край, е глух отвор от него
    с дълбочина за мерене."""
    import numpy as np
    from shapely.geometry import Point

    from app.cad import assemble as A

    body, notched = sharpener_end()
    outer = np.asarray(notched.exterior.coords)[:-1]
    hole = np.asarray(Point(20, 22).buffer(4.5, quad_segs=24).exterior.coords)[:-1]
    view = {"outer": outer, "holes": [], "U": 40.0, "V": 40.0, "each": [outer, outer],
            "blind": [{"pts": hole, "face": "left"}]}  # fmt: skip
    d = A.assemble({"end": view}, 57, 40, 40)
    channel = next(f for f in d["features"] if f["t"] == "rect")
    assert channel["open"] == {"v": "-"} and channel["axis"] == "x" and channel["face"] == "bottom"
    assert D.value(d, channel["w"]) == pytest.approx(14, abs=0.3)
    assert D.value(d, channel["h"]) == pytest.approx(2.7, abs=0.2)
    pencil = next(f for f in d["features"] if f["t"] == "hole")
    assert pencil["side"] == "-" and pencil["face"] == "left" and pencil.get("depth")
    names = [q["name"] for q in D.questions(d)]
    assert "Ширина на канала" in names and "Дълбочина на канала" in names and "Дълбочина на отвора" in names
    d = D.set_values(d, {channel["w"]: 14.0, channel["h"]: 2.7, pencil["d"]: 9.0, pencil["depth"]: 12.0})
    rep = B.build(measured(d), tmp_path)
    end_area = body.area - 14 * 2.7
    assert rep["volume_mm3"] == pytest.approx(end_area * 57 - math.pi * 4.5**2 * 12, rel=0.015)
    assert any("Канал по дължината" in e for e in rep["elements"])


def test_flat_side_of_a_smooth_shape_is_a_straight_line():
    """Плоската страна на гладка форма е права между два ъгъла, а не сплайн през много точки точно по страната
    (сплайнът в равнината на другото тяло проваляше сечението: точилка от 74 000 мм3 ставаше 3 000)."""
    import numpy as np

    from app.cad import fit

    body, _ = sharpener_end()
    pts, corners = fit.smooth_outline(np.asarray(body.exterior.coords)[:-1], 40, 40)
    on_bottom = [k for k, (u, v) in enumerate(pts) if v <= 1e-6]
    assert len(on_bottom) == 2 and set(on_bottom) <= set(corners)


def test_square_end_photo_turned_by_the_person_is_turned_back():
    """Краят 40 x 40 (точилка): по пропорциите не личи коя ос на снимката е ширината. Вдлъбнатината отгоре в края
    (по ширината) и тази на снимката на края показват коя е: каналът е на същата стена. Без такива белези - както
    казва приложението."""
    import numpy as np
    from shapely.geometry import box

    from app.cad import photos as P

    def shot(poly):
        return {"outer": (np.asarray(poly.exterior.coords)[:-1] * [1, -1]).tolist(), "size": [1000, 1000]}

    top = shot(box(0, 0, 570, 400).difference(box(-1, 130, 20, 270)))
    plain_top = shot(box(0, 0, 570, 400))
    square = box(0, 0, 400, 400).buffer(-60).buffer(60)
    beside = shot(square.difference(box(-1, 130, 27, 270)))  # каналът по оста, която приложението мисли за височина
    below = shot(square.difference(box(130, -1, 270, 27)))
    x, y = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for top_shot, end_shot, turn in ((top, beside, True), (top, below, False), (plain_top, beside, False)):
        frames = {"top": P.Frame(top_shot), "right": P.Frame(end_shot)}
        orient = {"top": ["top", x, y], "right": ["end", x, y]}
        assert P.end_turned(frames, orient, (57, 40, 40)) == turn
    assert not P.end_turned({"top": P.Frame(top), "right": P.Frame(beside)},
                            {"top": ["top", x, y], "right": ["end", x, y]}, (57, 21, 9))  # fmt: skip


def test_hole_removed_with_zero_still_draws_and_builds(tmp_path):
    """Лъжлив канал от отблясъка на светкавицата, махнат с 0 за дължината и ширината: чертежът не пада (сървърът
    връщаше грешка 500 на "Построй модела"), а отчетът е без него."""
    from app.cad import render as R

    d = D.new("block", 54, 23, 2)
    d = D.add_feature(d, {"t": "slot", "face": "top", "through": True, "centered": [True, True]})
    slot = d["features"][0]
    d = measured(D.set_values(d, {slot["len"]: 0.0, slot["w"]: 0.0}))
    R.drawing(d)
    rep = B.build(d, tmp_path)
    assert rep["volume_mm3"] == pytest.approx(54 * 23 * 2, rel=1e-4)
    assert not any("Канал" in e for e in rep["elements"])


def test_tapping_a_hole_on_the_photo_finds_it():
    """Докоснат отвор на снимката с очертанието (за махане на лъжлив отвор): вътре в него или до самия му ръб е
    той; в празно място - нищо."""
    from PIL import Image

    from app.cad import render as R

    d = D.new("block", 60, 30, 5)
    d = D.add_feature(d, {"t": "hole", "face": "top", "through": True, "centered": [True, True]})
    hole = d["features"][0]
    d = measured(D.set_values(d, {hole["d"]: 6.0}))
    photo = Image.new("RGB", (1200, 900), (120, 110, 100))
    A = [[10.0, 0.0, 300.0], [0.0, -10.0, 600.0]]  # 10 пиксела на мм, v нагоре
    sh, _, _, _ = R.overlay_sheet(d, "top", photo, A)
    w, h = sh.im.size
    cx, cy = sh.px(30, 15)
    assert R.feature_at(d, "top", photo, A, cx / w, cy / h) == hole["id"]
    ex, ey = sh.px(33.4, 15)  # до ръба отвън
    assert R.feature_at(d, "top", photo, A, ex / w, ey / h) == hole["id"]
    fx, fy = sh.px(5, 5)
    assert R.feature_at(d, "top", photo, A, fx / w, fy / h) is None


def test_back_of_a_usb_stick_is_a_rounded_piece_behind_a_slanted_end(tmp_path):
    """Задната част на флашка 52 x 21 x 8,5: тясна с по 1 мм от всяка страна, а краят на тялото е наклонен (долу
    свършва при 45,5 мм, горе при 49,5). Сечението на задната част е умаленото заоблено сечение на тялото (не
    правоъгълник), а между стъпалата тялото свършва по наклона, не на стъпало."""
    import numpy as np
    import trimesh
    from shapely.geometry import LineString, Polygon, box

    from app.cad import assemble as A

    end = LineString([(4.25, 4.25), (16.75, 4.25)]).buffer(4.25, quad_segs=32)
    top = box(0, 0, 49.5, 21).union(box(49, 1, 52, 20))
    side = Polygon([(0, 0), (45.5, 0), (45.5, 1), (52, 1), (52, 7.5), (49.5, 7.5), (49.5, 8.5), (0, 8.5)])
    views = {name: {"outer": np.asarray(g.exterior.coords)[:-1], "holes": [], "U": U, "V": V}
             for name, g, U, V in (("top", top, 52, 21), ("side", side, 52, 8.5), ("end", end, 21, 8.5))}
    d = A.assemble(views, 52, 21, 8.5)
    ends = D.narrow_ends(d)
    assert len(ends) == 1 and ends[0]["sign"] == "+" and ends[0]["round"]
    assert D.coord(d, ends[0]["bottom"]) == pytest.approx(45.5, abs=0.6)
    assert D.coord(d, ends[0]["top"]) == pytest.approx(49.5, abs=0.6)
    B.build(measured(d), tmp_path)
    mesh = trimesh.load(tmp_path / "part.stl", force="mesh")

    def area(x):
        s = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
        return s.to_2D()[0].area

    full, piece = end.area, end.area * (19 / 21) * (6.5 / 8.5)
    assert area(40) == pytest.approx(full, rel=0.03)
    assert area(51) == pytest.approx(piece, rel=0.06)  # заоблено, не правоъгълник 19 x 6,5 (123,5)
    assert piece * 1.05 < area(47.5) < full * 0.97  # по наклона: горе още тяло, долу само задната част


def test_gray_slot_on_a_printed_part_is_a_hole():
    """Отворът за връзката на бяла флашка с цветни лога: сивото петно (0,6 от яркостта на детайла) не минаваше прага,
    защото печатът увеличава разсейването. Сега е тъмно петно и, сиво и с форма на канал, е отвор."""
    import numpy as np

    from app.cad import photos as P
    from app.cad import vision as V

    rgb = np.full((400, 900, 3), 205.0)
    rng = np.random.default_rng(0)
    for k in range(12):  # цветни лога
        x, y = 80 + 60 * k, 60 + 20 * (k % 3)
        rgb[y:y + 50, x:x + 40] = rng.uniform(40, 230, 3)
    yy, xx = np.mgrid[0:400, 0:900]
    rgb[np.hypot(xx - np.clip(xx, 315, 505), yy - 265) <= 15] = 118.0  # сивият канал (0,58 от детайла)
    solid = np.ones((400, 900), bool)
    marks = V.dark_marks(rgb, solid)
    slot = [(m, i) for m, i in marks if 290 < m[:, 0].mean() < 530 and 240 < m[:, 1].mean() < 290]
    assert len(slot) == 1 and slot[0][1]["sat"] < 0.05
    # В мм (детайл 90 x 40), далеч от ръба: отвор от един изглед.
    m = slot[0][0] / 10.0
    p = {"outer": np.array([[0, 0], [90, 0], [90, 40], [0, 40]], float), "marks": [m], "mark_info": [slot[0][1]]}
    assert len(P.dark_holes(p, [], 90, 40)) == 1


def test_slanted_end_before_a_usb_plug_keeps_the_plug_rectangular(tmp_path):
    """Щепселът USB-A (навътре с 4,5 и 2,2 мм) остава правоъгълен, а краят на тялото пред него е наклонен."""
    import numpy as np
    import trimesh
    from shapely.geometry import LineString, Polygon, box

    from app.cad import assemble as A

    end = LineString([(4.5, 4.5), (16.5, 4.5)]).buffer(4.5, quad_segs=32)
    top = box(0, 0, 41, 21).union(box(40, 4.5, 52, 16.5))
    side = Polygon([(0, 0), (37, 0), (37, 2.2), (52, 2.2), (52, 6.8), (41, 6.8), (41, 9), (0, 9)])
    views = {name: {"outer": np.asarray(g.exterior.coords)[:-1], "holes": [], "U": U, "V": V}
             for name, g, U, V in (("top", top, 52, 21), ("side", side, 52, 9), ("end", end, 21, 9))}
    d = A.assemble(views, 52, 21, 9)
    ends = D.narrow_ends(d)
    assert len(ends) == 1 and not ends[0]["round"]
    B.build(measured(d), tmp_path)
    mesh = trimesh.load(tmp_path / "part.stl", force="mesh")
    plug = mesh.section(plane_origin=[47, 0, 0], plane_normal=[1, 0, 0]).to_2D()[0]
    assert plug.area == pytest.approx(12 * 4.6, rel=0.04)
    middle = mesh.section(plane_origin=[39, 0, 0], plane_normal=[1, 0, 0]).to_2D()[0].area
    assert 12 * 4.6 * 1.05 < middle < end.area * 0.97  # наклоненият край на тялото


def test_flat_pen_is_a_round_body_with_its_clip(tmp_path):
    """Химикалка 140 x 15 x 11, снимана само отгоре и отдолу (избран "плосък детайл"): дебела колкото е широко
    тялото ѝ, затова е кръгла (сечение кръг Ø11), а щипката встрани е тънка издатина - не плоча 11 мм с
    очертанието отгоре."""
    import numpy as np
    import trimesh
    from shapely.geometry import Polygon, box

    from app.cad import fit

    body = Polygon([(0, 3), (8, 3), (15, 0), (120, 0), (140, 4), (140, 7), (120, 11), (15, 11), (8, 8), (0, 8)])
    clip = box(20, 10, 60, 15)
    outer = np.asarray(body.union(clip).exterior.coords)[:-1]
    rb = fit.round_body(outer, 140, 15, 11)
    assert rb is not None and rb["axis"] == pytest.approx(5.5, abs=0.3)
    assert rb["fin"] > 0 and rb["fin_x"][0] == pytest.approx(20, abs=2) and rb["fin_x"][1] == pytest.approx(60, abs=2)
    assert fit.round_body(np.asarray(box(0, 0, 54, 41).exterior.coords)[:-1], 54, 41, 4) is None  # кука: плоска
    d = D.new("block", 140, 15, 11)
    d["flat"] = True
    d["views"]["top"] = {"pts": [[D.frac_level("x", u, 140), D.frac_level("y", v, 15), None] for u, v in outer],
                         "free": True}
    d = D.add_round(d, rb)
    d = D.set_values(d, {d["features"][0]["fin"]: 1.5})
    rep = B.build(measured(d), tmp_path)
    assert any("Кръгло тяло" in e for e in rep["elements"])
    mesh = trimesh.load(tmp_path / "part.stl", force="mesh")

    def section(x):
        return mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0]).to_2D()[0]

    barrel = section(90)
    assert barrel.area == pytest.approx(np.pi * 5.5**2, rel=0.03)  # кръг Ø11, не правоъгълник 11 x 11
    with_clip = section(40)
    assert with_clip.area == pytest.approx(np.pi * 5.5**2 + 1.5 * 4.0, rel=0.06)  # и щипката 1,5 x ~4


def test_partial_cutout_is_replaced_only_when_the_rest_looks_like_the_part():
    """IS-Net изрязва само върха на сива химикалка на сиво дърво: MobileSAM взима цялата и се ползва, защото
    добавеното е с цвета на химикалката. Ако добавеното е масата (с цвета на фона), остава IS-Net."""
    import numpy as np
    from PIL import Image
    from shapely.geometry import box

    from app.cad import vision as V

    rgb = np.zeros((900, 1200, 3), np.uint8)
    rgb[:] = (150, 135, 110)  # дърво
    rgb[420:480, 200:1000] = (172, 172, 178)  # химикалката
    im = Image.fromarray(rgb)
    tip = box(850, 420, 1000, 480)
    pen = box(200, 420, 1000, 480)
    assert V.partial(im, tip, pen)
    table = box(200, 300, 1000, 600)  # химикалката и масата около нея
    assert not V.partial(im, pen, table)
    assert not V.partial(im, pen, pen.buffer(2))  # почти същото: не е "само част"


def test_camera_bump_seen_from_the_side_is_a_round_boss_on_the_back(tmp_path):
    """Телефон 165 x 75,5 x 8,5 (тънък: плосък): камерата на гърба (кръг Ø30, 2 мм) личи на снимките отстрани като
    издатина, а измерената дебелина е без нея. Моделът е тялото с кръгла издатина навън, а не само правоъгълник."""
    import numpy as np
    from shapely.geometry import box

    from app.cad import assemble as A
    from app.cad import service as SV

    # Снимката отстрани в мм, свита до измерената дебелина 8,5 (тялото 8,5 + издатината 2 = 10,5 на снимката).
    side = box(0, 2, 165, 10.5).union(box(25, 0, 55, 2))
    q = np.asarray(side.exterior.coords)[:-1] * [1, 8.5 / 10.5]
    b = SV.side_bump({"outer": q, "ratio": 8.5 / 10.5, "U": 165.0})
    assert b and b["side"] == "-" and b["lo"] == pytest.approx(25, abs=2) and b["hi"] == pytest.approx(55, abs=2)
    assert b["rise"] == pytest.approx(2 / 8.5, abs=0.03)
    plain = np.asarray(box(0, 0, 165, 8.5).exterior.coords)[:-1]
    assert SV.side_bump({"outer": plain, "ratio": 1.0, "U": 165.0}) is None

    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    disk = np.c_[40 + 15 * np.cos(t), 37.75 + 15 * np.sin(t)]
    for outside, H, vol in ((True, 8.5, 165 * 75.5 * 8.5 + np.pi * 15**2 * 2),
                            (False, 10.5, 165 * 75.5 * 8.5 + np.pi * 15**2 * 2)):  # fmt: skip
        d = D.new("block", 165, 75.5, H)
        A.add_boss(d, {"pts": disk, "face": "bottom", "height": 2.0, "outside": outside}, 165)
        f = d["features"][0]
        assert f["boss"] and f["t"] == "hole" and f["side"] == "-"
        assert D.value(d, f["d"]) == pytest.approx(30, abs=0.5) and D.value(d, f["depth"]) == 2.0
        assert "издатината" in d["dims"][f["d"]]["name"]
        rep = B.build(measured(d), tmp_path / str(outside))
        assert rep["volume_mm3"] == pytest.approx(vol, rel=0.005)
        assert any("Издатина" in e for e in rep["elements"])


def test_ports_display_and_button_seen_on_the_photos_are_in_the_model(tmp_path):
    """Powerbank 98 x 39 x 30: USB-C (заоблен правоъгълник 9 x 3,2) и USB-A (13 x 5,4) в левия край, екранът (58 x
    32) отгоре и кръглият бутон (Ø7) отзад са глухи елементи от стената си с мерки, а не изгубени."""
    import numpy as np
    from shapely.geometry import LineString, Point, box

    from app.cad import assemble as A

    d = D.new("block", 98, 39, 30)
    usb_c = LineString([(27.6, 12), (33.4, 12)]).buffer(1.6, quad_segs=16)
    usb_a = box(5, 9.3, 18, 14.7)
    window = box(37, 3.5, 95, 35.5).buffer(-3).buffer(3, quad_segs=8)
    button = Point(84, 15).buffer(3.5, quad_segs=24)
    for poly, axis, view, size, face, kind in ((usb_c, "x", "end", 39, "left", "port"),
                                               (usb_a, "x", "end", 39, "left", "port"),
                                               (window, "z", "top", 98, "top", "panel"),
                                               (button, "y", "side", 98, "back", "button")):  # fmt: skip
        A.add_detail(d, np.asarray(poly.exterior.coords)[:-1], axis, view, size, face, kind)
    kinds = [f["kind"] for f in d["features"]]
    assert kinds == ["port", "port", "panel", "button"]
    c, a, w, b = d["features"]
    assert c["t"] == "rect" and D.value(d, c["r"]) == pytest.approx(1.6, abs=0.25)  # USB-C: полукръгли краища
    assert D.value(d, a["w"]) == pytest.approx(13, abs=0.2) and D.value(d, a["depth"]) == pytest.approx(13, abs=0.2)
    assert b["t"] == "hole" and D.value(d, b["d"]) == pytest.approx(7, abs=0.2)
    assert D.value(d, w["depth"]) == 0.5 and "панела" in d["dims"][w["w"]]["name"]
    rep = B.build(measured(d), tmp_path)
    cut = usb_c.area * 9.2 + usb_a.area * 13 + window.area * 0.5 + button.area * 0.3
    assert rep["volume_mm3"] == pytest.approx(98 * 39 * 30 - cut, rel=0.003)
    assert sum("Порт" in e for e in rep["elements"]) == 2 and any("Бутон" in e for e in rep["elements"])


def test_height_is_asked_on_the_end_photo_next_to_the_width():
    """Височината се пита на снимката отстрани, където се вижда и ширината (сива, вече измерена): отпред тесният
    край изглеждаше като ширината и тя се питаше втори път."""
    from app.cad import photos as P

    orient = {n: {"view": v, "eu": [1, 0], "ev": [0, 1], "primary": n in ("top", "front", "right")}
              for n, v in (("top", "top"), ("front", "side"), ("right", "end"))}  # fmt: skip
    qs = P.overall({"orient": orient, "round": {"top": False, "end": False}})
    h = next(q for q in qs if q["id"] == "H")
    assert h["photo"] == "right" and h["known"] == "W" and "не ширината" in h["ask"]
    orient.pop("right")
    h = next(q for q in P.overall({"orient": orient, "round": {"top": False, "end": False}}) if q["id"] == "H")
    assert h["photo"] == "front" and h["known"] == "L"
