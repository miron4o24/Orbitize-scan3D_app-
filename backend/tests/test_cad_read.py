"""Инженерният режим: четене на снимки (изрязване с IS-Net и очертания) на детайли с известни размери, нарисувани
като снимки отгоре (tests/cad_photos.py). Оценките от снимките трябва да са в рамките на 2% от размера: те само
подсказват, но трябва да намерят всеки елемент. Нужни са CadQuery и моделът за изрязване (на сървъра е там)."""

import pytest

pytest.importorskip("cadquery")
from app import segment as S  # noqa: E402

if not S.MODEL.exists():
    pytest.skip("няма модела за изрязване (tools/seg)", allow_module_level=True)

import trimesh  # noqa: E402

from app.cad import assemble as A  # noqa: E402
from app.cad import build as B  # noqa: E402
from app.cad import design as D  # noqa: E402
from app.cad import vision as V  # noqa: E402

from . import cad_photos as CP  # noqa: E402


@pytest.fixture(scope="module")
def sess():
    return S.session()


def read_all(sess, mesh, tmp_path, L, W, H):
    views = {}
    for k, (name, (U, VV)) in enumerate((("top", (L, W)), ("side", (L, H)), ("end", (W, H)))):
        path = tmp_path / f"{name}.jpg"
        CP.photo(mesh, name, angle=[7, -4, 3][k], seed=k).save(path, quality=92)
        v = V.read(sess, path, U, VV)
        views[name] = {"outer": v.outer, "holes": v.holes, "U": U, "V": VV}
    return A.assemble(views, L, W, H)


def measured_all(d):
    return D.set_values(d, {i: dim["v"] for i, dim in d["dims"].items()})


def test_plate_with_step_holes_and_round_corners(sess, tmp_path):
    truth = D.new("block", 60, 30, 10)
    r = D.add_dim(truth, "r", 5, "R", "R")
    truth["views"]["top"]["pts"] = [[u, v, r] for u, v, _ in truth["views"]["top"]["pts"]]
    x = D.level_for(truth, "x", 40, "С", "с", view="side")
    z = D.level_for(truth, "z", 6, "С", "с", view="side")
    truth["views"]["side"]["pts"] = [["x0", "z0", None], ["xL", "z0", None], ["xL", z, None], [x, z, None],
                                     [x, "zH", None], ["x0", "zH", None]]  # fmt: skip
    for _ in range(2):
        truth = D.add_feature(truth, {"t": "hole", "face": "top", "through": True, "centered": [False, True]})
    h1, h2 = truth["features"]
    truth = D.set_values(truth, {h1["d"]: 6, h2["d"]: 6, truth["levels"][h1["u"]["lvl"]]["dim"]: 10,
                                 truth["levels"][h2["u"]["lvl"]]["dim"]: 25})  # fmt: skip
    B.build(measured_all(truth), tmp_path / "truth")
    mesh = trimesh.load(tmp_path / "truth" / "part.stl", force="mesh")
    d = read_all(sess, mesh, tmp_path, 60, 30, 10)
    assert d["kind"] == "block"
    names = {q["name"]: q["est"] for q in D.questions(d)}
    holes = [f for f in d["features"] if f["t"] == "hole"]
    assert len(holes) == 2 and holes[0]["d"] == holes[1]["d"]  # еднакви: един размер
    assert d["dims"][holes[0]["d"]]["est"] == pytest.approx(6, abs=0.3)
    xs = sorted(D.coord(d, f["u"]["lvl"]) for f in holes)
    assert xs == pytest.approx([10, 25], abs=0.5)
    assert all(f["v"]["lvl"] == "yc" for f in holes)  # в средата: не се пита
    assert names["Стъпало по дължината (от десния край)"] == pytest.approx(20, abs=0.6)
    assert names["Стъпало по височината (от горната страна)"] == pytest.approx(4, abs=0.3)
    assert names["Радиус на заоблен ъгъл"] == pytest.approx(5, abs=0.5)
    assert len(D.questions(d)) == 9  # L, W, H, стъпалото (2), отворите (Ø и 2 положения), радиусът


def test_turned_shaft(sess, tmp_path):
    truth = D.new("turned", 50, 20, 20)
    x = D.level_for(truth, "x", 30, "Р", "р", view="profile")
    d2 = D.add_dim(truth, "d", 12, "Д", "Д", kind="diameter")
    truth["levels"]["Lr1"] = {"axis": "r", "dim": d2, "from": "diameter"}
    truth["views"]["profile"]["pts"] = [["x0", "r0", None], ["xL", "r0", None], ["xL", "Lr1", None],
                                        [x, "Lr1", None], [x, "rD", None], ["x0", "rD", None]]  # fmt: skip
    truth = D.add_feature(truth, {"t": "hole", "face": "left", "through": True})
    truth = D.set_values(truth, {truth["features"][0]["d"]: 6.0})
    B.build(measured_all(truth), tmp_path / "truth")
    mesh = trimesh.load(tmp_path / "truth" / "part.stl", force="mesh")
    d = read_all(sess, mesh, tmp_path, 50, 20, 20)
    assert d["kind"] == "turned"
    diam = [q for q in D.questions(d) if q["kind"] == "diameter"]
    assert sorted(round(q["est"]) for q in diam) == [6, 12]
    shoulder = next(q for q in D.questions(d) if q["kind"] == "position")
    assert shoulder["est"] == pytest.approx(20, abs=0.6)


def test_plate_with_slot_and_square_cutout(sess, tmp_path):
    """Канал и правоъгълен изрез с остри ъгли: сянката в изреза не бива да прави ъглите му заоблени."""
    truth = D.new("block", 60, 36, 5)
    truth = D.add_feature(truth, {"t": "slot", "face": "top", "through": True, "centered": [False, True]})
    truth = D.add_feature(truth, {"t": "rect", "face": "top", "through": True, "centered": [False, True]})
    sl, rc = truth["features"]
    truth = D.set_values(truth, {sl["len"]: 20, sl["w"]: 6, truth["levels"][sl["u"]["lvl"]]["dim"]: 18, rc["w"]: 12,
                                 rc["h"]: 16, truth["levels"][rc["u"]["lvl"]]["dim"]: 38})  # fmt: skip
    B.build(measured_all(truth), tmp_path / "truth")
    mesh = trimesh.load(tmp_path / "truth" / "part.stl", force="mesh")
    views = {}
    for k, (name, (U, VV)) in enumerate((("top", (60, 36)), ("side", (60, 5)))):
        path = tmp_path / f"{name}.jpg"
        CP.photo(mesh, name, angle=[5, -2][k], seed=k + 20, px_per_mm=12).save(path, quality=92)
        v = V.read(sess, path, U, VV)
        views[name] = {"outer": v.outer, "holes": v.holes, "U": U, "V": VV}
    d = A.assemble(views, 60, 36, 5)
    slot = next(f for f in d["features"] if f["t"] == "slot")
    rect = next(f for f in d["features"] if f["t"] == "rect")
    assert d["dims"][slot["len"]]["est"] == pytest.approx(20, abs=0.6)
    assert d["dims"][slot["w"]]["est"] == pytest.approx(6, abs=0.4)
    assert (d["dims"][rect["w"]]["est"], d["dims"][rect["h"]]["est"]) == pytest.approx((12, 16), abs=0.4)
    assert "r" not in rect  # остри ъгли
    assert D.coord(d, slot["u"]["lvl"]) == pytest.approx(8, abs=0.5)  # левият край на канала
