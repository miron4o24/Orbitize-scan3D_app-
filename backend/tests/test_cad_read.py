"""Инженерният режим: четене на снимките (изрязване с IS-Net, кой изглед е коя снимка и как е завъртяна, поправка на
перспективата) на детайли с известни размери, нарисувани като снимки с телефон отгоре (tests/cad_photos.py: камера
26 мм, предметът около половината кадър, обърнат с всяка страна нагоре, леко завъртян и изместен). Оценките от
снимките само подсказват, но трябва да намерят всеки елемент и да питат само за истинските размери. Нужни са
CadQuery и моделът за изрязване (на сървъра е там)."""

import numpy as np
import pytest

pytest.importorskip("cadquery")
from app import segment as S  # noqa: E402

if not S.MODEL.exists():
    pytest.skip("няма модела за изрязване (tools/seg)", allow_module_level=True)

import trimesh  # noqa: E402

from app.cad import assemble as A  # noqa: E402
from app.cad import build as B  # noqa: E402
from app.cad import design as D  # noqa: E402
from app.cad import photos as P  # noqa: E402
from app.cad import vision as V  # noqa: E402

from . import cad_photos as CP  # noqa: E402


@pytest.fixture(scope="module")
def sess():
    return S.session()


def measured(d):
    return D.set_values(d, {i: dim["v"] for i, dim in d["dims"].items()})


def read(sess, truth, tmp_path, names, sizes, yaw=7.0, seed=0, own_axis=False):
    """Снимките `names` на детайла `truth` -> план (общите размери) и описанието при истинските общи размери."""
    B.build(truth, tmp_path / "truth")
    mesh = trimesh.load(tmp_path / "truth" / "part.stl", force="mesh")
    rng = np.random.default_rng(seed)
    shots = {}
    for k, n in enumerate(names):
        path = tmp_path / f"{n}.jpg"
        off = (float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.08, 0.08)))
        img = CP.shot(mesh, n, yaw=yaw, spin=float(rng.uniform(-4, 4)), seed=k + 10 * seed, offset=off,
                      own_axis=own_axis)
        img.save(path, quality=92, exif=CP.exif())
        shots[n] = V.segment(sess, path)
    lay = P.layout(shots)
    plan = P.overall(lay)
    L, W, H = sizes
    lay = P.layout(shots, sizes)
    views, _ = P.views_mm(shots, lay, L, W, H)
    return plan, lay, A.assemble(views, L, W, H)


def solid_iou(a: dict, b: dict) -> float:
    sa, sb = B.solid(B.code(B.clean(a))), B.solid(B.code(B.clean(b)))
    inter = sa.intersect(sb).Volume()
    return inter / (sa.Volume() + sb.Volume() - inter)


def block():
    """70 x 40 x 20: изрез вдясно, стъпало отпред, два отвора, два ъгъла R6."""
    d = D.new("block", 70, 40, 20)
    x55 = D.level_for(d, "x", 55, "Н", "н", view="top")
    y15 = D.level_for(d, "y", 15, "Н", "н", view="top")
    y25 = D.level_for(d, "y", 25, "Н", "н", view="top")
    r = D.add_dim(d, "r", 6, "R", "R")
    d["views"]["top"]["pts"] = [["x0", "y0", r], ["xL", "y0", None], ["xL", y15, None], [x55, y15, None],
                                [x55, y25, None], ["xL", y25, None], ["xL", "yW", None], ["x0", "yW", r]]  # fmt: skip
    x25 = D.level_for(d, "x", 25, "С", "с", view="side")
    z12 = D.level_for(d, "z", 12, "С", "с", view="side")
    d["views"]["side"]["pts"] = [["x0", "z0", None], ["xL", "z0", None], ["xL", "zH", None], [x25, "zH", None],
                                 [x25, z12, None], ["x0", z12, None]]  # fmt: skip
    d = D.add_feature(d, {"t": "hole", "face": "top", "through": True, "centered": [False, True]})
    d = D.add_feature(d, {"t": "hole", "face": "top", "through": True, "centered": [False, False]})
    h1, h2 = d["features"]
    d = D.set_values(d, {h1["d"]: 8, h2["d"]: 5, d["levels"][h1["u"]["lvl"]]["dim"]: 12,
                         d["levels"][h2["u"]["lvl"]]["dim"]: 40, d["levels"][h2["v"]["lvl"]]["dim"]: 10})  # fmt: skip
    return measured(d)


def pill(L=140.0, Dm=64.0, r=None):
    """Легнал цилиндър (като колонка-хапче) със заоблени краища; r = D/2: полукълба."""
    d = D.new("turned", L, Dm, Dm)
    rid = D.add_dim(d, "r", r if r is not None else Dm / 2, "R", "R", kind="radius")
    d["views"]["profile"]["pts"] = [["x0", "r0", None], ["xL", "r0", None], ["xL", "rD", rid], ["x0", "rD", rid]]
    return measured(d)


def test_block_from_six_photos(sess, tmp_path):
    truth = block()
    plan, lay, d = read(sess, truth, tmp_path, P.NAMES, (70, 40, 20))
    assert [(q["id"], q["photo"]) for q in plan] == [("L", "top"), ("W", "top"), ("H", "front")]
    o = lay["orient"]
    assert {n: o[n]["view"] for n in o} == {"top": "top", "bottom": "top", "front": "side", "back": "side",
                                            "right": "end", "left": "end"}  # fmt: skip
    assert d["kind"] == "block"
    holes = sorted((f for f in d["features"] if f["t"] == "hole"), key=lambda f: D.coord(d, f["u"]["lvl"]))
    assert all(f["t"] == "hole" for f in d["features"])  # без лъжливи изрези
    # Дълбок отвор на снимка с перспектива се вижда отчасти: оценката е по-малка (диаметърът винаги се мери). Тесен
    # и дълбок отвор встрани от средата на кадъра може да не се види през детайла: тогава се добавя на ръка.
    truth_holes = [(8, 12), (5, 40)][: len(holes)]
    assert 1 <= len(holes) <= 2
    assert [D.value(d, f["d"]) for f in holes] == pytest.approx([t[0] for t in truth_holes], abs=1.2)
    assert [D.coord(d, f["u"]["lvl"]) for f in holes] == pytest.approx([t[1] for t in truth_holes], abs=1.0)
    names = {q["name"]: q["est"] for q in D.questions(d)}
    assert names["Стъпало по дължината (от левия край)"] == pytest.approx(25, abs=1.0)
    assert names["Стъпало по височината (от горната страна)"] == pytest.approx(8, abs=1.2)
    assert names["Радиус на заоблен ъгъл"] == pytest.approx(6, abs=1.0)
    end = {tuple(r) for r in d["views"]["end"]["pts"]}
    assert end == {tuple(r) for r in D.box("y", "z")}  # отстрани е правоъгълник: без лъжливи стъпала
    assert solid_iou(truth, measured(d)) > 0.95  # камерата е близо (перспектива), а това са само оценките


def test_pill_speaker(sess, tmp_path):
    """Легнал цилиндър със заоблени краища: само дължина и диаметър отначало, после радиусът на краищата."""
    truth = pill(140, 64, 14)
    plan, lay, d = read(sess, truth, tmp_path, ("top", "front", "right", "bottom"), (140, 64, 64), seed=1)
    assert [(q["id"], q["photo"]) for q in plan] == [("L", "top"), ("D", "right")]
    assert d["kind"] == "turned"
    asked = D.questions(d)
    assert [q["id"] for q in asked if q["kind"] != "radius"] == ["L", "D"]
    radius = [q for q in asked if q["kind"] == "radius"]
    assert len(radius) == 1 and radius[0]["est"] == pytest.approx(14, abs=2.5)
    assert solid_iou(truth, measured(d)) > 0.975


def test_capsule_and_upright_cylinder(sess, tmp_path):
    """Капсула (полукълба в краищата): радиусът е половината диаметър. Изправен цилиндър: кръг отгоре, без
    лъжливи стъпала отстрани."""
    truth = pill(150, 66)
    plan, _, d = read(sess, truth, tmp_path / "capsule", ("top", "front", "right"), (150, 66, 66), seed=2)
    assert d["kind"] == "turned"
    asked = D.questions(d)
    assert [q["id"] for q in asked if q["kind"] != "radius"] == ["L", "D"]
    assert [q["est"] for q in asked if q["kind"] == "radius"] == [pytest.approx(33, abs=0.05)]
    up = D.new("block", 60, 60, 90)
    up["views"]["top"] = {"pts": D.box("x", "y"), "round": True}
    up = measured(up)
    plan, _, d = read(sess, up, tmp_path / "upright", ("top", "front", "right"), (60, 60, 90), seed=3)
    assert [(q["id"], q["photo"]) for q in plan] == [("D", "top"), ("H", "front")]
    assert d["views"]["top"].get("round")
    assert [q["id"] for q in D.questions(d)] == ["L", "W", "H"]
    assert solid_iou(up, measured(d)) > 0.98


def test_flat_part_pointing_away(sess, tmp_path):
    """Като истинското сканиране на флашката от 15:58: отгоре предметът сочи напред (дължината е по височината на
    снимката), а за снимките отпред и отзад е претъркулен около дължината си. Отпред и отзад трябва да са изгледът
    отпред (с дебелината), а не краят: иначе височината се пита по ширината и детайлът става 51 x 21 x 21."""
    truth = D.new("block", 51, 21, 8)
    r = D.add_dim(truth, "r", 3, "R", "R")
    truth["views"]["top"]["pts"] = [[u, v, r] for u, v, _ in truth["views"]["top"]["pts"]]
    truth = measured(truth)
    plan, lay, d = read(sess, truth, tmp_path, P.NAMES, (51, 21, 8), yaw=88.0, seed=4, own_axis=True)
    views = {n: o["view"] for n, o in lay["orient"].items()}
    assert views == {"top": "top", "bottom": "top", "front": "side", "back": "side", "right": "end", "left": "end"}
    assert [(q["id"], q["photo"]) for q in plan] == [("L", "top"), ("W", "top"), ("H", "front")]
    assert not d["views"]["side"].get("round") and not d["views"]["end"].get("round")
    assert solid_iou(truth, measured(d)) > 0.95


def test_side_photo_with_the_part_turned_around(sess, tmp_path):
    """Като сканирането от 22:51: за снимките отпред и отзад флашката е и завъртяна наобратно на масата. Тесният край
    е отдясно отгоре и трябва да е отдясно и отпред (не при капачката), по формата и вида на двата края."""
    from shapely.geometry import LineString, Polygon

    truth = D.new("block", 51, 21, 9)
    xt = D.level_for(truth, "x", 45, "Т", "т", view="top")
    y1 = D.level_for(truth, "y", 1.5, "Т", "т", view="top")
    y2 = D.level_for(truth, "y", 19.5, "Т", "т", view="top")
    truth["views"]["top"]["pts"] = [["x0", "y0", None], [xt, "y0", None], [xt, y1, None], ["xL", y1, None],
                                    ["xL", y2, None], [xt, y2, None], [xt, "yW", None], ["x0", "yW", None]]  # fmt: skip
    zt1 = D.level_for(truth, "z", 1.5, "Т", "т", view="side")
    zt2 = D.level_for(truth, "z", 7.5, "Т", "т", view="side")
    r = D.add_dim(truth, "r", 4.0, "R", "R")
    truth["views"]["side"]["pts"] = [["x0", "z0", r], [xt, "z0", None], [xt, zt1, None], ["xL", zt1, None],
                                     ["xL", zt2, None], [xt, zt2, None], [xt, "zH", None], ["x0", "zH", r]]  # fmt: skip
    truth = measured(truth)
    B.build(truth, tmp_path / "truth")
    mesh = trimesh.load(tmp_path / "truth" / "part.stl", force="mesh")
    shots = {}
    for k, n in enumerate(("top", "front", "back", "right", "left", "bottom")):
        path = tmp_path / f"{n}.jpg"
        spin = 180.0 if n in ("front", "back") else 2.0
        CP.shot(mesh, n, yaw=4.0, spin=spin, seed=k + 30).save(path, quality=92, exif=CP.exif())
        shots[n] = V.segment(sess, path)
    lay = P.layout(shots, (51, 21, 9))
    assert lay["ends"]["front"] < -P.END_EVIDENCE  # обърнатата снимка е разпозната
    views, _ = P.views_mm(shots, lay, 51, 21, 9)
    d = A.assemble(views, 51, 21, 9)
    side = Polygon([(u, v) for u, v, _ in D.outline(d, "side")]).buffer(0)

    def height(x):
        return side.intersection(LineString([(x, -1), (x, 20)])).length

    assert height(49.5) < height(25) - 1.5  # тесният край е отдясно, както отгоре
    assert height(2.0) > height(49.5) + 1.0  # отляво е заобленият край на капачката, а не стъпалото
