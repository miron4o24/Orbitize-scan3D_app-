"""Картинки на инженерния режим (PIL, без външни услуги):

- question: изгледът на детайла от описанието, с това, което трябва да се измери, в цвят (размерна линия, отвор,
  заоблен ъгъл), и оценката от снимките;
- overlay: очертанието от описанието върху снимката (екранът "Какво виждам" и проверката след мерките);
- drawing: чертеж с изгледите и всички размери (оценените със звездичка)."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from . import design as D

FONTS = ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
BOLD = ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
INK = (38, 41, 48)
SOFT = (150, 156, 166)
FILL = (233, 236, 241)
ACCENT = (232, 93, 4)  # оранжево: това се мери
BLUE = (37, 99, 235)
PAPER = (252, 252, 253)
VIEW_NORMAL = {"top": "z", "side": "y", "end": "x", "profile": "y"}


def font(px: int, bold: bool = False):
    for name in BOLD if bold else FONTS:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default(px)


def mm(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",")


# --- Геометрията на изглед от описанието -----------------------------------------------------------------------------


def arc_outline(pts: list[tuple[float, float, float]], steps: int = 10) -> np.ndarray:
    """Ъглите (u, v, r) -> начупена линия със заоблените ъгли като дъги (същото като outline() в part.py)."""
    n = len(pts)
    out = []
    for k in range(n):
        u, v, r = pts[k]
        if r <= 0:
            out.append((u, v))
            continue
        pu, pv, _ = pts[k - 1]
        nu, nv, _ = pts[(k + 1) % n]
        a, b = np.array([pu - u, pv - v]), np.array([nu - u, nv - v])
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        ang = math.acos(max(-1.0, min(1.0, float(a @ b))))
        t = r / math.tan(ang / 2)
        bis = (a + b) / np.linalg.norm(a + b)
        c = np.array([u, v]) + bis * r / math.sin(ang / 2)
        p1, p2 = np.array([u, v]) + a * t, np.array([u, v]) + b * t
        a1, a2 = math.atan2(*(p1 - c)[::-1]), math.atan2(*(p2 - c)[::-1])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for s in np.linspace(0, 1, steps):
            out.append(tuple(c + r * np.array([math.cos(a1 + da * s), math.sin(a1 + da * s)])))
    return np.asarray(out)


def view_size(d: dict, view: str) -> tuple[float, float]:
    if view == "profile":
        return D.total(d, "x"), D.value(d, "D")
    a, b = D.VIEW_AXES[view]
    return D.total(d, a), D.total(d, b)


def silhouette(d: dict, view: str) -> np.ndarray:
    """Очертанието на изгледа в мм (u, v), с дъгите."""
    if view == "profile":  # профилът (x, r) огледан около оста: горе R + r, долу R - r
        R = D.value(d, "D") / 2
        pts = np.array([(u, r) for u, r, _ in D.outline(d, "profile")])
        top = Polygon(np.c_[pts[:, 0], R + pts[:, 1]]).buffer(0)
        bottom = Polygon(np.c_[pts[:, 0], R - pts[:, 1]]).buffer(0)
        shape = top.union(bottom)
        if shape.geom_type != "Polygon":
            shape = max(shape.geoms, key=lambda g: g.area)
        return np.asarray(shape.exterior.coords)[:-1]
    return arc_outline(D.outline(d, view))


def features_in(d: dict, view: str) -> list[tuple[dict, np.ndarray]]:
    """Елементите, които се виждат в изгледа като очертания: (елемент, точки в мм)."""
    normal = VIEW_NORMAL[view]
    out = []
    for f in d["features"]:
        if f["t"] not in ("hole", "rect", "slot"):
            continue
        place = D.feature_place(d, f)
        if f["axis"] != normal:
            continue
        cu, cv = place["u"], place["v"]
        if f["t"] == "hole":
            r = D.value(d, f["d"]) / 2
            t = np.linspace(0, 2 * math.pi, 48)
            out.append((f, np.c_[cu + r * np.cos(t), cv + r * np.sin(t)]))
        elif f["t"] == "rect":
            w, h = D.value(d, f["w"]), D.value(d, f["h"])
            r = D.value(d, f["r"]) if f.get("r") else 0.0
            r = min(r, w / 2 - 1e-3, h / 2 - 1e-3)
            out.append((f, arc_outline([(cu - w / 2, cv - h / 2, r), (cu + w / 2, cv - h / 2, r),
                                        (cu + w / 2, cv + h / 2, r), (cu - w / 2, cv + h / 2, r)])))  # fmt: skip
        else:
            ln, w = D.value(d, f["len"]), D.value(d, f["w"])
            lu, lv = (ln, w) if f.get("angle", 0) == 0 else (w, ln)
            r = min(lu, lv) / 2 - 1e-3
            out.append((f, arc_outline([(cu - lu / 2, cv - lv / 2, r), (cu + lu / 2, cv - lv / 2, r),
                                        (cu + lu / 2, cv + lv / 2, r), (cu - lu / 2, cv + lv / 2, r)])))  # fmt: skip
    return out


# --- Рисуване --------------------------------------------------------------------------------------------------------


class Sheet:
    """Изглед в мм върху картинка: v нагоре, с поле за размерните линии."""

    def __init__(self, im: Image.Image, box: tuple[int, int, int, int], U: float, V: float):
        self.im, self.draw = im, ImageDraw.Draw(im)
        x0, y0, x1, y1 = box
        self.s = min((x1 - x0) / U, (y1 - y0) / V)
        self.ox = x0 + ((x1 - x0) - U * self.s) / 2
        self.oy = y1 - ((y1 - y0) - V * self.s) / 2
        self.U, self.V = U, V

    def px(self, u: float, v: float) -> tuple[float, float]:
        return self.ox + u * self.s, self.oy - v * self.s

    def poly(self, pts: np.ndarray, fill=None, outline=INK, width=3):
        xy = [self.px(u, v) for u, v in pts]
        self.draw.polygon(xy, fill=fill)
        self.draw.line(xy + [xy[0]], fill=outline, width=width, joint="curve")

    def line(self, a, b, fill=INK, width=2, dash: float | None = None):
        pa, pb = self.px(*a), self.px(*b)
        if not dash:
            self.draw.line([pa, pb], fill=fill, width=width)
            return
        n = max(1, int(math.dist(pa, pb) / dash))
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1.0, (k + 1) / n)
            self.draw.line([(pa[0] + (pb[0] - pa[0]) * t0, pa[1] + (pb[1] - pa[1]) * t0),
                            (pa[0] + (pb[0] - pa[0]) * t1, pa[1] + (pb[1] - pa[1]) * t1)], fill=fill, width=width)  # fmt: skip

    def dimension(self, a, b, offset: tuple[float, float], text: str, color=ACCENT, width=4, size=30):
        """Размерна линия от a до b (мм), изместена с offset (пиксели), с текста по средата."""
        pa, pb = np.array(self.px(*a)), np.array(self.px(*b))
        off = np.array(offset, float)
        qa, qb = pa + off, pb + off
        for p, q in ((pa, qa), (pb, qb)):  # изводни линии
            if np.linalg.norm(off) > 0:
                self.draw.line([tuple(p), tuple(q + off / max(np.linalg.norm(off), 1) * 8)], fill=color, width=2)
        self.draw.line([tuple(qa), tuple(qb)], fill=color, width=width)
        d = qb - qa
        n = np.linalg.norm(d)
        if n > 1:
            d = d / n
            for p, s in ((qa, 1), (qb, -1)):
                tip = p
                back = p + s * d * 18
                side = np.array([-d[1], d[0]]) * 8
                self.draw.polygon([tuple(tip), tuple(back + side), tuple(back - side)], fill=color)
        mid = (qa + qb) / 2 + (off / max(np.linalg.norm(off), 1)) * 26
        label(self.draw, tuple(mid), text, color, size)


def label(draw: ImageDraw.ImageDraw, at, text: str, color, size=30):
    f = font(size, bold=True)
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    w, h = r - l, b - t
    W, H = draw.im.size
    x = min(max(at[0] - w / 2, 14), W - w - 14)  # в рамките на картинката
    y = min(max(at[1] - h / 2, 12), H - h - 14)
    draw.rounded_rectangle([x - 10, y - 7, x + w + 10, y + h + 9], radius=10, fill=(255, 255, 255), outline=color,
                           width=3)  # fmt: skip
    draw.text((x - l, y - t), text, font=f, fill=color)


def base_view(d: dict, view: str, size=(1000, 640), margin=120) -> Sheet:
    U, V = view_size(d, view)
    im = Image.new("RGB", size, PAPER)
    sh = Sheet(im, (margin, margin, size[0] - margin, size[1] - margin), U, V)
    sh.poly(silhouette(d, view), fill=FILL, outline=INK, width=4)
    for _, pts in features_in(d, view):
        sh.poly(pts, fill=PAPER, outline=INK, width=3)
    if view == "profile":  # оста и отворът по нея (скрит: пунктир)
        R = V / 2
        sh.line((-U * 0.04, R), (U * 1.04, R), SOFT, 2, dash=14)
        for f in d["features"]:
            if f["t"] == "hole" and f["axis"] == "x":
                r = D.value(d, f["d"]) / 2
                depth = D.value(d, f["depth"]) if f.get("depth") else U
                x0, x1 = (0, depth) if f.get("side") == "-" else (U - depth, U)
                for v in (R - r, R + r):
                    sh.line((x0, v), (x1, v), INK, 2, dash=12)
    return sh


def view_for(d: dict, dim_id: str) -> str:
    dim = d["dims"][dim_id]
    show = dim.get("show") or {}
    if d["kind"] == "turned":
        for f in d["features"]:
            if f["id"] == show.get("feature") and f["t"] == "hole" and f["axis"] == "x":
                return "end"
        return "profile"
    if dim_id == "H":
        return "side"
    if show.get("view") in ("top", "side", "end"):
        return show["view"]
    if show.get("feature"):
        f = next((f for f in d["features"] if f["id"] == show["feature"]), None)
        if f and f.get("axis"):
            return {"z": "top", "y": "side", "x": "end"}[f["axis"]]
    if show.get("axis") == "z":
        return "side"
    return "top"


def question(d: dict, dim_id: str) -> Image.Image:
    """Картинката към въпроса за размера `dim_id`."""
    dim = d["dims"][dim_id]
    show = dim.get("show") or {}
    view = view_for(d, dim_id)
    if view == "end" and d["kind"] == "turned":
        return turned_end(d, dim_id)
    sh = base_view(d, view)
    U, V = sh.U, sh.V
    est = f"≈ {mm(dim['est'])}" if dim.get("est") else "?"
    value = mm(dim["v"]) if dim.get("v") and dim["src"] == "measured" else est
    a, b = ("x", "r") if view == "profile" else D.VIEW_AXES[view]
    feature = next((f for f in d["features"] if f["id"] == show.get("feature")), None)
    if dim_id in ("L", "W", "H", "D"):
        if dim_id == "L":
            sh.dimension((0, 0), (U, 0), (0, 60), value)
        elif dim_id == "D" or (dim_id == "W" and view == "top") or (dim_id == "H" and view == "side"):
            sh.dimension((U, 0), (U, V), (60, 0), value)
    elif dim_id in d["dims"] and show.get("level") and show["level"] in d["levels"]:
        lvl = d["levels"][show["level"]]
        c = D.coord(d, show["level"])
        if lvl["axis"] == "r":  # диаметър на детайл от струг: на мястото, където го има
            xs = [u for u, r, _ in D.outline(d, "profile") if abs(r - c) < 1e-6]
            x = (min(xs) + max(xs)) / 2 if xs else U / 2
            R = V / 2
            sh.dimension((x, R - c), (x, R + c), (0, 0), "Ø " + value)
        else:
            on_u = lvl["axis"] == a
            ref = 0.0 if lvl["from"] == "start" else (U if on_u else V)
            if on_u:
                sh.line((c, -V * 0.08), (c, V * 1.08), ACCENT, 3, dash=12)
                sh.dimension((ref, V), (c, V), (0, -60), value)
            else:
                sh.line((-U * 0.05, c), (U * 1.05, c), ACCENT, 3, dash=12)
                sh.dimension((U, ref), (U, c), (60, 0), value)
    elif feature is not None and feature["t"] in ("hole", "rect", "slot"):
        for f, pts in features_in(d, view):
            if f is feature or f.get("d") == feature.get("d") and f["t"] == "hole" and dim_id == feature.get("d"):
                sh.poly(pts, fill=(255, 237, 224), outline=ACCENT, width=5)
        place = D.feature_place(d, feature)
        cu, cv = place["u"], place["v"]
        key = next((k for k in ("d", "w", "h", "len", "depth", "r") if feature.get(k) == dim_id), None)
        if key == "d":
            r = D.value(d, feature["d"]) / 2
            sh.dimension((cu - r, cv), (cu + r, cv), (0, -max(60, r * sh.s + 50)), "Ø " + value)
        elif key in ("w", "h", "len"):
            along = show.get("along") or (a if key in ("w", "len") else b)
            if feature["t"] == "slot":
                ln, w = D.value(d, feature["len"]), D.value(d, feature["w"])
                along_u = (feature.get("angle", 0) == 0) == (key == "len")
                size = ln if key == "len" else w
            else:
                size = D.value(d, feature[key])
                along_u = along == a
            if along_u:
                sh.dimension((cu - size / 2, cv), (cu + size / 2, cv), (0, -70), value)
            else:
                sh.dimension((cu, cv - size / 2), (cu, cv + size / 2), (70, 0), value)
        else:
            label(sh.draw, sh.px(cu, cv + (V * 0.18)), f"{'дълбочина' if key == 'depth' else 'радиус'} {value}",
                  ACCENT)  # fmt: skip
    elif show.get("radius") or dim["kind"] == "radius":
        pts = D.outline(d, view)
        for (u, v, _r), row in zip(pts, d["views"][view]["pts"]):
            if row[2] == dim_id:
                x, y = sh.px(u, v)
                sh.draw.ellipse([x - 26, y - 26, x + 26, y + 26], outline=ACCENT, width=5)
        label(sh.draw, (sh.im.width / 2, 50), f"R {value}", ACCENT)
    elif feature is not None:  # заобляне или фаска
        label(sh.draw, (sh.im.width / 2, 50), f"{D.label(d, feature)}: {value}", ACCENT)
    caption(sh, f"Изглед {D.VIEW_NAME[view]}")
    return sh.im


def turned_end(d: dict, dim_id: str) -> Image.Image:
    """Детайл от струг откъм края: кръгът и отворът по оста."""
    Dm = D.value(d, "D")
    im = Image.new("RGB", (1000, 640), PAPER)
    sh = Sheet(im, (200, 100, 800, 560), Dm, Dm)
    t = np.linspace(0, 2 * math.pi, 96)
    sh.poly(np.c_[Dm / 2 + Dm / 2 * np.cos(t), Dm / 2 + Dm / 2 * np.sin(t)], fill=FILL, outline=INK, width=4)
    dim = d["dims"][dim_id]
    for f in d["features"]:
        if f["t"] == "hole" and f["axis"] == "x":
            r = D.value(d, f["d"]) / 2
            hit = f["d"] == dim_id
            sh.poly(np.c_[Dm / 2 + r * np.cos(t), Dm / 2 + r * np.sin(t)], fill=(255, 237, 224) if hit else PAPER,
                    outline=ACCENT if hit else INK, width=5 if hit else 3)  # fmt: skip
            if hit:
                value = mm(dim["v"]) if dim["src"] == "measured" else (f"≈ {mm(dim['est'])}" if dim.get("est") else "?")
                sh.dimension((Dm / 2 - r, Dm / 2), (Dm / 2 + r, Dm / 2), (0, -(Dm / 2) * sh.s - 40), "Ø " + value)
    caption(sh, "Изглед откъм края")
    return im


def caption(sh: Sheet, text: str):
    f = font(26)
    sh.draw.text((24, sh.im.height - 46), text, font=f, fill=SOFT)


# --- Върху снимката -------------------------------------------------------------------------------------------------


def to_view(d: dict, view: str, pts: np.ndarray, flips: dict, outer: np.ndarray | None = None) -> np.ndarray:
    """Точки в осите на детайла за изгледа (u, v) -> точки в мм на снимката на изгледа (обратното на
    assemble.oriented: огледалата са сами себе си обратни)."""
    U, V = view_size(d, view if view != "profile" else "profile")
    fu, fv = flips.get(view, [False, False])
    if view == "end":
        fu = not fu
    if outer is not None and view in ("side", "end"):
        from .assemble import resting_down

        if resting_down(outer, U, V):
            fu, fv = not fu, not fv
    p = np.asarray(pts, float)
    return np.c_[U - p[:, 0] if fu else p[:, 0], V - p[:, 1] if fv else p[:, 1]]


def overlay(d: dict, view: str, photo: Image.Image, transform: dict, flips: dict, side: int = 1100,
            outer: np.ndarray | None = None) -> Image.Image:
    """Снимката на изгледа с очертанието и отворите от описанието, изрязана около детайла."""
    from .vision import View

    v = View(np.zeros((0, 2)), [], transform["U"], transform["V"], transform["angle"], tuple(transform["centre"]),
             transform["x0"], transform["y1"], transform["sx"], transform["sy"], tuple(transform["image_size"]))  # fmt: skip
    shape_view = "profile" if d["kind"] == "turned" and view in ("top", "side") else view
    if d["kind"] == "turned" and view == "end":
        Dm = D.value(d, "D")
        t = np.linspace(0, 2 * math.pi, 96)
        rings = [np.c_[Dm / 2 + Dm / 2 * np.cos(t), Dm / 2 + Dm / 2 * np.sin(t)]]
        for f in d["features"]:
            if f["t"] == "hole" and f["axis"] == "x":
                r = D.value(d, f["d"]) / 2
                rings.append(np.c_[Dm / 2 + r * np.cos(t), Dm / 2 + r * np.sin(t)])
        polys = rings
    else:
        polys = [silhouette(d, shape_view)] + [p for _, p in features_in(d, shape_view)]
    im = photo.convert("RGB").copy()
    draw = ImageDraw.Draw(im, "RGBA")
    allpx = []
    width = max(3, round(max(im.size) / 450))
    for k, pts in enumerate(polys):
        px = v.to_photo(to_view(d, view, pts, flips, outer))
        allpx.append(px)
        xy = [tuple(p) for p in px] + [tuple(px[0])]
        draw.line(xy, fill=(255, 255, 255, 230), width=width + 4, joint="curve")
        draw.line(xy, fill=ACCENT + (255,), width=width, joint="curve")
    allpx = np.vstack(allpx)
    lo, hi = allpx.min(0), allpx.max(0)
    pad = 0.18 * float(max(hi - lo))
    box = (max(0, int(lo[0] - pad)), max(0, int(lo[1] - pad)), min(im.width, int(hi[0] + pad)),
           min(im.height, int(hi[1] + pad)))  # fmt: skip
    im = im.crop(box)
    im.thumbnail((side, side), Image.LANCZOS)
    return im


# --- Чертеж ---------------------------------------------------------------------------------------------------------


def drawing(d: dict) -> Image.Image:
    """Изгледите един до друг с общите размери и списък на всички размери (оценените със *)."""
    views = ["profile", "end"] if d["kind"] == "turned" else ["top", "side", "end"]
    tiles = []
    for view in views:
        if view == "end" and d["kind"] == "turned":
            tile = turned_end(d, next((f["d"] for f in d["features"] if f["t"] == "hole" and f["axis"] == "x"), "D"))
        else:
            sh = base_view(d, view)
            U, V = sh.U, sh.V
            sh.dimension((0, 0), (U, 0), (0, 60), mm(U), color=BLUE, width=3, size=26)
            sh.dimension((U, 0), (U, V), (60, 0), ("Ø " if view == "profile" else "") + mm(V), color=BLUE, width=3,
                         size=26)  # fmt: skip
            caption(sh, f"Изглед {D.VIEW_NAME[view]}")
            tile = sh.im
        tiles.append(tile)
    lines = ["Размери (мм); * = оценено от снимките, не измерено:"]
    for i in D.used_dims(d):
        dim = d["dims"][i]
        if dim.get("hidden"):
            continue
        star = "" if dim["src"] == "measured" else " *"
        lines.append(f"{i:>5}  {dim['name']}: {mm(dim['v'])}{star}")
    for f in d["features"]:
        if f["t"] in ("fillet", "chamfer"):
            lines.append(f"       {D.label(d, f)}")
        if f.get("thread"):
            lines.append(f"       Резба {f['thread']} в отвор {f['id']}")
    f = font(24)
    text_h = 34 * len(lines) + 40
    cols = 2
    W = 1000 * cols
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (W, 640 * rows + text_h), PAPER)
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * 1000, (k // cols) * 640))
    draw = ImageDraw.Draw(sheet)
    y = 640 * rows + 20
    for k, line in enumerate(lines):
        draw.text((40, y + 34 * k), line, font=font(26, bold=k == 0) if k == 0 else f, fill=INK)
    return sheet
