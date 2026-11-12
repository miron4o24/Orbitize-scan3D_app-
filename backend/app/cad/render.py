"""Картинки на инженерния режим (PIL, без външни услуги):

- question: това, което трябва да се измери (размерна линия, отвор, заоблен ъгъл) в оранжево, нарисувано върху
  снимката на човека, ако изгледът има снимка, иначе върху чертеж на изгледа; overall: общите размери върху снимките,
  преди да има описание;
- overlay: очертанието от описанието върху снимката (екранът "Какво виждам");
- drawing: чертеж с изгледите и всички размери (оценените със звездичка)."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
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
    return np.asarray(D.arc_points(pts, steps))


def view_size(d: dict, view: str) -> tuple[float, float]:
    if view == "profile":
        return D.total(d, "x"), D.value(d, "D")
    a, b = D.VIEW_AXES[view]
    return D.total(d, a), D.total(d, b)


def silhouette(d: dict, view: str) -> np.ndarray:
    """Очертанието на изгледа в мм (u, v), с дъгите."""
    if view == "profile":  # профилът (x, r) огледан около оста: горе R + r, долу R - r
        R = D.value(d, "D") / 2
        pts = arc_outline(D.outline(d, "profile"))  # с дъгите на заоблените ъгли
        top = Polygon(np.c_[pts[:, 0], R + pts[:, 1]]).buffer(0)
        bottom = Polygon(np.c_[pts[:, 0], R - pts[:, 1]]).buffer(0)
        shape = top.union(bottom)
        if shape.geom_type != "Polygon":
            shape = max(shape.geoms, key=lambda g: g.area)
        return np.asarray(shape.exterior.coords)[:-1]
    if d["views"].get(view, {}).get("round"):
        U, V = view_size(d, view)
        t = np.linspace(0, 2 * math.pi, 144, endpoint=False)
        return np.c_[U / 2 + U / 2 * np.cos(t), V / 2 + V / 2 * np.sin(t)]
    pts = arc_outline(D.outline(d, view))
    cuts = [p for f, p in features_in(d, view, channels=True) if f.get("open")]
    if not cuts:
        return pts
    shape = Polygon(pts).buffer(0)
    for c in cuts:  # каналът до стената е вдлъбнатина в очертанието
        shape = shape.difference(Polygon(c).buffer(0.01))
    if shape.is_empty:
        return pts
    if shape.geom_type != "Polygon":
        shape = max(shape.geoms, key=lambda g: g.area)
    return np.asarray(shape.exterior.coords)[:-1]


def features_in(d: dict, view: str, channels: bool = False) -> list[tuple[dict, np.ndarray]]:
    """Елементите, които се виждат в изгледа като очертания: (елемент, точки в мм). Каналите до стената (вече
    изрязани от очертанието в silhouette) - само с channels, като правоъгълника в детайла."""
    normal = VIEW_NORMAL[view]
    out = []
    for f in d["features"]:
        if f["t"] not in ("hole", "rect", "slot") or (f.get("open") and not channels) or not D.present(d, f):
            continue  # размер 0: елементът го няма (отвор от отблясък, махнат с 0) - и не се рисува
        place = D.feature_place(d, f)
        if f["axis"] != normal:
            continue
        cu, cv = place["u"], place["v"]
        if f["t"] == "hole":
            r = D.value(d, f["d"]) / 2
            t = np.linspace(0, 2 * math.pi, 48)
            out.append((f, np.c_[cu + r * np.cos(t), cv + r * np.sin(t)]))
        elif f["t"] == "rect" and f.get("shape"):
            w, h = D.value(d, f["w"]), D.value(d, f["h"])
            out.append((f, np.array([(cu + (a - 0.5) * w, cv + (b - 0.5) * h) for a, b in f["shape"]])))
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

    halo = False  # бял кант под линиите (върху снимка)

    def __init__(self, im: Image.Image, box: tuple[int, int, int, int], U: float, V: float):
        self.im, self.draw = im, ImageDraw.Draw(im)
        x0, y0, x1, y1 = box
        self.s = min((x1 - x0) / U, (y1 - y0) / V)
        self.ox = x0 + ((x1 - x0) - U * self.s) / 2
        self.oy = y1 - ((y1 - y0) - V * self.s) / 2
        self.U, self.V = U, V

    def px(self, u: float, v: float) -> tuple[float, float]:
        return self.ox + u * self.s, self.oy - v * self.s

    def shift(self, a, offset) -> np.ndarray:
        """Точката a (мм), изместена с offset пиксела, както на изправен чертеж (x надясно, y надолу)."""
        return np.array(self.px(*a)) + np.asarray(offset, float)

    def stroke(self, xy: list, fill, width: int):
        if self.halo:
            self.draw.line(xy, fill=(255, 255, 255), width=width + 4, joint="curve")
        self.draw.line(xy, fill=fill, width=width, joint="curve")

    def poly(self, pts: np.ndarray, fill=None, outline=INK, width=3):
        xy = [self.px(u, v) for u, v in pts]
        self.draw.polygon(xy, fill=fill)
        self.draw.line(xy + [xy[0]], fill=outline, width=width, joint="curve")

    def line(self, a, b, fill=INK, width=2, dash: float | None = None):
        pa, pb = self.px(*a), self.px(*b)
        if not dash:
            self.stroke([pa, pb], fill, width)
            return
        n = max(1, int(math.dist(pa, pb) / dash))
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1.0, (k + 1) / n)
            self.stroke([(pa[0] + (pb[0] - pa[0]) * t0, pa[1] + (pb[1] - pa[1]) * t0),
                         (pa[0] + (pb[0] - pa[0]) * t1, pa[1] + (pb[1] - pa[1]) * t1)], fill, width)  # fmt: skip

    def dimension(self, a, b, offset: tuple[float, float], text: str, color=ACCENT, width=4, size=30):
        """Размерна линия от a до b (мм), изместена с offset (пиксели), с текста по средата."""
        pa, pb = np.array(self.px(*a)), np.array(self.px(*b))
        qa, qb = self.shift(a, offset), self.shift(b, offset)
        off = qa - pa
        n_off = max(float(np.linalg.norm(off)), 1.0)
        if np.linalg.norm(off) > 0:  # изводни линии
            for p, q in ((pa, qa), (pb, qb)):
                self.stroke([tuple(p), tuple(q + off / n_off * 8)], color, 2)
        self.stroke([tuple(qa), tuple(qb)], color, width)
        d = qb - qa
        n = np.linalg.norm(d)
        if n > 1:
            d = d / n
            for p, s in ((qa, 1), (qb, -1)):
                tip = p
                back = p + s * d * 18
                side = np.array([-d[1], d[0]]) * 8
                self.draw.polygon([tuple(tip), tuple(back + side), tuple(back - side)], fill=color)
        mid = (qa + qb) / 2 + (off / n_off) * 26
        label(self.draw, tuple(mid), text, color, size)


class PhotoSheet(Sheet):
    """Изглед в мм върху снимката: A (2 x 3) е от мм на изгледа към пиксели на снимката. Снимката се изрязва около
    детайла (с място за размерните линии), смалява се до `size` и леко се притъмнява, за да се виждат линиите."""

    halo = True

    def __init__(self, photo: Image.Image, A, U: float, V: float, size=(1000, 750), margin=0.42, dim: bool = True):
        A = np.asarray(A, float)
        corners = np.array([[0, 0], [U, 0], [U, V], [0, V]], float) @ A[:, :2].T + A[:, 2]
        lo, hi = corners.min(0), corners.max(0)
        span = float(max(hi - lo))
        lo, hi = lo - margin * span, hi + margin * span
        w, h = hi - lo
        aspect = size[0] / size[1]
        if w / h < aspect:
            lo[0] -= (h * aspect - w) / 2
            w = h * aspect
        else:
            lo[1] -= (w / aspect - h) / 2
            h = w / aspect
        k = size[0] / w
        box = (int(round(lo[0])), int(round(lo[1])), int(round(lo[0] + w)), int(round(lo[1] + h)))
        im = photo.convert("RGB").crop(box).resize(size, Image.LANCZOS)
        if dim:
            im = ImageEnhance.Brightness(ImageEnhance.Color(im).enhance(0.7)).enhance(0.88)
        self.im, self.draw = im, ImageDraw.Draw(im, "RGBA")
        k = size[0] / (box[2] - box[0])
        self.A = np.c_[A[:, :2] * k, (A[:, 2] - np.array([box[0], box[1]], float)) * k]
        self.U, self.V = U, V
        self.s = k * math.sqrt(abs(float(np.linalg.det(A[:, :2]))))

    def px(self, u: float, v: float) -> tuple[float, float]:
        p = self.A[:, :2] @ np.array([u, v], float) + self.A[:, 2]
        return float(p[0]), float(p[1])

    def shift(self, a, offset) -> np.ndarray:
        # Отместването е "навън" в осите на изгледа (мм), а не на екрана: снимката може да е завъртяна и огледална.
        off = np.array([offset[0], -offset[1]], float) / self.s
        return np.array(self.px(*(np.asarray(a, float) + off)))

    def poly(self, pts: np.ndarray, fill=None, outline=INK, width=3):
        xy = [self.px(u, v) for u, v in pts]
        if fill is not None and fill != PAPER:
            self.draw.polygon(xy, fill=ACCENT + (70,))
        self.stroke(xy + [xy[0]], outline, width)


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
    for x0, x1 in split_lines(d, view):  # граница между части (капачка и тяло): отдолу x0, отгоре x1
        sh.line((x0, -V * 0.04), (x1, V * 1.04), BLUE, 3, dash=16)
    slant, sections = narrow_marks(d, view)
    for x0, x1 in slant:  # наклоненият край на тялото до по-тесния край
        sh.line((x0, 0), (x1, V), INK, 3)
    for pts in sections:  # сечението на по-тесния край
        sh.poly(pts, outline=INK, width=2)
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


def split_lines(d: dict, view: str) -> list[tuple[float, float]]:
    """Границите между части в изглед, в който X е хоризонталата: (X долу, X горе) на изгледа. Отгоре се вижда
    горната страна; отпред (и в профила) границата може да е по диагонал от долната към горната страна."""
    if view not in ("top", "side", "profile"):
        return []
    out = []
    for f in d["features"]:
        if f["t"] == "split":
            try:
                top = D.coord(d, f["at"])
                low = D.coord(d, f["at2"]) if f.get("at2") else top
            except D.DesignError:
                continue
            out.append((top, top) if view == "top" else (low, top))
    return out


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


def value_text(dim: dict, star: bool = False) -> str:
    """Измереното число, иначе оценката с ≈ (на чертежа: числото със *)."""
    if dim.get("v") is not None and (dim["src"] == "measured" or star):
        return mm(dim["v"]) + ("*" if star and dim["src"] != "measured" else "")
    return f"≈ {mm(dim['est'])}" if dim.get("est") else "?"


def annotate(sh: Sheet, d: dict, view: str, dim_id: str, text: str, color=ACCENT, slot: int = 0,
             highlight: bool = True, width: int = 4, size: int = 30) -> str:  # fmt: skip
    """Рисува размера `dim_id` в изгледа: размерна линия (общите размери и нивата, на ред `slot` встрани от
    детайла), Ø при отвор, R при заоблен ъгъл. highlight: и мястото в цвят (въпросът). Връща накъде отиде
    линията: "u" (над детайла), "v" (вдясно) или "" (до елемента)."""
    dim = d["dims"][dim_id]
    show = dim.get("show") or {}
    U, V = sh.U, sh.V
    a, b = ("x", "r") if view == "profile" else D.VIEW_AXES[view]
    feature = next((f for f in d["features"] if f["id"] == show.get("feature")), None)
    up = (0, -(60 + 56 * slot))
    right = (60 + 86 * slot, 0)
    if dim_id in ("L", "W", "H", "D"):
        if dim_id == "L":
            sh.dimension((0, 0), (U, 0), (0, 60), text, color, width, size)
            return ""
        sh.dimension((U, 0), (U, V), right, ("Ø " if dim_id == "D" else "") + text, color, width, size)
        return "v"
    if show.get("level") and show["level"] in d["levels"]:
        lvl = d["levels"][show["level"]]
        c = D.coord(d, show["level"])
        if lvl["axis"] == "r":  # диаметър на детайл от струг: на мястото, където го има
            xs = [u for u, r, _ in D.outline(d, "profile") if abs(r - c) < 1e-6]
            x = (min(xs) + max(xs)) / 2 if xs else U / 2
            sh.dimension((x, V / 2 - c), (x, V / 2 + c), (0, 0), "Ø " + text, color, width, size)
            return ""
        on_u = lvl["axis"] == a
        ref = 0.0 if lvl["from"] == "start" else (U if on_u else V)
        if on_u:
            if highlight:
                sh.line((c, -V * 0.08), (c, V * 1.08), color, 3, dash=12)
            sh.dimension((ref, V), (c, V), up, text, color, width, size)
            return "u"
        if highlight:
            sh.line((-U * 0.05, c), (U * 1.05, c), color, 3, dash=12)
        if lvl["from"] == "start":  # от долния (предния) ръб: вляво, за да не се трупа с другите вдясно
            sh.dimension((0, ref), (0, c), (-(60 + 86 * slot), 0), text, color, width, size)
            return "vl"
        sh.dimension((U, ref), (U, c), right, text, color, width, size)
        return "v"
    if feature is not None and feature["t"] in ("hole", "rect", "slot"):
        if highlight:
            for f, pts in features_in(d, view, channels=True):
                if f is feature or (f["t"] == "hole" and f.get("d") == dim_id):
                    sh.poly(pts, fill=(255, 237, 224), outline=color, width=5)
        place = D.feature_place(d, feature)
        cu, cv = place["u"], place["v"]
        key = next((k for k in ("d", "w", "h", "len", "depth", "r") if feature.get(k) == dim_id), None)
        if key == "d":
            r = D.value(d, feature["d"]) / 2
            if highlight:
                sh.dimension((cu - r, cv), (cu + r, cv), (0, -max(60, r * sh.s + 50)), "Ø " + text, color, width, size)
            else:
                x, y = sh.px(cu, cv + r)
                label(sh.draw, (x, y - 26), "Ø " + text, color, size)
        elif key in ("w", "h", "len"):
            along = show.get("along") or (a if key in ("w", "len") else b)
            if feature["t"] == "slot":
                along_u = (feature.get("angle", 0) == 0) == (key == "len")
                size_mm = D.value(d, feature["len"] if key == "len" else feature["w"])
            else:
                size_mm = D.value(d, feature[key])
                along_u = along == a
            if along_u:
                sh.dimension((cu - size_mm / 2, cv), (cu + size_mm / 2, cv), (0, -50), text, color, width, size)
            else:
                sh.dimension((cu, cv - size_mm / 2), (cu, cv + size_mm / 2), (50, 0), text, color, width, size)
        else:
            what = "дълбочина" if key == "depth" else "R"
            x, y = sh.px(cu, cv)
            label(sh.draw, (x, y + 40), f"{what} {text}", color, size)
        return ""
    if dim["kind"] == "radius" or show.get("radius"):
        rows = d["views"][view]["pts"]
        outline = D.outline(d, view)
        if view == "profile":  # профилът е (x, радиус): на изгледа горната половина е R + r, долната R - r
            halves = [[(u, V / 2 + r, rr) for u, r, rr in outline], [(u, V / 2 - r, rr) for u, r, rr in outline]]
        else:
            halves = [outline]
        done = False
        for pts in halves:
            for k, ((u, v, r), row) in enumerate(zip(pts, rows)):
                if row[2] != dim_id or done:
                    continue
                pu, pv, _ = pts[k - 1]
                nu, nv, _ = pts[(k + 1) % len(pts)]
                arc = arc_outline([(pu, pv, 0.0), (u, v, r), (nu, nv, 0.0)], steps=16)[1:-1]
                if len(arc) >= 2:
                    xy = [sh.px(q[0], q[1]) for q in arc]
                    if highlight:
                        sh.stroke(xy, color, 9)
                    mid = xy[len(xy) // 2]
                    label(sh.draw, (mid[0] + (40 if u > U / 2 else -40), mid[1] + (34 if v < V / 2 else -34)),
                          "R " + text, color, size)  # fmt: skip
                    done = not highlight
        return ""
    if feature is not None:  # заобляне или фаска
        label(sh.draw, (sh.im.width / 2, 40), f"{D.label(d, feature)}: {text}", color, size)
    return ""


def question(d: dict, dim_id: str, photos: dict | None = None) -> Image.Image:
    """Картинката към въпроса за размера `dim_id`: върху снимката на изгледа, ако я има (photos: {изглед: {"image",
    "A", "name"}} от service), иначе върху чертеж на изгледа."""
    view = view_for(d, dim_id)
    photos = photos or {}
    if view == "end" and d["kind"] == "turned":
        if "end" in photos:
            Dm = D.value(d, "D")
            sh = PhotoSheet(photos["end"]["image"], photos["end"]["A"], Dm, Dm)
            turned_end_marks(sh, d, dim_id)
            photo_caption(sh, photos["end"]["name"])
            return sh.im
        return turned_end(d, dim_id)
    src = photos.get("top" if view == "profile" else view)
    if src is not None:
        U, V = view_size(d, view)
        sh = PhotoSheet(src["image"], src["A"], U, V)
        annotate(sh, d, view, dim_id, value_text(d["dims"][dim_id]))
        photo_caption(sh, src["name"])
        return sh.im
    sh = base_view(d, view)
    annotate(sh, d, view, dim_id, value_text(d["dims"][dim_id]))
    caption(sh, f"Изглед {D.VIEW_NAME[view]}")
    return sh.im


def photo_caption(sh: Sheet, name: str):
    from .photos import TITLE

    f = font(24, bold=True)
    text = f"Снимка {TITLE.get(name, name)}"
    l, t, r, b = sh.draw.textbbox((0, 0), text, font=f)
    label(sh.draw, (24 + (r - l) / 2 + 10, sh.im.height - 36), text, INK, 24)


def overall(photo: Image.Image | None, A, U: float, V: float, dim: str, along: str, name: str = "") -> Image.Image:
    """Общ размер върху снимката (преди да има описание): линия по цялата дължина ("u") или височина ("v") на
    изгледа, а диаметърът през средата. Без снимка (височина без снимка отстрани): схема."""
    text = "Ø ?" if dim == "D" else "?"
    if photo is None:
        im = Image.new("RGB", (1000, 750), PAPER)
        sh = Sheet(im, (170, 250, 830, 560), 3.0, 1.0)
        sh.poly(np.array([[0, 0], [3, 0], [3, 1], [0, 1]], float), fill=FILL, outline=INK, width=4)
        sh.line((-0.4, 0), (3.4, 0), SOFT, 3)
        sh.dimension((3, 0), (3, 1), (70, 0), text)
        caption(sh, "Отстрани: от масата до най-високата точка")
        return im
    sh = PhotoSheet(photo, A, U, V)
    if dim == "D":
        sh.dimension((0, V / 2), (U, V / 2), (0, 0), text)
    elif along == "u":
        sh.dimension((0, 0), (U, 0), (0, 70), text)
    else:
        sh.dimension((U, 0), (U, V), (70, 0), text)
    photo_caption(sh, name)
    return sh.im


def turned_end(d: dict, dim_id: str, star: bool = False, size=(1000, 640)) -> Image.Image:
    """Детайл от струг откъм края: кръгът и отворът по оста. star: за чертежа (синьо, без оцветяване)."""
    Dm = D.value(d, "D")
    im = Image.new("RGB", size, PAPER)
    m = round(size[1] * 0.16)
    sh = Sheet(im, (m, m, size[0] - m, size[1] - m), Dm, Dm)
    t = np.linspace(0, 2 * math.pi, 96)
    sh.poly(np.c_[Dm / 2 + Dm / 2 * np.cos(t), Dm / 2 + Dm / 2 * np.sin(t)], fill=FILL, outline=INK, width=4)
    turned_end_marks(sh, d, dim_id, star)
    caption(sh, "Изглед отстрани")
    return im


def turned_end_marks(sh: Sheet, d: dict, dim_id: str, star: bool = False):
    """Отворите по оста в кръга на края; този с размера `dim_id` е оцветен и с размерна линия."""
    Dm = D.value(d, "D")
    color = BLUE if star else ACCENT
    t = np.linspace(0, 2 * math.pi, 96)
    dim = d["dims"][dim_id]
    for f in d["features"]:
        if f["t"] == "hole" and f["axis"] == "x":
            r = D.value(d, f["d"]) / 2
            hit = f["d"] == dim_id
            mark = hit and not star
            sh.poly(np.c_[Dm / 2 + r * np.cos(t), Dm / 2 + r * np.sin(t)], fill=(255, 237, 224) if mark else PAPER,
                    outline=ACCENT if mark else INK, width=5 if mark else 3)  # fmt: skip
            if hit:
                value = value_text(dim, star)
                off = (0, -(Dm / 2) * sh.s - 40)
                sh.dimension((Dm / 2 - r, Dm / 2), (Dm / 2 + r, Dm / 2), off, "Ø " + value, color)


def caption(sh: Sheet, text: str):
    f = font(26)
    sh.draw.text((24, sh.im.height - 46), text, font=f, fill=SOFT)


# --- Върху снимката -------------------------------------------------------------------------------------------------


def overlay_shapes(d: dict, view: str, face: str | None = None) -> tuple[float, float, list]:
    """Какво се рисува върху снимката: размерите на изгледа и [(елемент или None за очертанието, точки в мм)].
    face: стената на снимката - глух отвор от другата стена не се вижда на нея."""
    if d["kind"] == "turned" and view == "end":
        Dm = D.value(d, "D")
        t = np.linspace(0, 2 * math.pi, 96)
        polys = [(None, np.c_[Dm / 2 + Dm / 2 * np.cos(t), Dm / 2 + Dm / 2 * np.sin(t)])]
        for f in d["features"]:
            if f["t"] == "hole" and f["axis"] == "x" and D.present(d, f):
                r = D.value(d, f["d"]) / 2
                polys.append((f, np.c_[Dm / 2 + r * np.cos(t), Dm / 2 + r * np.sin(t)]))
        return Dm, Dm, polys
    shape_view = "profile" if d["kind"] == "turned" else view
    U, V = view_size(d, shape_view)
    polys = [(None, silhouette(d, shape_view))] + [(f, p) for f, p in features_in(d, shape_view)
                                                   if not (face and f.get("depth") and f.get("face") != face)]
    return U, V, polys + [(None, p) for p in narrow_marks(d, shape_view, face)[1]]


def narrow_marks(d: dict, view: str, face: str | None = None) -> tuple[list, list]:
    """По-тесните краища (D.narrow_ends) в изгледа: наклоненият край на тялото отпред [(x долу, x горе)] и
    сечението на по-тесния край отстрани [точки в мм] (само на снимката на неговия край, ако е дадена)."""
    lines, sections = [], []
    if d["kind"] != "block":
        return lines, sections
    for n in D.narrow_ends(d):
        try:
            xb, xt = D.coord(d, n["bottom"]), D.coord(d, n["top"])
            y0, y1, z0, z1 = (D.coord(d, n[k][i]) for k in ("y", "z") for i in (0, 1))
        except D.DesignError:
            continue
        if view == "side" and abs(xb - xt) > 1e-6:
            lines.append((xb, xt))
        if view == "end" and n["round"] and (face is None or face == ("right" if n["sign"] == "+" else "left")):
            W, H = view_size(d, "end")
            pts = silhouette(d, "end")
            sections.append(np.c_[y0 + pts[:, 0] * (y1 - y0) / W, z0 + pts[:, 1] * (z1 - z0) / H])
    return lines, sections


def overlay_sheet(d: dict, view: str, photo: Image.Image, A, side: int = 1100, face: str | None = None):
    U, V, polys = overlay_shapes(d, view, face)
    return PhotoSheet(photo, A, U, V, size=(side, round(side * 0.75)), margin=0.16, dim=False), U, V, polys


def overlay(d: dict, view: str, photo: Image.Image, A, side: int = 1100, face: str | None = None) -> Image.Image:
    """Снимката с очертанието и отворите от описанието (изгледът `view` на описанието, A: от мм към снимката)."""
    sh, U, V, polys = overlay_sheet(d, view, photo, A, side, face)
    for _, pts in polys:
        xy = [sh.px(u, v) for u, v in pts]
        sh.stroke(xy + [xy[0]], ACCENT, 4)
    if not (d["kind"] == "turned" and view == "end"):
        for x0, x1 in split_lines(d, "profile" if d["kind"] == "turned" else view):
            sh.line((x0, 0), (x1, V), ACCENT, 3, dash=14)
        for x0, x1 in narrow_marks(d, view)[0]:  # наклоненият край на тялото до по-тесния край
            sh.line((x0, 0), (x1, V), ACCENT, 3)
    return sh.im


TAP_NEAR = 0.035  # докосване до толкова от ширината на снимката от ръба на отвор е върху него (пръстът е груб)


def feature_at(d: dict, view: str, photo: Image.Image, A, x: float, y: float, face: str | None = None) -> str | None:
    """Елементът (отвор, джоб, канал), докоснат на снимката с очертанието (overlay) в x, y от 0 до 1: вътре в него
    или до TAP_NEAR от ръба му; от няколко - най-малкият (отвор в джоб). None: там няма елемент."""
    from shapely.geometry import Point

    sh, _, _, polys = overlay_sheet(d, view, photo, A, face=face)
    w, h = sh.im.size
    at = Point(x * w, y * h)
    best = None
    for f, pts in polys:
        if f is None or f.get("open"):
            continue  # каналът до стената е вдлъбнатина в очертанието: маха се от списъка
        shape = Polygon([sh.px(u, v) for u, v in pts]).buffer(0)
        if shape.is_empty or shape.distance(at) > TAP_NEAR * w:
            continue
        if best is None or shape.area < best[1]:
            best = (f["id"], shape.area)
    return best[0] if best else None


# --- Чертеж ---------------------------------------------------------------------------------------------------------


def drawing(d: dict) -> Image.Image:
    """Изгледите с всички размери по тях (оценените със *) и списъкът на размерите отдолу."""
    views = ["profile", "end"] if d["kind"] == "turned" else ["top", "side", "end"]
    tiles = []
    TILE = (1100, 820)
    for view in views:
        if view == "end" and d["kind"] == "turned":
            bore = next((f["d"] for f in d["features"] if f["t"] == "hole" and f["axis"] == "x"), None)
            tile = turned_end(d, bore or "D", star=True, size=TILE)
        else:
            sh = base_view(d, view, size=TILE, margin=210)
            slots = {"u": 0, "v": 1, "vl": 0}
            for i in D.used_dims(d):
                dim = d["dims"][i]
                if dim.get("hidden"):
                    continue
                if i in ("L", "W", "H", "D"):
                    shown = {"L": ("top", "profile"), "W": ("top",), "H": ("side",), "D": ("profile",)}[i]
                    if view in shown:
                        annotate(sh, d, view, i, value_text(dim, star=True), BLUE, 0, False, 3, 24)
                    continue
                if view_for(d, i) != view:
                    continue
                side = level_side(d, view, i)
                went = annotate(sh, d, view, i, value_text(dim, star=True), BLUE, slots.get(side, 0), False, 3, 24)
                if went:
                    slots[went] = slots.get(went, 0) + 1
            caption(sh, f"Изглед {D.VIEW_NAME[view]}")
            tile = sh.im
        tiles.append(tile)
    lines = ["Размери (мм); * = оценено от снимките, не измерено с шублер:"]
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
    W = TILE[0] * cols
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (W, TILE[1] * rows + text_h), PAPER)
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * TILE[0], (k // cols) * TILE[1]))
    draw = ImageDraw.Draw(sheet)
    y = TILE[1] * rows + 20
    for k, line in enumerate(lines):
        draw.text((40, y + 34 * k), line, font=font(26, bold=True) if k == 0 else f, fill=INK)
    return sheet


def level_side(d: dict, view: str, dim_id: str) -> str:
    """Накъде отива размерната линия на ниво: "u" (над детайла), "v" (вдясно) или "" (диаметър)."""
    show = d["dims"][dim_id].get("show") or {}
    lvl = d["levels"].get(show.get("level") or "")
    if not lvl or lvl["axis"] == "r":
        return ""
    a = "x" if view == "profile" else D.VIEW_AXES[view][0]
    if lvl["axis"] == a:
        return "u"
    return "vl" if lvl["from"] == "start" else "v"
