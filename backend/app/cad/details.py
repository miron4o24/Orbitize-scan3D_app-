"""Детайлите по стените на предмета, видени на снимките (екранът, портовете, бутонът, светодиодът): MobileSAM по
решетка от точки върху детайла дава отделните предмети по стената; остават тези с правилна форма (кръг,
правоъгълник, канал), ясен ръб спрямо стената около тях и не до ръба на детайла. Отблясъкът (светъл и безцветен)
и печатът (между буквите е цветът на стената, тесни редове текст) не минават. Кръгъл бутон, тъмен като стената,
MobileSAM не отделя: него го дават кръговете по ръбовете (Hough) с почти пълна окръжност.

Всеки детайл е глух елемент от стената си с дълбочина за мерене (оценката е приблизителна): порт (тъмен или цветен
отвор), прозорец/панел (голям: екранът), бутон (кръг), отворче (малко)."""

import math

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import Point, Polygon

GRID = 7  # точки по по-късата страна на стената
AREA = (0.0025, 0.5)  # детайл: толкова от стената
SOLID = 0.85  # запълва поне толкова от изпъкналата си обвивка
EDGE = 0.012  # и е поне толкова (от по-късата страна) навътре от ръба
CONTRAST = 12.0  # ръбът му: средният цвят вътре и около него се различават поне толкова (Lab)
SAME_INSIDE = 0.45  # печат: толкова от вътрешността е с цвета на стената (празното между буквите)
LONG = 4.5  # канал, по-дълъг от толкова ширини, е ред текст
GLARE_L = 12.0  # отблясък или бял печат: вътре е толкова по-светло и почти без цвят
GLARE_CHROMA = 12.0
PANEL = 0.1  # детайл, голям поне толкова от стената: прозорец или панел (екран)
PORT_DARK = 10.0  # порт: вътре е толкова по-тъмно от стената
HOLE_DARK = 25.0  # отвор без цвят: вътре е поне толкова по-тъмно
HOLE_DARK_SMALL = 12.0  # а малко отворче (високоговорител, микрофон) - поне толкова
GAP_L = 2.5  # тъмният пръстен около езичето на USB е поне толкова по-тъмен от стената по-нататък
DOT = 0.012  # по-малък детайл (от стената): отворче или светодиод
PORT_LONG = 2.0  # освен ако е поне толкова пъти по-дълъг, отколкото широк: порт
BUTTON_R = (0.06, 0.2)  # бутонът: радиус толкова от по-късата страна на стената
BUTTON_RING = 0.85  # и ръбът минава по толкова от окръжността му
BUTTON_PLAIN = 0.06  # а вътре е гладък: ръбове по толкова от лицето му (кръгче в текст не е бутон)
SAME = 0.5  # два детайла с толкова общо са един
REGULAR = 0.88  # детайлът съвпада поне толкова с кръг или заоблен правоъгълник
COLOR_PX = 20.0  # цветен пиксел (наситеност в Lab)
COLORFUL = 0.25  # детайл с толкова цветни пиксели е цветен (езичето на USB, лого)
SHADE_L = 8.0  # пиксел, по-тъмен от стената с толкова (L), е от отвора
SHADED = 0.15  # цветен порт има поне толкова такива пиксели (тъмното около езичето), а печатът - не
BUTTON_PX = 12  # бутонът: радиус поне толкова пиксела
GROOVE = 0.012  # процепът около бутона е поне толкова по-тъмен (яркост 0..1) от вътре и от вън
GROOVED = 0.6  # по толкова от окръжността
BUTTON_FACE = 150  # и стената е поне толкова пиксела напряко
BUTTON_LONG = 6.0  # и не по-дълга от толкова пъти ширината си (химикалка)
DARK_FACE = 45.0  # стена, по-тъмна от това (L), е тъмна: цветно с тъмна цепка там е порт


def sam_masks(models, crop: Image.Image, points: list) -> list[np.ndarray]:
    """Маските на MobileSAM за всяка точка поотделно (без кутия: и малките части), с едно вграждане."""
    from .. import refine as RF

    enc, dec = models
    emb = RF.embedding(enc, crop)
    W, H = crop.size
    k = RF.SIZE / max(W, H)
    out = []
    for x, y in points:
        feed = {"image_embeddings": emb, "point_coords": (np.array([[x, y], [0, 0]], np.float32) * k)[None],
                "point_labels": np.array([1.0, -1.0], np.float32)[None],
                "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
                "orig_im_size": np.array([H, W], np.float32)}  # fmt: skip
        out += [m > 0 for m in dec.run(None, feed)[0][0]]
    return out


def outline_of(mask: np.ndarray) -> Polygon | None:
    from skimage import measure

    cs = measure.find_contours(np.pad(mask.astype(float), 1), 0.5)
    if not cs:
        return None
    poly = Polygon(max(cs, key=len)[:, ::-1] - 1).buffer(0)
    return poly if poly.geom_type == "Polygon" and poly.area > 0 else None


def judge(lab: np.ndarray, mask: np.ndarray) -> dict | None:
    """Цветът вътре и около маската: None, ако няма ясен ръб, или е отблясък или печат."""
    ring = ndimage.binary_dilation(mask, iterations=6) & ~ndimage.binary_dilation(mask, iterations=2)
    core = ndimage.binary_erosion(mask, iterations=2)
    if core.sum() < 30 or ring.sum() < 30:
        return None
    inside, around = np.median(lab[core], 0), np.median(lab[ring], 0)
    contrast = float(np.linalg.norm(inside - around))
    if contrast < CONTRAST:
        return None
    if inside[0] - around[0] > GLARE_L and math.hypot(inside[1], inside[2]) < GLARE_CHROMA:
        return None  # отблясък или светъл печат
    same = float((np.linalg.norm(lab[core] - around, axis=1) < 10).mean())
    if same > SAME_INSIDE:
        return None  # печат: между буквите е стената
    # Колко от пикселите вътре са цветни (не средният цвят: многоцветно лого е средно сиво, а пикселите му са цветни).
    colorful = float((np.hypot(lab[core][:, 1], lab[core][:, 2]) > COLOR_PX).mean())
    shade = float((lab[core][:, 0] < around[0] - SHADE_L).mean())  # по-тъмното от стената: отворът около езичето
    # MobileSAM често дава само езичето на USB: тогава отворът е тъмният пръстен точно около него, по-тъмен от
    # стената по-нататък. Печатът (лого) няма такъв.
    near = ndimage.binary_dilation(mask, iterations=4) & ~mask
    far = ndimage.binary_dilation(mask, iterations=14) & ~ndimage.binary_dilation(mask, iterations=8)
    gap = float(np.median(lab[far][:, 0]) - np.median(lab[near][:, 0])) if near.sum() >= 30 and far.sum() >= 30 else 0.0
    return {"contrast": contrast, "dark": float(around[0] - inside[0]), "colorful": colorful, "shade": shade,
            "gap": gap, "around": float(around[0])}  # fmt: skip


def kind_of(seen: dict, share: float, poly: Polygon) -> str | None:
    """Какво е детайлът по цвета вътре и около него (None: печат или лого). Светла стена (бяла флашка, сребрист
    телефон): отворите са ясно по-тъмни и сиви, а цветното и бледото е печат. Тъмна стена (черен powerbank): по-тъмно
    или с цветни части вътре (езичето на USB, контактите) е порт, а със същия цвят - бутон."""
    colorful = seen["colorful"] >= COLORFUL
    if share >= PANEL:
        return None if colorful else "panel"  # екранът или панелът; цветното голямо петно е лого
    x0, y0, x1, y1 = poly.bounds
    longish = max(x1 - x0, y1 - y0) >= PORT_LONG * min(x1 - x0, y1 - y0)
    small = share < DOT and not longish  # отворче или светодиод; USB-C на голяма стена е малък, но дълъг
    if seen["around"] >= DARK_FACE:
        if seen["dark"] >= (HOLE_DARK_SMALL if small else HOLE_DARK) and not colorful:
            return "dot" if small else "port"  # (решетката на високоговорителя е по-светла от отвор)
        return None
    opened = seen["shade"] >= SHADED or seen["gap"] >= GAP_L  # тъмно вътре или тъмен пръстен около
    if seen["dark"] >= PORT_DARK or (colorful and opened):  # цветно лого без отвор около него - не
        return "dot" if small else "port"
    if colorful:
        return None
    return None if abs(seen["dark"]) >= PORT_DARK else ("dot" if small else "button")


def regular(poly: Polygon) -> bool:
    """Кръг или правоъгълник със заоблени ъгли (екран, порт, бутон), а не случайно петно (стърготини в прозрачна
    пластмаса, сянка): съвпада с единия поне REGULAR."""
    from shapely import affinity

    c = poly.centroid
    circle = c.buffer(math.sqrt(poly.area / math.pi), 48)
    best = poly.intersection(circle).area / poly.union(circle).area
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    ang = math.degrees(math.atan2(rect[1][1] - rect[0][1], rect[1][0] - rect[0][0]))
    flat = affinity.rotate(poly, -ang, origin=c)
    x0, y0, x1, y1 = flat.bounds
    gaps = sorted(flat.exterior.distance(Point(x, y)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    r = min(gaps[1] / (math.sqrt(2) - 1), 0.499 * min(x1 - x0, y1 - y0))
    box = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    model = box.buffer(-r).buffer(r, 16) if r > 0 else box
    if not model.is_empty:
        best = max(best, flat.intersection(model).area / flat.union(model).area)
    return best >= REGULAR


def circles(gray: np.ndarray, part: Polygon, short: float) -> list[tuple[float, float, float, float]]:
    """Кръговете по ръбовете (Hough) в детайла: (x, y, r, дял от окръжността по ръб). Ръбовете по два начина:
    с обичайните прагове (бутон на снимка с шум и текстура) и по относителни (тъмен ръб на тъмна стена)."""
    from skimage import feature

    out: list[tuple[float, float, float, float]] = []
    for edges in (feature.canny(gray, sigma=2.0),
                  feature.canny(gray, sigma=2.0, low_threshold=0.8, high_threshold=0.92, use_quantiles=True)):
        for c in ring_circles(edges, part, short, gray):
            if all(math.hypot(c[0] - o[0], c[1] - o[1]) > 0.5 * max(c[2], o[2]) for o in out):
                out.append(c)
    return out


def groove(gray: np.ndarray, x: float, y: float, r: float) -> float:
    """Колко от окръжността е процеп: по-тъмна от вътре и от вън (бутонът е отделен с тъмна цепка; щампован надпис
    и драскотина имат ръб само от едната страна)."""
    t = np.linspace(0, 2 * math.pi, 72, endpoint=False)

    def at(k):
        px = np.clip((x + k * r * np.cos(t)).astype(int), 0, gray.shape[1] - 1)
        py = np.clip((y + k * r * np.sin(t)).astype(int), 0, gray.shape[0] - 1)
        return gray[py, px]

    ring = np.minimum.reduce([at(k) for k in (0.95, 1.0, 1.05)])
    return float((ring < np.minimum(at(0.7), at(1.3)) - GROOVE).mean())


def ring_circles(edges: np.ndarray, part: Polygon, short: float,
                 gray: np.ndarray) -> list[tuple[float, float, float, float]]:
    from skimage import transform

    radii = np.arange(max(BUTTON_PX, int(BUTTON_R[0] * short)), max(BUTTON_PX + 2, int(BUTTON_R[1] * short)), 2)
    if len(radii) == 0 or not edges.any():
        return []
    hough = transform.hough_circle(edges, radii)
    _, xs, ys, rs = transform.hough_circle_peaks(hough, radii, total_num_peaks=8, min_xdistance=int(radii[0]),
                                                 min_ydistance=int(radii[0]))  # fmt: skip
    near = ndimage.binary_dilation(edges, iterations=2)
    out = []
    for x, y, r in zip(xs, ys, rs):
        t = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        px = np.clip((x + r * np.cos(t)).astype(int), 0, edges.shape[1] - 1)
        py = np.clip((y + r * np.sin(t)).astype(int), 0, edges.shape[0] - 1)
        ring = float(near[py, px].mean())
        if ring < BUTTON_RING or not part.contains(Point(x, y).buffer(r)):
            continue
        yy, xx = np.ogrid[: edges.shape[0], : edges.shape[1]]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= (0.75 * r) ** 2
        if disk.sum() and float(edges[disk].mean()) > BUTTON_PLAIN:
            continue
        if groove(gray, x, y, r) < GROOVED:
            continue  # без процеп наоколо: щампован надпис или драскотина, а не бутон
        out.append((float(x), float(y), float(r), ring))
    return out


def find(models, im: Image.Image, outer: np.ndarray) -> list[dict]:
    """Детайлите по стената на снимката: [{"pts" (пиксели на снимката), "kind": port/panel/button/dot}]."""
    from skimage import color

    body = Polygon(np.asarray(outer, float)).buffer(0)
    if body.is_empty:
        return []
    x0, y0, x1, y1 = body.bounds
    short = min(x1 - x0, y1 - y0)
    pad = 0.06 * max(x1 - x0, y1 - y0)
    box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(im.width, x1 + pad)), int(min(im.height, y1 + pad)))
    crop = im.convert("RGB").crop(box)
    off = np.array(box[:2], float)
    lab = color.rgb2lab(np.asarray(crop, np.float32) / 255)
    local = Polygon(np.asarray(body.exterior.coords) - off)
    inner = local.buffer(-EDGE * short)
    grid = inner.buffer(-0.02 * short)
    step = short / GRID
    pts = [(x, y) for x in np.arange(x0 - box[0] + step / 2, x1 - box[0], step)
           for y in np.arange(y0 - box[1] + step / 2, y1 - box[1], step) if grid.contains(Point(x, y))]  # fmt: skip
    found = []
    for m in sam_masks(models, crop, pts):
        a = float(m.sum())
        if not AREA[0] * local.area <= a <= AREA[1] * local.area:
            continue
        poly = outline_of(m)
        if poly is None or poly.area < SOLID * poly.convex_hull.area or not inner.contains(poly):
            continue
        rect = poly.minimum_rotated_rectangle
        sides = sorted(np.linalg.norm(np.diff(np.asarray(rect.exterior.coords)[:3], axis=0), axis=1))
        if sides[0] <= 0 or sides[1] / sides[0] > LONG:
            continue  # ред текст или драскотина
        if not regular(poly):
            continue  # стърготини, сянка, неправилно петно
        seen = judge(lab, m)
        if seen is None:
            continue
        found.append((seen["contrast"], poly, seen))
    keep: list[tuple] = []
    for con, poly, seen in sorted(found, key=lambda t: -t[0]):
        if any(poly.intersection(k[1]).area >= SAME * min(poly.area, k[1].area) for k in keep):
            continue
        keep.append((con, poly, seen))
    out = []
    for _, poly, seen in keep:
        kind = kind_of(seen, poly.area / local.area, poly)
        if kind:
            out.append({"pts": np.asarray(poly.exterior.coords)[:-1] + off, "kind": kind})
    gray = np.asarray(crop.convert("L"), np.float32) / 255
    long = max(x1 - x0, y1 - y0)
    for x, y, r, _ in circles(gray, inner, short) if short >= BUTTON_FACE and long <= BUTTON_LONG * short else []:
        c = Point(x, y).buffer(r, 48)
        if any(c.intersection(Polygon(o["pts"] - off)).area >= SAME * min(c.area, Polygon(o["pts"] - off).area)
               for o in out):  # fmt: skip
            continue
        out.append({"pts": np.asarray(c.exterior.coords)[:-1] + off, "kind": "button"})
    return out
