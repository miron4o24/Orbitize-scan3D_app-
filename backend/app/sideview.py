"""Страничният профил на плосък предмет, измерен от кадрите на видеото, в които предметът се вижда точно отстрани. AI дълбочината
само предполага колко се издава едно копче (дистанционно: 1,2 мм при истински 3 мм), докато в такъв кадър копчето стърчи над лицето и
се вижда направо. Лентата до всяко лице се изправя и се заглажда по дължината: шарката на килим се осреднява, пръстите минават
напряко, а ръбът на копче остава. Всеки кадър е с друг наклон, затова местата се поправят с перспективата му, а кадрите се сравняват,
защото пръст в един кадър го няма в другите. Всичко е в дялове от дължината на предмета."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

FILE = "side_profile.npz"
SIDE_ON = 1.25  # кадър с толкова пъти по-голямо съотношение от най-тънкото показва предмета отстрани
MIN_FRAMES = 3  # пръстите се отсяват само при сравняване; с по-малко кадри остават AI височините
MOST_FRAMES = 8
POINTS = 400  # места по дължината на профила
SEARCH = 0.01  # от дължината: около грубия ред от маската се търси точното лице
OUT = 0.024  # от дължината: най-далечното, което може да е част от предмета (копче от 4,5 мм при 192 мм)
SKIP = 0.0016  # от дължината: по-близкото до лицето е самото лице (0,3 мм)
BEYOND = 0.01  # от дължината: лентата се взема с толкова по-широка
ALONG = 0.0026  # от дължината: заглаждане по дължината, което маха шарката на фона
EDGE = 14.0  # разлика в цвета от един пиксел към следващия, която е ръб
ACROSS = 3.0  # ръб, който се променя толкова пъти по-силно по дължината, минава напряко: пръст
STEADY = 0.0078  # от дължината: еднаква височина на такова разстояние (медиана), иначе е шум
SHIFT = 0.02  # от дължината: колко могат да се разминават кадрите след поправката на перспективата
SCALE = 0.02
PERSPECTIVE = 0.5  # колко от промяната в дебелината е истински наклон; останалото е малко по-широк изглед към стената (дистанционно: 1,28 измерено, 1,13 по мястото на копчетата)
# Напречното сечение от кадрите, в които предметът е обърнат с края към камерата (бяло дистанционно: плоско лице и заоблен гръб, издут с
# 20% от ширината), и дебелината по дължината от кадрите отстрани (скосеният му горен край).
SECTION_POINTS = 64  # места през ширината
SECTION_EDGE = 0.04  # от ширината: най-крайните колони са пиксел-два и не се мерят
UP = 4  # маските са с четвърт от размера на кадъра: увеличават се, за да е гладко очертанието
FULL = 80  # перцентил на дългата страна на очертанията: цялата дължина на предмета в кадъра
FACE_SIZE = 0.75  # кадър с толкова от цялата дължина и
FACE_ON = 2.5  # толкова пъти по-голямо съотношение от най-тънкото показва лицето
END_SIZE = (0.5, 1.6)  # кадър по оста: дългата страна на очертанието спрямо ширината на предмета в кадрите отпред
END_RATIO = (0.6, 1.6)  # и съотношението му спрямо дебелина / ширина
SECTION_AGREE = 0.06  # от ширината: средна разлика от общото сечение, над която кадърът е друг изглед (наклонен, с ръка)
MIN_SECTIONS = 2
TAPER_MIDDLE = (0.2, 0.8)  # от дължината: тук се мери правата на перспективата в профила на дебелината
SIDE_LOOSE = 1.5  # за профила на дебелината стига и кадър почти отстрани: той се дели на средата си (бяло дистанционно: 0,10 при най-тънко 0,078)
TAPER_AGREE = 0.1  # средна разлика от общия профил, над която кадърът е с пръст върху ръба или с непълна маска
TAPER_NOISE = 0.9  # по-малко изтъняване от това е шум на маските (вълни по стената на модела), а не форма


def robust_line(pos: np.ndarray, values: np.ndarray, valid: np.ndarray, floor: float = 0.7) -> tuple[np.ndarray, float]:
    """Гладка крива през стойностите, без отклоненията (пръсти, копчета). Връща кривата и дела на използваните точки."""
    keep = valid.copy()
    fit = np.zeros(4)
    for _ in range(10):
        if keep.sum() < 8:
            break
        fit = np.polyfit(pos[keep], values[keep], 3)
        off = values - np.polyval(fit, pos)
        spread = max(floor, 1.4826 * float(np.median(np.abs(off[keep]))))
        keep = valid & (np.abs(off) < 2 * spread)
    return np.polyval(fit, pos), float(keep.mean())


def upright_view(photo: Image.Image, m: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Кадърът и маската при неговия размер, завъртени така, че дължината на предмета да е отвесна; само редовете на предмета."""
    alpha = Image.fromarray(np.where(m > 127, 255, 0).astype(np.uint8)).resize(photo.size, Image.NEAREST)
    ys, xs = np.nonzero(np.asarray(alpha) > 127)
    if len(ys) < 200:
        return None
    x, y = xs - xs.mean(), ys - ys.mean()
    angle = float(np.degrees(0.5 * np.arctan2(2 * (x * y).mean(), (x * x).mean() - (y * y).mean())))
    turn = angle - 90.0 if angle > 0 else angle + 90.0  # PIL върти обратно на часовника, а y сочи надолу
    if abs(turn) >= 0.1:
        photo = photo.rotate(turn, Image.BICUBIC, expand=True)
        alpha = alpha.rotate(turn, Image.NEAREST, expand=True)
    on = np.asarray(alpha) > 127
    rows = np.flatnonzero(on.any(1))
    return np.asarray(photo, np.float32)[rows[0] : rows[-1] + 1], on[rows[0] : rows[-1] + 1]


def exact_face(picture: np.ndarray, coarse: np.ndarray, pos: np.ndarray, reach: int) -> tuple[np.ndarray, float]:
    """Лицето през редовете: най-силният ръб около грубата линия от маската, наложен на гладка крива."""
    blurred = ndimage.gaussian_filter1d(picture, 2.0, axis=0)
    change = np.linalg.norm(np.diff(blurred, axis=1), axis=2)
    h, w = change.shape
    offsets = np.arange(-reach, reach)
    cols = np.clip(np.round(coarse).astype(np.int64)[:, None] + offsets, 0, w - 1)
    found = change[np.arange(h)[:, None], cols]
    best = found.argmax(1)
    strong = found[np.arange(h), best] > 10.0
    return robust_line(pos, cols[np.arange(h), best] + 0.5, strong)


def raised_beside(picture: np.ndarray, face: np.ndarray, sign: int, length: float) -> np.ndarray:
    """На всеки ред: на колко пиксела от лицето е най-далечният ръб, успореден на лицето; 0, ако няма. sign: накъде е навън."""
    h = len(face)
    far = np.arange(-int(0.005 * length), int((OUT + BEYOND) * length))
    rows = np.broadcast_to(np.arange(h)[:, None], (h, len(far)))
    cols = face[:, None] + sign * far[None]
    strip = np.stack([ndimage.map_coordinates(picture[..., k], [rows, cols], order=1, mode="nearest") for k in range(3)], -1)
    smooth = ndimage.gaussian_filter1d(strip, max(1.0, ALONG * length), axis=0)
    away = np.linalg.norm(np.diff(smooth, axis=1), axis=2)
    soft = ndimage.gaussian_filter1d(strip, 1.0, axis=1)
    along = np.pad(np.linalg.norm(soft[2:] - soft[:-2], axis=2), ((1, 1), (0, 0)))[:, :-1]
    edge = (away > EDGE) & (along < ACROSS * away)
    start, stop = np.searchsorted(far[:-1], int(SKIP * length)), np.searchsorted(far[:-1], int(OUT * length))
    part = edge[:, start:stop]
    last = part.shape[1] - 1 - part[:, ::-1].argmax(1)
    height = np.where(part.any(1), far[:-1][start + last], 0).astype(np.float32)
    return ndimage.median_filter(height, size=max(3, int(STEADY * length)))


def frame_profile(frame: Path, m: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Профилите на двете дълги страни от един кадър, в дялове от дължината, по истинските места по нея (POINTS). Първа е страната с
    повече издадени части. None, ако лицата не се намират."""
    with Image.open(frame) as im:
        view = upright_view(im.convert("RGB"), m)
    if view is None:
        return None
    picture, on = view
    h = len(on)
    if h < 200:
        return None
    pos = np.arange(h) / h
    seen = on.any(1)
    left = np.where(seen, on.argmax(1), 0).astype(np.float64)
    right = np.where(seen, on.shape[1] - 1 - on[:, ::-1].argmax(1), 0).astype(np.float64)
    coarse_l, _ = robust_line(pos, left, seen, 1.5)
    coarse_r, _ = robust_line(pos, right, seen, 1.5)
    reach = max(3, int(SEARCH * h))
    face_l, fit_l = exact_face(picture, coarse_l, pos, reach)
    face_r, fit_r = exact_face(picture, coarse_r, pos, reach)
    across = face_r - face_l
    if min(fit_l, fit_r) < 0.4 or (across <= 2).any():
        return None
    # Перспективата по дебелината: по-близкият край изглежда по-дебел. pos: мястото в кадъра, t: истинското.
    mid = (pos > 0.1) & (pos < 0.9)
    slope, base = np.polyfit(pos[mid], across[mid], 1)
    near = 1 + PERSPECTIVE * (float(np.clip((base + slope) / max(base, 1e-6), 0.6, 1.7)) - 1)
    scale = float(across.mean()) / across
    t = near * pos / (1 - (1 - near) * pos)
    grid = (np.arange(POINTS) + 0.5) / POINTS
    out = []
    for face, sign in ((face_l, -1), (face_r, 1)):
        height = raised_beside(picture, face, sign, h) * scale / h
        out.append(np.interp(grid, t, height).astype(np.float32))
    out.sort(key=lambda p: float(np.minimum(p, OUT).mean()), reverse=True)
    return out[0], out[1]


def placed(profile: np.ndarray, shift: float = 0.0, scale: float = 1.0, flip: bool = False) -> np.ndarray:
    """Профилът, преместен, разтеглен и евентуално обърнат, върху същите места."""
    p = profile[::-1] if flip else profile
    grid = (np.arange(len(p)) + 0.5) / len(p)
    return np.interp((grid - 0.5) * scale + 0.5 + shift, grid, p)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a - a.mean(), b - b.mean()
    return float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-12))


def best_fit(profile: np.ndarray, ref: np.ndarray, reach: float = SHIFT, stretch: float = SCALE) -> tuple[float, float, float]:
    """(съвпадение, изместване, разтягане), при които профилът пасва най-добре на ref."""
    top = (-2.0, 0.0, 1.0)
    for scale in np.arange(1 - stretch, 1 + stretch + 1e-9, 0.005):
        for shift in np.arange(-reach, reach + 1e-9, 0.0025):
            c = correlation(placed(profile, shift, scale), ref)
            if c > top[0]:
                top = (c, float(shift), float(scale))
    return top


def consensus(busy: list[np.ndarray], quiet: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Страната с издадените части и другата: кадрите, наредени един спрямо друг, и медианата им, която отхвърля пръсти и шум."""
    ref = busy[int(np.argmax([float(np.minimum(b, OUT).mean()) for b in busy]))]
    moved_b, moved_q = list(busy), list(quiet)
    for _ in range(2):
        moved_b, moved_q = [], []
        for b, q in zip(busy, quiet):
            flip = correlation(placed(b, flip=True), ref) > correlation(b, ref) + 0.15
            b0, q0 = (b[::-1], q[::-1]) if flip else (b, q)
            _, shift, scale = best_fit(b0, ref)
            moved_b.append(placed(b0, shift, scale))
            moved_q.append(placed(q0, shift, scale))
        ref = np.median(np.stack(moved_b), axis=0)
    return ref.astype(np.float32), np.median(np.stack(moved_q), axis=0).astype(np.float32)


def side_on_frames(ratios: dict[str, float], thin: float) -> list[str]:
    """Кадрите, показващи предмета отстрани, равномерно до MOST_FRAMES. ratios: съотношението от очертанието на всеки кадър."""
    names = sorted(n for n, r in ratios.items() if 0 < r <= SIDE_ON * thin)
    if len(names) > MOST_FRAMES:
        names = [names[i] for i in np.linspace(0, len(names) - 1, MOST_FRAMES).round().astype(int)]
    return names


def measure(folder: Path, masks: dict[str, np.ndarray], ratios: dict[str, float], thin: float) -> dict | None:
    """{"busy", "quiet", "frames"}: профилите на двете лица на плоския предмет или None, ако кадрите отстрани са твърде малко."""
    found = []
    for name in side_on_frames(ratios, thin):
        try:
            got = frame_profile(folder / name, masks[name])
        except (OSError, ValueError, np.linalg.LinAlgError):
            got = None
        if got is not None:
            found.append(got)
    if len(found) < MIN_FRAMES:
        return None
    busy, quiet = consensus([f[0] for f in found], [f[1] for f in found])
    return {"busy": busy, "quiet": quiet, "frames": np.array(len(found))}


def axes_of(m: np.ndarray) -> tuple[float, float, float] | None:
    """(посоката на дължината в радиани, дължината, ширината) на очертанието по собствените му оси, в пиксели на маската."""
    ys, xs = np.nonzero(m > 127)
    if len(ys) < 50:
        return None
    x, y = xs - xs.mean(), ys - ys.mean()
    angle = 0.5 * np.arctan2(2 * float((x * y).mean()), float((x * x).mean() - (y * y).mean()))
    along, across = x * np.cos(angle) + y * np.sin(angle), y * np.cos(angle) - x * np.sin(angle)
    return float(angle), float(np.ptp(along)) + 1, float(np.ptp(across)) + 1


def smooth_outline(m: np.ndarray, turn: float) -> np.ndarray:
    """Маската, увеличена UP пъти с гладко очертание и завъртяна с turn градуса (обратно на часовника); само най-голямото парче."""
    big = np.asarray(Image.fromarray(m).resize((m.shape[1] * UP, m.shape[0] * UP), Image.BICUBIC), np.float32)
    on = Image.fromarray(((ndimage.gaussian_filter(big, 0.7 * UP) > 127) * 255).astype(np.uint8))
    on = np.asarray(on.rotate(turn, Image.BILINEAR, expand=True)) > 127
    labels, count = ndimage.label(on)
    if count > 1:
        on = labels == 1 + int(np.argmax(ndimage.sum_labels(on, labels, np.arange(1, count + 1))))
    return on


def end_on_frames(masks: dict[str, np.ndarray], ratios: dict[str, float], thin: float) -> list[str]:
    """Кадрите, в които предметът е с края си към камерата: очертанието е колкото ширината му в кадрите отпред, а съотношението е
    дебелина / ширина. Само по съотношение не стига: бялото дистанционно е 0,30 по оста и 0,25 отпред."""
    found = {n: a for n, a in ((n, axes_of(m)) for n, m in masks.items()) if a is not None}
    if not found:
        return []
    full = float(np.percentile([a[1] for a in found.values()], FULL))
    face = [ratios.get(n, 0.0) for n, a in found.items() if a[1] >= FACE_SIZE * full and ratios.get(n, 0.0) >= FACE_ON * thin]
    if len(face) < 2:
        return []
    aspect = float(np.median(face))  # ширина / дължина
    wide, ratio = aspect * full, thin / aspect
    return sorted(n for n, a in found.items()
                  if END_SIZE[0] * wide <= a[1] <= END_SIZE[1] * wide and END_RATIO[0] * ratio <= ratios.get(n, 0.0) <= END_RATIO[1] * ratio)  # fmt: skip


def section_of(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Сечението от един кадър по оста: (колко навътре от най-външната си точка е по-плоският контур, същото за другия, в дялове от
    ширината при SECTION_POINTS места през нея; дебелината спрямо ширината). Огледално осреднено, защото ляво и дясно не се различават."""
    axes = axes_of(m)
    if axes is None:
        return None
    on = smooth_outline(m, float(np.degrees(axes[0])))  # дължината на очертанието (ширината на предмета) хоризонтално
    cols = np.flatnonzero(on.any(0))
    width = cols[-1] - cols[0] + 1
    if width < 40:
        return None
    u = np.linspace(SECTION_EDGE, 1 - SECTION_EDGE, SECTION_POINTS)
    part = on[:, np.round(cols[0] + u * (width - 1)).astype(np.int64)]
    if not part.any(0).all():
        return None
    top = part.argmax(0).astype(np.float64)
    bottom = (len(part) - 1 - part[::-1].argmax(0)).astype(np.float64)
    first, second = (top - top.min()) / width, (bottom.max() - bottom) / width
    if first.mean() > second.mean():
        first, second = second, first
    return 0.5 * (first + first[::-1]), 0.5 * (second + second[::-1]), float(bottom.max() - top.min() + 1) / width


def cross_section(masks: dict[str, np.ndarray], ratios: dict[str, float], thin: float) -> dict | None:
    """{"section_flat", "section_dome", "section_thick", "section_frames"}: общото сечение на кадрите по оста или None."""
    found = [s for s in (section_of(masks[n]) for n in end_on_frames(masks, ratios, thin)) if s is not None]
    if len(found) < MIN_SECTIONS:
        return None
    flat, dome = np.stack([f[0] for f in found]), np.stack([f[1] for f in found])
    keep = np.ones(len(found), bool)
    for _ in range(2):
        mid_f, mid_d = np.median(flat[keep], 0), np.median(dome[keep], 0)
        keep = np.abs(flat - mid_f).mean(1) + np.abs(dome - mid_d).mean(1) < SECTION_AGREE
        if keep.sum() < MIN_SECTIONS:
            return None
    thick = float(np.median([f[2] for f, k in zip(found, keep) if k]))
    return {"section_flat": np.median(flat[keep], 0).astype(np.float32), "section_dome": np.median(dome[keep], 0).astype(np.float32),
            "section_thick": np.array(thick, np.float32), "section_frames": np.array(int(keep.sum()))}  # fmt: skip


def thickness_along(m: np.ndarray) -> np.ndarray | None:
    """Дебелината на кадър отстрани по дължината (POINTS места, горе първо), в дялове от дължината."""
    axes = axes_of(m)
    if axes is None:
        return None
    angle = float(np.degrees(axes[0]))
    on = smooth_outline(m, angle - 90.0 if angle > 0 else angle + 90.0)  # отвесно, без да се обръща (както upright_view)
    rows = np.flatnonzero(on.any(1))
    on = on[rows[0] : rows[-1] + 1]
    h = len(on)
    if h < 100:
        return None
    seen = on.any(1)
    left = on.argmax(1)
    right = on.shape[1] - 1 - on[:, ::-1].argmax(1)
    thick = np.where(seen, right - left + 1, 0) / h
    return np.interp((np.arange(POINTS) + 0.5) / POINTS, (np.arange(h) + 0.5) / h, thick).astype(np.float32)


def relative_thickness(t: np.ndarray) -> np.ndarray | None:
    """Профилът на дебелината, разделен на правата през средата му: перспективата (по-близкият край изглежда по-дебел) се маха и
    остават промените към краищата."""
    grid = (np.arange(len(t)) + 0.5) / len(t)
    mid = (grid >= TAPER_MIDDLE[0]) & (grid <= TAPER_MIDDLE[1])
    if t[mid].min() <= 0:
        return None
    line = np.polyval(np.polyfit(grid[mid], t[mid], 1), grid)
    return np.clip(t / np.maximum(line, 1e-6), 0.0, 1.5)


def side_thickness(masks: dict[str, np.ndarray], ratios: dict[str, float], thin: float) -> dict | None:
    """{"side_thickness", "side_frames"}: дебелината по дължината спрямо тази в средата (най-много 1: краищата и скосяванията), от
    кадрите отстрани с почти цялата дължина. Остават само съгласните помежду си: палец върху ръба или непълна маска дават друг профил."""
    found = {n: a for n, a in ((n, axes_of(m)) for n, m in masks.items()) if a is not None}
    if not found:
        return None
    full = float(np.percentile([a[1] for a in found.values()], FULL))
    names = [n for n, a in found.items() if 0 < ratios.get(n, 0.0) <= SIDE_LOOSE * thin and a[1] >= FACE_SIZE * full]
    rel = [r for r in (relative_thickness(t) for t in (thickness_along(masks[n]) for n in names) if t is not None) if r is not None]
    if len(rel) < 2:
        return None
    rel = np.stack([ndimage.median_filter(r, size=max(3, POINTS // 50), mode="nearest") for r in rel])
    keep = np.ones(len(rel), bool)
    for _ in range(2):
        keep = np.abs(rel - np.median(rel[keep], 0)).mean(1) < TAPER_AGREE
        if keep.sum() < 2:
            return None
    profile = np.clip(np.median(rel[keep], 0), 0.0, 1.0)
    grid = (np.arange(POINTS) + 0.5) / POINTS
    profile[(grid >= TAPER_MIDDLE[0]) & (grid <= TAPER_MIDDLE[1])] = 1.0  # средата е правата, спрямо която се мери
    profile[profile >= TAPER_NOISE] = 1.0
    return {"side_thickness": profile.astype(np.float32), "side_frames": np.array(int(keep.sum()))}


def main(folder: Path, masks_file: Path, thin: float, out: Path) -> None:
    """Собствен процес, за да не смята линейна алгебра в нишка на сървъра (OpenBLAS пада на Windows при две едновременно)."""
    if __package__:
        from . import segment as S
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from app import segment as S

    with np.load(masks_file) as z:
        masks = {n: z[n] for n in z.files}
    ratios = {n: S.outline_ratio(m) for n, m in masks.items()}
    got = {}
    heights = measure(folder, masks, ratios, thin)
    if heights is None:
        print("side view: too few frames from the side", flush=True)
    else:
        got |= heights
        print(f"side view: {int(heights['frames'])} frames, highest part {float(heights['busy'].max()):.4f} of the length", flush=True)
    section = cross_section(masks, ratios, thin)
    if section is None:
        print("cross-section: too few frames along the axis", flush=True)
    else:
        got |= section
        dome = section["section_dome"]
        print(f"cross-section: {int(section['section_frames'])} frames, thickness {float(section['section_thick']):.3f} of the width, "
              f"rounded side {float(dome[[0, -1]].mean() - dome.min()):.3f} of the width deep", flush=True)  # fmt: skip
    along = side_thickness(masks, ratios, thin)
    if along is not None:
        got |= along
        t = along["side_thickness"]
        print(f"thickness along the length: {int(along['side_frames'])} frames, ends {float(t[:8].mean()):.2f} and {float(t[-8:].mean()):.2f} of the middle", flush=True)
    if got:
        np.savez(out, **got)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4]))
