"""Инженерният режим върху папката на задачата (крайните точки са в main.py):

job/photos/top.jpg, front.jpg, back.jpg, right.jpg, left.jpg, bottom.jpg  - снимките (както при сканиране); нужна е
                      само отгоре, другите са по желание (колкото повече, толкова по-сигурно)
job/cad/shots.json  - очертанията на детайла на всяка снимка в пиксели (изрязването се прави веднъж на снимка)
job/cad/draft.json  - текущото описание (от него са картинките към въпросите и наложените очертания)
job/cad/out/        - последният построен детайл: part.step, part.py, part.stl, part.glb, drawing.png

Редът: read (без мерки) -> общите размери, всеки с линия върху снимка -> assemble (с тях) -> "Какво виждам" ->
мерките, по една, с линия върху снимката -> build. Всичко става веднага (секунди), не в опашката на сканиранията."""

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from . import assemble as AS
from . import design as D
from . import photos as P
from . import render as R

VIEWS = P.NAMES + tuple(P.LEGACY)  # имената на снимките (и от първия вариант: side, end)
FILES = {"part.step": "model/step", "part.py": "text/x-python; charset=utf-8", "part.stl": "model/stl",
         "part.glb": "model/gltf-binary", "drawing.png": "image/png"}  # fmt: skip
_session = None
_session_lock = threading.Lock()
_busy = threading.BoundedSemaphore(2)  # най-много две четения на снимки наведнъж (процесорът е общ със сканиранията)
_job_locks: dict[str, threading.Lock] = {}


log = logging.getLogger("uvicorn.error")


def lock(job: Path) -> threading.Lock:
    with _session_lock:
        return _job_locks.setdefault(job.name, threading.Lock())


def session():
    global _session
    with _session_lock:
        if _session is None:
            from .. import segment as S

            _session = S.session()
        return _session


def photo_path(job: Path, name: str) -> Path | None:
    names = [name] + [old for old, new in P.LEGACY.items() if new == name]
    for n in names:
        for ext in ("jpg", "jpeg"):
            p = job / "photos" / f"{n}.{ext}"
            if p.exists():
                return p
    return None


def present(job: Path) -> list[str]:
    return [n for n in P.NAMES if photo_path(job, n)]


KEPT = 10  # толкова от последните изоставени детайли на собственика се пазят за проверка


def kept_root() -> Path:
    from .. import jobs

    return jobs.DATA.parent / "cad-kept"  # извън DATA: не е задача (jobs.count, почистването)


def keep_abandoned(job: Path) -> bool:
    """Незавършен детайл на собственика, изоставен в приложението (то трие задачата): снимките и прочетеното от тях
    (cad/*.json) се пазят, последните KEPT, за да може да се види какво не е разчетено, вместо да се гадае.
    Моделите и останалото не се пазят."""
    if not present(job):
        return False
    root = kept_root()
    dest = root / job.name
    try:
        shutil.rmtree(dest, ignore_errors=True)
        (dest / "cad").mkdir(parents=True)
        for f in sorted((job / "photos").glob("*.jp*g")):
            shutil.copy2(f, dest / f.name)
        for f in sorted((job / "cad").glob("*.json")):
            shutil.copy2(f, dest / "cad" / f.name)
        write_json(dest / "kept.json", {"time": time.time(), "photos": present(job)})
        old = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
        for d in old[KEPT:]:
            shutil.rmtree(d, ignore_errors=True)
        return True
    except OSError as e:
        log.warning("keeping abandoned %s: %s", job.name, e)
        return False


def kept_list() -> list[dict]:
    root = kept_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        info = d / "kept.json"
        if d.is_dir() and info.exists():
            out.append({"id": d.name} | json.loads(info.read_text(encoding="utf-8")))
    return out


def kept_zip(job_id: str) -> bytes | None:
    """Пазените снимки и прочетеното от тях като zip (без компресия: JPEG не се свива)."""
    import io
    import zipfile

    d = kept_root() / job_id
    if not (d / "kept.json").exists():
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(d).as_posix())
    return buf.getvalue()


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def response(job: Path, d: dict, warnings: list[str] | None = None) -> dict:
    """Това, което приложението показва: описанието, разпознатото, въпросите и кои снимки има."""
    out = job / "cad"
    out.mkdir(exist_ok=True)
    write_json(out / "draft.json", d)
    stamp = str(int(time.time() * 1000))
    return {"design": d, "items": D.describe(d), "questions": D.questions(d), "notes": d.get("notes", []),
            "warnings": warnings or [], "views": present(job), "stamp": stamp}  # fmt: skip


def shots(job: Path) -> dict:
    """Очертанията на детайла на всички снимки (в пиксели), прочетени веднъж на снимка."""
    if not photo_path(job, "top"):
        raise D.DesignError("Липсва снимката отгоре.")
    out = job / "cad"
    out.mkdir(exist_ok=True)
    cache = out / "shots.json"
    with lock(job):
        from . import vision as VI

        stored = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        result, todo = {}, []
        for name in present(job):
            path = photo_path(job, name)
            key = [path.name, path.stat().st_mtime, path.stat().st_size]
            if name in stored and stored[name].get("key") == key:
                result[name] = stored[name]
            else:
                todo.append((name, path, key))
        if todo:
            with _busy:
                sess = session()
                for name, path, key in todo:
                    try:
                        result[name] = VI.segment(sess, path) | {"key": key}
                    except ValueError as e:
                        if str(e) == "cut":
                            raise D.DesignError(f"На снимката {P.TITLE[name]} предметът не се събира целият в кадъра: "
                                                "снимайте я отново от по-високо.") from e  # fmt: skip
                        raise D.DesignError(f"На снимката {P.TITLE[name]} не се вижда целият предмет: снимайте я "
                                            "отново, предметът в средата, на контрастна повърхност.") from e  # fmt: skip
        old = [n for n in result if result[n].get("profiles") != VI.PROFILES]  # от по-стара версия: без изрязване
        for name in old:
            im, outer = VI.load(photo_path(job, name)), np.asarray(result[name]["outer"])
            result[name] |= {"lines": VI.line_profiles(im, outer), "looks": VI.looks(im, outer),
                             "profiles": VI.PROFILES}  # fmt: skip
        if todo or old:
            write_json(cache, result)
        return result


CUT_ERRORS = {
    "cut": "Така очертанието стига до ръба на снимката: докоснете самия детайл, не повърхността или ръката.",
    "no sam": "Поправката с докосване още не е налична на сървъра: снимайте тази страна отново.",
    "no object": "Там не се вижда детайл: докоснете върху самия детайл.",
}


def retrace(job: Path, name: str, points: list[list[float]]) -> None:
    """Очертанието на снимката `name` наново, по точките, които потребителят е докоснал върху детайла (x, y от 0 до
    1; с трето число 0 - "това не е детайлът"), когато автоматичното е грешно (ръка, телефон, сянка). Без точки:
    отново автоматичното. Важи, докато снимката не се смени."""
    from . import vision as VI

    name = P.LEGACY.get(name, name)
    path = photo_path(job, name) if name in P.NAMES else None
    if not path:
        raise D.DesignError("Няма такава снимка.")
    shots(job)  # останалите снимки (и кешът) са готови
    try:
        with _busy:
            got = VI.segment(session(), path, points or None)
    except ValueError as e:
        raise D.DesignError(CUT_ERRORS.get(str(e), CUT_ERRORS["no object"])) from e
    key = [path.name, path.stat().st_mtime, path.stat().st_size]
    cache = job / "cad" / "shots.json"
    with lock(job):
        stored = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        stored[name] = got | {"key": key, **({"points": points} if points else {})}
        write_json(cache, stored)


OUTLINE_SIDE = 1400  # дългата страна на снимката с очертанието за проверка в приложението


def outline_jpg(job: Path, name: str) -> bytes:
    """Снимката (цялата, не завъртяна) с очертанието, което е изрязано от нея: за проверка в приложението, където
    докосване върху нея (дял от ширината и височината) е точка за retrace."""
    import io

    from PIL import ImageDraw

    from .vision import load

    name = P.LEGACY.get(name, name)
    path = photo_path(job, name) if name in P.NAMES else None
    if not path:
        raise D.DesignError("Няма такава снимка.")
    sh = shots(job)
    if name not in sh:
        raise D.DesignError("Снимката още не е прочетена.")
    im = load(path)
    k = OUTLINE_SIDE / max(im.size)
    im = im.resize((round(im.size[0] * k), round(im.size[1] * k)))
    d = ImageDraw.Draw(im)
    w = max(3, round(im.size[0] / 300))
    for poly in [sh[name]["outer"]] + sh[name]["holes"]:
        xy = [(x * k, y * k) for x, y in poly]
        d.line(xy + [xy[0]], fill=(20, 20, 20), width=w + 4, joint="curve")  # тъмна подложка: видимо и на светло
        d.line(xy + [xy[0]], fill=R.ACCENT, width=w, joint="curve")
    for p in sh[name].get("points", []):
        x, y, r = p[0] * im.size[0], p[1] * im.size[1], 3 * w
        ok = len(p) < 3 or p[2]
        d.ellipse((x - r, y - r, x + r, y + r), fill=(40, 200, 90) if ok else (230, 50, 50), outline="white",
                  width=max(2, w // 2))  # fmt: skip
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


FLAT_RATIO = 0.2  # дебелина под толкова от по-малката страна отгоре: плосък детайл (кука, пластина, шайба)


def read(job: Path, L: float | None = None, W: float | None = None, H: float | None = None,
         flips: dict | None = None, flat: bool = False) -> dict:  # fmt: skip
    """Чете снимките. Без мерки: кои общи размери да се измерят и на коя снимка (overall). С мерки (първият
    вариант на приложението): направо описанието, като assemble. flat: плосък детайл (третият размер е дебелина)."""
    if L and W and H:
        return assemble(job, {"L": L, "W": W, "H": H}, flips, flat)
    lay = P.layout(shots(job)) | {"flat": bool(flat)}
    write_json(job / "cad" / "plan.json", lay)
    return {"photos": present(job), "overall": P.overall(lay, flat), "warnings": lay["warnings"], "round": lay["round"],
            "stamp": str(int(time.time() * 1000))}  # fmt: skip


def assemble(job: Path, values: dict, flips: dict | None = None, flat: bool = False) -> dict:
    """Описанието от снимките и измерените общи размери (L, W, H или D за кръглото сечение). flat: плосък детайл
    (избран в приложението или тънък спрямо размерите отгоре): само очертанието отгоре с една дебелина, а ръбовете
    отгоре и отдолу може да са заоблени. Снимките отстрани на такъв детайл са в ръка и само развалят формата."""
    sh = shots(job)
    try:
        L, W, H = P.resolve(P.layout(sh), {k: v for k, v in values.items() if v})
    except ValueError as e:
        raise D.DesignError("Въведете общите размери в милиметри.") from e
    flips = sanitize_flips(flips)
    lay = P.layout(sh, (L, W, H), flips)
    views, warnings = P.views_mm(sh, lay, L, W, H)
    flat = bool(flat) or H <= FLAT_RATIO * min(L, W)
    if flat:
        sides = [P.TITLE[n] for n in ("front", "back", "right", "left")]  # предупрежденията за тях не важат
        warnings = [w for w in warnings if not any(t in w for t in sides)]
        lay = lay | {"warnings": [w for w in lay["warnings"] if not any(t in w for t in sides)]}
    d = AS.assemble(views, L, W, H, flat=flat)
    xs = [D.coord(d, lid) for lid, spec in d["levels"].items() if spec["axis"] == "x"]
    for x, x_bottom in seams(job, sh, lay, L, W, H):
        if all(abs(x - lx) > 0.04 * L for lx in xs):  # не е ръбът на стъпало
            d = D.add_split(d, x, x_bottom)
    if flat and d["kind"] == "block":
        d = D.add_feature(d, {"t": "fillet", "edges": "flat"})
        f = d["features"][-1]
        dim = d["dims"][f["r"]]
        dim["name"] = "Радиус на заоблените ръбове"
        dim["ask"] = "Радиус на заоблянето по дебелината (ръбовете отгоре и отдолу)"
        dim["hint"] = "Ако ръбовете са остри, въведете 0. Най-много половината от дебелината."
    elif d["kind"] == "block" and "end" not in views and not d["views"]["end"].get("round"):
        r = side_rounding(job, sh, lay, L, W, H)
        if r:
            rid = D.add_dim(d, "r", r, "Радиус на заоблените страни", "Радиус на заоблените дълги ръбове",
                            "Оценен по светлината на снимката; без снимка отстрани не се вижда точно.", "radius",
                            optional=True, show={"view": "end", "radius": True})  # fmt: skip
            d["views"]["end"] = {"pts": [["y0", "z0", rid], ["yW", "z0", rid], ["yW", "zH", rid], ["y0", "zH", rid]]}
            d["notes"].append("Заоблените страни са оценени по светлината на снимката: за точност снимайте и края.")
    d["flips"] = flips
    d["flat"] = flat
    return response(job, d, lay["warnings"] + warnings)


def side_rounding(job: Path, sh: dict, lay: dict, L: float, W: float, H: float) -> float | None:
    """Заоблени дълги ръбове без снимка от края: на снимката отгоре (или отдолу, която е по-равномерна, без печат)
    закръгленият ръб е ивица, в която светлината се мени, а равната страна е еднаква. Широчината на ивицата от двете
    страни е радиусът (ако е само от едната страна, е сянка или видяна стена, а не заобляне)."""
    from scipy import ndimage

    from .vision import load

    best = None
    for name in ("top", "bottom"):
        o = lay["orient"].get(name)
        if o is None or o["view"] != "top" or name not in sh:
            continue
        A = P.placement(sh[name], o, L, W)["A"]
        im = np.asarray(load(photo_path(job, name)).convert("L"), np.float32)
        us = np.linspace(0.3 * L, 0.6 * L, 60)
        vs = np.arange(0.0, W, W / 230)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        px = A[:, :2] @ np.stack([uu.ravel(), vv.ravel()]) + A[:, 2:3]
        prof = np.median(ndimage.map_coordinates(im, [px[1], px[0]], order=1).reshape(len(us), len(vs)), 0)
        mid = (vs > 0.35 * W) & (vs < 0.65 * W)
        flat = float(np.median(prof[mid]))
        mad = float(np.median(np.abs(prof[mid] - flat))) * 1.48
        if mad > 0.03 * flat:
            continue  # печат или неравна светлина по средата: не става
        off = np.abs(prof - flat) >= max(3 * mad, 0.03 * flat, 3.0)
        a, b = vs < 0.3 * W, vs > 0.7 * W
        w0 = float(vs[a][off[a]].max()) if off[a].any() else 0.0
        w1 = float(W - vs[b][off[b]].min()) if off[b].any() else 0.0
        if min(w0, w1) < 0.15 * H:
            continue
        if best is None or mad < best[0]:
            best = (mad, min((w0 + w1) / 2, 0.45 * H))
    return round(best[1], 2) if best else None


SEAM_COVER = 0.9  # колко от ширината трябва да пресича линия, за да е граница между части


def seams(job: Path, sh: dict, lay: dict, L: float, W: float, H: float = 0.0) -> list:
    """Границите между две части по дължината (капачка и тяло, два материала): права линия през цялата ширина на
    предмета на снимката отгоре (рязка смяна на яркостта по дължината в поне 90% от ширината; надписите и отворите
    не я пресичат цялата). Ако има и снимка отдолу, там трябва да има линия наблизо (до 10% от дължината); иначе
    е рисунка само отгоре. Връща [(X отгоре, X отдолу или None)] в мм (до две)."""
    from scipy import ndimage

    from .vision import load

    found = {}
    for name in ("top", "bottom"):
        o = lay["orient"].get(name)
        if o is None or o["view"] != "top" or name not in sh:
            continue
        A = P.placement(sh[name], o, L, W)["A"]
        im = np.asarray(load(photo_path(job, name)).convert("L"), np.float32)
        im = ndimage.gaussian_filter(im, 1.0)
        step = L / 500
        us = np.arange(0.06 * L, 0.94 * L, step)
        vs = np.linspace(0.12 * W, 0.88 * W, 60)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        px = A[:, :2] @ np.stack([uu.ravel(), vv.ravel()]) + A[:, 2:3]
        g = ndimage.map_coordinates(im, [px[1], px[0]], order=1).reshape(len(us), len(vs))
        k = max(2, round(0.4 / step))
        diff = np.abs(g[k:] - g[:-k])
        thr = max(3 * float(np.median(diff)), 6.0)
        cover = ndimage.uniform_filter1d((diff > thr).mean(1), 3)
        half = max(1, round(2.0 / step))
        peaks = [float(us[i + k // 2]) for i in range(len(cover))
                 if cover[i] >= SEAM_COVER and cover[i] == cover[max(0, i - half):i + half + 1].max()]  # fmt: skip
        merged = []
        for x in peaks:
            if not merged or x - merged[-1] > 2.0:
                merged.append(x)
        found[name] = merged
        # По-слабите линии (поне 60% от ширината): границата отдолу на размазана снимка.
        weak = [float(us[i + k // 2]) for i in range(len(cover))
                if cover[i] >= 0.6 and cover[i] == cover[max(0, i - half):i + half + 1].max()]  # fmt: skip
        found[name + "_weak"] = weak
        found[name + "_cover"] = [(float(us[i + k // 2]), float(cover[i])) for i in range(len(cover))]
    # Отгоре: линиите една до друга (ръбът на капачката и нещо под прозрачната пластмаса) са една граница - най-ясната;
    # другите от групата са "вътрешни" (виждат се през пластмасата и отдолу на същото място, затова не са границата).
    groups: list[list[float]] = []
    for x in sorted(found.get("top", [])):
        if groups and x - groups[-1][-1] <= 0.08 * L:
            groups[-1].append(x)
        else:
            groups.append([x])
    names = [n for n in ("front", "back") if n in lay["orient"] and lay["orient"][n]["view"] == "side"]
    out = []
    for g in groups:
        x = max(g, key=lambda v: top_cover(found, v))
        inner = [v for v in g if v != x]
        bottom = None
        if "bottom" in found:
            cands = [y for y in found["bottom"] + found.get("bottom_weak", []) if abs(x - y) <= 0.12 * L
                     and all(abs(y - v) > 0.6 for v in inner)]  # fmt: skip
            sides = [v for v in (slant(job, sh, lay, x, L, H, n) for n in names) if v is not None]
            side = float(np.mean(sides)) if sides and max(sides) - min(sides) <= 1.5 else None
            if cands and side is not None:  # отдолу и отстрани: линията отдолу, която пасва на диагонала
                y = min(cands, key=lambda c: abs(c - side))
                bottom = y if abs(y - side) <= 2.0 else side
            elif cands:
                bottom = min(cands, key=lambda c: abs(c - x))
            elif side is not None:
                bottom = side
            elif top_cover(found, x) < 0.97:
                continue  # не е потвърдена отдолу и не е съвсем ясна отгоре: печат, не граница
        else:
            sides = [v for v in (slant(job, sh, lay, x, L, H, n) for n in names) if v is not None]
            bottom = float(np.mean(sides)) if sides and max(sides) - min(sides) <= 1.5 else None
        out.append((x, bottom))
    return out[:2]


def slant(job: Path, sh: dict, lay: dict, x: float, L: float, H: float, name: str = "front") -> float | None:
    """Къде е границата на долната страна по снимката отпред: на 9 височини най-рязката смяна на светлината по
    дължината близо до x, права през тях (по най-много съгласни точки). Кой край на правата е горната страна се
    решава по x от снимката отгоре (отпред симетричен предмет може да е обърнат наопаки)."""
    from scipy import ndimage

    from .vision import load

    if name not in lay["orient"]:
        return None
    o = lay["orient"][name]
    A = P.placement(sh[name], o, L, H)["A"]
    im = ndimage.gaussian_filter(np.asarray(load(photo_path(job, name)).convert("L"), np.float32), 1.0)
    us = np.arange(max(0.0, x - 0.15 * L), min(L, x + 0.15 * L), L / 600)
    pts = []
    for z in np.linspace(0.1 * H, 0.9 * H, 9):
        zs = np.linspace(z - 0.04 * H, z + 0.04 * H, 5)
        uu, zz = np.meshgrid(us, zs, indexing="ij")
        px = A[:, :2] @ np.stack([uu.ravel(), zz.ravel()]) + A[:, 2:3]
        g = ndimage.map_coordinates(im, [px[1], px[0]], order=1).reshape(len(us), len(zs)).mean(1)
        d = np.abs(np.gradient(ndimage.gaussian_filter1d(g, 2)))
        if d.max() >= 2 * np.median(d) + 0.5:
            pts.append((float(z), float(us[int(np.argmax(d))])))
    if len(pts) < 4:
        return None
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (z1, x1), (z2, x2) = pts[i], pts[j]
            k = (x2 - x1) / (z2 - z1)
            inl = [(z, xx) for z, xx in pts if abs(x1 + k * (z - z1) - xx) < 0.8]
            if best is None or len(inl) > len(best):
                best = inl
    if len(best) < 4:
        return None
    zz, xx = np.array(best).T
    k, c = np.polyfit(zz, xx, 1)
    ends = (float(c), float(c + k * H))  # при z = 0 и z = H на снимката
    near = min(ends, key=lambda e: abs(e - x))
    if abs(near - x) > 0.06 * L:
        return None  # отпред не е същата граница
    other = ends[1] if near is ends[0] else ends[0]
    return float(np.clip(other, 0, L))


def top_cover(found: dict, x: float) -> float:
    return max((c for xx, c in found.get("top_cover", []) if abs(xx - x) < 0.6), default=0.0)


def edit(job: Path, raw: dict, op: dict) -> dict:
    """Промяна от приложението: values, remove, add, through, flip (наново от прочетеното, мерките се губят)."""
    from .build import clean

    d = clean(raw)
    d["flips"] = sanitize_flips(raw.get("flips"))
    d["flat"] = bool(raw.get("flat"))
    if "values" in op:
        vals = {}
        for k, v in (op["values"] or {}).items():
            vals[str(k)] = None if v is None else float(v)
        d = D.set_values(d, vals)
    elif "remove" in op:
        d = D.remove(d, str(op["remove"]))
    elif "add" in op:
        d = D.add_feature(d, op["add"] if isinstance(op["add"], dict) else {})
    elif "through" in op:
        fid, through = op["through"]
        d = set_through(d, str(fid), bool(through))
    elif "flip" in op:
        view, axis = op["flip"]
        view = {"front": "side", "back": "side", "right": "end", "left": "end"}.get(view, view)
        if view not in ("side", "end") or axis not in (0, 1):
            raise D.DesignError("Непознато обръщане.")
        flips = d["flips"]
        cur = flips.get(view, [False, False])
        cur[axis] = not cur[axis]
        flips[view] = cur
        L, W, H = sizes(d)
        return assemble(job, {"L": L, "W": W, "H": H}, flips, d["flat"])
    else:
        raise D.DesignError("Непозната промяна.")
    d["flips"] = sanitize_flips(raw.get("flips"))
    d["flat"] = bool(raw.get("flat"))
    return response(job, d)


def sizes(d: dict) -> tuple[float, float, float]:
    """(L, W, H) на описанието (детайл от струг: W = H = D)."""
    if d["kind"] == "turned":
        return D.value(d, "L"), D.value(d, "D"), D.value(d, "D")
    return D.value(d, "L"), D.value(d, "W"), D.value(d, "H")


def sanitize_flips(f) -> dict:
    out = {}
    if isinstance(f, dict):
        for k in ("side", "end"):
            v = f.get(k)
            if isinstance(v, list) and len(v) == 2:
                out[k] = [bool(v[0]), bool(v[1])]
    return out


def set_through(d: dict, fid: str, through: bool) -> dict:
    import copy

    d = copy.deepcopy(d)
    f = next((f for f in d["features"] if f["id"] == fid), None)
    if f is None or f["t"] not in ("hole", "rect", "slot"):
        raise D.DesignError("Непознат елемент.")
    if through:
        f.pop("depth", None)
        D.unused_cleanup(d)
    elif not f.get("depth"):
        face = f.get("face") or "top"
        f["depth"] = D.add_dim(d, "t", None, "Дълбочина", f"Измерете дълбочината от {D.FACE_NAME[face]}",
                               "С дълбокомера на шублера.", show={"feature": fid})  # fmt: skip
    return d


def draft(job: Path) -> dict:
    p = job / "cad" / "draft.json"
    if not p.exists():
        raise D.DesignError("Още няма прочетени снимки.")
    return json.loads(p.read_text(encoding="utf-8"))


def view_photos(job: Path, d: dict) -> dict:
    """{изглед на описанието: {"image", "A", "name"}}: основната снимка на всеки изглед и как мм от изгледа отиват на
    нея (по размерите на описанието). Празно, ако снимките още не са прочетени."""
    from .vision import load

    cache = job / "cad" / "shots.json"
    if not cache.exists():
        return {}
    sh = {k: v for k, v in json.loads(cache.read_text(encoding="utf-8")).items() if photo_path(job, k)}
    if "top" not in sh:
        return {}
    L, W, H = sizes(d)
    lay = P.layout(sh, (L, W, H), d.get("flips") or {})
    out = {}
    for name, o in lay["orient"].items():
        if not o["primary"]:
            continue
        U, V = P.view_sizes(o["view"], L, W, H)
        out[o["view"]] = {"image": load(photo_path(job, name)), "A": P.placement(sh[name], o, U, V)["A"],
                          "name": name}  # fmt: skip
    return out


def question_png(job: Path, dim: str) -> bytes:
    """Картинката към въпроса: върху снимката с линия, където да се мери. Общите размери преди описанието: от
    плана на четенето (plan.json)."""
    draft_path = job / "cad" / "draft.json"
    if draft_path.exists():
        d = draft(job)
        if dim in d["dims"]:
            return png(R.question(d, dim, view_photos(job, d)))
    if dim not in P.ASK:
        raise D.DesignError("Непознат размер.")
    plan_path = job / "cad" / "plan.json"
    if not plan_path.exists():
        raise D.DesignError("Още няма прочетени снимки.")
    from .vision import load

    lay = json.loads(plan_path.read_text(encoding="utf-8"))
    q = next((q for q in P.overall(lay, lay.get("flat", False)) if q["id"] == dim), None)
    if q is None:
        raise D.DesignError("Непознат размер.")
    if q["photo"] is None:
        return png(R.overall(None, None, 1, 1, dim, q["along"]))
    sh = shots(job)
    o = lay["orient"][q["photo"]]
    f = P.Frame(sh[q["photo"]])
    U, V = f.ext(o["eu"]), f.ext(o["ev"])  # пиксели като мм: само за положението на линията
    A = P.placement(sh[q["photo"]], o, U, V)["A"]
    return png(R.overall(load(photo_path(job, q["photo"])), A, U, V, dim, q["along"], q["photo"]))


def photo_jpg(job: Path, name: str) -> bytes:
    """Снимката с очертанието от текущото описание."""
    from .vision import load

    name = P.LEGACY.get(name, name)
    d = draft(job)
    cache = job / "cad" / "shots.json"
    path = photo_path(job, name)
    if not path or not cache.exists():
        raise D.DesignError("Няма такава снимка.")
    sh = json.loads(cache.read_text(encoding="utf-8"))
    if name not in sh:
        raise D.DesignError("Снимката още не е прочетена.")
    L, W, H = sizes(d)
    lay = P.layout({k: v for k, v in sh.items() if photo_path(job, k)}, (L, W, H), d.get("flips") or {})
    o = lay["orient"][name]
    U, V = P.view_sizes(o["view"], L, W, H)
    im = R.overlay(d, o["view"], load(path), P.placement(sh[name], o, U, V)["A"])
    import io

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return buf.getvalue()


def png(im) -> bytes:
    import io

    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def build(job: Path, raw: dict) -> dict:
    """Строи детайла, записва файловете и модела за приложението (model.ply с най-дълга страна 100, като всеки модел;
    истинският размер е real_mm). Връща отчета на задачата."""
    import trimesh

    from .. import mesh as M
    from .build import build as make
    from .build import clean

    d = clean(raw)
    d["flips"] = sanitize_flips(raw.get("flips"))
    missing = [q for q in D.questions(d) if d["dims"][q["id"]]["v"] is None]
    if missing:
        raise D.DesignError(f"Липсва размерът „{missing[0]['name']}“.")
    with lock(job):
        work = job / "cad" / "build.tmp"
        shutil.rmtree(work, ignore_errors=True)
        report = make(d, work)
        R.drawing(d).save(work / "drawing.png", optimize=True)
        out = job / "cad" / "out"
        shutil.rmtree(out, ignore_errors=True)
        os.replace(work, out)
        response(job, d)
        real = history_model(job, out)
        solid = trimesh.load(out / "part.stl", force="mesh")  # затворено (за обема и повърхнината в отчета)
        solid.apply_translation(-solid.bounds.mean(0) * [1, 1, 0] - [0, 0, solid.bounds[0][2]])
        solid.apply_scale(100.0 / real)
    return M.report(solid) | {"ai": "cad", "real_mm": round(real, 3), "object": "", "suitable": True,
                              "cad": report}  # fmt: skip


def history_model(job: Path, out: Path) -> float:
    """Моделът в историята (model.ply) е същият като прегледа: цветни части, остри ръбове (без сливане на върховете
    при зареждане: main.result), най-дълга страна 100 като всеки модел. Връща истинския размер (real_mm)."""
    import trimesh

    display = out / "display.ply"
    model = trimesh.load(display if display.exists() else out / "part.stl", force="mesh", process=False)
    real = float(max(model.extents))
    model.apply_translation(-model.bounds.mean(0) * [1, 1, 0] - [0, 0, model.bounds[0][2]])
    model.apply_scale(100.0 / real)
    tmp = job / "model.tmp.ply"
    model.export(tmp)
    os.replace(tmp, job / "model.ply")
    return real


def refresh(job: Path) -> bool:
    """Детайл, построен преди цветните части и острите ръбове в прегледа (без display.ply): прегледът и моделът в
    историята се правят наново от същото тяло (part.py, записан при строенето), така че да изглежда като новите.
    Размерите и файловете за изработка не се променят. False, ако не стане: остава старият вид."""
    from .build import glb, solid

    out = job / "cad" / "out"
    if (out / "display.ply").exists() or not (out / "part.py").exists():
        return False
    try:
        with lock(job):
            if not (out / "display.ply").exists():
                glb(out / "part.stl", out / "part.glb", solid((out / "part.py").read_text(encoding="utf-8")))
                history_model(job, out)
        return True
    except Exception as e:  # стар или повреден детайл: показва се както досега
        log.warning("cad refresh of %s: %s", job.name, e)
        return False


PREVIEW_STYLE = 2  # видът на прегледа (build.glb): при промяна запазените прегледи се правят наново


def preview(job: Path) -> Path:
    """GLB на текущото описание с оценките (за проверка на посоките преди мерките). Елементи без размер (добавени
    ръчно, още неизмерени) се пропускат."""
    import copy
    import hashlib

    from .build import build as make
    from .build import clean

    d = draft(job)
    stamp = hashlib.sha1(json.dumps([PREVIEW_STYLE, d], sort_keys=True).encode()).hexdigest()[:16]
    out = job / "cad" / "preview"
    glb = out / "part.glb"
    if glb.exists() and (out / "stamp").exists() and (out / "stamp").read_text() == stamp:
        return glb
    p = copy.deepcopy(d)
    missing = {i for i, dim in p["dims"].items() if dim["v"] is None}
    if missing:
        p["features"] = [f for f in p["features"] if not any(f.get(k) in missing for k in
                                                             ("d", "w", "h", "len", "depth", "r", "c"))]  # fmt: skip
        D.unused_cleanup(p)
    with lock(job):
        work = job / "cad" / "preview.tmp"
        shutil.rmtree(work, ignore_errors=True)
        make(clean(p), work)
        (work / "stamp").write_text(stamp)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(work, out)
    return glb
