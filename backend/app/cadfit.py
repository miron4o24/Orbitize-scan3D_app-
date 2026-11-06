"""Чист CAD модел от готово сканиране: безплатният Gemini (Google, ключ GEMINI_API_KEY) проектира предмета от прости части
(app/ai.py build: кутии, цилиндри, очертания, завъртени профили, отвори), а сървърът сам го сравнява със сканирането от девет
страни (очертания и дълбочина) и показва на Gemini къде греши, докато съвпадне. Така работи и инженерът от видеото с ChatGPT
(снимки, мерки, поправки), но поправките идват от истинската форма, а не на ръка. Подходящо за части с ясна форма (клипс,
държач, кутия, дистанционно); фигура остава сканиране.

Координатите са на сканирането (app/mesh.py make_printable): основата при z = 0, центрирано по x и y, най-дългата страна 100.
Истинският размер идва от мярката на потребителя в приложението (мащабира целия модел)."""

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from scipy.spatial import cKDTree

from . import ai as AI
from . import mesh as M
from . import pipeline as P

API = "https://generativelanguage.googleapis.com/v1beta/models/"
# Безплатните модели на Gemini (Pro не е безплатен): по ред, следващият при претоварване или лимит.
MODELS = [m for m in os.environ.get("GEMINI_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash").split(",") if m]
ROUNDS = int(os.environ.get("CAD_ROUNDS", "4"))  # поправки след първия проект
GOOD = 0.96  # оценка, при която поправките спират
SOURCE = "source.ply"
PHOTOS = "cad_photos"
PHOTO_COUNT = 5
PHOTO_SIDE = 1024
GRID = 220  # клетки по най-дългата страна на картинките за сравнение
SAMPLES = 400_000
# (име, ос на гледане, ос надясно, ос нагоре): x, y, z = 0, 1, 2; знакът е посоката на погледа
VIEWS = [("front", (1, 1), 0, 2), ("back", (1, -1), 0, 2), ("right", (0, -1), 1, 2), ("left", (0, 1), 1, 2),
         ("top", (2, -1), 0, 1), ("bottom", (2, 1), 0, 1)]  # fmt: skip

SYSTEM = AI.SYSTEM + """

Here you are copying one real object exactly, for a 1:1 printed copy. You get photos of it and, from the server, \
its 3D scan seen straight from six sides, in the scan's own coordinates and units (given in the request): use those \
numbers, not a guessed real size. The scan is the truth for the shape and size; the photos show the details (edges, \
holes, buttons, text-free surfaces) and which side is which. After each design the server builds it and shows you, \
side by side with the scan, where it differs: fix exactly those places."""

FIRST = """Design this object. The scan's box: x from {x0:.1f} to {x1:.1f}, y from {y0:.1f} to {y1:.1f}, z from 0 to \
{z1:.1f} (z up, it stands on z = 0). The second image shows the scan straight from six sides with a scale in the same \
units: front (looking along +y), back, right (looking along -x), left, top (looking down) and bottom; on each, the \
grey is the outline and darker is nearer. Answer with the design only."""

REVIEW = """The server built your design and compared it with the scan from six sides (same order and scale as \
before). Green: both. Red: the scan has it, your design does not (add there). Blue: your design has it, the scan does \
not (remove there). Outline match per side: {ious}; depth difference (where both are): {depths}. Overall score \
{score:.3f} of 1. Answer with the complete corrected design (all parts), keeping what already matches."""


# --- Gemini ---------------------------------------------------------------------------------------------------------


def picture_part(data: bytes, mime: str = "image/jpeg") -> dict:
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}


WAIT_S = 150  # един отговор
ASK_S = 200  # общо за един проект: претоварените (503) модели се опитват отново на кръгове, докато не изтече
GONE = (400, 401, 403, 404, 429)  # лимитът за деня свършил или моделът не е достъпен: не се пита повече в тази задача


def ask(contents: list, note: Callable[[str], None], skip: set[str]) -> tuple[dict, dict]:
    """Един проект от Gemini: (проектът, репликата на модела за разговора). Безплатните модели по ред; претоварен
    (503, бавен) се пробва пак след останалите, докато не изтече ASK_S. skip: моделите с изчерпан лимит в тази задача."""
    key = AI.key("gemini")
    if not key:
        raise P.Failed("Услугата за CAD модели още не е включена.")
    body = {"system_instruction": {"parts": [{"text": SYSTEM}]}, "contents": contents,
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": AI.DESIGN,
                                 "temperature": 0.4}}  # fmt: skip
    last = ""
    deadline = time.time() + ASK_S
    while True:
        models = [m for m in MODELS if m not in skip]
        if not models or time.time() > deadline:
            break
        answered = _ask_round(models, body, key, note, skip)
        if isinstance(answered, tuple):
            return answered
        last = answered
        time.sleep(5)  # всички бяха претоварени: малко почивка, после пак
    raise P.Failed(f"gemini: {last or 'no model left'}")


def _ask_round(models: list[str], body: dict, key: str, note: Callable[[str], None], skip: set[str]) -> tuple[dict, dict] | str:
    """Един опит с всеки модел по ред: (проект, реплика) от първия, който отговори, иначе последната грешка."""
    last = ""
    for model in models:
        for attempt in range(2):
            started = time.time()
            req = urllib.request.Request(API + model + ":generateContent", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})  # fmt: skip
            try:
                with urllib.request.urlopen(req, timeout=WAIT_S) as r:
                    answer = json.loads(r.read())
            except urllib.error.HTTPError as e:
                last = f"{model}: {e.code}"
                note(f"gemini {model}: {e.code} after {time.time() - started:.0f} s")
                if e.code == 400 and "responseJsonSchema" in body["generationConfig"]:  # модел без JSON схема
                    body["generationConfig"].pop("responseJsonSchema")
                    continue
                if e.code in GONE:
                    skip.add(model)
                break  # претоварен: следващият модел, а този пак в следващия кръг
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = f"{model}: {e}"
                note(f"gemini {model}: no answer after {time.time() - started:.0f} s")
                break
            parts = ((answer.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            try:
                design = json.loads(text[text.find("{") : text.rfind("}") + 1])
            except json.JSONDecodeError:
                last = f"{model}: no design in the answer"
                continue
            note(f"gemini {model}: {len(design.get('parts') or [])} parts in {time.time() - started:.0f} s")
            return design, {"role": "model", "parts": [{"text": json.dumps(design)}]}
    return last


# --- Сравнение със сканирането --------------------------------------------------------------------------------------


def frame_of(scan: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """Общата рамка на картинките: ъгълът на кутията на сканирането (с малко поле) и размерът на една клетка."""
    lo, hi = scan.bounds
    pad = 0.06 * float(scan.extents.max())
    return lo - pad, (float(scan.extents.max()) + 2 * pad) / GRID


def maps(mesh: trimesh.Trimesh, lo: np.ndarray, cell: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """За всяка страна: (очертание, дълбочина на най-близката точка) като решетки; дълбочината е от ръба на рамката."""
    points, _ = trimesh.sample.sample_surface(mesh, SAMPLES, seed=0)
    points = np.vstack([points, mesh.vertices])
    rel = (points - lo) / cell
    n = GRID
    out = {}
    for name, (axis, sign), right, up in VIEWS:
        u = np.clip(rel[:, right].astype(int), 0, n - 1)
        v = np.clip(n - 1 - rel[:, up].astype(int), 0, n - 1)
        if name in ("back", "left") or (name == "bottom"):
            u = n - 1 - u  # погледнато отзад / отляво / отдолу: огледално, както би се видяло
        d = rel[:, axis] if sign > 0 else n - rel[:, axis]
        depth = np.full((n, n), np.inf)
        np.minimum.at(depth, (v, u), d)
        seen = np.isfinite(depth)
        shape = ndimage.binary_fill_holes(ndimage.binary_closing(seen, iterations=2))
        depth = np.where(seen, depth, np.nan)
        out[name] = (shape, depth)
    return out


def compare(scan_maps: dict, cad_maps: dict, cell: float) -> tuple[float, dict, dict]:
    """(оценка 0-1, съвпадение на очертанията по страни, средна разлика в дълбочината по страни в единици)."""
    ious, depths = {}, {}
    for name, _, _, _ in VIEWS:
        a, da = scan_maps[name]
        b, db = cad_maps[name]
        ious[name] = float((a & b).sum()) / max(1.0, float((a | b).sum()))
        both = np.isfinite(da) & np.isfinite(db)
        depths[name] = float(np.nanmedian(np.abs(da - db)[both])) * cell if both.any() else float("nan")
    size = GRID * cell
    depth_err = np.nanmean(list(depths.values())) / size if np.isfinite(list(depths.values())).any() else 1.0
    score = float(np.mean(list(ious.values()))) - 2.0 * float(depth_err)
    return score, ious, depths


def crops(scan_maps: dict) -> dict[str, tuple[int, int, int, int]]:
    """За всяка страна: частта от решетката около сканирането (ред от, ред до, колона от, колона до), обща за сканирането
    и проекта, за да е предметът в средата и едър."""
    out = {}
    for name, (shape, _) in scan_maps.items():
        rows, cols = np.flatnonzero(shape.any(1)), np.flatnonzero(shape.any(0))
        if not len(rows):
            out[name] = (0, GRID, 0, GRID)
            continue
        m = 6
        out[name] = (max(0, rows[0] - m), min(GRID, rows[-1] + m + 1), max(0, cols[0] - m), min(GRID, cols[-1] + m + 1))
    return out


def sheet(images: list[tuple[str, np.ndarray]], cell: float, box: dict) -> bytes:
    """Картинките една до друга (3 x 2), всяка изрязана около предмета и с еднакъв мащаб, с името на страната и мащабна
    линия, като PNG."""
    tile = 420
    spans = [max(box[n][1] - box[n][0], box[n][3] - box[n][2]) for n, _ in images]
    zoom = min(8.0, tile / max(spans))  # пиксела на клетка, еднакво за всички страни
    out = Image.new("RGB", (tile * 3, (tile + 30) * 2), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    step = 10 if GRID * cell < 300 else 50
    for i, (name, rgb) in enumerate(images):
        r0, r1, c0, c1 = box[name]
        part = Image.fromarray(np.ascontiguousarray(rgb[r0:r1, c0:c1]))
        part = part.resize((max(1, int((c1 - c0) * zoom)), max(1, int((r1 - r0) * zoom))), Image.NEAREST)
        x, y = (i % 3) * tile, (i // 3) * (tile + 30)
        out.paste(part, (x + (tile - part.width) // 2, y + 30 + (tile - part.height) // 2))
        bar = int(step / cell * zoom)
        draw.text((x + 6, y + 4), f"{name}   bar = {step} units", fill=(0, 0, 0))
        draw.line([(x + tile - bar - 10, y + 22), (x + tile - 10, y + 22)], fill=(0, 0, 0), width=3)
        draw.rectangle([x, y, x + tile - 1, y + tile + 29], outline=(200, 200, 200))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def scan_picture(scan_maps: dict, cell: float) -> bytes:
    images = []
    for name, _, _, _ in VIEWS:
        shape, depth = scan_maps[name]
        rgb = np.full(shape.shape + (3,), 255, np.uint8)
        d = np.where(np.isfinite(depth), depth, np.nanmax(depth) if np.isfinite(depth).any() else 0)
        lo, hi = np.nanmin(d[shape]) if shape.any() else 0, np.nanmax(d[shape]) if shape.any() else 1
        shade = 60 + 160 * (d - lo) / max(hi - lo, 1e-6)
        rgb[shape] = np.clip(shade[shape], 0, 255).astype(np.uint8)[:, None]
        images.append((name, rgb))
    return sheet(images, cell, crops(scan_maps))


def difference_picture(scan_maps: dict, cad_maps: dict, cell: float) -> bytes:
    images = []
    for name, _, _, _ in VIEWS:
        a, da = scan_maps[name]
        b, db = cad_maps[name]
        rgb = np.full(a.shape + (3,), 255, np.uint8)
        rgb[a & b] = (120, 200, 120)
        rgb[a & ~b] = (220, 40, 40)
        rgb[b & ~a] = (40, 80, 230)
        both = np.isfinite(da) & np.isfinite(db) & a & b
        far = both & (np.abs(da - db) * cell > 0.02 * GRID * cell)  # дълбочината се различава с над 2%
        nearer = far & (db < da)  # проектът стърчи пред сканирането
        rgb[far & ~nearer] = (220, 40, 40)
        rgb[nearer] = (40, 80, 230)
        images.append((name, rgb))
    return sheet(images, cell, crops(scan_maps))


# --- Задачата -------------------------------------------------------------------------------------------------------


def copy_source(source_job: Path, job: Path) -> None:
    """Сканирането и няколко от кадрите му (предметът отвсякъде) в новата задача."""
    import shutil

    shutil.copy(source_job / "model.ply", job / SOURCE)
    frames = sorted((source_job / "photos").glob("*.jp*g"))
    pick = job / PHOTOS
    pick.mkdir(exist_ok=True)
    for f in (AI.spread(frames, PHOTO_COUNT) if frames else []):
        shutil.copy(f, pick / f.name)


def coloured(cad: trimesh.Trimesh, scan: trimesh.Trimesh) -> trimesh.Trimesh:
    """Цветовете на сканирането (най-близката точка), ако ги има, иначе тези на частите."""
    colours = getattr(scan.visual, "vertex_colors", None)
    if scan.visual.kind == "vertex" and colours is not None and len(colours) == len(scan.vertices):
        _, near = cKDTree(scan.vertices).query(cad.vertices)
        cad.visual.vertex_colors = np.asarray(colours)[near]
    return cad


def run(job: Path, params: dict, set_state: Callable[[str], None], step: Callable[[str], None]) -> dict:
    note = lambda text: P.note(job, text)  # noqa: E731
    set_state("reconstructing")
    scan = trimesh.load(job / SOURCE, force="mesh")
    lo, cell = frame_of(scan)
    scan_maps = maps(scan, lo, cell)
    x0, y0, _ = scan.bounds[0]
    x1, y1, z1 = scan.bounds[1]
    photos = []
    for f in sorted((job / PHOTOS).glob("*.jp*g")):
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((PHOTO_SIDE, PHOTO_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        photos.append(picture_part(buf.getvalue()))
    first = FIRST.format(x0=x0, x1=x1, y0=y0, y1=y1, z1=z1)
    contents = [{"role": "user", "parts": photos + [picture_part(scan_picture(scan_maps, cell), "image/png"), {"text": first}]}]
    best = None
    skip: set[str] = set()
    started = time.time()
    for round_ in range(ROUNDS + 1):
        step("ai designing" if round_ == 0 else "ai checking")
        try:
            design, turn = ask(contents, note, skip)
            model, skipped = AI.build(design)
        except P.Failed as e:
            if best is None:
                raise
            note(f"cad round {round_}: {e}; keeping the best so far")
            break
        cad_maps = maps(model, lo, cell)
        score, ious, depths = compare(scan_maps, cad_maps, cell)
        note(f"cad round {round_}: score {score:.3f}, outlines " + ", ".join(f"{k} {v:.2f}" for k, v in ious.items())
             + f", {len(design.get('parts') or [])} parts ({skipped} not buildable), {time.time() - started:.0f} s")  # fmt: skip
        if best is None or score > best[0]:
            best = (score, model, design)
        if score >= GOOD or round_ == ROUNDS:
            break
        review = REVIEW.format(ious=", ".join(f"{k} {v:.2f}" for k, v in ious.items()),
                               depths=", ".join(f"{k} {v:.1f}" for k, v in depths.items()), score=score)  # fmt: skip
        # Разговорът: първата заявка (снимките и сканирането), последният проект и сравнението му; по-старите кръгове не,
        # за да не расте заявката.
        contents = contents[:1] + [turn, {"role": "user", "parts": [picture_part(difference_picture(scan_maps, cad_maps, cell), "image/png"),
                                                                     {"text": review}]}]  # fmt: skip
    score, model, design = best
    set_state("cleaning")
    out = M.make_printable(coloured(model, scan), float(model.bounds[0, 2]), False, smoothing=0)
    note(f"cad: best score {score:.3f}, object {design.get('object')!r}")
    # Името от Gemini Flash понякога излиза като безсмислица на кирилица: заглавието идва от сканирането (приложението).
    return P.finish(job, out) | {"photos_used": len(photos), "photos_total": len(photos), "ai": "cad",
                                 "object": "", "suitable": bool(design.get("suitable")),
                                 "real_mm": None, "match": round(score, 3)}  # fmt: skip
