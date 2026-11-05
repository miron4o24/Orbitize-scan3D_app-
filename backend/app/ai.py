"""AI моделите на Pro: модел от всяко видео (или снимки), вместо обработката на сканиране (app/pipeline.py). single:
безплатен, на процесора на сървъра: кадрите се разполагат в 3D (COLMAP), AI изрязва предмета от тях, а измерената
между тях дълбочина и AI дълбочината се сливат в тези очертания и снимките стават текстура (app/outline.py).
Предположението на TripoSR от един кадър (app/single3d.py, гърбът е предположен), когато кадрите не могат да се
разположат. Двата метода по-долу изискват платени услуги и остават изключени, докато не се зададат ключовете им.
claude: езиков модел гледа няколко кадъра и проектира предмета от плътни части (кутии, цилиндри, изтеглени очертания,
завъртени профили...), всяка добавена или изрязана по ред. Частите се изграждат и съединяват тук в една затворена
мрежа, рисуват се (app/render.py) и се показват веднъж за корекция. Чисто, с плоски страни и верен размер за
изработени предмети (кутии, скоби, чаши, копчета, стойки); фигура или животно излиза като груб заместител. photo:
услугата image-to-3D на Meshy (платена) оформя всеки предмет от до четири кадъра около него; това, което никой кадър
не показва, се предполага, а размерите са само пропорции. Ключовете са на приложението, а не на потребителите:
ANTHROPIC_API_KEY / MESHY_API_KEY от средата, иначе се задават с API ключа (PUT /admin/ai; пазят се в таблицата с
настройки на базата данни и никога не се връщат)."""

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import requests
import trimesh
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.spatial import cKDTree

from . import db
from . import mesh as M
from . import outline as OL
from . import photo3d
from . import pipeline as P
from . import relief as RL
from . import segment as S
from . import single3d
from . import texture as TX

METHODS = ("single", "claude", "photo")
SERVICES = {"anthropic": "ANTHROPIC_API_KEY", "meshy": "MESHY_API_KEY"}
NEEDS = {"claude": "anthropic", "photo": "meshy"}

MODEL = os.environ.get("AI_CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("AI_EFFORT", "high")
CANDIDATES = 120  # кадри, прочетени от видеото; изпращат се най-резките от равномерно разпределени групи
CLAUDE_FRAMES = 6
MESHY_FRAMES = 4  # най-много, колкото приема multi-image-to-3d на Meshy
SEND_SIDE = 1280  # кадрите се изпращат с този размер (дълга страна)
REVIEWS = int(os.environ.get("AI_REVIEWS", "1"))  # кръгове "ето нарисувания ти модел, поправи го"
SECTIONS = 96  # сегменти около кръгли части: гладко при размера за печат
MESHY = "https://api.meshy.ai/openapi/v1"
MESHY_WAIT_S = 20 * 60


# --- Ключове --------------------------------------------------------------------------------------------------------


def key(service: str) -> str | None:
    return os.environ.get(SERVICES[service]) or db.get_setting(f"ai_key_{service}") or None


def set_key(service: str, value: str):
    db.set_setting(f"ai_key_{service}", value)  # "" го премахва


def key_status() -> dict:
    out = {}
    for service, env in SERVICES.items():
        k = key(service)
        out[service] = {"set": bool(k), "from": ("server" if os.environ.get(env) else "app") if k else None,
                        "ends": k[-4:] if k else None}  # fmt: skip
    return out


def available() -> dict:
    """Какво може да използва Pro на този сървър; express: сканирането за около 5 минути (app/pipeline.py), edit:
    текстови промени (app/edit.py; простите заявки работят без езиковия му модел), и двете винаги са налични."""
    paid = {method: bool(key(service)) for method, service in NEEDS.items()}
    return {"express": True, "edit": True, "single": single3d.ready()} | paid


# --- Кадри ----------------------------------------------------------------------------------------------------------


def duration(video: Path) -> float:
    r = subprocess.run([P.FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True, text=True, errors="replace")
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def sharpness(path: Path) -> float:
    with Image.open(path) as im:
        g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((320, 320)), float)
    return float(ndimage.laplace(g).var())


def frames(job: Path, sh: Callable[[str, list], None]) -> list[Path]:
    """Кадрите от видеото (до CANDIDATES, равномерно разпределени) като photos/img_0001.jpg... или качените снимки."""
    photos, video = job / "photos", job / P.VIDEO
    if video.exists():
        for p in photos.glob("*.jp*g"):  # повторен опит чете видеото отново
            p.unlink()
        seconds = duration(video)
        fps = min(4.0, CANDIDATES / seconds) if seconds > 0 else 2.0
        scale = "scale=w='trunc(iw*min(1,1600/max(iw,ih))/2)*2':h=-2"
        sh("reading video", [P.FFMPEG, "-v", "error", "-y", "-i", video, "-an", "-vf", f"fps={fps:.4f},{scale}",
            "-frames:v", str(CANDIDATES), "-q:v", "3", photos / "img_%04d.jpg"])  # fmt: skip
    found = sorted(photos.glob("*.jp*g"))
    if not found:
        raise P.Failed("Във видеото няма кадри, които да се прочетат. Опитайте с друго видео.")
    return found


def spread(found: list[Path], count: int) -> list[Path]:
    """Най-рязкият кадър от всяка от `count` равни групи: предметът отвсякъде, без размазване."""
    groups = np.array_split(np.arange(len(found)), min(count, len(found)))
    return [found[max(g, key=lambda i: sharpness(found[i]))] for g in groups if len(g)]


def jpeg(path: Path, side: int = SEND_SIDE) -> bytes:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def image_block(data: bytes, media: str = "image/jpeg") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.b64encode(data).decode()}}


# --- Езиков модел ---------------------------------------------------------------------------------------------------

NUM = {"type": "number"}
NUMS = {"type": "array", "items": NUM}
PART = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "note": {"type": "string"},
        "op": {"type": "string", "enum": ["add", "cut"]},
        "shape": {"type": "string", "enum": ["box", "rounded_box", "cylinder", "cone", "sphere", "torus", "extrude",
                                             "revolve"]},  # fmt: skip
        "size": NUMS,
        "radius": NUM,
        "radius2": NUM,
        "height": NUM,
        "points": {"type": "array", "items": NUMS},
        "position": NUMS,
        "rotation": NUMS,
        "color": {"type": "string"},
    },
    "required": ["note", "op", "shape", "size", "radius", "radius2", "height", "points", "position", "rotation",
                 "color"],  # fmt: skip
}
DESIGN = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "object": {"type": "string"},
        "suitable": {"type": "boolean"},
        "parts": {"type": "array", "items": PART},
    },
    "required": ["object", "suitable", "parts"],
}

SYSTEM = """You design 3D-printable models of real objects from photos of them, as a CAD designer would.

The photos are frames of one video (or photos) of a single object, usually filmed while walking around it. Find the \
main object (ignore the table, hands and the room) and build it from solid parts, in millimetres, at its real size: \
judge the size from what is around it (a hand, a table, a door, text, a plug) or from what the object is.

Coordinates: Z is up, the object stands on the plane z = 0 (its lowest point at z = 0), X is its width, Y its depth, \
centred on x = 0, y = 0.

Parts are applied in order: "add" joins a part to the model so far, "cut" removes it from the model so far (holes, \
slots, hollows, a cup's inside, a groove). Every added part must touch or overlap the rest: the model is one piece. \
Fill every field of every part; fields a shape does not use are 0, [] or [0, 0, 0].

Shapes, before rotation and position (the part's own origin goes to `position`):
- box: `size` [x, y, z], centred on its origin.
- rounded_box: a box with all its edges rounded by `radius` (less than half the smallest side).
- cylinder: `radius`, `height`, axis along Z, centred on its origin.
- cone: bottom radius `radius`, top radius `radius2` (0: a point), `height`, axis along Z, centred on its origin.
- sphere: `radius`; or an ellipsoid when `size` holds three radii [rx, ry, rz].
- torus: a ring around the Z axis: `radius` from the axis to the tube's centre, `radius2` the tube's radius.
- extrude: `points` [[x, y], ...] an outline in the XY plane (around its origin, no crossing edges), pulled up \
`height` along Z, centred on z = 0.
- revolve: `points` [[r, z], ...] a closed profile in the r-z half plane (r >= 0 is the distance from the axis), \
turned once round the Z axis: vases, cups, bottles, knobs, turned legs. Its origin is on the axis at z = 0.
`rotation` [rx, ry, rz] degrees about X, then Y, then Z, about the part's own origin, then moved to `position`.
`color` "#rrggbb": the part's colour in the photos. `note`: a few words on what the part is.

Make it look like the object: right proportions first, then every feature visible in the photos (bevels as \
rounded boxes, holes, slots, ribs, rims, text-free). Prefer extrude and revolve for outlines and turned shapes, and \
use as many parts as the object needs (often 5-60). Hollow things (cups, boxes, vases) get their inside cut, with \
walls at least 1.6 mm thick so they print.

"object": what it is, in Bulgarian, 1-4 words (it becomes the model's title). "suitable": false when the object \
cannot be told well from solid parts (a figure, an animal, a face, a plant): still give your best simple stand-in."""

FIRST = ("These are frames of the object, from around it. Design it. Answer with the design only.")
REVIEW = ("This is your design, built and drawn from the front right and a little from above (grey where no colour). "
          "Compare it with the photos: proportions and size, missing, extra or misplaced parts, holes that did not "
          "cut through, parts that do not touch. Answer with the complete corrected design (all parts, not only the "
          "changes), or the same design if it already matches.")  # fmt: skip


def ask_claude(client, messages: list) -> tuple[dict, list]:
    """Един проект: разчетеният отговор и репликата на асистента за пазене в разговора."""
    import anthropic

    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=SYSTEM,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": DESIGN}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # погрешен отказ от класификатора се изпълнява отново на друг модел, а не като неуспешен модел
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise P.Failed("Услугата Claude временно не работи. Опитайте по-късно.")
    except anthropic.RateLimitError:
        raise P.Failed("Claude е претоварен в момента. Опитайте отново след малко.")
    except anthropic.APIStatusError as e:
        raise P.Failed(f"Claude върна грешка ({e.status_code}). Опитайте отново.")
    except anthropic.APIConnectionError:
        raise P.Failed("Сървърът не можа да се свърже с Claude. Опитайте отново.")
    if msg.stop_reason == "refusal":
        raise P.Failed("Claude отказа да моделира това видео.")
    if msg.stop_reason == "max_tokens":
        raise P.Failed("Предметът е твърде сложен за Claude. Опитайте „AI от снимки“.")
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        design = json.loads(text)
    except json.JSONDecodeError:
        raise P.Failed("Claude не върна модел. Опитайте отново.")
    return design, [{"role": "assistant", "content": msg.content}]


def vec(v, n: int, fill: float = 0.0) -> np.ndarray:
    out = [float(x) for x in (v or [])[:n] if isinstance(x, (int, float))]
    return np.array(out + [fill] * (n - len(out)))


def part_mesh(p: dict) -> trimesh.Trimesh | None:
    """Една поставена част; None, когато размерите ѝ не дават плътно тяло."""
    shape = p.get("shape")
    size = np.clip(vec(p.get("size"), 3), 0, 5000)
    r, r2, h = (float(np.clip(p.get(k) or 0, 0, 5000)) for k in ("radius", "radius2", "height"))
    pts = np.array([vec(q, 2) for q in (p.get("points") or [])]).reshape(-1, 2)
    if shape == "box" and size.min() > 0:
        m = trimesh.creation.box(extents=size)
    elif shape == "rounded_box" and size.min() > 0:
        r = min(r, size.min() / 2 * 0.98)
        if r <= 0:
            m = trimesh.creation.box(extents=size)
        else:
            ball = trimesh.creation.icosphere(subdivisions=3, radius=r)
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * (size / 2 - r)
            m = trimesh.util.concatenate([ball.copy().apply_translation(c) for c in corners]).convex_hull
    elif shape == "cylinder" and r > 0 and h > 0:
        m = trimesh.creation.cylinder(radius=r, height=h, sections=SECTIONS)
    elif shape == "cone" and h > 0 and max(r, r2) > 0:
        m = M.revolve(np.array([[0, 0], [r, 0], [r2, h], [0, h]]), SECTIONS).apply_translation([0, 0, -h / 2])
    elif shape == "sphere" and (r > 0 or size.min() > 0):
        m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        m.apply_scale(size if size.min() > 0 else [r, r, r])
    elif shape == "torus" and r > 0 and 0 < r2 < r:
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        m = M.revolve(np.c_[r + r2 * np.cos(t), r2 * np.sin(t)], SECTIONS)
    elif shape == "extrude" and len(pts) >= 3 and h > 0:
        from shapely.geometry import Polygon

        poly = Polygon(pts).buffer(0)  # самопресичащо се очертание, направено валидно
        if poly.geom_type != "Polygon" or poly.area <= 0:
            if poly.geom_type != "MultiPolygon":
                return None
            poly = max(poly.geoms, key=lambda g: g.area)
        m = trimesh.creation.extrude_polygon(poly, h).apply_translation([0, 0, -h / 2])
    elif shape == "revolve" and len(pts) >= 3:
        m = M.revolve(pts, SECTIONS)
    else:
        return None
    if not m.is_volume:
        return None
    rot = np.radians(vec(p.get("rotation"), 3))
    m.apply_transform(trimesh.transformations.euler_matrix(*rot, "sxyz"))
    m.apply_translation(vec(p.get("position"), 3))
    return m


def colour(text: str) -> np.ndarray:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", (text or "").strip())
    return np.array([int(m[1][i : i + 2], 16) for i in (0, 2, 4)] + [255]) if m else np.array([170, 170, 170, 255])


def build(design: dict) -> tuple[trimesh.Trimesh, int]:
    """Частите, съединени по ред (съединенията прави manifold3d); оцветени по най-близката добавена част. Също колко
    части не са могли да се изградят."""
    model, added, skipped = None, [], 0
    for p in design.get("parts") or []:
        m = part_mesh(p)
        if m is None:
            skipped += 1
            continue
        if p.get("op") == "cut":
            if model is not None:
                model = trimesh.boolean.difference([model, m], engine="manifold")
        else:
            model = m if model is None else trimesh.boolean.union([model, m], engine="manifold")
            added.append((m, colour(p.get("color"))))
    if model is None or model.is_empty or not len(model.faces):
        raise P.Failed("Claude не успя да изгради модел от това видео. Опитайте отново или с „AI от снимки“.")
    pts, labels = [], []
    for i, (m, _) in enumerate(added):
        s, _ = trimesh.sample.sample_surface(m, 3000, seed=0)
        pts.append(s)
        labels.append(np.full(len(s), i))
    _, near = cKDTree(np.concatenate(pts)).query(model.vertices)
    palette = np.array([c for _, c in added], np.uint8)
    model.visual.vertex_colors = palette[np.concatenate(labels)[near]]
    return model, skipped


def drawing(work: Path, model: trimesh.Trimesh) -> bytes:
    from . import render

    d = work / "draft"
    d.mkdir(parents=True, exist_ok=True)
    out = M.make_printable(model.copy(), float(model.bounds[0, 2]), False, smoothing=0)
    out.export(d / "model.ply")
    buf = io.BytesIO()
    render.picture(d).save(buf, "PNG")
    return buf.getvalue()


def by_claude(job: Path, work: Path, found: list[Path], step: Callable[[str], None]) -> tuple[trimesh.Trimesh, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=key("anthropic"), timeout=900.0, max_retries=2)
    picked = spread(found, CLAUDE_FRAMES)
    messages = [{"role": "user", "content": [image_block(jpeg(f)) for f in picked] + [{"type": "text", "text": FIRST}]}]
    step("claude designing")
    design, turn = ask_claude(client, messages)
    model, skipped = build(design)
    P.note(job, f"claude: {design.get('object')!r}, {len(design.get('parts') or [])} parts ({skipped} not buildable),"
           f" suitable: {design.get('suitable')}")  # fmt: skip
    for _ in range(REVIEWS):
        step("claude checking")
        messages += turn + [{"role": "user", "content": [image_block(drawing(work, model), "image/png"),
                                                         {"type": "text", "text": REVIEW}]}]  # fmt: skip
        try:
            fixed, turn = ask_claude(client, messages)
            model, skipped = build(fixed)
            design = fixed
            P.note(job, f"claude review: {len(design.get('parts') or [])} parts ({skipped} not buildable)")
        except P.Failed as e:  # първият проект остава
            P.note(job, f"claude review failed ({e}); keeping the first design")
            break
    real = float(model.extents.max())
    return model, {"object": str(design.get("object") or "")[:60], "suitable": bool(design.get("suitable")),
                   "real_mm": round(real, 1) if 5 <= real <= 2000 else None}  # fmt: skip


# --- Meshy ----------------------------------------------------------------------------------------------------------

PICK = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"frames": {"type": "array", "items": {"type": "integer"}}, "object": {"type": "string"}},
    "required": ["frames", "object"],
}


def claude_picks(found: list[Path]) -> tuple[list[Path], str] | None:
    """Езиковият модел избира кадрите, които най-добре показват предмета от различни страни; None без ключ или при
    какъвто и да е проблем (кадрите тогава се разпределят равномерно)."""
    if not key("anthropic"):
        return None
    import anthropic

    shown = spread(found, 16)
    content = []
    for i, f in enumerate(shown):
        content += [{"type": "text", "text": f"Frame {i}:"}, image_block(jpeg(f, 512))]
    content.append({"type": "text", "text": (
        f"These frames are from a video of one object. Pick up to {MESHY_FRAMES} frame numbers for building a 3D model "
        "of the main object: sharp, the whole object in view, and from different sides (front, sides, back, a "
        "little from above), best first. Also name the object in Bulgarian, 1-4 words.")})  # fmt: skip
    try:
        client = anthropic.Anthropic(api_key=key("anthropic"), timeout=300.0, max_retries=2)
        msg = client.beta.messages.create(
            model=MODEL, max_tokens=16000, messages=[{"role": "user", "content": content}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": PICK}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )  # fmt: skip
        if msg.stop_reason != "end_turn":
            return None
        got = json.loads(next(b.text for b in msg.content if b.type == "text"))
        idx = [i for i in dict.fromkeys(got["frames"]) if 0 <= i < len(shown)][:MESHY_FRAMES]
        return ([shown[i] for i in idx], str(got["object"])[:60]) if idx else None
    except (anthropic.APIError, json.JSONDecodeError, KeyError, StopIteration):
        return None


def meshy(path: str, method: str = "GET", body: dict | None = None) -> dict:
    r = requests.request(method, f"{MESHY}/{path}", json=body, timeout=120,
                         headers={"Authorization": f"Bearer {key('meshy')}"})  # fmt: skip
    if r.status_code == 401:
        raise P.Failed("Услугата „AI от снимки“ временно не работи. Опитайте по-късно.")
    if r.status_code == 402:
        raise P.Failed("Услугата „AI от снимки“ временно не работи. Опитайте по-късно.")
    if r.status_code == 429:
        raise P.Failed("Meshy е претоварен в момента. Опитайте отново след малко.")
    if not r.ok:
        raise P.Failed(f"Meshy върна грешка ({r.status_code}). Опитайте отново.")
    return r.json()


def by_photo(job: Path, found: list[Path], step: Callable[[str], None]) -> tuple[trimesh.Trimesh, dict]:
    step("choosing frames")
    picks = claude_picks(found)
    picked, name = picks if picks else (spread(found, MESHY_FRAMES), "")
    P.note(job, f"meshy: {len(picked)} frames ({'picked by Claude' if picks else 'spread evenly'}): "
           + ", ".join(p.name for p in picked))  # fmt: skip
    urls = ["data:image/jpeg;base64," + base64.b64encode(jpeg(p, 2048)).decode() for p in picked]
    kind = "multi-image-to-3d" if len(urls) > 1 else "image-to-3d"
    body = {"should_texture": True, "target_formats": ["glb"]}
    body |= {"image_urls": urls} if len(urls) > 1 else {"image_url": urls[0]}
    task = meshy(kind, "POST", body)["result"]
    P.note(job, f"meshy: task {task}")
    started = time.time()
    while True:
        step("ai modelling")
        t = meshy(f"{kind}/{task}")
        if t["status"] == "SUCCEEDED":
            break
        if t["status"] in ("FAILED", "CANCELED"):
            reason = (t.get("task_error") or {}).get("message") or t["status"]
            P.note(job, f"meshy: {reason}")
            raise P.Failed("AI не успя да изгради модел от тези кадри. Опитайте с видео, в което предметът се вижда целият.")
        if time.time() - started > MESHY_WAIT_S:
            raise P.Failed("AI не завърши за 20 минути. Опитайте отново по-късно.")
        time.sleep(5)
    step("downloading the model")
    r = requests.get(t["model_urls"]["glb"], timeout=300)
    r.raise_for_status()
    scene = trimesh.load(io.BytesIO(r.content), file_type="glb", force="scene")
    pieces = []
    for node in scene.graph.nodes_geometry:
        transform, name_ = scene.graph[node]
        g = scene.geometry[name_].copy()
        if not isinstance(g, trimesh.Trimesh):
            continue
        try:
            g.visual = g.visual.to_color()  # текстурата като цветове на върховете: това показват екранът за преглед и рисунката
        except Exception:
            pass
        g.apply_transform(transform)
        pieces.append(g)
    if not pieces:
        raise P.Failed("AI върна празен модел. Опитайте отново.")
    model = trimesh.util.concatenate(pieces)
    model.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # glTF с Y нагоре -> Z нагоре
    return model, {"object": name, "suitable": True, "real_mm": None}


# --- Безплатният AI модел: AI очертания и дълбочина от цялото видео (app/outline.py)
# ---------------------------------

# Кадрите, които решаващият алгоритъм за камерите разполага: два бързи кръга за 33 с дадоха 24 кадъра през 30° и не
# повече от 4 можеха да се разположат заедно; 48 са през около 15°, като при бързо сканиране (app/video.py). Всеки се
# съпоставя със съседите си и с рядък набор през видеото (pipeline.match_pairs), а предметът се изрязва от всеки:
# формата се измерва и слива от всичките 48 (от всеки втори главата и тялото на Фродо излизаха на бучки, а ръката и
# мечът едно петно: всяко място е виждано от двойно по-малко кадри и шумът на дълбочината му се осреднява двойно
# по-малко). Потребителят избра детайла пред времето (2026-11-01).
AI_FRAMES = 48
MASK_EVERY = 1
MIN_FRAMES = 10  # извлечени са по-малко кадри: TripoSR само от един кадър
# Правилно разположени кадри (съгласувани с очертанията на предмета), от които се прави формата; при по-малко: TripoSR
# от един кадър. Брой, а не дял от изрязаните кадри: при изрязване на всеки втори половината от 24 бяха 12;
# изрязването на всичките 48 го направи 24, а дрелка, чиито два кръга излязоха на две части (най-добрата част: 19
# правилно разположени), отиде при модела от един кадър, петно със залепена дръжка на куфар (задача f4e8, 2026-11-01).
MIN_PLACED = 12
SINGLE_CANDIDATES = 12  # (само един кадър) кадри, в които се търси предметът: най-добрият се моделира, а друг може да покаже гърба му
SINGLE_SIDE = 2560  # снимките влизат с толкова: надписите отпред идват направо от тях


def single(job: Path, work: Path, flat_base: bool, set_state: Callable[[str], None],
           sh: Callable[[str, list], None], tune: dict | None = None) -> dict:  # fmt: skip
    """AI моделът от цялото видео; TripoSR от един кадър, когато кадрите не могат да се разположат в 3D (много кратко
    или размазано видео, едноцветен предмет върху едноцветна маса)."""
    photos, video = job / "photos", job / P.VIDEO
    tune = {**(tune or {}), "started": time.time()}
    frames_wanted = tune.get("frames") or AI_FRAMES
    if video.exists():
        try:
            P.extract_frames(video, photos, work, sh, P.frames_to_take(job, frames_wanted))
        except P.Failed as e:  # твърде кратко за решаващия алгоритъм за камерите; TripoSR иска един кадър
            P.note(job, f"AI model: {e}")
        P.merge_more_frames(job, photos)
    elif tune.get("frames"):
        photos = fewer_frames(work, photos, frames_wanted)  # тестова задача с вече избрани кадри
    found = sorted(photos.glob("*.jp*g"))
    if len(found) >= MIN_FRAMES and not tune.get("one_frame"):
        try:
            report = from_video(job, work, photos, len(found), flat_base, set_state, sh, tune)
            if report is not None:
                return report
        except Exception as e:  # каквото и да се е объркало, моделът от един кадър пак си заслужава
            if type(e).__name__ == "Cancelled":  # от app/jobs.py (импортирането му тук би било циклично)
                raise
            P.note(job, f"AI model from all the frames: failed ({e!r}); making it from one frame")
            TX.clear(job)  # текстура, направена за този модел, ако се е стигнало дотам
    if not found:
        found = frames(job, sh)
    set_state("reconstructing")
    model, info = by_single(job, work, found, sh, float(tune.get("relief") or 1.0))
    set_state("cleaning")
    two_sided = (work / photo3d.INFO).exists()
    # Фигура, снимана отгоре, е наклонена спрямо пода на TripoSR: плоска основа би отрязала ъгъл от нея.
    out = M.make_printable(model, float(model.bounds[0, 2]), flat_base and not two_sided,
                           smoothing=0 if two_sided else M.LIGHT, max_faces=photo3d.FACES if two_sided else M.MAX_FACES)  # fmt: skip
    if two_sided:
        try:
            P.note(job, f"one frame model: texture {photo3d.texture_model(job, work, out)}")
        except (OSError, ValueError, KeyError) as e:
            TX.clear(job)
            P.note(job, f"one frame model: no texture ({e}); the vertex colours")
    return P.finish(job, out) | {"photos_used": 1, "photos_total": len(found), "ai": "single"} | info


def from_video(job: Path, work: Path, photos: Path, n_photos: int, flat_base: bool,
               set_state: Callable[[str], None], sh: Callable[[str, list], None],
               tune: dict | None = None) -> dict | None:  # fmt: skip
    """Кадрите, разположени в 3D; предметът, изрязан от тях с AI и уточнен с MobileSAM; формата вътре във всички тези
    очертания; измерената между кадрите дълбочина (OpenMVS) с AI дълбочина (app/depth.py), слети върху нея
    (app/fuse.py): релефът, който очертанията не показват (ръб на основа, ръце, гънки, коса); текстурата от снимки
    върху него (app/outline.py). None, когато твърде малко кадри могат да се разположат."""
    tune = tune or {}
    all_masks = P.object_masks(work, photos, sh)
    forced = tune.get("solve")
    in_hand = forced == "object" if forced else P.camera_stayed_still(job, photos, all_masks)
    best, placed, stray = None, 0, []
    # Телефон, стоял на място, не може да разположи предмета по стаята. Телефон, който е обикалял, би трябвало да може,
    # но предмет върху еднообразна маса може да се разположи само по собствените си особености.
    for object_only in [in_hand] if in_hand or forced else [False, True]:
        best, placed, stray, all_masks, _ = P.solve_cameras(job, work, photos, n_photos, sh, paired=True,
                                                            mask_every=MASK_EVERY, features=tune.get("features"),
                                                            known_masks=all_masks, object_only=object_only)  # fmt: skip
        in_hand = object_only
        if best is not None and placed >= MIN_PLACED:
            break
    return model_from_solved(job, work, photos, flat_base, set_state, sh, best, placed, stray, all_masks, tune, in_hand)


ROUNDS = 2  # проходи на MobileSAM, всеки с подсказки от формата на очертанията от предишния
# Обещаното време. Когато не стига, отпада това, което се усеща най-малко: вторият проход на MobileSAM, проверяващият
# проход на гъстото съпоставяне и част от кадрите на детайла. Времената са от дрелка с 47 кадъра на сървъра.
TARGET_S = 285
DENSE_S = 3.0  # секунди гъсто съпоставяне на кадър
AFTER_DENSE_S = 150  # от края на гъстото съпоставяне до края на задачата
SECOND_ROUND_S = 15
DETAIL_S = 60  # детайлът от всички кадри и текстурата след него


def seconds_left(tune: dict) -> float:
    """Колко остава до обещаното време; безкрайно за тестова задача, която сравнява пълното качество (tune "full")."""
    if tune.get("full") or "started" not in tune:
        return float("inf")
    return TARGET_S - (time.time() - tune["started"])


def model_from_solved(job: Path, work: Path, photos: Path, flat_base: bool, set_state: Callable[[str], None],
                      sh: Callable[[str, list], None], best: Path | None, placed: int, stray: list[str],
                      all_masks: dict[str, np.ndarray], tune: dict | None = None, in_hand: bool = False) -> dict | None:  # fmt: skip
    """from_video след камерите: разположени кадри (`placed` от тях съгласувани с очертанията на търсача на предмета,
    `stray` несъгласувани) и маските на търсача. Те решават само откъде да се започне. При дрелка върху куфар той
    включваше куфара, дръжката му, стол или кутия отзад или пропускаше дръжката или патронника в 26 от 48 кадъра с
    напълно верни камери (едно цяло, задача 988c): съдени по тези маски, 26-те отпадаха, а формата се правеше от 22
    кадъра, някои от тях все още грешни, с издълбана до стъбло дръжка. Затова формата от съгласуваните кадри дава
    подсказки на MobileSAM за всеки разположен кадър (outlines_from_3d, ROUNDS пъти) и отпадат само кадри, чиито
    уточнени очертания все още се разминават с другите. in_hand: предметът е държан в ръка: няма маса, а сензорът за
    гравитация на телефона не казва нищо за него."""
    if best is None or placed < MIN_PLACED:
        P.note(job, f"AI model: only {placed} of the {len(all_masks)} frames with the object cut out placed in 3D; "
               "making it from one frame")  # fmt: skip
        return None
    tune = {k: v for k, v in (tune or {}).items() if v}  # Tune от app/main.py: настройките на тестова задача
    shown = {k: v for k, v in tune.items() if k != "started"}
    if shown:
        P.note(job, f"AI model: test settings {shown}")
    # Същите 14 минути, по-остър печат: при дрелка логото и етикетът излязоха по-ясни, с по-малко невидени стени (11%
    # -> 7%).
    if not tune.get("no_refine"):
        best = P.refine_cameras(job, work, best, sh)
    side = tune.get("side", OL.SIDE)
    solved = best
    # Един размер за всичко след това, както при експресно сканиране: картите на дълбочината, AI дълбочината и маските
    # трябва да делят пикселите си (AI дълбочина при 1920 спрямо карти при 1280 не пасваше на нито един кадър).
    dense = work / "dense"
    sh("undistorting", [P.COLMAP, "image_undistorter", "--image_path", photos, "--input_path", best,
        "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(side)])  # fmt: skip
    views = P.read_views(dense / "sparse")
    turned = P.undistort_masks(all_masks, best, dense / "sparse")
    if turned is None:
        masks_file = work / "undistorted_masks.npz"
        sh("finding the object", [sys.executable, S.__file__, dense / "images", masks_file])
        with np.load(masks_file) as z:
            masks = [z[n] for n in views.names]
    else:
        masks = [turned[0][n] for n in views.names]
    agreed = np.array([n in all_masks and n not in stray for n in views.names])

    up = None if in_hand else P.gravity_up(job, views)
    pts, cols = P.read_points_bin(dense / "sparse" / "points3D.bin")
    first = M.subset(views, agreed)
    first_masks = [m for m, a in zip(masks, agreed) if a]
    try:
        cloud = M.clean_cloud(pts, None, np.c_[cols / 255.0, np.ones(len(cols))], first, first_masks, S.MASK_SCALE, up,
                              find_table=not in_hand)  # fmt: skip
        P.note(job, f"AI model: {len(views)} frames placed, {placed} agreeing with the object finder's outlines, "
               f"{len(cloud.points)} of the solver's points on the object")  # fmt: skip
    except M.MeshError:  # твърде малко точки върху едноцветен предмет: само очертанията го разполагат
        cloud = outline_cloud(first, first_masks, up)
        if cloud is None:
            raise P.Failed("the frames' outlines give no shape")
        P.note(job, f"AI model: {len(views)} frames placed, {placed} agreeing, few points on the object")

    rest = DENSE_S * len(views) + AFTER_DENSE_S
    rounds = tune.get("rounds") or (ROUNDS if seconds_left(tune) >= rest + SECOND_ROUND_S else 1)
    masks, keep, sharpened = outlines_from_3d(job, work, dense, views, masks, agreed, cloud, pts, sh, rounds)
    if keep.sum() < MIN_PLACED:
        P.note(job, f"AI model: only {keep.sum()} frames agree on the object's outline; making it from one frame")
        return None
    if not keep.all():  # кадри с очертания, които още се разминават: извън гъстото съпоставяне и формата
        dropped = [n for n, k in zip(views.names, keep) if not k]
        kept = P.drop_stray(work, dense / "sparse", dropped, sh, "outline_kept")
        shutil.rmtree(dense / "sparse")
        shutil.copytree(kept, dense / "sparse")
        views, masks = M.subset(views, keep), [m for m, k in zip(masks, keep) if k]
    images = [dense / "images" / n for n in views.names]
    agree = tune.get("agree") or (OL.AGREE_SHARP if sharpened else M.HULL_AGREE)
    geometric = not tune.get("no_geo") and seconds_left(tune) >= rest
    P.note(job, f"AI model: {seconds_left(tune):.0f} s left of the promised time, {rounds} rounds of outlines, "
           f"second pass of dense matching: {geometric}")  # fmt: skip

    surface, cloud = OL.measured(job, work, dense, views, masks, cloud, up, agree, set_state, sh, side=side,
                                 voxel=tune.get("voxel", OL.VOXEL), fixed="agree" in tune,
                                 table=not in_hand and not tune.get("no_table"), flap=tune.get("flap"), wide=tune.get("wide_dense", False),
                                 geometric=geometric, passes=tune.get("passes"), iters=tune.get("iters"))  # fmt: skip
    set_state("cleaning")
    M.paint(surface, views, images, cloud.to_up)
    out = M.make_printable(surface, M.floor_z(cloud), flat_base and not in_hand)
    out = with_detail(job, work, dense / "images", views, masks, out, cloud, sh, tune)
    used = set(views.names)
    posed, texture_masks = texture_frames(work, solved, [n for n in all_masks if n not in used], all_masks, sh)
    P.photo_texture(job, work, photos, posed, out, cloud, sh, texture_masks)
    width = float(np.median([S.object_width(m) for m in masks]))
    return P.finish(job, out) | {"photos_used": len(views), "photos_total": len(all_masks),
                                 "object_width": round(width, 3), "ai": "single", "object": "", "suitable": True,
                                 "real_mm": None}  # fmt: skip


def with_detail(job: Path, work: Path, images: Path, views: M.Views, masks: list[np.ndarray], out: trimesh.Trimesh,
                cloud: M.Cloud, sh: Callable[[str, list], None], tune: dict) -> trimesh.Trimesh:  # fmt: skip
    """Моделът за печат с фините детайли от AI дълбочината (app/relief.py); какъвто е, ако това се провали."""
    if tune.get("no_relief"):
        return out
    to_world = np.eye(4)
    to_world[:3, :3] = cloud.to_up.T
    to_world = to_world @ np.linalg.inv(out.metadata["to_print"])
    inputs, result = work / "relief_inputs.npz", work / "relief.ply"
    colors = out.visual.vertex_colors if out.visual.kind == "vertex" else np.full((len(out.vertices), 4), 160, np.uint8)
    np.savez(inputs, names=np.array(views.names), R=views.R, t=views.t, K=views.K, mask_scale=S.MASK_SCALE,
             vertices=out.vertices, faces=out.faces, colors=colors, to_world=to_world, strength=tune.get("relief") or 1.0,
             views=tune.get("relief_views") or (RL.VIEWS if seconds_left(tune) >= DETAIL_S else RL.FEWER_VIEWS),
             faces_out=tune.get("relief_faces") or RL.FACES,
             **{f"masks_{i}": m for i, m in enumerate(masks)})  # fmt: skip
    try:
        sh("adding detail", [sys.executable, RL.__file__, images, inputs, result])
        info = json.loads(result.with_suffix(".json").read_text())
        fine = trimesh.load(result, force="mesh")
    except (P.Failed, OSError, ValueError, KeyError) as e:
        P.note(job, f"fine detail: failed ({e}); the model without it")
        return out
    fine.metadata["to_print"] = out.metadata["to_print"]
    P.note(job, f"fine detail: {info}")
    return fine


def fewer_frames(work: Path, photos: Path, count: int) -> Path:
    """Копие с равномерно избрани кадри (само за тестова задача, чиито кадри трябва да останат)."""
    names = sorted(p.name for p in photos.glob("*.jp*g"))
    if len(names) <= count:
        return photos
    fewer = work / "fewer_frames"
    fewer.mkdir(exist_ok=True)
    for i in np.linspace(0, len(names) - 1, count).round().astype(int):
        shutil.copy(photos / names[i], fewer / names[i])
    return fewer


def outlines_from_3d(job: Path, work: Path, dense: Path, views: M.Views, masks: list[np.ndarray], agreed: np.ndarray,
                     cloud: M.Cloud, pts: np.ndarray, sh, rounds: int = ROUNDS) -> tuple[list[np.ndarray], np.ndarray, bool]:  # fmt: skip
    """Очертанията на предмета във всеки разположен кадър (коригирани `views`, `masks` на търсача): формата от кадрите
    със съгласувани очертания (първоначално `agreed`) дава подсказки на MobileSAM за всички
    (pipeline.sharpened_masks); кадрите с нови очертания, съгласувани с другите (mesh.consistent_frames), дават
    формата за следващия кръг. Връща (маските, кои кадри са съгласувани, дали се използват тези на MobileSAM);
    собствените на търсача, когато MobileSAM не може да се използва."""
    current, trusted, sharpened = list(masks), agreed.copy(), False
    for round_ in range(rounds):
        idx = np.flatnonzero(trusted)
        found = M.hull_volume(M.subset(views, trusted), [current[i] for i in idx], S.MASK_SCALE, cloud)
        if found is None:
            break
        sharp = P.sharpened_masks(work, dense, views, current, cloud, found, pts, sh)
        if sharp is None:
            break
        keep = M.consistent_frames(views, sharp, S.MASK_SCALE)
        P.note(job, f"AI model: outlines round {round_ + 1}: {keep.sum()} of {len(views)} frames agree "
               f"(before: {trusted.sum()})")  # fmt: skip
        if keep.sum() < MIN_PLACED:  # очертанията на MobileSAM са твърде несъгласувани: остават последните
            break
        current, trusted, sharpened = sharp, keep, True
    return current, trusted, sharpened


def texture_frames(work: Path, solved: Path, stray: list[str], masks: dict[str, np.ndarray],
                   sh: Callable[[str, list], None]) -> tuple[Path, dict[str, np.ndarray]]:  # fmt: skip
    """Кадрите, от които се прави текстурата: тези с изрязан и правилно разположен предмет, а също и неизрязаните
    между тях (при MASK_EVERY > 1), чиито съседи от двете страни във видеото са правилно разположени: двойно повече
    кадри, така че всяка стена получава по-прав и по-близък изглед (Фродо: текстурата идваше от 19 от 48 кадъра, а
    6,7% от стените не се виждаха от нито един). Кадрите между тях взимат изрязването на съсед, за да се измери
    рязкостта им. Когато всеки кадър е изрязан, това са правилно разположените кадри. Връща (модела на камерите на
    тези кадри, {кадър: маска})."""
    names = P.read_views(solved).names
    placed = set(names)
    good = {n for n in names if n in masks and n not in stray}
    order = sorted(placed | set(masks))
    use, borrowed = set(good), dict(masks)
    for i, n in enumerate(order):
        if n in masks or not 0 < i < len(order) - 1:
            continue
        before, after = order[i - 1], order[i + 1]
        if before in good and after in good:
            use.add(n)
            borrowed[n] = masks[before]
    return P.drop_stray(work, solved, [n for n in names if n not in use], sh, "texture_frames"), borrowed


def outline_cloud(views: M.Views, masks: list[np.ndarray], up: np.ndarray | None) -> M.Cloud | None:
    """Заместител на точките от решаващия алгоритъм, когато малко от тях са върху предмета: вокселите на формата от
    очертанията (груби), завъртени така, че z да е нагоре. Маса не е известна; най-ниската точка е подът."""
    grid = M.hull_grid(views, masks, S.MASK_SCALE)
    if grid is None or len(grid[0]) < 10:
        return None
    to_up = M.rotation_to(up if up is not None else M.estimate_up(views.centers, views.dirs, views.ups))
    pts = grid[0] @ to_up.T
    return M.Cloud(pts, np.zeros_like(pts), np.ones((len(pts), 4)), None, to_up)


def by_single(job: Path, work: Path, found: list[Path], sh: Callable[[str, list], None],
              strength: float = 1.0) -> tuple[trimesh.Trimesh, dict]:  # fmt: skip
    pick = work / "single"
    pick.mkdir(parents=True, exist_ok=True)
    for f in spread(found, SINGLE_CANDIDATES):
        with Image.open(f) as im:  # снимка от телефон: изправена според данните в нея
            im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((SINGLE_SIDE, SINGLE_SIDE), Image.LANCZOS)
        im.save(pick / f.name, "JPEG", quality=95)
    masks, out = work / "single_masks.npz", work / "single.ply"
    sh("finding the object", [sys.executable, S.__file__, pick, masks])
    sh("ai modelling", [sys.executable, single3d.__file__, pick, masks, out, str(strength)])
    info = json.loads(out.with_suffix(".json").read_text())
    model = trimesh.load(out, force="mesh")
    P.note(job, f"single frame AI: {info}")
    return model, {"object": "", "suitable": True, "real_mm": None}


# --- Задачата -------------------------------------------------------------------------------------------------------


def run(job: Path, method: str, flat_base: bool, set_state: Callable[[str], None], sh: Callable[[str, list], None],
        step: Callable[[str], None], tune: dict | None = None) -> dict:  # fmt: skip
    """step(name): какво показва приложението в момента (хвърля грешка, ако задачата е отказана). Връща отчета за
    печата."""
    if not available().get(method):
        raise P.Failed("Тази AI услуга още не е включена. Опитайте по-късно.")
    work = job / "work"
    shutil.rmtree(work, ignore_errors=True)  # от повторно изпълнена задача
    work.mkdir(parents=True)
    TX.clear(job)  # текстура от снимки от сканиране, ако задачата преди е била сканиране
    set_state("reconstructing")
    if method == "single":
        saved = json.loads(db.get_setting("ai_tune") or "{}")  # настройките по подразбиране (PUT /admin/tune)
        return single(job, work, flat_base, set_state, sh, saved | (tune or {}))
    found = frames(job, sh)
    P.note(job, f"ai ({method}): {len(found)} frames")
    if method == "claude":
        model, info = by_claude(job, work, found, step)
        set_state("cleaning")
        out = M.make_printable(model, float(model.bounds[0, 2]), False, smoothing=0)  # проектиран да стои върху z = 0
    else:
        model, info = by_photo(job, found, step)
        set_state("cleaning")
        out = M.make_printable(model, float(model.bounds[0, 2]), flat_base)
    return P.finish(job, out) | {"photos_used": len(found), "photos_total": len(found), "ai": method} | info
