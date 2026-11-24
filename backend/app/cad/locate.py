"""Кой е предметът на снимката и точното му очертание: Florence-2 (MIT) описва снимката и посочва в нея всяко нещо от
описанието, а SAM 2.1 (Apache-2.0) изрязва посочения предмет отблизо.

IS-Net изрязва "това, което се откроява", а MobileSAM - "предмета под точката". И двете грешат там, където трябва да
се разбере какво има на снимката: ключ на бяла картичка (откроява се картичката), надпис или лого върху детайла
(MobileSAM взима само него), детайл между пръстите. Florence-2 описва снимката ("A bunch of keys sitting on top of a
piece of paper") и намира кутия за всяка част от описанието; от тях се взима предметът, а не повърхността под него
или ръката (SCENE), най-близо до средата на кадъра (там приложението държи детайла). SAM 2.1 изрязва по тази кутия.

Двата модела работят на процесора чрез transformers (вече използвана от моделите на Pro) и заедно държат около 4 GB
при четене на снимка. Без файловете им инженерният режим работи както преди (vision.cutout)."""

import os
import gc
import re
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS = Path(__file__).resolve().parents[2] / "tools"
FLORENCE = Path(os.environ.get("FLORENCE_DIR", TOOLS / "florence2" / "base"))
SAM2 = Path(os.environ.get("SAM2_DIR", TOOLS / "sam2" / "large"))

# Последната дума на част от описанието, която казва, че това не е детайлът: повърхността под него, ръката, която го
# държи, или човекът. ("a wooden table" -> table; "A bunch of keys" -> keys остава.)
SCENE = {
    "table", "tables", "surface", "desk", "counter", "countertop", "paper", "papers", "sheet", "notebook", "book",
    "page", "card", "cardboard", "envelope", "floor", "ground", "wall", "door", "board", "background", "mat", "cloth",
    "tablecloth", "towel", "fabric", "carpet", "rug", "blanket", "bed", "couch", "sofa", "chair", "wood", "shelf",
    "person", "man", "woman", "boy", "girl", "child", "people", "someone", "hand", "hands", "finger", "fingers",
    "thumb", "arm", "palm", "lap", "leg", "knee", "shadow", "light", "reflection", "room", "box",
}  # fmt: skip
BIG = 0.6  # кутия, заемаща толкова от кадъра, е повърхността (масата, листът), каквото и да пише
MARGIN = 0.3  # поле около кутията при изрязването за SAM 2.1, в размери на кутията
PAD = 0.03  # кутията за SAM 2.1 се разширява с толкова от размера си: Florence-2 понякога реже ръба
MIN_SCORE = 0.5  # под тази оценка на SAM 2.1 (очакваното припокриване) изрязването не се използва
# Отвор в изрязаното от SAM 2.1 е истински (през него се вижда повърхността), когато SAM 2.1, посочен в средата му,
# уверено (оценка поне MIN_SCORE) вижда там нещо отделно от детайла: всяко уверено изрязване покрива под HOLE_PART от
# детайла. При отблясък по метал уверените изрязвания са самият детайл (85-100% от него): на 9-те снимки на ключове и
# кука истинските отвори бяха 0%, отблясъците 30-100%. Цветът не ги различава: ключ на бял лист е сив като листа.
HOLE_PART = 0.2

# Двата модела държат около 4 GB, а на сървъра свободни са около 6 GB заедно със сканиранията. Затова се зареждат
# при четене на снимки и се освобождават, когато IDLE секунди никой не ги е ползвал (шестте снимки на един детайл
# се четат с едно зареждане).
IDLE = 120

_models = None
_users = 0
_timer: threading.Timer | None = None
_lock = threading.Lock()


def available() -> bool:
    return (FLORENCE / "model.safetensors").exists() and (SAM2 / "model.safetensors").exists()


def models(threads: int):
    """(Florence-2, процесорът му, SAM 2.1, процесорът му), заредени веднъж; None без файловете."""
    global _models
    with _lock:
        if _models is None and available():
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration, Sam2Model, Sam2Processor

            torch.set_num_threads(threads)
            _models = (Florence2ForConditionalGeneration.from_pretrained(FLORENCE).eval(),
                       AutoProcessor.from_pretrained(FLORENCE),
                       Sam2Model.from_pretrained(SAM2).eval(), Sam2Processor.from_pretrained(SAM2))  # fmt: skip
        return _models


@contextmanager
def held(threads: int):
    """Моделите (или None без файловете) за времето на блока; освобождават се IDLE секунди след последния блок."""
    global _users, _timer
    with _lock:
        _users += 1
        if _timer is not None:
            _timer.cancel()
            _timer = None
    try:
        yield models(threads)
    finally:
        with _lock:
            _users -= 1
            if _users == 0 and _models is not None:
                _timer = threading.Timer(IDLE, release)
                _timer.daemon = True
                _timer.start()


def release():
    """Освобождава паметта на моделите, ако никой не ги ползва."""
    global _models, _timer
    with _lock:
        if _users == 0:
            _models, _timer = None, None
    gc.collect()


def florence(m, im: Image.Image, task: str, text: str = ""):
    import torch

    model, proc = m[0], m[1]
    inp = proc(text=task + text, images=im, return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(**inp, max_new_tokens=128, num_beams=3, do_sample=False)
    return proc.post_process_generation(proc.batch_decode(ids, skip_special_tokens=False)[0], task=task,
                                        image_size=im.size)[task]  # fmt: skip


def head(phrase: str) -> str:
    words = re.findall(r"[a-z]+", phrase.lower())
    return words[-1] if words else ""


# Където свършва подлогът на описанието ("A bunch of keys | sitting on top of a piece of paper").
SUBJECT_END = re.compile(r"\s(?:sitting|lying|laying|resting|standing|placed|held|being|that|which|is|are|on|in|with|"
                         r"next|against|under|over|near|by|at)\s", re.I)
SENTENCE = 5  # част от описанието с повече думи е цялото изречение: казва какво има, но не кое е в кутията


def subject(caption: str) -> str:
    """Подлогът на описанието, без "A close up of": кое е снимано."""
    text = re.sub(r"^\s*(?:an?|the)\s+close[- ]up\s+(?:view\s+)?of\s+", "", caption.strip(), flags=re.I)
    return SUBJECT_END.split(" " + text + " ", maxsplit=1)[0].strip(" .")


# Florence-2 често дава на листа под детайла и името на детайла: "A computer mouse" и "paper" с почти една и съща кутия
# (пауърбанк на бял лист, задача 14a3b5). Кутия, която съвпада поне толкова (IoU) с кутия на повърхност, е повърхността.
TWIN = 0.85
# Кутия до ръба на снимката (дял от размера ѝ) е нещо друго в кадъра (мишка отстрани): детайлът в инженерния режим е
# целият в кадъра, иначе снимката така или иначе се отхвърля ("cut").
EDGE_BOX = 0.005


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_scene(box, label: str, W: int, H: int) -> bool:
    """Повърхност, ръка или човек по името (ако не е цялото изречение) или повърхност по размера (BIG)."""
    sentence = len(re.findall(r"[a-z]+", label.lower())) > SENTENCE
    return (not sentence and head(label) in SCENE) or (box[2] - box[0]) * (box[3] - box[1]) > BIG * W * H


def scene_boxes(found: dict, W: int, H: int) -> list[tuple]:
    return [tuple(b) for b, label in zip(found["bboxes"], found["labels"]) if is_scene(b, label, W, H)]


def candidates(found: dict, W: int, H: int, scene: list | None = None) -> list[tuple[float, tuple]]:
    """(разстояние до средата на кадъра, кутия) за частите, които не са повърхност, ръка или човек: нито по името или
    размера си (is_scene), нито като двойник на кутия на повърхност (TWIN, scene: и от предишно търсене), нито до ръба
    на снимката (EDGE_BOX)."""
    scene = scene_boxes(found, W, H) + list(scene or [])
    edge = EDGE_BOX * max(W, H)
    out = []
    for box, label in zip(found["bboxes"], found["labels"]):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0 or is_scene(box, label, W, H) or any(iou(box, s) >= TWIN for s in scene):
            continue
        if x0 <= edge or y0 <= edge or x1 >= W - edge or y1 >= H - edge:
            continue
        out.append((((x0 + x1) / 2 - W / 2) ** 2 + ((y0 + y1) / 2 - H / 2) ** 2, tuple(box)))
    return out


def object_box(m, im: Image.Image) -> tuple[tuple[float, float, float, float], str] | None:
    """Кутията на детайла (пиксели на снимката) и описанието на снимката; None, ако в описанието няма нещо, което не е
    повърхност, ръка или човек."""
    caption = florence(m, im, "<CAPTION>")
    W, H = im.size
    found = florence(m, im, "<CAPTION_TO_PHRASE_GROUNDING>", caption)
    cands = candidates(found, W, H)
    if not cands:  # посочена е само повърхността ("of paper"): още веднъж само подлогът ("A bunch of keys")
        name = subject(caption)
        if name and head(name) not in SCENE:
            cands = candidates(florence(m, im, "<CAPTION_TO_PHRASE_GROUNDING>", name), W, H, scene_boxes(found, W, H))
    if not cands:
        return None
    return min(cands)[1], caption


def cut(m, im: Image.Image, box) -> tuple[np.ndarray, tuple[int, int, int, int], float, Image.Image, object]:
    """SAM 2.1 по кутията, в изрязване около нея: (мека маска 0..1 на изрязването, изрязването, оценка, изображението
    на изрязването, вграждането му за through)."""
    import torch

    model, proc = m[2], m[3]
    W, H = im.size
    x0, y0, x1, y1 = box
    pad = PAD * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    margin = MARGIN * max(x1 - x0, y1 - y0)
    crop_box = (int(max(0, x0 - margin)), int(max(0, y0 - margin)), int(min(W, x1 + margin)), int(min(H, y1 + margin)))
    crop = im.crop(crop_box)
    local = [x0 - crop_box[0], y0 - crop_box[1], x1 - crop_box[0], y1 - crop_box[1]]
    inp = proc(images=crop, input_boxes=[[local]], return_tensors="pt")
    with torch.inference_mode():
        emb = model.get_image_embeddings(inp["pixel_values"])
        out = model(image_embeddings=emb, input_boxes=inp["input_boxes"], multimask_output=False)
    logits = proc.post_process_masks(out.pred_masks, inp["original_sizes"], binarize=False)[0][0, 0].numpy()
    soft = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return soft.astype(np.float32), crop_box, float(out.iou_scores.flatten()[0]), crop, emb


def through(m, crop: Image.Image, emb, point: tuple[float, float], solid: np.ndarray) -> bool:
    """Отворът с точка `point` (пиксели на изрязването) в средата е истински: виж HOLE_PART."""
    import torch

    model, proc = m[2], m[3]
    inp = proc(images=crop, input_points=[[[[float(point[0]), float(point[1])]]]], input_labels=[[[1]]],
               return_tensors="pt")  # fmt: skip
    with torch.inference_mode():
        out = model(image_embeddings=emb, input_points=inp["input_points"], input_labels=inp["input_labels"],
                    multimask_output=True)  # fmt: skip
    masks = proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0][0].numpy()
    area = max(int(solid.sum()), 1)
    sure = [mk for mk, s in zip(masks, out.iou_scores.flatten().tolist()) if s >= MIN_SCORE]
    return bool(sure) and all((mk & solid).sum() / area < HOLE_PART for mk in sure)
