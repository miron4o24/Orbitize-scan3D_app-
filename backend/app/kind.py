"""Какъв е предметът на една снимка, за да се избере как да се направи моделът (app/photo3d.py): плосък (дистанционно, телефон,
книга) става тяло с обичайната за вида дебелина, кръгъл (чаша, бутилка, ваза) се завърта около оста си, а всичко друго (фигура,
обувка, играчка) идва от формата на TripoSR. Разпознаването е с CLIP (openai/clip-vit-base-patch32, MIT, като ONNX в tools/clip;
scripts/models.ps1), безплатно на процесора: предметът върху сив фон срещу описанията в KINDS. CLIPSeg (за цветовете) не става
за това: и двете дистанционни, и фотоапаратът излизаха "телефон", докато CLIP ги позна (0,83 и 0,91; фотоапарат 0,40).
Собствен процес: python kind.py <снимка> <masks.npz> <out.json>; без модела не се записва нищо и остава обичайният път."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_DIR = Path(os.environ.get("CLIP_DIR", Path(__file__).resolve().parent.parent / "tools" / "clip"))
# (описание, име на български за заглавието на модела, вид, дебелина / дължина за плосък)
KINDS = [
    ("a remote control", "Дистанционно", "flat", 0.09),
    ("a mobile phone", "Телефон", "flat", 0.06),
    ("a book", "Книга", "flat", 0.14),
    ("a tablet computer", "Таблет", "flat", 0.035),
    ("a calculator", "Калкулатор", "flat", 0.08),
    ("a wallet", "Портфейл", "flat", 0.12),
    ("a cup", "Чаша", "round", None),
    ("a mug", "Чаша", "round", None),
    ("a drinking glass", "Чаша", "round", None),
    ("a bottle", "Бутилка", "round", None),
    ("a vase", "Ваза", "round", None),
    ("a tin can", "Консерва", "round", None),
    ("a jar", "Буркан", "round", None),
    ("a candle", "Свещ", "round", None),
    ("a bowl", "Купа", "round", None),
    ("a flower pot", "Саксия", "round", None),
    ("a ball", "Топка", "round", None),
    ("a toy figure", "Фигура", "other", None),
    ("a statue", "Статуетка", "other", None),
    ("a shoe", "Обувка", "other", None),
    ("a toy", "Играчка", "other", None),
    ("a stuffed animal", "Плюшена играчка", "other", None),
    ("a power tool", "Инструмент", "other", None),
    ("a camera", "Фотоапарат", "other", None),
    ("a cardboard box", "Кутия", "other", None),
    ("a cereal box", "Кутия", "other", None),
    ("a food package", "Опаковка", "other", None),
    ("a plant", "Растение", "other", None),
    ("a lamp", "Лампа", "other", None),
    ("a game controller", "Джойстик", "other", None),
    ("headphones", "Слушалки", "other", None),
]
SURE = 0.6  # плосък или кръгъл само ако описанията от този вид заедно имат поне толкова от вероятността: иначе обичайната форма
MARGIN_CROP = 0.1  # от размера на предмета: толкова около него остава в изображението
ROUND_SHARE = 0.75  # кръгъл предмет: поне толкова от очертанието е в огледално еднаквото тяло около оста (дръжката на чаша е извън него)
ROUND_AXIS = 0.08  # от ширината: средата на редовете се мести най-много толкова (изправен предмет)
SIDE = 224
MEAN, STD = np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711])
PAD_ID = 49407  # краят на текста в речника на CLIP, с него се допълва


def ready() -> bool:
    return (MODEL_DIR / "model_quantized.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()


def round_shape(inside: np.ndarray) -> tuple[float, float]:
    """(дял от очертанието в огледално еднаквото тяло около отвесна ос, колко се мести средата на редовете спрямо ширината)."""
    rows = np.flatnonzero(inside.any(1))
    sub = inside[rows]
    left = sub.argmax(1)
    right = sub.shape[1] - 1 - sub[:, ::-1].argmax(1)
    centres = (left + right) / 2
    axis = float(np.median(centres))
    radius = np.maximum(np.minimum(axis - left, right - axis), 0) + 0.5
    x = np.arange(sub.shape[1])[None, :] - axis
    body = np.abs(x) <= radius[:, None]
    width = float(np.median(right - left + 1))
    return float((body & sub).sum()) / max(1.0, float(sub.sum())), float(np.std(centres)) / max(width, 1.0)


def picture(photo: Image.Image, inside: np.ndarray) -> np.ndarray:
    """Предметът върху сив фон в квадрат SIDE, нормиран за CLIP (1, 3, SIDE, SIDE): без фона, който иначе подвежда."""
    ys, xs = np.nonzero(inside)
    pad = int(MARGIN_CROP * max(np.ptp(ys), np.ptp(xs)))
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(photo.width, xs.max() + pad + 1), min(photo.height, ys.max() + pad + 1))
    part = inside[box[1] : box[3], box[0] : box[2]]
    grey = np.where(part[..., None], np.asarray(photo.crop(box)), 128).astype(np.uint8)
    h, w = part.shape
    s = max(w, h)
    square = Image.new("RGB", (s, s), (128, 128, 128))
    square.paste(Image.fromarray(grey), ((s - w) // 2, (s - h) // 2))
    x = np.asarray(square.resize((SIDE, SIDE), Image.BICUBIC), np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def chances(photo: Image.Image, inside: np.ndarray) -> np.ndarray:
    """Вероятността на всяко описание от KINDS за предмета на снимката."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_padding(pad_id=PAD_ID, length=77)
    tok.enable_truncation(77)
    texts = tok.encode_batch([f"a photo of {prompt}." for prompt, *_ in KINDS])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(MODEL_DIR / "model_quantized.onnx"), opts, providers=["CPUExecutionProvider"])
    logits = session.run(["logits_per_image"], {"input_ids": np.array([t.ids for t in texts], np.int64), "pixel_values": picture(photo, inside),
                                                "attention_mask": np.array([t.attention_mask for t in texts], np.int64)})[0][0]  # fmt: skip
    p = np.exp(logits - logits.max())
    return p / p.sum()


def decide(found: np.ndarray, inside: np.ndarray) -> dict:
    """Видът от вероятностите на KINDS и очертанието: плосък или кръгъл само ако описанията от този вид заедно са сигурни (SURE), а
    кръгъл и само ако очертанието наистина е еднакво от двете страни на отвесна ос. Името е на най-вероятното описание."""
    best = int(np.argmax(found))
    prompt, name, group, thin = KINDS[best]
    share = float(sum(p for p, k in zip(found, KINDS) if k[2] == group))
    why = f"{group} {share:.2f}"
    if group != "other" and share < SURE:
        group, thin, why = "other", None, why + " not sure enough"
    if group == "round":
        whole, drift = round_shape(inside)
        why += f", round share {whole:.2f}, axis drift {drift:.3f}"
        if whole < ROUND_SHARE or drift > ROUND_AXIS:
            group = "other"
    top = [(KINDS[i][0], round(float(found[i]), 2)) for i in np.argsort(found)[::-1][:3]]
    return {"prompt": prompt, "name": name, "group": group, "thin": thin, "sure": round(float(found[best]), 3), "why": why, "top": top}


def main(photo_path: Path, masks_file: Path, out: Path) -> None:
    if not ready():
        print("object kind: the CLIP model is not installed", flush=True)
        return
    with Image.open(photo_path) as im:
        photo = ImageOps.exif_transpose(im).convert("RGB")
    with np.load(masks_file) as z:
        m = z[photo_path.name]
    inside = np.asarray(Image.fromarray(m).resize(photo.size, Image.BILINEAR)) > 127
    if inside.sum() < 500:
        print("object kind: no object", flush=True)
        return
    got = decide(chances(photo, inside), inside)
    out.write_text(json.dumps(got, ensure_ascii=False))
    print(f"object kind: {got['prompt']} -> {got['group']} ({got['why']}); best {got['top']}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
