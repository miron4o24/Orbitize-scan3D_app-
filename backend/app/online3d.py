"""Безплатен онлайн AI модел: TRELLIS (microsoft/TRELLIS, MIT) в Hugging Face Space на безплатните им видеокарти (ZeroGPU).
С безплатен ключ (HF_TOKEN) се получават около 5 минути видеокарта на ден, което е 3-5 модела; после до следващия ден
сървърът прави модела сам, както преди (app/ai.py single). Няколко кадъра от видеото (предметът отвсякъде, с изрязан фон)
стават един модел с текстура за около минута, без дупки, но формата на невидяното AI я досеща.

Собствен процес, за да може задачата да се откаже: python online3d.py <out.glb> <снимка.png> [още снимки...]
Ключът идва от средата (HF_TOKEN) и не се пише никъде. Изход 3: дневният лимит е свършил (out.json: "wait" в секунди)."""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

SPACE = os.environ.get("HF_3D_SPACE", "trellis-community/TRELLIS")
WAIT_S = float(os.environ.get("HF_3D_WAIT_S", "240"))  # опашка + изграждане; после моделът се прави на сървъра
TEXTURE = 2048
SIMPLIFY = 0.95  # дял от триъгълниците, които TRELLIS маха (същото като в демото му)
QUOTA = 3


def wait_from(message: str) -> float | None:
    """Колко секунди до новата дневна квота от съобщението на ZeroGPU ("Try again in 23:59:25"); None, ако не е за квота."""
    if "quota" not in message.lower():
        return None
    m = re.search(r"(\d+):(\d+):(\d+)", message)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else 3600.0


def make(out: Path, images: list[Path]) -> dict:
    from gradio_client import Client, handle_file

    started = time.time()
    client = Client(SPACE, hf_token=os.environ.get("HF_TOKEN") or None, verbose=False)
    client.predict(api_name="/start_session")  # папката на сесията в Space-а, иначе изграждането не намира къде да пише
    many = len(images) > 1
    client.predict(api_name="/lambda_1" if many else "/lambda")  # раздел "няколко снимки" / "една снимка" на демото
    files = [{"image": handle_file(str(p)), "caption": None} for p in images]
    ready = client.predict(images=files, api_name="/preprocess_images")  # центриране и мащаб както го иска TRELLIS
    ready = [{"image": handle_file(r["image"]), "caption": None} for r in ready]
    print(f"online AI: {len(ready)} pictures ready after {time.time() - started:.0f} s", flush=True)
    job = client.submit(image=ready[0]["image"], multiimages=ready, seed=0, ss_guidance_strength=7.5, ss_sampling_steps=12,
                        slat_guidance_strength=3.0, slat_sampling_steps=12, multiimage_algo="stochastic",
                        mesh_simplify=SIMPLIFY, texture_size=TEXTURE, api_name="/generate_and_extract_glb")  # fmt: skip
    result = job.result(timeout=max(30.0, WAIT_S - (time.time() - started)))
    glb = result[1] if isinstance(result[1], str) else (result[1] or {}).get("value") or (result[1] or {}).get("path")
    shutil.copy(glb, out)
    return {"space": SPACE, "pictures": len(images), "seconds": round(time.time() - started, 1), "bytes": out.stat().st_size}


def main(out: Path, images: list[Path]) -> int:
    info_file = out.with_suffix(".json")
    try:
        info = make(out, images)
    except Exception as e:  # всякаква грешка на услугата: сървърът прави модела сам
        message = str(e)
        wait = wait_from(message)
        info_file.write_text(json.dumps({"error": message[:500], "wait": wait}))
        print(f"online AI: failed: {type(e).__name__}: {message[:300]}", flush=True)
        return QUOTA if wait is not None else 1
    info_file.write_text(json.dumps(info))
    print(f"online AI: model in {info['seconds']} s from {info['pictures']} pictures", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]))
