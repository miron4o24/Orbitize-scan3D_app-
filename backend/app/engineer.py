"""Инженерен режим (метод "cad"): 8 до 40 снимки на малък детайл и мерки с шублер -> CAD модел 1:1 от елементи
(STEP за CAD програмите, кодът на CadQuery и STL), а не една мрежа.

1. Сканиране от снимките с проверения начин на AI модела (app/ai.py model_from_solved: камерите с COLMAP, очертанията,
   измерената и AI дълбочината), само с по-малко снимки: стига формата за положенията и размерите.
2. CAD модел: локално и безплатно (app/cad_local.py) за всеки или, ако собственикът го избере, от Claude по абонамента
   му (app/cad_claude.py), започвайки от локалния. Ако Claude не успее, остава локалният.
3. Моделът в приложението е CAD моделът (сив, без текстура); файловете са в cad/ (GET /jobs/{id}/cad/{файл})."""

import json
import shutil
import sys
from pathlib import Path
from typing import Callable

import trimesh
from PIL import Image, ImageOps

from . import ai as AI
from . import cad_claude
from . import cad_local
from . import mesh as M
from . import pipeline as P
from . import texture as TX

METHOD = "cad"
PHOTOS_MIN, PHOTOS_MAX = 8, 40
MIN_PLACED = 6  # снимки, разположени и съгласувани с очертанията (при видео: 12 от 48)
SIDE = 2400  # дългата страна на снимките за сканирането
FILES = {"part.step": "model/step", "part.py": "text/x-python; charset=utf-8", "part.stl": "model/stl",
         "part.json": "application/json", "compare.png": "image/png"}  # fmt: skip
TIPS = ("Снимките не можаха да се свържат в 3D. Снимайте отново: детайлът върху шарена (не едноцветна) повърхност, "
        "10 до 20 снимки в кръг около него, всяка с голямо припокриване със съседната, част от тях отгоре, без "
        "приближаване и без светкавица.")  # fmt: skip


def prepared(job: Path, work: Path) -> Path:
    """Снимките изправени (EXIF), не по-големи от SIDE и всички с еднакъв размер, с имената им.

    COLMAP пази една камера за всички снимки и пропуска тези с друг размер: телефонът записва снимка, държан изправен,
    като 2448x3264, а легнал като 3264x2448 (наблюдавано: 7 от 10 пропуснати). Затова снимките в другата посока се
    завъртат на 90° към посоката на повечето (за намирането на камерите посоката в кадъра няма значение), а снимка с
    друго съотношение се изрязва около средата до него."""
    out = work / "photos"
    out.mkdir(parents=True, exist_ok=True)
    ims = []
    for p in sorted((job / "photos").glob("*.jp*g")):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((SIDE, SIDE))
            ims.append((p.name, im))
    if not ims:
        return out
    wide = sum(im.width >= im.height for _, im in ims) * 2 >= len(ims)
    turned = [(n, im if (im.width >= im.height) == wide else im.transpose(Image.Transpose.ROTATE_90)) for n, im in ims]
    sizes = [im.size for _, im in turned]
    size = max(set(sizes), key=sizes.count)
    for name, im in turned:
        if im.size != size:
            im = ImageOps.fit(im, size, Image.Resampling.LANCZOS)  # изрязва около средата до съотношението
        im.save(out / name, quality=95)
    return out


def run(job: Path, params: dict, set_state: Callable[[str], None], sh: Callable[[str, list], None],
        step: Callable[[str], None]) -> dict:  # fmt: skip
    cad = params.get("cad") or {}
    meas = {k: cad.get(k) for k in ("length", "width", "height") if cad.get(k)}
    meas["features"] = [f for f in cad.get("features") or [] if f.get("value")]
    engine = cad.get("engine") or "local"
    work, out = job / "work", job / "cad"
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True)
    TX.clear(job)
    set_state("reconstructing")
    photos = prepared(job, work)
    n = len(list(photos.glob("*.jpg")))
    if n < PHOTOS_MIN:
        raise P.Failed(f"Нужни са поне {PHOTOS_MIN} снимки.")
    P.note(job, f"engineering: {n} photos, measures {json.dumps(meas, ensure_ascii=False)}, engine {engine}")

    # 1. Сканирането (подреден кръг от снимки: всяка с всяка, без двойките за видео).
    try:
        best, placed, stray, masks, _ = P.solve_cameras(job, work, photos, n, sh, paired=False, mask_every=1)
    except P.Failed as e:
        raise P.Failed(TIPS) from e
    scanned = AI.model_from_solved(job, work, photos, True, set_state, sh, best, placed, stray, masks, None,
                                   min_placed=MIN_PLACED)  # fmt: skip
    if scanned is None:
        raise P.Failed(TIPS)
    TX.clear(job)  # CAD моделът е сив: текстурата е на сканирането
    out.mkdir(parents=True)
    scan = out / "scan.ply"
    shutil.copy(job / "model.ply", scan)
    (out / "measures.json").write_text(json.dumps(meas, ensure_ascii=False), encoding="utf-8")

    # 2. CAD модел.
    set_state("meshing")
    try:
        sh("CAD model", [sys.executable, cad_local.__file__, scan, out / "measures.json", out])
    except P.Failed as e:
        raise P.Failed("От сканирането не се получи CAD модел. Проверете мерките и снимайте отново по-отблизо.") from e
    local = json.loads((out / "part.json").read_text(encoding="utf-8"))
    report = {"engine": "local", "local": local}
    if engine == "claude":
        step("CAD by Claude")
        for f in ("part.py", "part.step", "part.stl"):
            shutil.copy(out / f, out / f"local_{f}")
        originals = sorted((job / "photos").glob("*.jp*g"))  # work/ е изчистена от сканирането
        got = cad_claude.run(work / "claude", originals, meas, out / "scan_mm.stl",
                             out / "part.py", lambda t: P.note(job, t))  # fmt: skip
        if got["ok"]:
            for f in ("part.py", "part.step", "part.stl", "compare.png"):
                if (got["dir"] / f).exists():
                    shutil.copy(got["dir"] / f, out / f)
            report = {"engine": "claude", "local": local, "claude": got["check"], "rounds": got["rounds"]}
        else:
            P.note(job, f"CAD by Claude failed ({got.get('error')}); keeping the local model")
            report["claude_failed"] = True
    if not (out / "compare.png").exists():
        try:
            from . import cad_check

            cad_check.views([("scan", trimesh.load(out / "scan_mm.stl", force="mesh")),
                             ("model", trimesh.load(out / "part.stl", force="mesh"))], out / "compare.png")  # fmt: skip
        except Exception as e:  # само за прегледа
            P.note(job, f"compare views: {e!r}")

    # 3. Моделът в приложението: CAD моделът, пазен с най-дълга страна 100 мм като всеки модел (истинската е real_mm).
    set_state("cleaning")
    model = trimesh.load(out / "part.stl", force="mesh")
    real = float(max(model.extents))
    model.apply_scale(100.0 / real)
    model.export(job / "model.ply")
    shutil.rmtree(work, ignore_errors=True)
    (job / P.VIDEO).unlink(missing_ok=True)
    return M.report(model) | {"ai": METHOD, "real_mm": round(real, 2), "photos_used": scanned.get("photos_used"),
                              "photos_total": n, "object": "", "suitable": True, "cad": report}  # fmt: skip
