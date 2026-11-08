"""Инженерен режим с Claude: Claude Code на сървъра (влязъл с абонамента на собственика, без платен API ключ) гледа
снимките, мерките и рисунките на сканирането и пише модела като код на CadQuery (елемент по елемент, както би го
направил конструктор). Сървърът го изпълнява и проверява (app/cad_check.py: размери спрямо мерките, разминаване със
сканирането, рисунки модел/сканиране) и му връща проверката за поправки, до ROUNDS пъти.

Само за профила на собственика (accounts.OWNER_EMAILS): абонаментът е за собствено ползване. Настройка (веднъж) в
backend/.env: CLAUDE_CLI=<пълният път до claude.exe или claude.cmd> и CLAUDE_CODE_OAUTH_TOKEN=<от `claude setup-token`>.
Платен API ключ (ANTHROPIC_API_KEY) никога не се подава на Claude Code оттук, за да не се таксува нищо отделно."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import trimesh
from PIL import Image, ImageOps

from . import cad_check

ROUNDS = int(os.environ.get("CAD_CLAUDE_ROUNDS", "3"))  # проверки и поправки след първия модел
TIMEOUT_S = int(os.environ.get("CAD_CLAUDE_TIMEOUT", "900"))  # на кръг
PHOTO_SIDE = 1568  # дългата страна на снимките, които Claude вижда
MOST_PHOTOS = 12
PAID = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX", "AWS_BEARER_TOKEN_BEDROCK")  # fmt: skip


def cli() -> str | None:
    path = os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    return path if path and Path(path).exists() else None


def ready() -> bool:
    """Claude Code е инсталиран и има вход с абонамент (токен или папка с настройки)."""
    return cli() is not None and bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("CLAUDE_CONFIG_DIR"))


def plan_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in PAID}
    env["DISABLE_AUTOUPDATER"] = "1"
    env["DISABLE_TELEMETRY"] = "1"
    return env


def ask(folder: Path, prompt: str, session: str | None) -> dict:
    """Един кръг на Claude Code в `folder` (може само да чете и пише файлове там). Връща JSON изхода му."""
    argv = [cli(), "-p", "--output-format", "json", "--allowedTools", "Read,Write,Edit", "--max-turns", "40"]
    if os.environ.get("CAD_CLAUDE_MODEL"):
        argv += ["--model", os.environ["CAD_CLAUDE_MODEL"]]
    if session:
        argv += ["--resume", session]
    # Задачата е на стандартния вход: при claude.cmd аргумент с нови редове не минава през cmd.exe.
    r = subprocess.run(argv, input=prompt, cwd=folder, env=plan_env(), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=TIMEOUT_S)  # fmt: skip
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except (json.JSONDecodeError, IndexError):
        out = {}
    if r.returncode != 0 or out.get("is_error"):
        raise RuntimeError((out.get("result") or r.stderr or r.stdout or f"exit {r.returncode}")[-800:])
    return out


def task(meas: dict, n_photos: int, have_start: bool) -> str:
    overall = ", ".join(f"{k} {meas[k]:g} mm" for k in ("length", "width", "height") if meas.get(k))
    feats = "\n".join(f"- {f.get('label') or f['kind']}: {f['value']:g} mm ({f['kind']})" for f in meas.get("features") or [])
    return f"""You are a senior mechanical design engineer. Make an exact, editable CAD model of the real part shown in
this folder, the way you would draw it in SolidWorks or Fusion: from features, not a mesh.

Files here:
- photo_01.jpg ... photo_{n_photos:02d}.jpg: the part on a table, photographed from around it.
- measures.json: the user's caliper measurements in mm. They are exact and override anything else.
- scan_views.png: renders of a 3D scan made from these photos, already scaled to the measurements (front = looking
  along +Y, right = looking along -X, top = looking down, iso). Grid lines are a scale reference (10 mm, or 5 mm for
  parts under 40 mm). The scan is accurate for overall shape and positions (about ±0.3 mm) but blurs small details;
  the photos show the details.
{"- start.py: a model fitted automatically to the scan (CadQuery). Use it for positions and sizes it got right; fix or"
 " replace what is wrong or missing." if have_start else ""}

Overall size (exact): {overall or "not given"}.
Feature measurements:
{feats or "- none"}

Coordinate frame (same as the scan): X along the length (the longest side lying on the table), Y along the width, Z up.
The part's bottom is at Z = 0 and it is centred at X = 0, Y = 0.

Write the model to part.py:
- Python using only `import cadquery as cq` and `import math` (nothing else is allowed and it will be rejected).
- Put every dimension in a named variable at the top (mm), then build with sketches, extrude/revolve, cuts, holes
  (.hole / .cboreHole / .cskHole), fillets and chamfers that are visible in the photos.
- Use standard values when the photos clearly show a standard feature and no measurement is given (for example
  clearance holes for M3/M4/M5 screws, common fillet radii), and say so in a comment.
- Do not invent features you cannot see. Keep one solid.
- The final solid must be in a variable named `result`.
Only read files in this folder and write part.py. When done, reply with one short line about what you modelled."""


def feedback(report: dict) -> str:
    if not report.get("ok"):
        return (f"Running part.py failed:\n{report.get('error')}\nFix part.py (only cadquery and math are allowed) and "
                "write it again.")  # fmt: skip
    return f"""I ran your part.py. Check results:
- size (mm): {json.dumps(report["size_mm"])}; difference from the measurements (mm): {json.dumps(report["off_from_measured_mm"])}
- distance between your model and the scan: mean {report["deviation_from_scan_mm"]["mean"]} mm, 90% within
  {report["deviation_from_scan_mm"]["p90"]} mm
- bounding box min corner (should be about -length/2, -width/2, 0): {report["min_corner"]}
- compare.png: top row the scan, bottom row your model, same views and scale.
Compare carefully with the photos and compare.png: wrong or missing features, positions, sizes, holes, fillets.
If something is off, fix part.py. If the model already matches the photos and the measurements, reply exactly DONE
and do not change part.py."""


def run(folder: Path, photos: list[Path], meas: dict, scan_mm: Path, start: Path | None,
        note: Callable[[str], None]) -> dict:  # fmt: skip
    """Модел от Claude в `folder`. Връща {"ok", "rounds", "check" (на избрания), "dir" (с part.py, part.step,
    part.stl)} или {"ok": False, "error"}."""
    folder.mkdir(parents=True, exist_ok=True)
    chosen = photos if len(photos) <= MOST_PHOTOS else [photos[round(i)] for i in
                                                        [k * (len(photos) - 1) / (MOST_PHOTOS - 1) for k in range(MOST_PHOTOS)]]  # fmt: skip
    for k, p in enumerate(chosen, 1):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((PHOTO_SIDE, PHOTO_SIDE))
            im.save(folder / f"photo_{k:02d}.jpg", quality=90)
    (folder / "measures.json").write_text(json.dumps(meas, ensure_ascii=False, indent=1), encoding="utf-8")
    cad_check.views([("scan", trimesh.load(scan_mm, force="mesh"))], folder / "scan_views.png")
    if start is not None and start.exists():
        shutil.copy(start, folder / "start.py")
    session, best, prompt = None, None, task(meas, len(chosen), start is not None and start.exists())
    for k in range(ROUNDS + 1):
        try:
            out = ask(folder, prompt, session)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            note(f"CAD by Claude: round {k + 1} failed ({str(e)[:300]})")
            break
        session = out.get("session_id") or session
        said = (out.get("result") or "").strip()
        if not (folder / "part.py").exists():
            prompt = "part.py was not written. Write the model to part.py as described."
            continue
        if k > 0 and said.upper().startswith("DONE"):
            note(f"CAD by Claude: done after {k} checks")
            break
        checked = folder / f"check_{k}"
        r = subprocess.run([sys.executable, cad_check.__file__, folder / "part.py", scan_mm, folder / "measures.json",
                            checked], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)  # fmt: skip
        try:
            report = json.loads((checked / "check.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {"ok": False, "error": (r.stderr or r.stdout)[-1500:]}
        note(f"CAD by Claude: round {k + 1}: " + (f"deviation {report['deviation_from_scan_mm']}, off "
             f"{report['off_from_measured_mm']}" if report.get("ok") else f"error {report.get('error', '')[:200]}"))  # fmt: skip
        if report.get("ok"):
            shutil.copy(folder / "part.py", checked / "part.py")
            if (checked / "compare.png").exists():
                shutil.copy(checked / "compare.png", folder / "compare.png")
            best = {"check": report, "dir": checked, "round": k + 1}
        prompt = feedback(report)
    if best is None:
        return {"ok": False, "error": "Claude did not produce a model that runs"}
    return {"ok": True, "rounds": best["round"], "check": best["check"], "dir": best["dir"]}
