"""A text change the fixed rules (app/edit.py parse) could not read, turned into app/edit.py's operations by a small
language model on the CPU: Qwen2.5-1.5B-Instruct (Apache-2.0), in tools/qwen (scripts/models.ps1). Greedy decoding:
the same request gives the same answer. Its answer is checked by app/edit.py validate before anything is done.

Runs as its own process (step "understanding the request"), so its memory is freed afterwards:
    python edit_llm.py <request.json: {prompt, sizes}> <answer.json: {ops, unsupported}>"""

import json
import os
import re
import sys
from pathlib import Path

MODEL_DIR = Path(os.environ.get("EDIT_LLM_DIR", Path(__file__).resolve().parent.parent / "tools" / "qwen"))
MAX_NEW_TOKENS = 400

SYSTEM = """You turn a request to change a 3D-printable model into a list of operations, as JSON.
The model stands on a table. length = the longer side of its footprint, width = the shorter side, height = up.
All sizes are in millimetres (1 cm = 10 mm, 1 inch = 25.4 mm). The operations:
{"op":"resize","dimension":"longest|length|width|height","mm":N}   scale the whole model evenly so that this size becomes N mm
{"op":"scale","factor":F}   scale evenly (2 = twice as big, 0.5 = half the size)
{"op":"grow","axis":"longest|length|width|height","mm":N}   make that size N mm bigger (negative: smaller); longest scales evenly, the others stretch in one direction only
{"op":"stretch","axis":"length|width|height","factor":F}   stretch or squash in one direction only
{"op":"rotate","axis":"length|width|height","degrees":D}   height: spin it on the table; length or width: tip it over (90 lays it on its side)
{"op":"mirror","axis":"length|width|height"}
{"op":"hollow","wall_mm":N}   make it hollow inside with walls N mm thick
{"op":"hole","diameter_mm":N,"axis":"height|length|width","part":""}   a round hole straight through the middle along that direction (height: from the top down); with a part: through that part, into the surface
{"op":"cut","side":"bottom|top","mm":N}   cut N mm off the bottom or the top
{"op":"base","height_mm":N,"shape":"round|square"}   add a stand under it
{"op":"loop","diameter_mm":N,"part":""}   add a ring to hang it, e.g. on a keychain (N: the opening); on top, or on the part
{"op":"smooth","strength":1|2|3,"part":""}   smooth the surface (3: strongly), of the whole model or of the part
{"op":"thicken","mm":N,"part":""}   make the part (or the whole model) N mm thicker all round; negative: thinner
{"op":"simplify","percent":P}   keep P percent of the triangles
{"op":"color","part":"hair","color":"blue"}   paint the part a new colour; color: an English colour name or #rrggbb
"part": what the change is for: 1-3 English words naming a part as seen on the model ("hair", "shirt", "base", "left arm"); "here" when the request says here/this/there and the user tapped a spot; "" for the whole model.
Use them in the order asked. Pick sensible sizes when none are given (hole 5 mm, wall 2 mm, base 3 mm, loop 5 mm).
If part of the request cannot be done with these operations (text, patterns, adding other objects, changing the shape's details), leave it out and name it in "unsupported".
Answer with JSON only, in this form: {"ops":[...],"unsupported":""}"""

EXAMPLES = [
    ("Sizes now: length 80 mm, width 40 mm, height 120 mm.\nRequest: I need it a bit bigger, like 15cm high, and put a hole for a string at the top",
     '{"ops":[{"op":"resize","dimension":"height","mm":150},{"op":"loop","diameter_mm":5}],"unsupported":""}'),
    ("Sizes now: length 100 mm, width 60 mm, height 30 mm.\nRequest: save plastic, make the inside empty with thick walls of 3mm, and make it stand on a square plate",
     '{"ops":[{"op":"hollow","wall_mm":3},{"op":"base","height_mm":3,"shape":"square"}],"unsupported":""}'),
    ("Sizes now: length 50 mm, width 50 mm, height 70 mm.\nRequest: paint it red and turn it so it lies down",
     '{"ops":[{"op":"color","part":"","color":"red"},{"op":"rotate","axis":"length","degrees":90}],"unsupported":""}'),
    ("Sizes now: length 60 mm, width 55 mm, height 100 mm.\nRequest: give him blonde hair, a green jacket and a smiley face on the shirt",
     '{"ops":[{"op":"color","part":"hair","color":"blonde"},{"op":"color","part":"jacket","color":"green"}],"unsupported":"a smiley face on the shirt"}'),
    ("Sizes now: length 70 mm, width 40 mm, height 100 mm.\nThe user tapped a spot on the model.\nRequest: this bit is too rough and too thin, fix it and put a small hole through it for a string",
     '{"ops":[{"op":"smooth","strength":2,"part":"here"},{"op":"thicken","mm":1,"part":"here"},{"op":"hole","diameter_mm":3,"axis":"height","part":"here"}],"unsupported":""}'),
]  # fmt: skip


def ready() -> bool:
    return (MODEL_DIR / "model.safetensors").exists() and (MODEL_DIR / "tokenizer.json").exists()


def first_json(text: str) -> dict:
    """The first {...} in the answer that reads as JSON."""
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                try:
                    out = json.loads(text[start : end + 1])
                    if isinstance(out, dict):
                        return out
                except json.JSONDecodeError:
                    pass
                break
    return {"ops": [], "unsupported": "no answer"}


def main(request: Path, answer: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the answer goes to the job's log (UTF-8)
    req = json.loads(request.read_text())
    s = req["sizes"]
    tapped = "The user tapped a spot on the model.\n" if req.get("tapped") else ""
    question = (f"Sizes now: length {s['length']:g} mm, width {s['width']:g} mm, height {s['height']:g} mm.\n"
                f"{tapped}Request: {req['prompt']}")  # fmt: skip
    messages = [{"role": "system", "content": SYSTEM}]
    for q, a in EXAMPLES:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": question})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    model.eval()
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(text, flush=True)
    answer.write_text(json.dumps(first_json(text)))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
