"""Pro's text changes: the user writes what to change ("make it 15 cm tall and add a keychain loop"), and a new model is
made from the old one; the old one stays as it was.

The request becomes a list of operations (OPS). Plain requests are read by fixed rules (parse): exact, instant, and
no guessing. A request the rules do not fully cover goes to a small language model on the server (app/edit_llm.py:
Qwen2.5-1.5B-Instruct, Apache-2.0, on the CPU), whose answer is held to the same operations and limits (validate).
What was done is listed in the new model's report, in Bulgarian, for the app.

Sizes are the app's (the viewer's length, width and height): length is the longer side of the footprint, width the
shorter, height up. The model is changed at its real size (the size the user gave in the app), in millimetres.

Colours ("make the hair blue"): the part named is found by AI on pictures of the model (app/recolor.py) and its photo
texture and vertex colours are turned to the new colour, shading kept. The photo texture stays with the new model
when the shape changes only by moving, scaling, mirroring or hollowing (its outside unchanged); other changes leave
it with its vertex colours, as before."""

import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from . import mesh as M
from . import pipeline as P
from . import recolor as RC
from . import texture as TX

SOURCE = "source.ply"  # in the new job's directory: the model it was made from, copied when the job was made
AXES = ("length", "width", "height")
DIMENSIONS = ("longest", *AXES)


class EditError(P.Failed):
    """User-facing: the request cannot be done."""


# --- Operations: what each takes, and its limits ---------------------------------------------------------------------

OPS: dict[str, dict] = {
    "resize": {"dimension": DIMENSIONS, "mm": (1.0, 2000.0)},
    "scale": {"factor": (0.05, 20.0)},
    "grow": {"axis": DIMENSIONS, "mm": (-2000.0, 2000.0)},
    "stretch": {"axis": AXES, "factor": (0.05, 20.0)},
    "rotate": {"axis": AXES, "degrees": (-360.0, 360.0)},
    "mirror": {"axis": AXES},
    "hollow": {"wall_mm": (0.4, 50.0)},
    "hole": {"diameter_mm": (0.3, 500.0), "axis": AXES},
    "cut": {"side": ("bottom", "top"), "mm": (0.1, 2000.0)},
    "base": {"height_mm": (0.4, 200.0), "shape": ("round", "square")},
    "loop": {"diameter_mm": (1.0, 200.0)},
    "smooth": {"strength": (1, 3)},
    "simplify": {"percent": (5.0, 95.0)},
    "color": {},  # color: a name or #rrggbb (validate)
    "thicken": {"mm": (-5.0, 5.0)},  # the surface pushed out (or in) along its normals
}
# Operations that may work on a part of the model only ("part"): a few English words naming it ("hair"), HERE for the
# spot the user tapped in the app, or "" for the whole model (hole and loop: where they go).
REGIONAL = {"color", "smooth", "thicken", "hole", "loop"}
HERE = "here"
DEFAULTS = {"smooth": {"strength": 2}, "thicken": {"mm": 1.0}, "hole": {"diameter_mm": 5.0, "axis": "height"},
            "loop": {"diameter_mm": 5.0}}  # fmt: skip
# The photo texture stays when the surface keeps its points (moved, mirrored, smoothed, thickened) or its outside
# (hollowed): it is moved with them.
KEEPS_TEXTURE = {"color", "resize", "scale", "grow", "stretch", "rotate", "mirror", "hollow", "smooth", "thicken"}
AXIS_WORDS = {"vertical": "height", "z": "height", "up": "height", "tall": "height", "long": "length", "wide": "width"}


def validate(raw: dict) -> dict:
    """The operation with its values checked against OPS; EditError (in Bulgarian) when it is not one of them."""
    name = str(raw.get("op", "")).strip().lower()
    if name == "colour":
        name = "color"
    if name not in OPS:
        raise EditError(f"Непозната промяна: „{name}“.")
    if name == "color":
        colour = str(raw.get("color") or raw.get("colour") or "").strip().lower()
        if RC.colour_rgb(colour) is None:
            raise EditError(f"Непознат цвят: „{colour}“. Напишете името му на английски (red, blue, blonde…) или #RRGGBB.")
        return {"op": "color", "part": part_name(str(raw.get("part") or "")), "color": colour}
    op = {"op": name}
    part = part_name(str(raw.get("part") or "")) if name in REGIONAL else ""
    for field, allowed in OPS[name].items():
        value = raw.get(field)
        if value is None or value == "":
            value = DEFAULTS.get(name, {}).get(field)
        if isinstance(allowed[0], str):
            value = AXIS_WORDS.get(str(value).strip().lower(), str(value).strip().lower())
            if value not in allowed:
                raise EditError(f"Промяната „{name}“ не може да се направи с „{raw.get(field)}“.")
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise EditError(f"Промяната „{name}“ няма число за „{field}“.")
            lo, hi = allowed
            if not (np.isfinite(value) and lo <= value <= hi):
                raise EditError(f"Стойността {value:g} е извън границите за „{name}“.")
            if isinstance(lo, int):
                value = int(round(value))
        op[field] = value
    if name in ("grow", "thicken") and op["mm"] == 0:
        raise EditError("Промяна с 0 мм не променя нищо.")
    if name in REGIONAL:
        op["part"] = part
    return op


COLOUR_BG = {
    "red": "червено", "dark red": "тъмночервено", "orange": "оранжево", "yellow": "жълто", "gold": "златисто",
    "golden": "златисто", "green": "зелено", "dark green": "тъмнозелено", "light green": "светлозелено",
    "blue": "синьо", "dark blue": "тъмносиньо", "light blue": "светлосиньо", "navy": "тъмносиньо", "navy blue": "тъмносиньо",
    "purple": "лилаво",
    "violet": "виолетово", "pink": "розово", "brown": "кафяво", "dark brown": "тъмнокафяво",
    "light brown": "светлокафяво", "black": "черно", "white": "бяло", "grey": "сиво", "gray": "сиво",
    "silver": "сребристо", "blonde": "русо", "blond": "русо", "platinum blonde": "платинено русо",
    "ginger": "риждаво", "auburn": "кестеняво-червено", "brunette": "кестеняво", "chestnut": "кестеняво",
    "beige": "бежово", "turquoise": "тюркоазено", "cyan": "синьозелено", "copper": "медно", "bronze": "бронзово",
}  # fmt: skip
NAMES = {"longest": "най-дългата страна", "length": "дължината", "width": "ширината", "height": "височината"}
AROUND = {"length": "около дължината", "width": "около ширината", "height": "около височината"}


def where(op: dict) -> str:
    """Where a part-wise change was made, for describe: "" for the whole model."""
    part = op.get("part") or ""
    return "" if not part else " на посоченото място" if part == HERE else f" на „{part}“"


def describe(op: dict) -> str:
    """One line, in Bulgarian, for the app's list of what was done."""
    n = op["op"]
    if n == "resize":
        return f"Размер: {NAMES[op['dimension']]} {op['mm']:g} мм"
    if n == "scale":
        return f"Мащаб ×{op['factor']:g}"
    if n == "grow":
        return f"{NAMES[op['axis']].capitalize()} {'+' if op['mm'] > 0 else '−'}{abs(op['mm']):g} мм"
    if n == "stretch":
        return f"{NAMES[op['axis']].capitalize()} ×{op['factor']:g}"
    if n == "rotate":
        return f"Завъртане на {op['degrees']:g}° {AROUND[op['axis']]}"
    if n == "mirror":
        return f"Огледален образ по {NAMES[op['axis']]}"
    if n == "hollow":
        return f"Кух, със стени {op['wall_mm']:g} мм"
    if n == "hole":
        if op.get("part"):
            return f"Отвор Ø{op['diameter_mm']:g} мм{where(op)}, навътре през модела"
        side = {"height": "отгоре надолу", "length": "по дължината", "width": "по ширината"}[op["axis"]]
        return f"Отвор Ø{op['diameter_mm']:g} мм през средата, {side}"
    if n == "cut":
        return f"Отрязани {op['mm']:g} мм {'отдолу' if op['side'] == 'bottom' else 'отгоре'}"
    if n == "base":
        return f"{'Кръгла' if op['shape'] == 'round' else 'Квадратна'} основа, {op['height_mm']:g} мм висока"
    if n == "loop":
        if op.get("part"):
            return f"Халка за окачване{where(op)}, отвор Ø{op['diameter_mm']:g} мм"
        return f"Халка за окачване отгоре, отвор Ø{op['diameter_mm']:g} мм"
    if n == "color":
        part = "посоченото място" if op["part"] == HERE else f"„{op['part']}“" if op["part"] else "целият модел"
        return f"Цвят: {part} → {COLOUR_BG.get(op['color'], op['color'])}"
    if n == "smooth":
        return f"Изглаждане ({['леко', 'средно', 'силно'][op['strength'] - 1]}){where(op)}"
    if n == "thicken":
        return f"{'По-дебело' if op['mm'] > 0 else 'По-тънко'} с {abs(op['mm']):g} мм{where(op) or ' навсякъде'}"
    return f"Опростяване до {op['percent']:g}% от триъгълниците"


# --- Reading the request by fixed rules -------------------------------------------------------------------------------

NUM = r"(\d+(?:[.,]\d+)?)"
UNIT = r"\s*(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|inch(?:es)?|\")?(?![a-z])"
TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4, '"': 25.4}
FILLER = re.compile(r"\b(please|pls|thanks|thank you|can you|could you|would you|i want(?: you)?(?: to)?|i'd like(?: you)?(?: to)?|"
                    r"i would like(?: you)?(?: to)?|the model|this model|the object|this|it|its|a|an|the)\b")  # fmt: skip


def number(text: str) -> float:
    return float(text.replace(",", "."))


def unit_mm(unit: str | None) -> float:
    """mm per unit; no unit: millimetres."""
    if not unit:
        return 1.0
    u = unit.lower()
    for key, k in TO_MM.items():
        if u.startswith(key) and (key != "m" or u in ("m", "meter", "meters", "metre", "metres")):
            return k
    if u.startswith("millimet"):
        return 1.0
    if u.startswith("centimet"):
        return 10.0
    return 1.0


def length_mm(clause: str) -> float | None:
    """The first length in the clause, in mm (a bare number counts as mm); None if there is none."""
    for m in re.finditer(NUM + UNIT, clause):
        after = clause[m.end() :].lstrip()
        if after.startswith(("%", "percent", "°", "deg", "x ", "times")):
            continue
        return number(m[1]) * unit_mm(m[2])
    return None


def percent(clause: str) -> float | None:
    m = re.search(NUM + r"\s*(?:%|percent)", clause)
    return number(m[1]) if m else None


def degrees(clause: str) -> float | None:
    m = re.search(NUM + r"\s*(?:°|deg(?:ree)?s?)", clause)
    return number(m[1]) if m else None


def has(clause: str, pattern: str) -> bool:
    return re.search(pattern, clause) is not None


COMPARE = {  # word: (axis, sign)
    "taller": ("height", 1), "higher": ("height", 1), "shorter": ("height", -1), "lower": ("height", -1),
    "wider": ("width", 1), "narrower": ("width", -1), "thinner": ("width", -1),
    "longer": ("length", 1), "bigger": ("longest", 1), "larger": ("longest", 1), "smaller": ("longest", -1),
}  # fmt: skip


COLOUR_WORDS = re.compile(r"#[0-9a-f]{6}\b|\b(?:(?P<shade>light|pale|bright|dark|deep|lighter|paler|brighter|darker|deeper)"
                          r"\s+)?(?P<name>" + "|".join(sorted((re.escape(k) for k in RC.COLOURS), key=len, reverse=True))
                          + r")\b")  # fmt: skip
# A clause with any of these (or a number) is a shape change: the other rules or the language model read it.
NOT_COLOUR = re.compile(r"\b(add|adding|holes?|loop|keyring|keychain|hook|hollow|cut|trim|chop|smooth\w*|simplif\w*|"
                        r"rotate|spin|tilt|mirror\w*|flip\w*|scale|resize|bigger|smaller|taller|wider|longer|shorter|"
                        r"higher|lower|thinner|narrower|upside)\b|\d")  # fmt: skip
# Words around a colour that are not the part: "change the hair colour to blue" -> "hair".
COLOUR_FILLER = set("""
a an the it its it's his her their my your this that these those of to into in with and be is are should please pls
can could would you i want like id d make made turn change changed set give paint painted colour color colours colors
coloured colored recolour recolor dye dyed tint tinted shade all whole entire complete completely fully totally just
only so model figure figurine object thing everything him them one
""".split())  # fmt: skip
WHOLE = {"", "model", "figure", "figurine", "object", "thing", "everything", "whole", "all", "it"}
WHOLE_WORDS = WHOLE - {"", "it"} | {"entire"}
# "make this red", "smooth here", "this part": the spot the user tapped in the app (HERE).
HERE_WORDS = {"here", "this", "that", "there", "selected", "marked", "tapped", "picked", "chosen", "pointed"}
PLACE_WORDS = {"part", "area", "spot", "place", "bit", "piece", "region", "point", "section", "zone"}
# Words in a part-wise change that are not the part: "smooth the hair a lot" -> "hair".
REGION_FILLER = COLOUR_FILLER | set("""
smooth smoother smoothen smoothed smoothing out very much lot more strong strongly slightly slight bit little lightly
light gently gentle really thick thicker thicken thickened thin thinner fatter slimmer hole holes drill drilled through
add put create loop keyring key ring keychain chain eyelet hook hang hanging string for mm millimeter millimeters
millimetre millimetres cm centimeter centimeters centimetre centimetres diameter wide big small size by around about at on
""".split())  # fmt: skip


def part_name(text: str, filler: set[str] = COLOUR_FILLER) -> str:
    """The part a change is for, as CLIPSeg is asked about it: a few lowercase English words; HERE for the spot the
    user tapped ("this", "here", "this part"); "" for the whole model."""
    words = re.findall(r"[a-z]+", text.lower().replace("'s", ""))
    here = bool(set(words) & HERE_WORDS) and not set(words) & WHOLE_WORDS
    rest = [w for w in words if w not in filler and w not in HERE_WORDS and not (here and w in PLACE_WORDS)]
    if rest:
        part = " ".join(rest[:4])
        return "" if part in WHOLE else part
    return HERE if here else ""


def colour_rule(clause: str) -> dict | None:
    """"make the hair blue", "paint the base gold", "change his shirt colour to dark green", "make this #ff8800"."""
    found = list(COLOUR_WORDS.finditer(clause))
    if not found or NOT_COLOUR.search(COLOUR_WORDS.sub(" ", clause)):
        return None
    m = found[-1]
    rest = (clause[: m.start()] + " " + clause[m.end() :]).replace("'s", "")
    words = [w for w in re.findall(r"[a-z]+", rest) if w not in COLOUR_FILLER and w not in HERE_WORDS]
    if len(words) > 3 or len(found) > 1:  # "blue hair with red stripes": the language model reads it
        return None
    shade, name = m["shade"], m["name"]
    shade = {"lighter": "light", "paler": "pale", "brighter": "bright", "darker": "dark", "deeper": "deep"}.get(shade, shade)
    return {"op": "color", "part": part_name(rest), "color": f"{shade} {name}" if shade else m[0].strip()}


REGION_OP = re.compile(r"\b(smooth\w*|thick(?:er|en)?|thin(?:ner)?|fatter|slimmer|holes?|loop|keyring|keychain|"
                       r"key ?ring|key ?chain|eyelet|hook)\b")  # fmt: skip


def region_rule(clause: str) -> dict | None:
    """A change to one part or to the tapped spot: "smooth the hair", "smooth here a lot", "make this 1 mm thicker",
    "add a 3 mm hole here", "add a loop here". The whole model's ("smooth it") is left to rule()."""
    c = clause
    if not REGION_OP.search(c) or COLOUR_WORDS.search(c):
        return None
    part = part_name(re.sub(NUM + UNIT, " ", c), REGION_FILLER)
    if not part:
        return None
    words = set(re.findall(r"[a-z]+", c)) - REGION_FILLER - HERE_WORDS - PLACE_WORDS
    if len(words) > 3:  # more than a part's name: the language model reads it
        return None
    if has(c, r"\bholes?\b"):
        return {"op": "hole", "diameter_mm": length_mm(c) or 5.0, "axis": "height", "part": part}
    if has(c, r"\b(loop|keyring|key ?ring|keychain|key ?chain|eyelet|hook)\b"):
        return {"op": "loop", "diameter_mm": length_mm(c) or 5.0, "part": part}
    if has(c, r"\bsmooth\w*\b"):
        strength = 3 if has(c, r"\b(very|much|a lot|more|strong(?:ly)?|really)\b") else 1 if has(
            c, r"\b(slight(?:ly)?|a bit|bit|little|light(?:ly)?|gentl[ey])\b") else 2  # fmt: skip
        return {"op": "smooth", "strength": strength, "part": part}
    thinner = has(c, r"\b(thin|thinner|slimmer)\b")
    mm = length_mm(c) or (0.5 if thinner else 1.0)
    return {"op": "thicken", "mm": -mm if thinner else mm, "part": part}


def rule(clause: str) -> dict | None:
    """One clause as one operation, or None when no rule covers it."""
    c = clause
    if has(c, r"\bhollow\b"):
        m = re.search(NUM + UNIT + r"\s*(?:thick\s*)?walls?|walls?\s*(?:of\s*|thickness\s*(?:of\s*)?)?" + NUM + UNIT, c)
        if m:
            wall = number(m[1]) * unit_mm(m[2]) if m[1] else number(m[3]) * unit_mm(m[4])
        else:
            wall = length_mm(c) or 2.0
        return {"op": "hollow", "wall_mm": wall}
    if has(c, r"\b(loop|keyring|key ring|keychain|key chain|eyelet|hook)\b|\bring\b.*\bhang|\bhang"):
        return {"op": "loop", "diameter_mm": length_mm(c) or 5.0}
    if has(c, r"\bholes?\b"):
        if has(c, r"\b(front|back)\b"):
            axis = "width"
        elif has(c, r"\b(side|sides|sideways|horizontal(?:ly)?|left|right|end|ends)\b"):
            axis = "length"
        else:
            axis = "height"
        return {"op": "hole", "diameter_mm": length_mm(c) or 5.0, "axis": axis}
    if has(c, r"\bflat(?:ten)?\b") and has(c, r"\b(bottom|base|underside)\b"):
        return {"op": "cut", "side": "bottom", "mm": length_mm(c) or 1.0}
    if has(c, r"\b(base|stand|plinth|pedestal|platform)\b"):
        shape = "square" if has(c, r"\b(square|rectangular|rectangle|box)\b") else "round"
        return {"op": "base", "height_mm": length_mm(c) or 3.0, "shape": shape}
    if has(c, r"\b(cut|remove|trim|chop|slice|lop|shave)\b") and has(c, r"\b(bottom|top|tip|underside)\b"):
        mm = length_mm(c)
        if mm is None:
            return None
        return {"op": "cut", "side": "top" if has(c, r"\b(top|tip)\b") else "bottom", "mm": mm}
    if has(c, r"\bsmooth(?:er|en|ing)?\b"):
        strength = 3 if has(c, r"\b(very|much|a lot|more|strong(?:ly)?|really)\b") else 1 if has(
            c, r"\b(slight(?:ly)?|a bit|bit|little|light(?:ly)?|gentl[ey])\b") else 2  # fmt: skip
        return {"op": "smooth", "strength": strength}
    if has(c, r"\b(simplif\w*|low[- ]?poly|decimate|fewer (?:triangles|polygons|faces)|reduce (?:triangles|polygons|faces))\b"):
        return {"op": "simplify", "percent": percent(c) or 50.0}
    if has(c, r"\bupside[- ]down\b|\bflip\b.*\bover\b"):
        return {"op": "rotate", "axis": "length", "degrees": 180.0}
    if has(c, r"\b(mirror|mirrored)\b") or (has(c, r"\bflip\b") and not has(c, r"\b(rotate|turn)\b")):
        axis = "width" if has(c, r"\b(front|back)\b") else "height" if has(c, r"\b(vertical(?:ly)?|top|bottom)\b") else "length"
        return {"op": "mirror", "axis": axis}
    if has(c, r"\b(rotate|turn|spin|tilt|lay|lie|tip)\b"):
        deg = degrees(c)
        if deg is None:
            m = re.search(r"\b(rotate|turn|spin|tilt)\b[^0-9]*" + NUM + r"(?!\s*(?:mm|cm|%|x\b|times))", c)
            deg = number(m[2]) if m else 90.0
        if has(c, r"\b(on|onto)\b.*\bside\b|\bsideways\b|\blay\b|\blie\b|\btip(?:ped)? over\b"):
            axis = "length"
        elif has(c, r"\b(forward|forwards|backward|backwards|front|back)\b"):
            axis = "width"
        else:
            axis = "height"
            if has(c, r"\bclockwise\b") and not has(c, r"\b(counter|anti)[- ]?clockwise\b"):
                deg = -deg
        return {"op": "rotate", "axis": axis, "degrees": deg}
    return size_rule(c)


def size_rule(c: str) -> dict | None:
    """Sizes: "15 cm tall", "scale to 150 %", "20 % bigger", "twice as big", "20 mm taller", "3x bigger"."""
    words = re.findall(r"[a-z]+", c)
    comparative = next((w for w in words if w in COMPARE), None)
    if has(c, r"\b(twice|double|doubled)\b") and not comparative in ("smaller",):
        return {"op": "scale", "factor": 2.0}
    if has(c, r"\btriple\b"):
        return {"op": "scale", "factor": 3.0}
    if has(c, r"\bhalf\b") and has(c, r"\b(size|big|large|small)\b"):
        return {"op": "scale", "factor": 0.5}
    m = re.search(NUM + r"\s*(?:x|times)\s*(bigger|larger|smaller)\b", c)
    if m:
        f = number(m[1])
        if f <= 0:
            return None
        return {"op": "scale", "factor": f if m[2] != "smaller" else 1 / f}
    pct = percent(c)
    if pct is not None:
        if comparative:
            axis, sign = COMPARE[comparative]
            factor = 1 + sign * pct / 100
            return {"op": "scale", "factor": factor} if axis == "longest" else {"op": "stretch", "axis": axis, "factor": factor}
        if has(c, r"\b(scale|resize|size|make|set)\b"):
            return {"op": "scale", "factor": pct / 100}
        return None
    mm = length_mm(c)
    if mm is None:
        return None
    if comparative:
        axis, sign = COMPARE[comparative]
        return {"op": "grow", "axis": axis, "mm": sign * mm}
    if has(c, r"\b(tall|high|height|heigh)\b"):
        return {"op": "resize", "dimension": "height", "mm": mm}
    if has(c, r"\b(wide|width|thick|thickness|deep|depth)\b"):
        return {"op": "resize", "dimension": "width", "mm": mm}
    if has(c, r"\b(long|length)\b"):
        return {"op": "resize", "dimension": "length", "mm": mm}
    if has(c, r"\b(scale|resize|size|make|set|print|change|longest|biggest)\b"):
        return {"op": "resize", "dimension": "longest", "mm": mm}
    return None


# Every word the rules may meet in a clause they read. A clause with any other word ("make the ears bigger") is not
# theirs: it may be about a part of the model, which the rules cannot see, so the language model reads the request.
KNOWN = set("""
a an the it its this that model object thing print please pls thanks thank you can could would will i want like id to
d s ll me my make made making set change changing turn put give add adding create with and of by from in into on onto
at as so for up down off out over be is are should more less much very really bit little slightly slight lightly light
gently gentle strong strongly lot all whole entire size sized scale scaled resize resized
mm millimeter millimeters millimetre millimetres cm centimeter centimeters centimetre centimetres m meter meters metre
metres inch inches percent degree degrees deg x times
tall high height heigh wide width thick thickness deep depth long length longest biggest
taller higher shorter lower wider narrower thinner longer bigger larger smaller twice double doubled triple half big
large small
hole holes drill drilled through middle center centre straight vertical vertically horizontal horizontally side sides
sideways left right front back end ends top bottom underside tip
hollow empty inside walls wall
loop keyring key ring keychain chain eyelet hook hang hanging string
base stand plinth pedestal platform round square rectangular rectangle box
cut remove trim chop slice lop shave flat flatten
smooth smoother smoothen smoothing simplify simplified simplification low poly lowpoly decimate fewer reduce triangles
polygons faces
rotate rotated rotation spin tilt lay lie tip tipped forward forwards backward backwards clockwise counter anti
counterclockwise anticlockwise around about axis upside flip flipped mirror mirrored
""".split())  # fmt: skip


def covered(clause: str) -> bool:
    return all(w in KNOWN for w in re.findall(r"[a-z]+", clause))


SPLIT = re.compile(r"(?<!\d)[,;.](?!\d)|\s(?:and then|then|and|also|plus|after that)\s")


def parse(prompt: str) -> list[dict] | None:
    """The request as operations, if fixed rules cover every part of it; None otherwise (the language model reads it).
    Each operation is validated (EditError when a value is out of its limits)."""
    text = prompt.lower().replace("’", "'").replace("×", "x")
    ops = []
    for part in SPLIT.split(text):
        if not part or not part.strip():
            continue
        if not re.sub(r"\s+", "", FILLER.sub(" ", part)).strip(".!?"):
            continue  # "please", "thanks"
        op = colour_rule(part.strip()) or region_rule(part.strip()) or (rule(part.strip()) if covered(part) else None)
        if op is None:
            return None
        ops.append(validate(op))
    return ops or None


# --- Doing the operations -----------------------------------------------------------------------------------------


class Model:
    """The mesh being changed, at real size in mm (z up, standing on z = 0, centred), and the affine change so far of
    the original surface (to colour the result from the original at the end)."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.T = np.eye(4)

    def transform(self, matrix: np.ndarray):
        self.mesh.apply_transform(matrix)
        self.T = matrix @ self.T

    def seat(self):
        lo, hi = self.mesh.bounds
        self.transform(trimesh.transformations.translation_matrix([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]))

    def axis(self, name: str) -> int:
        """The mesh axis of the app's length, width or height."""
        if name == "height":
            return 2
        ex = self.mesh.extents
        longer = 0 if ex[0] >= ex[1] else 1
        return longer if name == "length" else 1 - longer

    def size(self, name: str) -> float:
        return float(self.mesh.extents.max()) if name == "longest" else float(self.mesh.extents[self.axis(name)])


def boolean(kind: str, a: trimesh.Trimesh, b: trimesh.Trimesh, what: str, note: Callable[[str], None]) -> trimesh.Trimesh:
    """Union or difference by manifold3d. A union of a mesh with openings falls back to the two overlapping (slicers
    print overlapping parts as one); a difference cannot."""
    try:
        if not a.is_volume:
            raise ValueError("not a closed solid")
        out = (trimesh.boolean.union if kind == "union" else trimesh.boolean.difference)([a, b], engine="manifold")
        if out.is_empty or not len(out.faces):
            raise ValueError("empty result")
        return out
    except Exception as e:
        if kind == "union":
            note(f"edit: {what}: joined as overlapping parts ({e})")
            return trimesh.util.concatenate([a, b])
        raise EditError(f"„{what}“ не може да се направи: моделът не е затворено тяло (има отвори).")


def axis_cylinder(radius: float, length: float, axis: int, centre: np.ndarray) -> trimesh.Trimesh:
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=96)  # along z
    if axis == 0:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 1:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation(centre)
    return cyl


def ring(major: float, minor: float) -> trimesh.Trimesh:
    """A torus around the z axis (its tube's circle turned round)."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    return M.revolve(np.c_[major + minor * np.cos(t), minor * np.sin(t)])


def hollowed(mesh: trimesh.Trimesh, wall: float) -> trimesh.Trimesh:
    """The mesh with a closed cavity at least WALL mm inside its surface everywhere, turned to face into the cavity.

    The filled voxels' distance inward is only a first guess: it counts from the voxels on the surface, not the surface,
    and left walls ~0.5 mm thin, near 0 in tight places. So the cavity's edge voxels are measured against the real
    surface (dense samples of it) and peeled off while closer than the wall, then the result is measured once more."""
    from skimage import measure

    if not mesh.is_watertight:
        raise EditError("„Кух“ не може да се направи: моделът не е затворено тяло (има отвори).")
    thin = EditError(f"Моделът е твърде тънък за стени от {wall:g} мм.")
    if wall * 2.2 >= float(mesh.extents.min()):
        raise thin
    pitch = max(wall / 4, float(mesh.extents.max()) / 240)
    vox = mesh.voxelized(pitch).fill()
    inside = ndimage.distance_transform_edt(np.pad(vox.matrix, 1)) * pitch > wall
    spacing = pitch / 3
    samples, _ = trimesh.sample.sample_surface(mesh, int(min(4e6, mesh.area / spacing**2)), seed=0)
    surface = cKDTree(samples)  # a sample's distance is at most `spacing` more than the surface's

    def points(idx: np.ndarray) -> np.ndarray:  # padded voxel indices -> model coordinates
        return trimesh.transform_points(idx.astype(np.float64) - 1, vox.transform)

    # The cavity's surface lies half a voxel outside its edge voxels' centres.
    need = wall + 0.5 * pitch + spacing
    for _ in range(80):  # a layer of voxels per round; thin, detailed figures need a few dozen
        idx = np.argwhere(inside & ~ndimage.binary_erosion(inside))
        if not len(idx):
            break
        d, _ = surface.query(points(idx), workers=-1)
        close = d < need
        if not close.any():
            break
        inside[tuple(idx[close].T)] = False
    if inside.sum() < 27:
        raise thin
    verts, faces, _, _ = measure.marching_cubes(inside.astype(np.float32), 0.5)
    cavity = trimesh.Trimesh(trimesh.transform_points(verts - 1, vox.transform), faces, process=True)
    d, _ = surface.query(cavity.vertices, workers=-1)
    if d.min() - spacing < 0.95 * wall:  # never expected after the peeling; a cavity too near the surface is not made
        raise EditError(f"Стени от {wall:g} мм не могат да се направят навсякъде в този модел.")
    # Thin parts split the hollow into pockets: each turned to face into itself; specks of a few voxels left solid.
    pockets = []
    for part in cavity.split(only_watertight=False):
        trimesh.repair.fix_normals(part)
        if abs(part.volume) < (3 * pitch) ** 3:
            continue
        if part.volume > 0:
            part.invert()  # its faces look into the hollow, away from the plastic
        pockets.append(part)
    if not pockets:
        raise thin
    return trimesh.util.concatenate([mesh, *pockets])


def apply(model: Model, op: dict, note: Callable[[str], None] = lambda text: None, regions: "Regions | None" = None):
    m, n = model.mesh, op["op"]
    if op.get("part") and n in ("smooth", "thicken", "hole", "loop"):
        if regions is None:
            raise EditError("Тази промяна иска място на модела.")
        local(model, op, regions, note)
    elif n in ("resize", "scale") or (n == "grow" and op["axis"] == "longest"):
        if n == "resize":
            k = op["mm"] / model.size(op["dimension"])
        elif n == "scale":
            k = op["factor"]
        else:
            k = (model.size("longest") + op["mm"]) / model.size("longest")
        if not 1 <= model.size("longest") * k <= 2000:
            raise EditError("Новият размер е извън границите (1–2000 мм).")
        model.transform(trimesh.transformations.scale_matrix(k))
    elif n in ("grow", "stretch"):
        i, now = model.axis(op["axis"]), model.size(op["axis"])
        k = (now + op["mm"]) / now if n == "grow" else op["factor"]
        if not 1 <= now * k <= 2000:
            raise EditError(f"{NAMES[op['axis']].capitalize()} трябва да остане между 1 и 2000 мм.")
        S = np.eye(4)
        S[i, i] = k
        model.transform(S)
    elif n == "rotate":
        vec = np.eye(3)[model.axis(op["axis"])]
        model.transform(trimesh.transformations.rotation_matrix(np.radians(op["degrees"]), vec, m.bounds.mean(0)))
    elif n == "mirror":
        S = np.eye(4)
        S[model.axis(op["axis"]), model.axis(op["axis"])] = -1
        model.transform(S)
        if model.mesh.is_watertight and model.mesh.volume < 0:
            model.mesh.invert()
    elif n == "hollow":
        model.mesh = hollowed(m, op["wall_mm"])
    elif n == "hole":
        i = model.axis(op["axis"])
        across = float(np.delete(m.extents, i).min())
        if op["diameter_mm"] >= 0.9 * across:
            raise EditError(f"Отвор Ø{op['diameter_mm']:g} мм е твърде голям за модел, широк {across:.0f} мм.")
        cyl = axis_cylinder(op["diameter_mm"] / 2, float(m.extents[i]) * 3, i, m.bounds.mean(0))
        model.mesh = boolean("difference", m, cyl, "Отвор", note)
    elif n == "cut":
        h = float(m.extents[2])
        if op["mm"] >= 0.95 * h:
            raise EditError(f"Не може да се отрежат {op['mm']:g} мм от модел, висок {h:.0f} мм.")
        z = op["mm"] if op["side"] == "bottom" else h - op["mm"]
        normal = [0, 0, 1] if op["side"] == "bottom" else [0, 0, -1]
        model.mesh = m.slice_plane([0, 0, z], normal, cap=True)
    elif n == "base":
        bottom = m.vertices[m.vertices[:, 2] <= m.bounds[0, 2] + 0.03 * m.extents[2]][:, :2]
        lo, hi = bottom.min(0), bottom.max(0)
        centre = (lo + hi) / 2
        margin = 0.08 * float(np.ptp(bottom, 0).max()) + 2.0
        h, overlap = op["height_mm"], min(0.4, op["height_mm"] / 4)
        if op["shape"] == "round":
            r = float(np.linalg.norm(bottom - centre, axis=1).max()) + margin
            stand = trimesh.creation.cylinder(radius=r, height=h + overlap, sections=128)
        else:
            stand = trimesh.creation.box(extents=[*(hi - lo + 2 * margin), h + overlap])
        stand.apply_translation([centre[0], centre[1], m.bounds[0, 2] - h + (h + overlap) / 2])
        model.mesh = boolean("union", m, stand, "Основа", note)
    elif n == "loop":
        d = op["diameter_mm"]
        tube = max(1.2, 0.3 * d)
        top = float(m.bounds[1, 2])
        near_top = m.vertices[m.vertices[:, 2] >= top - 0.02 * m.extents[2]]
        at = near_top[np.argmin(np.linalg.norm(near_top[:, :2] - near_top[:, :2].mean(0), axis=1))]
        hoop = ring(d / 2 + tube, tube)
        hoop.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # standing up
        hoop.apply_translation([at[0], at[1], top + d / 2 + tube])  # its lowest point a tube's width into the top
        model.mesh = boolean("union", m, hoop, "Халка", note)
    elif n == "smooth":
        model.mesh = M.smooth(m, op["strength"])
    elif n == "thicken":
        model.mesh = thickened(m, np.full(len(m.vertices), op["mm"]))
    elif n == "simplify":
        target = max(500, int(len(m.faces) * op["percent"] / 100))
        model.mesh = M.decimate(m, target, keep_closed=m.is_watertight)
    model.seat()  # every change ends standing on z = 0, centred (smoothing moves points near the base too)


# --- Changes to one part of the model -----------------------------------------------------------------------------

HERE_RADIUS = 0.12  # (share of the model's longest side) the tapped spot's area, for smoothing and thickening
# (a part) Light: Taubin's filter, which takes off fine ripples and keeps the shape (it does not shrink). Medium and
# strong: Laplacian smoothing, which also flattens lumps a few mm wide (a figure's hair strands: Taubin moved them at
# most 0.3 mm, 100 Laplacian rounds 0.8 mm), and sinks the area a little (0.27 mm).
SMOOTH_FILTERS = {1: ("taubin", 40), 2: ("laplacian", 30), 3: ("laplacian", 100)}


def thickened(m: trimesh.Trimesh, mm: np.ndarray) -> trimesh.Trimesh:
    """Each point pushed out along the surface's normal by its mm (in, if negative); the flat base stays on the table."""
    floor = m.bounds[0, 2]
    v = m.vertices + m.vertex_normals * mm[:, None]
    base = m.vertices[:, 2] <= floor + 1e-6 * m.extents.max()
    v[base, 2] = m.vertices[base, 2]
    out = m.copy()
    out.vertices = v
    return out


def local(model: Model, op: dict, regions: "Regions", note: Callable[[str], None]):
    """smooth, thicken, hole and loop on a part of the model or at the tapped spot."""
    m, n = model.mesh, op["op"]
    if n == "smooth":
        w = regions.weights(model, op["part"])
        smooth = m.copy()
        kind, rounds = SMOOTH_FILTERS[op["strength"]]
        if kind == "taubin":
            trimesh.smoothing.filter_taubin(smooth, lamb=0.5, nu=0.53, iterations=rounds)
        else:  # no volume constraint: it would rescale the whole model for a change to a part of it
            trimesh.smoothing.filter_laplacian(smooth, lamb=0.5, iterations=rounds, volume_constraint=False)
        v = m.vertices + w[:, None] * (smooth.vertices - m.vertices)
        v[:, 2] = np.maximum(v[:, 2], m.bounds[0, 2])
        out = m.copy()
        out.vertices = v
        model.mesh = out
    elif n == "thicken":
        model.mesh = thickened(m, op["mm"] * regions.weights(model, op["part"], tapped_part=True))
    elif n == "hole":
        d = op["diameter_mm"]
        if d >= 0.9 * float(m.extents.min()):
            raise EditError(f"Отвор Ø{d:g} мм е твърде голям за този модел.")
        at, normal = regions.place(model, op["part"], top=False)
        cyl = trimesh.creation.cylinder(radius=d / 2, height=3 * float(m.extents.max()), sections=96)
        cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], normal))
        cyl.apply_translation(at)
        model.mesh = boolean("difference", m, cyl, "Отвор", note)
    elif n == "loop":
        d = op["diameter_mm"]
        tube = max(1.2, 0.3 * d)
        at, normal = regions.place(model, op["part"], top=True)
        across = np.cross(normal, [0.0, 0.0, 1.0])  # the ring's axis: across the normal, so the ring stands out of it
        if np.linalg.norm(across) < 1e-3:
            across = np.array([1.0, 0.0, 0.0])
        hoop = ring(d / 2 + tube, tube)
        hoop.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], across / np.linalg.norm(across)))
        hoop.apply_translation(at + normal * (d / 2 + tube))  # its nearest point a tube's width into the surface
        model.mesh = boolean("union", m, hoop, "Халка", note)


class Regions:
    """Where part-wise changes act, found once on the model as it was before any change (in the coordinates where
    model.T is the identity), then followed through the changes by nearest point:
    - HERE: the spot the user tapped in the app: a colour or a thickening takes the part there (recolor.tapped:
      MobileSAM cuts out what is under the tap), smoothing fades out over HERE_RADIUS around it;
    - a part's name: found by CLIPSeg on pictures of the model drawn from its photo texture (app/recolor.py)."""

    def __init__(self, job: Path, mesh: trimesh.Trimesh, parts: list[dict] | None, spot: np.ndarray | None,
                 sh: Callable[[str, list], None]):  # fmt: skip
        self.job, self.sh, self.spot = job, sh, spot
        self.base = np.asarray(mesh.vertices, np.float64).copy()
        self.size = float(mesh.extents.max())
        self.found: dict[str, list[np.ndarray]] = {}
        self.atlases = [np.asarray(Image.open(io.BytesIO(p["jpeg"])).convert("RGB")) for p in parts or []]
        self.texels = [RC.texels(p, a) for p, a in zip(parts or [], self.atlases)]
        if mesh.visual.kind == "vertex":
            self.vcols = np.asarray(mesh.visual.vertex_colors[:, :3], np.uint8).copy()
        else:
            self.vcols = np.full((len(mesh.vertices), 3), 170, np.uint8)  # a grey model: one colour all over
        self.vnrm = np.asarray(mesh.vertex_normals, np.float32)
        self.mesh = mesh.copy()

    def queries(self, atlases: list[np.ndarray] | None = None, vcols: np.ndarray | None = None) -> list:
        """The texels of each texture image and, last, the vertices: (points, colours, normals) each."""
        atlases = self.atlases if atlases is None else atlases
        out = [(pts, a[where], nrm) for (where, pts, nrm), a in zip(self.texels, atlases)]
        return out + [(self.base.astype(np.float32), self.vcols if vcols is None else vcols, self.vnrm)]

    def draw_from(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Surface samples (points, colours, normals) the AI's pictures of the model are drawn from: its texels where
        it has a photo texture (the photos' detail), else points on its faces in their vertex colours."""
        queries = self.queries()
        if self.texels:
            allq = [np.concatenate(x) for x in zip(*queries[:-1])]
            pick = np.random.default_rng(0).choice(len(allq[0]), min(len(allq[0]), RC.RENDER_POINTS), replace=False)
            return allq[0][pick], allq[1][pick].astype(np.float64), allq[2][pick]
        pts, face = trimesh.sample.sample_surface(self.mesh, RC.RENDER_POINTS, seed=0)
        cols = self.vcols[self.mesh.faces[face]].mean(1).astype(np.float64)
        return pts, cols, self.mesh.face_normals[face]

    def tapped(self) -> list[np.ndarray]:
        """The part under the tapped spot (MobileSAM), per query set (the last: the vertices)."""
        if HERE not in self.found:
            _, _, tri = self.mesh.nearest.on_surface(self.spot[None])
            try:
                self.found[HERE] = RC.tapped(self.job / "work" / "recolor", self.draw_from(), self.queries(), self.spot,
                                             self.mesh.face_normals[tri[0]], self.sh, lambda text: P.note(self.job, text))  # fmt: skip
            except RC.NotFound:
                raise EditError("Не намерих частта на посоченото място. Посочете я отново, по-навътре в нея.")
        return self.found[HERE]

    def part(self, name: str) -> list[np.ndarray]:
        """CLIPSeg's weights for a named part, per query set (the last: the vertices)."""
        if name not in self.found:
            if not RC.ready():
                raise EditError("Промените на част от модела още не са включени на сървъра. Опитайте по-късно.")
            try:
                self.found[name] = RC.find(self.job / "work" / "recolor", name, self.draw_from(), self.queries(),
                                           self.sh, lambda text: P.note(self.job, text))  # fmt: skip
            except RC.NotFound:
                raise EditError(f"Не намерих „{name}“ на модела. Назовете частта с една-две думи на английски, както се "
                                "вижда на модела (например „hair“, „shirt“, „base“), или посочете мястото на модела.")  # fmt: skip
        return self.found[name]

    def base_weights(self, name: str, tapped_part: bool = False) -> np.ndarray:
        """Weights 0-1 over the vertices as they were. tapped_part: HERE is the part under the tap, not the area."""
        if not name:
            return np.ones(len(self.base))
        if name == HERE and tapped_part:
            return self.tapped()[-1]
        if name == HERE:
            d = np.linalg.norm(self.base - self.spot, axis=1) / (HERE_RADIUS * self.size)
            w = np.clip(1 - d, 0, 1)
            return w * w * (3 - 2 * w)  # down to 0 at the edge smoothly: no step in the surface
        return self.part(name)[-1]

    def weights(self, model: Model, name: str, tapped_part: bool = False) -> np.ndarray:
        """Weights 0-1 over the model's vertices now (moved, smoothed, holed...): each takes its nearest old one's."""
        w = self.base_weights(name, tapped_part)
        _, idx = cKDTree(trimesh.transform_points(self.base, model.T)).query(model.mesh.vertices, workers=-1)
        return w[idx]

    def place(self, model: Model, name: str, top: bool) -> tuple[np.ndarray, np.ndarray]:
        """Where a hole or loop goes, on the model's surface now, and the surface's outward normal there: the tapped
        spot, or a named part's top (a loop, to hang it by) or its point nearest its middle (a hole)."""
        if name == HERE:
            p0 = self.spot
        else:
            w = self.base_weights(name)
            sel = np.flatnonzero(w > 0.5)
            if not len(sel):
                raise EditError(f"Не намерих „{name}“ на модела.")
            pts = self.base[sel]
            if top:
                p0 = pts[np.argmax(pts[:, 2])]
            else:
                mid = (pts * w[sel, None]).sum(0) / w[sel].sum()
                p0 = pts[np.argmin(np.linalg.norm(pts - mid, axis=1))]
        p = trimesh.transform_points(np.asarray(p0, np.float64)[None], model.T)
        at, _, tri = model.mesh.nearest.on_surface(p)
        return at[0], model.mesh.face_normals[tri[0]]


def spot_point(mesh: trimesh.Trimesh, spot: list | None) -> np.ndarray | None:
    """The point the user tapped in the app, on the model's surface. The app sends it as shares (0-1) of the model's
    bounding box in the axes of the GLB it shows (glTF: y up; mesh.export turns z up into y up, y into -z)."""
    if not spot:
        return None
    u, v, w = (float(x) for x in spot)
    lo, hi = mesh.bounds
    ex = hi - lo
    p = np.array([lo[0] + u * ex[0], hi[1] - w * ex[1], lo[2] + v * ex[2]])  # glTF x, y, z = x, z, -y here
    at, _, _ = mesh.nearest.on_surface(p[None])
    return at[0]


def coloured(out: trimesh.Trimesh, original: trimesh.Trimesh, T: np.ndarray) -> trimesh.Trimesh:
    """Each vertex takes the colour of the nearest point of the original surface, moved as the model was."""
    if original.visual.kind != "vertex":
        return out
    _, idx = cKDTree(trimesh.transform_points(original.vertices, T)).query(out.vertices)
    out.visual.vertex_colors = original.visual.vertex_colors[idx]
    return out


# --- The job --------------------------------------------------------------------------------------------------------


def understand(job: Path, prompt: str, sizes: dict, sh: Callable[[str, list], None],
               tapped: bool = False) -> tuple[list[dict], bool]:  # fmt: skip
    """The operations, and whether part of the request was left out (the language model said so). tapped: the user
    marked a spot on the model."""
    ops = parse(prompt)
    if ops is not None:
        P.note(job, f"edit: read by rules: {ops}")
        return ops, False
    from . import edit_llm

    if not edit_llm.ready():
        raise EditError("Не разбрах промяната. Опишете я по-просто, например: „make it 15 cm tall“, "
                        "„add a 5 mm hole through the top“, „add a keychain loop“.")  # fmt: skip
    work = job / "work"
    work.mkdir(parents=True, exist_ok=True)
    request, answer = work / "edit_request.json", work / "edit_answer.json"
    request.write_text(json.dumps({"prompt": prompt, "sizes": sizes, "tapped": tapped}))
    sh("understanding the request", [sys.executable, edit_llm.__file__, request, answer])
    got = json.loads(answer.read_text())
    P.note(job, f"edit: language model: {got}")
    ops, problems = [], []
    for raw in got.get("ops") or []:
        try:
            ops.append(validate(raw if isinstance(raw, dict) else {}))
        except EditError as e:
            problems.append(str(e))
    if problems:
        raise EditError("Промяната не може да се направи: " + " ".join(problems))
    if not ops:
        raise EditError("Не разбрах промяната или тя не може да се направи. Опишете я по-просто, например: "
                        "„make it 15 cm tall“, „add a 5 mm hole through the top“, „hollow it with 2 mm walls“.")  # fmt: skip
    return ops, bool(str(got.get("unsupported") or "").strip())


def paint(regions: Regions, ops: list[dict]) -> list[np.ndarray]:
    """The colour operations on the model's vertex colours (regions.vcols, in place) and on the photo texture's
    images; returns the new images (none without a texture)."""
    atlases = [a.copy() for a in regions.atlases]
    vcols = regions.vcols
    for op in ops:
        rgb = RC.colour_rgb(op["color"])
        if op["part"] == HERE:  # the part under the tapped spot
            weights = regions.tapped()
        elif op["part"]:
            weights = regions.part(op["part"])
        else:
            weights = [np.ones(len(q[0])) for q in regions.queries()]
        for k, ((where, _, _), w) in enumerate(zip(regions.texels, weights)):
            atlases[k] = RC.recolour_atlas(atlases[k], where, w, rgb)
        vcols = RC.recoloured(vcols, weights[-1], rgb)
    regions.vcols = vcols
    return atlases


def save_texture(job: Path, parts: list[dict], atlases: list[np.ndarray] | None, vertices: np.ndarray,
                 normals: np.ndarray, links: list[np.ndarray], flip: bool):  # fmt: skip
    """The photo texture on the changed surface: each texture vertex at its model vertex's new place (links: the
    model vertex of each; the changes kept the model's vertices, in order), with the new images if colours changed.
    flip: mirrored an odd number of times (the faces turn the other way round)."""
    arrays = {}
    for k, (p, link) in enumerate(zip(parts, links)):
        faces = np.asarray(p["faces"])
        arrays |= {
            f"vertices_{k}": vertices[link].astype(np.float32),
            f"normals_{k}": normals[link].astype(np.float32),
            f"uv_{k}": np.asarray(p["uv"], np.float32),
            f"faces_{k}": (faces[:, ::-1] if flip else faces).astype(np.uint32),
        }
        if atlases is not None:
            Image.fromarray(atlases[k]).save(job / f"texture_{k}.jpg", quality=92)
    np.savez_compressed(job / TX.NPZ, count=len(parts), **arrays)


# --- A name plate: a stand under the model with the name in raised letters, the way the model looks ------------------

TEXT_FONTS = ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")  # bold, with Cyrillic; else Pillow's own
MIN_TEXT = 2.0  # mm: smaller letters do not print legibly
PLATE_COLOUR, LETTER_COLOUR = (31, 33, 38), (242, 242, 244)
Z_UP = np.array([0.0, 0.0, 1.0])


def text_font(px: int):
    from PIL import ImageFont

    for name in TEXT_FONTS:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default(px)


def text_outline(text: str) -> tuple[list[np.ndarray], np.ndarray]:
    """The letters' outlines (closed polygons: outsides and the holes of o, a, e...), in units of a capital letter's
    height, centred on (0, 0); and their (width, height) in those units."""
    from PIL import ImageDraw
    from skimage import measure

    font = text_font(160)
    l, t, r, b = font.getbbox(text)
    cap = font.getbbox("H")
    cap_h = float(cap[3] - cap[1])
    pad = 8
    img = Image.new("L", (int(r - l) + 2 * pad, int(b - t) + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=255)
    a = np.asarray(img, np.float32) / 255
    polys = []
    for c in measure.find_contours(np.pad(a, 1), 0.5):
        xy = np.c_[c[:, 1], -c[:, 0]] / cap_h
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if len(xy) >= 3:
            polys.append(xy)
    if not polys:
        raise EditError(f"Надписът „{text}“ няма букви, които мога да изпиша.")
    allp = np.concatenate(polys)
    mid = (allp.min(0) + allp.max(0)) / 2
    return [p - mid for p in polys], allp.max(0) - allp.min(0)


def text_solid(polys: list[np.ndarray], mm: float, z_lo: float, z_hi: float, drape: Callable[[np.ndarray], np.ndarray],
               fine: float | None) -> trimesh.Trimesh:  # fmt: skip
    """The letters `mm` high as a solid from z_lo to z_hi (z: out of the surface), laid on the stand by `drape`;
    fine: divided this small first, to follow a curve."""
    import manifold3d as mf

    cs = mf.CrossSection([np.asarray(p * mm, np.float64) for p in polys], mf.FillRule.EvenOdd).simplify(0.01)
    solid = mf.Manifold.extrude(cs, z_hi - z_lo).translate((0.0, 0.0, z_lo))
    if fine:
        solid = solid.refine_to_length(fine)
    solid = solid.warp_batch(lambda P: np.asarray(drape(np.asarray(P, np.float64)), np.float64))
    mesh = solid.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def nameplate(m: trimesh.Trimesh, text: str, shape: str, front: np.ndarray,
              note: Callable[[str], None]) -> tuple[trimesh.Trimesh, np.ndarray]:  # fmt: skip
    """The stand under the model (in its coordinates, mm) with `text` facing `front`. Returns it and which of its
    faces are the letters."""
    n = np.array([front[0], front[1], 0.0])
    n /= max(np.linalg.norm(n), 1e-9)
    right = np.cross(Z_UP, n)
    polys, (w1, h1) = text_outline(text)
    floor, height = float(m.bounds[0, 2]), float(m.extents[2])
    bottom = m.vertices[m.vertices[:, 2] <= floor + 0.03 * height]
    a, b = bottom @ right, bottom @ n
    centre_a, centre_b = (a.min() + a.max()) / 2, (b.min() + b.max()) / 2
    margin = 0.08 * max(float(np.ptp(a)), float(np.ptp(b))) + 2.0
    stand_h = float(np.clip(0.1 * m.extents.max(), 6.0, 24.0))
    mm = min(0.6 * stand_h, 0.8 * stand_h / h1)
    overlap = min(0.4, stand_h / 4)
    zmid = floor - stand_h / 2
    middle = centre_a * right + centre_b * n + Z_UP * zmid
    relief = float(np.clip(0.15 * mm, 0.6, 1.5))
    inset = 0.8
    if shape == "round":
        radius = float(np.hypot(a - centre_a, b - centre_b).max()) + margin
        mm = min(mm, 2.4 * radius / w1)  # about 140 degrees round: what is seen from the front
        stand = trimesh.creation.cylinder(radius=radius, height=stand_h + overlap, sections=128)

        def drape(P: np.ndarray) -> np.ndarray:
            angle, out = P[:, 0] / radius, radius + P[:, 2]
            return middle + (out * np.sin(angle))[:, None] * right + (out * np.cos(angle))[:, None] * n + P[:, 1:2] * Z_UP

        fine = max(0.3, mm / 10)
    else:
        pad = max(2.0, 0.5 * mm)
        width = max(float(np.ptp(a)) + 2 * margin, w1 * mm + 2 * pad)
        depth = float(np.ptp(b)) + 2 * margin
        radius = depth / 2
        stand = trimesh.creation.box(extents=[width, depth, stand_h + overlap])
        turn = np.eye(4)
        turn[:3, :3] = np.c_[right, -n, Z_UP]  # the box is the same both ways: only a turn, never a mirror
        stand.apply_transform(turn)

        def drape(P: np.ndarray) -> np.ndarray:
            return middle + P[:, :1] * right + (radius + P[:, 2])[:, None] * n + P[:, 1:2] * Z_UP

        fine = None
    if mm < MIN_TEXT:
        raise EditError(f"Надписът „{text}“ не се побира (буквите стават под {MIN_TEXT:g} мм). Съкратете го.")
    stand.apply_translation(middle + Z_UP * (overlap / 2))
    letters = text_solid(polys, mm, -inset, relief, drape, fine)
    plate = trimesh.boolean.union([stand, letters], engine="manifold")
    rel = plate.triangles_center - middle
    out = np.hypot(rel @ right, rel @ n) if shape == "round" else rel @ n
    raised = out > radius + 0.02
    note(f"edit: name plate „{text}“ {mm:.1f} mm letters on a {shape} stand {stand_h:.1f} mm high, facing "
         f"{np.round(n, 2).tolist()}, {len(plate.faces)} faces")  # fmt: skip
    return plate, raised


def plate_part(plate: trimesh.Trimesh, raised: np.ndarray, k: int, job: Path) -> dict:
    """The plate as one more texture part (its own small two-colour image), so the model above keeps its photos."""
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, :4], image[:, 4:] = PLATE_COLOUR, LETTER_COLOUR
    Image.fromarray(image).save(job / f"texture_{k}.jpg", quality=95)
    n = len(plate.faces)
    uv = np.repeat(np.where(raised, 0.75, 0.25), 3)
    return {f"vertices_{k}": plate.triangles.reshape(-1, 3).astype(np.float32),
            f"normals_{k}": np.repeat(plate.face_normals, 3, axis=0).astype(np.float32),
            f"uv_{k}": np.c_[uv, np.full(3 * n, 0.5)].astype(np.float32),
            f"faces_{k}": np.arange(3 * n).reshape(-1, 3).astype(np.uint32)}  # fmt: skip


def front_of(job: Path, mesh: trimesh.Trimesh) -> np.ndarray:
    """Which way the model looks: towards the side that shows the most (a face, a logo)."""
    from . import render

    pts, col, _ = render.surface(job, mesh, 400_000)
    return render.view_turn(render.best_side(pts, col)).T @ np.array([0.0, -1.0, 0.0])


def run_nameplate(job: Path, params: dict, set_state: Callable[[str], None]) -> dict:
    wanted = params["nameplate"]
    text = " ".join(str(wanted["text"]).split())
    longest_mm = float(params.get("longest_mm") or 100.0)
    source = trimesh.load(job / SOURCE, force="mesh")
    parts = TX.load(job)
    set_state("cleaning")
    front = front_of(job, source)
    model = Model(source)
    model.transform(trimesh.transformations.scale_matrix(longest_mm / float(source.extents.max())))
    model.seat()
    plate, raised = nameplate(model.mesh, text, wanted["shape"], front, lambda t: P.note(job, t))
    original = model.mesh.copy()
    model.mesh = boolean("union", model.mesh, plate, "Табелка", lambda t: P.note(job, t))
    before = model.T.copy()
    model.seat()
    plate.apply_transform(model.T @ np.linalg.inv(before))
    trimesh.repair.fix_normals(model.mesh)
    out = coloured(model.mesh, original, model.T @ np.linalg.inv(before))
    real = float(out.extents.max())
    shrink = trimesh.transformations.scale_matrix(100.0 / real)
    out.apply_transform(shrink)
    if parts:  # the model's photos as they were, moved with it; the plate as a part of its own
        plate.apply_transform(shrink)
        to = shrink @ model.T
        arrays = {}
        for k, p in enumerate(parts):
            arrays |= {f"vertices_{k}": trimesh.transform_points(p["vertices"], to).astype(np.float32),
                       f"normals_{k}": np.asarray(p["normals"], np.float32),
                       f"uv_{k}": np.asarray(p["uv"], np.float32),
                       f"faces_{k}": np.asarray(p["faces"], np.uint32)}  # fmt: skip
        arrays |= plate_part(plate, raised, len(parts), job)
        np.savez_compressed(job / TX.NPZ, count=len(parts) + 1, **arrays)
    changes = [f"Табелка с надпис „{text}“"]
    return P.finish(job, out) | {"real_mm": round(real, 1), "edit": {"prompt": text, "changes": changes,
                                                                     "source": params.get("source")}}  # fmt: skip


def run(job: Path, params: dict, set_state: Callable[[str], None], sh: Callable[[str, list], None]) -> dict:
    if params.get("nameplate"):
        return run_nameplate(job, params, set_state)
    prompt = str(params["prompt"])
    longest_mm = float(params.get("longest_mm") or 100.0)
    source = trimesh.load(job / SOURCE, force="mesh")
    parts = TX.load(job)  # the photo texture, copied with the model (copy_source); None if it has none
    model = Model(source)
    model.transform(trimesh.transformations.scale_matrix(longest_mm / float(source.extents.max())))
    model.seat()
    if parts:  # in the same coordinates, and tied to the model's vertices (the texture's are the same points)
        tree = cKDTree(model.mesh.vertices)
        for p in parts:
            p["vertices"] = trimesh.transform_points(p["vertices"], model.T)
        links = [tree.query(p["vertices"])[1] for p in parts]
    model.T = np.eye(4)
    n_vertices, n_faces = len(model.mesh.vertices), len(model.mesh.faces)
    sizes = {name: round(model.size(name), 1) for name in AXES}
    spot = spot_point(model.mesh, params.get("spot"))

    set_state("reconstructing")
    ops, partial = understand(job, prompt, sizes, sh, spot is not None)
    if spot is None and any(op.get("part") == HERE for op in ops):
        raise EditError("Посочете мястото на модела (бутона „Посочи място“) или назовете частта, например „the hair“.")
    set_state("cleaning")
    colours = [op for op in ops if op["op"] == "color"]
    regions = Regions(job, model.mesh, parts, spot, sh) if colours or any(op.get("part") for op in ops) else None
    atlases = None
    if colours:
        atlases = paint(regions, colours)
        rgba = np.full((n_vertices, 4), 255, np.uint8)
        rgba[:, :3] = regions.vcols
        model.mesh.visual = trimesh.visual.ColorVisuals(model.mesh, vertex_colors=rgba)
    original = model.mesh.copy()  # coloured, where the shape changes start from
    mirrors = 0
    for op in ops:
        if op["op"] != "color":
            apply(model, op, lambda text: P.note(job, text), regions)
            mirrors += op["op"] == "mirror"
    # The texture follows when the model's own vertices are all still there, in order (moved, smoothed, thickened,
    # or with a hollow's inside after them).
    keep = bool(parts) and all(op["op"] in KEEPS_TEXTURE for op in ops) and len(model.mesh.vertices) >= n_vertices
    if keep:
        outside = trimesh.Trimesh(model.mesh.vertices[:n_vertices], model.mesh.faces[:n_faces], process=False)
        surface, normals = outside.vertices.copy(), outside.vertex_normals.copy()
    out = model.mesh
    if out.is_empty or not len(out.faces):
        raise EditError("След промените не остана модел.")
    if len(out.faces) > M.MAX_FACES:
        out = M.decimate(out, M.MAX_FACES, keep_closed=out.is_watertight)
    trimesh.repair.fix_normals(out)
    out = coloured(out, original, model.T)
    real = float(out.extents.max())
    out.apply_scale(100.0 / real)  # stored at a 100 mm longest side, as every model (the app scales to real_mm)
    if keep:
        save_texture(job, parts, atlases, surface * (100.0 / real), normals, links, mirrors % 2 == 1)
    else:
        TX.clear(job)  # the shape changed under it: the vertex colours only
    changes = [describe(op) for op in ops] + (["Част от искането не може да се направи и е пропусната."] if partial else [])
    P.note(job, "edit: " + "; ".join(changes))
    return P.finish(job, out) | {"real_mm": round(real, 1), "edit": {"prompt": prompt[:500], "changes": changes,
                                                                     "source": params.get("source")}}  # fmt: skip


def copy_source(source_job: Path, job: Path):
    """The model the change starts from, with its photo texture if it has one."""
    shutil.copy(source_job / "model.ply", job / SOURCE)
    if (source_job / TX.NPZ).exists():
        for f in [source_job / TX.NPZ, *TX.texture_files(source_job)]:
            shutil.copy(f, job / f.name)
