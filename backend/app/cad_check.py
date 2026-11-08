"""Инженерен режим: проверка на CAD модел, написан като код на CadQuery (от app/cad_claude.py), и рисунки за сравнение.

Кодът се изпълнява в отделен процес и само ако използва cadquery и math, без файлове, мрежа или вътрешности на Python
(проверено по синтактичното дърво преди изпълнението). Като собствен процес:
python cad_check.py <part.py> <сканиране в мм .stl> <мерки.json> <изходна папка>
Пише part.step, part.stl, check.json (размери, разминаване със сканирането, грешка) и compare.png (горе
сканирането, долу моделът, от четири страни с мрежа през 10 мм).
python cad_check.py --views <сканиране в мм .stl> <out.png>: само рисунките на сканирането."""

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

ALLOWED_IMPORTS = {"cadquery", "math"}
FORBIDDEN = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "getattr", "setattr",
             "delattr", "vars", "breakpoint", "help", "exit", "quit", "memoryview", "type", "object", "super",
             "classmethod", "staticmethod", "property", "dir", "id", "hash"}  # fmt: skip
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in
                 ("abs", "min", "max", "round", "range", "len", "sum", "float", "int", "list", "tuple", "dict", "zip",
                  "enumerate", "sorted", "reversed", "map", "bool", "str", "print", "any", "all", "isinstance",
                  "ValueError", "Exception")}  # fmt: skip
VIEW = 360  # пиксела на изглед


class Unsafe(ValueError):
    pass


def safe(code: str) -> None:
    """Само cadquery и math; без забранените имена и без атрибути с долна черта."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise Unsafe(f"import {a.name} is not allowed (only cadquery and math)")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise Unsafe(f"from {node.module} import is not allowed (only cadquery and math)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN:
            raise Unsafe(f"{node.id} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise Unsafe(f"attribute {node.attr} is not allowed")


def run_code(code: str):
    import cadquery as cq

    safe(code)

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise Unsafe(f"import {name} is not allowed")
        return __import__(name, *args, **kwargs)

    ns = {"__builtins__": SAFE_BUILTINS | {"__import__": guarded_import}, "cq": cq, "math": math}
    exec(compile(code, "part.py", "exec"), ns)  # noqa: S102 (проверено от safe)
    result = ns.get("result")
    if result is None:
        raise ValueError("part.py must put the final solid in a variable named result")
    shape = result.val() if isinstance(result, cq.Workplane) else result
    if isinstance(result, cq.Workplane) and len(result.vals()) > 1:
        shape = cq.Compound.makeCompound([v for v in result.vals() if isinstance(v, cq.Shape)])
    if not isinstance(shape, cq.Shape):
        raise ValueError("result is not a solid")
    return shape


# --- Рисунки --------------------------------------------------------------------------------------------------------

VIEWS = (("front", np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
         ("right", np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
         ("top", np.array([0, 0, -1.0]), np.array([0, 1.0, 0])),
         ("iso", np.array([-1.0, 1.0, -0.8]), np.array([0, 0, 1.0])))  # fmt: skip


def render(mesh: trimesh.Trimesh, look: np.ndarray, up: np.ndarray, span: float, centre: np.ndarray) -> Image.Image:
    """Сива рисунка (паралелна проекция) с мрежа през 10 мм (5 мм при по-малки от 40 мм)."""
    fwd = look / np.linalg.norm(look)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    pts, face = trimesh.sample.sample_surface(mesh, 500_000, seed=0)
    n = mesh.face_normals[face]
    rel = pts - centre
    x, y, d = rel @ right, rel @ upv, rel @ fwd
    s = VIEW / (span * 1.15)
    u = (x * s + VIEW / 2).astype(int)
    v = (VIEW / 2 - y * s).astype(int)
    ok = (u >= 0) & (v >= 0) & (u < VIEW) & (v < VIEW)
    u, v, d, n = u[ok], v[ok], d[ok], n[ok]
    light = -fwd * 0.75 + upv * 0.45 - right * 0.3
    shade = np.clip(np.abs(n @ (light / np.linalg.norm(light))), 0, 1) * 0.8 + 0.15
    bg = Image.new("RGB", (VIEW, VIEW), (252, 252, 252))
    draw = ImageDraw.Draw(bg)
    grid = 10.0 if span >= 40 else 5.0
    for k in range(-int(span / grid) - 2, int(span / grid) + 3):
        p = int(VIEW / 2 + k * grid * s)
        for a in ((p, 0, p, VIEW), (0, p, VIEW, p)):
            draw.line(a, fill=(200, 218, 238), width=1)
    arr = np.asarray(bg).copy()
    order = np.argsort(-d)  # далечните първо, близките отгоре
    grey = (shade[order] * 255).astype(np.uint8)
    arr[v[order], u[order]] = grey[:, None]
    return Image.fromarray(arr)


def views(meshes: list[tuple[str, trimesh.Trimesh]], out: Path) -> None:
    lo = np.min([m.bounds[0] for _, m in meshes], axis=0)
    hi = np.max([m.bounds[1] for _, m in meshes], axis=0)
    centre, span = (lo + hi) / 2, float(np.max(hi - lo))
    rows = []
    for label, m in meshes:
        tiles = [render(m, look, up, span, centre) for _, look, up in VIEWS]
        row = Image.new("RGB", (VIEW * len(tiles), VIEW + 22), "white")
        d = ImageDraw.Draw(row)
        for k, (t, (name, _, _)) in enumerate(zip(tiles, VIEWS)):
            row.paste(t, (k * VIEW, 22))
            d.text((k * VIEW + 6, 5), f"{label}: {name}", fill=(20, 20, 20))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r_ in rows:
        sheet.paste(r_, (0, y))
        y += r_.height
    sheet.save(out)


# --- Проверка -------------------------------------------------------------------------------------------------------


def deviation(scan: trimesh.Trimesh, cad: trimesh.Trimesh) -> dict:
    from scipy.spatial import cKDTree

    a = trimesh.sample.sample_surface(scan, 30000, seed=0)[0]
    b = trimesh.sample.sample_surface(cad, 200000, seed=1)[0]
    d = np.r_[cKDTree(b).query(a)[0], cKDTree(a).query(trimesh.sample.sample_surface(cad, 30000, seed=2)[0])[0]]
    return {"mean": round(float(d.mean()), 3), "p90": round(float(np.percentile(d, 90)), 3)}


def check(part: Path, scan_path: Path, meas: dict, out: Path) -> dict:
    import cadquery as cq

    out.mkdir(parents=True, exist_ok=True)
    try:
        shape = run_code(part.read_text(encoding="utf-8"))
        if not shape.isValid():
            raise ValueError("the solid is not valid (self-intersecting or open); simplify the failing feature")
        cq.exporters.export(cq.Workplane().add(shape), str(out / "part.step"))
        size = max(shape.BoundingBox().xlen, shape.BoundingBox().ylen, shape.BoundingBox().zlen)
        cq.exporters.export(cq.Workplane().add(shape), str(out / "part.stl"), tolerance=max(0.01, size / 2000),
                            angularTolerance=0.1)  # fmt: skip
    except Exception as e:
        report = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}
        (out / "check.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        return report
    cad = trimesh.load(out / "part.stl", force="mesh")
    scan = trimesh.load(scan_path, force="mesh")
    bb = shape.BoundingBox()
    size = {"length": round(bb.xlen, 3), "width": round(bb.ylen, 3), "height": round(bb.zlen, 3)}
    off = {k: round(size[k] - float(meas[k]), 3) for k in size if meas.get(k)}
    report = {"ok": True, "size_mm": size, "off_from_measured_mm": off, "deviation_from_scan_mm": deviation(scan, cad),
              "volume_mm3": round(float(shape.Volume()), 1), "solids": len(shape.Solids()),
              "min_corner": [round(bb.xmin, 3), round(bb.ymin, 3), round(bb.zmin, 3)]}  # fmt: skip
    views([("scan", scan), ("model", cad)], out / "compare.png")
    (out / "check.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    if sys.argv[1] == "--views":
        views([("scan", trimesh.load(sys.argv[2], force="mesh"))], Path(sys.argv[3]))
    else:
        rep = check(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")),
                    Path(sys.argv[4]))  # fmt: skip
        print(json.dumps(rep))
