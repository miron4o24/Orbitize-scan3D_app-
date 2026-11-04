"""A photo texture for the model: the frames themselves wrapped onto it (OpenMVS TextureMesh), so print on the object,
down to small text, reads as in the frames. Vertex colours (mesh.paint) give one colour per point, a few millimetres
apart on a 10 cm print: a drill's battery label was a grey smear; textured, "LI-ION 4.0Ah 18V" reads.

The frames go in at full size (up to TEXTURE_SIDE), from their own undistortion: everything else works on frames of
pipeline.MAX_FRAME_SIDE, which is enough for the shape and keeps dense matching affordable.

TextureMesh's seam levelling (global and local) blacked out most of the texture with this OpenMVS build (a drill:
12-19% of the atlas pure black, in blobs over the faces), so it is off, and `level` does the global part here: each
patch of the texture comes from one frame, and frames differ in exposure and in the shine they catch, so without it
a black handle came out in grey and black blotches. Faces no frame saw (in crevices) come out in EMPTY and are
filled with their painted vertex colour instead. The atlas between the patches is filled with the nearest patch's
colour (`fill`): left in EMPTY, it bled into the patches' edges on the phone (its mipmaps), a pink fringe.

Kept in the job directory: textured.npz (per texture: vertices, normals, uv, faces, in the model's printed
coordinates) and texture_<k>.jpg; mesh.glb_textured builds the GLB from them.
"""

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

TEXTURE_SIDE = 3840  # frames up to 4K go in at full size
MAX_TEXTURE = 4096  # per texture image: phones load 4096 x 4096 (64 MB); more is split into several
EMPTY = (255, 0, 255)  # TextureMesh's colour for faces no frame sees: one that does not occur on real objects
CELL = 4  # texels per side of an unseen face's own colour square (see paint_unseen)
NPZ = "textured.npz"
# level: how much a patch's colour correction must stay smooth across the patch, against matching its neighbours at
# the seams (Waechter et al. 2014, "Let There Be Color!", use 1/λ = 10).
SMOOTH = 10.0
RASTER_BATCH = 2_000_000  # texel candidates per step of rasterize (memory: ~50 bytes each)
# Frames blurrier than this share of the median sharpness (on the object) stay out of the texture: TextureMesh picks
# each face's frame by how big and straight-on the face is in it, not how sharp, and a scan kept frames at a sixth of
# the median. At most TEXTURE_DROP of the frames are left out, so every side still has frames.
TEXTURE_SHARP = 0.5
TEXTURE_DROP = 0.3
SHARPNESS_SIDE = 1280  # frames are measured at this size


def texture_files(job: Path) -> list[Path]:
    return sorted(job.glob("texture_*.jpg"))


def clear(job: Path) -> None:
    """Removes a previous run's texture (a retried job)."""
    (job / NPZ).unlink(missing_ok=True)
    for p in texture_files(job):
        p.unlink(missing_ok=True)


def parts(loaded) -> list[trimesh.Trimesh]:
    """The textured meshes of an OBJ: one per material (texture image)."""
    if isinstance(loaded, trimesh.Scene):
        return [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    return [loaded]


def image_of(m: trimesh.Trimesh) -> Image.Image | None:
    mat = getattr(m.visual, "material", None)
    img = getattr(mat, "image", None) if mat is not None else None
    if img is None:
        img = getattr(mat, "baseColorTexture", None)
    return img


def unseen_faces(a: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """The faces whose texture is EMPTY (no frame saw them). a: the atlas (H, W, 3); uv: (faces, 3, 2), v up."""
    H, W = a.shape[:2]
    centre = uv.mean(1)
    x = np.clip(np.round(centre[:, 0] * (W - 1)), 0, W - 1).astype(int)
    y = np.clip(np.round((1 - centre[:, 1]) * (H - 1)), 0, H - 1).astype(int)
    return np.abs(a[y, x].astype(int) - EMPTY).sum(1) < 120  # JPEG shifts the colour a little


def rasterize(uv: np.ndarray, W: int, H: int, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which of `faces` covers each texel's centre, and where in it: (H, W) face index (-1: none) and (H, W, 3)
    barycentric weights. uv: (all faces, 3, 2), v up. Faces go in batches of similar size, each as a grid of its
    bounding boxes."""
    px = np.stack([uv[..., 0] * W, (1 - uv[..., 1]) * H], -1)[faces]  # (n, 3, 2) pixels, y down
    owner = np.full((H, W), -1, np.int32)
    bary = np.zeros((H, W, 3), np.float32)
    lo = np.floor(px.min(1) - 0.5).astype(np.int64)  # texels whose centre (i + 0.5) may lie inside
    hi = np.ceil(px.max(1) - 0.5).astype(np.int64)
    size = hi - lo + 1
    order = np.argsort(size[:, 0] * size[:, 1], kind="stable")
    i = 0
    while i < len(order):
        end = min(len(order), i + max(1, RASTER_BATCH // max(1, int(size[order[i]].prod()))))
        while True:
            bw, bh = size[order[i:end]].max(0)
            if bw * bh * (end - i) <= RASTER_BATCH or end == i + 1:
                break
            end = i + (end - i) // 2
        k = order[i:end]
        i = end
        X = lo[k, 0][:, None, None] + np.arange(bw)[None, None, :]
        Y = lo[k, 1][:, None, None] + np.arange(bh)[None, :, None]
        X, Y = np.broadcast_to(X, (len(k), bh, bw)), np.broadcast_to(Y, (len(k), bh, bw))
        a, b, c = (px[k, j][:, None, None, :] for j in range(3))
        v0, v1 = b - a, c - a
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        dx, dy = X + 0.5 - a[..., 0], Y + 0.5 - a[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = (dx * v1[..., 1] - v1[..., 0] * dy) / den
            l2 = (v0[..., 0] * dy - dx * v0[..., 1]) / den
        l0 = 1 - l1 - l2
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6) & (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        f, yy, xx = np.nonzero(inside)
        owner[Y[f, yy, xx], X[f, yy, xx]] = faces[k[f]]
        bary[Y[f, yy, xx], X[f, yy, xx]] = np.stack([l0[f, yy, xx], l1[f, yy, xx], l2[f, yy, xx]], -1)
    return owner, bary


def nearest_covered(covered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every texel, the nearest texel some face covers (its row and column)."""
    iy, ix = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    return iy, ix


def level(a: np.ndarray, uv: np.ndarray, corner: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Global seam levelling: a smooth colour correction per patch so that patches agree where they meet, while the
    detail inside them stays. a: atlas (H, W, 3) uint8; uv: (faces, 3, 2); corner: (faces, 3) the model's vertex at
    each corner; seen: the faces with a real texture. Returns the corrected atlas with the space between the patches
    filled from the nearest patch, the covered texels (H, W) and a line for the log."""
    H, W = a.shape[:2]
    owner, bary = rasterize(uv, W, H, np.flatnonzero(seen))
    covered = owner >= 0
    if not covered.any():
        return a, covered, "no texels covered"
    iy, ix = nearest_covered(covered)
    filled = a[iy, ix]
    # Texture vertices: one per distinct (model vertex, uv); a model vertex on a seam has one per patch around it.
    key = np.c_[corner.reshape(-1), np.round(uv.reshape(-1, 2) * 1e6)]
    _, tv = np.unique(key, axis=0, return_inverse=True)
    tv = tv.reshape(-1, 3)
    n = int(tv.max()) + 1
    # The colour each patch has at each vertex: sampled a little inside its faces (towards that corner), on the atlas
    # lightly blurred, averaged over the faces around it.
    soft = uniform_filter(filled.astype(np.float32), size=(3, 3, 1))
    w = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    pts = np.einsum("kc,fcd->fkd", w, uv[seen])
    x = np.clip((pts[..., 0] * W).astype(np.int64), 0, W - 1)
    y = np.clip(((1 - pts[..., 1]) * H).astype(np.int64), 0, H - 1)
    f = np.zeros((n, 3))
    count = np.zeros(n)
    np.add.at(f, tv[seen].reshape(-1), soft[y, x].reshape(-1, 3))
    np.add.at(count, tv[seen].reshape(-1), 1)
    known = count > 0
    f[known] /= count[known, None]
    # Seams: neighbouring texture vertices of one model vertex, both seen.
    pairs = np.unique(np.c_[corner.reshape(-1), tv.reshape(-1)], axis=0)
    pairs = pairs[known[pairs[:, 1]]]
    same = pairs[1:, 0] == pairs[:-1, 0]
    si, sj = pairs[:-1, 1][same], pairs[1:, 1][same]
    if not len(si):
        return np.where(covered[..., None], a, filled), covered, "one patch, nothing to level"
    edges = np.unique(np.sort(np.r_[tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]], 1), axis=0)

    def incidence(i, j, m):
        r = np.arange(len(i))
        return sp.csr_matrix((np.r_[np.ones(len(i)), -np.ones(len(i))], (np.r_[r, r], np.r_[i, j])), shape=(m, n))

    S, E = incidence(si, sj, len(si)), incidence(edges[:, 0], edges[:, 1], len(edges))
    A = (S.T @ S + SMOOTH * (E.T @ E) + 1e-4 * sp.identity(n)).tocsc()
    g = splu(A).solve(-(S.T @ (f[si] - f[sj])))  # (n, 3)
    before = np.abs(f[si] - f[sj]).mean()
    after = np.abs(f[si] + g[si] - f[sj] - g[sj]).mean()

    out = a.astype(np.float32)
    flat = np.flatnonzero(covered)
    for s in range(0, len(flat), RASTER_BATCH):
        t = flat[s : s + RASTER_BATCH]
        yy, xx = np.unravel_index(t, (H, W))
        out[yy, xx] += np.einsum("tk,tkc->tc", bary[yy, xx], g[tv[owner[yy, xx]]])
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    out = out[iy, ix]  # the covered texels keep their own colour; the rest take their nearest's
    note = f"{len(si)} seam points, colour step across seams {before:.1f} -> {after:.1f} of 255, mean change {np.abs(g).mean():.1f}"
    return out, covered, note


def paint_unseen(img: Image.Image, uv: np.ndarray, colors: np.ndarray, empty: np.ndarray,
                 start: int) -> tuple[Image.Image, np.ndarray, float]:  # fmt: skip
    """The `empty` faces (no frame saw them) in their painted colour. TextureMesh points all of them at one texel, so
    each gets its own CELL-wide square in rows added from `start` (below the last patch, as TextureMesh packs from the
    top), filled with the mean of its corners' colours. uv: (faces, 3, 2) per corner, v up; colors: (faces, 3, 3+)
    per corner. Returns the image, the new uv and the share of such faces."""
    a = np.asarray(img.convert("RGB"))
    H, W = a.shape[:2]
    n = int(empty.sum())
    if not n:
        return img, uv, 0.0
    # More rows are added if those below the patches are too few, but never past MAX_TEXTURE (smaller squares then).
    for cell in (CELL, 2, 1):
        rows = int(np.ceil(n / (W // cell))) * cell
        if start + rows <= MAX_TEXTURE or cell == 1:
            break
    total = max(H, start + rows)
    out = np.zeros((total, W, 3), np.uint8)
    out[:H] = a
    k = np.arange(n)
    per_row = W // cell
    cx, cy = (k % per_row) * cell, start + (k // per_row) * cell
    col = colors[empty][:, :, :3].mean(1).round().astype(np.uint8)
    for dy in range(cell):
        for dx in range(cell):
            out[cy + dy, cx + dx] = col
    uv = uv.copy()
    uv[..., 1] = 1 - (1 - uv[..., 1]) * H / total  # the old image now fills the top H of `total` rows
    uv[empty] = np.stack([(cx + cell / 2) / W, 1 - (cy + cell / 2) / total], -1)[:, None, :]
    return Image.fromarray(out), uv, n / len(uv)


def frame_sharpness(path: Path, mask: np.ndarray | None) -> float:
    """Variance of the Laplacian over the object (its mask, any size), on the frame scaled to SHARPNESS_SIDE."""
    with Image.open(path) as im:
        g = im.convert("L")
        if max(g.size) > SHARPNESS_SIDE:
            s = SHARPNESS_SIDE / max(g.size)
            g = g.resize((round(g.width * s), round(g.height * s)), Image.BILINEAR)
        a = np.asarray(g, np.float32)
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    if mask is not None:
        m = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).resize((a.shape[1], a.shape[0]))
        inside = np.asarray(m)[1:-1, 1:-1] > 127
        if inside.sum() > 500:
            return float(lap[inside].var())
    return float(lap.var())


def blurry_frames(photos: Path, names: list[str], masks: dict[str, np.ndarray]) -> list[str]:
    """The frames to leave out of the texture (see TEXTURE_SHARP)."""
    if len(names) < 10:
        return []
    s = np.array([frame_sharpness(photos / n, masks.get(n)) for n in names])
    limit, median = int(len(names) * TEXTURE_DROP), float(np.median(s))
    return [names[i] for i in np.argsort(s)[:limit] if s[i] < TEXTURE_SHARP * median]


def texture_model(job: Path, work: Path, photos: Path, best: Path, model: trimesh.Trimesh, to_print: np.ndarray,
                  to_up: np.ndarray, sh: Callable[[str, list], None], colmap: str, threads: str,
                  names: list[str] = (), masks: dict[str, np.ndarray] | None = None) -> dict:  # fmt: skip
    """Textures `model` (printed coordinates; to_print: 4x4 from the cloud's coordinates to them; to_up: world ->
    cloud rotation) from the frames in `photos`, posed by the COLMAP model `best` (its frames: `names`; masks: the
    object in each, for measuring sharpness). Writes textured.npz and the texture images into the job; returns a
    summary for the log. Raises on failure (the caller keeps the vertex colours)."""
    tex = work / "texture"
    shutil.rmtree(tex, ignore_errors=True)
    blurry = blurry_frames(photos, list(names), masks or {})
    posed = best
    if blurry:
        (work / "blurry.txt").write_text("\n".join(blurry) + "\n")
        sharp = work / "texture_sparse"
        shutil.rmtree(sharp, ignore_errors=True)
        sharp.mkdir(parents=True)
        try:
            sh("texturing", [colmap, "image_deleter", "--input_path", best, "--output_path", sharp,
                "--image_names_path", work / "blurry.txt"])  # fmt: skip
            posed = sharp
        except Exception:  # (pipeline.Failed) then all frames, as before
            blurry = []
    sh("texturing", [colmap, "image_undistorter", "--image_path", photos, "--input_path", posed, "--output_path", tex,
        "--output_type", "COLMAP", "--max_image_size", str(TEXTURE_SIDE)])  # fmt: skip
    sh("texturing", ["InterfaceCOLMAP", "-w", tex, "-i", tex, "-o", tex / "scene.mvs",
        "--image-folder", tex / "images"])  # fmt: skip
    # Back from printed coordinates to the cameras' (world) ones.
    cloud = trimesh.transform_points(model.vertices, np.linalg.inv(to_print))
    trimesh.Trimesh(cloud @ to_up, model.faces, process=False).export(tex / "model.ply")
    empty = (EMPTY[0] << 16) | (EMPTY[1] << 8) | EMPTY[2]
    sh("texturing", ["TextureMesh", "-w", tex, "-i", tex / "scene.mvs", "-m", tex / "model.ply",
        "-o", tex / "textured.obj", "--export-type", "obj", "--close-holes", "0", "--decimate", "1",
        "--global-seam-leveling", "0", "--local-seam-leveling", "0", "--max-texture-size", str(MAX_TEXTURE),
        "--empty-color", str(empty), "--max-threads", threads])  # fmt: skip
    meshes = parts(trimesh.load(tex / "textured.obj", process=False))
    if not meshes:
        raise ValueError("TextureMesh wrote no mesh")
    tree = cKDTree(model.vertices)
    normals = model.vertex_normals
    colors = model.visual.vertex_colors if model.visual.kind == "vertex" else np.full((len(model.vertices), 4), 160)
    arrays, unseen, sizes, levelled = {}, [], [], []
    clear(job)
    for k, m in enumerate(meshes):
        img = image_of(m)
        if img is None or m.visual.uv is None:
            raise ValueError("TextureMesh wrote a mesh without texture")
        verts = trimesh.transform_points(m.vertices @ to_up.T, to_print)
        _, near = tree.query(verts)  # the same points as the model's (TextureMesh keeps the surface)
        corner = near[m.faces]  # (faces, 3): the model's vertex at each corner
        uv = np.asarray(m.visual.uv, np.float64)[m.faces]
        a = np.asarray(img.convert("RGB"))
        empty = unseen_faces(a, uv)
        try:
            a, covered, note = level(a, uv, corner, ~empty)
        except (RuntimeError, ValueError, MemoryError) as e:  # then the atlas as TextureMesh made it
            covered, note = np.abs(a.astype(int) - EMPTY).sum(2) >= 120, f"not levelled ({e})"
        levelled.append(note)
        rows = np.flatnonzero(covered.any(1))
        start = int(rows[-1]) + 3 if len(rows) else 0
        img, uv, share = paint_unseen(Image.fromarray(a), uv, colors[corner], empty, start)
        unseen.append(share * len(m.faces))
        # One vertex per distinct (point, uv): TextureMesh writes three per face.
        corner, uv = corner.reshape(-1), uv.reshape(-1, 2)
        _, first, inverse = np.unique(np.c_[corner, np.round(uv * 1e6)], axis=0, return_index=True,
                                      return_inverse=True)  # fmt: skip
        arrays |= {
            f"vertices_{k}": model.vertices[corner[first]].astype(np.float32),
            f"normals_{k}": normals[corner[first]].astype(np.float32),
            f"uv_{k}": uv[first].astype(np.float32),
            f"faces_{k}": inverse.reshape(-1, 3).astype(np.uint32),
        }
        img.convert("RGB").save(job / f"texture_{k}.jpg", quality=92)
        sizes.append(img.size)
    np.savez_compressed(job / NPZ, count=len(meshes), **arrays)
    faces = sum(len(m.faces) for m in meshes)
    return {"textures": [list(s) for s in sizes], "unseen": round(sum(unseen) / max(faces, 1), 4),
            "levelled": "; ".join(levelled), "frames": f"{len(names) - len(blurry)} of {len(names)}"}  # fmt: skip


def load(job: Path) -> list[dict] | None:
    """The kept texture: per image {vertices, normals, uv, faces, jpeg (bytes), size (w, h)}; None if there is none."""
    path = job / NPZ
    if not path.exists():
        return None
    out = []
    with np.load(path) as z:
        for k in range(int(z["count"])):
            img = job / f"texture_{k}.jpg"
            with Image.open(img) as im:
                size = im.size
            out.append({"vertices": z[f"vertices_{k}"].astype(np.float64), "normals": z[f"normals_{k}"],
                        "uv": z[f"uv_{k}"], "faces": z[f"faces_{k}"], "jpeg": img.read_bytes(), "size": size})  # fmt: skip
    return out
