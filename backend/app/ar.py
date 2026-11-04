"""Моделът в добавена реалност: страница (PAGE), която приложението отваря в Chrome, показва модела, а бутонът ѝ го
поставя върху под или маса, видими за камерата, в реалния му размер: собственият AR на Chrome (WebXR) чрез
model-viewer на Google, иначе Scene Viewer на Google. Стартиран директно от приложението, Scene Viewer показваше само
празен екран на телефона на потребителя; страницата така или иначе първо показва модела и казва какво липсва, когато
телефонът не поддържа AR. Scene Viewer иска метри, най-много около 100 000 триъгълника (идеално 30-50 хил.), текстури
до 2048 x 2048 и най-много около 10 MB; едно сканиране има до 250 000 триъгълника и текстури до 4096 px (дрелка: 189
хил. триъгълника, 10 MB). Телефонът изтегля файла през Funnel, затова по-малкият се зарежда по-скоро: FACES
триъгълника, изпратени с gzip (с 44% по-малки). Браузърът взема страницата и файла без ключа на приложението, затова
приложението получава подписан линк с изтичане (main.ar_link); файлът се създава тогава и се пази в задачата
(ar_<десети от мм>.glb и .glb.gz)."""

import gzip
import html
import io
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from PIL import Image

from . import mesh as M
from . import texture as TX

FACES = 60_000
TEXTURE = 2048
log = logging.getLogger("uvicorn.error")


def small_texture(jpeg: bytes, size: tuple[int, int]) -> tuple[bytes, tuple[int, int]]:
    if max(size) <= TEXTURE:
        return jpeg, size
    with Image.open(io.BytesIO(jpeg)) as im:
        im = im.convert("RGB")
        im.thumbnail((TEXTURE, TEXTURE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return buf.getvalue(), im.size


def textured_part(p: dict, faces: int) -> dict:
    """Една част с текстура (texture.load) с най-много `faces` триъгълника и текстурата на мястото ѝ. Първо се сливат
    копията на точка по шевовете, така че повърхността да се опрости като едно цяло, с координати на текстурата за
    всеки ъгъл."""
    jpeg, size = small_texture(p["jpeg"], p["size"])
    p = p | {"jpeg": jpeg, "size": size}
    if len(p["faces"]) <= faces:
        return p
    points, weld = np.unique(p["vertices"], axis=0, return_inverse=True)
    f = weld.reshape(-1)[p["faces"].astype(np.int64)]
    # Чрез файл OBJ: мрежа, подадена като масиви, няма текстура за ъглите си и филтърът я отказва.
    ms = pymeshlab.MeshSet()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "m.mtl").write_text("newmtl t\nmap_Kd t.jpg\n")
        with open(Path(d) / "m.obj", "w") as fh:
            fh.write("mtllib m.mtl\n")
            np.savetxt(fh, points, "v %.6f %.6f %.6f")
            np.savetxt(fh, p["uv"][p["faces"]].reshape(-1, 2), "vt %.7f %.7f")
            fh.write("usemtl t\n")
            corners = np.arange(1, f.size + 1).reshape(-1, 3)
            np.savetxt(fh, np.stack([f + 1, corners], -1).reshape(-1, 6), "f %d/%d %d/%d %d/%d")
        ms.load_new_mesh(str(Path(d) / "m.obj"))
    ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=faces, preservenormal=True, planarquadric=True)
    m = ms.current_mesh()
    points, f, uv = m.vertex_matrix(), m.face_matrix(), m.wedge_tex_coord_matrix()
    normals = trimesh.Trimesh(points, f, process=False).vertex_normals
    # Разделяне отново: по един връх за всяка различна двойка (точка, uv), защото glTF има координати на текстурата за
    # всеки връх.
    corner = f.reshape(-1)
    _, first, inverse = np.unique(np.c_[corner, np.round(uv * 1e6)], axis=0, return_index=True, return_inverse=True)
    return p | {"vertices": points[corner[first]], "normals": normals[corner[first]], "uv": uv[first],
                "faces": inverse.reshape(-1, 3)}  # fmt: skip


def make(job: Path, longest_mm: float) -> bytes:
    """GLB за Scene Viewer: основата е на пода в началото на координатите, най-дълга страна longest_mm, в метри."""
    mesh = trimesh.load(job / "model.ply", force="mesh")
    textured = TX.load(job)
    if textured:
        total = sum(len(p["faces"]) for p in textured)
        try:
            textured = [textured_part(p, max(500, FACES * len(p["faces"]) // total)) for p in textured]
        except Exception as e:  # цветовете на върховете тогава, вместо да няма AR
            log.warning("AR file for %s without the texture: %s", job.name, e)
            textured = None
    if not textured and len(mesh.faces) > FACES:
        mesh = M.decimate(mesh, FACES, keep_closed=False)
    return M.export(mesh, "glb", longest_mm / 1000, textured, unlit=True)


def cached(job: Path, tenths: int) -> Path:
    """Запазеният файл за този размер, създаден наново, когато моделът е по-нов (повторно обработено сканиране). По
    един наведнъж: двойно натискане не бива да записва същия файл два пъти едновременно."""
    with _making:
        return _cached(job, tenths)


_making = threading.Lock()


def _cached(job: Path, tenths: int) -> Path:
    path = job / f"ar_{FACES // 1000}k_unlit_{tenths}.glb"  # FACES и unlit са в името: при промяна файловете се създават наново
    sources = [job / "model.ply", job / TX.NPZ]
    newest = max((s.stat().st_mtime for s in sources if s.exists()), default=0)
    if path.exists() and path.stat().st_mtime >= newest and zipped(path).exists():
        return path
    for old in job.glob("ar_*.glb*"):  # пази се един размер: всяка промяна на размера в приложението създава нов файл
        old.unlink(missing_ok=True)
    data = make(job, tenths / 10)
    for target, content in ((zipped(path), gzip.compress(data, 6)), (path, data)):
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(content)
        tmp.replace(target)
    return path


def zipped(path: Path) -> Path:
    return path.with_name(path.name + ".gz")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · View in your room</title>
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@4.1.0/dist/model-viewer.min.js"></script>
<style>
  :root {{ color-scheme: light; --bg: #f5f7fb; --card: #fff; --text: #0f172a; --muted: #64748b; --accent: #2563eb;
          --danger: #b91c1c; }}
  html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--text);
               font: 16px/1.45 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  main {{ display: flex; flex-direction: column; height: 100%; }}
  model-viewer {{ flex: 1; width: 100%; background: linear-gradient(#ffffff, #e9edf3); }}
  .progress {{ position: absolute; inset: 0 0 auto 0; height: 4px; }}
  .progress div {{ height: 100%; width: 0; background: var(--accent); transition: width .2s; }}
  .panel {{ display: grid; gap: 10px; padding: 18px 16px 22px; background: var(--card); border-radius: 22px 22px 0 0;
           box-shadow: 0 -6px 24px rgba(15, 23, 42, .08); }}
  h1 {{ margin: 0; font-size: 20px; }}
  p {{ margin: 0; color: var(--muted); font-size: 14px; }}
  p.error {{ color: var(--danger); }}
  button {{ appearance: none; border: 0; border-radius: 14px; padding: 16px; width: 100%; background: var(--accent);
           color: #fff; font: 600 17px system-ui, -apple-system, Roboto, sans-serif; }}
  button[disabled] {{ opacity: .45; }}
</style></head>
<body><main>
  <model-viewer id="mv" src="model.glb" alt="{title}" camera-controls touch-action="pan-y" auto-rotate
      shadow-intensity="1" interaction-prompt="none" ar ar-modes="webxr scene-viewer" ar-placement="floor"
      ar-scale="{scale}">
    <div slot="progress-bar" class="progress"><div id="bar"></div></div>
    <span slot="ar-button"></span>
  </model-viewer>
  <div class="panel">
    <h1>{title}</h1>
    <p>{size}</p>
    <p id="msg">Loading the model…</p>
    <button id="ar" disabled>View in your room</button>
  </div>
</main>
<script type="module">
const mv = document.getElementById('mv'), button = document.getElementById('ar'), msg = document.getElementById('msg');
const say = (text, error) => {{ msg.textContent = text; msg.className = error ? 'error' : ''; }};
mv.addEventListener('progress', (e) => {{ document.getElementById('bar').style.width = e.detail.totalProgress * 100 + '%'; }});
mv.addEventListener('error', () => say('The model could not be loaded. Go back to the app and try again.', true));
mv.addEventListener('load', () => {{
  document.querySelector('.progress').remove();
  // canActivateAR settles a moment after loading (Chrome answers whether this phone does AR).
  setTimeout(() => {{
    if (mv.canActivateAR) {{
      button.disabled = false;
      say('Tap the button, then point the phone at a floor or table and move it slowly until the model appears.');
    }} else {{
      say('This browser cannot show AR. Open the page in Chrome, and install "Google Play Services for AR" from the ' +
          'Play Store. You can still turn the model around here.', true);
    }}
  }}, 600);
}});
mv.addEventListener('ar-status', (e) => {{
  if (e.detail.status === 'object-placed') say('Drag to move it, twist with two fingers to turn it.');
  if (e.detail.status === 'failed')
    say('AR did not start. Install or update "Google Play Services for AR" from the Play Store and try again.', true);
}});
button.addEventListener('click', () => mv.activateAR());
</script></body></html>"""


def page(title: str, longest_mm: float, real_size: bool) -> str:
    size = f"Real size: {longest_mm:g} mm on its longest side" if real_size else "Pinch in AR to make it bigger or smaller"
    return PAGE.format(title=html.escape(title or "3D scan"), size=html.escape(size),
                       scale="fixed" if real_size else "auto")  # fmt: skip
