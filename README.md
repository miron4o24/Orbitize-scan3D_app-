# 3D Scan: phone camera to printable 3D model

Walk slowly around an object twice (once low, once from higher up) filming it with the app. A self-hosted backend
picks the sharpest, evenly spread frames from the video, reconstructs the object with photogrammetry and returns a
cleaned, watertight mesh (STL / 3MF / OBJ / GLB).

- `src/`: Expo app (SDK 57, expo-router, TypeScript). Record, upload, preview, export.
- `backend/`: FastAPI + ffmpeg (frame selection) + COLMAP (camera poses) + OpenMVS (dense points) + IS-Net/ONNX Runtime (object masks) + PyMeshLab/trimesh (mesh). CPU only. Runs natively on Windows or in Docker on Linux.

## How a video becomes frames

The capture screen tracks the phone's heading with the motion sensors: a ring of dots fills as the user walks around
the object, a second ring for a higher circle (it only counts once the phone looks down noticeably more than on the
first). Parts walked past too fast stay open, and recording stops by itself after both circles. The app records
1080p at 6 Mbit/s (~45 MB a minute) with stabilisation off (stabilisation warps each frame differently, which breaks
camera solving), keeps the screen awake, and uploads in 8 MB chunks, so a dropped connection resumes instead of
starting over. It also saves the phone's orientation during the recording (`motion.json`, ~20 readings a second) and
uploads it with the video, for plain objects (below).

The server scores 6 frames per second of video for sharpness and for how much the picture changed, splits the video
into slices of equal camera movement (2 per second, 20–80 in total) and keeps the sharpest frame of each slice, so
standing still or an uneven pace does not waste frames. Those frames then go through the same pipeline as photos.
Each job's `log.txt` has timestamps per step plus notes (frames kept, frames placed, point counts); the app shows its
tail under "Show server details" when a scan fails.

## How frames become a model

1. COLMAP finds where each frame was filmed from, with two solvers (incremental, and COLMAP 4's global one after
   focal-length calibration). Each result is scored by how many frames aim at a common point, as every frame of an
   object scan does; the best wins and its stray frames are removed. (The global solver once returned cameras that
   did not even face the object.)
2. A salient-object model (IS-Net, ONNX on the CPU, `app/segment.py`) cuts the object out of every frame.
   OpenMVS then computes dense 3D points at full frame resolution, but only in a box around the object (its masks),
   with each frame paired to the frames filmed 4–35° around the object from it (not to frames sharing sparse
   points, of which a plain table gives too few).
3. Dense points that fall inside the object's outline in at least 70% of the frames showing them are the object;
   the table, background and stray points are dropped, whatever the background looks like. Up comes from the
   phone's gravity sensor (the orientation log, each frame's gravity turned by its solved camera, used when the
   frames agree within 10°); without the log it is guessed from the cameras. The table plane is then found among
   the points around the object and used to level it.
4. The visual hull (the volume inside the object's outline in at least 75% of the frames) gives the shape where
   dense matching finds few points: dark, plain or shiny parts and the area near the table. Real holes in an
   outline stay open (a ring's middle, a mug handle's opening; only specks under 2% of it are filled), so the hull
   shows them. The dense points then carve the hull: a frame that measured a surface point saw through everything
   between it and the point (a cup's inside seen from above), while frames for which a place is hidden behind a
   surface do not vote on it. Tested on rendered scans of odd objects with known shape (two circles, like the app):
   a C-shaped bracket came out within 1 mm on average, a stool with its four legs and open space under it, a ring
   with its hole through, a mug with its handle open and its inside hollow as deep as the high circle sees into it.
5. AI depth (Depth Anything V2 Small, ONNX on the CPU, `app/depth.py`) carves the hull further: outlines cannot show a dip
   and leave a dome over a flat top seen only from above, but the network sees both. Its depth has an unknown scale
   and offset per frame, fitted robustly to the known 3D points (dense points, and hull samples where the hull does
   not overshoot). A point of the hull is cut away when the frames that see it see the surface behind it, on
   average, by more than 3% of the object's size; carving is skipped when the depth fits fewer than half the frames
   or would remove more than 30% of the hull. (AI depth used to go in as points, stacked from every frame: their
   small per-frame errors made rough, striped faces with flaps, and a box came out a third too flat.) Without the
   model (`install-task.ps1` and `setup-windows.ps1` download it) or if this step fails, the hull stays uncarved.
   Once the dense points are in, the masks are redone (`app/refine.py`): MobileSAM (promptable segmentation, ONNX
   on the CPU, about a second a frame) gets the hull's box in each frame, points deep inside the hull as "object" and
   measured points well outside it (table, labels, the room) as "not object". Of its four answers the one most like
   the hull's view is kept, trimmed to near that view and joined with its core where the first mask agreed. IS-Net
   erred in most frames of a figure and a drill (a chair, labels, a strip of table taken in; a face, a base, a black
   trigger left out); the new masks had none of that. Everything after (object points, hull, fusion, trimming) uses
   them. Not on the phone-sensor route: there IS-Net's masks of a plain speaker were already right and MobileSAM's
   added a flap.
6. The surface is fused from the frames' depth maps (`app/fuse.py`). OpenMVS's depth maps are striped with noise
   and empty on dark parts; each frame's AI depth is smooth and complete but off in scale and gently bent. Fitted to
   the frame's own depth map (scale and offset, then a smooth correction field), it keeps the measured shape with a
   clean surface and fills the gaps. Every frame votes where the surface is (a truncated signed distance averaged
   in a grid of 1/400 of the size, then lightly blurred); the uncarved outline shape fills in where no frame looked.
   Telling object from background: a frame's mask counts only where the outline shape (what most masks agree on)
   shows, so a chair, a poster or a strip of table one mask took in stays out, and a dark part one mask missed (a
   drill's black trigger) is filled; measured depth outside the object still marks the space in front of it empty;
   a thin sheet lying on the table and sticking out from the object (a red cushion under a red box) is cut off. A
   drill came out with flat panels, step edges and a round handle (Poisson: a crust of bumps); a box 1 : 0.99 : 0.95
   (its dense points: 1 : 1 : 0.96). When the fusion fails, uses fewer than half the frames, or the dense points sit
   more than 1% of the size off it, screened Poisson builds the surface from the dense points instead, with samples
   of the carved hull filling their gaps (depth 8: at 9 it followed the points' noise). The model is kept to at most
   250,000 triangles: the phone shows and exports it in JavaScript.
7. The model is turned so its footprint lines up with the axes (the smallest rectangle around it): the sizes
   reported and the longest side the app scales by are the object's own, not those of the footprint turned (a box
   standing at 29° read 100 x 100 instead of 100 x 97). Holes left by trimming and the flat-base cut are filled, and the few faces around edges shared by three or more
   faces (where duplicate points meet) are rebuilt, so the surface closes without being rebuilt as a whole (which
   loses fine detail).
8. Colours come from the photos: each point blends the frames that see it (facing it, and nearest to the camera
   where it lands, so a part hidden behind another takes nothing from that frame), weighted towards those facing it
   squarely, so exposure changes blend instead of leaving seams. The GLB for the phone has a matte material and
   linear colours: without a material, viewers use glTF's default, fully metallic one, and the phone showed models
   nearly black. Colours are what the camera recorded: on a red cloth, its white balance tints a silver object red.
9. On the camera-solver route the model also gets a photo texture (`app/texture.py`, OpenMVS TextureMesh): the frames
   themselves, at full size (the app films 4K; frames are kept up to 3840 px, everything else works on 1920),
   wrapped onto the surface, so print on the object reads: on a drill filmed in 1080p, "LI-ION 4.0Ah 18V",
   "Einhell" and "CHANGE" read on the textured model, a grey smear in vertex colours. TextureMesh's seam levelling
   blacked out 12-19% of the texture in blobs with this OpenMVS build, so it is off; faces no frame saw (the flat
   bottom, crevices) take their vertex colour in small squares of their own. The GLB (`result.glb`) carries the
   texture as JPEG; STL, OBJ and 3MF stay geometry (with vertex colours in 3MF/OBJ as before). The phone decodes
   the JPEG straight into a GPU texture through expo-gl (`src/lib/photoTexture.ts`), as React Native has no image
   decoder for three.js. About 20-60 s per scan.

The masks for dense matching are the first masks turned through COLMAP's lens model (same as segmenting the
undistorted frames again: 0.997 overlap, in a second instead of 2.5 minutes).

### Plain objects: poses from the phone's sensors

Frame matching needs texture. A plain speaker on red velvet had only 46 of 80 frames placed, from part of the circle,
and came out a wedge with flaps. When COLMAP places fewer than 85% of the frames and the app sent its orientation
log, `app/motion.py` places the frames instead: each frame's camera is turned as the phone was (sensor fusion, at the
frame's time in the video) and sits on the ray through the object's centre in its mask, at a distance from how big
the object looks (averaged over half a circle, so an oval's wide side does not look nearer than its narrow side).
The video's start delay and, per frame, the distance, centre and compass heading are fitted so all outlines agree.
The shape is then the volume inside the outline in 90% of the frames, coloured from the frame facing each spot.
Outlines cannot show a dip and leave a dome over a flat top, so AI depth (Depth Anything, scaled per frame to the
shape's walls, which the outlines get right) then carves away what the frames see through on average by more than 3%
of the object's size: the speaker's grille, sunk inside its rim, came out as a dish instead of a dome (height to
width 0.744 against the real 0.75; outlines alone: 0.777). Carving is skipped when the depth fits fewer than half the
frames or would remove more than 30% of the shape.
It is used when its outlines agree well (median overlap ≥ 75%, or ≥ 50% when COLMAP placed less than half), and only
if the frames looked at the object from at least 200° around. Deep hollows (a mug's inside) stay filled: carving one
would usually exceed the 30% limit. Filming the first circle low, at the object's height, keeps the walls exact. The job log
says which route was used (`sensor poses: ... using them`, then `AI depth carves N% ...`).

## Model by Claude (no server)

A second mode on the home screen. The phone sends 2–8 photos straight to the Claude API (`claude-opus-5-5`,
structured output), and Claude describes the object as solid parts in millimetres: lathe profiles for round things,
rounded boxes and cylinders, extrusions, plus raised features such as buttons (`src/lib/shape.ts` builds them with
three.js; `src/lib/claude.ts` makes the call). It is a clean redraw rather than a measurement, so it suits simple,
regular objects and gets fine details approximately. It needs the user's own API key, entered in the app and kept in
the phone's secure storage; one model costs roughly $0.10–0.40. The parts overlap; slicers merge them when slicing.

## Honest limits

- **Works well:** matte objects about 5–50 cm that stand out from the table, in even light, not moving.
- **Comes out rough:** plain or dark objects (their shape comes mostly from their outline, so hollows are filled in).
- **Fails:** shiny or transparent objects, objects the same colour as the table, thin parts (wires, hair), details under ~1 mm.
- **Scale:** photogrammetry has no absolute size. You enter the real length of the object's longest side.
- **Processing time (CPU):** minutes to tens of minutes per scan, depending on photo count, resolution and cores. A GPU would be several times faster but is not free. Measured numbers are in "Processing times" below.

## Processing times

| Dataset | Machine | Time |
|---|---|---|
| Middlebury templeRing, 47 photos 640×480 | 8-core laptop, CPU, Docker | 279 s (47/47 photos placed) |

## Backend

**Windows (Server 2025 / 11), no WSL or Docker, public over HTTPS: [docs/windows-server.md](docs/windows-server.md).**
In short: `scripts\setup-windows.ps1` installs COLMAP, OpenMVS and the Python packages into `backend\`;
`scripts\install-task.ps1` runs the server at boot; `scripts\e2e.ps1` tests it.

Linux, with Docker:

```bash
cd backend
cp .env.example .env            # set API_TOKEN to a long random string: openssl rand -hex 24
docker compose up --build -d    # first build compiles OpenMVS: ~20-40 min
curl localhost:8000/health      # {"ok":true}
```

### Test the pipeline without the app

```bash
cd backend
scripts/fetch_dataset.sh                                      # 47-photo sample -> samples/templeRing
API_TOKEN=$(grep API_TOKEN .env | cut -d= -f2) scripts/e2e.sh samples/templeRing   # -> e2e-out/model.{stl,glb,3mf}
API_TOKEN=$(grep API_TOKEN .env | cut -d= -f2) scripts/e2e.sh ~/orbit.mp4           # or a phone video of an object
```

### Unit tests

```bash
docker compose run --rm backend pytest -q            # in the image
# or locally: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pytest -q
```

### Accounts (Google sign-in, profiles, saved scans, share links)

`app/accounts.py` with tables in `app/db.py` (made on start): a SQLite file `backend/accounts.db` by default, MySQL
once `MYSQL_HOST` is in `backend/.env` (see `.env.example`). `DATABASE_URL=off` turns accounts off; the server then
works with the shared token only.

1. Optional, MySQL: `CREATE DATABASE scan3d CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;` and a user `scan3d`
   with all rights on it (for `'localhost'` and `'127.0.0.1'`). Put `MYSQL_*` in `backend/.env`, run
   `install-task.ps1`. The first start copies the SQLite file's accounts over (the file is renamed `.moved`).
2. Google Cloud (console.cloud.google.com), one project: OAuth consent screen (External; add test users while in
   Testing), an **Android** client (package `com.miron4o.scan3d`, SHA-1 of the EAS signing key:
   `9D:01:F2:C1:A6:0D:B5:83:5D:7B:AA:52:3B:E7:7F:26:EC:B3:6B:71`) and a **Web application** client.
3. Give the server the web client's ID, no server login needed:
   `PUT /admin/settings {"google_client_id": "...apps.googleusercontent.com"}` with the API token. The app reads it
   from `/health`, so no new app build is needed.

The app's sign-in sends Google's ID token to `POST /auth/google`; the server checks it with Google's keys and answers
with its own session token (stored hashed). Each job belongs to the account that made it; the shared `API_TOKEN` is
the admin key and sees every job. Finished scans in an account are kept (their frames are removed after
`JOB_TTL_DAYS`); jobs without an account are still deleted then. Share links: `/s/<id>` is a public page with a
three.js viewer, `/s/<id>/model.glb` the textured model.

### API

All endpoints except `/health`, the share pages `/s/...` and the signed AR links `/ar/...` need `Authorization: Bearer <API_TOKEN or session token>`.

| Method | Path | |
|---|---|---|
| POST | `/jobs` | create job, returns `{id, state: "uploading", received: []}` |
| PUT | `/jobs/{id}/video?offset=0` | raw MP4/MOV bytes appended at `offset` (≤ 16 MB per request, 600 MB total: `MAX_VIDEO_MB`); resume from `video_bytes` |
| PUT | `/jobs/{id}/photos/img_0001.jpg` | alternative to a video: raw JPEG body; idempotent, so a retry re-sends only missing names |
| PUT | `/jobs/{id}/motion` | `{"samples": [[t, alpha, beta, gamma], ...]}`: the phone's orientation while filming (s since recording was requested, radians, W3C angles), for plain objects |
| POST | `/jobs/{id}/start` | `{"flat_base": true, "video_bytes": 12345678, "quality": "detailed"}`; needs the complete video or ≥ 8 photos. `"quick"`: 48 frames instead of 80 and dense matching at 1440 px instead of 1920 (~10 instead of ~25 min). On a finished job: processes it again from its saved frames |
| GET | `/jobs/{id}/frames.zip` | the photos, or the frames picked from the video |
| GET | `/jobs/{id}` | `state`: uploading / queued / reconstructing / meshing / cleaning / done / failed, plus `step` (tool running now), `error`, `report`, `received`, `video_bytes` |
| GET | `/jobs/{id}/log?lines=60` | last lines of the job's processing log (plain text) |
| GET | `/jobs` | all jobs, newest first: `id`, `state`, `step`, `error`, `created`, `report` |
| GET | `/jobs/{id}/result.{stl,3mf,obj,glb}?longest_mm=120` | model scaled so its longest side is `longest_mm` (default 100) |
| POST | `/jobs/{id}/ar` | `{"longest_mm": 245}` → `{page, url, expires}`: signed links (6 h, no token needed) for the app's AR: `page` (`?title=&real=1`) shows the model in Chrome and places it in the room (model-viewer: WebXR, else Scene Viewer); `url` is the GLB, in metres, ≤ 60k triangles, textures ≤ 2048 px, gzipped (app/ar.py) |
| DELETE | `/jobs/{id}` | cancel and delete |
| GET | `/jobs/{id}/thumb.jpg` | square picture of the object (middle frame) |
| POST | `/auth/google` | `{"id_token": "..."}` → `{token, user}`; `POST /auth/logout` ends the session |
| GET / PATCH / DELETE | `/me` | profile with stats; `{"name": "..."}`; delete the account and its scans |
| PUT / DELETE / GET | `/me/avatar`, `/me/avatar.jpg` | own profile photo (JPEG ≤ 2 MB) |
| GET | `/me/scans` | the account's scans; `PATCH /me/scans/{id} {"title"}`, `POST/DELETE /me/scans/{id}/share` |
| POST | `/me/scans/claim` | `{"job_ids": [...]}`: scans made before signing in join the account |
| GET / PUT | `/admin/settings` | admin token only: `{"google_client_id"}`, database status |

Jobs without an account are deleted automatically after 7 days (`JOB_TTL_DAYS`).

## App

Requires Node 22, npm, and an Expo account (`npx eas-cli@latest login`).

```bash
npm install
cp .env.example .env.local      # EXPO_PUBLIC_API_URL = http://<laptop LAN IP>:8000
```

The app uses native modules (camera, sensors), so it needs a **development build**, not Expo Go:

```bash
npx eas-cli@latest build --profile development --platform android   # cloud build, installable APK
npx expo start                                                     # then open the dev build on the phone
```

Phone and laptop must be on the same Wi-Fi network for a LAN backend.

Checks:

```bash
npm run typecheck && npm run lint && npm test
```
