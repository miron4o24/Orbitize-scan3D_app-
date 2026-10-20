# 3D Scan: phone camera to printable 3D model

Walk around an object taking photos with the app. A self-hosted backend reconstructs it with
photogrammetry and returns a cleaned, watertight mesh (STL / 3MF / OBJ / GLB).

- `src/`: Expo app (SDK 57, expo-router, TypeScript). Capture, upload, preview, export.
- `backend/`: FastAPI + COLMAP (camera poses) + OpenMVS (dense points) + PyMeshLab/trimesh (mesh cleanup), in Docker. CPU only.

## Honest limits

- **Works well:** matte, textured objects about 5–50 cm, in even light, not moving.
- **Fails or comes out poor:** shiny, transparent or plain single-colour objects, thin parts (wires, hair), tiny details under ~1 mm.
- **Scale:** photogrammetry has no absolute size. You enter the real length of the object's longest side.
- **Processing time (CPU):** minutes to tens of minutes per scan, depending on photo count, resolution and cores. A GPU would be several times faster but is not free. Measured numbers are in "Processing times" below.

## Processing times

| Dataset | Machine | Time |
|---|---|---|
| Middlebury templeRing, 47 photos 640×480 | 8-core laptop, CPU, Docker | 279 s (47/47 photos placed) |

## Backend

Requires Docker. **Production setup on a Windows server, public over HTTPS: [docs/windows-server.md](docs/windows-server.md).**

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
```

### Unit tests

```bash
docker compose run --rm backend pytest -q            # in the image
# or locally: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pytest -q
```

### API

All endpoints except `/health` need `Authorization: Bearer $API_TOKEN`.

| Method | Path | |
|---|---|---|
| POST | `/jobs` | create job, returns `{id, state: "uploading", received: []}` |
| PUT | `/jobs/{id}/photos/img_0001.jpg` | raw JPEG body; idempotent, so a retry re-sends only missing names |
| POST | `/jobs/{id}/start` | `{"flat_base": true}`; needs ≥ 8 photos |
| GET | `/jobs/{id}` | `state`: uploading / queued / reconstructing / meshing / cleaning / done / failed, plus `error`, `report` |
| GET | `/jobs/{id}/result.{stl,3mf,obj,glb}?longest_mm=120` | model scaled so its longest side is `longest_mm` (default 100) |
| DELETE | `/jobs/{id}` | cancel and delete |

Jobs are deleted automatically after 7 days (`JOB_TTL_DAYS`).

## App

Requires Node 22, npm, and an Expo account (`npx eas-cli@latest login`).

```bash
npm install
cp .env.example .env.local      # EXPO_PUBLIC_API_URL = http://<laptop LAN IP>:8000, EXPO_PUBLIC_API_TOKEN = backend API_TOKEN
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
