# Run the backend on Windows Server 2025 (no WSL, no Docker)

Result: a public HTTPS address such as `https://myserver.tail1234.ts.net` that the app uses from any phone, anywhere.
Everything runs as normal Windows programs. No router port-forwarding, no domain name, no Linux.

```
Phone app ──HTTPS──> Tailscale Funnel ──> 127.0.0.1:8010 ──> Python server (scheduled task "scan3d")
                                                                 ├─ ffmpeg   (picks frames from the video)
                                                                 ├─ COLMAP   (finds where the camera was)
                                                                 ├─ OpenMVS  (dense 3D points)
                                                                 └─ IS-Net   (cuts the object out of each frame) -> mesh
```

Time needed: about 30 minutes.

What you need:

- Windows Server 2025 (Windows 11 works too), with an administrator account. Remote Desktop is fine.
- 4+ CPU cores, 8+ GB RAM (16 GB recommended), 20 GB free disk. More cores = faster scans.
- Internet access on the server (the setup downloads about 600 MB).
- `scan3d-backend.zip` from the development PC (`C:\3D\scan3d-backend.zip`). It contains `backend\.env` with the
  API token that the app already uses.
- A free Tailscale account (sign in with Google, Microsoft or GitHub).

"**PowerShell (Admin)**" below means: Start → type `PowerShell` → right-click *Windows PowerShell* →
*Run as administrator*.

---

## 1. Copy the backend to the server

1. Copy `scan3d-backend.zip` to the server, e.g. via Remote Desktop copy/paste, a USB stick or a network share.
2. Right-click it → *Extract All…* → type `C:\scan3d` → *Extract*.
3. Check that `C:\scan3d\backend\app\main.py` and `C:\scan3d\backend\.env` exist.

`.env` is a hidden-looking file without a name before the dot. Open it with Notepad to see the settings:

```
API_TOKEN=…     # the server's admin password; for your own tools only, never put it in the app
PORT=8010           # local port of the server
MAX_IMAGE_SIZE=1600 # lower (e.g. 1200) = faster and less RAM, less detail
```

## 2. Install Python 3.13

1. On the server, open https://www.python.org/downloads/windows/ and download the latest **Python 3.13.x –
   Windows installer (64-bit)**.
2. Run it:
   - tick **Add python.exe to PATH** at the bottom,
   - click **Customize installation** → *Next*,
   - tick **Install Python 3.13 for all users** (it then installs to `C:\Program Files\Python313`),
   - click *Install*, then *Close*.

## 3. Run the setup script

**PowerShell (Admin):**

```powershell
cd C:\scan3d\backend
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
```

It takes 5–10 minutes and does everything else:

| Step | What |
|---|---|
| Visual C++ runtime | Microsoft runtime that COLMAP and OpenMVS need (skipped if already installed) |
| COLMAP 4.2.1 | official Windows CPU build, into `backend\tools\colmap` |
| OpenMVS 2.4.0 | official Windows build, into `backend\tools\openmvs` |
| AI object finder | 180 MB, into `backend\tools\seg` (cuts the object out of each frame) |
| AI depth model | 100 MB, into `backend\tools\depth` (shape of plain and shiny surfaces) |
| Python packages | into `backend\.venv` (includes ffmpeg) |
| `.env` | kept as it is; only created (with a new random token) if missing |
| Self-test | runs COLMAP and the backend's tests |

It must end with `NN passed` and a green **Setup complete**. If it stops with a red error, see Troubleshooting.

## 4. Start it, now and at every boot

**PowerShell (Admin)**, still in `C:\scan3d\backend`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-task.ps1
```

This creates the scheduled task **scan3d**. It starts the server at every boot as SYSTEM, without anybody logging
in, and restarts the server if it ever crashes. The script ends with a green `Server is up`.

Check in a browser on the server: http://127.0.0.1:8010/health shows `{"ok":true}`.

### Optional: a full test with real photos (5–15 minutes)

The zip contains a 47-photo sample. This sends it through the server exactly like the app does:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\e2e.ps1 samples\templeRing
```

It prints each stage (`queued`, `reconstructing`, `meshing`, `cleaning`, `done`) and saves the model to
`C:\scan3d\backend\e2e-out\model.stl` (open it in any slicer, e.g. PrusaSlicer or Cura).

To test with a phone video instead: copy it to the server and run `scripts\e2e.ps1 C:\path\to\video.mp4`.

## 5. Make it reachable from the internet with Tailscale Funnel

1. Install Tailscale for Windows: https://tailscale.com/download/windows. Sign in when it asks.
2. **PowerShell (Admin):**

   ```powershell
   tailscale up --unattended
   tailscale funnel --bg 8010
   ```

   - `--unattended` keeps Tailscale connected when nobody is logged in to the server. Without it, the address stops
     working when you log out of Remote Desktop.
   - The first time, `funnel` prints a link to enable HTTPS and Funnel for your account. Open it, approve, and run
     `tailscale funnel --bg 8010` again.
3. It prints your public address, e.g. `https://myserver.tail1234.ts.net`. Write it down.
4. On your phone, **with Wi-Fi off**, open `https://myserver.tail1234.ts.net/health`. It must show `{"ok":true}`.

## 6. Reboot test

Restart the server, **don't log in**, wait 3 minutes, then open the `/health` address on the phone again. If it shows
`{"ok":true}`, everything starts on its own after power cuts and Windows updates.

## 7. Point the app at the server

On the development PC, open `C:\3D\.env.local` and replace the placeholder with your Funnel address (no slash at the
end):

```
EXPO_PUBLIC_API_URL=https://myserver.tail1234.ts.net
```

Then restart the dev server so the app picks it up: `npx expo start --clear`, and reload the app on the phone.
The home screen footer shows the server address it uses.

For an installable preview APK (`eas build --profile preview`), `.env.local` is not used: add both variables on
expo.dev → your project → *Environment variables*, environment **preview**.

---

## Everyday use

All in **PowerShell (Admin)** in `C:\scan3d\backend`, prefix each script with `powershell -ExecutionPolicy Bypass -File`.

| What | How |
|---|---|
| Is it running? | browser: http://127.0.0.1:8010/health |
| Server log | `C:\scan3d\backend\logs\server.log` (Notepad) |
| Log of one scan | `C:\scan3d\backend\data\<job id>\log.txt` (the job id is in the server log) |
| Frames picked from a video | `C:\scan3d\backend\data\<job id>\photos\` |
| Restart | `scripts\install-task.ps1` (stops, starts and checks it) |
| Stop | `scripts\stop-server.ps1` (it starts again at the next boot) |
| Run in a window to watch it | `scripts\stop-server.ps1`, then `scripts\run-server.ps1 -Console` (Ctrl+C to stop) |
| Remove completely | `scripts\stop-server.ps1`, `Unregister-ScheduledTask scan3d -Confirm:$false`, `tailscale funnel --bg off`, delete `C:\scan3d` |

Scans and their files are deleted automatically 7 days after they finish (abandoned uploads after 24 hours).

## Updating the backend later

1. On the development PC, in PowerShell in `C:\3D`, rebuild the zip:

   ```powershell
   tar.exe -a -c -f scan3d-backend.zip --exclude=.venv --exclude=.pytest_cache --exclude=__pycache__ --exclude=e2e-out --exclude=data --exclude=tools --exclude=logs backend
   ```

2. On the server: *Extract All* into the folder the server runs from (the one that contains `backend\`; `C:\scan3d`
   if you followed this guide) and choose **Replace** for the files. (If you changed `backend\.env` on the server,
   choose *Skip* for that one file.) Extracting anywhere else changes nothing: the scheduled task keeps running the
   old copy. A scan's log (`GET /jobs/<id>/log`) shows the folder it ran from on every tool line.
3. **PowerShell (Admin)** in that folder's `backend`:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
   powershell -ExecutionPolicy Bypass -File scripts\install-task.ps1
   ```

Scans that were in progress restart automatically.

## Troubleshooting

- **App: "Can't reach the server"**: open `<funnel address>/health` on the phone. If it fails but
  http://127.0.0.1:8010/health works on the server, the problem is Tailscale: run `tailscale funnel status`. If both
  fail, look at `logs\server.log` and run `scripts\install-task.ps1`.
- **Setup: "Python 3.13 (or 3.12) not found"**: install Python as in step 2, *for all users*, and close and reopen
  PowerShell.
- **Setup: "running scripts is disabled on this system"**: you ran the script directly. Use the full
  `powershell -ExecutionPolicy Bypass -File ...` command.
- **Setup: download errors**: the server needs to reach github.com and aka.ms. Check proxy/firewall, then run setup
  again; it continues where it stopped.
- **Scan fails with "ran out of memory"**: set `MAX_IMAGE_SIZE=1200` in `.env` and run `scripts\install-task.ps1`.
  Closing other programs on the server also helps.
- **Scan fails with "The object could not be told apart from the background"**: the table or background had the
  object's colour, or other things stood right next to it. Film again on a surface of a different colour.
- **Setup: "Segmentation model missing" or "model download is damaged"**: run `scripts\setup-windows.ps1` again.
- **Scan fails with "Only N of M frames could be placed in 3D"**: the video was too fast or blurry, or the object has
  too little texture. Look at the frames the server picked (table above). In the app, *Show server details* shows
  the end of the job's `log.txt` (same folder), with a timestamp for every step.
- **Scans are slow**: optional, makes Windows Defender skip the scan data:
  `Add-MpPreference -ExclusionPath C:\scan3d\backend\data`.

## Security notes

- The API token is inside every copy of the app, so treat it as a speed bump, not a password. Anyone with the APK can
  use your server. The backend therefore limits stored jobs (`MAX_JOBS`, default 30), waiting jobs (`MAX_QUEUED`, 5),
  photo size (15 MB) and video size (600 MB, `MAX_VIDEO_MB`). A finished job's video is deleted right away.
- The server only listens on 127.0.0.1, so it is reachable from outside only through Tailscale Funnel. No Windows
  Firewall changes are needed.
- To take it offline: `tailscale funnel --bg off`.

## Push notifications and Google Play (optional)

The app and server work without these; the feature just stays off.

**"Your model is ready" notifications** (free):

1. In the [Firebase console](https://console.firebase.google.com), create a project and add an Android app with the
   package `com.miron4o.scan3d`. Download `google-services.json` and put it in the app's project folder.
2. In Firebase, open Project settings > Service accounts and generate a new private key.
3. On expo.dev, open the project's Credentials page, select the Android app, and under FCM V1 service account key upload
   that file. Never put it in the project folder.
4. Make a new build.

**Pro through Google Play**:

1. In Google Play Console, add a subscription with the product ID `scanly_pro_monthly` (or set `PRO_PRODUCT_ID` in the
   server's `.env`). The app must be uploaded to a Play testing track for purchases to work.
2. In Google Cloud, create a service account, download its key, and give it access to the app in Play Console (financial
   data and orders). Keep the file on the server, outside the zip's folder.
3. In the server's `.env`, add `GOOGLE_PLAY_KEY_FILE=C:\path\to\the-key.json`, then run `scripts\install-task.ps1`.
