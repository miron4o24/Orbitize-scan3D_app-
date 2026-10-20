# Run the backend on Windows Server 2025 (public, always on)

Result: a public HTTPS address such as `https://myserver.tail1234.ts.net` that the app uses from any phone,
anywhere. Phones install nothing extra. The server needs no router port-forwarding and no domain name.

How it fits together: Docker runs inside WSL2 (Linux on Windows) → backend on `localhost:8000` →
Tailscale Funnel publishes it on the internet over HTTPS.

Recommended hardware: 4+ CPU cores, 8+ GB RAM, 20 GB free disk. More cores = faster scans.

---

## 1. Check the hardware (PowerShell)

```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object NumberOfLogicalProcessors, @{n='RAM_GB';e={[math]::Round($_.TotalPhysicalMemory/1GB)}}
```

## 2. Install WSL2 with Ubuntu (PowerShell as Administrator)

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when asked. After the reboot, an Ubuntu window opens and asks for a Linux user name and password.

Give WSL most of the RAM and cores (WSL uses only half the RAM by default). Create `C:\Users\<you>\.wslconfig`.
For example, on a 16 GB / 8-core server:

```ini
[wsl2]
memory=13GB
processors=8
```

## 3. Inside Ubuntu: enable systemd and install Docker

Open **Ubuntu** from the Start menu:

```bash
printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
```

In PowerShell run `wsl --shutdown`, then open Ubuntu again:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

Close and reopen Ubuntu, then check: `docker run --rm hello-world`.

## 4. Copy the backend to the server

On the development laptop, in the project folder:

```bash
git archive -o scan3d-backend.zip HEAD backend
```

Copy `scan3d-backend.zip` to the server, for example to `C:\scan3d\` (USB stick, RDP copy/paste, network share). Then in Ubuntu:

```bash
sudo apt-get install -y unzip
mkdir -p ~/scan3d && cd ~/scan3d
unzip -o /mnt/c/scan3d/scan3d-backend.zip
cd backend
cp .env.example .env
sed -i "s/^API_TOKEN=.*/API_TOKEN=$(openssl rand -hex 24)/" .env
nano .env    # MEM_LIMIT = WSL memory minus ~2g (e.g. 11g for 13GB); PORT=8000
```

## 5. Build and start (first build: 20–60 min, it compiles OpenMVS)

```bash
docker compose up -d --build
curl localhost:8000/health        # {"ok":true}
docker compose run --rm backend pytest -q    # all tests pass
```

Optional full test with real photos (~5–15 min):

```bash
scripts/fetch_dataset.sh
API_TOKEN=$(grep ^API_TOKEN .env | cut -d= -f2) scripts/e2e.sh samples/templeRing
```

## 6. Keep it running after reboots, without anyone logged in

WSL stops when no Windows program uses it. A startup task keeps it alive. Docker then starts automatically, and so does
the backend (`restart: unless-stopped`). Run in PowerShell as Administrator, replacing `<you>` with your Windows user
(it asks for that user's password):

```powershell
schtasks /create /tn "scan3d-wsl" /sc onstart /ru "$env:COMPUTERNAME\<you>" /rp * /rl highest `
  /tr "wsl.exe -d Ubuntu-24.04 --exec sleep infinity"
```

Test it: reboot the server, don't log in, and from another computer open `https://<your funnel address>/health` (step 7).

## 7. Publish it on the internet with Tailscale Funnel

1. Install Tailscale for Windows: https://tailscale.com/download/windows, and sign in (free account).
2. PowerShell as Administrator:

   ```powershell
   tailscale funnel --bg 8000
   ```

   The first time, it prints a link to enable HTTPS and Funnel for your account. Open it, approve, and run the command again.
3. It prints your public address, e.g. `https://myserver.tail1234.ts.net`. Open
   `https://myserver.tail1234.ts.net/health` on your phone **with Wi-Fi off**. It should show `{"ok":true}`.

Windows forwards `localhost:8000` into WSL automatically, so Funnel reaches the container.

## 8. Build the app for this server

Send the developer (or put into the EAS project settings):

- the Funnel address from step 7
- the `API_TOKEN` value from `~/scan3d/backend/.env`

Both are built into the preview APK.

---

## Security notes

- The API token is inside every copy of the app, so treat it as a speed bump, not a password. Anyone with the APK can
  use your server. The backend therefore limits stored jobs (`MAX_JOBS`, default 30), waiting jobs (`MAX_QUEUED`, 5)
  and photo size (15 MB), and deletes abandoned uploads after 24 h and finished jobs after 7 days.
- Uploaded photos are stored on your server until then.
- To shut the public address off: `tailscale funnel --bg off` (older Tailscale: `tailscale funnel reset`).

## Updating the backend later

Copy the new zip, unzip over `~/scan3d` (your `.env` is kept), then `docker compose up -d --build`.
