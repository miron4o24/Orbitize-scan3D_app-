# Runs the backend on http://127.0.0.1:$PORT (Tailscale Funnel publishes it). The "scan3d" scheduled task runs this
# at boot; it restarts the server if it ever exits. -Console: run once in this window with the log on screen.
param([switch]$Console)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location $root
$env:PYTHONUNBUFFERED = '1'
$uvicorn = "`"$python`" -m uvicorn app.main:app --host 127.0.0.1 --port $env:PORT"

if ($Console) {
  cmd /c $uvicorn
  exit $LASTEXITCODE
}

# A stopped task can leave the previous server running (and holding the port): end it and the tools it started.
foreach ($c in Get-NetTCPConnection -LocalPort $env:PORT -State Listen -ErrorAction SilentlyContinue) {
  taskkill.exe /PID $c.OwningProcess /T /F | Out-Null
}

$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs 'server.log'
while ($true) {
  if ((Test-Path $log) -and (Get-Item $log).Length -gt 50MB) { Move-Item $log "$log.old" -Force }
  Add-Content $log "`r`n=== starting $(Get-Date -Format s) ===" -Encoding UTF8
  # cmd does the redirect: PowerShell 5.1 would write the log as UTF-16.
  cmd /c "$uvicorn >> `"$log`" 2>&1"
  Add-Content $log "=== exited with code $LASTEXITCODE, restarting in 10 s ===" -Encoding UTF8
  Start-Sleep 10
}
