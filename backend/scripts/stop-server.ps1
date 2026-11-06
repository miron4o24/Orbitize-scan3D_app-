# Stops the backend: the scheduled task and whatever is still listening on $PORT, with any tool it started.
# PowerShell as Administrator: powershell -ExecutionPolicy Bypass -File scripts\stop-server.ps1
. (Join-Path $PSScriptRoot 'env.ps1')
Stop-ScheduledTask -TaskName 'scan3d' -ErrorAction SilentlyContinue
foreach ($c in Get-NetTCPConnection -LocalPort $env:PORT -State Listen -ErrorAction SilentlyContinue) {
  # /T: also the COLMAP/OpenMVS process of a scan in progress.
  taskkill.exe /PID $c.OwningProcess /T /F | Out-Null
}
Write-Host 'Stopped.'
