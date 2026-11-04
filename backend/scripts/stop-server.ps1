# Спира сървъра: планираната задача и всичко, което още слуша на $PORT, с инструментите, които е стартирало.
# PowerShell като администратор: powershell -ExecutionPolicy Bypass -File scripts\stop-server.ps1
. (Join-Path $PSScriptRoot 'env.ps1')
Stop-ScheduledTask -TaskName 'orbitize' -ErrorAction SilentlyContinue
foreach ($c in Get-NetTCPConnection -LocalPort $env:PORT -State Listen -ErrorAction SilentlyContinue) {
  # /T: и процесът на COLMAP/OpenMVS на текущо сканиране.
  taskkill.exe /PID $c.OwningProcess /T /F | Out-Null
}
Write-Host 'Stopped.'
