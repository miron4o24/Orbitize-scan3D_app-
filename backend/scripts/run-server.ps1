# Стартира сървъра на http://127.0.0.1:$PORT (Tailscale Funnel го публикува). Планираната задача "orbitize" го изпълнява
# при включване и го рестартира, ако спре. -Console: изпълнение веднъж в този прозорец с дневник на екрана.
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

# Спряна задача може да остави предишния сървър да работи (и да държи порта): спира се заедно със стартираните от него инструменти.
foreach ($c in Get-NetTCPConnection -LocalPort $env:PORT -State Listen -ErrorAction SilentlyContinue) {
  taskkill.exe /PID $c.OwningProcess /T /F | Out-Null
}

$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs 'server.log'
while ($true) {
  if ((Test-Path $log) -and (Get-Item $log).Length -gt 50MB) { Move-Item $log "$log.old" -Force }
  Add-Content $log "`r`n=== starting $(Get-Date -Format s) ===" -Encoding UTF8
  # cmd прави пренасочването: PowerShell 5.1 би записал дневника в UTF-16.
  cmd /c "$uvicorn >> `"$log`" 2>&1"
  Add-Content $log "=== exited with code $LASTEXITCODE, restarting in 10 s ===" -Encoding UTF8
  Start-Sleep 10
}
