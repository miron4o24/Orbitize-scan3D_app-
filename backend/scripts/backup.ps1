# Daily backup (task "scan3d-backup", registered by install-task.ps1): accounts, profile pictures and finished models
# into backend\backups (BACKUP_DIR in .env), keeping the last 7 (BACKUP_KEEP). Run by hand any time:
#   powershell -ExecutionPolicy Bypass -File scripts\backup.ps1
# Restore (stop the server first: Stop-ScheduledTask scan3d):
#   & .venv\Scripts\python.exe -m app.backup --restore backups\orbitize-<date>.zip
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Push-Location $root
try {
  $out = & $python -m app.backup 2>&1
  Add-Content (Join-Path $root 'logs\backup.log') "$(Get-Date -Format s) $out" -Encoding UTF8
  if ($LASTEXITCODE -ne 0) { throw "Backup failed: $out" }
  Write-Host $out
} finally { Pop-Location }
