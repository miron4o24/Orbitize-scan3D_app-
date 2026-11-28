# Health check every 5 minutes (task "scan3d-watch", registered by install-task.ps1): is the server answering here and
# at PUBLIC_URL, and is there disk space? One e-mail to OWNER_EMAILS when something breaks and one when it is fixed.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Push-Location $root
try { & $python -m app.watchdog } finally { Pop-Location }
