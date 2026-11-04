# Registers the "scan3d" scheduled task: starts the backend at boot as SYSTEM (nobody needs to log in), then starts
# it now and checks /health. Run again after moving the folder. Remove: Unregister-ScheduledTask scan3d -Confirm:$false
# PowerShell as Administrator: powershell -ExecutionPolicy Bypass -File scripts\install-task.ps1
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$run = Join-Path $PSScriptRoot 'run-server.ps1'

# An update may need Python packages the server does not have yet (quick when it has them all).
Write-Host 'Checking Python packages...'
& $python -m pip install --quiet --disable-pip-version-check -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Installing Python packages failed (exit code $LASTEXITCODE)." }

# An update may need an AI model the server does not have yet. Without it the server still runs (it skips AI depth).
. (Join-Path $PSScriptRoot 'models.ps1')
try { Install-Models } catch { Write-Host "Model download failed: $_. Scans still work; run this again later." -ForegroundColor Yellow }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$run`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
# No time limit; normal priority (tasks default to below normal); restart if it fails to start.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -Priority 4 -StartWhenAvailable `
  -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Stop-ScheduledTask -TaskName 'scan3d' -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName 'scan3d' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'scan3d'
Write-Host 'Task "scan3d" registered and started. Waiting for the server...'

for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep 2
  try {
    Invoke-RestMethod "http://127.0.0.1:$env:PORT/health" | Out-Null
    Write-Host "Server is up: http://127.0.0.1:$env:PORT/health" -ForegroundColor Green
    exit 0
  } catch {}
}
throw "The server did not answer within 2 minutes. Look at $root\logs\server.log"
