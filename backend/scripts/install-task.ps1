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
# Stopping the task can leave its server running (a run-server loop started another way, or the python it started),
# and that old server keeps the settings .env had when it started. End all of them so the new one reads .env again.
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
  Where-Object { $_.CommandLine -like '*run-server.ps1*' -and $_.ProcessId -ne $PID } |
  ForEach-Object { taskkill.exe /PID $_.ProcessId /T /F | Out-Null }
foreach ($c in Get-NetTCPConnection -LocalPort $env:PORT -State Listen -ErrorAction SilentlyContinue) {
  taskkill.exe /PID $c.OwningProcess /T /F | Out-Null
}
Start-Sleep 2
Register-ScheduledTask -TaskName 'scan3d' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'scan3d'
# Daily backup at 03:30 and a health check every 5 minutes (backup.ps1, watch.ps1), also as SYSTEM.
foreach ($t in @(@{ Name = 'scan3d-backup'; File = 'backup.ps1'; Trigger = (New-ScheduledTaskTrigger -Daily -At 3:30am) },
                 @{ Name = 'scan3d-watch'; File = 'watch.ps1'; Trigger = (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) `
                    -RepetitionInterval (New-TimeSpan -Minutes 5)) })) {
  $a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot $t.File)`""
  $st = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
  Register-ScheduledTask -TaskName $t.Name -Action $a -Trigger $t.Trigger -Principal $principal -Settings $st -Force | Out-Null
}
Write-Host 'Tasks "scan3d-backup" (daily 03:30) and "scan3d-watch" (every 5 minutes) registered.'
Write-Host 'Task "scan3d" registered and started. Waiting for the server...'

for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep 2
  try {
    $h = Invoke-RestMethod "http://127.0.0.1:$env:PORT/health"
    Write-Host "Server is up: http://127.0.0.1:$env:PORT/health (version $($h.version))" -ForegroundColor Green
    if ($h.mail) { Write-Host 'Mail for sign-up codes: ON' -ForegroundColor Green }
    else { Write-Host 'Mail for sign-up codes: OFF - add SMTP_HOST, SMTP_USER and SMTP_PASSWORD to .env, nobody can register without it' -ForegroundColor Yellow }
    exit 0
  } catch {}
}
throw "The server did not answer within 2 minutes. Look at $root\logs\server.log"
