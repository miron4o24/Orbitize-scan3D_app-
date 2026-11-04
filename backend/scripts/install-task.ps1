# Регистрира планираната задача "orbitize": стартира сървъра при включване като SYSTEM (без влизане в системата), после го стартира
# веднага и проверява /health. Изпълнете отново след преместване на папката. Премахване: Unregister-ScheduledTask orbitize -Confirm:$false
# PowerShell като администратор: powershell -ExecutionPolicy Bypass -File scripts\install-task.ps1
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$run = Join-Path $PSScriptRoot 'run-server.ps1'

# Обновяване може да изисква Python пакети, които сървърът още няма (бързо е, когато ги има всички).
Write-Host 'Checking Python packages...'
& $python -m pip install --quiet --disable-pip-version-check -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Installing Python packages failed (exit code $LASTEXITCODE)." }

# Обновяване може да изисква AI модел, който сървърът още няма. Без него сървърът пак работи (пропуска AI дълбочината).
. (Join-Path $PSScriptRoot 'models.ps1')
try { Install-Models } catch { Write-Host "Model download failed: $_. Scans still work; run this again later." -ForegroundColor Yellow }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$run`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
# Без ограничение на времето; нормален приоритет (задачите по подразбиране са под нормалния); рестарт при неуспешно стартиране.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -Priority 4 -StartWhenAvailable `
  -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
# Задачата със старото име отпреди преименуването на приложението.
if (Get-ScheduledTask -TaskName 'scan3d' -ErrorAction SilentlyContinue) {
  Stop-ScheduledTask -TaskName 'scan3d' -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName 'scan3d' -Confirm:$false
}
Stop-ScheduledTask -TaskName 'orbitize' -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName 'orbitize' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'orbitize'
Write-Host 'Task "orbitize" registered and started. Waiting for the server...'

for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep 2
  try {
    Invoke-RestMethod "http://127.0.0.1:$env:PORT/health" | Out-Null
    Write-Host "Server is up: http://127.0.0.1:$env:PORT/health" -ForegroundColor Green
    exit 0
  } catch {}
}
throw "The server did not answer within 2 minutes. Look at $root\logs\server.log"
