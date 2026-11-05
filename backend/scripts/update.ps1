# Обновява сървъра от GitHub: изтегля последната версия, слага я върху backend и го рестартира. Папките .venv, tools,
# data и файлът .env не са в хранилището, затова остават непокътнати. Стартирайте го в PowerShell като администратор,
# когато не тече сканиране: .\scripts\update.ps1
$ErrorActionPreference = "Stop"
$backend = Split-Path -Parent $PSScriptRoot
$zip = Join-Path $env:TEMP "orbitize-update.zip"
$dir = Join-Path $env:TEMP "orbitize-update"

Write-Host "Downloading the latest version..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest "https://github.com/miron4o24/Orbitize-scan3D_app-/archive/refs/heads/main.zip" -OutFile $zip -UseBasicParsing
if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
Expand-Archive $zip $dir
$source = Join-Path (Get-ChildItem $dir | Select-Object -First 1).FullName "backend"
if (-not (Test-Path (Join-Path $source "app\main.py"))) { throw "The download has no backend folder." }

Write-Host "Copying the files..."
robocopy $source $backend /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copying failed (robocopy code $LASTEXITCODE)." }
$global:LASTEXITCODE = 0
Remove-Item $zip -Force
Remove-Item $dir -Recurse -Force

Write-Host "Restarting the server..."
& (Join-Path $PSScriptRoot "install-task.ps1")
