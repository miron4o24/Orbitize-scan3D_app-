# Обновява уебсайта на IIS от GitHub: изтегля последната версия и слага папката website върху папката на сайта
# (по подразбиране D:\WEB_SITES\Scanly). Сайтът става точно като в GitHub: махнатите файлове изчезват. IIS не се
# рестартира: файловете се сменят на място и браузърите взимат новите при следващо отваряне.
# PowerShell като администратор: powershell -ExecutionPolicy Bypass -File scripts\update-website.ps1
param([string]$Site = "D:\WEB_SITES\Scanly", [string]$Branch = "main")
$ErrorActionPreference = "Stop"
$zip = Join-Path $env:TEMP "orbitize-website.zip"
$dir = Join-Path $env:TEMP "orbitize-website"

if (-not (Test-Path $Site)) { throw "There is no website folder at $Site. Give the right one: -Site D:\path\to\site" }
Write-Host "Downloading the latest website ($Branch)..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest "https://github.com/miron4o24/Orbitize-scan3D_app-/archive/refs/heads/$Branch.zip" -OutFile $zip -UseBasicParsing
if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
Expand-Archive $zip $dir
$source = Join-Path (Get-ChildItem $dir | Select-Object -First 1).FullName "website"
if (-not (Test-Path (Join-Path $source "index.html"))) { throw "The download has no website folder." }

# Копие на сегашния сайт до папката му, за връщане при нужда (последното копие се заменя).
$backup = "$Site.previous"
Write-Host "Keeping the current site in $backup ..."
robocopy $Site $backup /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Backing up the site failed (robocopy code $LASTEXITCODE)." }

Write-Host "Copying the new site to $Site ..."
robocopy $source $Site /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copying the site failed (robocopy code $LASTEXITCODE)." }
$global:LASTEXITCODE = 0
Remove-Item $zip -Force
Remove-Item $dir -Recurse -Force
Write-Host "Website updated: $Site" -ForegroundColor Green
