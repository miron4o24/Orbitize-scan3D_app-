# Зарежда се от другите скриптове за Windows: . (Join-Path $PSScriptRoot 'env.ps1')
# Зарежда backend\.env в този процес и добавя в PATH инсталираните от setup-windows.ps1 COLMAP и OpenMVS.
$root = Split-Path $PSScriptRoot -Parent
$envFile = Join-Path $root '.env'
if (-not (Test-Path $envFile)) { throw "Missing $envFile. Run scripts\setup-windows.ps1 first." }
foreach ($line in Get-Content $envFile) {
  if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { Set-Item "env:$($Matches[1])" $Matches[2] }
}
if (-not $env:PORT) { $env:PORT = '8010' }
if (-not $env:DATA_DIR) { $env:DATA_DIR = Join-Path $root 'data' }

$colmap = Get-ChildItem (Join-Path $root 'tools\colmap') -Recurse -Filter colmap.exe -ErrorAction SilentlyContinue | Select-Object -First 1
$mvs = Get-ChildItem (Join-Path $root 'tools\openmvs') -Recurse -Filter DensifyPointCloud.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $colmap -or -not $mvs) { throw 'COLMAP or OpenMVS is missing. Run scripts\setup-windows.ps1 first.' }
$env:PATH = "$($colmap.DirectoryName);$($mvs.DirectoryName);$env:PATH"
$env:QT_PLUGIN_PATH = Join-Path (Split-Path $colmap.DirectoryName -Parent) 'plugins'
$python = Join-Path $root '.venv\Scripts\python.exe'
