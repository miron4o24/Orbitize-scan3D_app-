# Еднократна инсталация на сървъра под Windows Server 2025 / Windows 11, без WSL и Docker:
# Visual C++, COLMAP и OpenMVS (за процесор), Python пакети, .env. Може да се изпълнява повторно.
# PowerShell като администратор, в папка backend:
#   powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue' # лентата за напредък прави Invoke-WebRequest около 10 пъти по-бавен
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = Split-Path $PSScriptRoot -Parent
$tools = Join-Path $root 'tools'
$COLMAP_URL = 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-nocuda.zip'
$OPENMVS_URL = 'https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip'

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)." } }

Step 'Python'
$python = @("$env:ProgramFiles\Python313\python.exe", "$env:ProgramFiles\Python312\python.exe") |
  Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python -and (Get-Command py.exe -ErrorAction SilentlyContinue)) {
  foreach ($v in '3.13', '3.12') {
    $found = & py.exe "-$v" -c 'import sys; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $found) { $python = $found.Trim(); break }
  }
}
if (-not $python) {
  throw 'Python 3.13 (or 3.12) not found. Install it from python.org (see the guide) and run this again.'
}
& $python --version

Step 'Visual C++ runtime (COLMAP and OpenMVS need it)'
$vc = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
if ($vc -and $vc.Installed -eq 1 -and $vc.Major -eq 14 -and $vc.Minor -ge 50) {
  Write-Host "Already installed ($($vc.Version))"
} else {
  $redist = Join-Path $env:TEMP 'vc_redist.x64.exe'
  Invoke-WebRequest 'https://aka.ms/vc14/vc_redist.x64.exe' -OutFile $redist -UseBasicParsing
  $p = Start-Process $redist -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
  # 0 installed, 1638 a newer one is already there, 3010 installed but wants a reboot
  if ($p.ExitCode -notin 0, 1638, 3010) { throw "Visual C++ runtime install failed (exit code $($p.ExitCode))." }
  Write-Host 'OK'
}

function Install-Tool($name, $url, $exe) {
  $dir = Join-Path $tools $name
  if (Get-ChildItem $dir -Recurse -Filter $exe -ErrorAction SilentlyContinue) { Write-Host 'Already installed'; return }
  $zip = Join-Path $env:TEMP "$name.zip"
  Write-Host "Downloading $url"
  Invoke-WebRequest $url -OutFile $zip -UseBasicParsing
  Expand-Archive $zip $dir -Force
  Remove-Item $zip
  Write-Host 'OK'
}
Step 'COLMAP 4.2.1 (finds the camera positions)'
Install-Tool 'colmap' $COLMAP_URL 'colmap.exe'
Step 'OpenMVS 2.4.0 (dense 3D points)'
Install-Tool 'openmvs' $OPENMVS_URL 'DensifyPointCloud.exe'

. (Join-Path $PSScriptRoot 'models.ps1')
Install-Models

Step 'Python packages (a few minutes)'
$venv = Join-Path $root '.venv'
if (-not (Test-Path "$venv\Scripts\python.exe")) { & $python -m venv $venv; Check 'Creating the virtual environment' }
& "$venv\Scripts\python.exe" -m pip install --upgrade --quiet pip; Check 'Upgrading pip'
& "$venv\Scripts\python.exe" -m pip install --quiet -r (Join-Path $root 'requirements.txt'); Check 'Installing packages'
Write-Host 'OK'

Step 'Settings (.env)'
$envFile = Join-Path $root '.env'
if (Test-Path $envFile) {
  Write-Host "Keeping the existing $envFile"
} else {
  $bytes = New-Object byte[] 24
  [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  $token = -join ($bytes | ForEach-Object { $_.ToString('x2') })
  $text = "API_TOKEN=$token`nPORT=8010`nMAX_IMAGE_SIZE=1600`nMAX_VIDEO_MB=600`n"
  [IO.File]::WriteAllText($envFile, $text) # UTF-8 без BOM
  Write-Host "Created $envFile with a new API_TOKEN. Put the same token in the app's .env.local." -ForegroundColor Yellow
}

Step 'Self-test'
. (Join-Path $PSScriptRoot 'env.ps1')
& colmap.exe version; Check 'COLMAP'
Push-Location $root
try { & $python -m pytest -q -p no:cacheprovider; Check 'Tests' } finally { Pop-Location }

Write-Host "`nSetup complete. Next: scripts\install-task.ps1 (see the guide)." -ForegroundColor Green
