# Sets a new password for an e-mail account (same rule as registration) and signs it out everywhere.
# Usage (backend folder): powershell -File scripts\set-password.ps1 <e-mail> <password>
param([Parameter(Mandatory = $true)][string]$Email, [Parameter(Mandatory = $true)][string]$Password)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Push-Location (Split-Path $PSScriptRoot -Parent)
try { & $python -m app.set_password $Email $Password } finally { Pop-Location }
