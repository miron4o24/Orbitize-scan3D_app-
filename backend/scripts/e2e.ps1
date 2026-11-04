# Тест от край до край през HTTP, точно като приложението: създаване на задача, качване, стартиране, следене, изтегляне.
# Употреба (папка backend): powershell -ExecutionPolicy Bypass -File scripts\e2e.ps1 <папка със снимки | video.mp4> [изходна папка] [адрес на сървъра]
param([Parameter(Mandatory = $true)][string]$Source, [string]$Out = 'e2e-out', [string]$Api)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
if (-not $Api) { $Api = "http://127.0.0.1:$env:PORT" }
$h = @{ Authorization = "Bearer $env:API_TOKEN" }

$id = (Invoke-RestMethod -Method Post "$Api/jobs" -Headers $h).id
Write-Host "job $id"
$item = Get-Item $Source
if ($item -is [IO.FileInfo]) {
  # Същите части от 8 MB като в приложението.
  $size = $item.Length
  $buf = New-Object byte[] 8MB
  $fs = [IO.File]::OpenRead($item.FullName)
  try {
    for ($off = 0; $off -lt $size; $off += $n) {
      $n = $fs.Read($buf, 0, $buf.Length)
      $part = New-Object byte[] $n
      [Array]::Copy($buf, $part, $n)
      Invoke-RestMethod -Method Put "$Api/jobs/$id/video?offset=$off" -Headers $h -ContentType 'application/octet-stream' -Body $part | Out-Null
    }
  } finally { $fs.Close() }
  Write-Host "uploaded video, $size bytes"
  $start = @{ flat_base = $true; video_bytes = $size }
} else {
  $photos = Get-ChildItem $item.FullName -Filter *.jpg
  foreach ($f in $photos) {
    Invoke-RestMethod -Method Put "$Api/jobs/$id/photos/$($f.Name)" -Headers $h -ContentType 'image/jpeg' -InFile $f.FullName | Out-Null
  }
  Write-Host "uploaded $($photos.Count) photos"
  $start = @{ flat_base = $true }
}
Invoke-RestMethod -Method Post "$Api/jobs/$id/start" -Headers $h -ContentType 'application/json' -Body ($start | ConvertTo-Json) | Out-Null

$clock = [Diagnostics.Stopwatch]::StartNew()
$last = ''
while ($true) {
  # Windows PowerShell 5.1 понякога губи повторно използвана връзка; приложението също просто опитва при следващата проверка.
  try { $status = Invoke-RestMethod "$Api/jobs/$id" -Headers $h -TimeoutSec 30 }
  catch { Write-Host "  (poll failed: $($_.Exception.Message); retrying)"; Start-Sleep 5; continue }
  $now = "$($status.state) $($status.step)".Trim()
  if ($now -ne $last) { Write-Host ('{0,5}s  {1}' -f [int]$clock.Elapsed.TotalSeconds, $now); $last = $now }
  if ($status.state -eq 'done') { break }
  if ($status.state -eq 'failed') {
    Write-Host (Invoke-RestMethod "$Api/jobs/$id/log?lines=40" -Headers $h)
    throw "failed: $($status.error)"
  }
  Start-Sleep 5
}

New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($fmt in 'stl', 'glb', '3mf') {
  Invoke-WebRequest "$Api/jobs/$id/result.$fmt`?longest_mm=100" -Headers $h -OutFile (Join-Path $Out "model.$fmt") -UseBasicParsing
}
Write-Host "report: $($status.report | ConvertTo-Json -Compress)"
Get-ChildItem $Out | Format-Table Name, Length
