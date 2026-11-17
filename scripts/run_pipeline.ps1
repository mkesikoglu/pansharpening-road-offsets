# Full processing chain after 09_build_jobs.py
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root
& (Join-Path $PSScriptRoot "11_run_matlab.ps1") -Stage rr
Set-Location $root
& $py -W ignore (Join-Path $root "scripts\10_run_zpnn.py")
if ($LASTEXITCODE -ne 0) { Write-Output "Z-PNN stage failed"; exit 1 }
& (Join-Path $PSScriptRoot "11_run_matlab.ps1") -Stage fuse
Set-Location $root
& $py -W ignore (Join-Path $root "scripts\12_measure.py")
