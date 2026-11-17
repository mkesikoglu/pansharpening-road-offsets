# Reduced-scale grid-phase experiment: MATLAB degradation and fusion (with retries), Z-PNN, measurement, analysis
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -W ignore (Join-Path $root "scripts\49_rr_gridphase.py") list
Set-Location (Join-Path $root "matlab")
for ($i = 0; $i -lt 40; $i++) {
    & "C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "rr_gridphase('../data/processed/rr_gridphase_list.txt','../data/processed/rr_gridphase')"
    if ($LASTEXITCODE -eq 0) { break }
    Write-Output "MATLAB exited with $LASTEXITCODE, retry $i"
}
Set-Location $root
& $py -W ignore (Join-Path $root "scripts\49_rr_gridphase.py") zpnn
& $py -W ignore (Join-Path $root "scripts\49_rr_gridphase.py") measure
& $py -W ignore (Join-Path $root "scripts\49_rr_gridphase.py") analyse
