# Reduced-scale fractional-phase experiment: MATLAB fusion (with retries), Z-PNN, measurement
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
Set-Location (Join-Path $root "matlab")
for ($i = 0; $i -lt 40; $i++) {
    & "C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "rr_fraction('../data/processed/rr_fraction_list.txt','../data/processed/rr_fraction')"
    if ($LASTEXITCODE -eq 0) { break }
    Write-Output "MATLAB exited with $LASTEXITCODE, retry $i"
}
Set-Location $root
& $py -W ignore (Join-Path $root "scripts\23_rr_fraction.py") zpnn
& $py -W ignore (Join-Path $root "scripts\23_rr_fraction.py") measure
