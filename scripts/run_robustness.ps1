# Second-review robustness experiments: side-dependent misregistration and intensity definitions
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$ml = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
& $py -W ignore (Join-Path $root "scripts\37_robustness.py") lists
Set-Location (Join-Path $root "matlab")
for ($i = 0; $i -lt 40; $i++) {
    & $ml -batch "rr_side('../data/processed/rr_side_list.txt','../data/processed/rr_side')"
    if ($LASTEXITCODE -eq 0) { break }
    Write-Output "MATLAB rr_side exited with $LASTEXITCODE, retry $i"
}
for ($i = 0; $i -lt 40; $i++) {
    & $ml -batch "fuse_full('../data/processed/intensity_list.txt','../data/processed/intensity')"
    if ($LASTEXITCODE -eq 0) { break }
    Write-Output "MATLAB fuse_full exited with $LASTEXITCODE, retry $i"
}
Set-Location $root
& $py -W ignore (Join-Path $root "scripts\37_robustness.py") zpnn_side
& $py -W ignore (Join-Path $root "scripts\37_robustness.py") measure_side
& $py -W ignore (Join-Path $root "scripts\37_robustness.py") zpnn_full
& $py -W ignore (Join-Path $root "scripts\37_robustness.py") intens
Write-Output "ROBUSTNESS DONE"
