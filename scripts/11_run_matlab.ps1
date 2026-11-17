# Run a MATLAB stage over all chips, retrying when MATLAB exits abnormally
# (the R2025a batch session occasionally crashes at start-up; finished chips are skipped).
param([string]$Stage = "fuse")   # "rr" or "fuse"
$root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $root "matlab")
$jobs = Import-Csv (Join-Path $root "data/processed/jobs.csv")
$list = Join-Path $root "data/processed/chip_list.txt"
($jobs | ForEach-Object { "../data/processed/chips/$($_.job).mat" }) | Set-Content $list -Encoding ascii
$cmd = if ($Stage -eq "rr") { "make_rr('../data/processed/chip_list.txt')" } else { "fuse_and_eval('../data/processed/chip_list.txt','../data/processed/fused')" }
for ($i = 0; $i -lt 30; $i++) {
    & "C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch $cmd
    if ($LASTEXITCODE -eq 0) { break }
    Write-Output "MATLAB exited with $LASTEXITCODE, retry $i"
}
