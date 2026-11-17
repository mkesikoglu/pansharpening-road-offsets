# Run MATLAB fusion on chips whose Z-PNN output already exists, repeatedly,
# until every job has a result file.
$root = Split-Path -Parent $PSScriptRoot
$jobs = Import-Csv (Join-Path $root "data/processed/jobs.csv")
$list = Join-Path $root "data/processed/chip_list_ready.txt"
while ($true) {
    $ready = $jobs | Where-Object {
        (Test-Path (Join-Path $root "data/processed/chips/$($_.job)_zpnn.mat")) -and
        -not (Test-Path (Join-Path $root "data/processed/fused/$($_.job)_res.mat")) }
    $done = ($jobs | Where-Object { Test-Path (Join-Path $root "data/processed/fused/$($_.job)_res.mat") }).Count
    if ($done -ge $jobs.Count) { break }
    if (-not $ready) { Start-Sleep 120; continue }
    ($ready | ForEach-Object { "../data/processed/chips/$($_.job).mat" }) | Set-Content $list -Encoding ascii
    Set-Location (Join-Path $root "matlab")
    & "C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "fuse_and_eval('../data/processed/chip_list_ready.txt','../data/processed/fused')"
    Write-Output "fused batch, exit $LASTEXITCODE, done before batch: $done"
}
