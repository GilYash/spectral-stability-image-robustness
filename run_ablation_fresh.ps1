# run_ablation_fresh_fixed.ps1
# Fresh 2x2 ablation:
#   fixed x grid
#   fixed x CE
#   geometric x grid
#   geometric x CE
#
# Uses NEW cache/result directories so old work is not overwritten.

# Do not use "Stop" here: Windows PowerShell can treat native-program stderr
# (for example Python warnings/progress output) as PowerShell errors even when
# the program itself is healthy. We check each Python process exit code instead.
$ErrorActionPreference = "Continue"

$Python = ".\.venv\Scripts\python.exe"
$FixedCache = "cache_fixed_fresh"
$GeometricCache = "cache_geometric_fresh"
$Results = "results_ablation_fresh"
$LogDir = "logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $Results | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "ablation_fresh_$Stamp.log"

function Run-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "============================================================"
    Write-Host "$Python $($CommandArgs -join ' ')"

    # IMPORTANT: use CommandArgs, not Args. $args is a PowerShell automatic
    # variable and using it as our parameter caused Python to receive no args.
    & $Python @CommandArgs 2>&1 | Tee-Object -FilePath $Log -Append

    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        Write-Host ""
        Write-Host "FAILED: $Name (exit code $ExitCode)" -ForegroundColor Red
        Write-Host "See log: $Log"
        exit $ExitCode
    }

    Write-Host "Completed: $Name" -ForegroundColor Green
}

Write-Host "Fresh 2x2 ablation"
Write-Host "Fixed cache:     $FixedCache"
Write-Host "Geometric cache: $GeometricCache"
Write-Host "Results:         $Results"
Write-Host "Log:             $Log"
Write-Host ""
Write-Host "Existing old cache/results directories are NOT deleted or overwritten."

# 1. Verify the new CE implementation before spending GPU time.
Run-Step -Name "1/4 - Unit tests" -CommandArgs @(
    "-m", "pytest", "-q",
    "test\test_analysis.py",
    "test\test_ce_fusion.py"
)

# 2. Rebuild the fixed ladder from scratch in a fresh directory.
Run-Step -Name "2/4 - Build fresh fixed-ladder cache" -CommandArgs @(
    "scripts\cache_profiles.py",
    "--ladder", "fixed",
    "--cache-dir", $FixedCache
)

# 3. Build the geometric ladder from scratch.
Run-Step -Name "3/4 - Build fresh geometric-ladder cache" -CommandArgs @(
    "scripts\cache_profiles.py",
    "--ladder", "geometric",
    "--cache-dir", $GeometricCache
)

# 4. Run all four ablation cells:
#    fixed/grid, fixed/CE, geometric/grid, geometric/CE
#    for both ResNet-18 and DeiT-Tiny.
Run-Step -Name "4/4 - Tune, evaluate, and summarize the 2x2 ablation" -CommandArgs @(
    "scripts\run_ablation.py",
    "--fixed-cache", $FixedCache,
    "--geometric-cache", $GeometricCache,
    "--results", $Results,
    "--force-tuning"
)

Write-Host ""
Write-Host "============================================================"
Write-Host "DONE"
Write-Host "============================================================"
Write-Host "Results are in: $Results"
Write-Host "Full console log: $Log"
Write-Host ""
Write-Host "Look for:"
Write-Host "  $Results\ablation_2x2_*.csv"
Write-Host "  $Results\ablation_2x2_*.json"
