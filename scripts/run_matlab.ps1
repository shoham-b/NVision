<#
.SYNOPSIS
    Run the SBED locator on every real ESR .mat file in data/matlab/.

.DESCRIPTION
    Runs `nv matlab-run --all`, which processes every .mat file in data/matlab/
    (or -MatlabDir) in turn. Each run:
      * writes a self-contained result bundle to artifacts/matlab_<stem>/
      * registers the run in the MATLAB-only cache at artifacts/matlab/cache/

    View them with:  uv run nv serve --dir artifacts/matlab
    then pick generator "MATLAB:<file>", noise "real", strategy "Bayesian-SBED".

    That cache is deliberately separate from the shared artifacts/cache/, which also
    holds the simulated grid — on a large simulation cache the shared manifest takes
    many minutes to build, whereas this one loads in about a second.

.EXAMPLE
    pwsh scripts/run_matlab.ps1
    pwsh scripts/run_matlab.ps1 -MaxSteps 300
    pwsh scripts/run_matlab.ps1 -Serve
#>
param(
    [int]$MaxSteps = 300,
    [switch]$Serve,
    [string]$MatlabDir
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $dirArgs = @()
    if ($MatlabDir) { $dirArgs = @("--dir", $MatlabDir) }

    uv run --no-sync nv matlab-run --all --max-steps $MaxSteps --no-progress @dirArgs

    if ($Serve) {
        uv run --no-sync nv serve --dir artifacts/matlab
    }
    else {
        Write-Host "`nDone. View with:  uv run nv serve --dir artifacts/matlab" -ForegroundColor Green
    }
}
finally { Pop-Location }
