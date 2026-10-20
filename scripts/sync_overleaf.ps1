<#
.SYNOPSIS
  Two-way sync of articles/nvision_paper.{tex,bib} with an Overleaf project via its Git bridge.

.USAGE
  $env:OVERLEAF_PROJECT_ID = "<id from the Overleaf project URL>"
  scripts/sync_overleaf.ps1 pull   # Overleaf -> articles/   (then review + commit)
  scripts/sync_overleaf.ps1 push   # articles/ -> Overleaf

  Auth: Overleaf Git bridge uses user "git" and a personal Git token
  (Overleaf > Account Settings > Git Integration). Git Credential Manager will
  prompt once and cache it; the token is never stored in this repo.
#>
param([Parameter(Mandatory)][ValidateSet("pull", "push")][string]$Direction)

$ErrorActionPreference = "Stop"
if (-not $env:OVERLEAF_PROJECT_ID) { throw "Set `$env:OVERLEAF_PROJECT_ID first." }

$repoRoot = Split-Path -Parent $PSScriptRoot
$articles = Join-Path $repoRoot "articles"
$files = @("nvision_paper.tex", "nvision_paper.bib")
$clone = Join-Path $repoRoot "scratch\overleaf_clone"
$url = "https://git@git.overleaf.com/$($env:OVERLEAF_PROJECT_ID)"

if (-not (Test-Path (Join-Path $clone ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path $clone) | Out-Null
    git clone $url $clone
} else {
    git -C $clone pull --ff-only
}

if ($Direction -eq "pull") {
    foreach ($f in $files) { Copy-Item (Join-Path $clone $f) (Join-Path $articles $f) -Force }
    Write-Host "Pulled from Overleaf. Review with: git diff articles/"
} else {
    foreach ($f in $files) { Copy-Item (Join-Path $articles $f) (Join-Path $clone $f) -Force }
    git -C $clone add -- $files
    if (git -C $clone status --porcelain) {
        git -C $clone -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -m "Sync from NVision repo"
        if ($LASTEXITCODE) { throw "commit failed" }
        git -C $clone push
        if ($LASTEXITCODE) { throw "push failed" }
        Write-Host "Pushed to Overleaf."
    } else {
        Write-Host "Overleaf already up to date."
    }
}
