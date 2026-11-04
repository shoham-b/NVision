<#
.SYNOPSIS
  Two-way sync of the article in articles/ with an Overleaf project via its Git bridge.

.DESCRIPTION
  Synced files: nvision_paper.bib, articles/sections/*.tex, and the main document, which is
  articles/nvision_paper.tex locally but main.tex on Overleaf (Overleaf's default main document).
  On push, the old nvision_paper.tex is removed from the Overleaf project.

.USAGE
  $env:OVERLEAF_PROJECT_ID = "<id from the Overleaf project URL>"
  scripts/sync_overleaf.ps1 pull   # Overleaf -> articles/   (then review + commit)
  scripts/sync_overleaf.ps1 push   # articles/ -> Overleaf

  Auth: Overleaf Git bridge uses user "git" and a personal Git token
  (Overleaf > Account Settings > Git Integration). Git Credential Manager will
  prompt once and cache it; the token is never stored in this repo.
#>
param([Parameter(Mandatory)][ValidateSet("pull", "push")][string]$Direction)

# git writes progress and CRLF notices to stderr; with "Stop" PowerShell 5.1 would treat those as errors.
# Native failures are caught through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
if (-not $env:OVERLEAF_PROJECT_ID) { throw "Set `$env:OVERLEAF_PROJECT_ID first." }

$repoRoot = Split-Path -Parent $PSScriptRoot
$articles = Join-Path $repoRoot "articles"
$clone = Join-Path $repoRoot "scratch\overleaf_clone"
$url = "https://git@git.overleaf.com/$($env:OVERLEAF_PROJECT_ID)"

# (local path under articles/) -> (path in the Overleaf project)
$map = [ordered]@{
    "nvision_paper.tex" = "main.tex"
    "nvision_paper.bib" = "nvision_paper.bib"
}
Get-ChildItem (Join-Path $articles "sections") -Filter *.tex | ForEach-Object {
    $map["sections/$($_.Name)"] = "sections/$($_.Name)"
}

if (-not (Test-Path (Join-Path $clone ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path $clone) | Out-Null
    git clone $url $clone
    if ($LASTEXITCODE) { throw "clone failed" }
} else {
    git -C $clone pull --ff-only
    if ($LASTEXITCODE) { throw "pull failed" }
}

if ($Direction -eq "pull") {
    foreach ($local in $map.Keys) {
        Copy-Item (Join-Path $clone $map[$local]) (Join-Path $articles $local) -Force
    }
    Write-Host "Pulled from Overleaf. Review with: git diff articles/"
} else {
    New-Item -ItemType Directory -Force (Join-Path $clone "sections") | Out-Null
    foreach ($local in $map.Keys) {
        Copy-Item (Join-Path $articles $local) (Join-Path $clone $map[$local]) -Force
    }
    git -C $clone add -- @($map.Values)
    # The main document used to be nvision_paper.tex on Overleaf; drop it so the project has one copy.
    if (Test-Path (Join-Path $clone "nvision_paper.tex")) { git -C $clone rm -q -- nvision_paper.tex }
    if (git -C $clone status --porcelain) {
        git -C $clone -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -q -m "Sync from NVision repo"
        if ($LASTEXITCODE) { throw "commit failed" }
        git -C $clone push
        if ($LASTEXITCODE) { throw "push failed" }
        Write-Host "Pushed to Overleaf."
    } else {
        Write-Host "Overleaf already up to date."
    }
}
