<#
.SYNOPSIS
  Two-way sync of the article in articles/ with an Overleaf project via its Git bridge.

.DESCRIPTION
  Synced files: nvision_paper.bib, articles/sections/*.tex, supplemental.tex (the Supplemental Material) with
  the latexmkrc that builds it first, the main document, which is
  articles/nvision_paper.tex locally but main.tex on Overleaf (Overleaf's default main document),
  and every figure under articles/figures/ (same path on Overleaf, subfolders included; see
  $figureExtensions). Figures are synced as a folder: pull also brings down figures that exist only on
  Overleaf, push uploads figures that exist only locally. Deletions are not propagated.
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
    # Supplemental Material: its own document, cross-referenced from main.tex through xr-hyper; latexmkrc makes
    # Overleaf build supplemental.aux before main.tex.
    "supplemental.tex"  = "supplemental.tex"
    "latexmkrc"         = "latexmkrc"
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

# Figures: same relative path on both sides, under figures/.
$figureExtensions = @(".pdf", ".png", ".jpg", ".jpeg", ".eps", ".tex", ".tikz")
function Get-FigurePaths([string]$root) {
    $dir = Join-Path $root "figures"
    if (-not (Test-Path $dir)) { return @() }
    $prefix = (Resolve-Path $root).Path.TrimEnd("\") + "\"
    Get-ChildItem $dir -Recurse -File |
        Where-Object { $figureExtensions -contains $_.Extension.ToLower() } |
        ForEach-Object { $_.FullName.Substring($prefix.Length).Replace("\", "/") }
}
function Copy-Figures([string]$fromRoot, [string]$toRoot) {
    foreach ($rel in Get-FigurePaths $fromRoot) {
        $dest = Join-Path $toRoot $rel
        New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
        Copy-Item (Join-Path $fromRoot $rel) $dest -Force
    }
}

if ($Direction -eq "pull") {
    foreach ($local in $map.Keys) {
        Copy-Item (Join-Path $clone $map[$local]) (Join-Path $articles $local) -Force
    }
    Copy-Figures $clone $articles
    Write-Host "Pulled from Overleaf. Review with: git diff articles/"
} else {
    New-Item -ItemType Directory -Force (Join-Path $clone "sections") | Out-Null
    foreach ($local in $map.Keys) {
        Copy-Item (Join-Path $articles $local) (Join-Path $clone $map[$local]) -Force
    }
    Copy-Figures $articles $clone
    git -C $clone add -- @($map.Values)
    if (Test-Path (Join-Path $clone "figures")) { git -C $clone add -- figures }
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
