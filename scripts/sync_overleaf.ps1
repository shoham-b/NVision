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

  Only some files: -Files takes paths relative to articles/ (the local names, so nvision_paper.tex, not
  main.tex). Every other file is left as it is on the other side, so pushing one section does not overwrite
  edits made on Overleaf, or by other branches, in the rest of the paper:
  scripts/sync_overleaf.ps1 push -Files sections/theory.tex
  scripts/sync_overleaf.ps1 push -Files sections/theory.tex, sections/methods.tex, figures/fig1.pdf

  -DryRun lists what would change (a diff of the selected files against the other side) and writes nothing.
  scripts/sync_overleaf.ps1 push -DryRun                          # which files differ from Overleaf?
  scripts/sync_overleaf.ps1 push -Files sections/theory.tex -DryRun

  Auth: Overleaf Git bridge uses user "git" and a personal Git token
  (Overleaf > Account Settings > Git Integration). Git Credential Manager will
  prompt once and cache it; the token is never stored in this repo.
#>
param(
    [Parameter(Mandatory)][ValidateSet("pull", "push")][string]$Direction,
    [string[]]$Files,
    [switch]$DryRun
)

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

$partial = [bool]$Files
if ($partial) {
    # Restrict the sync to the named files. Figures are matched by their path, which is the same on both sides.
    $selected = [ordered]@{}
    foreach ($f in $Files) {
        $rel = $f.Replace("\", "/").TrimStart("./")
        if ($rel.StartsWith("articles/")) { $rel = $rel.Substring("articles/".Length) }
        if ($map.Contains($rel)) {
            $selected[$rel] = $map[$rel]
        } elseif ($rel.StartsWith("figures/") -and ($figureExtensions -contains [IO.Path]::GetExtension($rel).ToLower())) {
            $selected[$rel] = $rel
        } else {
            throw "Not a synced file: '$f'. Give a path under articles/, e.g. sections/theory.tex or figures/x.pdf."
        }
    }
    $map = $selected
} else {
    # Full sync: every figure on the source side.
    $figureRoot = if ($Direction -eq "pull") { $clone } else { $articles }
    foreach ($rel in Get-FigurePaths $figureRoot) { $map[$rel] = $rel }
}

# (source, destination) of every selected file, in the direction of the sync.
$pairs = @(foreach ($local in $map.Keys) {
    $localPath = Join-Path $articles $local
    $remotePath = Join-Path $clone $map[$local]
    if ($Direction -eq "pull") { , @($remotePath, $localPath, $map[$local]) } else { , @($localPath, $remotePath, $local) }
})
foreach ($p in $pairs) {
    if (-not (Test-Path $p[0])) { throw "Missing on the source side: $($p[2])" }
}

if ($DryRun) {
    $changed = 0
    foreach ($p in $pairs) {
        if (-not (Test-Path $p[1])) { Write-Host "new: $($p[2])"; $changed++; continue }
        # --ignore-cr-at-eol: the local checkout may have CRLF where Overleaf has LF; that is not a change.
        git diff --no-index --ignore-cr-at-eol --quiet -- $p[1] $p[0]
        if ($LASTEXITCODE) {
            Write-Host "changed: $($p[2])"
            if ($partial) { git --no-pager diff --no-index --ignore-cr-at-eol -- $p[1] $p[0] }
            $changed++
        }
    }
    Write-Host "Dry run ($Direction): $changed of $($pairs.Count) file(s) would change. Nothing written."
    return
}

if ($Direction -eq "pull") {
    foreach ($p in $pairs) {
        New-Item -ItemType Directory -Force (Split-Path $p[1]) | Out-Null
        Copy-Item $p[0] $p[1] -Force
    }
    Write-Host "Pulled from Overleaf. Review with: git diff articles/"
} else {
    foreach ($p in $pairs) {
        New-Item -ItemType Directory -Force (Split-Path $p[1]) | Out-Null
        Copy-Item $p[0] $p[1] -Force
    }
    git -C $clone add -- @($map.Values)
    # The main document used to be nvision_paper.tex on Overleaf; drop it so the project has one copy.
    if (-not $partial -and (Test-Path (Join-Path $clone "nvision_paper.tex"))) { git -C $clone rm -q -- nvision_paper.tex }
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
