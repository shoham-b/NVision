---
name: nvision-plot-integrity-check
description: Bulk-verifies that NVision's results UI actually has working plot data across every (generator, noise, strategy, repeat) combination — not just the one combo you happen to click on — using `nv cache check-plots`, a script-only check that needs no browser. Use whenever the user asks to check that the UI/plots work well, that a run's output is actually renderable, or wants proof beyond "the page loaded" — especially mid-run, after a plotting/serialization code change, or after seeing any "Failed to load plot" error. Catches base64/encoding corruption and missing-graph-data bugs across the whole run in seconds instead of clicking through combinations one at a time.
---

# Checking the results UI actually works, across the whole run

`nvision-run-monitoring`'s verification recipe (its step 4) checks that *one* selected combo
renders in a real browser — necessary, but it doesn't tell you whether the other few thousand
repeats in the run are fine too. A plot-loading bug is often confined to a subset (one generator
variant, one strategy, one time window) that a single spot-check will never land on by chance.
This skill is the bulk version: verify every referenced plot file's data actually decodes, then
use the browser only to visually confirm the specific combos that came back bad.

## Why this doesn't need a browser

`static/plotly-utils.js`'s `_decodePlotlyFigure` does exactly three things to render a scan plot:
gzip-decompress, `JSON.parse` (after substituting `Infinity`/`NaN` -> `null`), then
base64-`atob()`-decode every `{bdata,dtype}`/`{__f32__}` typed-array payload. All three steps are
pure data transforms with no DOM/browser dependency — `nv cache check-plots` replays them in
Python against the actual files on disk. A payload that fails to base64-decode here is the exact
server-side reproduction of the UI's `Failed to load plot: ... atob ... not correctly encoded`
error. This is strictly better than browser-clicking for a bulk check: it's faster, it's
scriptable, and — critically — it does not hit the browser tools' large-manifest
screenshot-timeout issue documented in [[nvision-run-monitoring]] (section 3), since there's no
rendering involved at all.

## Running the check

```bash
uv run nv render                                          # snapshot current cache state first
uv run nv cache check-plots --type scan --sample 1000      # fast spot-check across everything
```

```
Checked 1000 plot file(s): 154 OK, 840 missing (rerun with --restore if unexpected), 6 corrupt (real bug -- data on disk that fails to decode).
```

Narrow with `--strategy` / `--generator` / `--noise` (substring match) to focus on a suspect
area, or drop `--sample` to check every matching entry exhaustively once you have time budget for
it. `--worst`/breakdown-style drilling isn't built in here the way it is for
[[nvision-convergence-check]] — instead, break down manually by re-running with different filters
(see the worked example below) once the aggregate count shows a problem exists.

## Reading the result: missing vs corrupt are different severities

The command deliberately reports these as two separate buckets, and so should you:

- **`corrupt`** (base64/JSON decode actually failed) is unambiguously a real bug — bytes exist on
  disk and don't decode. Every corrupt entry is worth taking to the `--worst`-style drill-down:
  note its `(generator, noise, strategy, repeat)` and go look at it in the browser (§ below) to
  confirm the user-facing symptom, then treat it as a genuine plotting/serialization bug to fix.
- **`missing`** (no file at that path at all) is *not* automatically a bug. Graph bytes live in
  the cache DB during a run and are only materialized to `<out>/graphs/...` on demand — by
  `nv render`, or by `nv serve`'s one-time `_restore_missing_graphs` walk at startup (see
  [[nvision-run-monitoring]] section 3). Pass `--restore` to run that same materialization pass
  yourself before concluding anything:
  ```bash
  uv run nv cache check-plots --type scan --strategy Bayesian-SBED --restore
  ```
  **If `--restore` does *not* reduce the missing count, that upgrades the finding** — it means
  the plot bytes were never captured in the cache in the first place, not merely un-materialized.
  That's the same failure class as the historical bug flagged in [[nvision-run-monitoring]]'s
  step 5 note ("~99% of a 44-hour run's output" went uncaptured silently). Confirm via the cache
  DB directly per that skill's step-1 recipe (check for `content_bin`/`content` on the specific
  combo's cache entry, not just a `path`) before reporting it as a capture-time bug rather than a
  materialization-timing artifact.

## Worked example: how a systemic bug actually looked in this check

A real run showed 840/1000 sampled scan plots missing even before `--restore`. Rather than
report "84% missing" as one number, breaking it down by generator family and strategy in a
one-off script (grouping the manifest by `"voigt" in generator` vs `"lorentzian" in generator`,
and by `strategy`) showed:

```
lorentzian:    0/2250   missing (0.0%)
voigt:     17987/19009  missing (94.6%)

SimpleSweep:    5931/7025  missing (84.4%)
Bayesian-SBED:  6033/7114  missing (84.8%)
SimpleSobol:    6023/7120  missing (84.6%)
```

The apparent generator split was a red herring: lorentzian combos in that run happened to still
be stuck on their first sub-task's share (scheduler hadn't gotten to their later sub-tasks yet),
while voigt combos had progressed further — so voigt was the one exposing the bug, not causing
it. `--restore` making no difference was the real signal: it meant `content_bin` was never in the
cache to restore, which traced back to a genuine bug in `nvision/runner/executor.py`'s
`_save_full_cache` — for a *split* sub-task (`_split_oversized_tasks` in `nvision/cli/run.py`,
active whenever `repeats > runners * 3`), the streaming-vs-non-streaming decision compared the
sub-task's own chunk size against `NVISION_STREAMING_REPEAT_THRESHOLD` instead of the
combination's full repeat count. Once a sub-task's chunk was at or under that threshold (default
0, but this project's `.env` sets it to 5 — and a 50-repeat combo over 10 runners chunks to
exactly 5 each), it took the non-streaming branch and re-embedded content from its own
already-stripped (heavy-fields-removed) results at the end of its run, finding neither in-memory
bytes nor an on-disk file to fall back on, and silently overwrote its own correctly-saved
per-repeat cache rows with content-less ones. Fixed by making that check use the combination's
total repeat count (`self.task.repeat_total or self.repeats`), matching what `_run_repeats`
already used to decide streaming mode — see the fix's commit for the full writeup. The general
lesson holds regardless: when a `check-plots` run shows a large missing/corrupt count, break it
down by generator, strategy, and noise before reporting a single "run is X% broken" number, but
also check whether the run used `--runners` > 1 with large `--repeats` (i.e. sub-task splitting
was active) before concluding the breakdown's grouping *is* the cause rather than a correlate of
which combos had progressed furthest.

## Confirming a flagged repeat visually (the browser step)

Once `check-plots` has narrowed things to specific `(generator, noise, strategy, repeat)` tuples,
confirm the user-facing symptom the way [[nvision-run-monitoring]]'s verification recipe (step 4)
describes: open the served UI, select that exact combo, and check for the literal
`Failed to load plot` text plus a nonzero `.js-plotly-plot` count via `javascript_tool` — screenshots
are unreliable on a large manifest, text/JS checks are not. Don't skip this step for a genuinely
`corrupt` entry: it's what turns "this base64 string fails to decode in Python" into "this is what
the user sees," which is the actual bug report.

## Code

`nv cache check-plots` lives in `nvision/cli/cache_cmd.py` next to `nv cache convergence`. It
walks `<out>/plots_manifest.json`, opens each entry's `path`, and replays
`static/plotly-utils.js`'s decode in Python (`_iter_typed_array_payloads` finds every
`{bdata,dtype}`/`{__f32__}` payload recursively, mirroring `_decodePlotlyFigure`). If the two
implementations ever drift — e.g. a new typed-array encoding is added to the frontend — update
`_iter_typed_array_payloads` alongside it, or this check will silently stop covering the new
format. See [[nvision-ui]] for the frontend decoder itself.
