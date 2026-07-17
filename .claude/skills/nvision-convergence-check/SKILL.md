---
name: nvision-convergence-check
description: Quantitatively checks whether an NVision locator's completed repeats are actually converging well mid-run — not just finishing — by comparing each repeat's own reported final uncertainty against its true error via `nv cache convergence`. Use whenever the user asks to check fit quality, calibration, or "is this run any good" for an in-progress or completed `nv run`/`nv groups` job, especially to compare across multiple locators/strategies at once or across the parameter grid to tell whether a problem is isolated or systemic. Flags premature convergence / particle-filter mode-lock (confidently wrong, not just imprecise) before a long run finishes and the results get used for evaluation.
---

# Checking convergence quality mid-run

A run finishing (`nv cache progress` hitting 100%) only tells you it *stopped* — not that it
stopped somewhere correct. A locator can terminate in a handful of steps every time and still
look "done." This skill is about catching that before hours/days of compute produce results
nobody should trust. Read [[locator-evaluation]] first — this is the fast, mid-run,
coarse-grained cousin of its "Trust framing: claimed uncertainty is the product" section, not a
replacement for it.

## The core idea: a locator's output is a claim, and claims can be audited

Every repeat reports `abs_err_x` (true error vs ground truth) and `uncert` (the locator's own
final claimed uncertainty), in the same units (Hz), already cross-checked against
`nvision/runner/metrics.py`'s `_promote_uncert`. Their ratio tells you whether the locator is
just imprecise (`abs_err_x` large, `uncert` also large — honest) or actively miscalibrated
(`abs_err_x` large, `uncert` tiny — the posterior collapsed onto a plausible-looking but wrong
answer and is confidently wrong about it). The second failure mode is far worse: it doesn't show
up as "noisy results," it shows up as a locator that looks *great* on every summary stat until
someone checks against ground truth.

## Running the check

```bash
uv run nv render                                   # snapshot the current cache state first
uv run nv cache convergence --breakdown strategy    # compare every locator in the run
```

```
         Convergence check (bad-ratio >= 10.0, low-measurements <= 5)
+-----------------------------------------------------------------------------+
| strategy      |    n | n w/ uncert | median ratio | % ratio>=bad | % low meas. | top failure_reason |
|----------------+------+-------------+---------------+--------------+-------------+---------------------|
| Bayesian-SBED  | 7117 |        7117 |         54.78 |        73.2% |        1.6% | max_steps (9%)      |
| SimpleSobol    | 7120 |        7120 |       1429.94 |        89.9% |        0.0% | -                   |
| SimpleSweep    | 7025 |        7025 |         30.44 |        71.0% |        0.0% | -                   |
+-----------------------------------------------------------------------------+
```

`--breakdown` accepts any `locator_results.csv` column, comma-separated:
- `strategy` (default) — is the problem in one locator or all of them? A shared bug in
  `nvision/runner/metrics.py`/`executor.py`'s finalize path looks like *every* strategy being
  bad simultaneously; a locator-specific algorithmic bug looks like one strategy standing out.
- `grid_linewidth,grid_c_total` (or whatever grid columns the run uses) — is it isolated to hard
  corners of the parameter grid (narrow linewidth, low contrast) or uniform across the whole
  grid? Uniform-and-bad across the entire grid is the more alarming pattern: it rules out "this
  is just a hard case" and points at a bug, not a physics limit.
- `strategy,noise` — does it track noise level, or is it noise-independent (another point toward
  "bug" over "fundamental difficulty")?

Add `--worst N` to list the N individual repeats with the highest ratio — take their
`(generator, noise, strategy, attempt)` straight into the results UI (see
[[nvision-plot-integrity-check]] for how to inspect a specific repeat without a browser, or just
select it in the UI) to look at the actual posterior/measurement plot and confirm what went
wrong (e.g. locked onto the wrong Zeeman dip).

## Reading the numbers — don't stop at one aggregate percentage

- **A "% ratio>=bad" over ~20-30% across the board is worth treating as a real problem**, not
  noise. In one case audited this way, three unrelated strategies (SBED, Sobol, Sweep) in the
  same run all showed 70-90% bad-ratio rates, uniformly across every linewidth × contrast × noise
  cell in the grid — that uniformity, more than the specific number, is what made it clearly a
  bug rather than three independently hard locators.
- **`% low meas.`** (repeats stopping at `--low-measurements` or fewer steps, default 5) flags
  suspiciously fast termination — useful on its own even when `uncert` is missing/unreliable for
  a locator (e.g. don't fully trust SimpleSweep's mid-run belief per [[locator-evaluation]]'s
  "Known historical traps": its `uncert`/`abs_err_x` only reflect the *final* least-squares fit,
  fixed 2026-06-12, so post-fix numbers are meaningful but the measurement-count signal is the
  more robust one for sweep specifically).
- **Cross-reference `final_est_zeeman_split` / `final_est_frequency` against `true_zeeman_split`
  / `true_frequency`** (both are in the CSV) for flagged repeats — a locator that's locked onto
  the wrong Zeeman-split solution will show a badly wrong split estimate alongside the tiny
  claimed uncertainty. This is the single most common concrete cause of a bad ratio: a low-`n`
  SMC/particle posterior converges and resamples around a plausible-but-wrong mode before ever
  visiting the true one.
- **Before concluding "this locator is broken," check whether an uncommitted/recent fix already
  addresses it.** `nvision/runner/executor.py`'s finalize-record re-sync (re-syncing
  `result.snapshots[-1].belief` after a locator's deferred `finalize()`/buffered flush) explicitly
  does **not** touch locators that update their belief every step (SBED) — so if SBED alone shows
  a bad ratio, that's not explained by a stale-snapshot bug and is a genuine finding; if a
  buffered/chunked locator (SimpleSobol, sweep-family) shows one, check whether the run predates
  that fix before concluding it's a new bug.
- **`stop_reason`/`failure_reason`** distinguish "hit `max_steps` without converging" (censored —
  see [[locator-evaluation]]'s budget-censored-vs-stalled distinction) from a clean
  `locator_stop`. A high bad-ratio rate combined with `stop_reason == locator_stop` everywhere
  (not `max_steps`) means the locator is *confidently* declaring victory too early, which is the
  worse of the two problems.

## What this check is not

It's a fast mid-run smoke test using a raw ratio threshold, not the rigorous end-of-run
evaluation. For a final writeup use [[locator-evaluation]]'s coverage-with-Wilson-CI framing
(fraction of repeats with `|true_err| <= 1sigma`/`2sigma` against 68%/95% nominal) instead of this
ratio, and its fair-comparison rules (matched budget / matched promise) before comparing
strategies head-to-head. This skill exists to answer "should I keep this run going, or stop and
investigate first" — not "which strategy wins."

## Code

`nv cache convergence` lives in `nvision/cli/cache_cmd.py` next to `nv cache progress`. It reads
`<out>/locator_results.csv` (written by `nv render`) with polars, computes `abs_err_x / uncert`
per repeat, and groups by `--breakdown`. Read-only, safe against a cache a live run is still
writing to — it never touches the cache itself, only the CSV snapshot. Run `nv render` again for
a fresher snapshot before re-checking; it's the same safe-against-live-cache operation documented
in [[nvision-run-monitoring]] (step 2).
