---
name: locator-evaluation
description: Encodes the hard-won rules for designing and interpreting NVision locator/strategy comparisons (Bayesian-SBED, Bayesian-SMC, SimpleSweep, etc.) — which comparisons are statistically fair, how to read the Highlights and Dashboard UI views, and known historical bugs that silently invalidated past results. Use this whenever the user wants to compare locator/strategy performance, interpret convergence or steps-to-converge numbers, design a new evaluation or parameter-grid study, or asks why one strategy looks better/worse than another. Also flags stale-artifact traps (pre-fix SimpleSweep numbers, sweep "steps to converge" not meaning efficiency) before they get reused in an analysis.
---

# Locator Evaluation

These rules came out of repeated debugging of "why does locator X look better than Y" —
several looked-plausible comparisons turned out to be artifacts of measurement, not real
performance differences. Read this before building or interpreting any comparison.

## The one metric that matters: frequency convergence

The deliverable of every experiment is a precise **frequency** estimate. `freq_converged_step`
is the primary axis everywhere. "Full convergence" (`all_converged_step`, all parameters) and
"full run" are **not co-equal goals** — they exist only to verify the frequency milestone can be
trusted (that stopping early didn't sacrifice the answer, and the frequency estimate doesn't
drift once everything else converges). When in doubt about which metric to default to or plot,
prefer `freq_converged` over `full`.

## Fair-comparison rules

Only two comparisons are actually fair; everything else either has survivorship bias or answers
a question nobody asked:

1. **Quality at matched budget** — full run vs full run. Use anytime error-vs-steps curves
   (median + IQR) so the specific step budget chosen doesn't bias the result.
2. **Cost at matched promise** — `freq_converged` vs `freq_converged` at the *same* threshold.
   Use survival curves, not mean steps-to-converge — averaging only over converged repeats has
   survivorship bias (repeats that never converge are silently dropped from the mean).

**Never mix criteria**: comparing strategy X's full-run quality against strategy Y's
freq-converged cost answers neither question. If a UI selector or analysis lets you pick these
independently, that's a bug, not a feature.

**SimpleSweep is a valid baseline only for quality, never for its own steps-to-converge.** Its
"convergence step" reflects where the dip happens to sit in a fixed scan order, not search
efficiency — a dip near the start of the sweep converges in fewer steps than the identical dip
near the end. Don't include sweep in any steps-to-converge comparison.

**Savings decomposition** — "how much cheaper is the adaptive strategy":
- *vs sweep* = matched-quality savings (step where the error curve crosses sweep's final error).
- *vs another adaptive baseline* = matched-promise savings, paired **per repeat** (same ground
  truth signal), not just averaged separately.

## Trust framing: claimed uncertainty is the product

A locator's output is its claimed uncertainty (σ); the true error against ground truth is only
an audit of that claim, not a separate metric to optimize alone. The aggregate trust metric is
**coverage**: fraction of repeats where `|true_err| ≤ 1σ` (or `2σ`) claimed, compared against the
68%/95% Gaussian nominal, with Wilson confidence intervals (small-n binomial CIs, not normal
approximation). Only call a strategy's uncertainty untrustworthy when the whole CI sits below
nominal — a point estimate crossing nominal isn't enough given typical repeat counts.

## Budget-censoring vs stalled (reading convergence-vs-noise)

When a repeat never converges within `max_steps`, distinguish two causes before concluding the
strategy is bad at that noise level:
- **Budget-censored**: claimed uncertainty was still falling >5% over the last 20% of the budget
  — the run just needed more steps. Fix: raise `--loc-max-steps`, don't blame the strategy.
- **Stalled**: uncertainty is flat — the convergence threshold is genuinely unreachable at that
  noise level. This is a real finding, not a budget problem.

The convergence threshold (`tau`, default `NVISION_FREQ_CONVERGENCE_THRESHOLD` = 1e5 Hz) is a
fixed product requirement — don't loosen it to paper over high noise. If noise-sensitivity
matters, use the τ-sensitivity toggle (×0.5/×1/×2) in the UI rather than changing the constant.

## Where to look in the UI

- **Highlights view** (per generator/noise, curated): anytime error curves, convergence survival
  curves at τ×0.5/×1/×2, coverage vs 68/95% nominals, paired matched-promise savings with a
  search/refine decomposition (search = steps until claimed uncertainty halves from its initial
  value).
- **Dashboard tab**: aggregates across *all* generators/noise levels with no selection needed —
  advantage-at-sweep-quality heatmap, promise-kept coverage heatmap, converged-within-budget vs
  noise (hover shows the budget-censored/stalled diagnosis above), pooled paired savings vs
  `f_span` scatter.
- Full rendering details: `docs/ui_architecture.md` (sections 9 "Highlights view" and 11
  "Dashboard tab").

## Parameter-grid studies

`nvision/viz/grid_study.py` activates automatically when result rows carry grid-coordinate
columns (`grid_saturation`/`grid_sigma_inhom` for the saturation-Voigt study, or
`grid_linewidth`/`grid_c_total` for the Lorentzian study) — one fixed signal-parameter point per
grid cell, aggregated over repeats. Cell aggregation is **censored**: step metrics report the
median (+ IQR) over *converged* repeats only, always paired with `convergence_rate`; cells that
never converge report the failure breakdown (`infeasible_crlb` vs other) instead of a step
number. Don't read a grid heatmap cell's step value without also checking its convergence rate —
a low rate with a fast median is likely cherry-picked from the easy repeats.

## Known historical traps — check before trusting old artifacts

These were real bugs that silently corrupted past comparisons. If a result looks like one of
these symptoms, don't trust it without regenerating from a run made after the fix date.

- **SimpleSweep prior-mean bug (fixed 2026-06-12)**: before the fix, SimpleSweep's recorded
  `abs_err_x`/`uncert` were the un-updated prior (a constant `uncert` equal to the domain's
  uniform-prior std, for *every* repeat and noise level) because the dip-fit result never
  reached the finalize record. Any "vs sweep" comparison (Dashboard advantage heatmap, Highlights
  sweep-quality line, error-vs-noise) built from artifacts generated before this date is invalid
  — regenerate the sweep runs.
- **Sweep's mid-run belief is flat by design, not a bug**: SimpleSweep doesn't do an incremental
  fit, so its anytime error curve stays at prior level until the very last point. This is
  expected, not something to "fix" — mid-sweep partial fits aren't implemented.
- **The red "locator most likely signal" curve is the SMC belief mode, not the curve fit** — it's
  `belief_mode_estimates(locator.belief)`, the marginal argmax per parameter. During a sweep the
  belief is batch-updated without resampling and can collapse, producing a garbage curve (e.g. a
  phantom third dip) even on clean data — that's a plotting/interpretation artifact, not evidence
  the fit is wrong. `GenericSweepLocator.fit_mode_estimates()` is the actual fitted curve.
- **Sweep-fit bounds aliasing (fixed 2026-07-08)**: a shared `param_bounds_phys` dict could get
  mutated in place by SMC belief narrowing before the curve fit ran, producing nondeterministic
  or badly-wrong fits (frequency pinned at a narrowed window edge). Guarded by the 4-seed stress
  test in `tests/test_generic_sweep_locator_fit.py` — if sweep fits regress to "right frequency,
  wrong shape" or become nondeterministic across runs, this is the first thing to check.
- **`NVISION_SMC_EIG_CACHE`**: SBED's expected-information-gain has a cached and a fused
  computation path (default cached). They're not bit-identical step-to-step (different particle
  subsampling), though acquisition *decisions* rank identically. If you need to rule this out as
  a source of a comparison discrepancy, set `NVISION_SMC_EIG_CACHE=0` to A/B against the fused
  path.
