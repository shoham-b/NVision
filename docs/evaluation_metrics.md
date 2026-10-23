# Evaluation metrics: what to measure, per stage

Read this before adding or changing any locator-comparison metric or results view. It records what the comparison
is for, which quantity drives each stage, and the rules that keep a metric meaningful. The stage detector itself
(search → focus → refine, the Fisher-efficiency rule) is defined in [locator_comparison.md](locator_comparison.md).

## What the comparison is for

**Bayesian SBED is the subject.** The other locators are a yardstick that shows what SBED gains. Scenario: the
line shape was profiled before, the magnetic field moved, and the new resonance must be found.

The yardstick is a ladder, each rung tuned before the next:

| Rung | Locator | Tuned / differs by |
|---|---|---|
| 0 | `SimpleSweep-<k>perFWHM`, k = 2, 3, 4 | grid step (k points per **prior** FWHM); passes averaged until the stop rule fires |
| 1 | `Smart-Sweep` | coarse sweep, then sweep inside the dip, then the max-Fisher points |
| 2 | `Smart-SBED` | same coarse grid as Smart-Sweep: only the acquisition after the dip is seen differs |
| 3 | `Bayesian-SBED` | no coarse grid |
| 4 | `Bayesian-SBED-Drive` | also picks the MW drive amplitude (strong to search, near the slope optimum to refine) |

Expected story: the search stage is a tie (blind, cost of order n for everyone); SBED's distinctive win over
Smart-Sweep is in the focus stage; in refine SBED and Smart-Sweep both measure at the Fisher-optimal points, while a
plain sweep keeps spreading its measurements over the whole dip.

## What drives each stage

Choose the axis from the stage's physics, not from whatever grid columns the results rows happen to have.

| Stage | What drives it | Axis | Metric |
|---|---|---|---|
| **Search** | whether the grid puts enough points inside the true dip, and how deep each point sees it | **grid points per true FWHM** = k × FWHM_true / FWHM_prior, with SNR = contrast / noise σ | measurements until a dip is seen (better: in grid **passes**), dip-found rate |
| **Focus** | going from "dip seen" (uncertainty ≈ FWHM) to "located near the steepest slope" | precision relative to the line: FWHM_true / SNR | measurements in the focus stage |
| **Refine** | Fisher-limited precision | measurements an oracle needs for σ = τ, which scale like (FWHM_true / (SNR · τ))² | Fisher efficiency η (`stage3_sens_eff`), cost / Fisher-limited cost |

`sigma_inhom` and the hyperfine isotope matter only through the dip's effective width: plot against the effective
FWHM, not against those columns.

**k is relative to the prior FWHM, not the true one.** The sweep spacing is FWHM_prior / k. When the prior is wide
(e.g. prior HWHM ≈ 2.5 MHz against a true 0.5 MHz) a "k = 1" grid puts ~0.2 points per true FWHM and can step over the
dip. k = 1 is therefore not studied; compare spacings by points per true FWHM.

## Rules

1. **Never use the accuracy at the end of a run or stage as a quality metric.** Every run stops once its claimed σ
   stays below the target τ, so the error it happens to end with mostly reflects τ. What matters is that the claimed
   σ is right: report the share of runs with |error| ≤ 1σ and ≤ 2σ against the Gaussian 68% / 95%, or |error| /
   claimed σ. Below nominal = overconfident (the belief is known to be; see the locator memory notes).
2. **Every locator gets the same budget.** `NVISION_MAX_MEASUREMENTS` (default 2500, for every signal and every window
   of the SNR x window study) is the SBED family's `max_steps` and caps a plain sweep's total measurements. A run that hit it is censored: the stop rule never fired
   (`primary_converged_step` is null; a sweep still ends as a normal locator stop). Count it at the budget, as a lower
   bound, and show the capped share.
3. **Show the raw metrics before ranking anything.** No arbitrary reference locator (e.g. "divided by k = 3"), no
   "cheapest k" summaries, until the plain per-locator metrics have been looked at.
4. **Use physical axes.** SNR = contrast / noise σ (∞ without noise) and the effective FWHM, per the stage table
   above.
5. **Runs are paired.** Repeat *i* has the same true signal for every locator and noise level
   (`signal_repeat_key`), so per-repeat comparisons are valid; measurement noise differs, as it must.
6. **Stops are compared only at matched reliability.** Comparing measurements-to-stop is fair only if the locators'
   claimed σ are equally honest (rule 1); otherwise one could look faster by stopping riskier.
7. **Medians hide the tail.** Show p75 and the capped share next to the median cost; the spacings differ more in the
   tail than at the median.

## Where it lives

* Per-run metrics: `nvision/metrics/stages.py` (stage boundaries, η), `nvision/metrics/milestones.py` (stop step),
  `nvision/metrics/series.py` (per-step series), `nvision/metrics/compare.py` (`nv compare` tables).
* Results UI views: **Locators** (every locator side by side, chips to show/hide, cost, search, σ coverage vs SNR;
  `/api/run-rows`), **Stages**, **Accuracy** (stop-quality panels), **Failed Runs**, and **Run** for one repeat.
  See [ui_architecture.md](ui_architecture.md).

## Not done yet

* The search axis needs per-run widths the rows don't carry: the true effective FWHM (from the model's own width
  function) and the prior FWHM / grid step (`prior_fwhm_phys`). Compute them on the server from the cached repeat
  for existing runs, and record them as per-run metrics for new ones; then plot search cost in passes against points
  per true FWHM.
* Focus and refine axes (FWHM_true / SNR, Fisher-limited cost) in the Locators view.
* The signed z = (estimate − truth) / claimed σ is not stored (only |error|), so σ honesty is checked against a
  half-normal, not N(0, 1).
* SBED-Drive against every fixed drive amplitude, including the best one in hindsight.
