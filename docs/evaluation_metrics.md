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
| **Focus** | going from "dip seen" to "located near the steepest slope" (within ε, where the sensitivity is still ≥ 90% of its best) | information per measurement ∝ SNR², so cost ∝ 1/SNR²; expected independent of the window (a hypothesis, checked in the Scaling view) | measurements in the focus stage, efficiency = Fisher bound / steps |
| **Refine** | Fisher-limited precision | measurements an oracle needs for σ = τ, which scale like (FWHM_true / (SNR · τ))² | Fisher efficiency η (`stage3_sens_eff`), efficiency = Fisher bound / steps (`stage3_efficiency`) |

`sigma_inhom` and the hyperfine isotope matter only through the dip's effective width: plot against the effective
FWHM, not against those columns.

**k is relative to the prior FWHM, not the true one.** The sweep spacing is FWHM_prior / k. When the prior is wide
(e.g. prior HWHM ≈ 2.5 MHz against a true 0.5 MHz) a "k = 1" grid puts ~0.2 points per true FWHM and can step over the
dip. k = 1 is therefore not studied; compare spacings by points per true FWHM.

## The Fisher oracle behind the stage metrics

A stage is judged against the fewest measurements any locator could need, computed from the *true* signal
(`nvision/metrics/stages.py`, simulation only). One measurement of the primary parameter θ (the dip position, e.g.
`zeeman_shift`) at drive frequency x, with Gaussian noise σ per shot, carries Fisher information

$$I(x) = \frac{g(x)^2}{\sigma^2}, \qquad g = \frac{\partial\,\mathrm{signal}}{\partial\theta}.$$

Independent measurements add their information, so the Cramér-Rao bound after N of them is
σ_θ = 1/√(N I). The best point is the steepest slope of the dip: for a Lorentzian of HWHM γ and depth C,
|g\*| ≈ 0.65 C/γ, so I\* = max g² / σ² ≈ 0.42 SNR²/γ² (SNR = C/σ). Hence the steps needed for a precision ε are
N ≈ 1/(ε² I\*): **they fall as 1/SNR²**. `stage_fisher_star` stores I\* (Hz⁻² per step; shots enter as a factor).

* **Focus ends at ε**, the half-width of the region around the best point where g² ≥ 0.9 max g² (η ≥ 1 − δ with
  δ = 0.1). For a Lorentzian η ≈ 1 − 4.5 (ε/γ)², so ε = √(0.1/4.5) γ ≈ 0.15 γ; the code finds it numerically from the
  true curve (`stage_eps_target_phys`), so it holds for any line shape. Oracle: `stage2_oracle_steps` = 1/(ε² I\*), about
  100/SNR² for a Lorentzian (the constant depends on the SNR and width definitions; the exponent is the point).
  Not defined for drive-controlled models (the best point is a drive setting; pulse fringes can be narrower than the grid).
* **Refine ends at the stop target τ** (`stage_tau_phys`). Oracle: `stage3_oracle_steps` = (τ⁻² − σ₀⁻²)/I\* with σ₀
  the claimed σ at refine start. It scales like (FWHM_true / (SNR τ))²: the Scaling view's refine panel axis.
* **Search** has no Fisher bound (nothing is known about where the dip is); its scale is the number of dip-widths to cover,
  window / FWHM_true (`stage_window_fwhm`), times the passes needed to see the dip at the given SNR.
* **Efficiency** = oracle steps / steps spent (`stage2_efficiency`, `stage3_efficiency`): 1 = Fisher-limited. The oracle is a
  lower bound that assumes every measurement is at the best point and the other parameters are known, so it is not a target.
  `stage{2,3}_wasted_share` is the share of the stage's measurements with η < 1 − δ.
* The Fisher curve in the Stages view's claimed-uncertainty plots, (σ₀⁻² + j I\*)^(-1/2) after j in-stage measurements,
  starts from the stage's own claimed σ and is the same bound; below it the belief claims more precision than the data carry.

Per-run axes stored with them: `stage_fwhm_true_phys` (2 × `effective_hwhm` of the true signal; none for drive-controlled
models), `stage_prior_fwhm_phys` (mean prior FWHM), `stage_window_phys`, `stage_window_fwhm`.

**What the first check showed** (2026-10-27, read-only, on the stored pure-Lorentzian SNR × window rows, written before
`NVISION_DIP_P_FA` went to 0.1 and before measurements are confined to the focus window; re-check on fresh runs):
SBED's focus stage took 25-150× the bound and flattened at roughly 20-40 steps at high SNR instead of following 1/SNR²;
the claimed σ at focus start was already 0.1-0.5 γ (not ≈ the FWHM) while the true error was 2-4× larger (overconfidence);
focus steps grew about 2× from window 5 to 10 × FWHM; SBED search steps were far flatter in the window than proportional.
With the focus stage's mean η only 0.3-0.5, the stage length looks set by exploration the tail rule counts, not by information.

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
7. **Metrics are per use case.** A use case is one signal at one noise level: window, line shape and SNR (generator ×
   noise for other studies). Compute every headline number (median and p75 cost, capped share, σ coverage, stage
   durations) within one use case first; never read it off runs pooled across windows, shapes or SNRs, since the cost
   spans orders of magnitude between them. A selection of several use cases is summarised as the **median over the
   use cases of each use case's own number** (every use case counts once, whatever its number of repeats) and is
   labelled with what it covers ("Median over 45 use cases · window 5, 10, 20 × FWHM · … · SNR 1…20"); the per-use-case
   numbers stay one click away. Such a median mixes easy and hard use cases, so it is a summary of the selection, not a
   property of any one signal. Trends across use cases are still shown as curves, one point per use case.
8. **Medians hide the tail.** Show p75 and the capped share next to the median cost; the spacings differ more in the
   tail than at the median.

## Where it lives

* Per-run metrics: `nvision/metrics/stages.py` (stage boundaries, η), `nvision/metrics/milestones.py` (stop step),
  `nvision/metrics/series.py` (per-step series), `nvision/metrics/compare.py` (`nv compare` tables).
* Results UI views: **Stages** (scope selector Whole run /
  Search / Focus / Refine; **Whole run** also carries **Cost & search**: every locator side by side, chips to show/hide, a
  median-over-selection block and a block per use case, and cost and search vs SNR across use cases, `/api/run-rows`;
  per stage the claimed-σ and true-error curves with the Fisher bound (dashed) and a -1/2 slope
  reference, the median with an interquartile band, a "runs left" strip (curves stop where fewer than half the repeats remain
  in the stage), a table with the stage's Fisher bound, efficiency and off-optimum share, and for a single use case the
  paired per-repeat difference against SBED; Whole run shows the stacked stage bar and the total / capped table),
  **Scaling** (fixed locator, many use cases: search and focus steps vs window, focus steps vs SNR, refine steps vs
  (FWHM/(SNR τ))², each with its predicted scaling dotted and the Fisher bound dashed, a panel only when the selection spans
  at least two values of its axis), **Sanity** (σ coverage vs SNR "Is the claimed σ right?", then the stop-quality panels, then the failed runs; the failed-run
  count is in the tab label), and **Run** for one repeat. Stages and Sanity default to the median over the selected use cases.
  See [ui_architecture.md](ui_architecture.md).

## Not done yet

* The oracle columns exist only for runs recorded after they were added; older runs show a dash (the stage metrics are
  computed from the belief snapshots at run time and the cache keeps no snapshots, so there is nothing to backfill from).
  Re-run to get them.
* Search is judged by its steps against window / FWHM_true in the Scaling view; there is no analytic "passes" model for it
  yet, and the prior FWHM / grid step enters only through `stage_prior_fwhm_phys`.
* Stage boundaries depend on the dip detector (`NVISION_DIP_P_FA`) and δ = 0.1; a one-off sensitivity check (δ = 0.05 / 0.1 /
  0.2) that the locator ranking survives is still to do.
* The signed z = (estimate − truth) / claimed σ is not stored (only |error|), so σ honesty is checked against a
  half-normal, not N(0, 1).
* SBED-Drive against every fixed drive amplitude, including the best one in hindsight.
