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
(`nvision/metrics/stages.py`, simulation only). The signal is the photoluminescence (PL) as a function of the drive
frequency x. Notation: Γ = FWHM of the dip (Γ = 2 HWHM), C = its depth, σ = noise standard deviation of **one** shot
(dimensionless, like the PL), SNR = C/σ, k = number of shots averaged into one measurement (`obs.n_shots`, `k_t` in
`stages.py`). One measurement of the primary parameter θ (the dip position, e.g. `zeeman_shift`) at drive frequency x
carries Fisher information

$$I(x) = \frac{k\,g(x)^2}{\sigma^2}, \qquad g(x) = \frac{\partial\,\mathrm{PL}(x;\theta)}{\partial\theta}\ [\mathrm{Hz}^{-1}].$$

Averaging k shots cuts the variance to σ²/k, so the information is multiplied by k. I has units Hz⁻² per measurement
(Fisher information is not unitless). Independent measurements add their information, so the Cramér-Rao bound after N of
them is σ_θ = 1/√(N I). The best point is the steepest slope of the dip: for a Lorentzian, |g\*| ≈ 1.3 C/Γ, so
I\* = k max g² / σ² ≈ 1.7 k SNR²/Γ². Hence the steps needed for a precision ε are N ≈ 1/(ε² I\*): **they fall as
1/SNR²**. `stage_fisher_star` stores I\* (Hz⁻² per step; shots enter as the factor k).

* **Focus ends at ε**, the half-width of the region around the best point where g² ≥ 0.9 max g² (η ≥ 1 − δ with
  δ = 0.1). For a Lorentzian η ≈ 1 − 18 (ε/Γ)², so ε = √(0.1/18) Γ ≈ 0.075 Γ; the code finds it numerically from the
  true curve (`stage_eps_target_phys`), so it holds for any line shape. Oracle: `stage2_oracle_steps` = (ε⁻² − J₀)/I\*,
  about 100/SNR² for a Lorentzian when J₀ is small (the constant depends on the SNR and width definitions; the exponent is
  the point). J₀ is the prior, see below.
  Not defined for drive-controlled models (the best point is a drive setting; pulse fringes can be narrower than the grid).
* **Refine starts when the claimed σ has stayed ≤ ε** for a few steps (`refine_start_step`; ε taken on the line as the run
  estimated it, so the boundary needs no true parameter and applies to an experiment). **It ends at the stop target τ**
  (`stage_tau_phys`). Oracle: `stage3_oracle_steps` = (τ⁻² − ε⁻²)/I\*. It scales like (FWHM_true / (SNR τ))²: the Scaling
  view's refine panel axis.
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
the claimed σ at focus start was already 0.05-0.25 Γ (not ≈ the FWHM) while the true error was 2-4× larger (overconfidence);
focus steps grew about 2× from window 5 to 10 × FWHM; SBED search steps were far flatter in the window than proportional.
With the focus stage's mean η only 0.3-0.5, the stage length looks set by exploration the tail rule counts, not by information.
(These were measured with the old boundaries: search→focus at a 0.5 false-alarm dip scan, refine = first step whose whole
remaining tail had mean η ≥ 0.9. In those rows the true error was already ≤ ε at "dip found" in about a third of SBED runs,
and refine was reached in about 15% of runs. The boundaries above replace both; the old `max_info_step` column is now
`refine_start_step`, so older rows show a dash and need a re-run.)

## The prior is information

A search window is a prior: a uniform window of width W says "the dip is inside, nowhere in particular", so it already
holds information before any shot. The Cramér-Rao line σ = 1/√(N I\*) is therefore not the reference for a Bayesian
posterior; the next sections give the prior, the stages it separates, and the bound that includes it.

### Information in the window: why σ₀² = W²/12

A uniform prior p = 1/W on [x₀ − W/2, x₀ + W/2] has variance

$$\sigma_0^2 = \int_{-W/2}^{W/2} \frac{u^2}{W}\,du = \frac{1}{W}\cdot\frac{2\,(W/2)^3}{3} = \frac{W^2}{12},
\qquad \sigma_0 = \frac{W}{\sqrt{12}} \approx 0.29\,W .$$

It is W² (Hz²), not W. The prior information scale is J₀ = 12/W² (Hz⁻²). A uniform prior has no finite Fisher
information at its edges, so the inverse variance is used as the convention: after N measurements 1/σ² = 12/W² + Σ I_j.
Fisher information is not unitless (Hz⁻²); the unitless quantity is the gain log₂(σ₀/σ) = log₂(W/(√12 σ)) bits.
`stage_prior_info` stores J₀ as σ₀⁻² with σ₀ the claimed σ after the first measurement (the prior to one measurement's
accuracy).

* Window widths are not comparable without J₀: a narrower window starts lower for free. A focus window built from the
  search posterior was paid for with shots; a window placed with the truth (e.g. the Zeeman edge pad) is information the
  locator did not earn.
* The focus oracle is `(ε⁻² − J₀)/I*`, not "from scratch". Refine already started from ε⁻². Together:
  `(τ⁻² − J₀)/I*` for the whole run.

### When the three stages are distinct

The stages are regimes of one continuous accumulation of information; the boundaries are thresholds (δ = 0.1, the dip
false-alarm rate, the stop target τ). Whether all three show depends on the run:

* **Search** is visible only for large W/Γ: its plateau lasts about W/Γ points. If σ₀ ≲ Γ, i.e. W ≲ 3.5 Γ, there is no
  plateau, only focus and refine.
* **Focus** needs a gap between "dip seen" and ε = 0.075 Γ. At low SNR search and focus merge; at very high SNR focus is a
  few steps.
* **Refine** needs τ well below ε: its length is (τ⁻² − ε⁻²)/I\*. With τ ≈ ε the run stops on the cliff and has no −½
  tail. It also needs the locally linear regime (within ≈ 0.075 Γ of the best point).

Focus and refine are expected to be independent of W; only search grows with it. So a large window shows three stages and a
small one two.

### The CRLB with the real prior

$$\sigma_{\mathrm{CRLB}}(N) = \Big(\frac{12}{W^2} + N\,I^*\Big)^{-1/2}.$$

It starts at W/√12, bends where N I\* ≈ 12/W², then has slope −½ (van Trees form). A claimed σ below the plain
1/√(N I\*) at small N is expected; only a true error (averaged over repeats) below this curve would indicate a defect.
The Stages view's Whole run chart draws it from the run's own start (J₀ from the claimed σ after the first measurement,
every measurement counted); the focus and refine charts restart from each stage's own claimed σ. Not yet done: starting the
whole-run curve exactly at W/√12.

The stages are regimes of one continuous accumulation of information, J = 1/σ² from J₀. The Whole run scope therefore
also shows two stage-free charts (`renderHlInfoCurve`), both from the claimed σ: the **prior share** J₀/J (1 → 0, dashed =
J₀/(J₀ + N I\*)) and the **information rate** (J(N) − J(N′))/((N − N′) I\*), where 1 is every measurement at the
Fisher-optimal point and a value above 1 means the claimed σ fell faster than any measurements could justify
(overconfidence). Both are computed from the claimed σ, so they inherit its overconfidence; they say nothing about whether
the dip was found, and search (detection) is not a Fisher regime, so the first steps need care. The stage boundaries and
columns remain for the per-stage views.

A fit of three straight log-log segments to the claimed σ is not used: the bound above is the reference. On the
2026-11-02 check (3 use cases, SNR 5, 50 repeats each), the existing `refine_start_step` sat on the cliff, not on a −1/2
tail, in 72 of 149 Smart-SBED and 113 of 149 Smart-Sweep runs.

### Sensitivity and bandwidth

η = δB √T is a constant only when σ ∝ T^(−1/2) exactly (refine). With the gyromagnetic ratio γ_e ≈ 28 GHz/T,
θ = γ_e B along the NV axis, so σ_B = σ_θ/γ_e. The operational quantity is the first passage time of the true error to a
target, `stage_true_passage_step` = T(τ); the bandwidth is 1/T and `stage_eta_eff` = τ √T (Hz √steps) is the effective
sensitivity *at that target*. It tends to the refine sensitivity as τ → 0 and is not a constant of the locator. Per-step
time is not in the data yet, so T is in steps.

Stored only for runs recorded after these columns were added; re-run to get them (the cache has no belief snapshots).

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
* The boundaries inherit the claimed σ's overconfidence (on the SNR × window study σ is ≈ 2 ε after the first measurement
  while the true error is ≈ 5 ε, and a few runs start refine with the error still several ε). They are as good as the
  Sanity coverage; a σ-free observable for "position known to within ε" is still to find.
* The search→focus factor `FOCUS_SIGMA_DROP` = 0.7 is arbitrary (0.9 fires before the data have localised anything; 0.25 fires later than the true error reaches ε, at
  step 37 / 200); the stage boundaries also depend on δ = 0.1; a one-off sensitivity check (δ = 0.05 / 0.1 /
  0.2) that the locator ranking survives is still to do.
* The signed z = (estimate − truth) / claimed σ is not stored (only |error|), so σ honesty is checked against a
  half-normal, not N(0, 1).
* SBED-Drive against every fixed drive amplitude, including the best one in hindsight.
