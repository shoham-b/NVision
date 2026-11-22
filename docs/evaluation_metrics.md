# Evaluation metrics: the article's claims and how each is tested

Read this before adding or changing any locator-comparison metric or results view. Every metric here tests one claim of
the article (`articles/sections/theory.tex`, `methods.tex`; Overleaf is the source of truth). A metric that does not
test a claim does not belong in the comparison. The UI has one panel per claim, on the **Simulation** page (simulated
runs) and the **Lab** page (the lab's own runs), each with the theory curve beside the data.

## What the comparison is for

**Bayesian SBED is the subject**, and `SimpleSweep-optimal` (a uniform sweep at the task's optimal grid step k\*) is the
yardstick. `SimpleSweep-optimal-x0.5` and `-x2` check that k\* is the cheapest step; the speedup is always measured
against the best of the three. The scenario: the line shape is known, the magnetic field moved, and the resonance must be
found again in a drive window of width W. The `article` run group is the Lorentzian line
`NVCenter-voigt-span<W>-gs0-hfunresolved` at seven windows and five SNRs, SBED and `SimpleSweep-optimal`, 50 repeats each;
the Simulation page opens on it.

## The four regimes of a run

A run passes four regimes. Their boundaries are read from **each run's own history**, never at a fixed number of
measurements after the start, because runs find the dip at different times:

| Regime | Ends at | Stored as | Bound that applies |
|---|---|---|---|
| **Search + localize** | the run's own dip test fires (`NVISION_DIP_P_FA`), held for 3 steps | `dip_detected_step` | change-of-measure bound (Eq. search-bound); the elimination model (Eq. mean-curve) is a reference |
| **Converge** | the posterior is one Gaussian peak: negentropy stays < 0.05 | per-step negentropy series `g`; the onset is computed (`detection_aligned.gaussian_onset`) | van Trees restarted at detection with the posterior's own information J_τ |
| **Refine** | the end of the run | — | 1/MSE grows as q·I_max·(n − n₀), q ≤ 1 |

The van Trees bound does **not** apply to the uniform global prior: its Fisher information is infinite (hard edges), so
van Trees from n = 0 says nothing. It holds from each run's detection, with J_τ (`posterior_J_tau_phys`, the KDE Fisher
information of the posterior at detection) taking the place of J₀ (Eq. after-detection). The prior enters only through
its spread σ₀ = W/√12 (Eq. prior), with W the range `zeeman_shift` is drawn from.

## Per-run quantities (stored at run time)

`nvision/metrics/stages.py` (`compute_stage_metrics`), identical for every locator:

* `dip_detected_step`, `stage_primary_param` (`zeeman_shift` for the NV generators);
* `stage_fisher_star` = **I_max**: the Fisher information about the primary parameter of one measurement at the steepest
  point of the true dip (shots × max g²/σ², Hz⁻² per measurement; every locator takes one shot per measurement);
* the axes: `stage_tau_phys` (the stop target τ), `stage_fwhm_true_phys` (Γ), `stage_prior_fwhm_phys`,
  `stage_window_phys` (the drive window), `stage_window_fwhm`, `stage_prior_info_var` = 1/Var(prior)
  (12/W² when uniform), and `stage_true_passage_step` (first step from which the true error stays ≤ τ).

`nvision/metrics/posterior.py` adds the posterior series per step — `z` (signed error / σ̂), `g` (negentropy), `m`, `j`
(KDE Fisher information), `h`, `x` (far placement) — and the scalars `posterior_J_tau_phys`, `posterior_J_tau_2h_phys`,
`posterior_negentropy_tau`, `posterior_far_mass_tau`, `posterior_far_placed_share`. The PIT series `p` and
`posterior_pit_final` (posterior CDF at the truth) are being added in a separate change; runs recorded before it carry
none.

**Claimed uncertainty** σ̂_N is the standard deviation the locator reports after N measurements (the posterior's spread).
It is compared with the true error; it is not an accuracy measure on its own.

## The claims

Every number below is computed from stored data (per-step series and scalars), so `nv serve` on an existing cache is
enough: no re-run and no `nv render`. A number the data cannot give is `None` with the reason (`why` / `unavailable`).

### 1. Error over the run (Fig. overview)

The run-averaged error stays near the prior W/√12 while the runs search, then falls as 1/√n at q × the data limit
(n I_max)^(-1/2). Reference curves: the elimination model (Eq. mean-curve) and the change-of-measure bound.

* Data: `rms_curve` of `GET /api/claims` (rms over the runs of the true error `e` and of σ̂ `u` on a common step grid;
  a stopped run keeps its last value), `regimes` (median detection step, median Gaussian onset, converge end, plateau,
  tail slope).
* Panel: *Error over the run* (one use case).

### 2. Search against the bounds (Fig. search)

During the search no strategy, adaptive or not, beats the change-of-measure bound, Eq. search-bound
W²/(8e)·e^{−nκ̄}/(1+ln2+nκ̄)², with κ̄ = k∫dip²dx/(2σ²W) the mean KL rate per measurement. At high SNR the finite-reach
form (W − nΔd)³/(12W) (Eq. search-mean) and the elimination model also hold, Δd the single-measurement reach (eq.
sm-reach). The search ratio ρ_search(n) = bound MSE / measured MSE must stay ≤ 1 before detection.

* Code: `nvision/metrics/search_bounds.py`, `GET /api/search-bounds` (`nvision/cli/search_bounds_api.py`). W is the
  range `zeeman_shift` is drawn from (checked against the generator). κ̄ is integrated on the true line rebuilt from the
  generator, so it holds for any line shape. SNR = line contrast / σ (the article convention), computed from σ.
* Also returned: the survival S(n) of |error| > one HWHM, its decay rate κ_eff against κ̄, and the detection summary.
* Panel: *Search against the bounds* (one use case).

### 3. Converge: van Trees restarted at detection (Fig. localize)

After its own detection n_d each run is compared with 1/(J_τ + k I_max), k = n − n_d. The localize ratio
ρ(k) = mean bound / mean squared true error must stay ≤ 1 (`bound_violated` flags a lower bootstrap limit above 1.01);
ρ near 1 means information at the data limit. The posterior becomes one Gaussian peak once its negentropy stays < 0.05.

* Code: `nvision/metrics/detection_aligned.py`, `GET /api/detection-aligned`. Also the elimination-model curve with
  J_d = 12/Δd², the claimed rate q̂ and the far placement.
* Panels: *After detection: ρ against restarted van Trees*, *Posterior negentropy after detection* (one use case).

### 4. Refine: the information rate q (Fig. strategy_plane)

Late in a run 1/MSE = q·I_max·(n − n₀) (Eq. strategy-numbers). SBED's q is about 0.5 at every window and SNR. A sweep
spreads its measurements over the window and gets q_sweep = I_sweep/I_max = 8πΓ/(27W) (Eq. sweep-info).

* `q` (true error): `claims.fit_with_gate` = `refine_rates.fit_rate_line` on the late window, with bootstrap intervals
  and a **linearity gate** (the q of the two halves of the window must agree); `linear = false` means the late regime is
  not reached and q is not a refine rate. Also `fit_claimed` (the same on σ̂) and `rate_ratio_claimed_over_true`
  (above 1 = overconfident).
* `q_place` (placement): the mean η = I(d)/I_max of the measurements after detection, judged on the true signal
  (`nvision/metrics/refine_rates_runs.py`, `GET /api/refine-rates`; SBED's positions from its scan figure or
  `meas_drive_freq_unit`, a sweep's from its design). `q_err / q_place` is what the filter keeps of the information the
  placement offered.
* Offset histogram: where the measurements after detection fall from the nearest true line centre, in units of the true
  HWHM γ; the information peaks at ±γ/√3.
* Panels: *Information rate q vs window* (every use case), *Where measurements fall after detection* (one use case).

### 5. Speedup over the sweep (Eq. speedup)

For a precise target the speedup is the ratio of the rates, R = q_SBED/q_sweep ≈ 1.07 q W/Γ (27/(8π) W/Γ at q = 1),
independent of SNR.

* Data: `speedup` of `GET /api/refine-rates`: `r_inf_err` (ratio of the fitted q, the best sweep spacing as yardstick),
  `r_inf_place`, and `ratio` = T_sweep(τ)/T_SBED(τ) on targets τ = σ₀/gain, σ₀ = W/√12, counted where at least half of
  each locator's runs reached τ (the cost of a target is the first step from which the true error stays ≤ τ, median over
  all runs).
* Panel: *Speedup over the sweep vs window*: measured R∞ and the cost ratio at the most precise common target, against
  1.07 q W/Γ with SBED's own q.

### 6. Calibration: coverage, not end-of-run accuracy

The accuracy at the end of a run is not a quality metric: a run stops once σ̂ is below the target, so its final error
mostly reflects the target. What is tested is whether σ̂ is honest:

* rms true error ÷ rms σ̂ over the run (`rms_curve`); 1 = honest on average.
* Coverage from the signed z (`coverage_z_at_budget`: share of runs within 1 and 2 σ̂, Wilson intervals, mean and std
  of z) and, per n, `coverage_curve` counted from each run's Gaussian onset, since σ̂ is a 68 % interval only once the
  posterior is Gaussian (`within_k_all` without the gate, `gaussian_share` the share of runs that got there).
* Coverage by the posterior's own quantiles (PIT): valid before the posterior is Gaussian. Read from
  `coverage_pit_at_budget` once the claims report carries it; until the runs store `posterior_pit_final` the panel says
  "needs re-run".
* Panels: *True error vs claimed σ over the run* (one use case), *Coverage at the end of the run vs window*,
  *Coverage by the posterior's own quantiles (PIT)*.

## Lab claims (Lab page)

Lab runs have no ground truth and store no per-step particles. The reference is the median final mean of the converged
runs (`nvision/metrics/lab_study.py`); the line drifts 0.4–0.6 MHz between recordings, which floors every error. Detection
is the run's first steep, lasting drop of its claimed σ (`lab_search_fit.detection_step`).

| Claim | Data | Panel |
|---|---|---|
| The search cost grows linearly in W at about 25× Eq. search-cost, n_search ≳ (8/π)(W/Γ)/(k SNR²) | median detection step per window (`GET /api/lab-detection-aligned`, `nvision/metrics/lab_detection_aligned.py`), Γ and SNR = C/σ of the runs' own model and noise | *Search cost vs window* |
| After detection q̂ and the placement are near 0.4–0.5 in narrow windows and fall to the sweep rate 8πΓ/(27W) above about 25 Γ | `claimed_rate.q_hat` (slope of 1/⟨σ̂²⟩ per measurement over the second half of the lags, in units of I_max) and `placement.efficiency_median` | *Information per measurement after detection* |
| The claimed σ matches the scatter of repeated runs | per window: median claimed σ, across-run std of the final estimates, within 1σ / 2σ (`GET /api/lab-study`) | *Claimed σ vs repeated-run scatter* and its table |

The per-window claimed-σ curves against the elimination model fitted at import (`lab_search_fit`) and the per-run
replay stay on the page for inspection.

## Rules

1. **Read localize and refine from each run's own detection.** Never compare locators at a fixed lag after the start, and
   never at "detection + k" without aligning each run at its own detection (the detection time itself differs).
2. **No van Trees from n = 0 for a global search.** The bound holds from detection with J_τ; before detection use the
   search bounds.
3. **The yardstick is the best sweep spacing**, never an arbitrary one.
4. **One budget for all.** Every locator gets the same measurement budget and stop target, so costs compare.
5. **No end-of-run accuracy as a quality metric.** Calibration is judged by coverage.
6. **Raw metrics before ranking.** Show each locator's number with its interval before any summary or ranking.
7. **One use case at a time, then the median over use cases.** A use case is one simulated signal at one noise level;
   repeats share the true signal across locators, so locators are compared run by run. Never pool runs across use cases.

## Where it lives

| What | Code | Endpoint |
|---|---|---|
| Per-run markers and axes | `nvision/metrics/stages.py` | stored in the results row |
| Posterior series | `nvision/metrics/posterior.py`, `series.py` | stored in the scan entry |
| Regimes, (n₀, q), coverage, rms curves | `nvision/metrics/claims.py`, `claims_report.py` | `/api/claims` |
| Search bounds | `nvision/metrics/search_bounds.py`, `nvision/cli/search_bounds_api.py` | `/api/search-bounds` |
| Van Trees from detection, negentropy | `nvision/metrics/detection_aligned.py` | `/api/detection-aligned` |
| Placement, offsets, speedup | `nvision/metrics/refine_rates.py`, `refine_rates_runs.py` | `/api/refine-rates` |
| Lab | `nvision/metrics/lab_*.py` | `/api/lab-study`, `/api/lab-detection-aligned` |
| Simulation page | `static/article-view.js` (plots from `static/detection-aligned-view.js`) | — |
| Lab page | `static/lab-view.js` | — |

The first view of a use case reads every run from the cache (up to a minute); the server memoises the result until
`/api/reload`.

## Removed (2026-11-22)

These contradicted the article and are gone from the code, the API and the UI: the σ-drop stage model
(`focus_found_step`, `refine_start_step`, `stage{1,2,3}_*`, the ε ≈ 0.075 Γ localize/refine oracles
`stage2/3_oracle_steps`, `*_efficiency`, `*_wasted_share`, `stage_eta_eff`, the Fisher-efficiency tail rule), the lock
steps (`lock_step_true`, `lock_step_claimed`, `nvision/metrics/localize.py`), the milestone metrics (`*_at_milestone`,
`*_at_all_converged`), van Trees from n = 0 with J₀ = 4π²/W² (`stage_prior_info`, the R(n) ratio and the n₀/e cost
ladder) and the Stages / Information, Scaling and Sanity views. Rows recorded before still carry those columns; nothing
reads them. The stop rule's own convergence test (`milestones.target_params_converged`) and the converged-step markers
of the Run view are locator behaviour, not metrics, and stay.
