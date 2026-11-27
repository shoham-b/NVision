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
| **Search + localize** | the run's own dip test fires (`NVISION_DIP_P_FA`), held for 3 steps | `detection_step` | change-of-measure bound (Eq. search-bound); the elimination model (Eq. mean-curve) is a reference |
| **Converge** | the posterior is one Gaussian peak: negentropy stays < 0.05 | per-step `negentropy` series; the onset is computed (`detection_aligned.gaussian_onset`) | van Trees restarted at detection with the posterior's own information J_τ |
| **Refine** | the end of the run | — | 1/MSE grows as q·I_max·(n − n₀), q ≤ 1 |

The van Trees bound does **not** apply to the uniform global prior: its Fisher information is infinite (hard edges), so
van Trees from n = 0 says nothing. It holds from each run's detection, with J_τ (`j_tau_phys`, the KDE Fisher
information of the posterior at detection) taking the place of J₀ (Eq. after-detection). The prior enters only through
its spread σ₀ = W/√12 (Eq. prior), with W the range `zeeman_shift` is drawn from.

## Per-run quantities (stored at run time)

`nvision/metrics/run_axes.py` (`compute_run_axes`), identical for every locator:

* `detection_step`, `primary_param` (`zeeman_shift` for the NV generators);
* `i_max` = **I_max**: the Fisher information about the primary parameter of one measurement at the steepest
  point of the true dip (shots × max g²/σ², Hz⁻² per measurement; every locator takes one shot per measurement);
* the axes: `sigma_target_phys` (the stop target τ), `fwhm_phys` (Γ), `prior_fwhm_phys`,
  `window_phys` (the drive window), `window_over_fwhm`, `sigma0_phys` = σ₀, the prior's standard deviation
  (W/√12 when uniform), `prior_kind` (`uniform`: global search; `gaussian`: tracking), and `target_cost_step` (first step from which the true error stays ≤ τ).

`nvision/metrics/series.py` stores the per-step series `step`, `error`, `claimed_sigma`, `sigma_target` (and
`focus_windows` for the timeline); `nvision/metrics/posterior.py` adds `z_score` (signed error / σ̂), `pit` (the
posterior CDF at the truth), `negentropy`, `far_mass`, `posterior_fisher` (KDE Fisher information), `kde_bandwidth`,
`far_placed` and the scalars `j_tau_phys`, `j_tau_2h_phys`, `negentropy_tau`, `far_mass_tau`, `far_placed_share`,
`pit_final`. The names are the article's (table in [the glossary](glossary.md#metric-names-renamed-2026-11-24)); rows
recorded before the rename fail fast and need a re-run.

**Claimed uncertainty** σ̂_N is the standard deviation the locator reports after N measurements (the posterior's spread).
It is compared with the true error; it is not an accuracy measure on its own.

## The shared run record

Simulation and lab are the same metrics fed by two sources (`nvision/metrics/run_record.py`). One `RunRecord` per run
holds the stored steps, the estimate, the claimed σ and the error per step, the drive frequencies and outcomes (when the
source keeps them), the detection, I_max, W, Γ, SNR, the budget and the line-shape parameters. Every metric reads the
record; the two sources differ only in four explicit, named places, never in scattered branches:

| Difference | Simulation | Lab | Where |
|---|---|---|---|
| **Truth** (`TruthSource`) | `SIM_TRUTH`: the true parameter, drawn from the prior; ensemble quantities (MSE against the bounds, ρ, the fitted rate, PIT) are valid | `LAB_REFERENCE`: the reference sweep fit, whose standard error adds in quadrature and whose 0.4–0.6 MHz drift floors every error; only claimed quantities (q̂, placement, claimed-σ search cost, scatter of repeated runs) are valid | `RunRecord.truth.ensemble_valid`; the Lab page shows no ensemble panel |
| **Detection** (`Detector`) | `DIP_TEST`: the run's dip test fires (stored as `detection_step`); restart at the posterior before that measurement (its J_τ is stored) | `SIGMA_DROP`: the first steep, lasting drop of the claimed σ (`lab_search_fit.detection_step`); restart at the first posterior after the drop | `detection_aligned.align(record, detector)` |
| **Nuisances** | contrast and width inferred | held fixed (known) | `RunRecord.known_nuisances`; an input of I_max only |
| **Stop and budget** | every run measures the whole 2500 | set 1 stops at claimed σ < 1 MHz or 300; set 2 measures the full 300 | `RunRecord.stop_rule`, `budget`; only lags every run reaches are used |

`SIGMA_DROP` needs only the claimed σ, so it also runs on the simulated runs: `detector_comparison` of
`/api/detection-aligned` reports, per use case and locator, the median step of each definition and their difference
over the runs both detect (the Simulation page's *Two detection definitions* panel). On a simulated run the sigma drop
is located to the stored steps (at most 80 per run). First reading (span 5, SNR 5): the sigma drop fires about 2× later
than the dip test (median 12 vs 5 for SBED, 34 vs 16 for the sweep).

Three panels are one component on both pages (`static/claim-panels.js`), each page feeding it through its adapter:
*Search cost vs window* (Eq. search-cost), *Information per measurement after detection* (q̂ from `claimed_rate` of
`detection_aligned`, the same function for both sources, and the placement) and *Claimed σ vs the spread of the runs*
(the rms true error over the prior ensemble in the simulation, the scatter of repeated runs in the lab).

## The claims

Every number below is computed from stored data (per-step series and scalars), so `nv serve` on an existing cache is
enough: no re-run and no `nv render`. A number the data cannot give is `None` with the reason (`why` / `unavailable`).

### 1. Error over the run (Fig. overview)

The run-averaged error stays near the prior W/√12 while the runs search, then falls as 1/√n at q × the data limit
(n I_max)^(-1/2). Reference curves: the elimination model (Eq. mean-curve) and the change-of-measure bound.

* Data: `rms_curve` of `GET /api/claims` (rms over the runs of the true error and of σ̂ on a common step grid, `rms_curve.error` / `.claimed_sigma`;
  a stopped run keeps its last value), `regimes` (median detection step, median Gaussian onset, converge end, plateau,
  tail slope).
* **Moving line.** In every run-group signal the true `zeeman_shift` random-walks during the run (step std q Hz per
  measurement, run axis `line_walk_step_phys`, setting `NVISION_LINE_JITTER_STEP_FWHM`). The locators assume a still
  line. Every per-step error, z-score, PIT and milestone is scored against the truth **at that step**
  (`RunResult.truth_trace`); `true_params` and the `true_*` columns hold the position at the first measurement.
  The bound is the recursion J_n = 1/(1/J_{n-1} + q²) + I_max (`moving_line_bound`; J_n = J₀ + n I_max when q = 0),
  whose error levels off near (q/√I_max)^(1/2). It is used by the tracking bound below and by the detection-restarted ρ.
  **Still computed for a still line** (read them as references): the data limit, the elimination and
  change-of-measure search curves, `i_max`, the refine-rate theory and placement offsets (relative to the start
  position; the walk is typically about 0.015 FWHM over 2500 measurements). With a moving line 1/MSE saturates, so the
  late-window q fit can fail its linearity gate.
* Tracking (`prior_kind = gaussian`, the `article-tracking` group): the prior holds J₀ = 1/σ₀², so the bound applies
  from the first measurement; the panel draws it (`van_trees_from_start.bound_sigma`) and reports
  ρ = bound variance / MSE ≤ 1.
  σ₀ here is the std of the honest Gaussian prior the truth is drawn around (the generator's
  `true_params.bounds._priors.zeeman_shift`), not the spread of the locator's belief, which mixes a broad share
  (`PRIOR_BROAD_FRACTION`) over the whole window into it; the belief's spread is reported as `locator_prior_sigma_phys`.
  `/api/claims` returns `van_trees_from_start` (ρ per n, `rho_at`, `rho_max` with its bootstrap CI, `violated` when the
  CI's lower end exceeds 1). The detection-restarted ρ is not the tracking test: with the shape known the dip test fires
  after the posterior is already Gaussian.
* Panel: *Error over the run* (one use case). The Line picker also offers the `article-lab-mirror` family (one
  generator per lab window, with `Bayesian-SBED-KnownShape`) and the tracking family (one per prior width).

### 2. Search against the bounds (Fig. search)

During the search no strategy, adaptive or not, beats the change-of-measure bound, Eq. search-bound
W²/(8e)·e^{−nκ̄}/(1+ln2+nκ̄)², with κ̄ = k∫dip²dx/(2σ²W) the mean KL rate per measurement. At high SNR the finite-reach
form (W − nΔd)³/(12W) (Eq. search-mean) and the elimination model also hold, Δd the single-measurement reach (eq.
sm-reach). The search ratio ρ_search(n) = bound MSE / measured MSE must stay ≤ 1 before detection.

* Code: `nvision/metrics/search_bounds.py`, `GET /api/search-bounds` (`nvision/cli/search_bounds_api.py`). The rate is
  reported as `n_search` = 1/κ̄, the measurements per e-fold of the bound (Eq. search-cost). W is the
  range `zeeman_shift` is drawn from (checked against the generator). κ̄ is integrated on the true line rebuilt from the
  generator, so it holds for any line shape. SNR = line contrast / σ (the article convention), computed from σ.
* Also returned: the survival S(n) of |error| > one HWHM, `n_search_eff` = 1/κ_eff from its decay and
  `search_efficiency` = `n_search` / `n_search_eff`, and the detection summary.
* Panel: *Search against the bounds* (one use case).

### 3. Converge: van Trees restarted at detection (Fig. localize)

After its own detection n_d each run is compared with 1/(J_τ + k I_max), k = n − n_d. The localize ratio
ρ(k) = mean bound / mean squared true error must stay ≤ 1 (`bound_violated` flags a lower bootstrap limit above 1.01);
ρ near 1 means information at the data limit. The posterior becomes one Gaussian peak once its negentropy stays < 0.05.

* Code: `nvision/metrics/detection_aligned.py`, `GET /api/detection-aligned`. Also the elimination-model curve with
  J_d = 12/Δd² (`rho_elimination`), the claimed rate q̂ (`claimed_rate`; `claimed_rate_late` is the lab's definition)
  and the far placement. ρ is `rho_van_trees` (`rho_van_trees_at`, `rho_van_trees_max`).
* Panels: *After detection: ρ against restarted van Trees*, *Posterior negentropy after detection* (one use case).

### 4. Refine: the information rate q (Fig. strategy_plane)

Late in a run 1/MSE = q·I_max·(n − n₀) (Eq. strategy-numbers). SBED's q is about 0.5 at every window and SNR. A sweep
spreads its measurements over the window and gets q_sweep = I_sweep/I_max = 8πΓ/(27W) (Eq. sweep-info).

* `fitted_rate` (q from the true error, with `search_offset` = n₀): `claims.fit_with_gate` = `refine_rates.fit_rate_line` on the late window, with bootstrap intervals
  and a **linearity gate** (the q of the two halves of the window must agree); `linear = false` means the late regime is
  not reached and q is not a refine rate. Also `claimed_rate` (the same on σ̂) and `claimed_over_fitted_rate`
  (above 1 = overconfident).
* `placement_efficiency` (placement): the mean η = I(d)/I_max of the measurements after detection, judged on the true signal
  (`nvision/metrics/refine_rates_runs.py`, `GET /api/refine-rates`; SBED's positions from its scan figure or
  `meas_drive_freq_unit`, a sweep's from its design). `fitted_over_placement` is what the filter keeps of the information the
  placement offered.
* Offset histogram: where the measurements after detection fall from the nearest true line centre, in units of the true
  HWHM γ; the information peaks at ±γ/√3.
* Panels: *Information rate q vs window* (every use case), *Where measurements fall after detection* (one use case).

### 5. Speedup over the sweep (Eq. speedup)

For a precise target the speedup is the ratio of the rates, R = q_SBED/q_sweep ≈ 1.07 q W/Γ (27/(8π) W/Γ at q = 1),
independent of SNR.

* Data: `speedup` of `GET /api/refine-rates`: `speedup_fitted` (ratio of the fitted q, the best sweep spacing as yardstick),
  `speedup_placement`, and `ratio` = T_sweep(τ)/T_SBED(τ) on targets τ = σ₀/gain, σ₀ = W/√12, counted where at least half of
  each locator's runs reached τ (the cost of a target is the first step from which the true error stays ≤ τ, median over
  all runs).
* Panel: *Speedup over the sweep vs window*: measured R∞ and the cost ratio at the most precise common target, against
  1.07 q W/Γ with SBED's own q.

### 6. Calibration: coverage, not end-of-run accuracy

The accuracy at the end of a run is not a quality metric: a run stops once σ̂ is below the target, so its final error
mostly reflects the target. What is tested is whether σ̂ is honest:

* rms true error ÷ rms σ̂ over the run (`rms_curve`); 1 = honest on average.
* Coverage from the signed z (`coverage_z_final`: share of runs within 1 and 2 σ̂, Wilson intervals, mean and std
  of z) and, per n, `coverage_curve` counted from each run's Gaussian onset, since σ̂ is a 68 % interval only once the
  posterior is Gaussian (`within_k_all` without the gate, `gaussian_share` the share of runs that got there).
* Coverage by the posterior's own quantiles (PIT, `coverage_pit_final` from the series `pit`): valid before the
  posterior is Gaussian. Runs recorded before the `pit` series show "needs re-run".
* Panels: *True error vs claimed σ over the run* (one use case), *Coverage at the end of the run vs window*,
  *Coverage by the posterior's own quantiles (PIT)*.

## Lab claims (Lab page)

The lab runs are the lab's own MATLAB BayesianESR, not NVision's SBED, and the page labels them so. They go through
the same record and functions (`lab_detection_aligned.record_from_trace`), with the lab's
truth (`LAB_REFERENCE`) and detector (`SIGMA_DROP`); only claimed quantities are reported.

| Claim | Data | Panel |
|---|---|---|
| The search cost grows linearly in W at about 25× Eq. search-cost, n_search ≳ (8/π)(W/Γ)/(k SNR²) | median detection step per window (`GET /api/lab-detection-aligned`, `nvision/metrics/lab_detection_aligned.py`), Γ and SNR = C/σ of the runs' own model and noise | *Search cost vs window* |
| After detection q̂ and the placement are near 0.4–0.5 in narrow windows and fall to the sweep rate 8πΓ/(27W) above about 25 Γ | `claimed_rate.claimed_rate` (slope of 1/⟨σ̂²⟩ per measurement over the second half of the lags, in units of I_max) and `placement.placement_efficiency_median` | *Information per measurement after detection* |
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
| Per-run markers and axes | `nvision/metrics/run_axes.py` | stored in the results row |
| Shared run record, truth sources, detectors | `nvision/metrics/run_record.py` | — |
| Posterior series | `nvision/metrics/posterior.py`, `series.py` | stored in the scan entry |
| Regimes, (n₀, q), coverage, rms curves | `nvision/metrics/claims.py`, `claims_report.py` | `/api/claims` |
| Search bounds | `nvision/metrics/search_bounds.py`, `nvision/cli/search_bounds_api.py` | `/api/search-bounds` |
| Van Trees from detection, negentropy | `nvision/metrics/detection_aligned.py` | `/api/detection-aligned` |
| Placement, offsets, speedup | `nvision/metrics/refine_rates.py`, `refine_rates_runs.py` | `/api/refine-rates` |
| Lab | `nvision/metrics/lab_*.py` | `/api/lab-study`, `/api/lab-detection-aligned` |
| Panels shared by both pages | `static/claim-panels.js` | — |
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
