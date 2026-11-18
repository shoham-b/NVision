# Article claim -> exact quantity -> metric (v2, 2026-11-18, master 67ed334e)

Article read: `theory.tex`, `methods.tex`, `results.tex` (all `\tbd`), SM titles. Strategies in the article: SBED and
SimpleSweep-optimal (k*, x0.5, x2). Smart-Sweep / Smart-SBED are NOT in the article (and are removed from the code).
Notation: `e` true error, `u` claimed sigma_hat (stored per-step series `s`,`e`,`u`,`z`, <= 80 points), `J0` =
`stage_prior_info`, `I*` = `stage_fisher_star` (= I_max), `n` measurements (k = 1 shot each in the simulator).
Status: **EXISTS** (master, file:line), **PARTIAL**, **MISSING** (-> this PR, `nvision/metrics/claims.py`),
**RE-RUN** (needs run-time data the cache does not hold).

| # | Claim (article) | Exact quantity | Status |
|---|---|---|---|
| 1 | Prior gap sqrt(J0/J0_var) = pi/sqrt3 for a uniform window (eq. vantrees discussion; `evaluation_metrics.md`) | `stage_prior_info / stage_prior_info_var` (both stored, `stages.py:135`) | MISSING (trivial) -> `claims.prior_gap` |
| 2 | Run-averaged error never below the bound, tracking/after-detection form sigma_vT(n) = (J0 + n I*)^-1/2 (eq. vantrees, Sec. protocol) | R(n) = 1 / mean_runs(e^2 (J0 + n I*)), true and claimed; "violated" if R > 1 + scatter | PARTIAL: R(n) exists only in the browser (`static/app.js` `hlInfoRatioAt`), nothing in Python, no flag -> `claims.bound_ratio`, `bound_violation` |
| 3 | Global search: mean over runs against the search bound eq. search-mean (finite reach) | mean e^2 vs (W - n dd)^3/(12 W); R_search = bound / mean e^2 | EXISTS `search_bounds.finite_reach_curve` (search_bounds.py:173), `mean_squared_error` (:262), `/api/search-bounds` |
| 4 | Low-SNR search bound eq. search-cost / change of measure | kappa_bar, change_of_measure_curve, survival S(n), kappa_eff | EXISTS `search_bounds.py:88,180,186,272` |
| 5 | Elimination model eq. elimination / after-detection / mean-curve (picture, not prediction) | `mean_curve` | EXISTS `search_bounds.mean_curve` (:198) |
| 6 | Three regimes: plateau near sigma_0, one detection step, -1/2 tail in the mean curve | plateau = rms e / sigma_0 before half the window is eliminated; n_half = first n with rms e <= sigma_0/2; drop factor across it; log-log slope of rms e on the late window | MISSING -> `claims.regimes`. (Per-run stage boundaries `stages.py` are claimed-sigma thresholds, a different object) |
| 7 | After each run's own detection: van Trees restarted at tau, J_tau = posterior information | J_tau, bound steps to lock | EXISTS `posterior.py` (`posterior_J_tau_phys`), `localize.py:57` |
| 8 | Localize cost lock - tau vs sweep | L, share locked | EXISTS `localize.py:32,50`, `refine_rates_runs._localize_summary` |
| 9 | Per-strategy (n0, q): sigma^-2 ~ q I_max (n - n0) late in the run (eq. strategy-numbers); q <= 1 | LS line of 1/mean(e^2) on a late window | EXISTS `refine_rates.fit_rate_line` (:177), `late_window` (:160), R^2 reported. **PARTIAL**: no test that the window is late/linear (R^2 is not a gate, bootstrap CI absent) -> `claims.linearity_gate` (two halves of the window must give overlapping q) + `claims.bootstrap_q` |
| 10 | q of what the measurements carry (placement) | mean eta = g^2/max g^2 after detection | EXISTS `refine_rates.place_efficiency` (:93), `refine_rates_runs.build_run_trace`; stored `stage{1,2,3}_sens_eff` |
| 11 | Sweep q = 1/R_inf, n0 ~ 0 | `exact_sweep_ratio`, `lorentzian_sweep_q` | EXISTS `refine_rates.py:41,56` (+ checked against fits in `refine_rates_runs`) |
| 12 | R_inf = I_max/I_sweep = 27 W/(8 pi Gamma) (eq. speedup), independent of SNR | formula vs exact (from the true line) vs q_err ratio SBED/sweep | EXISTS `refine_rates.lorentzian_speedup` (:51), `speedup_curve` (:247), `refine_rates_runs._speedup` (:466) |
| 13 | Cost to a precise target n0 + tau^-2/(q I_max) | predicted vs first-passage measured | EXISTS `refine_rates.predicted_cost` (:199), `cost_curve` (:230) |
| 14 | Information rate <= 1 for the true error; claimed rate > 1 means overconfident (Sec. protocol, Metrics) | (J(n) - J(n'))/((n-n') I_max), J = 1/mean e^2 and 1/mean u^2 | PARTIAL: claimed rate per run in the browser (`renderHlInfoCurve`); run-averaged true twin and late summary MISSING -> `claims.information_rate`, `rate_summary` |
| 15 | Coverage of the true error within 1 and 2 claimed sigma, "only once the posterior is Gaussian, or with particle quantiles" | share of runs with |e| <= u, <= 2u at each n; gated to n >= `refine_start_step`; Wilson interval | PARTIAL: Sanity view at the stop only (`app.js`); per-n, gated, with interval MISSING -> `claims.coverage_curve`. Series `z` (signed, `posterior.py:388`) gives the same without the |e| half-normal approximation -> `claims.coverage_from_z` |
| 16 | Coverage with quantiles of the particle distribution (PIT) | posterior CDF at the truth | RE-RUN (cache holds no particles; old branch had `claim_columns.py`). Not ported: touches `runner/metrics.py` / schema, outside a read-only PR. Listed in docs |
| 17 | SBED posterior Gaussian only in refine (negentropy) | `posterior_negentropy_tau`, series `g` | EXISTS `posterior.py` (`g`) |
| 18 | Share of localize measurements > 3 FWHM from the peak vs residual mass there (Sec. eig, results.tex) | `posterior_far_placed_share_stage2` vs `posterior_far_mass_tau` | EXISTS `posterior.py` (`x`, `m`, `FAR_FWHM`); cached rows from the fit-gap era carry it, older ones are None |
| 19 | Sensitivity eta = 4/(3 sqrt3) Gamma sqrt(t_s)/(gamma_e SNR) ideal; eta/sqrt(q) real (eq. sensitivity) | dimensionless eta_meas/eta_th = sqrt(I_max) rms e(n) sqrt(n) late, expected 1/sqrt(q) | MISSING -> `claims.sensitivity_ratio`. (`stage_eta_eff` = tau sqrt(T) is at the stop target and "not a constant", `stages.py:441`) |
| 20 | Bandwidth: 1/(n0 t_m) for loose targets | n0 | covered by 9 |
| 21 | Cost vs SNR, window (results (c)) | first passage at tau | EXISTS `stage_true_passage_step`, Cost & search panel |
| 22 | Stuck belief / overconfidence cause (Methods: miscalibration "at any N") | D = (SS_belief - SS_fit)/sigma_fit^2 | EXISTS `fit_gap.py:71` |
| 23 | Settings quoted in SM (N_p = 10,000, resample < 0.5 N_p, a = 0.85, SNR = C/sigma) | constants | audited by the align sessions (#305/#308), not re-done here |

What this PR adds (everything is computed from stored data; no re-run, no fingerprint / CACHE_SCHEMA_VERSION change):
rows 1, 2, 6, 9 (gate + bootstrap), 14, 15, 19 in `nvision/metrics/claims.py`, one report per use case and locator
(`claims_report.py`) served read-only at `GET /api/claims`. Not ported: PIT (16) and `stage_fisher_sweep` (the exact
sweep information is rebuilt from the true line by master's `refine_rates_runs`), both would need run-time columns.
