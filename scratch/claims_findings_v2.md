# What the cached data say about the article's claims (v2, 2026-11-18)

**Data (read-only; cache and artifacts untouched).** Pure-Lorentzian SNR x window study
`NVCenter-voigt-span{1.6,2,3,5}-gs0-hfunresolved`, SNR 1/2/5/10/20, SBED and `SimpleSweep-optimal`, `-x0.5`, `-x2`,
**50 repeats, physics fingerprint `196210189ea7`** (recorded 2026-11-18 01:00-03:40: 10,000 particles, Liu-West a = 0.85,
resample below 0.2 N_p, article SNR = C/sigma; 77 use-case x locator combos, 17 of them SBED: three of the 20 SBED use
cases were not in the cache). Run with `scratch/claims_v2_run.py <fingerprint> <out.json>` (summaries
`claims_v2_summary.py`, `claims_v2_summary2.py`; results `claims_results_v2.json`).
The **current** fingerprint `ad11261c13ae` (resample below 0.5 N_p, #333) is only partly cached: sweeps 50 repeats, but SBED
only 12-15 repeats in four use cases (`claims_results_v2_cur.json`). That is too thin for medians; it is used only as a
direction check below. Every run measures the whole 2500-measurement budget; series are stored at <= 80 points per run.

Numbers are medians over use cases (rule 7) unless a range is given. "q" = late-window slope of 1/MSE in units of
`I_max` (`stage_fisher_star`), fitted on `[max(median dip step, n_hi/2), n_hi]` = [1216, 2432] (median).

| # | Article claim | What the data say | Verdict |
|---|---|---|---|
| 1 | Gap before any measurement sqrt(J0/J0_var) = pi/sqrt3 = 1.81; truths uniform over the prior | stored gap 1.814 in every use case; rms truth / (W/sqrt12) = 1.00 (SBED 1.001) | **supported** (the rms-truth ratio is the part that could fail) |
| 2 | Run-averaged true error never below the van Trees bound | R(n) <= 0.54 everywhere (max over use cases and n in {1,10,100,1000}); 0 of 77 violate (interval lower end above 1.01) | **supported**, but the bound is not tight: R(1) = 0.22-0.24, R(1000) = 0.12-0.31 |
| 2' | The mean-square error approaches the bound at large n (data limit) | R(1000): sweeps 0.12-0.20, SBED 0.31 (max 0.54), i.e. the mean-square error is 3x (SBED) to 5-8x (sweeps) above the bound at the end of the budget; R(n) does not trend to 1 | **not shown**: no strategy reaches the data limit within 2500 measurements in this study |
| 3 | Three regimes in the mean curve: plateau at the prior, one detection step, -1/2 tail | plateau level 0.76-0.98 sigma_0 before n_half/2 (consistent); no cliff in a run average: drop across n_half only 1.04-1.09 (detection steps differ per run and smear; the article calls the elimination model a picture of one run); log-log tail slope -0.38 to -0.58 (SBED -0.54, sweeps -0.52 / -0.38 / -0.46) | plateau **consistent**; cliff **not visible in a mean** (as the article says); tail slope **roughly -1/2**, but it is steeper than -1/2 by n/(n - n0) so it is not a clean test |
| 4 | 1/MSE linear in n late in the run (Eq. strategy-numbers) | linearity gate passes in 76 of 77 fits (one x2 sweep fails) **but its power is low**: a half of the window is 4-5 grid points of ~10 stored samples, so the half-window q intervals are wide; R^2 of the line 0.95-0.97 | **consistent, weakly tested**. Not demonstrated that this window is the late regime (see 5) |
| 5 | Sweeps: q = 1/R_inf = (8 pi / 27) Gamma/W (Lorentzian W >> Gamma form) | fitted q / that form: sweep-optimal 0.53 (range 0 to 2.1), x0.5 0.21, x2 0.40. Not close, and the form is not valid at W/Gamma = 1.6-5 (use the exact ratio of the Refine & speedup section) | **not confirmed at these windows**; window too narrow for the W >> Gamma formula. The earlier "sweeps match theory" is not reproduced by this form |
| 6 | q <= 1 for every strategy; SBED q near 1 | q (true error) for SBED: median 0.26, range 0.01-0.59; each use case's bootstrap interval has its upper end < 1 (17/17); sweeps median 0.08-0.23 (range -0.001 to 0.64) | q <= 1 **holds**; "SBED near 1" **contradicted** (the same earlier finding: SBED carries ~45% of I_max by placement, 26% by error decay) |
| 7 | Speedup R_inf = I_max / I_sweep ~ 1.07 W/Gamma (3.2 median over these use cases, 1.7-5.4) | q_SBED / q_sweep-optimal, span 1.6: 1.3-1.4 (predicted 1.7); span 2: 0.4-0.8 (one outlier 214 from a near-zero sweep q; predicted 2.2); span 3: 0.2-0.4 (and 2.8; predicted 3.2); span 5: 2.6-5.9 (one outlier 366; predicted 5.4) | **in practice 1x or lower at W/Gamma <= 3, 3-6x at W/Gamma = 5**: the predicted rise with W/Gamma is visible only at the widest window; sweep q is noisy where it is near zero |
| 8 | Claimed information rate > 1 means overconfident; claimed sigma honest | claimed rate / true rate on the same window: SBED 1.9, sweeps 1.2 / 2.3 / 1.7: the claimed sigma falls faster than the error. Claimed-sigma R(1000) 0.42 (SBED) vs true 0.31 | **overconfident, all four** (the SBED and x0.5 / x2 sweeps most) |
| 9 | Coverage 68 % / 95 % (within sigma_hat / 2 sigma_hat) | at the budget, all runs: SBED 0.62 / 0.92, sweeps 0.61-0.66 / 0.88-0.91; z = (estimate - truth)/sigma_hat: SBED 0.62 / 0.92 with std(z) = 1.44 (sweeps 1.12-1.24); by SNR (SBED) 1: 0.61/0.88, 2: 0.62/0.93, 5: 0.58/0.90, 10: 0.66/0.93, 20: 0.65/0.92. `refine_start_step` is set in 100 % of runs, so the "Gaussian only in refine" gate is vacuous | **below nominal (overconfident), SBED worst in spread (z std 1.4)**; mean z ~ 0 (no bias). Particle-quantile (PIT) coverage **cannot be tested** (needs a re-run) |
| 10 | Sensitivity eta = (article) / sqrt(q) | measured / article: SBED 1.8, sweeps 2.2 / 3.0 / 2.2 = sqrt(I_max MSE n) on the window, equal to 1/sqrt(q) sqrt(n/(n-n0)) by construction | **consistent with 5-6, not independent**: SBED is 1.8x worse than the ideal in field uncertainty at equal integration time |
| 11 | Search / localize costs (search bounds, localize vs bound) | `n_half` (SBED median) 39 / 30 / 13 / 11 for W/Gamma 1.6 / 2 / 3 / 5: the detection step *falls* with window, because the prior sigma_0 = W/sqrt12 is already within a factor 2 of the dip scale. With W <= 5 Gamma there is no search plateau to speak of | search and localize claims **not testable at W <= 5 Gamma** (as before); the existing `/api/search-bounds` panel says the same |
| 12 | Posterior near-Gaussian only in refine; far-placement of localize measurements vs residual mass | `posterior_far_placed_share_stage2` and `posterior_far_mass_tau` exist on master but were not read here (outside the new metrics) | not re-tested |

**Direction check on the current fingerprint (resample below 0.5 N_p).** SBED z std 0.81-1.28 (4 use cases, 12-15 repeats each)
against 1.2-1.5 at the old threshold, coverage 0.42-0.83 / 0.85-1.0 (noisy). Suggests the 0.5 threshold improves calibration;
**not established** with this few repeats. Re-run the SBED use cases to decide.

**Agreement with the earlier "paper story check".** Agree: SBED mean-square error stays 3x above the bound (error floor) and is
overconfident (z std 1.4, coverage 0.58-0.66); search and localize are untestable at W <= 5 Gamma. Disagree: "sweeps match
theory": the Lorentzian W >> Gamma form is off by a factor ~2 on average at these windows, so I cannot confirm it (it is the
wrong tool here; the exact ratio of the Refine & speedup section is the right one and was not re-run in this read-only pass).

**Caveats that apply to every number.** (i) Windows only 1.6-5 Gamma; the article's claims about wide windows are not
reached. (ii) Series are 80 stored points per run and interpolated; the fit window [1216, 2432] is a factor 2 wide, so q and
n0 have wide bootstrap intervals (n0 median is negative for several locators, i.e. the late slope is not yet linear through
the origin). (iii) Sweep `e`/`u` are the SMC belief's (their own fit is the domain midpoint, #197 aside). (iv) 17 of 20 SBED
use cases are cached; the median is over those. (v) Three use cases of one SNR (5, 10, 20) have different W/Gamma, so SNR
trends above mix windows.
