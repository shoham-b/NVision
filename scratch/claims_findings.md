# Where the cached data contradict (or only weakly test) the article (2026-11-08)

Data: the cached pure-Lorentzian SNR x window study (`NVCenter-voigt-span{1.6,2,3,5}-gs0-hfunresolved`, SNR 1/2/5/10/20,
six locators, 50 repeats each; 120 combos, physics fingerprint `368f4f5d4fdc`, recorded 2026-11-04/05). Read-only
(`scratch/claims_analysis.py` -> `scratch/claims_results.json`; summaries `claims_summary*.py`). Cache and artifacts untouched.

**SNR caveat (applies to every finding).** These rows use the old noise level (label SNR = contrast_total/sigma, so the
article's SNR = C/sigma is half the label). `I*`, `J0`, `R_inf`, R, q, n0, eta-bar and coverage come from the true signal and
the stored errors, so they do not depend on the label; only SNR axes and absolute eta do. Re-run after the SNR branch
(`claude/align-snr-article-e05091`, fingerprint `snr-per-line-contrast-v1`) lands to put the points at the article's SNRs.

**Method caveats.**
* Series are downsampled to 80 points (dense head, log-spaced tail) and interpolated log-log. Stopped runs leave the average
  (never held); a fit needs >= 90% of the repeats running, so the (n0, q) window is short and only 9-12 of 20 use cases per locator
  have one (6 per locator have no detection step at sigma_0/2: SNR 1-2 runs to the 2500 budget; 2-5 are too short).
* The fit window is **not demonstrably the late regime.** Late log-log slopes of the run average are -0.2 to -1.1 (the -1/2 tail is
  not reached), the fitted n0 is -30 to -200, rms/median of the error at the stop is 1.7-1.9 (4.9 at worst: heavy tails),
  and the linearity gate (the two halves of the window must give overlapping q intervals) **passes in every fitted use case, but
  only because the intervals are wide** (3-8 points per half). So fitted q is "the information growth of the run average over the
  window the runs live in", which includes the worst runs still being found, not a refine rate.
* Sweep `u`/`e` are the SMC belief's, not the sweep's own fit (whose reported value is the domain midpoint, memory note), and fall like
  the others, so sweep q is not an artifact of that bug.
* span1.6: the theta prior is [1.00, 1.28] MHz (sigma_0 = 80 kHz) and the stop target tau = 100 kHz is *above* sigma_0, so the
  stored first-passage step is 1 in 49/50 runs (checked on the series: first-step error 71 kHz median). That is genuine (the prior mean is
  already inside tau), not a bug: the cost table has no information for span1.6 and no search exists there.

| # | Article claim | What the data say | Verdict |
|---|---|---|---|
| i | Gap before any measurement = pi/sqrt3 = 1.81 (uniform truth) | R(1) = 0.16-0.38, median 0.27; implied sigma gap 1.6-2.0 (article: R = 0.30). Truth spread vs uniform-over-the-prior: 0.98 / 0.88 / 1.03 / 1.09 for span 1.6 / 2 / 3 / 5 (span2: truths kept off the prior edges) | consistent. **docs/evaluation_metrics.md said "up to 1.8x the variance, R ~ 0.55" - wrong (1.8x is in sigma, R = 0.30); fixed** |
| i | Averaged true error is on or above the bound (R <= 1 + 1%) | R <= 1 at every step with >= 90% of repeats running (0 violations in 120 combos) | consistent: no defect in J0 / I* |
| ii | Plateau, one detection step, -1/2 tail in the run average | plateau ratio 0.73-1.0 of sigma_0 before n_half: consistent. No cliff in a run average (adjacent-grid drop 1.0-1.2; detection steps differ per run and smear; the elimination model is a picture of one run, as the article says). The -1/2 tail is not reached before the runs stop (slopes -0.2 to -1.1) | plateau consistent; cliff and tail **not testable on the run average of these runs** |
| iii | q of the information the measurements *carry* (placement): sweeps 1/R_inf, SBED near 1 | direct measure (mean g^2/max g^2 over all measurements, stored `stage_k_sens_eff`, no fit, no re-run): **sweeps 0.32-0.33, i.e. q x R_inf = 0.83-0.86 (claim 1)**; Smart-Sweep 0.76 (x R_inf = 1.95); **SBED 0.45**, Smart-SBED 0.45, SBED refine-only 0.47 | sweep claim **confirmed** (R_inf approximate; W >> Gamma form is within 15% here); **"SBED near 1" not supported: SBED places measurements at 45% of the optimal information, Smart-Sweep (76%) does better** |
| iii | q of the *error* decay (fitted late-window 1/MSE slope) | median 0.11 SBED / Smart-SBED, 0.09 Smart-Sweep, 0.05-0.08 sweeps (upper bootstrap bound < 1 in 100% of fits, mostly < 0.5) | far below the direct measure (SBED 0.11 vs 0.45): the information the measurements carry is **not converted to error reduction** at the rate the Fisher bound says (heavy tails, nuisance parameters, the belief's errors). Two findings, not one: placement (0.45) and inference/tails (x4). Because the window is not the late regime (caveat above) the 0.11 is itself provisional |
| iv | R_inf = 27 W/(8 pi Gamma) as the ratio of rates | from the direct rates: SBED/sweep3 = 0.45/0.33 = 1.4 vs predicted 3.2-5.4 (W/Gamma 3-5); the new `stage_fisher_sweep` column will give the exact ratio per run once re-recorded (dry-run check: measured 36.2 vs formula 36.2 at W/Gamma 33.7) | the *formula* holds (sweep direct q x R_inf 0.85, dry run exact); the *speedup in practice* is **1.4x, not 3-5x**, because SBED carries 0.45 of I*, not 1 |
| iv | Finite-target cost n0 + sigma_tgt^-2/(q I*) | predicted/measured first-passage step: median 2.9 (SBED, 2.1-6.7), 4.5 (sweep k=3), 2.7 (Smart-Sweep): over-predicts by 3-4x because the fitted window is not the late regime and ends before the target | **not testable from a fit on this window** (the number stays in the report, flagged by the fit's own caveats) |
| v | Information rate <= 1 (true); claimed > 1 means overconfident | claimed-sigma q is 1.2-2.6x the true-error q (median over fitted use cases): SBED 2.0, Smart-SBED 2.4, Smart-Sweep 1.2, sweeps 2.4-2.6 | **overconfident, all six** (a ratio of two fits on the same window, so the window caveat cancels to first order) |
| vi | Coverage 68% / 95% | at the stop: SBED 48% / 81%, Smart-SBED 54% / 82%, Smart-Sweep 57% / 84%, sweeps 64-66% / 89-91% (medians over use cases). At a fixed n (gated on `refine_start_step`, 98-100% of runs, so the gate is vacuous) SBED is about 60% / 87% at n = 100: the stop selects the runs whose claimed sigma just got small, which is part of the shortfall | **all overconfident; SBED the most** at the stop and at fixed n. Particle-quantile coverage (PIT) needs a re-run |
| ix | eta measured / eta article = 1/sqrt(q) | eta ratio 2.2-5.5 from the late window, = 1/sqrt(q_fit) by construction (a check of the fit) | with the direct q the article's eta is **1/sqrt(0.45) = 1.5x** too optimistic for SBED from placement alone, 3x or more from the error decay |
| vii | share of localize measurements far from the peak vs residual mass | **not testable from the cache** (needs run-time columns); one dry run each: SBED 4% far vs 1% mass (n = 26 localize steps), a sweep 80% far vs 0.2% mass (by design) | re-run needed |

Also worth the author's attention:
* The runs stop at the stop target (0.05 FWHM = 100 kHz) when the mean-square error is still 3-15x above the bound (R(30) = 0.04-0.3).
  The claimed sigma is what stopped them (overconfidence), so "reaches the data limit" is not demonstrated by these runs, only "reaches
  the stop rule".
* `n_c = J0/I*` ranges from 0.1 (SNR 20, window 5) to 100 (SNR 5, window 1.6): for the high-SNR use cases the bound is in its data limit
  from n = 1, so the prior-limited regime does not exist there; Fig. budget's wide-window picture applies only to the low-SNR / large-window
  corner, and that corner has no late fit.
* The theta prior (zeeman_shift, 0.28-3 MHz) is not the drive window (3.2-10 MHz here). `R_inf` in the article uses the drive window (what a
  sweep covers); J0 uses the theta prior. Both are stored (`stage_window_phys`, `stage_prior_info`).
