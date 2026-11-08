# Where the cached data contradict (or only weakly test) the article (2026-11-08)

Data: the cached pure-Lorentzian SNR x window study (`NVCenter-voigt-span{1.6,2,3,5}-gs0-hfunresolved`, SNR 1/2/5/10/20,
six locators, 50 repeats each; 120 combos, physics fingerprint `368f4f5d4fdc`, recorded 2026-11-04/05). Read-only
(`scratch/claims_analysis.py` -> `scratch/claims_results.json`; summaries `claims_summary*.py`). Cache and artifacts untouched.

**SNR caveat (applies to every finding).** These rows use the old noise level (label SNR = contrast_total/sigma, so the
article's SNR = C/sigma is half the label). `I*`, `J0`, `R_inf`, R, q, n0 and coverage are computed from the true signal
and the stored errors, so they do not depend on the label; only SNR axes and absolute eta do. Re-run after the SNR branch
(`claude/align-snr-article-e05091`, fingerprint `snr-per-line-contrast-v1`) lands to put the points at the article's SNRs.
**Method caveats.** Series are downsampled to 80 points (dense head, log-spaced tail) and interpolated log-log. Stopped runs
leave the average (never held) and a fit needs >= 90% of the repeats running, so the (n0, q) window is short: only 9-12 of 20
use cases per locator have one (the rest stop before a late regime, or at SNR 1-2 run to the 2500 budget with no detection step
at sigma_0/2). q intervals are wide (bootstrap over 50 repeats). Sweep `u`/`e` are the SMC belief's, not the sweep's own fit
(its reported fit is the domain midpoint, memory note), and fall like the others, so sweep q is not an artifact of that bug.

| # | Article claim | What the data say | Verdict |
|---|---|---|---|
| i | Gap before any measurement = pi/sqrt3 = 1.81 (uniform truth) | R(1) = 0.16-0.38, median 0.27; implied sigma gap 1.6-2.0 (article 0.30 -> 1.81). The truths are within 0.88-1.09 of uniform-over-the-prior spread (span2 is 0.88: truths are kept off the prior edges) | consistent. **docs/evaluation_metrics.md said "up to 1.8x the bound's variance (R ~ 0.55)" - that is wrong (1.8x is in sigma, R = 0.30); fixed** |
| i | Averaged true error is on or above the bound (R <= 1 + 1%) | R <= 1 at every step with >= 90% of repeats running (0 violations in 120 combos) | consistent (not a defect in J0 / I*) |
| i | Claimed sigma_hat run-averaged vs bound | claimed R tracks true R at small n (0.2-0.3) and is above it later: the claimed sigma is too small | see (v) |
| iii | SBED reaches q near 1 | q (late-run fit) median 0.11 (range 0.02-0.32); the 95% interval upper end is < 1 in 10 of 10 fitted use cases. Smart-SBED 0.11, Smart-Sweep 0.09 | **contradicted in this data**: no locator gets near I* in the window before it stops |
| iii/iv | Sweep q = 1/R_inf, n0 ~ 0 | q x R_inf (1 = claim) median 0.15 / 0.24 / 0.30 for sweep k = 2 / 3 / 4: sweeps carry 3-7x *less* information than the window average predicts; n0 is large and negative (-30 to -90), i.e. growth is slower than linear from zero | **contradicted** (a sweep that repeats whole passes does not gain a constant rate over the window) |
| iv | R_inf = 27 W/(8 pi Gamma) as the measured ratio of rates | SBED q / sweep q = 1.6-4 in 5 of 6 comparable use cases vs predicted 3.2-5.4, and 0.17 in one; intervals overlap everything | inconclusive (q intervals too wide); `stage_fisher_sweep` (new) gives the rate ratio without the fit once runs are re-recorded |
| iv | Finite-target cost n0 + sigma_tgt^-2/(q I*) | predicted/measured passage step median 2.9 (SBED, range 2.1-6.7), 4.5 (sweep k=3), 2.7 (Smart-Sweep): the extrapolated fit over-predicts 3-4x. The fit window ends before the target is reached, so this is also an extrapolation test the fit fails | **contradicted** as a predictor from a late-window fit |
| v | Information rate <= 1 for the true error; > 1 for the claimed sigma means overconfidence | claimed-sigma q is 1.5-2.6x the true q for every locator (median) | **overconfident**, all six |
| vi | Coverage 68% / 95% | at the stop: SBED 48% / 81%, Smart-SBED 54% / 82%, Smart-Sweep 57% / 84%, sweeps 64-66% / 89-91% (medians over use cases). Gating on `refine_start_step` (the proxy for "Gaussian", 98-100% of runs reach it) does not fix it | **SBED is the most overconfident**; the Gaussian-posterior assumption is not what is missing. Particle-quantile coverage (PIT) needs a re-run |
| ii | Plateau, one detection step, -1/2 tail in the run average | plateau ratio 0.73-1.0 of sigma_0 before n_half (consistent); the run average shows **no cliff** (adjacent-grid drop 1.0-1.2: detection steps differ per run and smear); late log-log slope -0.2 to -1.1 (median ~ -0.4), steeper than -1/2 only where n0 > 0 | plateau consistent; "cliff" is a single-run feature not visible in a run average (the elimination model is a picture of one run, as the article says); the -1/2 tail is not reached before runs stop |
| ix | eta measured / eta article = 1/sqrt(q) | eta ratio 2.2-5.5, consistent with 1/sqrt(q) (the check, by construction) | follows from (iii): SBED's refine sensitivity is ~3x worse than the article's formula in these runs |
| vii | share of localize measurements far from the peak vs residual mass | **not testable from the cache** (needs run-time columns) | re-run needed |

Also worth the author's attention:

* The raw runs stop at the stop target (0.05 FWHM = 100 kHz) when the mean-square error is still 3-15x above the bound
  (R(30) = 0.04-0.3). The claimed sigma is what stopped them (overconfidence above), so "reaches the data limit" is
  not demonstrated by these runs, only "reaches the stop rule".
* `n_c = J0/I*` ranges from 0.1 (SNR 20, window 5) to 100 (SNR 5, window 1.6): for the high-SNR use cases the bound is in its
  data limit from n = 1, so "search" and the prior-limited regime do not exist there; Fig. budget's wide-window picture
  applies only to the low-SNR / large-window corner (and that corner has no late fit).
* The theta prior (zeeman_shift, 3 MHz) is not the drive window (W = 3.2-10 MHz here). `R_inf` in the article uses the drive
  window (what a sweep covers); J0 uses the theta prior. Both are stored (`stage_window_phys`, `stage_prior_info`).
