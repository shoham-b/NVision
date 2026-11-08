# Constants audit: articles/sections/{sm_algorithm,methods}.tex vs code

Read-only audit. Result: every checked number and rule already matches; no code was changed.

| Item | Article | Code | Where | Status |
|---|---|---|---|---|
| Particles N_p | 1000 | 1000 | nvision/belief/smc_marginal.py:29 | ok |
| Resample trigger | ESS < 0.2 N_p, systematic | 0.2; `_systematic_resample_indices` | smc_marginal.py:30, :1076 | ok |
| Liu-West a | 0.98, cov (1-a^2) Sigma, weighted mean/cov before resample | 0.98, `nudge_cov=(1-a^2)*cov`, pre-resample mean | smc_marginal.py:31, :1080-1122 | ok |
| Variance floor | (0.01 e^{-n/25})^2 of unit range | `min_exploration_frac=0.01`, decay `exp(-step/25)` on cov diagonal in unit space | smc_marginal.py:32, :70, :1093-1098 | ok |
| Reflection | fold back into unit cube | `1-abs(mod(x,2)-1)` | smc_marginal.py:1131-1134 | ok |
| Noise prior | alpha0=10, beta0=alpha0*sigma_nom^2, sigma_nom geometric mean of range | `noise_prior_strength=10`, `sqrt(lo*hi)` | smc_marginal.py:419, :564-569 | ok |
| Discount | d=0.99 on alpha and beta before update; +1/2 alpha, +1/2 k r^2 beta | 0.99; same recursion (plus theta-independent within-batch term for k>1) | smc_marginal.py:418, :689-697 | ok |
| Predictive | Student-t_{2alpha}, scale^2 = beta/(k alpha), pre-update alpha,beta | gammaln form with beta/(k alpha) | smc_marginal.py:672-679 | ok |
| beta after rejuvenation | recomputed from full history | `_recompute_noise_betas` | smc_marginal.py:1138, :1144 | ok |
| Stop ceilings | 100 kHz theta, 500 kHz widths, 0.01 contrast | 1e5 (zeeman_shift), 5e5 (hwhm_lorentz, sigma_inhom), 0.01 (contrast_total) | nvision/sim/defaults.py:275-281, :310 | ok |
| Hold | 8 consecutive measurements + dip found | `NVISION_CONVERGENCE_PATIENCE=8` | nvision/sim/locs/bayesian/sequential_bayesian_locator.py:29, :58 | ok |
| Dip re-run | at most every 10 | `NVISION_DIP_SCAN_EVERY=10` | sim/defaults.py:67 | ok |
| Budget | 2500, censored | `NVISION_MAX_MEASUREMENTS=2500`; executor returns it for sweeps and the SBED family (overrides the stale 200 in combinations.py) | sim/defaults.py:77; runner/executor.py:1191-1205 | ok |
| Dip baseline | 70th percentile minus t^-1(0.7) s | `_BASELINE_PERCENTILE=70`, `student_t.ppf(0.7,nu)*s` | belief/dip_detection.py:55, :249-250 | ok |
| Dip score/threshold | boxcar sum z/sqrt(n); t_nu^-1(1-p_FA/N_tests), p_FA=0.1 | same; `NVISION_DIP_P_FA=0.1` | dip_detection.py:158, :264; sim/defaults.py:63 | ok |
| SimpleSweep spacing | Gamma0/p, p=2,3,4 | `SWEEP_SPACING_DIVISORS=(2,3,4)` | sim/combinations.py:47 | ok |
| Smart-Sweep search | Gamma0/3 until dip | `COARSE_SEARCH_POINTS_PER_FWHM=3` | sim/locs/search_grid.py:39 | ok |
| Smart-Sweep focus | Gamma0/10 | `FOCUS_SWEEP_POINTS_PER_FWHM=10` | search_grid.py:48 | ok |
| Smart-Sweep refine | steepest-slope once >=90% of optimal information | `SMART_SWEEP_REFINE_INFO_LOSS=0.1` (eta >= 0.9) | sim/locs/bayesian/sbed_locator.py:360, :481 | ok |
| SBED candidates | 800 | `NVISION_SMC_EPOCH_CANDIDATE_BUDGET=800` | smc_marginal.py:64 | ok |
| Candidate thinning | >=100 kHz | `NVISION_SMC_CANDIDATE_STEP_HZ` defaults to the 100 kHz theta ceiling | sim/defaults.py:132 | ok (the 10 kHz `NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ` is the pre-thinning density grid, not the thinning) |
| EIG subsample | 500 particles | `NVISION_SMC_EIG_PARTICLES=500` | smc_marginal.py:44 | ok |
| Blocks | 64 | `_EIG_CHUNK_SIZE=64` | smc_marginal.py:66 | ok |
| Softmax temperature | 0.01 nats | `_EIG_SELECTION_TEMPERATURE_NATS=0.01` | smc_marginal.py:84 | ok |
| Focus padding | 1 posterior-mean HWHM | `NVISION_SMC_FOCUSING_COVER_FACTOR=1.0` | belief/focus_window.py:38; sbed_locator.py:220-233 | ok |
| Narrowing | after 8 measurements, >=5% shrink | `NVISION_MIN_STEPS_BEFORE_NARROWING=8`, `_MIN_NARROWING_FRACTION=0.05` | focus_window.py:34, :40, :280-286 | ok |
| Falsification probe | 0.02 + 0.08 g | floor 0.02, max 0.10, linear in largest gap | sbed_locator.py:52-84 | ok |

## Notes for the article / follow-ups (no code change made)

- `NVISION_SBED_MAX_STEPS=200` (sim/defaults.py:83, used in sim/combinations.py:295,317) is a stale default; the
  executor overrides it with 2500 for the SBED family. Harmless, but a misleading constant. Left alone.
- `NVISION_NOISE_SIGMA_CONVERGENCE_THRESHOLD=0.001` exists in the ceilings dict; the article's stop rule lists only
  theta, widths and contrast. Not verified whether noise_sigma is a target parameter in the stop check (it is not a
  particle dimension); worth one sentence in the article only if it is.
- Focus "re-decided after every resampling" and "candidates placed after every resampling" were not traced line by
  line; the constants above were.

## Staleness

No code changed, so no results are stale and the physics fingerprint is not bumped.
