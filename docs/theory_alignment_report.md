# Theory alignment report: where the data contradicts the article

This report checks the article's theory against the simulation and the lab data, one item at a time (2026-11-22). The
article's theory has four regimes: search and localize before the line is located, refine after it, and tracking
from a narrow prior. Each item gives the evidence, a verdict, and, when the text should change, a proposed wording.
The verdicts are (a) a bug in the code or in how it was run, (b) a gap that needs new runs, and (c) a theory or text
statement that should change. The article itself is not edited here; the wording is a proposal for the author.

Every number comes from a read-only script in `debug/`, named in its item, run on the main cache
(`artifacts/locator_results.parquet`, `artifacts/cache/`) or the lab's `.mat` runs
(`artifacts/downloads/<drive id>/`). The new dry-run studies are in-process runs that write nothing to the cache.

| # | Item | Verdict |
|---|---|---|
| 1 | Lab refine falls to the sweep rate in wide windows | (a) running manner: the lab's own locator never locks its focus at W >= 50 MHz |
| 2 | Lab claimed sigma ~2x too small | (a) running manner: a mis-calibrated fixed contrast (set 1); a jumping line (set 2) |
| 3 | SBED's refine q ~ 0.5, not 1 | (c) text: q = 1 holds for a known shape (measured 0.94-0.96); inferring C and Gamma costs the rest |
| 4 | Calibration claimed only for 1.6-5 Gamma | (c) text: the claim extends to 50 Gamma |
| 5 | The sweep gains ~10 % less than 8 pi Gamma / (27 W) | (c) text: W is the swept window; finite-window terms explain the rest |
| 6 | Tracking regime has no scenario | (b) new scenario `article-tracking`; (c) text: "from the first measurement" holds only with the shape known, after ~10 measurements |

## The lab runs are not NVision's SBED

The lab data are runs of the lab's own MATLAB `BayesianESR` (`myStruct.BayesianESR`), not of NVision's
`Bayesian-SBED`. NVision only reads the stored traces (`nvision/metrics/lab_process.py`), so `nv matlab` has no
focus or candidate code path of its own: the lab and the simulation are two different implementations. The stored
config of the lab locator (`debug/lab_config_dump.py`) is:

| setting | value | NVision SBED |
|---|---|---|
| particles / utility particles | 10000 / 500 | 10000 / 500 (`NVISION_SMC_EIG_PARTICLES`) |
| candidates | 201 global, up to 800 in focus | uniform grid of 0.1 HWHM, restricted to the focus |
| selection | softmax, temperature 1 | per-chunk argmax, then softmax at 0.01 nats |
| exploration | 10 %, decaying over 25 steps | gap probe, 2-10 % |
| shape | baseline, contrast, width fixed (`inferBaseline/Contrast/Linewidth = 0`) | contrast and width inferred |
| focus | locks on a dip-support test (`dipSigma`, `minimumDipPoints`, `focusJointTailThreshold` 1e-3) | one padded section per dip of the boxcar GLRT (`find_dips`) |

The MATLAB source is not in the data folder, so the causes below are inferred from the stored traces and this
config. The lab's line also always sits near the window centre (the windows are symmetric about the expected line;
offset 1-10 % of W), while the simulation draws it uniformly over the window.

## Item 1. Lab refine falls to the sweep rate in wide windows

**Evidence.** `debug/lab_placement_mechanism.py` reads every run's per-step focus window and lock flag. It aligns
the run at its detection (`lab_search_fit.detection_step`) and measures placement about the run's own final mean.
The table gives medians over runs of the second set (FWHM 1.82 MHz, budget 300, about 19 runs per window).

| W (MHz) | detection step | steps locked after detection | placement on locked steps (uniform-in-focus prediction) | placement on unlocked steps (uniform-in-window prediction) | all steps |
|---|---|---|---|---|---|
| 5 | 2 | 95 % | 0.45 (0.46) | 0.33 (0.34) | 0.45 |
| 10 | 3 | 91 % | 0.43 (0.43) | 0.19 (0.17) | 0.40 |
| 20 | 7 | 62 % | 0.44 (0.45) | 0.08 (0.09) | 0.31 |
| 50 | 31 | 0 % | 0.40 (0.38) | 0.033 (0.034) | 0.036 |
| 100 | 81 | 0 % | 0.23 (0.32) | 0.020 (0.017) | 0.021 |

The first set (FWHM 3.5 MHz, stop at sigma < 1 MHz) never locks at 40/100/150 MHz either; its placement is
0.10/0.042/0.039.

Two facts explain the whole table:

- **The lab locator samples nearly uniformly within its candidate set.** On both locked and unlocked steps the
  measured placement equals what uniform sampling over that set would give. With softmax at temperature 1, scores
  much smaller than 1 make the choice nearly flat. This also explains the lab's narrow-window q of about 0.45: that is
  uniform sampling over a focus of about 2 FWHM, not a refine at the steepest slope.
- **In wide windows the focus never locks within 300 steps.** The lock is a dip-support test on the measurements
  near the line. Unlocked, the locator spreads its measurements over the whole window, so only about
  (focus width / W) x 300 land near the line, about 11 at W = 100 MHz. At the end of the 100 MHz runs the candidate
  support is (10 of 12) and (3 of 6) points, and the multiplicity-adjusted tail is 1.0
  (`debug/lab_focus_state.py`). This is chicken and egg: the locator needs a lock to measure near the line, and it
  needs measurements near the line to lock. Unlocked placement is exactly the sweep's 8 pi Gamma / (27 W), because
  uniform sampling over the window is what a sweep does.

The other candidates are ruled out:

- Known C and Gamma do not cause it: locked steps reach 0.40-0.45 at every window, 50 MHz included.
- Spurious-dip widening does not cause it: the 50/100 MHz focus window is the full window, not a widened one.
- The detection definition does not cause it: the far share is 89-94 % over all post-detection steps, whatever
  step detection is taken at.
- The budget does not cause it directly: the lock simply does not fire before 300.

**Verdict: (a) running manner**, in the lab's MATLAB locator, outside this repository. Two fixes would remove it:
select by argmax, or by a softmax whose temperature is scaled to the score, as NVision does at 0.01 nats; or let
the focus follow the posterior mass (`focusPosteriorMass`) without the support test. The simulated lab mirror
(`article-lab-mirror`, below) runs NVision's SBED with the lab's known shape at the lab's windows. If it refines at
q ~ 0.5 at 100 MHz, the theory needs no change and the lab text should name the locator's focus rule as the cause.

**Proposed wording (Results, lab).** Replace the explanation of the slow wide-window tail with: "In windows wider
than about 25 linewidths the lab's locator kept its candidates spread over the whole window after detection,
because its focus did not lock within the 300 measurements, so it gained information at the sweep's rate
$8\pi\Gamma/(27W)$. Where the focus locked, it measured at an efficiency of 0.40-0.45 at every window."

## Item 2. Lab claimed sigma is ~2x too small

There are two separate causes, one per lab set.

**Set 1 (`runs_to_convergence`, 100 runs per window at 40/100/150 MHz, stop at sigma < 1 MHz).** The runs held
the contrast at C = 0.14 and the FWHM at 3.5 MHz. A free Lorentzian fit to each run's own measurements
(`debug/lab_pooled_lineshape.py`, runs with >= 40 points within 3 FWHM of their mean) gives C = 0.061/0.070/0.064
and FWHM 4.4/3.5/3.2 MHz. The contrast was about 2.15x too large. The information per measurement scales as C^2,
so the model thought each measurement worth about 4.6x what it was, and its claimed sigma is about 2.1x too small.
That is exactly the ratio seen:

- between consecutive runs (two minutes apart, so no slow drift), the median |m_i+1 - m_i| / sqrt(s_i^2 + s_i+1^2)
  is 1.41/1.48/1.48 against 0.674 calibrated (`debug/lab_segment_calibration.py`);
- the robust z-scale is 1.96/2.50/2.10 (`debug/lab_z_distribution.py`).

The second set uses the same MW amplitude (-17 dBm) and C = 0.14, and fits C = 0.12-0.14, FWHM 1.8-2.0 MHz. Set 1's
contrast was set from a calibration that did not hold on its day.

**Set 2 (`Full runs ... Nov_2_2026`, ~20 runs per window, full 300 steps, C = 0.14, FWHM 1.82 MHz).** The final
means do not scatter: they sit on a few levels 0.5-1 MHz apart and stay on one level for many consecutive runs
(`debug/lab_means_vs_time.py`). At W = 10 MHz, runs 5-19 lie within 1466.03-1466.14 MHz (spread about 0.04 MHz
against a claimed 0.027 MHz); runs 1-4 and 20 sit at about 1465.6 MHz. The line jumps between recordings taken
minutes apart. The scatter over all runs (0.2-0.7 MHz) measures those jumps, not the locator. Between consecutive
runs the median ratio is 1.16/0.94/0.77/0.79/0.63 at 5/10/20/50/100 MHz, against 0.674 calibrated: within about
1.2-1.4x of calibrated, plus a tail from the jumps.

**Noise model.** The residuals around each run's own model are Gaussian at the recorded shot noise
(resid/SE = 1.0-1.2). Their lag-1 autocorrelation is |rho| <= 0.05, so there is no structured in-run drift. The
noise variance is inferred (inverse-gamma prior), so a noise level 1.3-1.6x the shot noise would not make sigma too
small. The cause is not the noise model.

**Verdict: (a) running manner.** Set 1's fixed contrast was mis-calibrated. Set 2 has no single truth to calibrate
against. A drift-free reference needs interleaved reference sweeps (the even/odd holdout of `lab_reference.py` does
this per raw sweep file) or a jump-aware reference, the level the line sits on in that run. The dry-run
`s20_snr5_ks_c15` (below) tests the mechanism in simulation: the known shape's contrast is set 1.5x the truth.

**Proposed wording (Results, lab).** "The claimed uncertainty of the first set is about twice the scatter of
repeated runs because that set held the contrast at 0.14 while the line's contrast that day was about 0.065, so the
model credited each measurement with about 4.6 times its information. In the second set, with the contrast right,
consecutive runs agree within 1.2-1.4 times their claimed uncertainty, while the line itself moved by 0.5-1 MHz
between some recordings."

## Item 3. SBED refines at q ~ 0.5, not 1

**Evidence.** `debug/repro_known_shape_q.py` (in-process dry runs, nothing cached) runs `Bayesian-SBED` and the new
`Bayesian-SBED-KnownShape` on the same 16 truths, and `debug/analyse_known_shape_q.py` summarises them. Known shape
means the contrast and the width are centred on the truth at 1 %, as the lab holds them. q is the mean oracle
efficiency eta over a fixed step range, not "k steps after detection", so arms that detect at different times are
compared at the same run time. The offsets (in HWHM gamma) say where the measurements sit: |u| < 0.3 is the dip
centre, 0.3-1 the steep sides (the optimum is 0.58), 1-3 the wings, > 3 far.

| use case | arm | q, steps 100-300 | q, steps 300-800 | centre / sides / wings / far |
|---|---|---|---|---|
| span 20, SNR 5 | shape inferred | 0.57 | 0.56 | 0.32 / 0.63 / 0.02 / 0.03 |
| span 20, SNR 5 | shape known | 0.95 | 0.94 | 0.01 / 0.84 / 0.00 / 0.15 |
| span 50, SNR 5 | shape inferred | 0.58 | 0.58 | 0.34 / 0.64 / 0.00 / 0.02 |
| span 50, SNR 5 | shape known | 0.95 | 0.94 | 0.00 / 0.83 / 0.03 / 0.14 |
| span 50, SNR 10 | shape inferred | 0.56 | 0.56 | 0.33 / 0.64 / 0.01 / 0.03 |
| lab mirror, W = 100 MHz, FWHM 1.82 MHz, SNR 5 (budget 300) | shape known | 0.96 | - | 0.00 / 0.83 / 0.03 / 0.14 |

With the shape inferred, a third of the measurements go to the dip centre. There a shift of theta changes nothing,
but the depth tells the contrast. Those measurements alone take q from about 0.9 to 0.57. With the shape known, the
centre share drops to zero and q reaches 0.94-0.96. The far share (about 0.14) is the falsification probes (2-10 %)
plus measurements in the pads. The claimed sigma stays honest in both arms (cov2 0.94-1.00 at n = 300).

The predictive-variance score is not wrong. It is the right greedy choice for the joint posterior over
(theta, Gamma, C). It differs from the theory's assumption that only theta is unknown. The article already says
the bounds treat Gamma and C as known, and that inferring them "adds nothing when the measurements are placed
symmetrically": that is true of the information on theta at a given placement, but the placement itself changes,
because the greedy score spends measurements on C.

**Verdict: (c) text.** The theory's q = 1 is reached by SBED when the shape is known; with it inferred, SBED spends
about 40 % of its refine budget on the contrast. A code change (a score on theta alone, the Schur-complement
variance) was tried before and was slower in paired runs (memory `split-targeted-acquisition-negative-2026-11-02`),
so this report does not propose one.

**Proposed wording (after Eq. strategy-numbers, or Results, refine).** "Greedy EIG reaches $q\approx1$ only when
$\theta$ is the sole unknown: with the contrast and the width known, SBED measures at the steepest slope and gains
$q=0.94$-$0.96$. Inferring them as well, it spends about a third of its measurements at the bottom of the dip, where
they tell the depth but nothing about $\theta$, and gains $q\approx0.57$."

**Runs needed (b).** The numbers above are 16 dry-run repeats per arm. The `article-lab-mirror` member runs both
arms on the lab's signals at 50 repeats; see "Runs to make".

## Item 4. Calibration at 10-50 Gamma

**Evidence.** `debug/article_claims_coverage.py` computes `/api/claims` in process (FastAPI TestClient, read-only
against the main cache). That is the article's own coverage at the budget, gated at each run's Gaussian onset. The
table gives SBED's within-2-sigma share per SNR (1/2/5/10/20), 50 runs each:

| span | within 2 sigma | within 1 sigma |
|---|---|---|
| 1.6-5 | 0.92-0.98 | 0.56-0.78 |
| 10 | 0.94/0.98/0.92/0.98/0.94 | 0.84/0.62/0.56/0.68/0.60 |
| 20 | 0.98/0.96/0.98/0.98/0.94 | 0.78/0.74/0.70/0.74/0.66 |
| 50 | 0.96/1.00/0.96/0.98/0.94 | 0.66/0.68/0.56/0.72/0.62 |

All 50 runs pass the Gaussian-onset gate at spans 10-50. The ungated end-of-run z from the parquet agrees: pooled over
SNR, cov2 is 0.95/0.97/0.97 at spans 10/20/50, 250 runs each (`debug/article_grid_calibration.py`). The rows are
post-#348: the largest true Zeeman shift reaches 0.98 W_drive at span 50. The 1-sigma share at span 10, SNR 5 (0.56)
and span 50, SNR 5 (0.56) sits below 0.68, at the edge of its 95 % interval with 50 runs. The sweep's gated coverage
is 0.84-0.98.

**Verdict: (c) text.** The claim can extend to 1.6-50 Gamma. Nothing in the data limits it to 5 Gamma.

**Proposed wording (Results, calibration).** "The claimed uncertainty is honest at every window, 1.6 to 50
linewidths: 92-100 % of the runs lie within $2\hat\sigma_N$ at the budget, at every SNR."

## Item 5. The sweep gains ~10 % less than 8 pi Gamma / (27 W)

**Evidence.** `debug/sweep_q_finite_window.py` takes the measured placement of `SimpleSweep-optimal` (the
shot-weighted mean of `stage{1,2,3}_sens_eff`, i.e. mean eta over the run) on the article grid, 250 runs per span.
It compares that with the closed form using the swept drive window W_drive = (span + 1) Gamma, and with two exact
finite-window averages. The first is the continuous one, with the dip drawn uniformly in [pad, W - pad]. The second
uses the sweep's own grid, `linspace(lo, hi, n)` with both ends at 3 points per FWHM.

| span (W_prior / Gamma) | W_drive / Gamma | measured | 8 pi Gamma / (27 W_drive) | 8 pi Gamma / (27 W_prior) | exact, continuous | exact, grid | measured / closed (drive) | measured / exact grid |
|---|---|---|---|---|---|---|---|---|
| 1.6 | 2.6 | 0.308 | 0.358 | 0.582 | 0.343 | 0.312 | 0.86 | 0.99 |
| 2 | 3 | 0.278 | 0.310 | 0.465 | 0.300 | 0.275 | 0.90 | 1.01 |
| 3 | 4 | 0.213 | 0.233 | 0.310 | 0.227 | 0.213 | 0.92 | 1.00 |
| 5 | 6 | 0.147 | 0.155 | 0.186 | 0.153 | 0.146 | 0.95 | 1.01 |
| 10 | 11 | 0.082 | 0.085 | 0.093 | 0.084 | 0.082 | 0.97 | 1.01 |
| 20 | 21 | 0.044 | 0.044 | 0.047 | 0.044 | 0.044 | 0.98 | 1.00 |
| 50 | 51 | 0.018 | 0.018 | 0.019 | 0.018 | 0.018 | 1.01 | 1.01 |

Measured over the exact grid average is 0.99-1.01 at every span, so nothing is unexplained. The shortfall against
the closed form has three parts:

- **Which W.** The sweep scans the drive window, the prior range plus a half-FWHM pad on each side. Against the
  prior range, the gap is span / (span + 1): 9 % at span 10, 17 % at span 5. All the metrics code
  (`refine_rates_runs`, `stages.py`) already uses the drive window.
- **Edge truncation.** A dip near the window edge loses part of its slope to outside the window: 1-4 % at spans
  <= 5, under 1 % at spans >= 10.
- **The grid's endpoints.** n points span W with n - 1 gaps, so the grid mean is about (n - 1)/n of the continuous
  one: 3-10 % at spans <= 5, under 3 % at spans >= 10.

The last two vanish as W >> Gamma, which is the limit the closed form states.

**Verdict: (c) text.** No code or run is wrong.

**Proposed wording (Speedup over a drive-frequency sweep).** After Eq. (sweep-info): "Here $W$ is the swept window,
which in the simulations is the prior range of the line plus half a linewidth at each end; for windows of a few
linewidths the window's edges and the sweep grid's end points lower $I_{\rm sweep}$ by a further 5-14 % (under 3 %
from $W = 10\,\Gamma$)." In Results, quote the use cases by the swept window (2.6-51 Gamma) or state that the span
is the prior range.

A side observation: the sweep's prior FWHM (`stage_prior_fwhm_phys`) is 3.85x the true FWHM on the article grid,
so its grid is sized off a much wider line than the true one. `optimal_sweep_spacing` compensates by choosing a
large k*, and the result is about 3 points per true FWHM. Worth knowing when reading `sweep_k`.

## Item 6. Tracking regime

**Decision.** A scenario is needed. The theory makes a quantitative claim ("a tracking run with
$\sigma_0^{\rm track}\lesssim\Gamma$ ... is all refine", following van Trees from the start), and nothing tested it.
The new member `article-tracking` (`sim.presets.tracking_generators`) is the article grid's span-20 Lorentzian with
an honest Gaussian prior on the line: std sigma_0 = 0.25, 0.5 or 1 FWHM, and mean drawn from N(truth, sigma_0)
(`NVCenterCoreGenerator.tracking_prior_std_fwhm`). An honest prior is required for J0 = 1/sigma_0^2: the
generator's usual nuisance priors are deliberately offset by 3 sigma, which would invalidate the bound. It runs
`Bayesian-SBED`, `Bayesian-SBED-KnownShape` and the sweep at k* over the SNR grid. The prior is drawn last in
`generate()`, so no other scenario's draws change.

**Evidence.** Dry runs at sigma_0 = 0.5 FWHM, SNR 5, 16 runs per arm. R(n) = sigma_vT(n)^2 / mean err(n)^2 with
sigma_vT^-2 = J0 + n I_max and the run's own I_max; R <= 1 is required, and R near 1 means the bound is followed.
n_c = J0 / I_max = 0.09, so the prior is worth a tenth of one measurement. With 16 runs, R carries about 35 %
relative error.

| arm | R(1) | R(3) | R(10) | R(30) | R(100) | R(300) |
|---|---|---|---|---|---|---|
| shape known, belief prior as shipped (10 % uniform mixture) | 0.05 | 0.08 | 0.39 | 0.67 | 1.25 | 1.15 |
| shape known, belief prior = the stated Gaussian (`BROAD_FRACTION=0`) | 0.22 | 0.20 | 0.87 | 0.68 | 0.47 | 0.51 |
| shape inferred, prior as shipped | TRACK_SBED |
| shape inferred, prior = the stated Gaussian | 0.18 | 0.10 | 0.09 | 0.19 | 0.05 | 0.30 |

Two things stop SBED from following the bound "from the first measurement":

- **The belief does not use the stated prior.** `PRIOR_BROAD_FRACTION = 0.1` (`belief/smc_marginal.py`) draws 10 %
  of the prior's particles uniformly over the whole window. This is a robustness choice against wrongly centred
  priors (memory `locator-overconfident-calibration`). After one measurement the shipped belief claims 3-7 MHz
  against sigma_0 = 1 MHz, and its estimate is pulled toward the window. With the mixture off, R(1) rises from 0.05
  to 0.22 and R(10) from 0.39 to 0.87.
- **The bound is not tight while the prior is as wide as the line.** At sigma_0 = Gamma/2 the first measurements
  cannot all sit at the steepest slope of a line whose position is uncertain by a half width, so I(d) < I_max even
  for an ideal design, and R(1-3) of about 0.2 is the nonlinear regime, not a flaw. It closes by n of about 10,
  once the posterior is well inside the line.
- **With the shape inferred**, tracking stays at R of 0.05-0.3 throughout: the bound treats C and Gamma as known,
  and SBED must learn them from a run that starts on the line. That is item 3's cost, without a search to hide it.

**Verdict: (b) runs, and (c) text.** Run the member (see below) for 50-run numbers at three sigma_0 and every SNR.
The text should qualify the claim. Whether to turn the broad mixture off for tracking priors is a design choice for
the author: it would make SBED follow the bound sooner, at the cost of robustness to a wrong prior. It is not
changed here.

**Proposed wording (Theory, regimes).** Replace "A tracking run with $\sigma_0^{\rm track}\lesssim\Gamma$ has the
dip located from the start, so it has no search and is all refine" with "A tracking run with
$\sigma_0^{\rm track}\lesssim\Gamma$ has the dip located from the start, so it has no search: once its posterior is
well inside the line, after about ten measurements, it follows the van Trees bound, provided the width and the
contrast are known."

## Detection: the dip test against the lab-style claimed-sigma drop

The orchestrator asked how far apart the two detectors are on simulated runs. Both will become pluggable. The dip
test is `dip_detected_step`, the locator's held boxcar GLRT (`stages.py`). The lab-style detector is
`lab_search_fit.detection_step`, the first steep, lasting drop of the claimed sigma. `debug/detector_gap.py` applies
both per run to the cached article grid (50 runs per use case). The claimed sigma comes from the stored series,
which may be thinned, so the lab step is the stored step at the drop.

DETECTOR_TABLE

## Runs to make

None of this needs a re-run of existing results. The new strategy and generators add new cache keys only; existing
combinations keep their keys (the key hashes the names and the physics fingerprint, not the generator's fields).
`Bayesian-SBED` is unchanged. **No physics fingerprint bump is needed and nothing cached is stale.**

```powershell
# The article group: the existing 70 search combinations are served from cache; this runs only the new members
# (lab mirror 2 x 6 x 3 = 36 tasks at SNR 5; tracking 3 x 5 x 3 = 45 tasks), 50 repeats, budget 2500
uv run nv groups run article --repeats 50
```

```powershell
# The lab mirror at the lab's own budget of 300, in a separate folder (the sweep's step k* depends on the budget)
$env:NVISION_MAX_MEASUREMENTS = "300"; uv run nv run --run-group article-lab-mirror --repeats 50 --out artifacts_lab300
```

These will show the following:

- the lab mirror: q of SBED with the shape known against inferred at the lab's windows. The dry run gives 0.96 at
  100 MHz, so the lab's wide-window tail is its locator, not the physics. Also the claimed-sigma coverage with the
  lab's known shape.
- tracking: R(n) at three sigma_0 over the SNR grid, and the sweep's constant rate as a reference.

## Code changes in this PR

- `Bayesian-SBED-KnownShape` (opt-in strategy): `belief_builders.apply_known_prior` / `apply_known_shape` (applied
  after the Voigt-to-Lorentzian renaming), `_TaskRunner._known_shape_prior_phys` (truth-centred, std
  `KNOWN_SHAPE_STD_FRACTION = 0.01`), `StrategyFilter.BAYESIAN_SBED_KNOWN_SHAPE`. Simulation-only (an oracle mean).
  `KNOWN_SHAPE_STD_FRACTION` is a module constant. Whether to expose it in `.env` is open for the user.
- `NVCenterCoreGenerator.tracking_prior_std_fwhm` (honest Gaussian prior on `zeeman_shift`).
- `sim.presets.lab_mirror_generators`, `tracking_generators`, registered in `CombinationGrid`.
- `run_groups`: `article-lab-mirror` and `article-tracking`, members of `article` and runnable alone.
- No lab file (`nvision/metrics/lab_*.py`, `static/lab-view.js`) is changed; the lab analysis is in `debug/` only.
- Tests: `tests/test_known_shape_prior.py`, the article-group tests in `tests/test_run_groups_param_grid.py`.
