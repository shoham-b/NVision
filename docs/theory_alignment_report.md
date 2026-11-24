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
| 3 | SBED's refine q ~ 0.5, not 1 | see item 3 |
| 4 | Calibration claimed only for 1.6-5 Gamma | (c) text: the claim extends to 50 Gamma |
| 5 | The sweep gains ~10 % less than 8 pi Gamma / (27 W) | (c) text: W is the swept window; finite-window terms explain the rest |
| 6 | Tracking regime has no scenario | (b) new scenario `article-tracking`; see item 6 |

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
