# Core Architecture: Bayesian Inference and Simulation

## Overview

The core architecture drives simulated experiments and Bayesian inference using a highly modular combination of Generative models, Noise models, and Locator strategies. The system relies heavily on Sequential Monte Carlo (SMC) to represent belief and a strict Unit-Cube scaling system to handle bounded optimization safely.

(Note: Legacy concepts such as `ParameterWithPosterior` and 1D `BeliefSignal` grids have been fully superseded by the N-dimensional SMC architecture).

## Key Components

### 1. Simulation Orchestration (`nvision/runner/`)

Simulations are constructed as Cartesian products (`CombinationGrid`) of three primary components:
- **Generators (`nvision/sim/gen/`)**: Defines the physical experiment parameters and the true ground signal. `nv_center_generator.py`'s `NVCenterCoreGenerator` covers three NV-center lineshapes — Lorentzian, Voigt, and Saturation-Voigt (`nvision/spectra/nv_center.py`) — each optionally with Zeeman splitting and hyperfine structure. All three share the same population-normalized `contrast_total` contrast convention (see [`contrast_reparametrization.md`](contrast_reparametrization.md)). `zero_field_freq` (the zero-field center) defaults to fixed at the domain midpoint for every draw (`with_fixed_zero_field_freq=True`) — like a known, calibrated instrument constant — so only hwhm/split/hyperfine/contrast vary between repeats unless a locator explicitly opts into inferring it.
- **Noise Models (`nvision/spectra/noise_model.py`)**: Defines the noise layered over the generator's true signal — `GaussianNoiseSignalModel`, `DriftNoiseSignalModel` (slow time-varying drift), and `CompositeNoiseSignalModel` (combines multiple noise sources).
- **Locators (`nvision/sim/locs/`)**: The strategy that iteratively decides where to sample next and decides when the simulation is confident enough to stop.

The orchestration pipeline resolves these combinations into atomic `LocatorTask` units, executing them concurrently while heavily leveraging the caching database (`artifacts/cache/`).

### 2. Bayesian Belief & SMC (`nvision/belief/`)

Every strategy runs the same loop: choose a drive frequency $d_n$, take one measurement $y_n$ (the mean of $k$ shots),
update the posterior, and stop when the stop rule fires. The strategies differ only in the first step; the posterior
and its update are shared (article `methods.tex` §Bayesian inference, numbers in `sm_algorithm.tex`).

- **Posterior.** A weighted particle cloud (`smc_marginal.py`, $N_p = 1000$) over the parameter of interest
  $\theta = D - \delta_Z$ (particle dimension `zeeman_shift`) and the nuisances: the linewidth (`hwhm`, or
  `hwhm_lorentz` + `sigma_inhom` for pseudo-Voigt) and the contrast. The mechanisms that set the width are not inferred.
- **Noise.** $\sigma$ is not a particle dimension. Each particle carries an inverse-gamma posterior on $\sigma^2$
  (prior shape 10, scale from the geometric mean of the $\sigma$ prior range, discount 0.99 per measurement), so $\sigma^2$
  is integrated out exactly (Rao-Blackwellized filter) and a measurement reweights each particle by a Student-$t$
  predictive. `noise_model` is mandatory.
- **Resampling.** Systematic, when the effective sample size falls below $0.2\,N_p$, followed by Liu-West
  rejuvenation ($a = 0.98$) with a variance floor $(0.01\,e^{-n/25})^2$ of each unit range and reflection of particles that
  leave the unit cube. Each particle's noise posterior is recomputed from the full history afterwards.
- **Reported uncertainty.** `uncertainty()` is the weighted particle standard deviation $\hat\sigma_N$, the
  **claimed uncertainty**. It is a moment of the posterior, defined at every $N$, and is only a 68 % interval once the
  posterior is Gaussian; coverage claims must be tested against the particle distribution or only in the data regime.
  `entropy()` assumes a Gaussian and overstates the entropy before that.

### 3. Unit-Cube Scaling Architecture

A critical design feature of the inference engine is the strict separation between internal algorithmic state and external physical representation.

- **Unit Normalized Parameters (`[0, 1]`)**: The core SMC engine and likelihood algorithms operate strictly on the unit-cube `[0, 1]`. This ensures uniform convergence thresholds, prevents scale imbalances during multidimensional acquisition optimizations, and makes the core algorithms completely agnostic to the underlying physical dimensions.
- **Physically Scaled Parameters**: The physical bounds and scaling logic are abstracted away from the core particle math.
- **`SMCMarginalDistribution`**: Particles and likelihoods live on the unit cube, while the public summaries (`.estimates()`, `.uncertainty()`, covariance matrices, `.get_candidate_drive_freq_phys()`) transparently denormalize back into physical scales for the CLI monitors and UI plots. The drive-frequency axis is always the full domain; which part of it may be scanned (the focus) is owned by the locator, not the belief.

### 4. Locators: sweep family and Bayesian SBED

All strategies infer with the same particle filter, so the comparison isolates the choice of measurements from
the inference, and $\hat\sigma_N$ means the same for all of them. All run under the **same stop rule and the same budget**
(`methods.tex` §Measurement strategies; `sm_algorithm.tex`):

- **Stop rule.** The claimed uncertainty of every parameter stays below its ceiling for 8 consecutive measurements
  (`NVISION_CONVERGENCE_PATIENCE`) *and* the dip test finds a dip. Ceilings: 100 kHz for $\theta$, 500 kHz for each width
  (`NVISION_WIDTH_CONVERGENCE_THRESHOLD`), 0.01 for the contrast. The dip test is re-run at most every 10
  measurements (`NVISION_DIP_SCAN_EVERY`).
- **Budget.** 2500 measurements for every strategy (`NVISION_MAX_MEASUREMENTS`); a sweep's total is capped in whole
  passes. A run that reaches the budget is **censored** and counted at the budget, a lower bound on its cost.
- **Dip test** (`find_dips`, `belief/dip_detection.py`). Each measurement is scored $z_n = (b - y_n)/s$ against a baseline
  $b$ (70th percentile of the $y_n$, minus the expected offset under pure noise) in units of the noise posterior's
  scale. Every contiguous stretch of measured drive frequencies whose span lies in the prior range of the linewidth gets
  the boxcar score $\sum z_n/\sqrt n$; it is significant when it passes the Bonferroni threshold at family-wise
  false-alarm probability $p_{FA} = 0.1$ (`NVISION_DIP_P_FA`). Each connected run of significant stretches is one dip.
- **Sweep family.** `SimpleSweep-<k>perFWHM` measures a uniform grid of spacing $\Gamma_0/p$, $p = 2, 3, 4$, and repeats
  whole passes; `Smart-Sweep` sweeps at $\Gamma_0/3$ until the dip test finds a dip, then at $\Gamma_0/10$ inside the
  focus, then measures only at the maxima of $(\partial\mu/\partial\theta)^2$ at the posterior mean once the posterior expects each point to
  carry at least 90 % of the optimal information. Details: [locator comparison](locator_comparison.md).
- **Bayesian SBED** (`nvision/sim/locs/bayesian/sbed_locator.py`). After every resampling, 800 candidate drive
  frequencies are placed at quantiles of a density with Gaussian kernels on the two sides of the posterior-mean dip
  ($\pm$ HWHM from its centre), kernels on the dips found and a flat share, thinned to at least 100 kHz spacing. The
  expected information gain (variance form, article Eq. `eig-var`) is evaluated on a 500-particle subsample with $\sigma^2$ the
  posterior mean of the noise; candidates are taken in blocks of 64 and the block maxima compete through a softmax of
  temperature 0.01 nats, so the choice is near-greedy.
- **Focus.** The locator owns a `FocusRegion` (`nvision/belief/focus_window.py`): one section per detected dip, the dip's
  extent padded by one posterior-mean HWHM on each side (`NVISION_SMC_FOCUSING_COVER_FACTOR`), overlapping sections
  merged. It restricts which candidates may be picked. A new dip **widens** the focus at once; **narrowing** waits 8 measurements
  (`NVISION_MIN_STEPS_BEFORE_NARROWING`) and must shrink the focus by at least 5 %. The focus is re-decided after every
  resampling. With no significant dip the focus is unchanged. The UI's per-step focus overlay draws exactly these sections.
- **Falsification probe.** A strategy that has piled its measurements on a suspected dip cannot be contradicted
  elsewhere, so with probability $0.02 + 0.08\,g$ ($g$ = the widest unmeasured gap as a fraction of the window)
  SBED measures the middle of that gap instead of the EIG maximiser, outside the focus.

## Known deviations (docs/code vs. article)

Items where the article and the code differ, or where this doc could not confirm the code; none were changed here.

- **Prior.** The article's simulated runs are global searches: uniform prior on $\delta_Z$ over the window the truth is
  drawn from. The generators additionally build *narrowed Gaussian priors* (`PRIOR_STD_FRACTION`, random prior mean offset
  `PRIOR_MEAN_OFFSET_SIGMAS`) for the other parameters; a Gaussian *tracking* run (truth drawn from the prior) is not simulated in
  the article yet. Whether the `zeeman_shift` prior is uniform in the shipped grids was not re-verified.
- **Stage names.** Article: search / localize / refine. Code and UI: search / focus / refine, with boundaries from the
  claimed uncertainty (0.7 drop, then within the tolerance), not from the first detection of the dip. See
  [locator comparison](locator_comparison.md).
- **SNR.** The code label is `contrast_total / noise_sigma` with `contrast_total` $= 2C$; the article's SNR is $C/\sigma$.
- **Bound name.** The code and several docs say CRLB for what the article calls the van Trees bound (CRLB plus prior information $J_0$).
- **Ceiling variables.** The $\theta$ ceiling of 100 kHz is stated for the primary parameter; in `nvision/sim/defaults.py` the
  named 100 kHz threshold is `NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD`. Which variable binds a `zeeman_shift` run was not traced.

---

*For CLI usage and caching logic, refer to [`cli_reference.md`](cli_reference.md) and [`caching.md`](caching.md).*
