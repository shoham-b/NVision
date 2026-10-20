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

The system tracks uncertainty and posteriors using **Sequential Monte Carlo (SMC)** (`smc_marginal.py`). 
- Belief is represented by an N-dimensional cloud of discrete particles.
- The locator calculates likelihoods against the current noise model and resamples particles to narrow the posterior around the true parameter values as new observations are collected.

### 3. Unit-Cube Scaling Architecture

A critical design feature of the inference engine is the strict separation between internal algorithmic state and external physical representation.

- **Unit Normalized Parameters (`[0, 1]`)**: The core SMC engine and likelihood algorithms operate strictly on the unit-cube `[0, 1]`. This ensures uniform convergence thresholds, prevents scale imbalances during multidimensional acquisition optimizations, and makes the core algorithms completely agnostic to the underlying physical dimensions.
- **Physically Scaled Parameters**: The physical bounds and scaling logic are abstracted away from the core particle math.
- **`SMCMarginalDistribution`**: Particles and likelihoods live on the unit cube, while the public summaries (`.estimates()`, `.uncertainty()`, covariance matrices, `.get_candidate_drive_freq_phys()`) transparently denormalize back into physical scales for the CLI monitors and UI plots. The drive-frequency axis is always the full domain; which part of it may be scanned (the focus) is owned by the locator, not the belief.

### 4. Sequential Bayesian Experiment Design (SBED)

The flagship locator strategy is the SBED locator (`nvision/sim/locs/bayesian/sbed_locator.py`). It works across all three lineshape families above (Lorentzian/Voigt/Saturation-Voigt, with or without Zeeman/hyperfine structure) under Gaussian or drift noise — it is not limited to the plain Lorentzian-under-Gaussian-noise case.

- **Prior Initialization**: When a simulation starts, the generator provides the deterministic parameter boundaries. To ensure efficient convergence, the SBED locator does not use flat uniform priors — particles are initialized using dynamically narrowed **Gaussian priors** drawn around the underlying values, sized via `PRIOR_STD_FRACTION` (`nvision/spectra/nv_center.py`). `zero_field_freq` itself is fixed by default (see above), so in the default configuration it is not part of this randomized initialization at all — only the free shape parameters (hwhm, split, hyperfine, contrast, and `zero_field_freq` itself when a locator explicitly enables `with_fixed_zero_field_freq=False`) are.
  The prior's *mean* is itself a random draw, `gauss(true_value, PRIOR_STD_FRACTION-width * PRIOR_MEAN_OFFSET_SIGMAS)` — deliberately wider than the prior's own reported std (default multiplier 3.0), so the prior is usually centered a couple of sigma away from the true value rather than suspiciously close to it, simulating a real experimentalist's imperfect calibration guess instead of implicitly leaking the answer. Particle sampling truncates (not clips) to the parameter's physical bounds, so an out-of-range prior mean is handled correctly rather than collapsing particles onto the boundary (see `_sample_truncated_normal` in `nvision/belief/smc_marginal.py`).
- **Acquisition Strategy**: The locator iteratively proposes new experimental coordinates (e.g., candidate_drive_freq points) that are explicitly calculated to maximize the expected information gain (reducing the entropy) of the particle cloud.
- **Focus sections (dip-based)**: The locator owns a `FocusRegion` (`nvision/belief/focus_window.py`) — a set of disjoint drive-frequency sections — that restricts which epoch candidates EIG may pick (the 10% decaying uniform probe still samples the full axis). It is deduced from measurements, not from the particle belief: after each resample the locator takes the significant dips from `find_dips` (`nvision/belief/dip_detection.py` — a shape-agnostic boxcar GLRT scan over windows whose widths span the hwhm prior, kept if the score passes the Bonferroni-corrected family-wise false-alarm test `NVISION_DIP_P_FA`, each connected run of significant windows being one dip) and gives each dip its own section: its `[f_min, f_max]` run extent (at least the line core) padded by `NVISION_SMC_FOCUSING_COVER_FACTOR` (= 3.0) × the posterior-mean effective HWHM per side, clamped to the axis, with overlapping sections merged. A dip section outside the current focus **expands** it (union) immediately; **narrowing** waits for `NVISION_MIN_STEPS_BEFORE_NARROWING` steps (and as many since the last expansion) and must shrink the total covered width by ≥5%. With no significant dip the focus is unchanged (a full axis stays full). Because dips are model-independent, this is active in default fixed-`zero_field_freq` runs too. Two well-separated dips give two sections, and the empty axis between them is not scanned. The UI's per-step focus overlay draws exactly these sections; the end-of-run `focus_window` field is their outer extent.

---

*For CLI usage and caching logic, refer to [`cli_reference.md`](cli_reference.md) and [`caching.md`](caching.md).*
