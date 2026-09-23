# Core Architecture: Bayesian Inference and Simulation

## Overview

The core architecture drives simulated experiments and Bayesian inference using a highly modular combination of Generative models, Noise models, and Locator strategies. The system relies heavily on Sequential Monte Carlo (SMC) to represent belief and a strict Unit-Cube scaling system to handle bounded optimization safely.

(Note: Legacy concepts such as `ParameterWithPosterior` and 1D `BeliefSignal` grids have been fully superseded by the N-dimensional SMC architecture).

## Key Components

### 1. Simulation Orchestration (`nvision/runner/`)

Simulations are constructed as Cartesian products (`CombinationGrid`) of three primary components:
- **Generators (`nvision/sim/gen/`)**: Defines the physical experiment parameters and the true ground signal. `nv_center_generator.py`'s `NVCenterCoreGenerator` covers three NV-center lineshapes — Lorentzian, Voigt, and Saturation-Voigt (`nvision/spectra/nv_center.py`) — each optionally with Zeeman splitting and hyperfine structure. All three share the same population-normalized `c_total` contrast convention (see [`dip_depth_reparametrization.md`](dip_depth_reparametrization.md)). `frequency` (the zero-field center) defaults to fixed at the domain midpoint for every draw (`with_fixed_frequency=True`) — like a known, calibrated instrument constant — so only linewidth/split/hyperfine/contrast vary between repeats unless a locator explicitly opts into inferring it.
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
- **`UnitCubeSMCMarginalDistribution`**: This critical wrapper acts as the bridge. It encapsulates the raw unit-cube SMC engine, intercepting requests for public summaries like `.estimates()`, `.uncertainty()`, and covariance matrices to transparently denormalize the `[0, 1]` values back into their true physical scales for the CLI monitors and UI plots.

### 4. Sequential Bayesian Experiment Design (SBED)

The flagship locator strategy is the SBED locator (`nvision/sim/locs/bayesian/sbed_locator.py`). It works across all three lineshape families above (Lorentzian/Voigt/Saturation-Voigt, with or without Zeeman/hyperfine structure) under Gaussian or drift noise — it is not limited to the plain Lorentzian-under-Gaussian-noise case.

- **Prior Initialization**: When a simulation starts, the generator provides the deterministic parameter boundaries. To ensure efficient convergence, the SBED locator does not use flat uniform priors — particles are initialized using dynamically narrowed **Gaussian priors** drawn around the underlying values, sized via `PRIOR_STD_FRACTION` (`nvision/spectra/nv_center.py`). `frequency` itself is fixed by default (see above), so in the default configuration it is not part of this randomized initialization at all — only the free shape parameters (linewidth, split, hyperfine, contrast, and frequency itself when a locator explicitly enables `with_fixed_frequency=False`) are.
  The prior's *mean* is itself a random draw, `gauss(true_value, PRIOR_STD_FRACTION-width * PRIOR_MEAN_OFFSET_SIGMAS)` — deliberately wider than the prior's own reported std (default multiplier 3.0), so the prior is usually centered a couple of sigma away from the true value rather than suspiciously close to it, simulating a real experimentalist's imperfect calibration guess instead of implicitly leaking the answer. Particle sampling truncates (not clips) to the parameter's physical bounds, so an out-of-range prior mean is handled correctly rather than collapsing particles onto the boundary (see `_sample_truncated_normal` in `nvision/belief/smc_marginal.py`).
- **Acquisition Strategy**: The locator iteratively proposes new experimental coordinates (e.g., measurement frequencies) that are explicitly calculated to maximize the expected information gain (reducing the entropy) of the particle cloud.

---

*For CLI usage and caching logic, refer to [`cli_reference.md`](cli_reference.md) and [`caching.md`](caching.md).*
