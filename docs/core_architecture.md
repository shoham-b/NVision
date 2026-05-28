# Core Architecture: Bayesian Inference and Simulation

## Overview

The core architecture drives simulated experiments and Bayesian inference using a highly modular combination of Generative models, Noise models, and Locator strategies. The system relies heavily on Sequential Monte Carlo (SMC) to represent belief and a strict Unit-Cube scaling system to handle bounded optimization safely.

(Note: Legacy concepts such as `ParameterWithPosterior` and 1D `BeliefSignal` grids have been fully superseded by the N-dimensional SMC architecture).

## Key Components

### 1. Simulation Orchestration (`nvision/runner/`)

Simulations are constructed as Cartesian products (`CombinationGrid`) of three primary components:
- **Generators (`nvision/sim/gen/`)**: Defines the physical experiment parameters and the true ground signal (e.g., `nv_center_generator.py` for Lorentzian signals).
- **Noise Models**: Defines the noise floor (e.g., Gaussian noise) layered over the generator's true signal.
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

The flagship locator strategy is the SBED locator (`nvision/sim/locs/bayesian/sbed_locator.py`), primarily optimized for finding **Lorentzian signals under Gaussian noise**.

- **Prior Initialization**: When a simulation starts, the generator provides the deterministic parameter boundaries. To ensure efficient convergence, the SBED locator does not use flat uniform priors. Instead, particles are initialized using dynamically narrowed **Gaussian priors** drawn around the underlying values, with specific randomized initialization rules applied to the core resonant frequency parameter (`f_b`).
- **Acquisition Strategy**: The locator iteratively proposes new experimental coordinates (e.g., measurement frequencies or sweep times) that are explicitly calculated to maximize the expected information gain (reducing the entropy) of the particle cloud.

---

*For detailed documentation on the CLI integration or caching logic, refer to `cli_integration.md` and `caching.md`.*
