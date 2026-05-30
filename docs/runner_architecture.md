# Runner Architecture & Orchestration

## Overview

The `nvision/runner/executor.py` module is responsible for orchestrating the execution of experiments. Because Bayesian inference is computationally intensive and experiments often involve thousands of simulated measurements across multiple strategies and noise levels, the runner employs a highly parallelized, streaming architecture.

## Execution Lifecycle

### 1. Repeat Artifact Generation
Instead of running a single long simulation, the runner breaks down tasks into **repeats** (independent runs with different random seeds) for statistical robustness.
- Each repeat generates its own independent random number generator (RNG).
- Measurements within a single repeat are deterministic given that repeat's specific seed.
- If a repeat is aborted, it can be resumed later without affecting others.

### 2. The Sobol Baseline (`_run_sobol_baseline`)
For every **Bayesian** strategy being tested (e.g., `SequentialBayesianExperimentDesignLocator`), the runner automatically performs a "Sobol Baseline" measurement first.

- **Purpose**: To provide a ground-truth benchmark of how a completely uniform, un-targeted random sequence (a van der Corput Sobol sequence) would converge given the same amount of time/noise. It calculates the expected uniform points needed.
- **Isolation**: The baseline is completely decoupled from the actual strategy execution. It instantiates a fresh `SimpleSobolBayesianLocator` and a fresh `UnitCubeSMCMarginalDistribution` (belief). 
- **Decoupling**: Non-Bayesian locators (like `StagedSobolSweepLocator` in `coarse/sobol_locator.py`) might still track a "belief" internally to observe data, but the executor explicitly avoids running the 10,000-step Bayesian baseline on them to prevent heavy inference algorithms from throttling coarse, fast sweep strategies.

### 3. The Main Locator Phase
After the baseline completes (or is skipped), a fresh belief is instantiated for the primary locator strategy.
- The `_run_single_repeat` method drives the primary locator's `.next()`, `.observe()`, and `.done()` hooks.
- A hard timeout is enforced via a threading monitor (e.g., stopping the loop if it exceeds `timeout_s`).
- At the end of the acquisition loop, if it's a Bayesian locator, the runner dumps the full posterior sample array into a parquet/feather artifact for downstream visualization.

### 4. Memory & Streaming Optimizations
When tasks request hundreds of combinations or repeats, storing all artifacts in memory before saving them would cause a memory exhaustion crash. 
The runner implements a streaming mode:
- If `repeats > STREAMING_REPEAT_THRESHOLD`, the runner yields results one-by-one as a generator.
- Results are saved to the cache on-the-fly (`save_cached_combination`).
- The in-memory history is cleared aggressively to keep the RAM footprint stable.
