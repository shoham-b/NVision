# NVision

NVision is a Python-based repository focused on Bayesian localization and signal modeling.

## Implemented Algorithms

NVision implements various algorithms for signal generation, noise simulation, and Bayesian localization (peak-finding).

### Locators (Search Strategies)
Locators are strategies used to find signal peaks. They implement the `Locator` protocol.
- **Coarse Locators**: E.g., `SobolLocator` and `SweepLocator` for initial wide-area scanning.
- **Bayesian Locators**: Use probabilistic models to guide the search.
  - `SequentialBayesianLocator`: Main sequential search.
  - `MaximumLikelihoodLocator`: Chooses points maximizing current likelihood.
  - `SBEDLocator`: Uses Sequential Bayesian Experimental Design.
  - `UtilitySamplingLocator`, `StudentsTLocator`.

### Beliefs (Posterior Representations)
Represent the current knowledge about the parameter space.
- Includes `AbstractBeliefDistribution`.
- Support for grid-based and SMC (Sequential Monte Carlo) implementations, as well as unit-cube variants.

### Generators & Spectra
Models for generating synthetic signals to test locators.
- **Generators**: E.g., `NVCenterGenerator` for simulating Nitrogen-Vacancy center physics, `MultiPeakGenerator`, `SymmetricTwoPeakGenerator`.
- **Spectra**: Underlying mathematical models like `Gaussian`, `Lorentzian`, and composite models like `VoigtZeeman`.

## Installation & Developer Workflows

- **Dependency Management**: Always use [uv](https://github.com/astral-sh/uv) for installing and running Python commands.
  - Example: `uv sync --group dev` to install all dependencies.

- **Running Experiments**:
  - Run full experiment pipelines using the CLI:
    `uv run --no-sync python -m nvision run --repeats 5 --loc-max-steps 150`
  - Results are cached and written to `artifacts/`.
  - **Note on `--no-sync`**: On Windows, `uv run` may fail with `os error 32` because `nvision.exe` is locked by a prior process. `--no-sync` skips the sync step and uses the existing venv.

- **Re-render Reports (No Re-run)**:
  - Rebuild `plots_manifest.json` and the static UI from cache:
    `uv run --no-sync python -m nvision render`

- **Testing & Linting**:
  - `uv run --no-sync pytest -q` for tests.
  - `uv run --no-sync ruff check` and `uv run --no-sync ruff format --check` for linting/formatting.
  - **Fuzz Testing**: `uv run --no-sync python -m fuzz.run_fuzz` for robustness checks.

- **Docker**: Build and run containers with `docker build` and `docker run`.

## Project-Specific Conventions

- **Reproducibility**: All experiments use a fixed RNG seed (`nvision.tools.utils.NVISION_RNG_SEED`) and a scenario grid for deterministic results.
- **DataFrames**: Polars is used for all tabular data.
- **Plotting**: All visualizations are generated as HTML/PNG in `artifacts/` using Plotly.
