# NVision

NVision is a Python-based repository focused on Bayesian localization and signal modeling.

## Project Architecture

- **Modular Python `src/` layout**: All core logic is under `src/nvision/` with submodules for CLI, simulation, core utilities, and visualization. *(Note: in the actual repo, the layout is simply `nvision/` at the root).*
- **Beliefs** (`nvision/belief/`): Posterior representations (`AbstractBeliefDistribution`, grid and SMC implementations, unit-cube variants).
- **Simulation Core**:
  - `nvision/sim/` contains experiment generators (`gen/`), noise models, and locator strategies (`locs/`).
  - Locators (peak-finding strategies) follow the `Locator` protocol (see `nvision/sim/locs/base.py`).
  - NV center simulation variants are implemented in `nvision/sim/gen/generators/nv_center_generator.py`.
- **CLI Entrypoint**:
  - Main Typer CLI in `nvision/__main__.py` and `nvision/cli/main.py`.
  - The `run` command (see `nvision/cli/run.py`) orchestrates full experiment pipelines, including caching, result aggregation, and plotting.
- **Visualization**:
  - All plots and summaries are generated via the `Viz` class (`nvision/viz/__init__.py`), which combines mixins for experiment, measurement, and Bayesian visualizations.
- **Artifacts**: Results, plots, and caches are written to the `artifacts/` directory by default.

## Installation & Developer Workflows

- **Dependency Management**: Always use [uv](https://github.com/astral-sh/uv) for installing and running Python commands.
  - Example: `uv sync --group dev` to install all dependencies.
- **Running Experiments**:
  - Use `uv run --no-sync python -m nvision run --repeats 5 --loc-max-steps 150`.
  - Results are cached and written to `artifacts/`.
  - **Why `--no-sync`?** On Windows, `uv run` may fail with `os error 32` because `nvision.exe` is locked by a prior process (e.g., `nvision serve`). `--no-sync` skips the sync step and uses the existing venv, avoiding the lock.
- **Re-render reports (no re-run)**:
  - `uv run --no-sync python -m nvision render` (default `--out` is the repo `artifacts/` directory, same as `nvision run`) rebuilds `plots_manifest.json` and the static UI from cache.
- **Testing & Linting**:
  - `uv run --no-sync pytest -q` for tests.
  - `uv run --no-sync ruff check` and `uv run --no-sync ruff format --check` for linting/formatting.
- **Fuzz Testing**: `uv run --no-sync python -m fuzz.run_fuzz` for robustness checks.
- **Docker**: Build and run containers with `docker build` and `docker run`.

## Project-Specific Conventions

- **Reproducibility**: All experiments use a fixed RNG seed (`nvision.tools.utils.NVISION_RNG_SEED`) and a scenario grid for deterministic results.
- **Caching**: Results and intermediate data are cached in `artifacts/cache/` for efficient repeat runs.
- **Locator Protocol**: New locator strategies must implement the `Locator` interface (`propose_next`, `should_stop`, `finalize`).
- **DataFrames**: Polars is used for all tabular data (not pandas).
- **Plotting**: All visualizations are generated as HTML/PNG in `artifacts/` using Plotly and custom mixins.
- **Naming**: Use clear, descriptive names for new strategies, generators, and noise models. Follow the structure of existing modules.

## Integration & Extension

- To add a new locator or generator, follow the structure and registration patterns in `nvision/sim/locs/` and `nvision/sim/gen/generators/`.
- For new CLI commands, extend `nvision/cli/main.py` and register with Typer.
- For new plots, add mixins to `nvision/viz/` and register with the `Viz` facade.
