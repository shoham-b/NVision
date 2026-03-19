# AGENTS.md

## NVision AI Agent Guide

This document provides essential knowledge for AI coding agents to be productive in the NVision codebase. It summarizes architecture, workflows, and project-specific conventions.

---

### 1. Project Architecture
- **Modular Python src/ layout**: All core logic is under `src/nvision/` with submodules for CLI, simulation, core utilities, and visualization.
- **Simulation Core**:
  - `sim/` contains experiment generators (`gen/`), noise models, and locator strategies (`locs/`).
  - Locators (peak-finding strategies) follow the `Locator` protocol (see `sim/locs/base.py`).
  - NV center simulation variants are implemented in `sim/gen/generators/nv_center_generator.py`.
- **CLI Entrypoint**:
  - Main Typer CLI in `src/nvision/__main__.py` and `cli/main.py`.
  - The `run` command (see `cli/run.py`) orchestrates full experiment pipelines, including caching, result aggregation, and plotting.
- **Visualization**:
  - All plots and summaries are generated via the `Viz` class (`viz/__init__.py`), which combines mixins for experiment, measurement, and Bayesian visualizations.
- **Artifacts**: Results, plots, and caches are written to the `artifacts/` directory by default.

---

### 2. Developer Workflows
- **Dependency Management**: Always use [uv](https://github.com/astral-sh/uv) for installing and running Python commands (see `.github/copilot-instructions.md`).
  - Example: `uv sync --group dev` to install all dependencies.
- **Running Experiments**:
  - `uv run python -m nvision run --repeats 5 --loc-max-steps 150`
  - Results are cached and written to `artifacts/`.
- **Testing & Linting**:
  - `uv run pytest -q` for tests.
  - `uv run ruff check` and `uv run ruff format --check` for linting/formatting.
- **Fuzz Testing**: `uv run python -m fuzz.run_fuzz` for robustness checks.
- **Docker**: Build and run containers with `docker build` and `docker run` (see README for details).
- **Makefile**: POSIX make targets are available for common tasks (install, lint, test, coverage, docker-build, etc.).

---

### 3. Project-Specific Conventions
- **Reproducibility**: All experiments use a fixed RNG seed (`nvision.tools.utils.NVISION_RNG_SEED`) and a scenario grid for deterministic results.
- **Caching**: Results and intermediate data are cached in `artifacts/cache/` for efficient repeat runs.
- **Locator Protocol**: New locator strategies must implement the `Locator` interface (`propose_next`, `should_stop`, `finalize`).
- **DataFrames**: Polars is used for all tabular data (not pandas).
- **Plotting**: All visualizations are generated as HTML/PNG in `artifacts/` using Plotly and custom mixins.
- **Configuration**: Main config is in `pyproject.toml` (Ruff, Pytest, setuptools). Pre-commit hooks and CI/CD are configured for code quality.
- **Naming**: Use clear, descriptive names for new strategies, generators, and noise models. Follow the structure of existing modules.

---

### 4. Key Files & Directories
- `src/nvision/cli/run.py`: Main experiment runner and CLI logic.
- `src/nvision/sim/locs/base.py`: Locator protocol and base class.
- `src/nvision/sim/gen/generators/nv_center_generator.py`: NV center signal generator.
- `src/nvision/viz/`: Visualization facade and mixins.
- `artifacts/`: Output directory for all experiment results and plots.
- `pyproject.toml`: Project configuration.
- `.github/copilot-instructions.md`: AI agent conventions (always use `uv`).
- `README.md`: High-level overview and workflow examples.

---

### 5. Integration & Extension
- To add a new locator or generator, follow the structure and registration patterns in `sim/locs/` and `sim/gen/generators/`.
- For new CLI commands, extend `cli/main.py` and register with Typer.
- For new plots, add mixins to `viz/` and register with the `Viz` facade.

---

*For further details, consult code comments and referenced files above. Follow existing patterns for maximum compatibility and reproducibility.*

