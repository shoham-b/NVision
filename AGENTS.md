# AGENTS.md

## NVision AI Agent Guide

This document provides essential knowledge for AI coding agents to be productive in the NVision codebase. It summarizes architecture, workflows, and project-specific conventions.

---

### 1. Project Architecture & Documentation

To understand the core design of the inference engine, you **MUST** read the documentation located in the `docs/` folder. `AGENTS.md` is strictly for agent behavioral rules and workflows. 

**Mandatory Reading for Architecture:**
- **[core_architecture.md](file:///c:/Users/shoha/git/NVision/docs/core_architecture.md)**: Explains the Simulation Orchestration, the Sequential Monte Carlo (SMC) engine, the Bayesian SBED locator, and the critical **Unit-Cube Parameter Scaling** conventions.
- **[caching.md](file:///c:/Users/shoha/git/NVision/docs/caching.md)**: Explains how the SQLite cache operates and the `--no-cache` / `--dry-run` flag behaviors.

**Brief Repository Map**:
- `nvision/sim/`: Generators, Noise Models, and Locators.
- `nvision/belief/`: SMC implementation and Unit-Cube marginal wrappers.
- `nvision/runner/`: Task orchestration, CLI combinations, and execution.
- `nvision/viz/`: Plotly mixins and UI facades.
- `static/`: Frontend assets (Do NOT analyze directly, run the server instead).
- `artifacts/`: Output directory for all experiment results, plots, and caches.

---

### 2. Developer Workflows
- **Dependency Management**: Always use [uv](https://github.com/astral-sh/uv) for installing and running Python commands (see `.github/copilot-instructions.md`). Never run python directly (e.g., `python file.py`); always use the `uv run nv <command>` interface.
- **Running Experiments & Verification**:
  - Use `uv run nv run --repeats 5 --loc-max-steps 150` to execute regular simulation runs.
  - To quickly verify that the CLI, locators, and task orchestration are working without performing heavy simulations or overwriting cached results, run this lightweight dry-run:
    `uv run nv run-single NVCenter-lorentzian "Gauss(0.01)" Bayesian-SBED --loc-max-steps 3 --repeats 1 --runners 1 --dry-run`
  - Results are cached and written to `artifacts/`.
- **Serving the UI (Server)**:
  - To view the UI and interact with artifacts, run the server: `uv run nv serve`
  - **What to connect to**: By default, this serves on `http://localhost:18080`. Connect your browser to this address to view the UI.
- **Re-render reports (no re-run)**: `uv run nv render` (default `--out` is the repo `artifacts/` directory, same as `nvision run`) rebuilds `plots_manifest.json` and the static UI from cache.
- **Testing & Linting**:
  - `uv run pytest -q` for tests.
  - `uv run ruff check` and `uv run ruff format --check` for linting/formatting.
  - **Constraint**: Do NOT test or run lint checks on files that were not modified by the task requested. Only target your modified files and their corresponding unit tests to verify your changes, rather than running formatting checks or full-suite runs over unmodified files in the repository.
- **Fuzz Testing**: `uv run python -m fuzz.run_fuzz` for robustness checks.
- **Docker**: Build and run containers with `docker build` and `docker run` (see README for details).
- **Makefile**: POSIX make targets are available for common tasks.

---

### 3. Project-Specific Conventions
- **Scope & Focus**: Do NOT get rabbit-holed or obsessed with micro-optimizations, algorithmic minutiae (like the Welford algorithm), or complex `polars` filtering pipelines unless explicitly requested by the user. If you find yourself spending too much time debugging a tiny isolated detail, step back and address the broader macro-objective.
- **Reproducibility**: All experiments use a fixed RNG seed (`nvision.tools.utils.NVISION_RNG_SEED`) and a scenario grid for deterministic results.
- **Caching**: Results and intermediate data are cached in `artifacts/cache/` for efficient repeat runs. Caching is enabled by default. See [caching.md](file:///c:/Users/shoha/git/NVision/docs/caching.md) for details on `--no-cache` and `--dry-run` flag behaviors.
- **Locator Protocol**: New locator strategies must implement the `Locator` interface (`propose_next`, `should_stop`, `finalize`).
- **DataFrames**: Polars is used for all tabular data (not pandas).
- **Plotting**: All visualizations are generated as HTML/PNG in `artifacts/` using Plotly and custom mixins.
- **Configuration**: Main config is in `pyproject.toml` (Ruff, Pytest, setuptools). Pre-commit hooks and CI/CD are configured for code quality.
- **Configuration Management**: When there is an algorithmic constant that is likely to be needed for fine-tuning or modification, ask the user if they want to add it to `.env`. If they do, use `nvision/cli/defaults.py` to expose the environment variable default, and make sure to add it to both `.env` and `.env.example`.
- **Cache & Render Invalidation**: If you modify a core mathematical or algorithmic detail (e.g., SMC resampling, Bayesian likelihood), you MUST notify the user whether they need to completely re-run the simulation (to update `artifacts/cache/`) or just run `uv run nv render` (via `nvision/cli/render.py`) to update the visuals.
- **Array Documentation**: When adding or modifying functions that process multi-dimensional numpy/polars arrays, always document the expected array shape and physical meaning in the docstring (e.g., `shape: (n_particles, n_parameters)`).
- **Plotting**: Always use **Plotly** for visualizations to maintain UI interactivity. Do NOT use static Matplotlib unless explicitly requested for an academic paper.
- **Scratch & Debug Files**: All temporary scratch files, check scripts, or refactoring scripts MUST be placed inside the `scratch/` directory. All debug or reproduction scripts MUST be placed inside the `debug/` directory. Do not write scratch or debug files in the repository root.

---

### 4. Integration & Extension
- To add a new locator or generator, follow the structure and registration patterns in `sim/locs/` and `sim/gen/generators/`.
- For new CLI commands, extend `cli/main.py` and register with Typer.
- For new plots, add mixins to `viz/` and register with the `Viz` facade.

---

### 5. Documentation (`docs/`)
The `docs/` directory contains permanent project documentation generated via Sphinx (using Markdown/MyST).
- **When to use**: Write or update files in `docs/` for permanent architectural overviews, complex mathematical/physics derivations (e.g., `dip_depth_reparametrization.md`), core subsystem explanations (e.g., `caching.md`, `cli_integration.md`), and broad design patterns.
- **When NOT to use**: Do NOT put quick scratch notes, temporary debug scripts, or task plans here. Do NOT place agent behavior rules here (those go in `AGENTS.md` or `.github/copilot-instructions.md`). 
- **Format**: Prefer **Markdown (`.md`)** for general explanations and system architecture. Use **LaTeX** inside markdown only for complex math or physics equations.
- **Index**: When creating a new documentation file, always update `docs/index.md` (or the relevant toctree) to include a link to it.

---

*For further details, consult code comments and referenced files above. Follow existing patterns for maximum compatibility and reproducibility.*

