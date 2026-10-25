# AGENTS.md

## NVision AI Agent Guide

This document provides essential knowledge for AI coding agents to be productive in the NVision codebase. It summarizes architecture, workflows, and project-specific conventions.

---

### 1. Project Architecture & Documentation

To understand the core design of the inference engine, you **MUST** read the documentation located in the `docs/` folder. `AGENTS.md` is strictly for agent behavioral rules and workflows. 

**Mandatory Reading for Architecture:**
- **[core_architecture.md](file:///c:/Users/shoha/git/NVision/docs/core_architecture.md)**: Explains the Simulation Orchestration, the Sequential Monte Carlo (SMC) engine, the Bayesian SBED locator, and the critical **Unit-Cube Parameter Scaling** conventions.
- **[caching.md](file:///c:/Users/shoha/git/NVision/docs/caching.md)**: Explains how the SQLite cache operates and the `--no-cache` / `--dry-run` flag behaviors.
- **[evaluation_metrics.md](file:///c:/Users/shoha/git/NVision/docs/evaluation_metrics.md)**: Read before adding or changing any locator-comparison metric or results view: what each stage (search / focus / refine) is driven by, which axes to use, and the rules (no end-of-run accuracy as a quality metric, one shared budget, raw metrics before ranking).

**Brief Repository Map**:
- `nvision/sim/`: Generators, Noise Models, and Locators.
- `nvision/belief/`: SMC implementation and Unit-Cube marginal wrappers.
- `nvision/runner/`: Task orchestration, CLI combinations, and execution.
- `nvision/viz/`: Plotly mixins and UI facades.
- `static/`: Frontend assets (Do NOT analyze directly, run the server instead).
- `docs/design_system/`: The UI design system (tokens, component markup, full-page skeleton). Read its `README.md` before writing or restyling any UI.
- `artifacts/`: Output directory for all experiment results, plots, and caches.

---

### 2. Developer Workflows
- **Dependency Management**: Always use [uv](https://github.com/astral-sh/uv) for installing and running Python commands (see `.github/copilot-instructions.md`). Never run python directly (e.g., `python file.py`); always use the `uv run nv <command>` interface. For full CLI documentation, see **[CLI Reference](file:///c:/Users/shoha/git/NVision/docs/cli_reference.md)**.
- **Running Experiments & Verification**:
  - Use `uv run nv run --repeats 5` to execute regular simulation runs.
  - To quickly verify that the CLI, locators, and task orchestration are working without performing heavy simulations or overwriting cached results, run this lightweight dry-run:
    `uv run nv run-single NVCenter-lorentzian "Gauss(0.01)" Bayesian-SBED --repeats 1 --runners 1 --dry-run`
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
- **Locator Protocol**: New locator strategies must implement the `Locator` ABC (`nvision/models/locator.py`): the classmethod `create()`, `next()`, `done()`, `result()`, and `observe()`. Some concrete sweep locators additionally implement a `finalize()` hook (called once after the last observation, before `result()`) — it is an addition on top of the ABC, not a substitute for `done()`/`result()`.
- **DataFrames**: Polars is used for all tabular data (not pandas).
- **Plotting**: All visualizations are generated as HTML/PNG in `artifacts/` using Plotly and custom mixins.
- **Configuration**: Main config is in `pyproject.toml` (Ruff, Pytest, setuptools). Pre-commit hooks and CI/CD are configured for code quality.
- **Configuration Management**: When there is an algorithmic constant that is likely to be needed for fine-tuning or modification, ask the user if they want to add it to `.env`. If they do, use `nvision/cli/defaults.py` to expose the environment variable default, and make sure to add it to both `.env` and `.env.example`.
- **Cache & Render Invalidation**: If you modify a core mathematical or algorithmic detail (e.g., SMC resampling, Bayesian likelihood), you MUST notify the user whether they need to completely re-run the simulation (to update `artifacts/cache/`) or just run `uv run nv render` (via `nvision/cli/render.py`) to update the visuals.
- **Array Documentation**: When adding or modifying functions that process multi-dimensional numpy/polars arrays, always document the expected array shape and physical meaning in the docstring (e.g., `shape: (n_particles, n_parameters)`).
- **Plotting**: Always use **Plotly** for visualizations to maintain UI interactivity. Do NOT use static Matplotlib unless explicitly requested for an academic paper.
- **Scratch & Debug Files**: All temporary scratch files, check scripts, or refactoring scripts MUST be placed inside the `scratch/` directory. All debug or reproduction scripts MUST be placed inside the `debug/` directory. Do not write scratch or debug files in the repository root.
- **Vocabulary (never plain "frequency")**: `drive_freq` (measured axis: the swept microwave drive frequency), `candidate_drive_freq` (point to measure next), `zero_field_freq` (dip-centre model parameter), `zeeman_shift` (never a frequency). See **[glossary.md](file:///c:/Users/shoha/git/NVision/docs/glossary.md)**.
- **Unit vs Physical Coordinate Suffixes**: Always strictly suffix variables, getters, and setters with `_phys` (physical space, e.g. Hz) or `_unit` (normalized space, in `[0, 1]`) when dealing with parameter bounds, observations, or candidates (e.g. `obs_drive_freqs_phys`, `obs_drive_freqs_unit`, `lo_phys`, `lo_unit`). This prevents coordinate space mismatches between normalized SMC particles/observations and physical algorithms.
- **No Algorithmic Fallbacks & Fail Fast**: Fallbacks that silently alter the algorithmic behavior (e.g. swallowing errors, parameter truncations, or coordinate scaling fallbacks) are strictly forbidden. When an inconsistency is encountered (like a coordinate mismatch, shape mismatch, or invalid parameter state), the code must **fail fast** by raising an explicit exception (like `ValueError` or `AssertionError`). This ensures logic bugs are caught immediately rather than causing hard-to-debug silent failures.



---

### 3b. Fast, Low-Token Dev Loop (SBED / SMC / Metrics)
Mirrored in `.agent/rules/fast-dev-loop.md` (Antigravity) and `.github/copilot-instructions.md`; keep the three in sync.
- **Run tests without coverage.** Coverage is opt-in (it made a 3 s test take ~30 s and printed ~20 KB). Plain `pytest` is fast; CI passes `--cov=nvision --cov-report=...` explicitly. For iteration use: `uv run pytest tests/<file>.py` (add `--cov=nvision --cov-report=term-missing` only when you need coverage).
- **Pick tests by area, do not run the suite.**
  - SBED / focus: `test_sbed_acquire.py`, `test_sbed_thinning.py`, `test_focus_*.py`, `test_refocus.py`, `test_dip_detection.py`
  - SMC / belief: `test_smc_*.py`, `test_unit_cube_belief.py`
  - Metrics: `test_stage_metrics.py`, `test_grid_study_metrics.py`, `test_milestone_metrics_fit_override.py`
- **Never open these** (huge, generated, or irrelevant): `cprofile_output.txt` (~1 MB), `pytest_output.txt`, `notebook.pdf`, `uv.lock`, `artifacts/`, `logs/`, `htmlcov/`, `coverage.xml`, `static/app.js` (~8k lines).
- **Large modules: grep, then read a line range.** Do not read whole files: `spectra/nv_center.py` (2.4k lines), `belief/smc_marginal.py` (1.7k), `cli/run.py` (1.7k), `runner/executor.py` (1.6k), `sim/locs/coarse/generic_sweep_locator.py` (1.4k).
- **Where things live:** SBED `sim/locs/bayesian/sbed_locator.py`; SMC `belief/smc_marginal.py`; focus `belief/focus_window.py`; dips `belief/dip_detection.py`; metrics `metrics/` and `runner/metrics.py`.

---

### 3c. Guardrails (no unintended actions)
Mirrored in `.agent/rules/guardrails.md` (Antigravity) and `.github/copilot-instructions.md`.
- **Stay in scope.** Change only what the task asks for. No drive-by refactors, renames, reformatting of untouched files, or dependency/`pyproject.toml`/CI changes unless requested. If you spot something else worth fixing, say so instead of doing it.
- **Never touch the shared cache or results.** `artifacts/` (incl. `artifacts/cache/`, tens of GB and shared across worktrees), `data/`, `logs/` are read-only for you. No `nv cache clean`, no deleting or overwriting cache/DB files, no full `nv run` / `nv groups run`. Use `--dry-run` or `run-single --repeats 1 --dry-run`.
- **Restricted Git Workflow.** A restricted set of git commands is allowed: creating and switching to a dedicated task/feature branch (`git checkout -b <branch>`), staging and committing changes (`git add`, `git commit`), pushing that branch to remote (`git push -u origin <branch>` / `git push`), committing further to that branch, and opening PRs (e.g. via `gh pr create`). Never commit or push directly to `main` or the default branch. Never use destructive or history-rewriting commands (`git reset --hard`, `git push --force`, rebase, or checking out over uncommitted changes) and never use `--no-verify`.
- **No destructive or outward-facing commands** (rm -r, drop/truncate, kill, deploys, GCP/docker pushes, anything under `.github/workflows` or `scripts/` that talks to a remote) without explicit confirmation in chat.
- **Physics/algorithm changes need a heads-up.** If you change SMC, SBED, likelihood, focus, or a physics constant, state whether old results are stale (re-run vs `nv render`) and whether the physics fingerprint must be bumped. Do not bump it silently.
- **Fail fast, no silent fallbacks.** Raise on shape/coordinate mismatches; do not add try/except, clipping, or defaults that hide a bug. Keep `_unit` / `_phys` suffixes strict.
- **Scratch only in `scratch/` and `debug/`.** Never in the repo root.
- **Uncertain or ambiguous? Ask first**, with a recommendation, rather than choosing a risky interpretation.

---

### 4. Integration & Extension
- To add a new locator or generator, follow the structure and registration patterns in `sim/locs/` and `sim/gen/generators/`.
- For new CLI commands, extend `cli/main.py` and register with Typer.
- For new plots, add mixins to `viz/` and register with the `Viz` facade.

---

### 5. Documentation (`docs/`)
The `docs/` directory contains permanent project documentation generated via Sphinx (using Markdown/MyST).
- **When to use**: Write or update files in `docs/` for permanent architectural overviews, complex mathematical/physics derivations (e.g., `contrast_reparametrization.md`), core subsystem explanations (e.g., `caching.md`, `cli_reference.md`), and broad design patterns.
- **When NOT to use**: Do NOT put quick scratch notes, temporary debug scripts, or task plans here. Do NOT place agent behavior rules here (those go in `AGENTS.md` or `.github/copilot-instructions.md`). 
- **Format**: Prefer **Markdown (`.md`)** for general explanations and system architecture. Use **LaTeX** inside markdown only for complex math or physics equations.
- **Index**: When creating a new documentation file, always update `docs/index.md` (or the relevant toctree) to include a link to it.

---

*For further details, consult code comments and referenced files above. Follow existing patterns for maximum compatibility and reproducibility.*

