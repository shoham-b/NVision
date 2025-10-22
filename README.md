# NvCenter

Project template scaffold with:
- src/ package layout
- Pytest tests (including Hypothesis-based fuzzing)
- Lightweight custom fuzz runner (fuzz/run_fuzz.py)
- Ruff linting and formatting with strict checks
- GitHub Actions CI (Ruff + Pytest)
- Comprehensive .gitignore

Additionally, includes a clean, extensible simulation framework for NV-center measurements under nvcenter.sim:
- Multiple data generators (Rabi oscillation, T1 decay)
- Multiple noise models and compound noise application
- Multiple measurement strategies
- A runner to evaluate strategy performance across noise types
- Uses Polars for tabular experiment summaries (fast DataFrame + CSV export)

Note: The noise API has been modernized to operate on DataBatch objects (with an internal Polars DataFrame) rather than raw lists. This enables broader Polars usage throughout the simulation pipeline. See nvcenter.sim.core.DataBatch and nvcenter.sim.noise.* classes.

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install in editable mode with dev tools:

   pip install -e .[dev]

3. Lint and format checks (no changes applied):

   ruff format --check
   ruff check

4. Run tests:

   pytest -q

5. Run the combined simulations (writes CSVs under ./artifacts by default):

   python -m nvcenter --repeats 5 --seed 123 --loc-max-steps 150

   This runs both the time-series ExperimentRunner (Rabi/T1) and the 1-D LocatorRunner (OnePeak/TowPeak with gaussian/rabi/t1_decay modes) across several noise presets using Polars for scenario grids and result tables.

   Results are cached by scenario (seed, repeats, strategies, noises) under ./artifacts/cache to avoid re-computation on repeated runs. Caching uses the diskcache library for robust on-disk storage. You can safely delete this folder to force recomputation.

   The command also generates comparison plots under ./artifacts:
   - experiment_summary_<GEN>.png (RMSE by noise/strategy for each generator)
   - locator_summary_single_<GEN>.png (single-peak errors/uncertainty)
   - locator_summary_double_<GEN>.png (two-peak errors/uncertainty)

6. Run the simple fuzz loop locally (infinite loop, press Ctrl+C to stop):

   python -m fuzz.run_fuzz

## Configuration

Reproducibility: All random generation (data + noise + iterative measurements) is driven by a single seed passed via the CLI `--seed`. Using the same seed with the same scenario inputs will yield identical results thanks to caching and consistent RNG usage.
- Ruff configuration is in pyproject.toml under [tool.ruff]* keys.
- Pytest configuration is in pyproject.toml under [tool.pytest.ini_options].
- Setuptools uses a src/ layout defined in pyproject.toml.

## CI
CI runs on push and pull requests:
- ruff format --check
- ruff check
- pytest

You can adjust versions and steps in .github/workflows/ci.yml.


## Docker

Build and run the project in a container (multi-stage image):

- Build runtime image:

  docker build -t nvcenter:dev -f Dockerfile --target runtime .

- Run combined simulations (writes CSVs under ./artifacts on the host):

  docker run --rm -v %cd%/artifacts:/workspace/artifacts nvcenter:dev --repeats 5 --seed 123 --loc-max-steps 150

On Linux/macOS, replace %cd% with $(pwd).

A GitHub Actions workflow (.github/workflows/docker.yml) also builds the image on PRs/pushes and, on tags or main, pushes to GHCR using the repository’s GITHUB_TOKEN.

## CI pipelines

The CI workflow (.github/workflows/ci.yml) runs on Ubuntu, Windows, and macOS for Python 3.9–3.12 and performs:
- Ruff format check and lint
- Pytest with coverage (XML + HTML)
- Uploads coverage files as artifacts

A separate Docker workflow builds the runtime image and smoke‑tests it on PRs.

## Pre-commit hooks

Install and enable pre-commit to run linting locally before commits:

- pip install pre-commit
- pre-commit install

This repository ships .pre-commit-config.yaml configured with:
- ruff-format (code formatting)
- ruff (linting with autofix)
- prettier for Markdown/YAML/JSON

## Makefile helpers

Common tasks (POSIX make):
- make install
- make lint
- make format
- make test
- make coverage
- make docker-build
- make docker-run

On Windows without make, you can invoke the commands shown in the Makefile directly.
