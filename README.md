# NVision

<div align="center">

[![CI Status](https://github.com/shoham-b/NVision/actions/workflows/ci.yml/badge.svg)](https://github.com/shoham-b/NVision/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shoham-b/NVision/branch/main/graph/badge.svg)](https://codecov.io/gh/shoham-b/NVision)
[![Docker Build](https://github.com/shoham-b/NVision/actions/workflows/docker.yml/badge.svg)](https://github.com/shoham-b/NVision/actions/workflows/docker.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://pre-commit.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.12%20--%203.14-blue)](https://www.python.org/)
[![View UI](https://img.shields.io/badge/View_Live_UI-WebApp-green.svg)](https://shoham-b.github.io/NVision/)

</div>

A modular framework for simulating, analyzing, and benchmarking **Nitrogen-Vacancy (NV) centers in diamond**—enabling next-generation quantum sensing experiments and reproducible scientific workflows.

---

## 1. Scientific & Technical Overview

NVision is a simulation framework designed to accelerate research in quantum sensing by providing a platform to develop, test, and benchmark measurement strategies for NV centers. It combines realistic physical models with modern, data-driven analysis pipelines.

- **Advanced Locator Strategies**: Implements and compares multiple algorithms for locating ODMR peaks, from simple grid scans to sophisticated adaptive methods. A key feature is the `SequentialBayesianLocator`, based on the work of Dushenko et al. (2020), which uses Bayesian inference to dramatically speed up measurements.
- **Realistic Physics Models**: Simulates authentic ODMR spectra, including the characteristic three-dip `NVCenterManufacturer` and generic `CauchyLorentz` peak shapes, under configurable noise models.
- **Modular & Extensible**: The `LocatorStrategy` protocol allows for easy integration of new custom search algorithms, making it an ideal testbed for novel sensing protocols.
- **Reproducible Science**: Every experiment is fully controlled by a single seed and scenario grid, ensuring identical results for repeated runs.
- **Analysis Pipelines**: Data and results are processed with fast, modern tabular tools (Polars DataFrames), supporting advanced statistical summaries and error analysis.
- **Visualization**: Automatically generates comparison plots to easily interpret strategy performance, uncertainty, and real-world impact.
- **Interactive Visualizer**: Experiment with NV-center parameters in this [Desmos NV center calculator](https://www.desmos.com/calculator/ynsoiy3ofk) to build intuition before running simulations.
- **Applications**: Designed for quantum magnetometry, microscopy, and custom quantum sensing protocol development.

---

## 2. Installation Guide

Get started quickly on any platform.

### Prerequisites

- Python ≥3.12
- Git (for cloning)
- [uv](https://github.com/astral-sh/uv) (for installation)
- Optional: Docker (for containers)

### Steps

1. **Clone the repository**
```

git clone https://github.com/shoham-b/NVision.git
cd NVision

```
2. **Create and activate a virtual environment**
```

python -m venv .venv
source .venv/bin/activate

```
3. **Install dependencies**
```

uv sync --group dev

```

---

## 3. User Workflow

Run simulations, analyze results, and visualize outcomes—all in a reproducible pipeline.

### Basic Usage

- **Run NV-center simulations**
```

uv run python -m nvision --repeats 5 --seed 123 --loc-max-steps 150

```
- Generates experimental datasets across strategies and noise conditions
- Automatically produces CSV summaries and performance plots in `./artifacts`
- Results are cached for efficient repeat runs

- **Test code integrity** (optional)
```

uv run ruff format --check
uv run ruff check
uv run pytest -q

```

- **Fuzz Testing** (stress-test for robustness)
```

uv run python -m fuzz.run_fuzz

```

- **Inspect Results**
  - CSV files: scenario grids and measurement outcomes
  - Plots:
    - `experiment_summary_.png`: RMSE by noise/strategy
    - `locator_summary_single_.png`, `locator_summary_double_.png`: uncertainty/error for single/double peak contexts

### Docker Usage

- **Build runtime container**
```

docker build -t nvcenter:dev -f Dockerfile --target runtime .

```
- **Run in isolation**
```

docker run --rm -v $(pwd)/artifacts:/workspace/artifacts nvcenter:dev --repeats 5 --seed 123 --loc-max-steps 150

```
- Ensures clean, reproducible environments and portable results

---

## 4. Developer Guide

Join research and development efforts or extend NVision for custom projects.

### Core Features

- **Modern Python src/ layout** for modularity and clarity
- **Comprehensive CI/CD**:
- GitHub Actions check linting (Ruff), formatting, and tests (Pytest) across OSes and Python versions (3.12–3.14)
- Automated Docker builds and smoke-tests for reliability
- **Pre-commit hooks** for instant code quality
- **Makefile commands** for developer efficiency (POSIX make, or use commands directly):
```

make install
make lint
make format
make test
make coverage
make docker-build
make docker-run

```
- **Fuzz testing** with Hypothesis + custom runner for scientific resilience

### Configuration At-a-Glance

- `pyproject.toml`: main config for Ruff, Pytest, and setuptools
- `.pre-commit-config.yaml`: pre-commit settings
- `.github/workflows/ci.yml`, `.github/workflows/docker.yml`: GitHub Actions pipelines
- `Dockerfile`: build container images

### Contribution

Feedback, feature requests, and PRs are welcome!
Please read code comments, follow style guides (Ruff/Prettier), and add tests for new modules.

---

## 5. Contact & License

- Maintainer: Shoham Baris (shoham.baris@mail.huji.ac.il)
- License: MIT

---

*NVision empowers scientists and developers to push the boundaries of quantum sensing research, with robust, efficient, and reproducible workflows from code to discovery.*
