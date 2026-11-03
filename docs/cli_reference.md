# CLI Reference

The NVision CLI (`uv run nv`) is a Typer-driven command-line interface for running experiments, rendering reports, and managing caches.

## Main Commands

| Command | Description |
|---|---|
| `run` | Run a batch of simulation experiments based on provided parameters. |
| `run-single` | Run a single (generator, noise, strategy) combination. |
| `demo` | Quick demo to validate improvements - fast, focused, visual. |
| `groups` | Run preset simulation groups. |
| `graphs` | Build the graphs still queued by an earlier run (see [Deferred Graph Generation](graph_queue.md)). |
| `render` | Render reports and graphs from cache without running simulations. |
| `serve` | Start a local HTTP server for viewing NVision results. |
| `cache` | Manage simulation cache (list, clean, recalc). |
| `matlab` | Load the lab's MATLAB data (Drive link, folder or `.mat`): Bayesian-run lab studies and SBED on raw sweeps. |

## Usage and Examples

### `nv run`
Runs a batch of simulations across multiple locators and noise levels.

**Common Options:**
- `--repeats`: Number of repeat experiments (default: 5).
- Step budgets are not CLI options; set `NVISION_DEFAULT_LOC_MAX_STEPS` (see `.env.example`) instead. `nv demo` still takes `--loc-max-steps`.
- `--filter-category`, `--filter-strategy`, `--filter-generator`, `--filter-noise`: Filter scenarios to run.
- `--runners`: Number of parallel runner processes (default: `min(8, cpu_count // 2)`, scales with the machine — see `NVISION_DEFAULT_RUNNERS`; use 1 for sequential execution). Each worker's numba/BLAS thread count is capped to the leftover cores (`cpu_count // runners`) so processes and intra-worker threads don't oversubscribe.
- `--no-cache`: Force bypass of the simulation cache.
- `--dry-run`: Do not write results to cache.
- `--open`: Open the results browser after completing.

**Example Use Cases:**
```bash
# Standard batch run with 5 repeats
uv run nv run --repeats 5

# Run only NVCenter Lorentzian generators across all noises for the Bayesian locator
uv run nv run --filter-generator NVCenter-lorentzian --filter-strategy Bayesian

# Force a clean run without reading from cache, but save to cache
uv run nv run --no-cache

# Run using a single process for easier debugging and traceback
uv run nv run --runners 1
```

### `nv run-single`
Runs exactly one combination of generator, noise, and strategy. Extremely useful for debugging or quick verification.

**Arguments:**
1. `generator`: Generator name (e.g., `NVCenter-lorentzian`).
2. `noise`: Noise descriptor (e.g., `Gauss(0.01)`, `Poisson(5000)`).
3. `strategy`: Strategy name (e.g., `Bayesian-SBED`).

**Example Use Cases:**
```bash
# Run a fast dry-run for a specific strategy and noise level to verify code changes
uv run nv run-single NVCenter-lorentzian "Gauss(0.01)" Bayesian-SBED --repeats 1 --runners 1 --dry-run
```

### `nv groups`
Run preset combinations of simulations defined in `nvision.sim.run_groups.RunGroup`.

**Example Use Cases:**
```bash
# List all available run groups
uv run nv groups list

# Run a group (shortcut: `uv run nv groups voigt-grid`)
uv run nv groups run voigt-grid

# Run a subset of a group's locators/signals (what the former *-only groups did)
uv run nv run --run-group voigt-grid --filter-strategy "^Bayesian-SBED$"
uv run nv run --run-group voigt-grid --filter-signal fw40

# Only the sweeps, across every scenario
uv run nv run --run-group all --filter-strategy Sweep

# Resume an interrupted run: keeps completed/partial results from the latest session,
# and runs unstarted combinations fresh without reading stale pre-session cache:
uv run nv groups voigt-grid --repeats 50 --resume
```

### `nv render`
Re-render reports and regenerate interactive HTML/Plotly visuals from the cache, without actually running the simulations.

**Example Use Cases:**
```bash
# After modifying plotting logic, update the UI without re-running simulations
uv run nv render

# Render only specific strategies
uv run nv render --filter-strategy Bayesian
```

### `nv serve`
Start the local HTTP server to interactively view simulation results.

**Example Use Cases:**
```bash
# Start the UI server (default http://localhost:18080)
uv run nv serve
```

### `nv matlab`
The one command for the lab's MATLAB data. `SOURCE` is a Google Drive folder link (or id), a local folder, or a single
`.mat` file; every `.mat` file below it is handled by what it holds:

- **The lab's Bayesian ESR runs** (`myStruct.BayesianESR`): processed into one lab study per folder (claimed σ vs the
  scatter between runs, steps to convergence) and listed as "Lab: …" in the UI's Study dropdown. A run that stopped on
  precision keeps its own stop; a full-budget run has its stop replayed from its history. A folder the lab's own
  `Gather_and_Process_data.m` already summarised keeps those statistics.
- **Raw ESR sweeps** (`myStruct.ESR`): run through the Bayesian SBED locator instead of a simulated generator. Results
  land in the artifact store next to simulated runs — see [`ui_architecture.md`](ui_architecture.md) for the
  "MATLAB (real data)" study bucket. Runs only when `SOURCE` is a single file, or with `--sweeps` (a folder can hold
  many sweeps and each SBED run takes minutes).

A Drive link is downloaded first with `rclone` (read-only `gdrive` remote; `.png`/`.fig` skipped) into
`<dir>/downloads/<folder id>`; if the login expired run `rclone config reconnect gdrive:`. Everything is stored under
`--dir`, and `nv serve --dir` shows it with no other flags. Run it again on the same `SOURCE` to pick up new files.

**Common Options:**
- `source`: Drive folder link/id, local folder, or `.mat` file.
- `--dir`: artifacts directory results are stored under (default: `artifacts/`; give `nv serve` the same one).
- `--sweeps` / `--no-sweeps`: run SBED on raw sweeps (default: only for a single `.mat` file).
- `--noise-std`, `--max-steps` (default 300), `--valid-shots`, `--particles` (default 10000), `--out`, `--no-ui`,
  `--no-progress`, `--infer-center-freq` / `--no-infer-center-freq`: options of the sweep (SBED) runs.
- `--holdout`: check SBED's claimed σ on real data. SBED gets only the even sweeps; the odd ones are fitted with
  the same model and bounds as an independent reference, and each run reports
  z = (estimate − ref) / √(σ² + σ_ref²) for `zeeman_shift` and `zero_field_freq`, plus the fit's χ²/dof (above 2 a
  warning: the model does not describe the data, so z is too optimistic). A folder run ends with the share of files
  with |z| ≤ 1 / ≤ 2 against 68% / 95%. One file is one independent z, so coverage is across files, never across
  seeds on one file. Stored as noise `real-holdout` (its own cache entry), columns `ref_*` and `holdout_z_*`. See
  `nvision/metrics/lab_reference.py`. Every sweep run also records `shot_reuse`, draws per distinct recorded shot:
  shots are drawn with replacement and each reuse counts as new evidence, so a value well above 1 shrinks the
  claimed σ (and halving the data with `--holdout` doubles it).

**Example Use Cases:**
```bash
# The lab's shared Drive folder: download, process the Bayesian runs, then view every lab
uv run nv matlab "https://drive.google.com/drive/folders/<id>"
uv run nv serve

# One raw sweep recording through SBED
uv run nv matlab data/matlab/sample.mat

# Every raw sweep in a folder, writing per-file JSON summaries
uv run nv matlab data/matlab --sweeps --out artifacts/matlab_summaries

# Is SBED's claimed sigma honest on real data? (held-out sweeps as the reference)
uv run nv matlab data/matlab --sweeps --holdout --no-ui

# View sweep results in their own small, fast-to-reload cache
uv run nv serve --dir artifacts/matlab
```

### `nv cache`
Manage the simulation SQLite cache.

**Example Use Cases:**
```bash
# List all simulations in cache
uv run nv cache list

# Delete specific runs from cache
uv run nv cache clean --filter-strategy Sweep

# Snapshot, list, restore, and prune whole-cache "generations" (see docs/caching.md #4)
uv run nv cache gen save --label before-optimization
uv run nv cache gen list
uv run nv cache gen restore before-optimization
uv run nv cache gen prune --keep 2
```
