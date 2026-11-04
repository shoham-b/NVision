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
| `matlab` | Add the lab's MATLAB data (Drive link, folder or `.mat`) and sync it now: Bayesian-run lab studies and SBED on raw sweeps. |

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
Add the lab's MATLAB data and sync it now. `SOURCE` is a Google Drive folder link (or id), a local folder, or a
single `.mat` file. It is remembered in `<dir>/labs/sources.json`. `nv serve` never syncs by itself; to pick up
later changes run `nv matlab --dir <dir>` with no `SOURCE`, or press **Sync now** on any lab page (it runs that in a
subprocess, log in `<dir>/labs/sync.log`). A sync downloads only the files missing or changed on the Drive
(`rclone copy`) and processes only new or changed files: a file's kind is remembered by size and mtime, and a raw sweep is redone only when its file
changes, when the processing version (`PROCESSING_VERSION` in `nvision/tools/lab_sources.py`) is bumped, or when it
failed last time. Failures (an expired rclone login, an unreadable file, a sweep that crashed) are listed on the
lab page.

Every `.mat` file below `SOURCE` is handled by what it holds:

- **The lab's Bayesian ESR runs** (`myStruct.BayesianESR`): processed into one lab study per folder (claimed σ vs the
  scatter between runs, steps to convergence) and listed as "Lab: …" in the UI's Study dropdown. A run that stopped on
  precision keeps its own stop; a full-budget run has its stop replayed from its history. A folder the lab's own
  `Gather_and_Process_data.m` already summarised keeps those statistics.
- **Raw ESR sweeps** (`myStruct.ESR`): run through SBED twice, a few seconds each, seeded from the file's path so a
  re-sync reproduces them:
  - on all the data (noise `real`);
  - on the even sweeps only (noise `real-holdout`), with the odd sweeps fitted by the same model and bounds as an
    independent reference: z = (estimate − ref) / √(σ² + σ_ref²) for `zeeman_shift` and `zero_field_freq`, plus the
    fit's χ²/dof (above 2 the model does not describe the data, so z is too optimistic). See
    `nvision/metrics/lab_reference.py`.

  Both appear per run under "MATLAB (real data)" (generator `MATLAB:<source>/<path>`), and every file is a row of
  **"Lab: raw sweeps (SBED)"**: per folder (one experiment), the estimates, the reference, z, χ²/dof, `shot_reuse` and
  the share of files with |z| ≤ 1 / ≤ 2 against 68% / 95%. One file is one independent z, so the shares are over the
  files of a folder, never pooled across folders or taken over seeds. `shot_reuse` is draws per distinct recorded
  shot: SBED draws with replacement and counts each draw as new evidence, so a value well above 1 (a file with few
  sweeps) shrinks its claimed σ.

A Drive link is downloaded with `rclone` (read-only `gdrive` remote; `.png`/`.fig` skipped) into
`<dir>/downloads/<folder id>`; if the login expired run `rclone config reconnect gdrive:`. Everything is stored under
`--dir` (results in `<dir>/cache` and `<dir>/matlab/cache`, per-file records in `<dir>/labs/<source>/sweeps/`).

**Common Options:**
- `source`: Drive folder link/id, local folder, or `.mat` file.
- `--dir`: artifacts directory results are stored under (default: `artifacts/`; give `nv serve` the same one).
- `--no-sweeps`: skip the raw sweeps this time (they are still picked up by the next sync).
- `--noise-std`, `--max-steps` (default 300), `--valid-shots`, `--particles` (default 10000), `--out` (also copy each
  file's record there), `--no-ui`, `--no-progress`, `--infer-center-freq` / `--no-infer-center-freq`: options of
  this run's sweeps. `nv matlab` with no `SOURCE` and the Sync button always use the defaults.

**Example Use Cases:**
```bash
# Once: the lab's shared Drive folder
uv run nv matlab "https://drive.google.com/drive/folders/<id>"

# Later, after new files land on the Drive
uv run nv matlab

# Open the UI
uv run nv serve

# One raw sweep recording through SBED
uv run nv matlab data/matlab/sample.mat
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
