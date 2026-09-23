# CLI Reference

The NVision CLI (`uv run nv`) is a Typer-driven command-line interface for running experiments, rendering reports, and managing caches.

## Main Commands

| Command | Description |
|---|---|
| `run` | Run a batch of simulation experiments based on provided parameters. |
| `run-single` | Run a single (generator, noise, strategy) combination. |
| `demo` | Quick demo to validate improvements - fast, focused, visual. |
| `groups` | Run preset simulation groups. |
| `render` | Render reports and graphs from cache without running simulations. |
| `serve` | Start a local HTTP server for viewing NVision results. |
| `cache` | Manage simulation cache (list, clean, recalc). |
| `matlab-run` | Run the SBED locator on real ESR measurements from a MATLAB `.mat` file. |

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

# Run the 'sbed-only' group
uv run nv groups run sbed-only

# Or use the shortcut alias
uv run nv groups sbed-only

# Resume an interrupted run: keeps completed/partial results from the latest session,
# and runs unstarted combinations fresh without reading stale pre-session cache:
uv run nv groups both-sbed --repeats 50 --resume
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

### `nv matlab-run`
Run the Bayesian SBED locator against real ESR measurements recorded in a MATLAB
`.mat` file, instead of a simulated generator. Results land in the artifact store
next to simulated runs, so they show up in `nv serve` — see
[`ui_architecture.md`](ui_architecture.md) for how the UI groups these under the
"MATLAB (real data)" study bucket.

**Common Options:**
- `matlab_file`: path to the `.mat` file (or a bare filename resolved via `data/matlab/`). Omit when using `--all`.
- `--all`: run every `.mat` file in `data/matlab/` (or `--dir`) in turn, continuing past any single file's failure so one bad recording doesn't abort the rest.
- `--dir`: directory to scan for `.mat` files with `--all` (default: `data/matlab/`).
- `--noise-std`: override the auto-estimated measurement noise std instead of deriving it from the file's own shot spread.
- `--max-steps`: maximum SBED measurement steps (default: 300).
- `--infer-frequency` / `--no-infer-frequency`: fit the NV zero-field-splitting center instead of fixing it to 2.87 GHz (default: infer — real samples run 1-2 MHz off the textbook value from strain/temperature).
- `--particles`: SMC particle count (default: 10000, 10x the simulation default).
- `--out`: write a JSON result summary to this path.
- `--no-ui`: skip artifact writing (no `nv serve` integration) — useful for a quick numeric check.

**Example Use Cases:**
```bash
# Run one recording
uv run nv matlab-run data/matlab/sample.mat

# Run every recording in a directory, writing per-file JSON summaries
uv run nv matlab-run --all --dir data/matlab --out artifacts/matlab_summaries

# View results in their own small, fast-to-reload cache
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
