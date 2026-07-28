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

## Usage and Examples

### `nv run`
Runs a batch of simulations across multiple locators and noise levels.

**Common Options:**
- `--repeats`: Number of repeat experiments (default: 5).
- `--loc-max-steps`: Maximum steps for Bayesian locator (default: 1200).
- `--filter-category`, `--filter-strategy`, `--filter-generator`, `--filter-noise`: Filter scenarios to run.
- `--runners`: Number of parallel runner processes (default: `min(8, cpu_count // 2)`, scales with the machine — see `NVISION_DEFAULT_RUNNERS`; use 1 for sequential execution). Each worker's numba/BLAS thread count is capped to the leftover cores (`cpu_count // runners`) so processes and intra-worker threads don't oversubscribe.
- `--no-cache`: Force bypass of the simulation cache.
- `--dry-run`: Do not write results to cache.
- `--open`: Open the results browser after completing.

**Example Use Cases:**
```bash
# Standard batch run with 5 repeats
uv run nv run --repeats 5 --loc-max-steps 150

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
uv run nv run-single NVCenter-lorentzian "Gauss(0.01)" Bayesian-SBED --loc-max-steps 3 --repeats 1 --runners 1 --dry-run
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

### `nv cache`
Manage the simulation SQLite cache.

**Example Use Cases:**
```bash
# List all simulations in cache
uv run nv cache list

# Delete specific runs from cache
uv run nv cache clean --filter-strategy Sweep
```
