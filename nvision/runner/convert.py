"""Result conversion utilities — RunResult → Polars DataFrames.

Pure functions with no side-effects; used by the task executor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from nvision.belief.grid_marginal import GridMarginalDistribution
from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution
from nvision.models.observer import RunResult


def denormalize_x(x_norm: float, x_min: float, x_max: float) -> float:
    """Convert normalized [0,1] x to physical domain."""
    return x_min + x_norm * (x_max - x_min)


def run_result_to_history_df(
    result: RunResult,
    repeat_id: int,
    x_min: float,
    x_max: float,
) -> pl.DataFrame:
    """Convert a RunResult's snapshots to a history DataFrame in physical domain.

    Returns a DataFrame with columns: repeat_id, step, x, signal_values.
    Returns an empty typed DataFrame when the result has no snapshots.
    """
    rows = []
    for step, snapshot in enumerate(result.snapshots):
        x = snapshot.obs.x
        # Detect if x is normalized [0,1] or already physical
        # Normalized x is in [0, 1]; physical x for NV centers is ~2.6e9-3.1e9 (Hz)
        x_phys = (
            denormalize_x(x, x_min, x_max) if 0 <= x <= 1 else x
        )  # x is already in physical coordinates (e.g., from SweepingLocator)
        rows.append(
            {
                "repeat_id": repeat_id,
                "step": step,
                "x": x_phys,
                "signal_values": snapshot.obs.signal_value,
            }
        )

    if not rows:
        return pl.DataFrame(
            {
                "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                "step": pl.Series("step", [], dtype=pl.Int64),
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


def extract_peak_estimates(
    belief_estimates: dict[str, float],
    locator_result: dict[str, float],
    x_min: float,
    x_max: float,
) -> dict[str, float]:
    """Map raw locator/belief estimates to physical-domain peak positions.

    Locator result values take priority. Position-like keys (containing
    'x', 'pos', or 'freq') are denormalized when they appear to be in [0, 1].
    """
    estimates: dict[str, float] = {}

    for key, value in locator_result.items():
        if not isinstance(value, (int, float)):
            continue
        if "x" in key.lower() or "pos" in key.lower() or "freq" in key.lower():
            estimates[key] = denormalize_x(value, x_min, x_max) if 0 <= value <= 1 else value
        else:
            estimates[key] = value

    if "frequency" in belief_estimates:
        freq_phys = belief_estimates["frequency"]
        estimates.setdefault("peak_x", freq_phys)
        estimates.setdefault("x1_hat", freq_phys)

    if "split" in belief_estimates:
        estimates["split"] = belief_estimates["split"]

    return estimates


def belief_mode_estimates(belief: object) -> dict[str, float]:
    """Approximate most likely parameters in physical units (for plotting the locator's best guess).

    Grid beliefs: independent marginal argmax on each 1D PMF (product approximation).
    SMC: posterior mean (expected value) via estimates().
    """
    # Unit-cube grid belief: map argmax on unit grid back to physical bounds.
    if isinstance(belief, UnitCubeGridMarginalDistribution):
        modes: dict[str, float] = {}
        for p in belief.parameters:
            name = p.name
            base_param = GridMarginalDistribution.get_grid_param(belief, name)
            idx = int(np.argmax(base_param.posterior))
            u_mode = float(base_param.grid[idx])
            lo, hi = belief.physical_param_bounds[name]
            modes[name] = lo + u_mode * (hi - lo)
        return modes

    # Plain grid belief: argmax directly on each parameter grid.
    if isinstance(belief, GridMarginalDistribution):
        modes = {}
        for p in belief.parameters:
            idx = int(np.argmax(p.posterior))
            modes[p.name] = float(p.grid[idx])
        return modes

    # Fallback for non-grid beliefs (e.g., SMC): use mode estimates if available, else posterior mean.
    if hasattr(belief, "mode_estimates") and callable(belief.mode_estimates):
        out = belief.mode_estimates()
        return {k: float(v) for k, v in out.items() if isinstance(v, (int, float))}
    if hasattr(belief, "estimates") and callable(belief.estimates):
        out = belief.estimates()
        return {k: float(v) for k, v in out.items() if isinstance(v, (int, float))}
    return {}


def run_result_to_finalize_record(
    result: RunResult,
    locator_result: dict[str, float],
    repeat_id: int,
    x_min: float,
    x_max: float,
) -> dict[str, Any]:
    """Flatten a RunResult into a single dict suitable for the finalize DataFrame.

    Includes peak estimates, parameter uncertainties, entropy, convergence flag,
    and measurement count.
    """
    record: dict[str, Any] = {"repeat_id": repeat_id}

    belief_estimates = result.final_estimates()
    record.update(extract_peak_estimates(belief_estimates, locator_result, x_min, x_max))

    for key, value in belief_estimates.items():
        record.setdefault(key, value)

    if result.snapshots:
        last_belief = result.snapshots[-1].belief
        # Same marginal stds as :meth:`AbstractMarginalDistribution.uncertainty` (and as the
        # parameter convergence plot): physical units for unit-cube beliefs, grid/particle
        # empirical std otherwise — no extra domain scaling.
        for param_name, uncert in last_belief.uncertainty().items():
            record[f"uncert_{param_name}"] = uncert

        record["entropy"] = last_belief.entropy()
        record["converged"] = last_belief.converged(threshold=0.01)
        record["measurements"] = len(result.snapshots)

    return record
