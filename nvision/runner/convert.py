"""Result conversion utilities — RunResult → Polars DataFrames.

Pure functions with no side-effects; used by the task executor.
"""

from __future__ import annotations

from typing import Any

import polars as pl

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
        # Normalized x is in [0, 1]; physical x for NV centers is ~2.72e9-3.02e9 (Hz)
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

    return pl.from_dicts(rows, infer_schema_length=None)


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
        if not isinstance(value, int | float):
            continue
        key_lc = key.lower()
        is_position = (
            key_lc in ("x", "x1", "x2", "peak_x", "x_hat", "x1_hat", "x2_hat")
            or key_lc.endswith("_x")
            or key_lc.endswith("_x1")
            or key_lc.endswith("_x2")
            or "pos" in key_lc
            or "freq" in key_lc
        )
        if is_position:
            estimates[key] = denormalize_x(value, x_min, x_max) if 0 <= value <= 1 else value
        else:
            estimates[key] = value

    # Seed peak_x/x1_hat from the locator's own frequency fit when present;
    # the belief value is only a fallback (it may still be the prior for
    # locators that don't drive their belief, e.g. sweep locators).
    freq_phys = estimates.get("frequency", belief_estimates.get("frequency"))
    if freq_phys is not None:
        estimates.setdefault("peak_x", freq_phys)
        estimates.setdefault("x1_hat", freq_phys)

    if "split" in belief_estimates:
        estimates["split"] = belief_estimates["split"]

    return estimates


def belief_mode_estimates(belief: object) -> dict[str, float]:
    """Most likely parameters in physical units (for plotting the locator's best guess).

    Delegates to the belief's own ``mode_estimates()`` — every concrete belief
    type is required to implement it as one internally consistent joint state
    (see ``AbstractMarginalDistribution.mode_estimates``), so there is no
    fallback here to a different quantity (e.g. the posterior mean).
    """
    out = belief.mode_estimates()
    return {k: float(v) for k, v in out.items() if isinstance(v, int | float)}


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
    if result.fit_mode_estimates:
        # Sweep locators (GenericSweepLocator) batch-update their belief
        # without resampling during the sweep, so it can collapse — the same
        # reason the plotted "locator most likely signal" curve prefers
        # fit_mode_estimates over the belief mode (see
        # nvision/runner/executor.py). Every final_est_<param> column below
        # was silently reading the collapsed belief instead of the actual
        # least-squares fit for every parameter except frequency (which
        # extract_peak_estimates seeds from locator_result); apply the same
        # preference here.
        belief_estimates = {**belief_estimates, **result.fit_mode_estimates}
    record.update(extract_peak_estimates(belief_estimates, locator_result, x_min, x_max))

    for key, value in belief_estimates.items():
        record.setdefault(key, value)

    if result.snapshots:
        last_belief = result.snapshots[-1].belief
        # Same marginal stds as :meth:`AbstractMarginalDistribution.uncertainty` (and as the
        # parameter convergence plot): physical units for unit-cube beliefs, grid/particle
        # empirical std otherwise — no extra domain scaling.
        for param_name, uncert in last_belief.reported_uncertainty().items():
            record[f"uncert_{param_name}"] = uncert

        record["entropy"] = last_belief.entropy()
        record["converged"] = last_belief.converged(threshold=0.01)
        record["measurements"] = len(result.snapshots)

    return record
