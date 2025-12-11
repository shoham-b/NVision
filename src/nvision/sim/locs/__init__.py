from __future__ import annotations

import logging
import random
import time

import polars as pl

from nvision.sim.core import OverFrequencyNoise, OverProbeNoise

from .base import Locator, ScanBatch
from .nv_center import (
    AnalyticalBayesianLocator,  # noqa: F401
    NVCenterSequentialBayesianLocator,
    NVCenterSweepLocator,
    ProjectBayesianLocator,
    SimpleSequentialLocator,
)
from .one_peak import OnePeakGoldenLocator, OnePeakGridLocator, OnePeakSweepLocator
from .two_peak import TwoPeakGoldenLocator, TwoPeakGridLocator, TwoPeakSweepLocator

log = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when a locator run exceeds its time limit."""

    pass


def run_locator(
    *,
    locator: Locator,
    scan: ScanBatch,
    seed: int,
    over_frequency_noise: OverFrequencyNoise | None = None,
    over_probe_noise: OverProbeNoise | None = None,
    max_steps: int = 200,
    timeout_s: float = 300.0,
) -> pl.DataFrame:
    """Orchestrates a peak-finding simulation using a given locator and noise model.

    Args:
        locator: The locator strategy to use for finding peaks.
        scan: The scan batch containing the signal to analyze.
        seed: Random seed for reproducibility.
        over_frequency_noise: Optional noise to apply to the frequency axis.
        over_probe_noise: Optional noise to apply to the signal measurements.
        max_steps: Maximum number of measurement steps to perform.
        timeout_s: Maximum time in seconds to run before timing out.

    Returns:
        A DataFrame containing the measurement history.

    Raises:
        TimeoutError: If the operation exceeds the specified timeout.
    """
    rng = random.Random(seed)
    start_time = time.perf_counter()

    def check_timeout() -> None:
        if time.perf_counter() - start_time > timeout_s:
            raise TimeoutError(f"Locator run timed out after {timeout_s:.1f} seconds")

    # Build repeat tracking DataFrame for single repeat
    repeats_df = pl.DataFrame({"repeat_id": [0], "active": [True]})
    history_rows = []

    try:
        for step_num in range(max_steps):
            check_timeout()

            # Build history DataFrame with repeat_id
            if history_rows:
                history_df = pl.DataFrame(history_rows)
            else:
                # Initialize with basic schema, but allow it to grow
                history_df = pl.DataFrame(
                    {
                        "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                        "step": pl.Series("step", [], dtype=pl.Int64),
                        "x": pl.Series("x", [], dtype=pl.Float64),
                        "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
                    }
                )

            # Check stop condition
            stop_decisions = locator.should_stop(history_df, repeats_df, scan)
            if not stop_decisions.is_empty() and stop_decisions.get_column("stop")[0]:
                break

            # Propose next measurement
            proposals = locator.propose_next(history_df, repeats_df, scan)
            if proposals.is_empty():
                break
            x_next = float(proposals.get_column("x")[0])

            y_ideal = scan.signal(x_next)

            y_measured = over_probe_noise.apply(y_ideal, rng, locator) if over_probe_noise is not None else y_ideal

            # Capture current estimates if available
            row_data = {
                "repeat_id": 0,
                "step": step_num,
                "x": x_next,
                "signal_values": y_measured,
            }

            if hasattr(locator, "current_estimates") and isinstance(locator.current_estimates, dict):
                for key, value in locator.current_estimates.items():
                    # Prefix with est_ to avoid collisions and clearly mark as estimates
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        row_data[f"est_{key}"] = value

            history_rows.append(row_data)
    except (KeyboardInterrupt, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            log.warning("%s. Finalizing with current measurements...", str(e))
        else:
            log.warning("Keyboard interrupt received. Finalizing partial run...")

    if not history_rows:
        return pl.DataFrame()

    # Return history without repeat_id for backward compatibility
    result_df = pl.DataFrame(history_rows).drop("repeat_id")
    return result_df


__all__ = [
    "Locator",
    "NVCenterSequentialBayesianLocator",
    "NVCenterSweepLocator",
    "OnePeakGoldenLocator",
    "OnePeakGridLocator",
    "OnePeakSweepLocator",
    "ProjectBayesianLocator",
    "ScanBatch",
    "SimpleSequentialLocator",
    "TwoPeakGoldenLocator",
    "TwoPeakGridLocator",
    "TwoPeakSweepLocator",
    "run_locator",
]
