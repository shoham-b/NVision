from __future__ import annotations

import logging
import random
import time

import polars as pl

from nvision.sim.core import OverFrequencyNoise, OverProbeNoise

from .base import Locator, ScanBatch
from .nv_center import (
    NVCenterSequentialBayesianLocator,
    NVCenterSweepLocator,
)
from .one_peak import OnePeakGoldenLocator, OnePeakGridLocator, OnePeakSweepLocator
from .two_peak import TwoPeakGoldenLocator, TwoPeakGridLocator, TwoPeakSweepLocator

log = logging.getLogger(__name__)


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
    """Orchestrates a peak-finding simulation using a given locator and noise model."""
    rng = random.Random(seed)

    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    try:
        for _ in range(max_steps):
            if time.perf_counter() - start_time > timeout_s:
                log.warning(f"Timeout of {timeout_s}s reached. Finalizing run.")
                break

            current_history_df = pl.DataFrame(history)
            if locator.should_stop(current_history_df, scan):
                break

            x_next = locator.propose_next(current_history_df, scan)
            y_ideal = scan.signal(x_next)

            y_measured = (
                over_probe_noise.apply(y_ideal, rng, locator)
                if over_probe_noise is not None
                else y_ideal
            )

            history.append({"x": x_next, "signal_values": y_measured})
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt received. Finalizing partial run...")

    if not history:
        return pl.DataFrame()

    return pl.DataFrame(history)


__all__ = [
    "Locator",
    "NVCenterSequentialBayesianLocator",
    "NVCenterSweepLocator",
    "OnePeakGoldenLocator",
    "OnePeakGridLocator",
    "OnePeakSweepLocator",
    "ScanBatch",
    "TwoPeakGoldenLocator",
    "TwoPeakGridLocator",
    "TwoPeakSweepLocator",
    "run_locator",
]
