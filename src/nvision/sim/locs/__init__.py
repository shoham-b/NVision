from __future__ import annotations

import random

import numpy as np
import polars as pl

from nvision.sim.core import DataBatch, OverFrequencyNoise, OverProbeNoise

from .base import Locator, ScanBatch
from .nv_center import (
    NVCenterSequentialBayesianLocator,
    NVCenterSweepLocator,
)
from .one_peak import OnePeakGoldenLocator, OnePeakGridLocator, OnePeakSweepLocator
from .two_peak import TwoPeakGoldenLocator, TwoPeakGridLocator, TwoPeakSweepLocator


def run_locator(
    locator: Locator,
    scan: ScanBatch,
    over_frequency_noise: OverFrequencyNoise | None = None,
    over_probe_noise: OverProbeNoise | None = None,
    max_steps: int = 100,
    seed: int = 0,
) -> pl.DataFrame:
    """Orchestrates a peak-finding simulation using a given locator and noise model."""
    rng = random.Random(seed)

    # If over-frequency noise is present, it transforms the underlying signal.
    if over_frequency_noise is not None:
        # Create a dense representation of the signal to apply the noise.
        xs = [scan.x_min + (scan.x_max - scan.x_min) * i / 1000 for i in range(1001)]
        ys = [scan.signal(x) for x in xs]
        original_batch = DataBatch.from_arrays(x=xs, signal_values=ys, meta=scan.meta)
        noisy_batch = over_frequency_noise.apply(original_batch, rng)

        # Create a new, noisy signal function for the locator to use.
        def noisy_signal(x_val: float) -> float:
            return float(np.interp(x_val, noisy_batch.x, noisy_batch.signal_values))

        scan = ScanBatch(
            x_min=scan.x_min,
            x_max=scan.x_max,
            signal=noisy_signal,
            meta=scan.meta,
            truth_positions=scan.truth_positions,
        )

    history: list[dict[str, float]] = []
    for _ in range(max_steps):
        current_history_df = pl.DataFrame(history) if history else pl.DataFrame()

        if locator.should_stop(current_history_df, scan):
            break

        x_next = locator.propose_next(current_history_df, scan)
        y_ideal = scan.signal(x_next)

        # Over-probe noise is applied at each measurement step.
        y_measured = (
            over_probe_noise.apply(y_ideal, rng) if over_probe_noise is not None else y_ideal
        )

        history.append({"x": x_next, "signal_values": y_measured})

    return pl.DataFrame(history)


__all__ = [
    "Locator",
    "run_locator",
    "ScanBatch",
    # Category-specific locators
    "OnePeakGridLocator",
    "OnePeakGoldenLocator",
    "OnePeakSweepLocator",
    "TwoPeakGridLocator",
    "TwoPeakGoldenLocator",
    "TwoPeakSweepLocator",
    "NVCenterSweepLocator",
    "NVCenterSequentialBayesianLocator",
]
