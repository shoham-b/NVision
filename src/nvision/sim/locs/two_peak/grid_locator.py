from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class TwoPeakGridLocator(Locator):
    """Grid scan locator optimized for two peak detection.

    Performs a uniform grid scan and identifies the two strongest peaks.
    """

    coarse_points: int = 25
    min_separation_frac: float = 0.05

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        lo, hi = scan.x_min, scan.x_max
        grid = np.linspace(lo, hi, self.coarse_points).tolist()

        if history.is_empty():
            return grid[0]

        taken = {round(x, 12) for x in history["x"]}
        for g in grid:
            if round(g, 12) not in taken:
                return g
        return grid[-1]

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        if history.is_empty():
            return False
        return history["x"].n_unique() >= self.coarse_points

    def finalize(self, history: pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        if history.is_empty():
            return {
                "n_peaks": 2.0,
                "x1_hat": math.nan,
                "x2_hat": math.nan,
                "uncert": math.inf,
                "uncert_pos": math.inf,
                "uncert_sep": math.inf,
            }

        sorted_history = history.sort("signal_values", descending=True)
        w = (scan.x_max - scan.x_min) * self.min_separation_frac

        picks: list[float] = []
        for x in sorted_history["x"]:
            if not picks or all(abs(x - p) > w for p in picks):
                picks.append(x)
            if len(picks) == 2:
                break

        picks.sort()
        if len(picks) == 1:
            return {
                "n_peaks": 2.0,
                "x1_hat": float(picks[0]),
                "x2_hat": float(picks[0]),
                "uncert": 0.0,
                "uncert_pos": 0.0,
                "uncert_sep": 0.0,
            }

        dist = abs(picks[1] - picks[0])
        return {
            "n_peaks": 2.0,
            "x1_hat": float(picks[0]),
            "x2_hat": float(picks[1]),
            "uncert": float(0.5 * dist),
            "uncert_pos": float(0.5 * dist),
            "uncert_sep": float(0.5 * dist),
        }
