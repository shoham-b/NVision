from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class TwoPeakGoldenLocator(Locator):
    """Greedy locator for two peak detection using golden-section-inspired refinement.

    Performs coarse scan then greedily refines the two strongest peaks.
    """

    coarse_points: int = 25
    refine_points: int = 5
    min_separation_frac: float = 0.05

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        lo, hi = scan.x_min, scan.x_max
        if history.is_empty() or len(history) < self.coarse_points:
            grid = np.linspace(lo, hi, self.coarse_points).tolist()
            if history.is_empty():
                return grid[0]
            taken = {round(x, 12) for x in history["x"]}
            for g in grid:
                if round(g, 12) not in taken:
                    return g

        if history.is_empty():
            return 0.5 * (lo + hi)

        # Identify two best peaks
        sorted_history = history.sort("signal_values", descending=True)
        x1 = sorted_history["x"][0]
        w = (hi - lo) * self.min_separation_frac

        second_peak_candidates = sorted_history.filter(pl.col("x").is_not_nan())
        second_peak_candidates = second_peak_candidates.filter((pl.col("x") - x1).abs() >= w)

        x2 = second_peak_candidates["x"][0] if not second_peak_candidates.is_empty() else x1

        # Greedily refine the less-sampled peak
        n1 = history.filter((pl.col("x") - x1).abs() <= w).height
        n2 = history.filter((pl.col("x") - x2).abs() <= w).height

        target = x2 if n2 < n1 else x1

        if target <= x1 and target <= x2:
            return max(lo, target - 0.5 * w)
        if target >= x1 and target >= x2:
            return min(hi, target + 0.5 * w)
        return target

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        return len(history) >= (self.coarse_points + 2 * self.refine_points)

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
        picks: list[float] = []
        for x in sorted_history["x"]:
            if not picks or all(abs(x - p) > 1e-9 for p in picks):
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
