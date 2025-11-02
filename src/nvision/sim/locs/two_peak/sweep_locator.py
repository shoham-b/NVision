from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class TwoPeakSweepLocator(Locator):
    """Sweep locator optimized for two peak detection.

    Two-phase strategy: coarse scan followed by refinement around the two best peaks.
    """

    coarse_points: int = 25
    refine_points: int = 10
    min_separation_frac: float = 0.05

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        lo, hi = scan.x_min, scan.x_max

        # --- 1. Coarse Scan Phase ---
        if history.is_empty() or len(history) < self.coarse_points:
            coarse_candidates = np.linspace(lo, hi, self.coarse_points)
            candidates_df = pl.DataFrame({"candidate": coarse_candidates})

            history_points = (
                history.select(pl.col("x").round(12))
                if not history.is_empty()
                else pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
            )

            untaken_candidates = candidates_df.join(
                history_points,
                left_on=pl.col("candidate").round(12),
                right_on=pl.col("x"),
                how="anti",
            )

            if not untaken_candidates.is_empty():
                return untaken_candidates["candidate"][0]

        if history.is_empty():
            return 0.5 * (lo + hi)

        # --- 2. Refinement Phase ---
        # Identify two best peaks
        sorted_history = history.sort("signal_values", descending=True)
        x1 = sorted_history["x"][0]
        w = (hi - lo) * self.min_separation_frac

        second_peak_candidates = sorted_history.filter(pl.col("x").is_not_nan())
        second_peak_candidates = second_peak_candidates.filter((pl.col("x") - x1).abs() >= w)

        x2 = second_peak_candidates["x"][0] if not second_peak_candidates.is_empty() else x1

        # Refine around both peaks
        width = w * 0.5
        refine_candidates = [
            x1 - width,
            x1,
            x1 + width,
            x2 - width,
            x2,
            x2 + width,
        ]

        taken = set(history["x"])
        for candidate in refine_candidates:
            if lo <= candidate <= hi and candidate not in taken:
                return candidate

        return x1

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        return len(history) >= (self.coarse_points + self.refine_points)

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
