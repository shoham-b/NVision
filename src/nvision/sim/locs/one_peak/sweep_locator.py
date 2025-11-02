from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class OnePeakSweepLocator(Locator):
    """Sweep locator optimized for single peak detection.

    Two-phase strategy: coarse scan followed by refinement around the best point.
    """

    coarse_points: int = 20
    refine_points: int = 10

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        if history.is_empty():
            return scan.x_min

        # --- 1. Coarse Scan Phase ---
        if len(history) < self.coarse_points:
            coarse_candidates = np.linspace(scan.x_min, scan.x_max, self.coarse_points)
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

        # --- 2. Refinement Phase ---
        best_obs = (
            history.sort("signal_values", descending=True).row(0, named=True)
            if not history.is_empty()
            else {"x": scan.x_min}
        )
        best_x = best_obs["x"]

        width = (scan.x_max - scan.x_min) * 0.1
        refine_candidates = [best_x - width / 2, best_x, best_x + width / 2]
        return next((p for p in refine_candidates if p not in history["x"]), best_x)

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        return len(history) >= (self.coarse_points + self.refine_points)

    def finalize(self, history: pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        if history.is_empty():
            return {"n_peaks": 0.0, "x1_hat": 0.0, "uncert": float("inf")}
        best = history.sort("signal_values", descending=True).row(0, named=True)
        return {"n_peaks": 1.0, "x1_hat": best["x"], "uncert": 0.0}
