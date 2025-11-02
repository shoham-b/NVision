from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs import Locator, ScanBatch


@dataclass
class SweepLocator(Locator):
    """A locator that mimics a common two-stage ODMR measurement strategy: a sweep.

    This strategy consists of two phases:
    1.  **Coarse Scan**: A fixed number of points (`coarse_points`) are measured
        evenly across the entire domain to get a rough overview of the spectrum.
        This is effectively a frequency sweep.
    2.  **Refinement**: After the coarse scan, the locator identifies the most
        promising region (around the point with the highest intensity) and
        proposes additional points to refine the measurement in that area.

    This approach balances initial exploration with subsequent exploitation.
    """

    coarse_points: int = 20
    refine_points: int = 10
    min_separation_frac: float = 0.05
    uncertainty_threshold: float = 0.1

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        if history.is_empty():
            return scan.x_min

        # --- 1. Coarse Scan Phase ---
        if len(history) < self.coarse_points:
            coarse_candidates = np.linspace(scan.x_min, scan.x_max, self.coarse_points)
            candidates_df = pl.DataFrame({"candidate": coarse_candidates})

            untaken_candidates = candidates_df.join(
                history.select(pl.col("x").round(12)),
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
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}
        best = history.sort("signal_values", descending=True).row(0, named=True)
        return {"n_peaks": 1.0, "x1": best["x"], "uncert": 0.0}  # Placeholder for uncert
