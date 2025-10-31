from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.models.obs import Obs


@dataclass
class ODMRLocator:
    """A locator that mimics a common two-stage ODMR measurement strategy.

    This strategy consists of two phases:
    1.  **Coarse Scan**: A fixed number of points (`coarse_points`) are measured
        evenly across the entire domain to get a rough overview of the spectrum.
    2.  **Refinement**: After the coarse scan, the locator identifies the most
        promising region (around the point with the highest intensity) and
        proposes additional points to refine the measurement in that area.

    This approach balances initial exploration with subsequent exploitation.
    """

    coarse_points: int = 20
    refine_points: int = 10
    min_separation_frac: float = 0.05
    uncertainty_threshold: float = 0.1

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        domain_lo, domain_hi = domain

        if not history:
            # If there's no history, start with the first coarse point.
            return domain_lo

        # Use polars for efficient processing of history and candidates.
        history_df = pl.DataFrame([{"x": o.x, "intensity": o.intensity} for o in history])

        # --- 1. Coarse Scan Phase ---
        if len(history) < self.coarse_points:
            # Generate all potential coarse scan points.
            coarse_candidates = np.linspace(domain_lo, domain_hi, self.coarse_points)
            candidates_df = pl.DataFrame({"candidate": coarse_candidates})

            # Find the first candidate that is not in the history.
            # A left anti-join returns candidates that have no match in history.
            # We round to handle potential floating point inaccuracies.
            untaken_candidates = candidates_df.join(
                history_df.select(pl.col("x").round(12)),
                left_on=pl.col("candidate").round(12),
                right_on=pl.col("x"),
                how="anti",
            )

            if not untaken_candidates.is_empty():
                return untaken_candidates["candidate"][0]

        # --- 2. Refinement Phase ---
        # Find the observation with the highest intensity.
        best_obs = history_df.row(by="intensity", descending=True, named=True)
        best_x = best_obs["x"]

        # Propose new points around the best observation.
        width = (domain_hi - domain_lo) * 0.1
        refine_candidates = [best_x - width / 2, best_x, best_x + width / 2]
        return next((p for p in refine_candidates if p not in history_df["x"]), best_x)

    def should_stop(self, history: Sequence[Obs]) -> bool:
        if len(history) >= (self.coarse_points + self.refine_points):
            recent = history[-5:] if len(history) >= 5 else history
            avg_u = sum(o.uncertainty for o in recent) / max(1, len(recent))
            return avg_u < self.uncertainty_threshold
        return False

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
        if not history:
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}
        best = max(history, key=lambda o: o.intensity)
        return {"n_peaks": 1.0, "x1": best.x, "uncert": best.uncertainty}
