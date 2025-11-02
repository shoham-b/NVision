from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class NVCenterSweepLocator(Locator):
    """Sweep locator optimized for NV center signals with potential multiple peaks.

    Handles both single peak (f_b=0) and triple peak (f_b!=0) scenarios.
    Uses coarse scan followed by adaptive refinement around detected peaks.
    """

    coarse_points: int = 30
    refine_points: int = 10
    min_separation_frac: float = 0.03  # Smaller for closer NV center peaks

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
        # Identify up to 3 peaks
        sorted_history = history.sort("signal_values", descending=True)
        w = (hi - lo) * self.min_separation_frac

        peaks: list[float] = []
        for x in sorted_history["x"]:
            if not peaks or all(abs(x - p) >= w for p in peaks):
                peaks.append(x)
            if len(peaks) >= 3:  # NV centers have at most 3 peaks
                break

        # Refine around all detected peaks
        refine_candidates: list[float] = []
        width = w * 0.5
        for peak in peaks:
            refine_candidates.extend([peak - width, peak, peak + width])

        taken = set(history["x"])
        for candidate in refine_candidates:
            if lo <= candidate <= hi and candidate not in taken:
                return candidate

        return peaks[0] if peaks else 0.5 * (lo + hi)

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        return len(history) >= (self.coarse_points + self.refine_points)

    def finalize(self, history: pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        if history.is_empty():
            return {
                "n_peaks": 1.0,
                "x1_hat": math.nan,
                "x2_hat": math.nan,
                "x3_hat": math.nan,
                "uncert": math.inf,
                "uncert_pos": math.inf,
            }

        sorted_history = history.sort("signal_values", descending=True)
        w = (scan.x_max - scan.x_min) * self.min_separation_frac

        picks: list[float] = []
        for x in sorted_history["x"]:
            if not picks or all(abs(x - p) > w for p in picks):
                picks.append(x)
            if len(picks) >= 3:
                break

        picks.sort()

        result: dict[str, float] = {
            "n_peaks": float(len(picks)),
            "x1_hat": float(picks[0]) if len(picks) >= 1 else math.nan,
            "x2_hat": float(picks[1]) if len(picks) >= 2 else math.nan,
            "x3_hat": float(picks[2]) if len(picks) >= 3 else math.nan,
            "uncert": float(w),
            "uncert_pos": float(w),
        }

        return result
