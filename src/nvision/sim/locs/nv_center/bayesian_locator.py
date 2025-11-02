from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class NVCenterBayesianLocator(Locator):
    """Bayesian locator optimized for NV center signals.

    Uses a simplified Bayesian approach with uniform prior to adaptively
    sample the domain and locate NV center peaks.
    """

    max_steps: int = 40
    coarse_points: int = 15
    min_separation_frac: float = 0.03

    _sampled_coarse: bool = field(default=False, init=False, repr=False)

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        lo, hi = scan.x_min, scan.x_max

        if coarse := self._coarse_proposal(history, lo, hi):
            return coarse

        if history.is_empty():
            return 0.5 * (lo + hi)

        if undersampled := self._undersampled_bin_center(history, lo, hi):
            return undersampled

        return self._fallback_peak_sample(history, lo, hi)

    def _coarse_proposal(
        self,
        history: pl.DataFrame,
        lo: float,
        hi: float,
    ) -> float | None:
        if self._sampled_coarse:
            return None

        if not history.is_empty() and len(history) >= self.coarse_points:
            return None

        coarse_grid = np.linspace(lo, hi, self.coarse_points).tolist()
        if history.is_empty():
            return coarse_grid[0]

        taken = {round(x, 12) for x in history["x"]}
        for candidate in coarse_grid:
            if round(candidate, 12) not in taken:
                return candidate

        self._sampled_coarse = True
        return None

    def _undersampled_bin_center(
        self,
        history: pl.DataFrame,
        lo: float,
        hi: float,
    ) -> float | None:
        n_bins = 20
        bin_width = (hi - lo) / n_bins

        bin_counts = [0] * n_bins
        for x in history["x"]:
            bin_idx = min(int((x - lo) / bin_width), n_bins - 1)
            bin_counts[bin_idx] += 1

        min_count = min(bin_counts)
        undersampled_bins = [i for i, count in enumerate(bin_counts) if count == min_count]
        if not undersampled_bins:
            return None

        target_bin = undersampled_bins[len(undersampled_bins) // 2]
        proposed_x = lo + (target_bin + 0.5) * bin_width

        taken = set(history["x"])
        if proposed_x not in taken:
            return proposed_x
        return None

    def _fallback_peak_sample(
        self,
        history: pl.DataFrame,
        lo: float,
        hi: float,
    ) -> float:
        best = history.sort("signal_values", descending=True).row(0, named=True)
        best_x = float(best["x"])
        width = (hi - lo) * 0.05

        candidates = [best_x - width, best_x + width]
        taken = set(history["x"])
        for candidate in candidates:
            if lo <= candidate <= hi and candidate not in taken:
                return candidate

        return best_x

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        return len(history) >= self.max_steps

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
