from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class GridScanLocator(Locator):
    n_points: int = 21

    def _grid(self, lo: float, hi: float) -> list[float]:
        if self.n_points <= 1:
            return [0.5 * (lo + hi)]
        return np.linspace(lo, hi, self.n_points).tolist()

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        grid = self._grid(scan.x_min, scan.x_max)
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
        return history["x"].n_unique() >= self.n_points

    def finalize(self, history: pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        if history.is_empty():
            return {
                "n_peaks": 1.0,
                "x1_hat": math.nan,
                "x2_hat": math.nan,
                "uncert": math.inf,
                "uncert_pos": math.inf,
                "uncert_sep": math.nan,
            }

        best = history.sort("signal_values", descending=True).row(0, named=True)
        best_x = best["x"]

        xs = sorted(history["x"].unique().to_list())
        if len(xs) < 2:
            dx = 0.5
        else:
            idx = xs.index(min(xs, key=lambda xv: abs(xv - best_x)))
            left = xs[idx - 1] if idx > 0 else xs[idx]
            right = xs[idx + 1] if idx + 1 < len(xs) else xs[idx]
            if right != left:
                dx = 0.5 * (right - left)
            elif len(xs) > 1:
                dx = (xs[1] - xs[0]) / 2
            else:
                dx = 0.5
        return {
            "n_peaks": 1.0,
            "x1_hat": float(best_x),
            "x2_hat": float("nan"),
            "uncert": float(abs(dx)),
            "uncert_pos": float(abs(dx)),
            "uncert_sep": float("nan"),
        }
