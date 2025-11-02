from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class OnePeakGoldenLocator(Locator):
    """Golden section search locator optimized for single peak detection.

    Uses the golden-section search algorithm to efficiently find a single peak maximum.
    """

    max_evals: int = 25
    samples_per_point: int = 3
    _golden_ratio: float = (5**0.5 - 1) / 2

    _lower_bound: float | None = field(default=None, init=False, repr=False)
    _upper_bound: float | None = field(default=None, init=False, repr=False)
    _inner_point_c: float | None = field(default=None, init=False, repr=False)
    _inner_point_d: float | None = field(default=None, init=False, repr=False)

    def _get_averaged_history(self, history: pl.DataFrame) -> dict[float, float]:
        """Averages intensities for points that were sampled multiple times."""
        if history.is_empty():
            return {}
        averaged_df = history.group_by("x").agg(pl.mean("signal_values"))
        return dict(zip(averaged_df["x"], averaged_df["signal_values"], strict=False))

    def propose_next(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        """Proposes the next point to sample using the golden-section search logic."""
        if history.is_empty():
            self._lower_bound, self._upper_bound = scan.x_min, scan.x_max
            self._inner_point_c = self._upper_bound - self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            self._inner_point_d = self._lower_bound + self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            return self._inner_point_c

        point_counts_df = history.group_by("x").agg(pl.len().alias("len"))
        point_counts = dict(zip(point_counts_df["x"], point_counts_df["len"], strict=False))

        if (
            self._inner_point_c
            and point_counts.get(self._inner_point_c, 0) < self.samples_per_point
        ):
            return self._inner_point_c
        if (
            self._inner_point_d
            and point_counts.get(self._inner_point_d, 0) < self.samples_per_point
        ):
            return self._inner_point_d

        averaged_history = self._get_averaged_history(history)

        if len(averaged_history) == 1:
            self._lower_bound, self._upper_bound = scan.x_min, scan.x_max
            self._inner_point_c = next(iter(averaged_history.keys()))
            self._inner_point_d = self._lower_bound + self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            return self._inner_point_d

        f_at_c = averaged_history.get(self._inner_point_c)
        f_at_d = averaged_history.get(self._inner_point_d)

        if f_at_c is None or f_at_d is None:
            return self._inner_point_c if f_at_c is None else self._inner_point_d

        if f_at_c > f_at_d:
            self._upper_bound = self._inner_point_d
            self._inner_point_d = self._inner_point_c
            self._inner_point_c = self._upper_bound - self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            return self._inner_point_c
        else:
            self._lower_bound = self._inner_point_c
            self._inner_point_c = self._inner_point_d
            self._inner_point_d = self._lower_bound + self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            return self._inner_point_d

    def should_stop(self, history: pl.DataFrame, scan: ScanBatch) -> bool:
        """Stops after a fixed number of evaluations."""
        return len(history) >= self.max_evals

    def finalize(self, history: pl.DataFrame, scan: ScanBatch) -> dict[str, float]:
        """Returns the point with the highest observed intensity."""
        if history.is_empty():
            return {"n_peaks": 0.0, "x1_hat": 0.0, "uncert": float("inf")}

        best_obs = history.sort("signal_values", descending=True).row(0, named=True)
        return {"n_peaks": 1.0, "x1_hat": best_obs["x"], "uncert": 0.0}
