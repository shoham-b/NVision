from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class OnePeakGridLocator(Locator):
    """Grid scan locator optimized for single peak detection.

    Performs a uniform grid scan across the domain to find one peak.
    """

    n_points: int = 21

    def _grid(self, lo: float, hi: float) -> list[float]:
        if self.n_points <= 1:
            return [0.5 * (lo + hi)]
        return np.linspace(lo, hi, self.n_points).tolist()

    def propose_next(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        grid = self._grid(scan.x_min, scan.x_max)
        if not grid:
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        max_idx = len(grid) - 1

        taken_counts = (
            history.filter(pl.col("repeat_id").is_in(active.get_column("repeat_id")))
            .group_by("repeat_id")
            .agg(pl.col("x").n_unique().alias("n_taken"))
        )

        proposals = (
            active.select("repeat_id")
            .join(taken_counts, on="repeat_id", how="left")
            .with_columns(pl.col("n_taken").fill_null(0).cast(pl.Int64))
            .with_columns(
                pl.when(pl.col("n_taken") > max_idx)
                .then(max_idx)
                .otherwise(pl.col("n_taken"))
                .map_elements(lambda idx: float(grid[int(idx)]), return_dtype=pl.Float64)
                .alias("x")
            )
            .select("repeat_id", "x")
        )

        return proposals

    def should_stop(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        counts = (
            history.group_by("repeat_id")
            .agg(pl.col("x").n_unique().alias("n_unique"))
            .with_columns(pl.col("n_unique").cast(pl.Int64))
        )

        result = (
            repeats.select("repeat_id")
            .join(counts, on="repeat_id", how="left")
            .with_columns(pl.col("n_unique").fill_null(0))
            .with_columns((pl.col("n_unique") >= self.n_points).alias("stop"))
            .select("repeat_id", "stop")
        )

        return result

    def finalize(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        base = repeats.select("repeat_id")

        if history.is_empty():
            return base.with_columns(
                pl.lit(1.0).alias("n_peaks"),
                pl.lit(math.nan).alias("x1_hat"),
                pl.lit(math.inf).alias("uncert"),
                pl.lit(math.inf).alias("uncert_pos"),
                pl.lit(0).alias("measurements"),
            )

        grouped = history.group_by("repeat_id").agg(
            pl.col("x").sort_by(pl.col("signal_values"), descending=True).first().alias("x1_hat"),
            pl.col("x").unique().sort().alias("grid_points"),
            pl.len().alias("measurements"),
        )

        def _estimate_uncert(row: dict) -> float:
            xs = row.get("grid_points") or []
            x_hat = row.get("x1_hat")
            if x_hat is None or (isinstance(x_hat, float) and math.isnan(x_hat)):
                return math.inf
            xs_sorted = sorted(float(x) for x in xs)
            if len(xs_sorted) < 2:
                if self.n_points <= 1:
                    return (scan.x_max - scan.x_min) / 2
                span = scan.x_max - scan.x_min
                return span / max(2 * (self.n_points - 1), 2)
            idx = min(range(len(xs_sorted)), key=lambda i: abs(xs_sorted[i] - x_hat))
            left = xs_sorted[idx - 1] if idx > 0 else xs_sorted[idx]
            right = xs_sorted[idx + 1] if idx + 1 < len(xs_sorted) else xs_sorted[idx]
            if right != left:
                return abs(right - left) / 2
            return abs(xs_sorted[1] - xs_sorted[0]) / 2

        enriched = grouped.with_columns(
            pl.struct(["grid_points", "x1_hat"])
            .map_elements(lambda s: _estimate_uncert(s), return_dtype=pl.Float64)
            .alias("uncert")
        ).with_columns(pl.col("uncert").abs().alias("uncert"))

        result = (
            base.join(enriched.drop("grid_points"), on="repeat_id", how="left")
            .with_columns(pl.col("x1_hat").cast(pl.Float64))
            .with_columns(pl.col("uncert").fill_null(math.inf))
            .with_columns(pl.col("measurements").fill_null(0).cast(pl.Int64))
            .with_columns(pl.lit(1.0).alias("n_peaks"))
            .with_columns(pl.col("uncert").alias("uncert_pos"))
        )

        return result.select(
            "repeat_id", "n_peaks", "x1_hat", "uncert", "uncert_pos", "measurements"
        )
