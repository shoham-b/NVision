from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class TwoPeakGridLocator(Locator):
    """Grid scan locator optimized for two peak detection.

    Performs a uniform grid scan and identifies the two strongest peaks.
    """

    coarse_points: int = 25
    min_separation_frac: float = 0.05

    def propose_next(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        grid = np.linspace(scan.x_min, scan.x_max, self.coarse_points).tolist()
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
        counts = history.group_by("repeat_id").agg(pl.col("x").n_unique().alias("n_unique"))
        result = (
            repeats.select("repeat_id")
            .join(counts, on="repeat_id", how="left")
            .with_columns(pl.col("n_unique").fill_null(0).cast(pl.Int64))
            .with_columns((pl.col("n_unique") >= self.coarse_points).alias("stop"))
            .select("repeat_id", "stop")
        )
        return result

    def finalize(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        base = repeats.select("repeat_id")
        if history.is_empty():
            return base.with_columns(
                pl.lit(2.0).alias("n_peaks"),
                pl.lit(math.nan).alias("x1_hat"),
                pl.lit(math.nan).alias("x2_hat"),
                pl.lit(math.inf).alias("uncert"),
                pl.lit(math.inf).alias("uncert_pos"),
                pl.lit(math.inf).alias("uncert_sep"),
                pl.lit(0).alias("measurements"),
            )

        w = (scan.x_max - scan.x_min) * self.min_separation_frac

        def _find_two_peaks(repeat_id: int) -> dict:
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
            if repeat_history.is_empty():
                return {
                    "repeat_id": repeat_id,
                    "x1_hat": math.nan,
                    "x2_hat": math.nan,
                    "uncert": math.inf,
                    "uncert_pos": math.inf,
                    "uncert_sep": math.inf,
                    "measurements": 0,
                }

            sorted_history = repeat_history.sort("signal_values", descending=True)
            picks: list[float] = []
            for x in sorted_history["x"]:
                if not picks or all(abs(x - p) > w for p in picks):
                    picks.append(x)
                if len(picks) == 2:
                    break

            picks.sort()
            if len(picks) == 0:
                return {
                    "repeat_id": repeat_id,
                    "x1_hat": math.nan,
                    "x2_hat": math.nan,
                    "uncert": math.inf,
                    "uncert_pos": math.inf,
                    "uncert_sep": math.inf,
                    "measurements": repeat_history.height,
                }
            if len(picks) == 1:
                return {
                    "repeat_id": repeat_id,
                    "x1_hat": float(picks[0]),
                    "x2_hat": float(picks[0]),
                    "uncert": 0.0,
                    "uncert_pos": 0.0,
                    "uncert_sep": 0.0,
                    "measurements": repeat_history.height,
                }

            dist = abs(picks[1] - picks[0])
            return {
                "repeat_id": repeat_id,
                "x1_hat": float(picks[0]),
                "x2_hat": float(picks[1]),
                "uncert": float(0.5 * dist),
                "uncert_pos": float(0.5 * dist),
                "uncert_sep": float(0.5 * dist),
                "measurements": repeat_history.height,
            }

        results = [_find_two_peaks(rid) for rid in base.get_column("repeat_id")]
        results_df = pl.DataFrame(results)

        final = base.join(results_df, on="repeat_id", how="left").with_columns(
            pl.lit(2.0).alias("n_peaks")
        )
        return final.select(
            "repeat_id",
            "n_peaks",
            "x1_hat",
            "x2_hat",
            "uncert",
            "uncert_pos",
            "uncert_sep",
            "measurements",
        )
