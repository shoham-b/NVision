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

    def propose_next(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        coarse_grid = np.linspace(scan.x_min, scan.x_max, self.coarse_points).tolist()

        counts = history.group_by("repeat_id").agg(pl.len().alias("n_measurements"))

        active_with_counts = active.join(counts, on="repeat_id", how="left").with_columns(
            pl.col("n_measurements").fill_null(0).cast(pl.Int64)
        )

        # Coarse phase: propose next grid point
        coarse_phase = active_with_counts.filter(pl.col("n_measurements") < self.coarse_points)
        coarse_proposals = []
        for row in coarse_phase.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            n_taken = row["n_measurements"]
            x_next = coarse_grid[n_taken] if n_taken < len(coarse_grid) else coarse_grid[-1]
            coarse_proposals.append({"repeat_id": repeat_id, "x": x_next})

        # Refine phase: propose around best point
        refine_phase = active_with_counts.filter(pl.col("n_measurements") >= self.coarse_points)
        refine_proposals = []
        if not refine_phase.is_empty():
            best_per_repeat = (
                history.filter(pl.col("repeat_id").is_in(refine_phase.get_column("repeat_id")))
                .sort(["repeat_id", "signal_values"], descending=[False, True])
                .group_by("repeat_id")
                .agg(pl.col("x").first().alias("best_x"))
            )
            refine_with_best = refine_phase.join(best_per_repeat, on="repeat_id", how="left")
            width = (scan.x_max - scan.x_min) * 0.1
            for row in refine_with_best.iter_rows(named=True):
                repeat_id = row["repeat_id"]
                best_x = row.get("best_x", scan.x_min)
                n_taken = row["n_measurements"]
                offset_idx = n_taken - self.coarse_points
                if offset_idx == 0:
                    x_next = best_x - width / 2
                elif offset_idx == 1:
                    x_next = best_x
                elif offset_idx == 2:
                    x_next = best_x + width / 2
                else:
                    x_next = best_x
                refine_proposals.append({"repeat_id": repeat_id, "x": x_next})

        all_proposals = coarse_proposals + refine_proposals
        if not all_proposals:
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})
        return pl.DataFrame(all_proposals)

    def should_stop(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        counts = history.group_by("repeat_id").agg(pl.len().alias("n_measurements"))
        result = (
            repeats.select("repeat_id")
            .join(counts, on="repeat_id", how="left")
            .with_columns(pl.col("n_measurements").fill_null(0).cast(pl.Int64))
            .with_columns(
                (pl.col("n_measurements") >= (self.coarse_points + self.refine_points)).alias(
                    "stop"
                )
            )
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
                pl.lit(float("nan")).alias("x1_hat"),
                pl.lit(float("inf")).alias("uncert"),
                pl.lit(0).alias("measurements"),
            )

        best_per_repeat = (
            history.sort(["repeat_id", "signal_values"], descending=[False, True])
            .group_by("repeat_id")
            .agg(
                pl.col("x").first().alias("x1_hat"),
                pl.len().alias("measurements"),
            )
        )

        result = (
            base.join(best_per_repeat, on="repeat_id", how="left")
            .with_columns(pl.col("x1_hat").cast(pl.Float64))
            .with_columns(pl.col("x1_hat").fill_null(float("nan")))
            .with_columns(pl.col("measurements").fill_null(0).cast(pl.Int64))
            .with_columns(pl.lit(1.0).alias("n_peaks"))
            .with_columns(pl.lit(0.0).alias("uncert"))
        )
        return result.select("repeat_id", "n_peaks", "x1_hat", "uncert", "measurements")
