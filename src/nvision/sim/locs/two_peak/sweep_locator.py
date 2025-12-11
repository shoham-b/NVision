from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class TwoPeakSweepLocator(Locator):
    """Sweep locator optimized for two peak detection.

    Two-phase strategy: coarse scan followed by refinement around the two best peaks.
    """

    coarse_points: int = 25
    refine_points: int = 10
    min_separation_frac: float = 0.05

    def propose_next(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        lo, hi = scan.x_min, scan.x_max
        coarse_grid = np.linspace(lo, hi, self.coarse_points).tolist()
        w = (hi - lo) * self.min_separation_frac

        counts = history.group_by("repeat_id").agg(pl.len().alias("n_measurements"))
        active_with_counts = active.join(counts, on="repeat_id", how="left").with_columns(
            pl.col("n_measurements").fill_null(0).cast(pl.Int64)
        )

        # Coarse phase
        coarse_phase = active_with_counts.filter(pl.col("n_measurements") < self.coarse_points)
        coarse_proposals = []
        for row in coarse_phase.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            n_taken = row["n_measurements"]
            x_next = coarse_grid[n_taken] if n_taken < len(coarse_grid) else coarse_grid[-1]
            coarse_proposals.append({"repeat_id": repeat_id, "x": x_next})

        # Refine phase
        refine_phase = active_with_counts.filter(pl.col("n_measurements") >= self.coarse_points)
        refine_proposals = []
        for row in refine_phase.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
            if repeat_history.is_empty():
                refine_proposals.append({"repeat_id": repeat_id, "x": 0.5 * (lo + hi)})
                continue

            sorted_history = repeat_history.sort("signal_values", descending=True)
            x1 = sorted_history["x"][0]
            second_peak_candidates = sorted_history.filter((pl.col("x") - x1).abs() >= w)
            x2 = second_peak_candidates["x"][0] if not second_peak_candidates.is_empty() else x1

            width = w * 0.5
            refine_candidates = [
                x1 - width,
                x1,
                x1 + width,
                x2 - width,
                x2,
                x2 + width,
            ]
            taken = set(repeat_history["x"])
            x_next = x1
            for candidate in refine_candidates:
                if lo <= candidate <= hi and candidate not in taken:
                    x_next = candidate
                    break
            refine_proposals.append({"repeat_id": repeat_id, "x": x_next})

        all_proposals = coarse_proposals + refine_proposals
        if not all_proposals:
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})
        return pl.DataFrame(all_proposals)

    def should_stop(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        counts = history.group_by("repeat_id").agg(pl.len().alias("n_measurements"))
        result = (
            repeats.select("repeat_id")
            .join(counts, on="repeat_id", how="left")
            .with_columns(pl.col("n_measurements").fill_null(0).cast(pl.Int64))
            .with_columns((pl.col("n_measurements") >= (self.coarse_points + self.refine_points)).alias("stop"))
            .select("repeat_id", "stop")
        )
        return result

    def finalize(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
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

        final = base.join(results_df, on="repeat_id", how="left").with_columns(pl.lit(2.0).alias("n_peaks"))
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
