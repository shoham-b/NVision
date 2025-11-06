from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs import Locator, ScanBatch


@dataclass
class TwoPeakGreedyLocator(Locator):
    """A greedy locator designed to find two distinct peaks in the domain.

    This strategy operates in two main phases:
    1.  **Coarse Grid Scan**: It first samples a predefined number of points
        (`coarse_points`) evenly across the domain.
    2.  **Greedy Refinement**: After the initial scan, it identifies the two
        strongest candidate peaks that are separated by at least a minimum
        distance. It then greedily proposes new measurement points around
        whichever of the two candidate peaks has been sampled the least, in an
        attempt to balance the refinement of both peaks.
    """

    coarse_points: int = 25
    refine_points: int = 5
    min_separation_frac: float = 0.05

    def propose_next(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        lo, hi = scan.x_min, scan.x_max
        grid = np.linspace(lo, hi, self.coarse_points).tolist()
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
            x_next = grid[n_taken] if n_taken < len(grid) else grid[-1]
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

            n1 = repeat_history.filter((pl.col("x") - x1).abs() <= w).height
            n2 = repeat_history.filter((pl.col("x") - x2).abs() <= w).height
            target = x2 if n2 < n1 else x1

            if target <= min(x1, x2):
                x_next = max(lo, target - 0.5 * w)
            elif target >= max(x1, x2):
                x_next = min(hi, target + 0.5 * w)
            else:
                x_next = target
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
                (pl.col("n_measurements") >= (self.coarse_points + 2 * self.refine_points)).alias(
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
                pl.lit(2.0).alias("n_peaks"),
                pl.lit(math.nan).alias("x1_hat"),
                pl.lit(math.nan).alias("x2_hat"),
                pl.lit(math.inf).alias("uncert"),
                pl.lit(math.inf).alias("uncert_pos"),
                pl.lit(math.inf).alias("uncert_sep"),
                pl.lit(0).alias("measurements"),
            )

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
                if not picks or all(abs(x - p) > 1e-9 for p in picks):
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
