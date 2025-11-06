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

    def _get_coarse_proposals(
        self,
        active_with_counts: pl.DataFrame,
        coarse_grid: list[float],
    ) -> list[dict]:
        """Generate coarse scan proposals."""
        coarse_phase = active_with_counts.filter(pl.col("n_measurements") < self.coarse_points)
        coarse_proposals = []
        for row in coarse_phase.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            n_taken = row["n_measurements"]
            x_next = coarse_grid[n_taken] if n_taken < len(coarse_grid) else coarse_grid[-1]
            coarse_proposals.append({"repeat_id": repeat_id, "x": x_next})
        return coarse_proposals

    def _find_peaks(self, x: np.ndarray, y: np.ndarray, w: float) -> list[float]:
        """Find peaks in the signal."""
        peaks: list[float] = []
        for px in x:
            if not peaks or all(abs(px - p) >= w for p in peaks):
                peaks.append(px)
            if len(peaks) >= 3:
                break
        return peaks

    def _get_refine_point(
        self, center: float, width: float, lo: float, hi: float, step: int
    ) -> float:
        """Get next refinement point around a peak."""
        # Create a grid of points around the peak
        radius = min(center - lo, hi - center, 2 * width)
        start = max(lo, center - radius)
        end = min(hi, center + radius)

        # Use a golden ratio pattern for refinement
        golden_ratio = 0.618033988749895
        offset = (end - start) * (golden_ratio ** (step % 10))
        return start + offset

    def _get_refine_proposals(
        self,
        active_with_counts: pl.DataFrame,
        history: pl.DataFrame,
        w: float,
        lo: float,
        hi: float,
    ) -> list[dict]:
        """Generate refinement proposals around detected peaks."""
        refine_phase = active_with_counts.filter(pl.col("n_measurements") >= self.coarse_points)
        refine_proposals = []

        for row in refine_phase.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
            if repeat_history.is_empty():
                refine_proposals.append({"repeat_id": repeat_id, "x": 0.5 * (lo + hi)})
                continue

            # Find peaks in the signal
            x = repeat_history.get_column("x").to_numpy()
            y = repeat_history.get_column("signal_values").to_numpy()
            peaks = self._find_peaks(x, y, w)

            if not peaks:
                refine_proposals.append({"repeat_id": repeat_id, "x": 0.5 * (lo + hi)})
            else:
                # Find the highest peak
                peak_ys = [y[np.argmin(np.abs(x - px))] for px in peaks]
                main_peak_x = peaks[np.argmax(peak_ys)]
                refine_x = self._get_refine_point(main_peak_x, w, lo, hi, len(refine_proposals))
                refine_proposals.append({"repeat_id": repeat_id, "x": refine_x})

        return refine_proposals

    def propose_next(
        self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch
    ) -> pl.DataFrame:
        """Propose next measurement points for active repeats."""
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        lo, hi = scan.x_min, scan.x_max
        coarse_grid = np.linspace(lo, hi, self.coarse_points).tolist()
        w = (hi - lo) * self.min_separation_frac

        # Get measurement counts for active repeats
        counts = history.group_by("repeat_id").agg(pl.len().alias("n_measurements"))
        active_with_counts = active.join(counts, on="repeat_id", how="left").with_columns(
            pl.col("n_measurements").fill_null(0).cast(pl.Int64)
        )

        # Generate proposals for both phases
        coarse_proposals = self._get_coarse_proposals(active_with_counts, coarse_grid)
        refine_proposals = self._get_refine_proposals(active_with_counts, history, w, lo, hi)

        # Combine and return all proposals
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
                pl.lit(math.nan).alias("x1_hat"),
                pl.lit(math.nan).alias("x2_hat"),
                pl.lit(math.nan).alias("x3_hat"),
                pl.lit(math.inf).alias("uncert"),
                pl.lit(math.inf).alias("uncert_pos"),
                pl.lit(0).alias("measurements"),
            )

        w = (scan.x_max - scan.x_min) * self.min_separation_frac

        def _find_peaks(repeat_id: int) -> dict:
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
            if repeat_history.is_empty():
                return {
                    "repeat_id": repeat_id,
                    "n_peaks": 1.0,
                    "x1_hat": math.nan,
                    "x2_hat": math.nan,
                    "x3_hat": math.nan,
                    "uncert": math.inf,
                    "uncert_pos": math.inf,
                    "measurements": 0,
                }

            sorted_history = repeat_history.sort("signal_values", descending=True)
            picks: list[float] = []
            for x in sorted_history["x"]:
                if not picks or all(abs(x - p) > w for p in picks):
                    picks.append(x)
                if len(picks) >= 3:
                    break

            picks.sort()
            return {
                "repeat_id": repeat_id,
                "n_peaks": float(len(picks)),
                "x1_hat": float(picks[0]) if len(picks) >= 1 else math.nan,
                "x2_hat": float(picks[1]) if len(picks) >= 2 else math.nan,
                "x3_hat": float(picks[2]) if len(picks) >= 3 else math.nan,
                "uncert": float(w),
                "uncert_pos": float(w),
                "measurements": repeat_history.height,
            }

        results = [_find_peaks(rid) for rid in base.get_column("repeat_id")]
        results_df = pl.DataFrame(results)

        final = base.join(results_df, on="repeat_id", how="left")
        return final.select(
            "repeat_id",
            "n_peaks",
            "x1_hat",
            "x2_hat",
            "x3_hat",
            "uncert",
            "uncert_pos",
            "measurements",
        )
