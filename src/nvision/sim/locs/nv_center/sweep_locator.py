from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch


@dataclass
class NVCenterSweepLocator(Locator):
    """Sweep locator optimized for NV center signals with potential multiple peaks."""

    coarse_points: int = 30
    refine_points: int = 10
    min_separation_frac: float = 0.03

    def propose_next(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
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

        # Refine phase: propose around detected peaks
        refine_phase = active_with_counts.filter(pl.col("n_measurements") >= self.coarse_points)
        refine_proposals = []
        if not refine_phase.is_empty():
            for row in refine_phase.iter_rows(named=True):
                repeat_id = row["repeat_id"]
                n_taken = row["n_measurements"]

                # Get history for this repeat
                repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
                if not repeat_history.is_empty():
                    x_vals = repeat_history.get_column("x").to_numpy()
                    y_vals = repeat_history.get_column("signal_values").to_numpy()
                    peaks = self._find_peaks(x_vals, y_vals, self.min_separation_frac * (scan.x_max - scan.x_min))

                    if peaks:
                        main_peak = peaks[0]
                        x_next = self._get_refine_point(
                            main_peak, 0.1 * (scan.x_max - scan.x_min), scan.x_min, scan.x_max, n_taken
                        )
                    else:
                        x_next = (scan.x_min + scan.x_max) / 2
                else:
                    x_next = (scan.x_min + scan.x_max) / 2

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
                pl.lit(1.0).alias("n_peaks"),
                pl.lit(float("nan")).alias("x1_hat"),
                pl.lit(float("inf")).alias("uncert"),
                pl.lit(0).alias("measurements"),
            )

        # Find peaks for each repeat
        results = []
        for repeat_id in repeats.get_column("repeat_id").to_list():
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
            if repeat_history.is_empty():
                results.append(
                    {
                        "repeat_id": repeat_id,
                        "n_peaks": 1.0,
                        "x1_hat": float("nan"),
                        "uncert": float("inf"),
                        "measurements": 0,
                    }
                )
            else:
                x_vals = repeat_history.get_column("x").to_numpy()
                y_vals = repeat_history.get_column("signal_values").to_numpy()
                peaks = self._find_peaks(x_vals, y_vals, self.min_separation_frac * (scan.x_max - scan.x_min))
                results.append(
                    {
                        "repeat_id": repeat_id,
                        "n_peaks": 1.0,
                        "x1_hat": peaks[0] if peaks else float("nan"),
                        "uncert": 0.0,
                        "measurements": len(repeat_history),
                    }
                )

        return pl.DataFrame(results).select("repeat_id", "n_peaks", "x1_hat", "uncert", "measurements")

    def _find_peaks(self, x: np.ndarray, y: np.ndarray, w: float) -> list[float]:
        # Sort by signal value descending to prioritize high peaks
        sorted_indices = np.argsort(-y)
        peaks: list[float] = []
        for idx in sorted_indices:
            px = x[idx]
            if not peaks or all(abs(px - p) >= w for p in peaks):
                peaks.append(px)
            if len(peaks) >= 3:
                break
        return peaks

    def _get_refine_point(self, center: float, width: float, lo: float, hi: float, step: int) -> float:
        radius = min(center - lo, hi - center, 2 * width)
        start = max(lo, center - radius)
        end = min(hi, center + radius)
        golden_ratio = 0.618033988749895
        offset = (end - start) * (golden_ratio ** (step % 10))
        return start + offset
