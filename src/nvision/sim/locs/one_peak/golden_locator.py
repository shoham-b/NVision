from __future__ import annotations

from dataclasses import dataclass

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

    # Per-repeat state is now stored in the repeats DataFrame columns:
    # lower_bound, upper_bound, inner_c, inner_d

    def _get_averaged_history_per_repeat(self, history: pl.DataFrame, repeat_id: int) -> dict[float, float]:
        """Averages intensities for a single repeat."""
        repeat_history = history.filter(pl.col("repeat_id") == repeat_id)
        if repeat_history.is_empty():
            return {}
        averaged_df = repeat_history.group_by("x").agg(pl.mean("signal_values"))
        return dict(zip(averaged_df["x"], averaged_df["signal_values"], strict=False))

    def propose_next(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        """Proposes the next point for each active repeat using golden-section search."""
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        proposals = []
        for row in active.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            lower = row.get("lower_bound")
            upper = row.get("upper_bound")
            inner_c = row.get("inner_c")
            inner_d = row.get("inner_d")

            repeat_history = history.filter(pl.col("repeat_id") == repeat_id)

            # Initialize bounds on first call
            if lower is None or upper is None:
                lower, upper = scan.x_min, scan.x_max
                inner_c = upper - self._golden_ratio * (upper - lower)
                inner_d = lower + self._golden_ratio * (upper - lower)
                proposals.append(
                    {
                        "repeat_id": repeat_id,
                        "x": inner_c,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "inner_c": inner_c,
                        "inner_d": inner_d,
                    }
                )
                continue

            point_counts_df = repeat_history.group_by("x").agg(pl.len().alias("count"))
            point_counts = dict(zip(point_counts_df["x"], point_counts_df["count"], strict=False))

            # Check if we need more samples at inner_c or inner_d
            if inner_c is not None and point_counts.get(inner_c, 0) < self.samples_per_point:
                proposals.append(
                    {
                        "repeat_id": repeat_id,
                        "x": inner_c,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "inner_c": inner_c,
                        "inner_d": inner_d,
                    }
                )
                continue
            if inner_d is not None and point_counts.get(inner_d, 0) < self.samples_per_point:
                proposals.append(
                    {
                        "repeat_id": repeat_id,
                        "x": inner_d,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "inner_c": inner_c,
                        "inner_d": inner_d,
                    }
                )
                continue

            averaged_history = self._get_averaged_history_per_repeat(history, repeat_id)

            if len(averaged_history) == 1:
                lower, upper = scan.x_min, scan.x_max
                inner_c = next(iter(averaged_history.keys()))
                inner_d = lower + self._golden_ratio * (upper - lower)
                proposals.append(
                    {
                        "repeat_id": repeat_id,
                        "x": inner_d,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "inner_c": inner_c,
                        "inner_d": inner_d,
                    }
                )
                continue

            f_at_c = averaged_history.get(inner_c)
            f_at_d = averaged_history.get(inner_d)

            if f_at_c is None or f_at_d is None:
                x_next = inner_c if f_at_c is None else inner_d
                proposals.append(
                    {
                        "repeat_id": repeat_id,
                        "x": x_next,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "inner_c": inner_c,
                        "inner_d": inner_d,
                    }
                )
                continue

            if f_at_c > f_at_d:
                upper = inner_d
                inner_d = inner_c
                inner_c = upper - self._golden_ratio * (upper - lower)
                x_next = inner_c
            else:
                lower = inner_c
                inner_c = inner_d
                inner_d = lower + self._golden_ratio * (upper - lower)
                x_next = inner_d

            proposals.append(
                {
                    "repeat_id": repeat_id,
                    "x": x_next,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "inner_c": inner_c,
                    "inner_d": inner_d,
                }
            )

        if not proposals:
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})
        return pl.DataFrame(proposals)

    def should_stop(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        """Stops after a fixed number of evaluations per repeat."""
        counts = history.group_by("repeat_id").agg(pl.len().alias("n_measurements"))
        result = (
            repeats.select("repeat_id")
            .join(counts, on="repeat_id", how="left")
            .with_columns(pl.col("n_measurements").fill_null(0).cast(pl.Int64))
            .with_columns((pl.col("n_measurements") >= self.max_evals).alias("stop"))
            .select("repeat_id", "stop")
        )
        return result

    def finalize(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        """Returns the point with the highest observed intensity per repeat."""
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
