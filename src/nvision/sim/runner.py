from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from .core import CompositeNoise, DataBatch, DataGenerator, MeasurementStrategy


def _common_keys(truth: Mapping[str, float], est: Mapping[str, float]) -> list[str]:
    return [k for k in truth.keys() if k in est]


def _rmse(truth: Mapping[str, float], est: Mapping[str, float], keys: Sequence[str]) -> float:
    if not keys:
        return 0.0
    return math.sqrt(sum((est[k] - truth[k]) ** 2 for k in keys) / len(keys))


@dataclass
class ExperimentRunner:
    """Runs measurement strategies on generated data with noise and summarizes metrics."""

    rng_seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def run_once(
        self,
        generator: DataGenerator,
        noise: CompositeNoise | None,
        strategies: Sequence[MeasurementStrategy],
    ) -> dict[str, dict[str, float]]:
        """Run a single dataset through strategies and compute simple metrics.

        Returns a dict: {strategy_name: {metric: value, ...}}
        Metrics include per-key absolute error and overall RMSE.
        """
        clean = generator.generate(self._rng)
        if noise is not None:
            data = noise.apply(clean, self._rng)
        else:
            data = DataBatch(
                time_points=clean.time_points,
                signal_values=list(clean.signal_values),
                meta=clean.meta,
            )
        results: dict[str, dict[str, float]] = {}
        for strat in strategies:
            name = strat.__class__.__name__
            est = strat.estimate(data)
            keys = _common_keys(clean.meta, est)
            summary: dict[str, float] = {}
            for k in keys:
                summary[f"abs_err_{k}"] = abs(est[k] - clean.meta[k])
            summary["rmse"] = _rmse(clean.meta, est, keys)
            # keep the estimates too for reference
            for k, v in est.items():
                summary[f"est_{k}"] = v
            results[name] = summary
        return results

    def sweep(
        self,
        generator: DataGenerator,
        noises: Sequence[CompositeNoise | None],
        strategies: Sequence[MeasurementStrategy],
        repeats: int = 10,
    ) -> pl.DataFrame:
        """Run multiple repeats for each noise combo and collect average metrics.

        Returns a Polars DataFrame with columns: noise, strategy, and metric columns (floats).
        """
        rows: list[dict[str, float | str]] = []
        for noise in noises:
            noise_name = self._noise_name(noise)
            accum: dict[str, dict[str, float]] = {}
            counts: dict[str, int] = {}
            for _ in range(repeats):
                res = self.run_once(generator, noise, strategies)
                for strat_name, metrics in res.items():
                    if strat_name not in accum:
                        accum[strat_name] = {}
                        counts[strat_name] = 0
                    for k, v in metrics.items():
                        accum[strat_name][k] = accum[strat_name].get(k, 0.0) + v
                    counts[strat_name] += 1
            for strat_name, sums in accum.items():
                n = max(1, counts[strat_name])
                avg = {k: v / n for k, v in sums.items()}
                row: dict[str, float | str] = {"noise": noise_name, "strategy": strat_name}
                row.update(avg)
                rows.append(row)
        if not rows:
            return pl.DataFrame({"noise": [], "strategy": []})
        # Construct DataFrame; Polars will infer dtypes; ensure metric columns are Float64
        df = pl.DataFrame(rows)
        # Try to cast non-string columns except noise/strategy to Float64
        metric_cols = [c for c in df.columns if c not in ("noise", "strategy")]
        if metric_cols:
            df = df.with_columns([pl.col(metric_cols).cast(pl.Float64)])
        return df

    @staticmethod
    def _noise_name(noise: CompositeNoise | None) -> str:
        if noise is None:
            return "NoNoise"
        parts = getattr(noise, "_parts", [])  # type: ignore[attr-defined]
        if not parts:
            return "EmptyNoise"
        return "+".join([p.__class__.__name__ for p in parts])

    @staticmethod
    def to_csv(
        rows_or_df: list[tuple[str, str, dict[str, float]]] | pl.DataFrame, path: str,
    ) -> None:
        """Write results to CSV using Polars.

        Accepts either:
        - A Polars DataFrame returned by sweep, or
        - Legacy List[Tuple[noise, strategy, metrics_dict]] from older API.
        """
        if isinstance(rows_or_df, pl.DataFrame):
            df = rows_or_df
        else:
            # Build DataFrame from legacy rows
            rows: list[dict[str, float | str]] = []
            metric_keys: list[str] = []
            for _, _, metrics in rows_or_df:
                for k in metrics.keys():
                    if k not in metric_keys:
                        metric_keys.append(k)
            for noise_name, strat_name, metrics in rows_or_df:
                row: dict[str, float | str] = {"noise": noise_name, "strategy": strat_name}
                for k in metric_keys:
                    row[k] = metrics.get(k, math.nan)
                rows.append(row)
            df = pl.DataFrame(rows)
            met_cols = [c for c in df.columns if c not in ("noise", "strategy")]
            if met_cols:
                df = df.with_columns([pl.col(met_cols).cast(pl.Float64)])
        df.write_csv(path)
