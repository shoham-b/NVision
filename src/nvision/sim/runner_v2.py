"""V2 locator runner that replaces the legacy LocatorRunner."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from nvision.sim.core import CompositeNoise, ScanGenerator
from nvision.sim.locs.v2.base import Locator
from nvision.sim.locs.v2.experiment import Experiment
from nvision.sim.locs.v2.runner import Runner
from nvision.sim.scan_batch import ScanBatch


def _pairing_error(truth: list[float], est: Mapping[str, float]) -> dict[str, float]:
    """Calculate pairing errors between truth and estimates."""
    if len(truth) == 1:
        xh = est.get("x1_hat", est.get("x_hat", est.get("peak_x")))
        err = (
            abs(float(xh) - truth[0])
            if xh is not None and isinstance(xh, int | float) and math.isfinite(float(xh))
            else math.inf
        )
        return {"abs_err_x": err}

    if len(truth) == 2:
        x1h = est.get("x1_hat")
        x2h = est.get("x2_hat")

        if (
            x1h is not None
            and isinstance(x1h, int | float)
            and math.isfinite(x1h)
            and x2h is not None
            and isinstance(x2h, int | float)
            and math.isfinite(x2h)
        ):
            xs = sorted([float(x1h), float(x2h)])
            t = sorted(truth)
            err1 = abs(xs[0] - t[0])
            err2 = abs(xs[1] - t[1])
            return {
                "abs_err_x1": err1,
                "abs_err_x2": err2,
                "pair_rmse": math.sqrt(0.5 * (err1 * err1 + err2 * err2)),
            }

        return {"abs_err_x1": math.inf, "abs_err_x2": math.inf, "pair_rmse": math.inf}

    # For 3+ peaks, return basic error
    return {"error": math.inf}


def _aggregate_repeat_records(records: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across multiple repeats."""
    if not records:
        return {}

    metrics_df = pl.DataFrame(records)
    numeric_cols = [name for name, dtype in metrics_df.schema.items() if dtype.is_numeric()]
    if not numeric_cols:
        return {}

    metrics_df = metrics_df.with_columns([pl.col(name).cast(pl.Float64) for name in numeric_cols])
    agg_exprs = []
    for name in numeric_cols:
        if name in {"measurements", "duration_ms"}:
            agg_exprs.append(pl.col(name).mean().alias(name))
        else:
            agg_exprs.append(pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).mean().alias(name))

    aggregated_row = metrics_df.select(agg_exprs).to_dicts()[0]
    return {name: float(value) if value is not None else float("nan") for name, value in aggregated_row.items()}


@dataclass
class LocatorRunStatsV2:
    """Statistics from a single locator run."""

    history: pl.DataFrame
    estimate: Mapping[str, float]
    measurements: int
    duration_ms: float
    timed_out: bool = False


@dataclass
class LocatorRunnerV2:
    """V2 locator runner that uses the stateless v2 architecture."""

    rng_seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def _run_configuration(
        self,
        gen_name: str,
        gen: ScanGenerator,
        noise_name: str,
        noise: CompositeNoise | None,
        strat_name: str,
        strat: Locator,
        repeats: int,
        max_steps: int,
    ) -> dict:
        """Run a single configuration (generator + noise + strategy) for multiple repeats."""
        repeat_records: list[dict[str, float]] = []

        for _repeat_idx in range(repeats):
            scan = gen.generate(self._rng)
            run_stats = self.run_once(scan, strat, noise, max_steps)
            est = run_stats.estimate
            metrics = _pairing_error(scan.truth_positions, est)

            # Add uncertainty metrics if available
            for key in ("uncert", "uncert_pos", "uncert_sep"):
                value = est.get(key)
                if isinstance(value, int | float):
                    metrics[key] = float(value)

            record: dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, int | float):
                    record[key] = float(value)
            record["measurements"] = float(run_stats.measurements)
            record["duration_ms"] = float(run_stats.duration_ms)
            repeat_records.append(record)

        aggregated = _aggregate_repeat_records(repeat_records)

        row = {
            "generator": gen_name,
            "noise": noise_name,
            "strategy": strat_name,
            "repeats": repeats,
        }
        row.update(aggregated)
        return row

    def run_once(
        self,
        scan: ScanBatch,
        strategy: Locator,
        noise: CompositeNoise | None,
        max_steps: int = 200,
        timeout_s: float = 300.0,
    ) -> LocatorRunStatsV2:
        """Run a single locator instance on a scan."""
        start = time.perf_counter()

        # Create experiment from scan and noise
        experiment = Experiment(scan=scan, noise=noise)

        # Use v2 runner
        runner = Runner()

        try:
            results, histories, stop_reasons = runner.run(
                locator=strategy,
                experiment=experiment,
                repeats=1,
                rng=self._rng,
                max_steps=max_steps,
                timeout_s=timeout_s,
                return_history=True,
            )

            result = results[0] if results else {}
            history_df = (
                histories[0] if histories else pl.DataFrame(schema={"x": pl.Float64, "signal_value": pl.Float64})
            )
            timed_out = bool(stop_reasons and stop_reasons[0] == "repeat_timeout")

        except Exception as e:
            # Handle any errors gracefully
            result = {"error": str(e)}
            history_df = pl.DataFrame(schema={"x": pl.Float64, "signal_value": pl.Float64})
            timed_out = False

        duration_ms = (time.perf_counter() - start) * 1000.0

        return LocatorRunStatsV2(
            history=history_df,
            estimate=result,
            measurements=history_df.height,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    def sweep(
        self,
        generators: Sequence[tuple[str, ScanGenerator]],
        strategies: Sequence[tuple[str, Locator]],
        noises: Sequence[tuple[str, CompositeNoise | None]],
        repeats: int = 10,
        max_steps: int = 200,
    ) -> pl.DataFrame:
        """Run a parameter sweep across generators, strategies, and noise models."""
        rows: list[dict] = []
        for gen_name, gen in generators:
            for noise_name, noise in noises:
                for strat_name, strat in strategies:
                    rows.append(
                        self._run_configuration(
                            gen_name,
                            gen,
                            noise_name,
                            noise,
                            strat_name,
                            strat,
                            repeats,
                            max_steps,
                        )
                    )
        if not rows:
            return pl.DataFrame(
                {
                    "generator": [],
                    "noise": [],
                    "strategy": [],
                    "repeats": [],
                }
            )
        df = pl.DataFrame(rows)
        metric_cols = [c for c in df.columns if c not in ("generator", "noise", "strategy", "repeats")]
        if metric_cols:
            df = df.with_columns(pl.col(metric_cols).cast(pl.Float64))
        return df
