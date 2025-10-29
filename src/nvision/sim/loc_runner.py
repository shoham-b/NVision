from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from .core import CompositeNoise, ScanGenerator
from .locs import LocatorStrategy, MeasurementProcess, ScalarMeasure, ScanBatch


def _pairing_error(truth: list[float], est: Mapping[str, float]) -> dict[str, float]:
    if len(truth) == 1:
        xh = est.get("x1_hat", est.get("x_hat"))
        err = (
            abs(float(xh) - truth[0])
            if xh is not None and isinstance(xh, int | float) and math.isfinite(float(xh))
            else math.inf
        )
        return {"abs_err_x": float(err)}
    x1h = est.get("x1_hat")
    x2h = est.get("x2_hat")
    if (
        x1h is None
        or x2h is None
        or not (
            isinstance(x1h, int | float)
            and isinstance(x2h, int | float)
            and math.isfinite(x1h)
            and math.isfinite(x2h)
        )
    ):
        return {"abs_err_x1": math.inf, "abs_err_x2": math.inf, "pair_rmse": math.inf}
    xs = sorted([float(x1h), float(x2h)])
    t = sorted(truth)
    err1 = abs(xs[0] - t[0])
    err2 = abs(xs[1] - t[1])
    return {
        "abs_err_x1": err1,
        "abs_err_x2": err2,
        "pair_rmse": math.sqrt(0.5 * (err1 * err1 + err2 * err2)),
    }


@dataclass
class LocatorRunner:
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
        strat: LocatorStrategy,
        repeats: int,
        max_steps: int,
        history_callback: Callable[[str, str, str, int, ScanBatch, pl.DataFrame], None] | None,
    ) -> dict[str, float | str]:
        sum_metrics: dict[str, float] = {}
        count = 0
        for repeat_idx in range(repeats):
            scan = gen.generate(self._rng)
            history_df, est = self.run_once(scan, strat, noise, max_steps)
            if history_callback is not None:
                history_callback(
                    gen_name,
                    noise_name,
                    strat_name,
                    repeat_idx,
                    scan,
                    history_df,
                )
            metrics = _pairing_error(scan.truth_positions, est)
            for key in ("uncert", "uncert_pos", "uncert_sep"):
                if key in est and isinstance(est[key], int | float):
                    metrics[key] = float(est[key])  # type: ignore[arg-type]
            for k, v in metrics.items():
                sum_metrics[k] = sum_metrics.get(k, 0.0) + float(v)
            count += 1

        avg = {k: v / max(1, count) for k, v in sum_metrics.items()}
        row: dict[str, float | str] = {
            "generator": gen_name,
            "noise": noise_name,
            "strategy": strat_name,
        }
        row.update(avg)
        return row

    def run_once(
        self,
        scan: ScanBatch,
        strategy: LocatorStrategy,
        noise: CompositeNoise | None,
        max_steps: int = 200,
    ) -> tuple[pl.DataFrame, dict[str, float]]:
        proc = MeasurementProcess(
            scan=scan,
            meas=ScalarMeasure(noise=noise),
            strategy=strategy,
            max_steps=max_steps,
        )
        return proc.run(self._rng)

    def sweep(
        self,
        generators: Sequence[tuple[str, ScanGenerator]],
        strategies: Sequence[tuple[str, LocatorStrategy]],
        noises: Sequence[tuple[str, CompositeNoise | None]],
        repeats: int = 10,
        max_steps: int = 200,
        history_callback: Callable[[str, str, str, int, ScanBatch, pl.DataFrame], None]
        | None = None,
    ) -> pl.DataFrame:
        rows: list[dict[str, float | str]] = []
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
                            history_callback,
                        )
                    )
        if not rows:
            return pl.DataFrame({"generator": [], "noise": [], "strategy": []})
        df = pl.DataFrame(rows)
        metric_cols = [c for c in df.columns if c not in ("generator", "noise", "strategy")]
        if metric_cols:
            df = df.with_columns(pl.col(metric_cols).cast(pl.Float64))
        return df
