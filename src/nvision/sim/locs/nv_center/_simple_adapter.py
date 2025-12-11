"""Batched adapter for SimpleSequentialLocator.

This adapter wraps the SimpleSequentialLocator to work with the batched API
used by the CLI runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch
from nvision.sim.locs.nv_center.simple_sequential_locator import (
    SimpleSequentialLocator as _OriginalSimpleLocator,
)


@dataclass
class SimpleSequentialLocatorBatched(Locator):
    """Batched adapter for the Simple Sequential locator.

    This adapter maintains one instance of the original per-repeat locator for each
    repeat_id and delegates calls to them individually.
    """

    max_evals: int = 50
    prior_bounds: tuple[float, float] = (2.6e9, 3.1e9)
    noise_model: str = "gaussian"
    acquisition_function: str = "expected_information_gain"
    convergence_threshold: float = 1e-8
    min_uncertainty_reduction: float = 1e-9
    n_monte_carlo: int = 100
    grid_resolution: int = 1000
    linewidth_prior: tuple[float, float] = (1e6, 50e6)
    distribution: str = "lorentzian"
    gaussian_width_prior: tuple[float, float] = (1e6, 50e6)
    split_prior: tuple[float, float] = (1e6, 10e6)
    k_np_prior: tuple[float, float] = (2.0, 4.0)
    amplitude_prior: tuple[float, float] = (0.01, 1.0)
    background_prior: tuple[float, float] = (0.99, 1.01)
    bo_enabled: bool = False
    bo_acq_func: str = "ucb"
    bo_kappa: float = 2.576
    bo_xi: float = 0.0
    bo_random_state: int | None = None
    utility_history_window: int = 9
    n_warmup: int = 10

    _locators: dict[int, _OriginalSimpleLocator] = field(default_factory=dict, init=False, repr=False)

    def _get_locator(self, repeat_id: int) -> _OriginalSimpleLocator:
        """Get or create a locator instance for a given repeat_id."""
        if repeat_id not in self._locators:
            self._locators[repeat_id] = _OriginalSimpleLocator(
                max_evals=self.max_evals,
                prior_bounds=self.prior_bounds,
                convergence_threshold=self.convergence_threshold,
                grid_resolution=self.grid_resolution,
                linewidth_prior=self.linewidth_prior,
                distribution=self.distribution,
                gaussian_width_prior=self.gaussian_width_prior,
                split_prior=self.split_prior,
                amplitude_prior=self.amplitude_prior,
                background_prior=self.background_prior,
                n_warmup=self.n_warmup,
            )
        return self._locators[repeat_id]

    def propose_next(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        """Propose next measurement for each active repeat."""
        active = repeats.filter(pl.col("active"))
        if active.is_empty():
            return pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

        proposals = []
        for row in active.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id).drop("repeat_id", "step")
            locator = self._get_locator(repeat_id)
            x_next = locator.propose_next(repeat_history, scan)
            proposals.append({"repeat_id": repeat_id, "x": float(x_next)})

        return pl.DataFrame(proposals) if proposals else pl.DataFrame(schema={"repeat_id": pl.Int64, "x": pl.Float64})

    def should_stop(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        """Check stopping criteria for each repeat."""
        stop_flags = []
        for row in repeats.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id).drop("repeat_id", "step")
            locator = self._get_locator(repeat_id)
            stop = locator.should_stop(repeat_history, scan)
            stop_flags.append({"repeat_id": repeat_id, "stop": bool(stop)})

        return pl.DataFrame(stop_flags)

    def finalize(self, history: pl.DataFrame, repeats: pl.DataFrame, scan: ScanBatch) -> pl.DataFrame:
        """Finalize all repeats and return per-repeat metrics."""
        results = []
        for row in repeats.iter_rows(named=True):
            repeat_id = row["repeat_id"]
            repeat_history = history.filter(pl.col("repeat_id") == repeat_id).drop("repeat_id", "step")
            locator = self._get_locator(repeat_id)
            metrics = locator.finalize(repeat_history, scan)
            results.append({"repeat_id": repeat_id, **metrics, "measurements": repeat_history.height})

        return (
            pl.DataFrame(results)
            if results
            else repeats.select("repeat_id").with_columns(
                pl.lit(math.nan).alias("x1_hat"),
                pl.lit(math.inf).alias("uncert"),
                pl.lit(0).alias("measurements"),
            )
        )
