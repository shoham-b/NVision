"""
Base class for Sequential Bayesian Experiment Design Locators for NV Centers.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch
from nvision.sim.locs.nv_center._bayesian_plotting import (
    plot_bo as _plot_bo,
)
from nvision.sim.locs.nv_center._bayesian_plotting import (
    plot_convergence_stats as _plot_convergence_stats,
)
from nvision.sim.locs.nv_center._bayesian_plotting import (
    plot_posterior_history as _plot_posterior_history,
)
from nvision.sim.locs.nv_center._jit_kernels import _update_posterior_math
from nvision.sim.locs.nv_center._lineshape_optimizer import (
    get_optim_config,
    optimize_lineshape_params,
)
from nvision.sim.locs.nv_center._odmr_model import (
    calculate_log_likelihoods_grid,
    coerce_measurement,
    compute_likelihood,
    odmr_model,
)
from nvision.sim.locs.nv_center.sweep_locator import NVCenterSweepLocator

log = logging.getLogger(__name__)


@dataclass
class NVCenterBayesianLocatorBase(Locator):
    """
    Abstract base class for Bayesian locators for NV center magnetometry.

    Provides shared state management, posterior updates, likelihood calculations,
    and global stopping conditions.
    """

    max_evals: int = 500
    prior_bounds: tuple[float, float] = (2.6e9, 3.1e9)
    noise_model: str = "gaussian"
    acquisition_function: str = "expected_information_gain"
    convergence_threshold: float = 1e-10
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
    sweeper: NVCenterSweepLocator = None

    # Internal state fields
    _locators: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize the locator after dataclass creation."""
        if self.n_warmup >= self.max_evals:
            raise ValueError("n_warmup must be smaller than max_evals")
        self.sweeper = NVCenterSweepLocator(coarse_points=self.n_warmup, refine_points=self.n_warmup)
        self._bo: Any = None
        self._bo_utility: Any = None

        self._unscaled_prior_bounds = self.prior_bounds
        self._unscaled_linewidth_prior = self.linewidth_prior
        self._unscaled_gaussian_width_prior = self.gaussian_width_prior
        self._unscaled_split_prior = self.split_prior

        self.reset_posterior()
        self.measurement_history: list[dict[str, float]] = []
        self.utility_history: list[float] = []
        self.posterior_history: list[np.ndarray] = []
        self.parameter_history: list[dict[str, float]] = []

    def reset_posterior(self):
        """Reset posterior distributions to priors."""
        self.freq_grid = np.linspace(self.prior_bounds[0], self.prior_bounds[1], self.grid_resolution)
        self.freq_posterior = np.ones(self.grid_resolution) / self.grid_resolution
        self.current_estimates = {
            "frequency": np.mean(self.prior_bounds),
            "linewidth": np.mean(self.linewidth_prior),
            "amplitude": np.mean(self.amplitude_prior),
            "background": np.mean(self.background_prior),
            "uncertainty": np.inf,
            "gaussian_width": np.mean(self.gaussian_width_prior),
            "split": np.mean(self.split_prior),
            "k_np": np.mean(self.k_np_prior),
            "entropy": np.log(self.grid_resolution),  # Max entropy for uniform
            "max_prob": 1.0 / self.grid_resolution,  # Uniform probability
        }
        self._init_bayes_optimizer()

    def reset_run_state(self) -> None:
        """Clear accumulated histories and reinitialize the posterior."""
        if hasattr(self, "measurement_history"):
            self.measurement_history.clear()
        if hasattr(self, "posterior_history"):
            self.posterior_history.clear()
        if hasattr(self, "utility_history"):
            self.utility_history.clear()
        if hasattr(self, "parameter_history"):
            self.parameter_history.clear()
        self.reset_posterior()

    def _init_bayes_optimizer(self) -> None:
        self._bo = None
        self._bo_utility = None

    # -- Model / likelihood (delegated) ----------------------------------

    def odmr_model(
        self,
        frequency: float | np.ndarray,
        params: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        return odmr_model(frequency, params, self.distribution)

    def _coerce_measurement(self, measurement: dict[str, float]) -> dict[str, float]:
        return coerce_measurement(measurement)

    def likelihood(
        self,
        measurement: dict[str, float],
        params: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        return compute_likelihood(measurement, params, self.distribution, self.noise_model)

    # -- Optimisation (delegated) ----------------------------------------

    def _get_optim_config(self):
        return get_optim_config(
            self.distribution,
            self.linewidth_prior,
            self.gaussian_width_prior,
            self.split_prior,
            self.k_np_prior,
            self.amplitude_prior,
            self.background_prior,
        )

    def _optimize_lineshape_params(self):
        param_keys, bounds = self._get_optim_config()
        optimize_lineshape_params(
            self.measurement_history,
            self.current_estimates,
            self.distribution,
            self.noise_model,
            param_keys,
            bounds,
        )

    # -- Grid log-likelihoods & posterior update --------------------------

    def _calculate_log_likelihoods(
        self,
        measurement: dict[str, float],
        base_params: dict[str, float],
    ) -> np.ndarray:
        """Calculate log-likelihoods over the frequency grid using vectorized operations."""
        return calculate_log_likelihoods_grid(
            self.freq_grid, measurement, base_params, self.distribution, self.noise_model
        )

    def update_posterior(self, measurement: dict[str, float]):
        measurement = self._coerce_measurement(measurement)

        base_params = {k: v for k, v in self.current_estimates.items() if k != "frequency"}
        log_likelihoods = self._calculate_log_likelihoods(measurement, base_params)

        new_posterior, est_freq, uncertainty, entropy, max_prob = _update_posterior_math(
            self.freq_grid, self.freq_posterior, log_likelihoods
        )

        self.freq_posterior = new_posterior
        self.current_estimates["frequency"] = est_freq
        self.current_estimates["uncertainty"] = uncertainty
        self.current_estimates["entropy"] = entropy
        self.current_estimates["max_prob"] = max_prob

        self.measurement_history.append(measurement.copy())
        self.posterior_history.append(self.freq_posterior.copy())
        self._optimize_lineshape_params()
        self.parameter_history.append(self.current_estimates.copy())

    # -- Plotting (delegated) --------------------------------------------

    def plot_bo(
        self,
        output_path: Path,
        fig_size: tuple[int, int] = (12, 8),
        dpi: int = 100,
    ) -> Path | None:
        return _plot_bo(output_path, fig_size, dpi)

    def plot_posterior_history(self, output_path: Path) -> Path | None:
        return _plot_posterior_history(self.freq_grid, self.posterior_history, output_path)

    def plot_convergence_stats(self, output_path: Path) -> Path | None:
        return _plot_convergence_stats(self.freq_grid, self.posterior_history, self.utility_history, output_path)

    # -- History helpers -------------------------------------------------

    def _history_iter(self, history: Sequence | pl.DataFrame) -> Iterable[dict[str, float]]:
        if isinstance(history, pl.DataFrame):
            yield from history.iter_rows(named=True)
        else:
            for entry in history:
                if isinstance(entry, dict):
                    yield entry
                else:
                    entry_map = {
                        "x": getattr(entry, "x", None),
                        "signal_values": getattr(entry, "signal_values", getattr(entry, "intensity", None)),
                        "uncertainty": getattr(entry, "uncertainty", 0.05),
                    }
                    if entry_map["x"] is None or entry_map["signal_values"] is None:
                        raise KeyError("History entries must provide x and signal_values/intensity")
                    yield entry_map

    def _ingest_history(self, history: Sequence | pl.DataFrame) -> None:
        for row in self._history_iter(history):
            is_new = not any(existing["x"] == row["x"] for existing in self.measurement_history)
            if is_new:
                self.update_posterior(row)

    def _rescale_priors_if_needed(self, scan: ScanBatch | None):
        """Rescale priors if scan bounds suggest a normalized coordinate system."""
        if scan is None:
            return

        scan_bounds = (scan.x_min, scan.x_max)
        is_normalized_scan = np.isclose(scan_bounds[0], 0.0) and np.isclose(scan_bounds[1], 1.0)
        is_unscaled = np.isclose(self.prior_bounds[0], self._unscaled_prior_bounds[0])

        if is_normalized_scan and is_unscaled:
            scale_factor = self._unscaled_prior_bounds[1] - self._unscaled_prior_bounds[0]
            if scale_factor > 0:
                self.prior_bounds = (0.0, 1.0)
                self.linewidth_prior = tuple(p / scale_factor for p in self._unscaled_linewidth_prior)
                self.gaussian_width_prior = tuple(p / scale_factor for p in self._unscaled_gaussian_width_prior)
                self.split_prior = tuple(p / scale_factor for p in self._unscaled_split_prior)
                self.reset_run_state()

    # -- Acquisition & propose_next --------------------------------------

    @abstractmethod
    def _optimize_acquisition(self, domain: tuple[float, float]) -> tuple[float, float]:
        """Determine next frequency point and its utility."""
        raise NotImplementedError

    def propose_next(
        self,
        history: Sequence | pl.DataFrame,
        scan: ScanBatch | None = None,
        repeats: pl.DataFrame | None = None,
    ) -> float | pl.DataFrame:
        """Propose the next measurement point.

        Matches Locator interface but delegates to _optimize_acquisition.
        """
        # Handle argument swapping if called as (history, repeats, scan)
        if isinstance(scan, pl.DataFrame) and repeats is not None and not isinstance(repeats, pl.DataFrame):
            real_repeats = scan
            real_scan = repeats
            scan = real_scan
            repeats = real_repeats
        elif isinstance(repeats, ScanBatch) and scan is None:
            pass

        # Original single-value interface
        if not isinstance(history, pl.DataFrame):
            history = pl.DataFrame(history)

        hist_len = history.height
        self._rescale_priors_if_needed(scan)

        self._ingest_history(history)

        if hist_len < self.n_warmup:
            dummy_repeats = pl.DataFrame({"repeat_id": [0], "active": [True]})
            history_with_id = history.with_columns(pl.lit(0, dtype=pl.Int64).alias("repeat_id"))

            proposals = self.sweeper.propose_next(history_with_id, dummy_repeats, scan)

            if not proposals.is_empty():
                return proposals.item(0, "x")

            return (scan.x_min + scan.x_max) / 2.0

        next_freq, utility = self._optimize_acquisition(self.prior_bounds)
        self.utility_history.append(utility)
        if len(self.utility_history) > self.utility_history_window:
            self.utility_history.pop(0)
        return float(next_freq)

    # -- Stopping --------------------------------------------------------

    def should_stop(
        self,
        history: Sequence | pl.DataFrame,
        repeats: pl.DataFrame | None = None,
        scan: ScanBatch | None = None,
    ) -> bool | pl.DataFrame:
        if isinstance(repeats, ScanBatch) and scan is None:
            scan = repeats
            repeats = None

        if scan is None:
            raise ValueError("scan must be provided")

        hist_len = history.height if isinstance(history, pl.DataFrame) else len(history)
        if hist_len >= self.max_evals:
            return True
        if self.current_estimates["uncertainty"] < self.convergence_threshold:
            return True

        if self._check_crb_stopping(hist_len):
            return True

        return len(self.utility_history) == self.utility_history_window and all(
            u < self.min_uncertainty_reduction for u in self.utility_history
        )

    def _check_crb_stopping(self, hist_len: int) -> bool:
        if hist_len <= 10:
            return False

        from nvision.sim.locs.nv_center.fisher_information import calculate_crb

        if not self.measurement_history:
            return False

        x_vals = [m["x"] for m in self.measurement_history]
        crb_history = calculate_crb(x_vals, self.current_estimates, noise_model=self.noise_model, noise_params=None)

        if crb_history:
            current_crb = crb_history[-1]
            if current_crb > 0 and self.current_estimates["uncertainty"] <= 2.0 * current_crb:
                return True
        return False

    # -- Finalize --------------------------------------------------------

    def finalize(
        self,
        history: Sequence | pl.DataFrame,
        repeats: pl.DataFrame | None = None,
        scan: ScanBatch | None = None,
    ) -> dict[str, float] | pl.DataFrame:
        if isinstance(repeats, ScanBatch) and scan is None:
            scan = repeats
            repeats = None

        if scan is None:
            raise ValueError("scan must be provided")

        self._ingest_history(history)

        uncertainty = float(self.current_estimates["uncertainty"])
        frequency = float(self.current_estimates["frequency"])

        results = {
            "n_peaks": 1.0,
            "x1_hat": frequency,
            "x2_hat": math.nan,
            "x3_hat": math.nan,
            "uncert": uncertainty,
            "uncert_pos": uncertainty,
        }

        if self.distribution == "voigt-zeeman":
            split = float(self.current_estimates.get("split", 0.0))
            results.update(
                {
                    "n_peaks": 3.0,
                    "x1_hat": frequency - split,
                    "x3_hat": frequency + split,
                }
            )

        return results
