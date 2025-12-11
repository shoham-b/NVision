"""Base class for NV Center locators with shared functionality."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import Locator, ScanBatch
from nvision.sim.locs.nv_center.sweep_locator import NVCenterSweepLocator


@dataclass
class NVCenterLocatorBase(Locator):
    """Base class for NV Center locators with shared configuration and logic.

    Provides common parameters, state management, warmup logic, and finalization.
    Subclasses implement `_propose_measurement()` for their specific strategy.
    """

    # Common configuration parameters
    max_evals: int = 50
    prior_bounds: tuple[float, float] = (2.6e9, 3.1e9)
    convergence_threshold: float = 1e-8
    grid_resolution: int = 1000
    linewidth_prior: tuple[float, float] = (1e6, 50e6)
    distribution: str = "lorentzian"
    gaussian_width_prior: tuple[float, float] = (1e6, 50e6)
    split_prior: tuple[float, float] = (1e6, 10e6)
    k_np_prior: tuple[float, float] = (2.0, 4.0)
    amplitude_prior: tuple[float, float] = (0.01, 1.0)
    background_prior: tuple[float, float] = (0.99, 1.01)
    n_warmup: int = 10
    sweeper: NVCenterSweepLocator = None

    def __post_init__(self):
        """Initialize the locator after dataclass creation."""
        if self.n_warmup >= self.max_evals:
            raise ValueError("n_warmup must be smaller than max_evals")

        self.sweeper = NVCenterSweepLocator(coarse_points=self.n_warmup, refine_points=self.n_warmup)

        # Store unscaled priors for potential rescaling
        self._unscaled_prior_bounds = self.prior_bounds
        self._unscaled_linewidth_prior = self.linewidth_prior
        self._unscaled_gaussian_width_prior = self.gaussian_width_prior
        self._unscaled_split_prior = self.split_prior

        # Initialize state
        self.current_estimates = {
            "frequency": np.mean(self.prior_bounds),
            "linewidth": np.mean(self.linewidth_prior),
            "amplitude": np.mean(self.amplitude_prior),
            "background": np.mean(self.background_prior),
            "uncertainty": np.inf,
            "gaussian_width": np.mean(self.gaussian_width_prior),
            "split": np.mean(self.split_prior),
            "k_np": np.mean(self.k_np_prior),
        }
        self.measurement_history: list[dict[str, float]] = []
        self.parameter_history: list[dict[str, float]] = []  # Initialize history

    def reset_run_state(self) -> None:
        """Clear accumulated histories and reinitialize estimates."""
        if hasattr(self, "measurement_history"):
            self.measurement_history.clear()
        if hasattr(self, "parameter_history"):
            self.parameter_history.clear()

        self.current_estimates = {
            "frequency": np.mean(self.prior_bounds),
            "linewidth": np.mean(self.linewidth_prior),
            "amplitude": np.mean(self.amplitude_prior),
            "background": np.mean(self.background_prior),
            "uncertainty": np.inf,
            "gaussian_width": np.mean(self.gaussian_width_prior),
            "split": np.mean(self.split_prior),
            "k_np": np.mean(self.k_np_prior),
        }

    def _history_iter(self, history: Sequence | pl.DataFrame) -> Iterable[dict[str, float]]:
        """Iterate over history entries, normalizing format."""
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
        """Ingest measurement history, updating internal state."""
        for row in self._history_iter(history):
            is_new = not any(existing["x"] == row["x"] for existing in self.measurement_history)
            if is_new:
                self.measurement_history.append(row)
                # Update estimates based on a new measurement
                self._update_estimates(row)

                # Auto-track history for non-Bayesian locators (Bayesian track themselves)
                if "Bayesian" not in self.__class__.__name__:
                    self.parameter_history.append(self.current_estimates.copy())

    def _update_estimates(self, measurement: dict[str, float]) -> None:
        """Update current estimates based on a new measurement.

        Base implementation does simple averaging. Subclasses can override for
        more sophisticated updates (e.g., Bayesian posterior updates).
        """
        # Simple implementation: just track that we have a measurement
        # Subclasses override for more sophisticated estimation
        pass

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

    @abstractmethod
    def _propose_measurement(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        """Propose the next measurement point (after warmup).

        Subclasses must implement their specific acquisition strategy.

        Args:
            history: Measurement history as DataFrame
            scan: Scan batch information

        Returns:
            Next frequency to measure
        """
        raise NotImplementedError

    def propose_next(
        self,
        history: Sequence | pl.DataFrame,
        scan: ScanBatch | None = None,
        repeats: pl.DataFrame | None = None,
    ) -> float | pl.DataFrame:
        """Propose the next measurement point."""
        # Handle argument swapping if called as (history, repeats, scan)
        if isinstance(scan, pl.DataFrame) and repeats is not None and not isinstance(repeats, pl.DataFrame):
            real_repeats = scan
            real_scan = repeats
            scan = real_scan
            repeats = real_repeats
        elif isinstance(repeats, ScanBatch) and scan is None:
            pass

        if repeats is not None:
            # For now, if we don't have a proper batched implementation,
            # we assume single-instance behavior (valid for repeats=1)
            # or return a dummy stop for all.
            # Ideally this class should not be used directly in batched mode without an adapter.
            # But to prevent crashing:
            return repeats.select("repeat_id").with_columns(pl.lit(0.0).alias("x"))

        # Convert to DataFrame if needed
        if not isinstance(history, pl.DataFrame):
            history = pl.DataFrame(history) if history else pl.DataFrame()

        hist_len = history.height

        self._rescale_priors_if_needed(scan)
        self._ingest_history(history)

        # Warmup phase: use sweep locator
        if hist_len < self.n_warmup:
            dummy_repeats = pl.DataFrame({"repeat_id": [0], "active": [True]})
            history_with_id = history.with_columns(pl.lit(0, dtype=pl.Int64).alias("repeat_id"))
            proposals = self.sweeper.propose_next(history_with_id, dummy_repeats, scan)

            if not proposals.is_empty():
                return proposals.item(0, "x")

            # Fallback if sweeper gives no proposal
            return (scan.x_min + scan.x_max) / 2.0 if scan else np.mean(self.prior_bounds)

        # Post-warmup: delegate to subclass strategy
        return float(self._propose_measurement(history, scan))

    def should_stop(
        self,
        history: Sequence | pl.DataFrame,
        repeats: pl.DataFrame | None = None,
        scan: ScanBatch | None = None,
    ) -> bool | pl.DataFrame:
        """Check if locator should stop acquiring measurements."""
        if isinstance(repeats, ScanBatch) and scan is None:
            scan = repeats
            repeats = None

        # Handle batched return if repeats is provided
        if repeats is not None:
            # For now, if we don't have a proper batched implementation,
            # we assume single-instance behavior (valid for repeats=1)
            # or return a dummy stop for all.
            # Ideally this class should not be used directly in batched mode without an adapter.
            # But to prevent crashing:
            return repeats.select("repeat_id").with_columns(pl.lit(False).alias("stop"))

        hist_len = history.height if isinstance(history, pl.DataFrame) else len(history)
        if hist_len >= self.max_evals:
            return True
        return self.current_estimates["uncertainty"] < self.convergence_threshold

    def finalize(
        self,
        history: Sequence | pl.DataFrame,
        repeats: pl.DataFrame | None = None,
        scan: ScanBatch | None = None,
    ) -> dict[str, float] | pl.DataFrame:
        """Finalize and return estimated peak positions."""
        if isinstance(repeats, ScanBatch) and scan is None:
            scan = repeats
            repeats = None

        if repeats is not None:
            # Fallback for batched call - return empty/dummy results
            # This is a hack to allow repeats=1 to work if the caller passes repeats
            # But really, for repeats=1, the caller might treat it as single if we return dict?
            # No, cli.py expects DataFrame if it passed repeats.

            # If repeats=1, we can try to compute the result using the single logic
            if repeats.height == 1:
                # Ingest history (assuming it's filtered or we just take it all)
                self._ingest_history(history)
                res = self.finalize(history, scan=scan)  # Call single version
                return repeats.select("repeat_id").with_columns(
                    pl.lit(res.get("n_peaks", 1.0)).alias("n_peaks"),
                    pl.lit(res.get("x1_hat", math.nan)).alias("x1_hat"),
                    pl.lit(res.get("x2_hat", math.nan)).alias("x2_hat"),
                    pl.lit(res.get("x3_hat", math.nan)).alias("x3_hat"),
                    pl.lit(res.get("uncert", math.inf)).alias("uncert"),
                    pl.lit(res.get("uncert_pos", math.inf)).alias("uncert_pos"),
                    pl.lit(res.get("measurements", 0)).alias("measurements"),
                )

            return repeats.select("repeat_id").with_columns(
                pl.lit(1.0).alias("n_peaks"),
                pl.lit(math.nan).alias("x1_hat"),
                pl.lit(math.nan).alias("x2_hat"),
                pl.lit(math.nan).alias("x3_hat"),
                pl.lit(math.inf).alias("uncert"),
                pl.lit(math.inf).alias("uncert_pos"),
                pl.lit(0).alias("measurements"),
            )

        self._ingest_history(history)

        # Extract peak positions from current estimates
        uncertainty = float(self.current_estimates["uncertainty"])
        frequency = float(self.current_estimates["frequency"])

        # Check if we have multiple peaks based on distribution
        if self.distribution == "voigt-zeeman":
            split = float(self.current_estimates["split"])
            return {
                "n_peaks": 3.0,
                "x1_hat": frequency - split,
                "x2_hat": frequency,
                "x3_hat": frequency + split,
                "uncert": uncertainty,
                "uncert_pos": uncertainty,
            }
        else:
            return {
                "n_peaks": 1.0,
                "x1_hat": frequency,
                "x2_hat": math.nan,
                "x3_hat": math.nan,
                "uncert": uncertainty,
                "uncert_pos": uncertainty,
            }
