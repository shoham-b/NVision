"""Simple sequential locator for NV centers using deterministic point selection."""

from __future__ import annotations

import math
import warnings

import polars as pl
import numpy as np

from nvision.sim.locs.nv_center._jit_kernels import _lorentzian_model
from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center._base_locator import NVCenterLocatorBase


class SimpleSequentialLocator(NVCenterLocatorBase):
    """A simplified sequential locator that selects measurement points based on
    characteristic features of the estimated distribution (peaks ± gamma/sqrt(3)).

    This locator uses a deterministic strategy instead of Monte Carlo simulations,
    selecting points at the inflection points of Lorentzian peaks.
    """

    def _update_estimates(self, measurement: dict[str, float]) -> None:
        """Update estimates by fitting a Lorentzian to the history."""
        # Need at least a few points to fit
        if len(self.measurement_history) < 5:
            return

        try:
            from scipy.optimize import curve_fit
        except ImportError:
            warnings.warn("scipy.optimize.curve_fit not available", stacklevel=2)
            return

        # Prepare data
        x_data = np.array([m["x"] for m in self.measurement_history])
        y_data = np.array([m["signal_values"] for m in self.measurement_history])

        # Define model function (Lorentzian)
        def lorentzian(x, f0, gamma, amp, bg):
            return _lorentzian_model(x, f0, gamma, amp, bg)

        # Initial guesses
        # Find the point with the minimum signal (dip) to guess f0
        min_idx = np.argmin(y_data)
        f0_guess = x_data[min_idx]

        p0 = [
            f0_guess,
            self.current_estimates["linewidth"],
            self.current_estimates["amplitude"],
            self.current_estimates["background"],
        ]

        # Bounds
        domain_width = self.prior_bounds[1] - self.prior_bounds[0]
        bounds = (
            [self.prior_bounds[0], 1e3, 0.0, 0.0],
            [self.prior_bounds[1], domain_width, 2.0, 2.0],
        )

        try:
            popt, pcov = curve_fit(lorentzian, x_data, y_data, p0=p0, bounds=bounds, maxfev=1000)

            # Update estimates
            self.current_estimates["frequency"] = popt[0]
            self.current_estimates["linewidth"] = min(popt[1], domain_width)
            self.current_estimates["amplitude"] = popt[2]
            self.current_estimates["background"] = popt[3]

            # Calculate uncertainty (std dev of frequency)
            perr = np.sqrt(np.diag(pcov))
            self.current_estimates["uncertainty"] = perr[0]

        except Exception as e:
            # Fitting failed, keep previous estimates
            # log.debug(f"Fitting failed: {e}")
            pass

    def _propose_measurement(self, history: pl.DataFrame, scan: ScanBatch) -> float:
        """Select the next measurement point deterministically based on current estimates.

        Uses the formula x = μ ± γ/√3 for Lorentzian inflection points.
        """
        domain_low, domain_high = self.prior_bounds

        # Get current estimates
        f0 = self.current_estimates["frequency"]
        gamma = self.current_estimates["linewidth"]

        # Determine centers based on distribution
        centers = []
        if self.distribution in ("lorentzian", "voigt"):
            # Single peak at f0
            centers = [f0]
        elif self.distribution == "voigt-zeeman":
            # Three peaks, but we focus on the outer ones (most left and most right)
            split = self.current_estimates["split"]
            centers = [f0 - split, f0 + split]
        else:
            # Fallback for unknown distributions (treat as single peak)
            warnings.warn(
                f"Unknown distribution '{self.distribution}', treating as single peak.",
                stacklevel=2,
            )
            centers = [f0]

        # Calculate candidate points: center +/- gamma/sqrt(3)
        # gamma/sqrt(3) approx 0.577 * gamma
        offset = gamma / math.sqrt(3)
        candidates = []
        for c in centers:
            candidates.append(c - offset)
            candidates.append(c + offset)

        # Select one candidate based on history length (round-robin)
        # We use the length of measurement_history to cycle through candidates
        idx = len(self.measurement_history) % len(candidates)
        selected_freq = candidates[idx]

        # Add jitter to avoid measuring the exact same points repeatedly
        # Jitter is +/- 5% of linewidth
        jitter = (np.random.random() - 0.5) * 0.1 * gamma
        selected_freq += jitter

        # Ensure within bounds
        selected_freq = max(domain_low, min(selected_freq, domain_high))

        return selected_freq


# Alias for backward compatibility
SimpleSequentialLocatorBatched = SimpleSequentialLocator
