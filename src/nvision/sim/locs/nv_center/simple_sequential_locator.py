"""Simple sequential locator for NV centers using deterministic point selection."""

from __future__ import annotations

import math
import warnings

import polars as pl

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center._base_locator import NVCenterLocatorBase


class SimpleSequentialLocator(NVCenterLocatorBase):
    """A simplified sequential locator that selects measurement points based on
    characteristic features of the estimated distribution (peaks ± gamma/sqrt(3)).

    This locator uses a deterministic strategy instead of Monte Carlo simulations,
    selecting points at the inflection points of Lorentzian peaks.
    """

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

        # Ensure within bounds
        selected_freq = max(domain_low, min(selected_freq, domain_high))

        return selected_freq
