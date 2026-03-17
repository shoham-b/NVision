"""Stateless sweep locator for NV centers using v2 architecture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.v2.base import Locator


@dataclass
class NVCenterSweepLocatorV2(Locator):
    """Stateless sweep locator for NV center signals with potential multiple peaks.

    Two-phase strategy:
    1. Coarse scan: uniform grid across the domain
    2. Refinement: focus around detected peaks

    This implementation is fully stateless - the same instance can be safely
    reused across multiple repeats. All state is reconstructed from the history
    DataFrame on each call.
    """

    coarse_points: int = 30
    refine_points: int = 10
    min_separation_frac: float = 0.03

    def next(self, history: pl.DataFrame) -> float:
        """Propose next measurement point based on history.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with columns ['x', 'signal_value']

        Returns
        -------
        float
            Next position to measure (in range [0, 1])
        """
        n = history.height

        if n < self.coarse_points:
            # Coarse scan phase: uniform grid from 0 to 1
            return float(np.linspace(0, 1, self.coarse_points)[n])

        # Refinement phase: focus around detected peaks
        x_vals = history.get_column("x").to_numpy()
        y_vals = history.get_column("signal_value").to_numpy()

        peaks = self._find_peaks(x_vals, y_vals, self.min_separation_frac)

        if not peaks:
            # No peaks found, default to midpoint
            return 0.5

        # Refine around the main (strongest) peak
        main_peak = peaks[0]
        refine_step = n - self.coarse_points

        return self._get_refine_point(main_peak, 0.1, 0.0, 1.0, refine_step)

    def done(self, history: pl.DataFrame) -> bool:
        """Check if we've completed both coarse and refinement phases.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with columns ['x', 'signal_value']

        Returns
        -------
        bool
            True if both phases are complete
        """
        return history.height >= (self.coarse_points + self.refine_points)

    def result(self, history: pl.DataFrame) -> dict[str, float]:
        """Extract final peak location from history.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with columns ['x', 'signal_value']

        Returns
        -------
        dict[str, float]
            Dictionary with 'peak_x' and 'peak_signal' keys
        """
        if history.is_empty():
            return {"peak_x": float("nan"), "peak_signal": float("nan")}

        x_vals = history.get_column("x").to_numpy()
        y_vals = history.get_column("signal_value").to_numpy()

        peaks = self._find_peaks(x_vals, y_vals, self.min_separation_frac)

        if not peaks:
            # Fall back to maximum signal point
            max_idx = np.argmax(y_vals)
            return {"peak_x": float(x_vals[max_idx]), "peak_signal": float(y_vals[max_idx])}

        # Return the strongest peak
        peak_x = peaks[0]
        # Find signal value at peak
        peak_idx = np.argmin(np.abs(x_vals - peak_x))

        return {"peak_x": float(peak_x), "peak_signal": float(y_vals[peak_idx])}

    def _find_peaks(self, x: np.ndarray, y: np.ndarray, w: float) -> list[float]:
        """Find up to 3 peaks separated by at least w, prioritizing highest signal.

        Parameters
        ----------
        x : array of positions
        y : array of signal values
        w : minimum separation fraction

        Returns
        -------
        list of peak positions, sorted by signal strength (highest first)
        """
        # Sort by signal value descending to prioritize high peaks
        sorted_indices = np.argsort(-y)
        peaks: list[float] = []

        for idx in sorted_indices:
            px = float(x[idx])
            # Check if this peak is far enough from existing peaks
            if not peaks or all(abs(px - p) >= w for p in peaks):
                peaks.append(px)
            if len(peaks) >= 3:
                break

        return peaks

    def _get_refine_point(self, center: float, width: float, lo: float, hi: float, step: int) -> float:
        """Generate refinement point around peak using golden-ratio spacing.

        Parameters
        ----------
        center : peak center position
        width : initial search width
        lo, hi : domain bounds
        step : refinement step number (0, 1, 2, ...)

        Returns
        -------
        Next refinement point
        """
        radius = min(center - lo, hi - center, 2 * width)
        start = max(lo, center - radius)
        end = min(hi, center + radius)

        golden_ratio = 0.618033988749895
        offset = (end - start) * (golden_ratio ** (step % 10))

        return float(start + offset)
