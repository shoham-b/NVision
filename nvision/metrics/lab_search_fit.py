"""Fit of the article's search-phase error model (theory.tex, Eq. ``mean-curve``) to a claimed-sigma history.

The elimination model has a window ``W`` and two free parameters: the reach ``delta_d`` of one measurement and the
information ``i_max`` of one measurement at the steepest slope. Its error after ``n`` measurements, averaged over where
the dip lies, is the search term plus the restarted after-detection terms::

    var(n) = (W - n dd)^3 / (12 W)  +  dd/W * sum_{nd=1..min(n, W/dd)} (12/dd^2 + (n - nd) i_max)^-1

(the search term is zero once ``n dd >= W``). All lengths in MHz, ``i_max`` in 1/MHz^2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def search_model_sigma(steps: np.ndarray, window: float, delta_d: float, i_max: float) -> np.ndarray:
    """Model sigma (MHz) after each measurement count in *steps*.

    *steps*: shape (n_steps,), 1-based measurement counts. *window*, *delta_d*: MHz, ``0 < delta_d <= window``.
    *i_max*: 1/MHz^2, ``> 0``. Returns shape (n_steps,), MHz.
    """
    if not (0 < delta_d <= window) or i_max <= 0:
        raise ValueError(f"need 0 < delta_d <= window and i_max > 0, got {delta_d=}, {window=}, {i_max=}")
    n = np.asarray(steps, dtype=float)
    search = np.clip(window - n * delta_d, 0.0, None) ** 3 / (12.0 * window)
    n_cells = round(window / delta_d)
    found = np.zeros_like(n)
    for i, n_i in enumerate(n):
        m = int(min(n_i, n_cells))
        if m > 0:
            lags = n_i - np.arange(1, m + 1)  # n - nd, shape (m,)
            found[i] = delta_d / window * np.sum(1.0 / (12.0 / delta_d**2 + lags * i_max))
    return np.sqrt(search + found)


def fit_search_curve(sigma: np.ndarray, window_mhz: float | None = None) -> dict:
    """Fit the model to a claimed-sigma history, least squares on log sigma.

    *sigma*: shape (n_steps,), MHz, claimed sigma after step 1..n_steps, all positive. *window_mhz*: the search
    window ``W`` when known (the drive window of a lab run); otherwise the prior spread of Eq. ``prior``,
    ``W = sqrt(12) * sigma[0]``. Returns ``window_mhz``, ``delta_d_mhz``, ``i_max_per_mhz2``,
    ``rms_log`` (rms of ``ln(model/data)``) and ``model``, shape (n_steps,), MHz. Raises if the fit fails.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 1 or sigma.size < 3 or not np.all(sigma > 0):
        raise ValueError("sigma must be a positive 1-D history of at least 3 steps")
    steps = np.arange(1, sigma.size + 1, dtype=float)
    window = float(np.sqrt(12.0) * sigma[0]) if window_mhz is None else float(window_mhz)
    log_data = np.log(sigma)

    def residual(p: np.ndarray) -> np.ndarray:
        return np.log(search_model_sigma(steps, window, float(np.exp(p[0])), float(np.exp(p[1])))) - log_data

    # Late-time sigma ~ 1/sqrt(n i_max) fixes the i_max start; the reach starts at a tenth of the window.
    i0 = 1.0 / (sigma[-1] ** 2 * steps[-1])
    best = None
    for dd0 in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        for i_scale in (0.3, 1.0, 3.0):
            res = least_squares(
                residual,
                x0=np.log([dd0 * window, i0 * i_scale]),
                bounds=(np.log([window / 1000.0, 1e-12]), np.log([window, 1e12])),
            )
            if res.success and (best is None or res.cost < best.cost):
                best = res
    if best is None:
        raise RuntimeError("search-phase fit did not converge")
    delta_d, i_max = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    return {
        "window_mhz": window,
        "delta_d_mhz": delta_d,
        "i_max_per_mhz2": i_max,
        "rms_log": float(np.sqrt(np.mean(best.fun**2))),
        "model": search_model_sigma(steps, window, delta_d, i_max),
    }
