"""Fit of the article's mean error curve (SM Eq. ``mean-curve``) to a claimed-sigma history.

The elimination model has a window ``W`` and three free parameters, as in the article: the search rate ``v`` (window
cleared per measurement, Eq. ``search-rate``), ``J_d``, the inverse of the mean error at detection (Eq.
``after-detection``, ``12 / Delta d^2`` in the article; fitted here, so an effective value, not the Fisher information
``J_tau`` of the posterior), and the information ``I_max`` of one measurement at the steepest slope. A fraction
``v / W`` of the runs detects the dip at each measurement ``n_d``, and from its own detection on each follows the van
Trees form ``(J_d + (n - n_d) I_max)^-1``. Averaged over where the dip lies::

    var(n) = (W - n v)^3 / (12 W)  +  sum_{n_d=1..min(n, N)} w_{n_d} / (J_d + (n - n_d) I_max)

with ``N = ceil(W / v)`` detections, ``w_{n_d} = v / W`` and the last one the remainder so the weights sum to one
(the search term is zero once ``n v >= W``). All lengths in MHz, ``J_d`` and ``I_max`` in 1/MHz^2.

The search rate and the information at detection are separate parameters: a search that needs several measurements
per cell (low SNR, ``m > 1``) is slow without the dip being pinned to a narrow cell when it is found.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# Stored fits (window*MHz_search_fit.json, a MATLAB record's search_fit) carry this; a fit without it is from the
# older models (version 2 called J_d "j_tau"; before that v = delta_d and J_d = 12 / delta_d^2) and is refitted
# when read.
SEARCH_FIT_VERSION = 4  # 4 adds the van Trees fit from the detection step ("detection")

# A claimed sigma below this fraction of the prior width W / sqrt(12) counts as detected (detection_step).
DETECTION_FRACTION = 0.1


def detection_weights(window: float, v: float) -> np.ndarray:
    """Share of the runs that detect the dip at measurement ``n_d = 1..N``, ``N = ceil(W / v)``.

    Returns shape (N,), summing to 1: ``v / W`` each, the last one the remainder.
    """
    n_det = int(np.ceil(window / v - 1e-9))
    w = np.full(n_det, v / window)
    w[-1] = 1.0 - v / window * (n_det - 1)
    return w


def search_model_sigma(steps: np.ndarray, window: float, v: float, j_d: float, i_max: float) -> np.ndarray:
    """Model sigma (MHz) after each measurement count in *steps*.

    *steps*: shape (n_steps,), 1-based measurement counts. *window*, *v*: MHz and MHz per measurement,
    ``0 < v <= window``. *j_d*, *i_max*: 1/MHz^2, ``> 0``. Returns shape (n_steps,), MHz.
    """
    if not (0 < v <= window) or j_d <= 0 or i_max <= 0:
        raise ValueError(f"need 0 < v <= window, j_d > 0, i_max > 0, got {v=}, {window=}, {j_d=}, {i_max=}")
    n = np.asarray(steps, dtype=float)
    search = np.clip(window - n * v, 0.0, None) ** 3 / (12.0 * window)
    w = detection_weights(window, v)  # shape (N,)
    lags = n[:, None] - np.arange(1, w.size + 1)[None, :]  # n - n_d, shape (n_steps, N)
    vt = np.where(lags >= 0, w[None, :] / (j_d + np.clip(lags, 0.0, None) * i_max), 0.0)
    return np.sqrt(search + vt.sum(axis=1))


def fit_search_curve(sigma: np.ndarray, window_mhz: float | None = None) -> dict:
    """Fit the model to a claimed-sigma history, least squares on log sigma.

    *sigma*: shape (n_steps,), MHz, claimed sigma after step 1..n_steps, all positive. *window_mhz*: the search
    window ``W`` when known (the drive window of a lab run); otherwise the prior spread of Eq. ``prior``,
    ``W = sqrt(12) * sigma[0]``. Returns ``version``, ``window_mhz``, ``v_mhz_per_step``, ``j_d_per_mhz2``,
    ``i_max_per_mhz2``, ``search_end_step`` (``W / v``), ``rms_log`` (rms of ``ln(model/data)``), ``model``,
    shape (n_steps,), MHz, and ``detection`` (:func:`fit_van_trees_from_detection`). Raises if the fit fails.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 1 or sigma.size < 3 or not np.all(sigma > 0):
        raise ValueError("sigma must be a positive 1-D history of at least 3 steps")
    steps = np.arange(1, sigma.size + 1, dtype=float)
    window = float(np.sqrt(12.0) * sigma[0]) if window_mhz is None else float(window_mhz)
    log_data = np.log(sigma)

    def residual(p: np.ndarray) -> np.ndarray:
        v, j_d, i_max = np.exp(p)
        return np.log(search_model_sigma(steps, window, float(v), float(j_d), float(i_max))) - log_data

    # Late-time sigma ~ 1/sqrt(n I_max) fixes the I_max start; J_d starts at a cell of a tenth or a hundredth of W.
    i0 = 1.0 / (sigma[-1] ** 2 * steps[-1])
    lo = np.log([window / 1000.0, 12.0 / window**2, 1e-12])
    hi = np.log([window, 1e12, 1e12])
    best = None
    for v0 in (0.3, 0.05, 0.01, 0.003):
        for cell in (0.1, 0.01):
            for i_scale in (0.3, 3.0):
                x0 = np.clip(np.log([v0 * window, 12.0 / (cell * window) ** 2, i0 * i_scale]), lo + 1e-9, hi - 1e-9)
                res = least_squares(residual, x0=x0, bounds=(lo, hi))
                if res.success and (best is None or res.cost < best.cost):
                    best = res
    if best is None:
        raise RuntimeError("search-phase fit did not converge")
    v, j_d, i_max = (float(x) for x in np.exp(best.x))
    return {
        "version": SEARCH_FIT_VERSION,
        "window_mhz": window,
        "v_mhz_per_step": v,
        "j_d_per_mhz2": j_d,
        "i_max_per_mhz2": i_max,
        "search_end_step": window / v,
        "rms_log": float(np.sqrt(np.mean(best.fun**2))),
        "model": search_model_sigma(steps, window, v, j_d, i_max),
        "detection": fit_van_trees_from_detection(sigma, window),
    }


def detection_step(sigma: np.ndarray, window: float) -> int | None:
    """First step (1-based) at which *sigma* (shape (n_steps,), MHz) falls below ``DETECTION_FRACTION`` of the prior
    width ``W / sqrt(12)``; None if it never does. On the median claimed sigma this is the median run's detection.
    """
    below = np.flatnonzero(np.asarray(sigma, dtype=float) < DETECTION_FRACTION * window / np.sqrt(12.0))
    return int(below[0]) + 1 if below.size else None


def fit_van_trees_from_detection(sigma: np.ndarray, window: float) -> dict | None:
    """Van Trees form ``(J_d + (n - n_d) I_max)^-1/2`` fitted only to the steps from the detection step on.

    *sigma*: shape (n_steps,), MHz. Returns ``step`` (n_d), ``j_d_per_mhz2``, ``i_max_per_mhz2``, ``rms_log`` and
    ``steps`` / ``model``, plain lists of the steps ``n >= n_d`` and the curve there (MHz). None when sigma never falls
    to the detection level or fewer than 3 steps follow it (nothing to fit).
    """
    sigma = np.asarray(sigma, dtype=float)
    n_d = detection_step(sigma, window)
    if n_d is None or sigma.size - n_d + 1 < 3:
        return None
    steps = np.arange(n_d, sigma.size + 1, dtype=float)
    log_data = np.log(sigma[n_d - 1 :])

    def residual(p: np.ndarray) -> np.ndarray:
        j_d, i_max = np.exp(p)
        return -0.5 * np.log(j_d + (steps - n_d) * i_max) - log_data

    best = None
    for j0 in (1.0 / sigma[n_d - 1] ** 2, 10.0 / sigma[n_d - 1] ** 2):
        for i0 in (0.1, 3.0):
            res = least_squares(residual, x0=np.log([j0, i0]))
            if res.success and (best is None or res.cost < best.cost):
                best = res
    if best is None:
        raise RuntimeError("van Trees fit from detection did not converge")
    j_d, i_max = (float(x) for x in np.exp(best.x))
    return {
        "step": n_d,
        "j_d_per_mhz2": j_d,
        "i_max_per_mhz2": i_max,
        "rms_log": float(np.sqrt(np.mean(best.fun**2))),
        "steps": steps.astype(int).tolist(),
        "model": (1.0 / np.sqrt(j_d + (steps - n_d) * i_max)).tolist(),
    }


def is_current(fit: dict | None) -> bool:
    """Whether a stored fit is from the current model (see ``SEARCH_FIT_VERSION``)."""
    return fit is not None and fit.get("version") == SEARCH_FIT_VERSION
