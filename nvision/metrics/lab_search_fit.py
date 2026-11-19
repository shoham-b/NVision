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
# 4 added the van Trees fit from detection ("detection"); 5 aligned each run at its own detection; 6 detects the dip
# by the run's first steep, lasting drop of sigma instead of a sigma level.
SEARCH_FIT_VERSION = 6

# A one-step drop of ln sigma counts as steep when it is at least this fraction of the run's steepest one-step drop
# (detection_step). 0.7 is where the lab runs' detection steps stop moving with it and the van Trees fit after
# detection is best; lower values detect on an earlier step of a wide window's staircase.
DETECTION_STEEP_FRACTION = 0.7


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


def fit_search_curve(
    sigma: np.ndarray, window_mhz: float | None = None, run_sigmas: list[np.ndarray] | None = None
) -> dict:
    """Fit the model to a claimed-sigma history, least squares on log sigma.

    *sigma*: shape (n_steps,), MHz, claimed sigma after step 1..n_steps, all positive. *window_mhz*: the search
    window ``W`` when known (the drive window of a lab run); otherwise the prior spread of Eq. ``prior``,
    ``W = sqrt(12) * sigma[0]``. *run_sigmas*: every run's own claimed-sigma history (MHz, each shape
    (n_steps_run,), lengths may differ) for the van Trees fit from detection; a single run passes ``[sigma]``, and
    None means no detection fit. Returns ``version``, ``window_mhz``, ``v_mhz_per_step``, ``j_d_per_mhz2``,
    ``i_max_per_mhz2``, ``search_end_step`` (``W / v``), ``rms_log`` (rms of ``ln(model/data)``), ``model``,
    shape (n_steps,), MHz, and ``detection`` (:func:`fit_van_trees_from_run_detection`, None without *run_sigmas*).
    Raises if the fit fails.
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
        "detection": None if run_sigmas is None else fit_van_trees_from_run_detection(run_sigmas),
    }


def detection_step(sigma: np.ndarray) -> int | None:
    """One run's detection: the step (1-based) right after its first steep, lasting one-step drop of sigma.

    *sigma*: shape (n_steps,), the run's claimed sigma after each measurement, all positive. A drop is steep when
    ``ln sigma`` falls by at least ``DETECTION_STEEP_FRACTION`` of the run's steepest one-step fall, and lasting when
    sigma never later climbs back above the geometric mean of its values before and after the drop (a candidate the
    run settled on and then rejected does not count). Uses only the run's own history: no sigma level, no window.
    None if sigma never falls or no steep drop lasts.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 1 or sigma.size < 2 or not np.all(sigma > 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be a positive, finite 1-D history of at least 2 steps")
    log_s = np.log(sigma)
    drops = np.diff(log_s)  # drops[k]: change of ln sigma from step k+1 to step k+2, shape (n_steps - 1,)
    steepest = drops.min()
    if steepest >= 0:
        return None
    for k in np.flatnonzero(drops <= DETECTION_STEEP_FRACTION * steepest):
        if np.all(log_s[k + 1 :] <= 0.5 * (log_s[k] + log_s[k + 1])):
            return int(k) + 2
    return None


def fit_van_trees_from_run_detection(run_sigmas: list[np.ndarray]) -> dict | None:
    """Van Trees form ``(J_d + k I_max)^-1/2`` fitted to the runs aligned at their own detection, ``k = n - n_d``.

    Each run's ``n_d`` is :func:`detection_step` of its own history, so the median at each lag ``k`` is not smeared
    by when the runs detected (the median over runs at a fixed step mixes runs still searching with runs that already
    found the dip). *run_sigmas*: per-run histories in MHz, each shape (n_steps_run,). A lag is used while at least
    half of the detected runs still have a step there. ``J_d`` is not fitted but pinned to the aligned median at
    detection, ``J_d = 1 / sigma(k=0)^2``, so the curve starts on the data; only ``I_max`` is fitted, to the later lags.

    Returns ``step`` (median ``n_d`` over the detected runs, where the curve is drawn from), ``step_q25`` /
    ``step_q75``, ``n_detected`` / ``n_runs``, ``j_d_per_mhz2``, ``i_max_per_mhz2``, ``rms_log``, and plain lists over
    the lags: ``steps`` (``step + k``), ``aligned_median`` (median sigma at each lag, MHz) and ``model`` (MHz). None
    when no run detects or fewer than 3 lags remain (nothing to fit).
    """
    histories = [np.asarray(s, dtype=float) for s in run_sigmas]
    if not histories or any(h.ndim != 1 for h in histories):
        raise ValueError("run_sigmas must be a non-empty list of 1-D histories")
    detected = [(n_d, h) for h in histories if (n_d := detection_step(h)) is not None]
    if not detected:
        return None
    n_ds = np.array([n_d for n_d, _ in detected], dtype=float)
    tails = [h[n_d - 1 :] for n_d, h in detected]  # each run from its own detection on: lag k = 0, 1, ...
    min_runs = int(np.ceil(len(tails) / 2))
    n_lags = 0
    while sum(t.size > n_lags for t in tails) >= min_runs:
        n_lags += 1
    if n_lags < 3:
        return None
    aligned = np.array([np.median([t[k] for t in tails if t.size > k]) for k in range(n_lags)])  # (n_lags,)
    lags = np.arange(n_lags, dtype=float)
    log_data = np.log(aligned)
    j_d = float(1.0 / aligned[0] ** 2)  # pinned: the curve passes through the data at detection

    def residual(p: np.ndarray) -> np.ndarray:
        return -0.5 * np.log(j_d + lags * np.exp(p[0])) - log_data

    best = None
    for i0 in (0.1 * j_d, j_d, 10.0 * j_d):
        res = least_squares(residual, x0=np.log([i0]))
        if res.success and (best is None or res.cost < best.cost):
            best = res
    if best is None:
        raise RuntimeError("van Trees fit from detection did not converge")
    i_max = float(np.exp(best.x[0]))
    step = round(float(np.median(n_ds)))
    return {
        "step": step,
        "step_q25": float(np.quantile(n_ds, 0.25)),
        "step_q75": float(np.quantile(n_ds, 0.75)),
        "n_detected": len(detected),
        "n_runs": len(histories),
        "j_d_per_mhz2": j_d,
        "i_max_per_mhz2": i_max,
        "rms_log": float(np.sqrt(np.mean(best.fun**2))),
        "steps": (step + lags).astype(int).tolist(),
        "aligned_median": aligned.tolist(),
        "model": (1.0 / np.sqrt(j_d + lags * i_max)).tolist(),
    }


def is_current(fit: dict | None) -> bool:
    """Whether a stored fit is from the current model (see ``SEARCH_FIT_VERSION``)."""
    return fit is not None and fit.get("version") == SEARCH_FIT_VERSION
