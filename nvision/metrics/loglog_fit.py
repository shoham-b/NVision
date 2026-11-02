"""Segmented fit of a claimed-uncertainty curve in log-log: the shape behind the three stages.

A locator's claimed uncertainty ``sigma(n)`` against the number of measurements ``n`` is, when everything goes right,
three straight lines on log-log axes:

1. **search**: flat (nothing is known yet, ``sigma`` stays at the prior width);
2. **focus**: a steep fall (the belief collapses onto the dip);
3. **refine**: slope ``-1/2``, because ``sigma ~ 1/sqrt(n)`` once every measurement adds Fisher information (CRLB).

Not every run has all three (a run may never find the dip, stop on the cliff, or level off like a plain sweep), so
:func:`fit_segments` does not assume them: it fits continuous hinges with 0, 1 or 2 breakpoints
(``y = a + s1 x + c1 (x - k1)+ + c2 (x - k2)+``, ``x = ln n``, ``y = ln sigma``; breakpoints searched on a grid, the
rest solved by least squares, so the result is deterministic) and keeps the simplest model the data support: extra
breakpoints must lower the BIC *and* change the slope by at least ``MIN_SLOPE_CHANGE``. The last segment's slope is left
free; its distance from ``-1/2`` is the diagnostic, and a refine stage exists only if that segment lies in
``REFINE_SLOPE_BAND`` and spans ``REFINE_MIN_DECADES``.

The fit uses the whole curve, so it is **not causal**: unlike the boundaries in :mod:`nvision.metrics.stages` it
cannot be read off while the run is going. It describes a finished run (or the median curve of many).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Points of the log-spaced resampling grid. Raw steps put thousands of points in the tail and a handful in the search
# plateau, so the tail would dictate the fit; a log grid weights every decade equally.
FIT_GRID_POINTS = 48
# Fewest grid points in each segment (a segment needs more than its two defining points).
MIN_SEGMENT_POINTS = 4
# A breakpoint counts only if the slope changes by at least this much across it (the fit is in ln sigma vs ln n).
MIN_SLOPE_CHANGE = 0.15
# Residual floor per grid point (ln units squared) so a noiseless curve does not send the BIC to -infinity.
RSS_FLOOR_PER_POINT = 1e-6
# Refine is "reached" when the last segment's slope lies in this band around -1/2 ...
REFINE_SLOPE_BAND = (-0.8, -0.25)
# ... and that segment covers at least this many decades of measurements (shorter is a corner, not a segment).
REFINE_MIN_DECADES = 0.3
THEORY_SLOPE = -0.5


def _resample_log_grid(n: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(ln n, ln sigma)`` on ``FIT_GRID_POINTS`` log-spaced ``n``, holding the last value (a step function).

    ``n``: shape ``(n_points,)``, measurement counts (>= 1, increasing). ``sigma``: shape ``(n_points,)``, claimed
    uncertainty in Hz (> 0). Returns two arrays of shape ``(<= FIT_GRID_POINTS,)``, duplicates dropped.
    """
    grid = np.unique(np.round(np.geomspace(n[0], n[-1], FIT_GRID_POINTS)))
    idx = np.searchsorted(n, grid, side="right") - 1
    return np.log(grid), np.log(sigma[idx])


def _knot_candidates(n_grid: int, n_knots: int) -> np.ndarray:
    """Grid indices of every admissible breakpoint set, shape ``(n_sets, n_knots)``; segments keep enough points."""
    m = MIN_SEGMENT_POINTS
    if n_knots == 0:
        return np.empty((1, 0), dtype=int)
    if n_knots == 1:
        return np.arange(m - 1, n_grid - m + 1)[:, None]
    return np.array([(i, j) for i in range(m - 1, n_grid - 2 * m + 1) for j in range(i + m, n_grid - m + 1)])


def _fit_hinge(x: np.ndarray, y: np.ndarray, n_knots: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Best continuous hinge with ``n_knots`` breakpoints on grid points: ``(rss, knot indices, coefficients)``.

    ``x``, ``y``: shape ``(n_grid,)``. Coefficients, shape ``(2 + n_knots,)``: ``a, s1, c1, ...`` (the slope change at
    each breakpoint). Every breakpoint set is solved at once by batched normal equations, ``(n_sets, n_grid, p)`` ->
    ``(n_sets, p, p)``.
    """
    knots = _knot_candidates(x.size, n_knots)
    ones = np.ones((len(knots), x.size))
    cols = [ones, np.broadcast_to(x, ones.shape)]
    cols += [np.clip(x[None, :] - x[knots[:, k : k + 1]], 0.0, None) for k in range(n_knots)]
    design = np.stack(cols, axis=-1)
    gram = np.einsum("pgk,pgl->pkl", design, design)
    rhs = np.einsum("pgk,g->pk", design, y)
    coefs = np.linalg.solve(gram, rhs[..., None])[..., 0]
    rss_all = np.sum((np.einsum("pgk,pk->pg", design, coefs) - y) ** 2, axis=1)
    best = int(np.argmin(rss_all))
    return float(rss_all[best]), knots[best], coefs[best]


def _bic(rss: float, n_points: int, n_knots: int) -> float:
    """Bayesian information criterion of a hinge (parameters: ``a``, ``s1``, and a knot plus a slope change each)."""
    rss_eff = max(rss, RSS_FLOOR_PER_POINT * n_points)
    return n_points * math.log(rss_eff / n_points) + (2 + 2 * n_knots) * math.log(n_points)


def fit_segments(n: Any, sigma: Any) -> dict[str, Any] | None:
    """Fit the simplest 1-, 2- or 3-segment log-log line the claimed uncertainty ``sigma`` (Hz) vs ``n`` supports.

    ``n``: shape ``(n_points,)``, measurement counts; ``sigma``: shape ``(n_points,)``, claimed uncertainty (Hz).
    Non-finite and non-positive points are dropped. Returns ``None`` when fewer than ``2 * MIN_SEGMENT_POINTS``
    distinct log-grid points remain, else a dict:

    * ``n_segments``: 1, 2 or 3, the segments the data support (see the module docstring).
    * ``breaks_n``: measurement counts of the ``n_segments - 1`` breakpoints, in order.
    * ``slopes``: log-log slope of each segment (``n_segments`` values; the last is the tail).
    * ``a``, ``s1``, ``c1``, ``c2``, ``k1``, ``k2``: the hinge in ``x = ln n`` (``c`` = slope change at each breakpoint,
      zero for a breakpoint that does not exist), so the curve is drawn without refitting.
    * ``tail_decades``: ``log10(n_last / last breakpoint)`` (the whole span for one segment).
    * ``refine``: ``n_segments >= 2``, the last slope is in ``REFINE_SLOPE_BAND`` and ``tail_decades >=
      REFINE_MIN_DECADES``. A single straight line is not a refine stage, whatever its slope.
    * ``rss``: residual sum of squares of ``ln sigma`` on the grid.
    """
    n_arr = np.asarray(n, dtype=float)
    s_arr = np.asarray(sigma, dtype=float)
    if n_arr.shape != s_arr.shape or n_arr.ndim != 1:
        raise ValueError(f"n and sigma must be 1-D arrays of equal shape, got {n_arr.shape} and {s_arr.shape}")
    ok = np.isfinite(n_arr) & np.isfinite(s_arr) & (n_arr >= 1) & (s_arr > 0)
    n_arr, s_arr = n_arr[ok], s_arr[ok]
    if n_arr.size < 2 or n_arr[-1] <= n_arr[0]:
        return None
    order = np.argsort(n_arr, kind="stable")
    x, y = _resample_log_grid(n_arr[order], s_arr[order])
    if x.size < 2 * MIN_SEGMENT_POINTS:
        return None

    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None  # (bic, n_knots, rss, knots, coef)
    for n_knots in (0, 1, 2):
        if x.size < (n_knots + 1) * MIN_SEGMENT_POINTS:
            break
        rss, knots, coef = _fit_hinge(x, y, n_knots)
        if n_knots and np.any(np.abs(coef[2:]) < MIN_SLOPE_CHANGE):
            continue  # a breakpoint without a real change of slope is not a stage boundary
        bic = _bic(rss, x.size, n_knots)
        if best is None or bic < best[0]:
            best = (bic, n_knots, rss, knots, coef)
    assert best is not None  # the straight line is always admissible
    _, n_knots, rss, knots, coef = best

    a, s1 = float(coef[0]), float(coef[1])
    changes = [float(v) for v in coef[2:]] + [0.0] * (2 - n_knots)
    k: list[float | None] = [float(x[i]) for i in knots] + [None] * (2 - n_knots)
    slopes = [s1 + sum(changes[:s]) for s in range(n_knots + 1)]
    last_start = float(k[n_knots - 1]) if n_knots else float(x[0])  # type: ignore[arg-type]
    tail_decades = (float(x[-1]) - last_start) / math.log(10.0)
    return {
        "n_segments": n_knots + 1,
        "breaks_n": [math.exp(v) for v in k[:n_knots]],  # type: ignore[arg-type]
        "slopes": slopes,
        "a": a,
        "s1": s1,
        "c1": changes[0],
        "c2": changes[1],
        "k1": k[0],
        "k2": k[1],
        "tail_decades": tail_decades,
        "refine": bool(
            n_knots >= 1
            and REFINE_SLOPE_BAND[0] <= slopes[-1] <= REFINE_SLOPE_BAND[1]
            and tail_decades >= REFINE_MIN_DECADES
        ),
        "rss": rss,
    }
