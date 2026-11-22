"""Refine rates and the speedup over a sweep: the maths behind the Information tab's "Refine & speedup" section.

Pure array functions, no cache or run access (see :mod:`nvision.metrics.refine_rates_runs` for that).

The model (``articles/sections/theory.tex``, Sec. "Speedup over a drive-frequency sweep"). With ``I_max`` the Fisher
information one measurement carries at the steepest slope (``i_max``, per measurement, Hz^-2), a locator
gains information at ``q * I_max`` per measurement after a lag ``n0``::

    1 / mean(err^2)(n)  =  q * I_max * (n - n0)            (err: true error of the primary parameter, Hz)

Two different things are called ``q`` and the gap between them is the point:

* ``q_place`` = mean oracle efficiency ``eta_t = g(x_t)^2 / max g^2`` of the measurements after the dip was detected.
  It is exact (from where the locator measured, with the true signal) and says how well the *placement* is chosen.
* ``q_err`` = the slope of ``1 / mean(err^2)`` in units of ``I_max``. It is what the *estimate* gained. The gap
  ``q_err / q_place`` is what the particle filter loses (an unbiased, efficient estimator would give 1).

A sweep spreads its measurements uniformly over the window ``W``; for a single Lorentzian line of FWHM ``Gamma``
``q_sweep = I_sweep / I_max = 8 pi / 27 * Gamma / W`` and the speedup of an adaptive locator that always measures at
the steepest slope is ``R_inf = 27 / (8 pi) * W / Gamma`` (``lorentzian_sweep_q`` / ``lorentzian_speedup``);
:func:`exact_sweep_ratio` is the same ratio ``mean(g^2) / max(g^2)`` for any line shape (Voigt, hyperfine triplet) from
the sensitivity on a grid, which is the reference to compare with whenever the line is not a single Lorentzian.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

# Where a Lorentzian's squared slope is largest, in units of its HWHM gamma: d/dx of 1 / (1 + (x/gamma)^2) peaks at
# |x| = gamma / sqrt(3). Greedy EIG measures there once the line is found (theory.tex, Sec. eig).
LORENTZIAN_SLOPE_OFFSET_GAMMA = 1.0 / math.sqrt(3.0)

# A grid point of the late-part fit needs at least this share of a use case's repeats still running.
MIN_ALIVE_SHARE = 0.5


def lorentzian_sweep_q(fwhm_phys: float, window_phys: float) -> float:
    """Closed form ``q_sweep = I_sweep / I_max = 8 pi / 27 * Gamma / W`` of a single Lorentzian (W >> Gamma).

    ``fwhm_phys``: Gamma, the FWHM of the line (Hz). ``window_phys``: W, the swept drive-frequency range (Hz).
    """
    if not (fwhm_phys > 0.0 and window_phys > 0.0):
        raise ValueError(f"Need a positive FWHM and window, got {fwhm_phys!r} and {window_phys!r}.")
    return 8.0 * math.pi / 27.0 * fwhm_phys / window_phys


def lorentzian_speedup(fwhm_phys: float, window_phys: float) -> float:
    """Closed form ``R_inf = 27 / (8 pi) * W / Gamma`` (about 1.07 W / Gamma) of a single Lorentzian."""
    return 1.0 / lorentzian_sweep_q(fwhm_phys, window_phys)


def exact_sweep_ratio(g2_grid: np.ndarray) -> tuple[float, float]:
    """``(q_sweep, R_inf)`` = ``(mean(g^2) / max(g^2), max / mean)`` of any line shape.

    ``g2_grid``: shape (n_grid,), squared sensitivity ``(d signal / d primary)^2`` of the true signal on an evenly
    spaced grid of the whole drive window, (signal / Hz)^2. The mean over an even grid is the average a uniform sweep
    takes; this is ``I_sweep / I_max`` for that signal, with no Lorentzian assumption.
    """
    g2 = np.asarray(g2_grid, dtype=np.float64)
    if g2.ndim != 1 or g2.size < 2 or not float(g2.max()) > 0.0:
        raise ValueError(f"g2_grid must be a 1-D array with a positive maximum, got shape {g2.shape}.")
    q = float(g2.mean() / g2.max())
    return q, 1.0 / q


def sweep_n_points(window_phys: float, fwhm_prior_phys: float, spacing_divisor: float) -> int:
    """Points of one pass of a ``SimpleSweep-{spacing_divisor}perFWHM`` over the window (ends included).

    The spacing is ``fwhm_prior_phys / spacing_divisor`` (``uniform_grid_phys`` in
    :mod:`nvision.sim.locs.search_grid`), so ``n = ceil(W / spacing) + 1``.
    """
    if not (window_phys > 0.0 and fwhm_prior_phys > 0.0 and spacing_divisor > 0.0):
        raise ValueError(f"Bad sweep geometry: {window_phys!r}, {fwhm_prior_phys!r}, {spacing_divisor!r}.")
    return math.ceil(window_phys / (fwhm_prior_phys / spacing_divisor)) + 1


def sweep_positions_phys(x_lo_phys: float, x_hi_phys: float, n_points: int, n_measurements: int) -> np.ndarray:
    """Drive frequencies (Hz) a sweep measured: its ``n_points`` grid, repeated pass after pass.

    Returns shape (n_measurements,): measurement ``t`` (0-based) is at grid point ``t % n_points`` of
    ``linspace(x_lo, x_hi, n_points)``, which is exactly what ``GenericSweepLocator.next`` proposes.
    """
    if n_points < 2 or n_measurements < 1:
        raise ValueError(f"Need n_points >= 2 and n_measurements >= 1, got {n_points}, {n_measurements}.")
    grid = np.linspace(x_lo_phys, x_hi_phys, n_points)
    return grid[np.arange(n_measurements) % n_points]


def place_efficiency(eta: np.ndarray, detection_step: int) -> float | None:
    """``q_place`` of one run: mean ``eta`` over the measurements after ``detection_step``.

    ``eta``: shape (n_steps,), per-measurement Fisher efficiency in [0, 1] (every measurement has the same shot
    count, so the plain mean is the shot-weighted one). ``detection_step``: 1-indexed step the dip test first showed
    a dip and kept showing one; the measurements counted are steps ``detection_step + 1 ... n_steps``.
    ``None`` when the run ends at or before the detection (no measurement after it).
    """
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 1:
        raise ValueError(f"eta must be 1-D (n_steps,), got shape {eta.shape}.")
    if detection_step < 1:
        raise ValueError(f"detection_step is 1-indexed, got {detection_step}.")
    after = eta[detection_step:]
    return float(after.mean()) if after.size else None


class InverseMse(NamedTuple):
    """``1 / mean(err^2)`` of a use case against the measurement count."""

    n: np.ndarray  # shape (n_grid,), measurement counts
    inv_mse: np.ndarray  # shape (n_grid,), Hz^-2; NaN where fewer than half the repeats still run
    alive_share: np.ndarray  # shape (n_grid,), share of the repeats still running at n


def inverse_mse_curve(
    steps: Sequence[np.ndarray], errs: Sequence[np.ndarray], ends: np.ndarray, n_grid: np.ndarray
) -> InverseMse:
    """``1 / mean over the repeats of err(n)^2`` on ``n_grid``, with stopped runs keeping their final error.

    ``steps[i]``, ``errs[i]``: shape (m_i,), the per-step series of repeat ``i`` (measurement counts, ascending, and the
    true error ``|estimate - truth|`` in Hz after each; the series is downsampled to about 80 points, see
    ``nvision.metrics.series``). Between samples the error is interpolated linearly (a step hold would lag by the
    sampling interval, up to several percent of n on the geometric tail, which would bias ``n0``); beyond the last
    sample it is held, which is the "stopped runs keep their final error" rule. ``ends``: shape (n_runs,), each
    repeat's measurement count. ``n_grid``: shape (n_grid,), counts to evaluate at.
    A grid point where fewer than :data:`MIN_ALIVE_SHARE` of the repeats still run is NaN: the survivors are the slow,
    unlucky runs, and the mean over them is not the use case's.
    """
    n_grid = np.asarray(n_grid, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if not (len(steps) == len(errs) == ends.size and ends.size > 0):
        raise ValueError("steps, errs and ends must describe the same non-empty set of runs.")
    sq = np.empty((ends.size, n_grid.size))
    for i, (s, e) in enumerate(zip(steps, errs, strict=True)):
        s, e = np.asarray(s, dtype=np.float64), np.asarray(e, dtype=np.float64)
        if s.ndim != 1 or s.shape != e.shape or s.size == 0:
            raise ValueError(f"Run {i}: steps and errs must be equal-length 1-D arrays, got {s.shape} and {e.shape}.")
        sq[i] = np.interp(n_grid, s, e) ** 2
    alive = (ends[:, None] >= n_grid[None, :]).mean(axis=0)
    mse = sq.mean(axis=0)
    inv = np.where((alive >= MIN_ALIVE_SHARE) & (mse > 0.0), 1.0 / np.where(mse > 0.0, mse, 1.0), np.nan)
    return InverseMse(n_grid, inv, alive)


class RateFit(NamedTuple):
    """``1 / mean(err^2) = slope * n + intercept`` over a window, as ``q * I_max * (n - n0)``."""

    q: float  # slope / I_max; the share of the best information rate the estimate gained
    n0: float  # -intercept / slope, measurements; NaN when the slope is not positive
    slope: float  # Hz^-2 per measurement
    intercept: float  # Hz^-2
    r2: float  # coefficient of determination of the line over the window (1 = straight)
    n_lo: float  # fitted window, measurements
    n_hi: float


def late_window(inv: InverseMse, dip_steps: Sequence[int]) -> tuple[float, float] | None:
    """The part of a curve the line is fitted on: ``[max(median dip step, n_hi / 2), n_hi]``.

    ``n_hi`` is the largest ``n`` where at least half the repeats still run (``inv`` finite); the line is only
    expected after the dip was found (the median over the repeats that detected one) and the second half of the
    usable range keeps the search out of it. ``None`` when that leaves no range.
    """
    ok = np.isfinite(inv.inv_mse)
    if not ok.any():
        return None
    n_hi = float(inv.n[ok].max())
    n_lo = n_hi / 2.0
    if len(dip_steps):
        n_lo = max(n_lo, float(np.median(np.asarray(dip_steps, dtype=np.float64))))
    return (n_lo, n_hi) if n_lo < n_hi else None


def fit_rate_line(inv: InverseMse, i_max: float, window: tuple[float, float]) -> RateFit | None:
    """Least-squares line through ``1 / mean(err^2)`` over ``window``, expressed as ``q`` and ``n0``.

    ``i_max``: ``I_max`` (Hz^-2 per measurement). ``None`` when the window holds fewer than 5 finite points of the
    grid. A line with a non-positive slope (no information growing) returns ``q <= 0`` and ``n0 = NaN``: that is the
    measurement, not a fit failure.
    """
    if not i_max > 0.0:
        raise ValueError(f"I_max must be positive, got {i_max!r}.")
    lo, hi = window
    sel = np.isfinite(inv.inv_mse) & (inv.n >= lo) & (inv.n <= hi)
    if sel.sum() < 5:
        return None
    x, y = inv.n[sel], inv.inv_mse[sel]
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0.0 else float("nan")
    n0 = float(-intercept / slope) if slope > 0.0 else float("nan")
    return RateFit(float(slope / i_max), n0, float(slope), float(intercept), r2, float(lo), float(hi))


def predicted_cost(tau_phys: np.ndarray, q: float, n0: float, i_max: float) -> np.ndarray:
    """Measurements to reach a true error ``tau`` on the line: ``n0 + tau^-2 / (q * I_max)``.

    ``tau_phys``: shape (n_tau,), Hz. Same shape out. It is the cost of the *mean square* error reaching ``tau^2``,
    while the measured cost (:func:`cost_curve`) is the median first passage of single runs.
    """
    if not (q > 0.0 and i_max > 0.0) or not math.isfinite(n0):
        raise ValueError(f"Need q > 0, I_max > 0 and a finite n0, got {q!r}, {i_max!r}, {n0!r}.")
    return n0 + np.asarray(tau_phys, dtype=np.float64) ** -2 / (q * i_max)


class CostCurve(NamedTuple):
    """Cost to reach each precision target, over the runs of a use case."""

    t_median: np.ndarray  # shape (n_tau,), median over ALL runs (unreached = +inf); NaN where share < 0.5
    t_p75: np.ndarray  # shape (n_tau,), 75th percentile over ALL runs (unreached = +inf); NaN where share < 0.75
    share: np.ndarray  # shape (n_tau,), share of the runs that reached the target (always shown next to the median)


def first_passage(steps: np.ndarray, errs: np.ndarray, tau_phys: float) -> float | None:
    """First measurement from which the true error stays ``<= tau`` to the end of the run; ``None`` if it ends above.

    ``steps``, ``errs``: shape (m,), the downsampled per-step series (ascending counts, Hz). The first sampled step of
    the final stretch at or below ``tau`` is returned, as in the Information tab's cost ladder.
    """
    above = np.flatnonzero(np.asarray(errs) > tau_phys)
    if above.size and above[-1] == len(errs) - 1:
        return None
    return float(steps[above[-1] + 1]) if above.size else float(steps[0])


def cost_curve(steps: Sequence[np.ndarray], errs: Sequence[np.ndarray], tau_phys: np.ndarray) -> CostCurve:
    """Median / p75 first-passage cost over all runs, and the share of runs that reached each target.

    ``steps[i]``, ``errs[i]``: shape (m_i,), the series of run ``i``. ``tau_phys``: shape (n_tau,), Hz. The cost of a
    run is the number of measurements after which its error stays below the target to the end (:func:`first_passage`).
    A run that never gets there counts as ``+inf``, so the median is the plain median of all runs and is defined only
    where at least half of them reached the target (``share >= 0.5``); the p75 needs ``share >= 0.75``. Elsewhere NaN:
    a median over the survivors alone would be biased low.
    """
    tau_phys = np.asarray(tau_phys, dtype=np.float64)
    med, p75, share = (np.full(tau_phys.size, np.nan) for _ in range(3))
    for j, tau in enumerate(tau_phys):
        passed = [first_passage(s, e, float(tau)) for s, e in zip(steps, errs, strict=True)]
        srt = np.sort(np.array([np.inf if t is None else t for t in passed], dtype=np.float64))  # (n_runs,)
        share[j] = float(np.isfinite(srt).mean())
        for out, q in ((med, 0.5), (p75, 0.75)):
            # linear-interpolated quantile over all runs; undefined (NaN) when it touches a censored (inf) run
            pos = q * (srt.size - 1)
            lo, hi = math.floor(pos), math.ceil(pos)
            if np.isfinite(srt[hi]):
                out[j] = srt[lo] + (pos - lo) * (srt[hi] - srt[lo])
    return CostCurve(med, p75, share)


def speedup_curve(sbed: CostCurve, sweeps: Sequence[CostCurve], min_share: float = MIN_ALIVE_SHARE) -> np.ndarray:
    """``T_sweep(tau) / T_SBED(tau)``, the sweep being the best spacing at each target.

    ``sbed``, ``sweeps[k]``: :class:`CostCurve` on the same ``tau`` grid, shape (n_tau,). A locator's median counts at a
    target only where at least ``min_share`` of its runs got there; the best sweep is the cheapest of those. NaN where
    SBED or every sweep does not count. Rule 3: the yardstick is the best of the sweep spacings, nothing arbitrary.
    """
    t_sbed = np.where(sbed.share >= min_share, sbed.t_median, np.nan)
    t_sweeps = np.array([np.where(c.share >= min_share, c.t_median, np.nan) for c in sweeps])
    if t_sweeps.size == 0:
        return np.full_like(t_sbed, np.nan)
    best = np.where(np.isnan(t_sweeps), np.inf, t_sweeps).min(axis=0)
    best[np.isinf(best)] = np.nan
    return best / t_sbed


class OffsetHistogram(NamedTuple):
    """Where measurements fell relative to the true line centre, in units of its HWHM."""

    edges: np.ndarray  # shape (n_bins + 1,), bin edges in gamma
    fraction: np.ndarray  # shape (n_bins,), share of the measurements in each bin
    outside: float  # share of the measurements beyond the outermost edges


def nearest_centre_offsets(x_phys: np.ndarray, centres_phys: np.ndarray, gamma_phys: float) -> np.ndarray:
    """Signed offset of each measurement from the nearest true line centre, in units of the HWHM ``gamma``.

    ``x_phys``: shape (n,), drive frequencies (Hz). ``centres_phys``: shape (m,), the true line centres (Hz): one for a
    single line, three for an N-14 triplet. Returns shape (n,). A measurement is assigned to its nearest centre.
    """
    if not gamma_phys > 0.0:
        raise ValueError(f"gamma must be positive, got {gamma_phys!r}.")
    centres = np.asarray(centres_phys, dtype=np.float64)
    if centres.ndim != 1 or centres.size == 0:
        raise ValueError(f"centres_phys must be a non-empty 1-D array, got shape {centres.shape}.")
    d = np.asarray(x_phys, dtype=np.float64)[:, None] - centres[None, :]
    return np.take_along_axis(d, np.abs(d).argmin(axis=1)[:, None], axis=1)[:, 0] / gamma_phys


def offset_histogram(offsets_gamma: np.ndarray, edges: np.ndarray) -> OffsetHistogram | None:
    """Histogram of ``offsets_gamma`` (shape (n,)) over ``edges`` as shares of ``n``; ``None`` for no measurements."""
    offsets = np.asarray(offsets_gamma, dtype=np.float64)
    if offsets.size == 0:
        return None
    edges = np.asarray(edges, dtype=np.float64)
    counts, _ = np.histogram(offsets, bins=edges)
    return OffsetHistogram(edges, counts / offsets.size, 1.0 - counts.sum() / offsets.size)
