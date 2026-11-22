"""Tests of the article's quantitative claims that master's other metric modules do not already make.

What already exists is reused, not redone: the ``(n0, q)`` line and the sweep theory (``refine_rates``), the search
bounds
(``search_bounds``), the detection-aligned localize ratio (``detection_aligned``) and the posterior series
(``posterior``). This module adds the missing
pieces, each a measurement that can fail its claim (nothing is tuned toward the prediction; a window that cannot support
a number says why instead of widening):

* ``regimes``: detection, Gaussian onset and converge end of the runs, plateau and tail slope of the run-averaged
  error (claim: four regimes: search + localize, collapse, converge, refine);
* ``fit_with_gate``: the ``(n0, q)`` line of ``refine_rates.fit_rate_line`` plus a linearity gate and bootstrap
intervals
  (claim: ``1/MSE`` is linear in ``n`` late in the run);
* ``rate_ratio``: claimed information rate over the true one (claim: > 1 means overconfident);
* ``coverage_curve``, ``coverage_from_z``: share of runs within 1 and 2 claimed sigma (claim: 68 % / 95 %);
  ``coverage_from_pit``: the same from quantiles of the particle posterior (the PIT series ``p``);
* ``sensitivity_ratio``: measured over article sensitivity (claim: ``eta / sqrt(q)``).

Conventions: ``steps[i]``, ``vals[i]`` are one repeat's stored per-step series (measurement counts ascending, values in
Hz), as ``refine_rates.inverse_mse_curve`` takes them. Every run measures the whole budget (fingerprint
``stratified-zeeman-v1`` on), so a run's last stored value is held after its last stored step (the master rule) and all
repeats are present at every ``n``. ``docs/evaluation_metrics.md`` ("Testing the article's claims") says how to read
each.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nvision.metrics.refine_rates import RateFit, fit_rate_line, inverse_mse_curve

BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 20261118
GAUSS_WITHIN = {1: 0.6826894921370859, 2: 0.9544997361036416}  # share of a Gaussian within k sigma
MIN_RUNS = 10  # fewer repeats make every share and every bootstrap meaningless
HALF_MIN_POINTS = 3  # grid points each half of the fit window needs for the linearity gate


def step_grid(max_n: int, dense: int = 10, points: int = 70) -> np.ndarray:
    """Measurement counts to evaluate at: the first ``dense`` verbatim, then log-spaced to ``max_n``; shape
    (n_grid,)."""
    if max_n < 1:
        raise ValueError(f"max_n must be >= 1, got {max_n}")
    if max_n <= dense:
        return np.arange(1, max_n + 1, dtype=np.float64)
    tail = np.unique(np.round(np.geomspace(dense, max_n, points)))
    return np.unique(np.concatenate([np.arange(1, dense, dtype=np.float64), tail]))


def value_matrix(steps: Sequence[np.ndarray], vals: Sequence[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Per-run series on ``grid``: linear interpolation between stored samples, last value held after the last one.

    ``steps[i]``, ``vals[i]``: shape (m_i,); ``grid``: shape (n_grid,). Returns shape (n_runs, n_grid) in the series'
    unit.
    """
    if len(steps) != len(vals) or not len(steps):
        raise ValueError("steps and vals must list the same non-empty set of runs")
    out = np.empty((len(steps), grid.size))
    for i, (s, v) in enumerate(zip(steps, vals, strict=True)):
        s, v = np.asarray(s, dtype=np.float64), np.asarray(v, dtype=np.float64)
        if s.ndim != 1 or s.shape != v.shape or s.size == 0 or not np.all(np.isfinite(v)):
            raise ValueError(
                f"Run {i}: steps and values must be equal-length finite 1-D arrays, got {s.shape}, {v.shape}"
            )
        out[i] = np.interp(grid, s, v)
    return out


def rms(mat: np.ndarray) -> np.ndarray:
    """Root mean square over the runs. ``mat``: shape (n_runs, n_grid) -> (n_grid,)."""
    return np.sqrt(np.mean(mat * mat, axis=0))


# --- bound -------------------------------------------------------------------------------------------------------
def truth_spread_ratio(truth_phys: np.ndarray, lo_phys: float, hi_phys: float) -> float:
    """``rms(truth - window centre) / (W / sqrt12)``: 1 when the truths are uniform over the prior window.

    ``truth_phys``: shape (n_runs,), Hz. The claim that a uniform window's mean error is ``W / sqrt12`` assumes 1;
    truths
    kept off the edges give less.
    """
    t = np.asarray(truth_phys, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or not hi_phys > lo_phys:
        raise ValueError("need >= 2 truths and hi > lo")
    return float(np.sqrt(np.mean((t - 0.5 * (lo_phys + hi_phys)) ** 2)) / ((hi_phys - lo_phys) / math.sqrt(12.0)))


# --- regimes -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Regimes:
    """Run-level boundaries of the article's four regimes and the shape of the run-averaged error.

    ``None`` fields are what the data cannot support (see ``why``). The boundaries are in measurements ``n``:
    search + localize end at detection, the collapse ends at the Gaussian onset, converge ends where the run-averaged
    error is within 10 % of the data limit; refine is what follows.
    """

    plateau_ratio: float | None  # mean rms err over n < detection, in units of sigma_0 (~1 = plateau at the prior)
    detection_step: float | None  # median detection_step: end of search + localize
    gaussian_onset: float | None  # median first step from which the negentropy stays below the Gaussian threshold
    converge_end: float | None  # first n from which rms err <= CONVERGE_TOLERANCE / sqrt(q n I_max)
    tail_slope: float | None  # log-log slope of rms err on n >= 2 detection (-0.5 = Cramer-Rao; an offset steepens it)
    tail_points: int
    why: list[str] = field(default_factory=list)


CONVERGE_TOLERANCE = 1.1  # within 10 % of the data limit (n q I_max)^-1/2


def regimes(
    grid: np.ndarray,
    err: np.ndarray,
    sigma0: float,
    *,
    detection_step: float | None,
    gaussian_onset: float | None,
    q: float | None,
    imax: float,
) -> Regimes:
    """Regime boundaries of the article, from the stored series (Fig. budget).

    ``grid``: shape (n_grid,), measurement counts. ``err``: shape (n_runs, n_grid), Hz. ``sigma0``: Hz, the prior's
    standard deviation ``(1/J0_var)^(1/2)``. ``detection_step`` / ``gaussian_onset``: medians over the runs that have
    one (``None`` = no run has). ``q``: fitted ``q`` of the true-error line (``None`` = no fit); ``imax``: ``I_max``,
    Hz^-2 per measurement. The converge end is the first grid ``n`` from which ``rms(err) <= 1.1 / sqrt(q n I_max)``
    at every later grid point.
    """
    if not (sigma0 > 0 and imax > 0):
        raise ValueError(f"sigma0 and I_max must be positive, got {sigma0}, {imax}")
    r = rms(err)
    why: list[str] = []
    plateau = None
    slope = None
    tail_points = 0
    if detection_step is None:
        why.append("no run detected the dip: no detection step, plateau or tail slope")
    else:
        early = grid < detection_step
        if early.sum() >= 2:
            plateau = float(np.mean(r[early]) / sigma0)
        else:
            why.append("fewer than 2 grid steps before detection: no plateau level")
        late = grid >= 2.0 * detection_step
        tail_points = int(late.sum())
        if tail_points >= 4:
            slope = float(np.polyfit(np.log(grid[late]), np.log(r[late]), 1)[0])
        else:
            why.append(f"only {tail_points} grid steps after 2 x detection: no tail slope (the budget ends first)")
    if gaussian_onset is None:
        why.append("no run's posterior became a single Gaussian peak: no Gaussian onset")
    converge = None
    if q is None or not q > 0:
        why.append("no true-error (n0, q) fit: no converge end")
    else:
        within = r <= CONVERGE_TOLERANCE / np.sqrt(q * grid * imax)
        stays = np.flatnonzero(~within)
        first = 0 if stays.size == 0 else int(stays[-1]) + 1
        if first < grid.size:
            converge = float(grid[first])
        else:
            why.append("the run-averaged error is not within 10 % of the data limit at the end of the budget")
    return Regimes(plateau, detection_step, gaussian_onset, converge, slope, tail_points, why)


# --- (n0, q) with a linearity gate -------------------------------------------------------------------------------
@dataclass(frozen=True)
class GatedFit:
    """``refine_rates.fit_rate_line`` on a window, plus bootstrap intervals and a gate on the window being linear.

    ``q_halves``: q of the first and second half of the window; ``linear`` is True when their bootstrap intervals
    overlap. ``None`` fit / ``why`` set when the window is too short or 1/MSE does not grow.
    """

    fit: RateFit | None
    q_ci: tuple[float, float] | None
    n0_ci: tuple[float, float] | None
    q_halves: tuple[float, float] | None
    linear: bool | None
    why: str | None = None
    q_halves_ci: tuple[tuple[float, float] | None, tuple[float, float] | None] | None = None


def _boot_fit(
    steps: list[np.ndarray],
    vals: list[np.ndarray],
    grid: np.ndarray,
    imax: float,
    window: tuple[float, float],
    draws: int,
) -> tuple[RateFit | None, list[RateFit]]:
    """The fit of all runs and of ``draws`` resamples of the runs (None entries dropped), same window and grid."""
    ends = np.array([s[-1] for s in steps], dtype=np.float64)
    full = fit_rate_line(inverse_mse_curve(steps, vals, ends, grid), imax, window)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots: list[RateFit] = []
    for _ in range(draws):
        pick = rng.integers(0, len(steps), len(steps))
        f = fit_rate_line(
            inverse_mse_curve([steps[i] for i in pick], [vals[i] for i in pick], ends[pick], grid), imax, window
        )
        if f is not None:
            boots.append(f)
    return full, boots


def _ci(values: list[float]) -> tuple[float, float] | None:
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if len(values) >= 20 else None


def fit_with_gate(
    steps: list[np.ndarray],
    vals: list[np.ndarray],
    grid: np.ndarray,
    imax: float,
    window: tuple[float, float],
    *,
    bootstrap: int = BOOTSTRAP_SAMPLES,
) -> GatedFit:
    """``1/mean(vals^2) = q I_max (n - n0)`` over ``window`` with bootstrap intervals and the linearity gate.

    ``vals`` is the true error (or the claimed sigma for the claimed twin). The gate splits the window in two halves
    (>= :data:`HALF_MIN_POINTS` grid points each) and requires their q intervals to overlap; a window that fails it
    has not
    reached the late regime, and the q is then NOT a refine rate (it is still returned so it can be read with that
    caveat). A half whose slope is not positive is not an error: its interval, which then includes 0, is what is
    compared, so a short window gives a wide interval and passes weakly; ``q_halves_ci`` shows how weak.
    """
    full, boots = _boot_fit(steps, vals, grid, imax, window, bootstrap)
    if full is None:
        return GatedFit(None, None, None, None, None, "fewer than 5 grid points in the fit window")
    if not full.slope > 0:
        return GatedFit(full, None, None, None, None, "1/MSE does not grow with n in the window (no information gain)")
    q_ci = _ci([b.q for b in boots])
    n0_ci = _ci([b.n0 for b in boots if b.slope > 0])
    inside = grid[(grid >= window[0]) & (grid <= window[1])]
    if inside.size < 2 * HALF_MIN_POINTS:
        return GatedFit(full, q_ci, n0_ci, None, None, f"{inside.size} grid points: too few to test linearity")
    mid = float(inside[inside.size // 2])
    halves = []
    for sub in ((window[0], float(inside[inside.size // 2 - 1])), (mid, window[1])):
        f, bs = _boot_fit(steps, vals, grid, imax, sub, bootstrap)
        if f is None:
            return GatedFit(full, q_ci, n0_ci, None, None, "fewer than 5 grid points in one half of the window")
        halves.append((f.q, _ci([b.q for b in bs])))
    (q1, ci1), (q2, ci2) = halves
    if ci1 is None or ci2 is None:
        return GatedFit(
            full, q_ci, n0_ci, (q1, q2), None, "too few bootstrap draws in a half to test linearity", (ci1, ci2)
        )
    linear = max(ci1[0], ci2[0]) <= min(ci1[1], ci2[1])
    why = (
        None if linear else f"not linear: q of the halves differ ({q1:.3g} vs {q2:.3g}), the late regime is not reached"
    )
    return GatedFit(full, q_ci, n0_ci, (q1, q2), linear, why, (ci1, ci2))


def rate_ratio(claimed: GatedFit, true: GatedFit) -> float | None:
    """``q_claimed / q_true`` over one window: > 1 means the claimed sigma fell faster than the true error
    (overconfidence).

    Both fits must have a positive slope on the same window; otherwise ``None`` (never a stand-in).
    """
    if claimed.fit is None or true.fit is None or not (claimed.fit.q > 0 and true.fit.q > 0):
        return None
    return float(claimed.fit.q / true.fit.q)


def sensitivity_ratio(grid: np.ndarray, err: np.ndarray, imax: float, window: tuple[float, float]) -> float | None:
    """``eta_measured / eta_article = sqrt(I_max * MSE(n) * n)``, median over the window (article Eq. sensitivity).

    ``t_m``, ``gamma_e`` and the SNR cancel against ``I_max``, so the dimensionless ratio is what the data give. It
    tends to
    ``1/sqrt(q)`` times ``sqrt(n / (n - n0))``: a consistency check of the fit, not an independent confirmation.
    ``None``
    with fewer than 4 grid points in ``window``.
    """
    sel = (grid >= window[0]) & (grid <= window[1])
    if sel.sum() < 4:
        return None
    return float(np.median(np.sqrt(imax * np.mean(err[:, sel] ** 2, axis=0) * grid[sel])))


# --- coverage ----------------------------------------------------------------------------------------------------
def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95 % for the default ``z``) of ``hits / total``."""
    if total < 1 or not 0 <= hits <= total:
        raise ValueError(f"need 0 <= hits <= total and total >= 1, got {hits}/{total}")
    p, d = hits / total, 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return centre - half, centre + half


def coverage_curve(
    err: np.ndarray, unc: np.ndarray, gate: np.ndarray, grid: np.ndarray
) -> list[dict[str, float | int | None]]:
    """Share of runs with ``err <= k * unc`` (k = 1, 2) at every grid step, with Wilson intervals.

    ``err``, ``unc``: shape (n_runs, n_grid), Hz (true error and claimed sigma). ``gate``: shape (n_runs,), the step
    from
    which a run counts (its Gaussian onset; NaN = never reached, excluded) because the article tests coverage only
    once
    the posterior is Gaussian; ``within_k_all`` is the same over all runs so the effect of the gate is visible. A step
    with fewer than :data:`MIN_RUNS` counted runs reports ``None``. Below 0.683 / 0.954 is overconfident.
    """
    gate = np.asarray(gate, dtype=np.float64)
    if err.shape != unc.shape or gate.shape != (err.shape[0],) or err.shape[1] != grid.size:
        raise ValueError("err, unc, gate and grid shapes disagree")
    if not np.all(unc > 0):
        raise ValueError("claimed sigma must be positive")
    gated = grid[None, :] >= gate[:, None]  # a NaN gate compares False: never counted
    rows = []
    for j in range(grid.size):
        row: dict[str, float | int | None] = {"n": int(grid[j]), "runs_gated": int(gated[:, j].sum())}
        for tag, mask in (("", gated[:, j]), ("_all", np.ones(err.shape[0], dtype=bool))):
            for k in (1, 2):
                m = int(mask.sum())
                row[f"within_{k}{tag}"] = float((err[mask, j] <= k * unc[mask, j]).mean()) if m >= MIN_RUNS else None
                if tag == "" and m >= MIN_RUNS:
                    hits = int((err[mask, j] <= k * unc[mask, j]).sum())
                    row[f"within_{k}_ci"] = list(wilson_interval(hits, m))
        rows.append(row)
    return rows


def coverage_from_z(z: np.ndarray) -> dict[str, float | list[float] | int]:
    """Coverage from the signed ``z = (estimate - truth) / sigma_hat`` of one step (series ``z``): |z| <= 1 and <= 2.

    ``z``: shape (n_runs,). A calibrated Gaussian posterior gives 0.683 / 0.954; ``mean`` (bias) and ``std`` of z (1
    when
    the claimed sigma is the spread of the error) separate a shifted posterior from a too-narrow one.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size < MIN_RUNS or not np.all(np.isfinite(z)):
        raise ValueError(f"need >= {MIN_RUNS} finite z values, got shape {z.shape}")
    out: dict[str, float | list[float] | int] = {"runs": int(z.size), "mean": float(z.mean()), "std": float(z.std())}
    for k in (1, 2):
        hits = int((np.abs(z) <= k).sum())
        out[f"within_{k}"] = hits / z.size
        out[f"within_{k}_ci"] = list(wilson_interval(hits, z.size))
    return out


def coverage_from_pit(pit: np.ndarray) -> dict[str, float | list[float] | int]:
    """Quantile coverage from the PIT ``p`` (posterior CDF at the truth, series ``p``) of one step.

    ``pit``: shape (n_runs,), values in [0, 1]. A run is inside the central 68 % interval of its own particle
    posterior when ``0.16 <= p <= 0.84`` and inside the central 95 % one when ``0.025 <= p <= 0.975``
    (``within_1`` / ``within_2``, nominal 0.68 / 0.95, with Wilson intervals). Needs no Gaussian peak. ``mean`` and
    ``std`` of ``p`` are 1/2 and 1/sqrt(12) when calibrated (a U-shaped histogram, std above that, is overconfident).
    """
    pit = np.asarray(pit, dtype=np.float64)
    if pit.ndim != 1 or pit.size < MIN_RUNS or not np.all(np.isfinite(pit)):
        raise ValueError(f"need >= {MIN_RUNS} finite PIT values, got shape {pit.shape}")
    if pit.min() < 0.0 or pit.max() > 1.0:
        raise ValueError("PIT values must lie in [0, 1]")
    out: dict[str, float | list[float] | int] = {
        "runs": int(pit.size),
        "mean": float(pit.mean()),
        "std": float(pit.std()),
    }
    for k, lo, hi in ((1, 0.16, 0.84), (2, 0.025, 0.975)):
        hits = int(((pit >= lo) & (pit <= hi)).sum())
        out[f"within_{k}"] = hits / pit.size
        out[f"within_{k}_ci"] = list(wilson_interval(hits, pit.size))
    return out
