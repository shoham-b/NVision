"""Search-phase bounds and the measured search, per use case (article Sec. global, SM Sec. prior).

During the search no strategy, adaptive or not, can beat the bounds of ``articles/sections/theory.tex`` (Sec. global)
and ``articles/sections/sm_prior.tex``. This module holds their formulas and the statistics that are compared to them.
It is pure numpy: the cache plumbing lives in :mod:`nvision.cli.search_bounds_api`.

Notation (all lengths in Hz, the primary parameter being ``zeeman_shift``):

* ``W``: the width the generator DRAWS ``zeeman_shift`` from (``zeeman_min_phys`` .. ``zeeman_max_phys``): not the drive
  window and not a van Trees prior. The bounds average over ``theta`` uniform in this range.
* ``kappa_bar = k * int dip(x)^2 dx / (2 sigma^2 W)``: the mean Kullback-Leibler rate (per measurement) between the
  baseline and a dip at a uniformly drawn ``theta``. For a Lorentzian of depth ``C`` and HWHM ``gamma`` it equals
  ``pi gamma k C^2 / (4 W sigma^2) = pi gamma k SNR^2 / (4 W)`` with ``SNR = C / sigma``. It is computed from the true
  line numerically so it holds for any line shape (Voigt, hyperfine multiplets).
* ``dd`` (``Delta d``): the reach of one measurement (``eq:sm-reach``), the width of the cell it rules out.
* ``I_max``: Fisher information (Hz^-2) of one measurement at the steepest slope, the stored ``stage_fisher_star``.

Truth-agnostic interface: nothing here reads the generator's true signal. The caller passes a reference centre
(simulation: the true ``zeeman_shift``; lab: the holdout fit's centre) and gets errors from :func:`abs_errors`, plus the
line's width ``gamma``, shots ``k`` and ``SNR`` (or the reference ``sigma``) as plain numbers. What each output needs:

* single run, no ensemble over ``theta``: the bound curves (:func:`finite_reach_curve`, :func:`change_of_measure_curve`,
  :func:`mean_curve`), :func:`kappa_bar_lorentzian`, :func:`probe_spacing`, and the detection step of one run against
  ``1 / kappa_bar`` and ``W / dd`` (:func:`detection_summary` on one value);
* simulation-only (they average over many ``theta`` drawn uniformly in the window): :func:`mean_squared_error`,
  :func:`survival`, :func:`fit_kappa_eff` and the :func:`survival_floor_curve` they are compared with.

Array conventions: ``n`` is a measurement count, shape ``(n_grid,)``; error matrices have shape ``(n_runs, n_grid)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# Survival threshold t, in units of the line's HWHM: the event |true error| > t is the one the change-of-measure bound
# (eq:search-bound) is on.
SURVIVAL_TOLERANCE_HWHM = 1.0
# Survival range over which the log-slope kappa_eff is fitted (share of runs still wrong): from the first clear drop
# below 1 to before the tail is dominated by a few runs.
KAPPA_FIT_S_HIGH = 0.9
KAPPA_FIT_S_LOW = 0.05
KAPPA_FIT_MIN_POINTS = 4
# Missed-detection margin z_beta of eq:sm-reach: z = z_alpha + z_beta, z_alpha = sqrt(2 ln(W / dd)) (rare false alarms).
Z_BETA = 1.28
# eq:sm-fixed-cost: a fixed-averaging sweep pays ~2.4 ln(W / dd) times the change-of-measure bound.
SWEEP_COST_FACTOR = 2.4
# The best probe averages up to u = 4/3 (z units) when one measurement alone is weaker, spacing Gamma / sqrt(3).
SWEEP_SINGLE_SHOT_LIMIT = 4.0 / 3.0
_REACH_ITERATIONS = 200
_REACH_TOLERANCE = 1e-10
# The simulator never sets ``n_shots`` (every measurement is one shot); ``implied_shots`` checks it against the cache.
SIM_SHOTS_PER_MEASUREMENT = 1.0
# Stored stage_fisher_star and the line rebuilt here must agree on k to this relative tolerance.
SHOTS_CROSS_CHECK_REL_TOL = 0.05


# --------------------------------------------------------------------------- line overlap, kappa_bar, reach


class LineIntegrals(NamedTuple):
    """Integrals of the true line: ``dip_sq_integral`` = int dip^2 dx (signal^2 Hz), ``max_grad_sq`` = max g^2."""

    dip_sq_integral: float
    max_grad_sq: float


def line_integrals(x_phys: np.ndarray, dip: np.ndarray, grad_sq: np.ndarray) -> LineIntegrals:
    """``int dip^2 dx`` and ``max g^2`` of a line sampled on a grid.

    ``x_phys``: shape (n_x,), drive frequency (Hz), uniformly spaced, wide enough that the dip has decayed at both ends.
    ``dip``: shape (n_x,), baseline minus signal (signal units, >= 0). ``grad_sq``: shape (n_x,), the squared
    sensitivity ``(d signal / d zeeman_shift)^2`` (signal^2 Hz^-2) on the same grid.
    """
    x = np.asarray(x_phys, dtype=float)
    d = np.asarray(dip, dtype=float)
    g2 = np.asarray(grad_sq, dtype=float)
    if not (x.ndim == d.ndim == g2.ndim == 1 and x.size == d.size == g2.size and x.size >= 3):
        raise ValueError(f"x, dip, grad_sq must be equal-length 1-D arrays, got {x.shape}, {d.shape}, {g2.shape}")
    steps = np.diff(x)
    if not np.allclose(steps, steps[0], rtol=1e-6):
        raise ValueError("x_phys must be uniformly spaced")
    return LineIntegrals(float(np.sum(d**2) * steps[0]), float(np.max(g2)))


def kappa_bar(
    dip_sq_integral: float, width_phys: float, sigma: float, shots: float = SIM_SHOTS_PER_MEASUREMENT
) -> float:
    """Mean KL rate per measurement ``k int dip^2 dx / (2 sigma^2 W)`` (1 / measurement)."""
    if not (dip_sq_integral > 0 and width_phys > 0 and sigma > 0 and shots > 0):
        raise ValueError(f"kappa_bar needs positive inputs, got {(dip_sq_integral, width_phys, sigma, shots)}")
    return float(shots * dip_sq_integral / (2.0 * sigma**2 * width_phys))


def kappa_bar_lorentzian(width_phys: float, hwhm_phys: float, snr: float, shots: float = 1.0) -> float:
    """Closed form ``pi gamma k SNR^2 / (4 W)`` for a Lorentzian dip (reference for tests and docs)."""
    return math.pi * hwhm_phys * shots * snr**2 / (4.0 * width_phys)


def implied_shots(fisher_star: float, sigma: float, max_grad_sq: float) -> float:
    """``k = I* sigma^2 / max g^2``: the shots per measurement implied by the stored ``stage_fisher_star``.

    Used to check the cache against the line rebuilt here: it must equal ``SIM_SHOTS_PER_MEASUREMENT``.
    """
    return float(fisher_star * sigma**2 / max_grad_sq)


class Reach(NamedTuple):
    """Probe spacing of a fixed-averaging sweep.

    ``dd_phys``: cell width Delta d (Hz). ``z``: detection threshold z_alpha + z_beta. ``kind``: ``"reach"`` when one
    measurement per probe suffices (eq:sm-reach), ``"core"`` when the best probe averages (Delta d = Gamma / sqrt 3) or
    ``"window"`` when the reach is at least the whole window (Delta d = W: one measurement covers it, the bound is 0).
    """

    dd_phys: float
    z: float
    kind: str


def probe_spacing(
    width_phys: float, fwhm_phys: float, snr: float, shots: float = SIM_SHOTS_PER_MEASUREMENT
) -> Reach | None:
    """Spacing ``Delta d`` of eq:sm-reach / eq:sm-fixed-cost, or ``None`` when no single measurement detects the dip.

    ``eq:sm-reach``: ``Delta d = Gamma sqrt(sqrt(k) SNR / z - 1)`` with ``z = sqrt(2 ln(W / Delta d)) + z_beta``, a
    fixed point in ``Delta d`` solved by iteration. If ``a = sqrt(k) SNR / z`` is at most 1 there is no reach. If
    ``a <= 4/3`` the best probe averages repeats and the spacing is ``Gamma / sqrt 3`` (eq:sm-fixed-cost), with ``z``
    evaluated at that spacing.
    """
    if not (width_phys > 0 and fwhm_phys > 0 and snr > 0 and shots > 0):
        raise ValueError(f"probe_spacing needs positive inputs, got {(width_phys, fwhm_phys, snr, shots)}")
    amp = math.sqrt(shots) * snr
    dd = fwhm_phys
    z = math.nan
    for _ in range(_REACH_ITERATIONS):
        cells = width_phys / dd
        if cells <= 1.0:
            return Reach(float(width_phys), Z_BETA, "window")  # one measurement rules out the whole window
        z = math.sqrt(2.0 * math.log(cells)) + Z_BETA
        ratio = amp / z
        if ratio <= 1.0:
            return None
        new_dd = fwhm_phys * math.sqrt(ratio - 1.0)
        if abs(new_dd - dd) <= _REACH_TOLERANCE * fwhm_phys:
            dd = new_dd
            break
        dd = new_dd
    else:
        raise ValueError(f"eq:sm-reach did not converge for W={width_phys:g}, Gamma={fwhm_phys:g}, SNR={snr:g}")
    cells = width_phys / dd
    z = math.sqrt(2.0 * math.log(cells)) + Z_BETA
    if amp / z > SWEEP_SINGLE_SHOT_LIMIT:
        return Reach(float(dd), float(z), "reach")
    dd_core = fwhm_phys / math.sqrt(3.0)
    if width_phys / dd_core <= 1.0:
        return Reach(float(width_phys), Z_BETA, "window")
    return Reach(float(dd_core), float(math.sqrt(2.0 * math.log(width_phys / dd_core)) + Z_BETA), "core")


def predicted_sweep_ratio(width_phys: float, dd_phys: float) -> float:
    """``kappa_eff / kappa_bar`` predicted for a sweep: ``1 / (2.4 ln(W / Delta d))`` (eq:sm-fixed-cost)."""
    if not width_phys > dd_phys > 0:
        raise ValueError(f"need W > dd > 0, got W={width_phys:g}, dd={dd_phys:g}")
    return 1.0 / (SWEEP_COST_FACTOR * math.log(width_phys / dd_phys))


# --------------------------------------------------------------------------- bound curves


def finite_reach_curve(n: np.ndarray, width_phys: float, dd_phys: float) -> np.ndarray:
    """eq:search-mean, ``(W - n dd)^3 / (12 W)`` (Hz^2), shape (n_grid,); 0 once the window is covered."""
    n = np.asarray(n, dtype=float)
    left = np.clip(width_phys - n * dd_phys, 0.0, None)
    return left**3 / (12.0 * width_phys)


def change_of_measure_curve(n: np.ndarray, width_phys: float, kappa: float) -> np.ndarray:
    """eq:search-bound, ``W^2 / (8e) exp(-n kappa) / (1 + ln 2 + n kappa)^2`` (Hz^2), shape (n_grid,)."""
    a = 1.0 + math.log(2.0) + np.asarray(n, dtype=float) * kappa
    return width_phys**2 / (8.0 * math.e) * np.exp(-(a - 1.0 - math.log(2.0))) / a**2


def survival_floor_curve(n: np.ndarray, width_phys: float, kappa: float, tolerance_phys: float) -> np.ndarray:
    """The bound on the survival ``S(n) >= exp(-(n kappa + ln 2) / (1 - 2 t / W))`` (capped at 1), shape (n_grid,).

    This is the inequality eq:search-bound is derived from, for the chosen ``t``; a measured survival below it
    would be a violation of the bound.
    """
    if not 0.0 < 2.0 * tolerance_phys < width_phys:
        raise ValueError(f"need 0 < 2t < W, got t={tolerance_phys:g}, W={width_phys:g}")
    expo = (np.asarray(n, dtype=float) * kappa + math.log(2.0)) / (1.0 - 2.0 * tolerance_phys / width_phys)
    return np.minimum(1.0, np.exp(-expo))


def mean_curve(n: np.ndarray, width_phys: float, dd_phys: float, fisher_max: float) -> np.ndarray:
    """eq:mean-curve, the elimination model's mean squared error (Hz^2), shape (n_grid,).

    ``search``: ``(W - n dd)^3 / (12 W)``. ``found``: the runs that detected at step ``n_d`` (a fraction ``dd / W``
    each, the last cell narrower if ``W / dd`` is not an integer) follow ``(12 / dd^2 + (n - n_d) I_max)^-1``
    (eq:after-detection). ``n_d`` runs over the cells, so past ``W / dd`` the whole window has detected.
    """
    n = np.asarray(n, dtype=float)
    if not (width_phys > 0 and dd_phys > 0 and fisher_max > 0):
        raise ValueError(f"mean_curve needs positive inputs, got {(width_phys, dd_phys, fisher_max)}")
    cells = math.ceil(width_phys / dd_phys)
    n_d = np.arange(1, cells + 1, dtype=float)  # shape (cells,)
    cell_width = np.minimum(dd_phys, width_phys - (n_d - 1.0) * dd_phys)  # shape (cells,)
    weight = cell_width / width_phys
    j_tau = 12.0 / cell_width**2  # information at detection: the cell's own variance
    out = finite_reach_curve(n, width_phys, dd_phys)
    for i, n_i in enumerate(n):
        found = n_d <= n_i
        if found.any():
            out[i] += float(np.sum(weight[found] / (j_tau[found] + (n_i - n_d[found]) * fisher_max)))
    return out


# --------------------------------------------------------------------------- measured error and survival


def abs_errors(estimates_phys: np.ndarray, reference_centre_phys: float) -> np.ndarray:
    """``|estimate - reference centre|`` (Hz); shape (n_steps,) like ``estimates_phys``.

    The reference centre is the true ``zeeman_shift`` in simulation and the holdout fit's centre on lab data.
    """
    return np.abs(np.asarray(estimates_phys, dtype=float) - float(reference_centre_phys))


def align_errors(
    steps_list: list[np.ndarray], errs_list: list[np.ndarray], n_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-run true error on a common step grid by carry-forward, plus whether each run is still running.

    ``steps_list[i]``: shape (m_i,), the 1-indexed kept steps of run ``i`` (the stored series is downsampled);
    ``errs_list[i]``: shape (m_i,), the true error ``|estimate - truth|`` (Hz) at those steps, NaN-free.
    ``n_grid``: shape (n_grid,), measurement counts. Returns ``errors`` shape (n_runs, n_grid): the error at the last
    kept step <= n (a run's first kept step is step 1; a stopped run keeps its final error) and ``running`` of the same
    shape: ``n <= `` the run's last step.
    """
    n_grid = np.asarray(n_grid, dtype=float)
    errors = np.empty((len(steps_list), n_grid.size))
    running = np.empty_like(errors, dtype=bool)
    for i, (s, e) in enumerate(zip(steps_list, errs_list, strict=True)):
        s = np.asarray(s, dtype=float)
        e = np.asarray(e, dtype=float)
        if s.size == 0 or s.size != e.size or not np.all(np.diff(s) > 0) or s[0] != 1:
            raise ValueError(f"run {i}: series must start at step 1 and have increasing steps")
        idx = np.searchsorted(s, n_grid, side="right") - 1
        errors[i] = e[np.clip(idx, 0, None)]
        running[i] = n_grid <= s[-1]
    return errors, running


def usable_columns(running: np.ndarray) -> np.ndarray:
    """Grid columns where at least half of the runs still run (shape (n_grid,), bool)."""
    return running.mean(axis=0) >= 0.5


def mean_squared_error(errors: np.ndarray) -> np.ndarray:
    """Mean over runs of ``error^2`` (Hz^2) per grid column; shape (n_grid,). Rule 8: the tail is not trimmed."""
    return np.mean(np.asarray(errors, dtype=float) ** 2, axis=0)


def survival(errors: np.ndarray, tolerance_phys: float) -> np.ndarray:
    """Share of runs with ``error > t`` per grid column; shape (n_grid,)."""
    return np.mean(np.asarray(errors, dtype=float) > tolerance_phys, axis=0)


def fit_kappa_eff(n: np.ndarray, surv: np.ndarray) -> float | None:
    """Rate ``kappa_eff = -d ln S / dn`` by least squares of ``ln S`` on ``n`` where ``S`` is in (0.05, 0.9).

    ``None`` when fewer than ``KAPPA_FIT_MIN_POINTS`` grid points fall in that range (the curve never decays through
    it, or falls through it between two grid points) or the fitted rate is not positive.
    """
    n = np.asarray(n, dtype=float)
    surv = np.asarray(surv, dtype=float)
    keep = (surv <= KAPPA_FIT_S_HIGH) & (surv >= KAPPA_FIT_S_LOW)
    if keep.sum() < KAPPA_FIT_MIN_POINTS:
        return None
    slope = float(np.polyfit(n[keep], np.log(surv[keep]), 1)[0])
    return -slope if slope < 0 else None


class DetectionSummary(NamedTuple):
    """``dip_detected_step`` of a use case's runs, with runs that never detected counted at the budget (rule 2).

    ``median`` / ``p75``: steps. ``capped_share``: share of runs that never detected (censored at ``budget``).
    ``n_runs``: number of runs. A run that never detected is a lower bound, so ``median`` / ``p75`` are lower bounds
    whenever ``capped_share`` reaches 0.5 / 0.25.
    """

    median: float
    p75: float
    capped_share: float
    n_runs: int


def detection_summary(detected_steps: list[int | None], budget: int) -> DetectionSummary:
    """Median, p75 and capped share of ``dip_detected_step`` (``None`` = never, counted at ``budget``)."""
    if not detected_steps:
        raise ValueError("detection_summary needs at least one run")
    steps = np.array([budget if s is None else min(int(s), budget) for s in detected_steps], dtype=float)
    never = np.array([s is None for s in detected_steps])
    return DetectionSummary(
        float(np.median(steps)), float(np.percentile(steps, 75)), float(never.mean()), len(detected_steps)
    )
