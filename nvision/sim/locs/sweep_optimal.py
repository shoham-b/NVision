"""The analytically optimal grid step of a plain sweep: grid points per prior FWHM, ``k*``, chosen per task.

A plain sweep measures a uniform grid of step ``FWHM / k`` and repeats whole passes over it until Bayesian-SBED's
stopping rules fire (:class:`~nvision.sim.locs.bayesian.sequential_bayesian_locator.ConvergenceStopRules`), but it only
stops at the end of a pass. Two effects set the cost, both functions of ``k``:

* **Ripple.** Fisher information about the dip position from a grid depends on where the dip sits relative to the
  grid (Poisson summation: the sum over grid points of the squared slope equals the continuum integral plus
  harmonics at multiples of ``1 / step`` that decay like ``exp(-pi * FWHM * m / step)``). For a Lorentzian the
  information per measurement relative to a dense grid is, worst/best offset: k=1 0.13/1.37, k=1.5 0.43/1.48,
  k=2 0.71/1.28, k=2.5 0.88/1.12, k=3 0.96/1.04, k=4 1.00/1.00. Detection energy is flat (>= 0.97) from k = 2.
* **Pass rounding.** One pass costs ``n_points(k) ~ k * W / FWHM`` measurements and the sweep stops only after a whole
  number of passes, so the measurements actually spent are ``P * n_points`` with ``P`` an integer. Above k ~ 2.5 the
  information per measurement no longer changes, so ``k`` matters only through this rounding: a fractional ``k`` lets
  a whole number of passes land on the measurement count the stopping rules need, and at high SNR one pass already
  overshoots, so a smaller ``k`` (accepting some ripple) is cheaper.

``k*`` minimises, over a scan of ``k`` in [``K_MIN``, ``K_MAX``] (step ``K_STEP``),

    cost(k) = mean over grid offsets phi of  P(k, phi) * n_points(k),

``phi`` running over ``N_OFFSETS`` evenly spaced positions of the dip within one grid step, and
``P(k, phi) = max(P_prec, P_det)`` the whole passes the stopping rules need.

The sweep makes at most ``max(1, M // n_points)`` passes (``M = max_measurements``). A ``k`` with
``cost(k) > M`` is *unreachable*: the sweep will stop at the cap with the stop rule unmet. Any reachable ``k`` beats
every unreachable one. When *no* ``k`` is reachable, the cost is meaningless (it would pick an arbitrary ``k`` near
``K_MAX``) and the objective becomes the expected final error at the cap: with ``P_cap(k) = max(1, M // n_points)``,

    err(k) = mean over phi of  s*(P_cap * F_pass),

where ``s*`` is the binding relative uncertainty, the factor by which the CRLB stds would have to shrink for
:func:`target_params_converged` to accept them (``s* < 1``: converged; found by bisection on that function, so the
per-parameter ceilings, the RMS condition and the saturation-Voigt derived quantities are the stop rule's own). The
``k`` with the smallest ``err`` wins (precision only; detection is not scored in this case). The reported
``passes_pred`` / ``cost_pred`` are then ``P_cap`` and ``P_cap * n_points`` and ``within_budget`` is False.

* ``P_prec``: the smallest ``P`` for which, with the Fisher matrix ``P * F_pass`` of ``P`` passes over the actual
  grid, every target parameter's Cramer-Rao standard deviation is under its convergence ceiling and the RMS relative
  condition holds -- :func:`nvision.metrics.milestones.target_params_converged` itself is evaluated, so the ceilings,
  the absolute zero_field_freq ceiling, the exclusion of drive-response parameters and the saturation-Voigt derived
  quantities are the stop rule's own. ``F_pass`` is built from the belief's own signal model (numerical central
  differences of every parameter the belief infers, Gaussian noise ``sigma``), at the prior-mean parameters, so
  Voigt / hyperfine / saturation lineshapes need no special case. The CRLB uses the same unit-normalised, ridged
  pseudo-inverse as :class:`~nvision.models.fisher_information.CumulativeFisher`.
* ``P_det``: the smallest ``P`` for which the boxcar GLRT of :func:`nvision.belief.dip_detection.find_dips` is
  expected to clear its threshold. The score of a window of ``m`` grid points after ``P`` passes is
  ``sqrt(P) * sum_i(depth_i / sigma) / sqrt(m)`` (the noise-free expectation; the windows are the ones ``find_dips``
  tests, so the best score is ``sqrt(P)`` times the best one-pass score), the threshold is
  ``Phi^-1(1 - p_fa / N_windows)``. Approximation: ``find_dips`` uses a Student-t threshold whose degrees of freedom
  grow with the data; here sigma is known, so the normal quantile (``nu -> inf``) is used. The expected score ignores
  the positive bias of a maximum over windows, and the stop rule's patience (``NVISION_CONVERGENCE_PATIENCE``
  consecutive passing checks) is not modelled; both are second-order against the pass rounding.

Oracle / prior split: ``sigma`` is the TRUE noise sigma of the task (an oracle yardstick -- the belief's own noise
prior is a wide window around it); every signal quantity (contrast, widths, shifts) is the belief's *prior mean*, as
:func:`~nvision.sim.locs.search_grid.prior_fwhm_phys` reads it. The sweep therefore needs no knowledge of the true
signal, only of the noise level it runs at.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from nvision.belief.dip_detection import _window_scores, hwhm_fwhm_prior_hz, window_width_range_hz
from nvision.metrics.milestones import target_params_converged
from nvision.models.fisher_information import ranges_from_bounds, single_shot_marginal_stds_from_fim
from nvision.sim.defaults import NVISION_CONVERGENCE_THRESHOLD, NVISION_DIP_P_FA
from nvision.sim.locs.search_grid import prior_fwhm_phys, prior_mean_params_phys, uniform_grid_phys

# Strategy-config value of ``spacing_divisor`` that asks for the optimal step (see GenericSweepLocator).
OPTIMAL_SPACING = "optimal"

# Scan of k (grid points per prior FWHM): inclusive range and step.
K_MIN: float = 0.5
K_MAX: float = 8.0
K_STEP: float = 0.05
# Dip offsets within one grid step the cost is averaged over (the offset dependence is smooth and periodic, so a
# handful of evenly spaced offsets average it accurately).
N_OFFSETS: int = 8

# Upper bound of the passes a requirement can ask for (an unidentifiable parameter never converges).
_P_UNBOUNDED: int = 1_000_000

# The signal and its parameter derivatives are tabulated once on a dense grid of this many points per FWHM and
# interpolated linearly to every scanned grid (interpolation error ~ (1/DENSE)^2 / 8 of the local curvature).
_DENSE_POINTS_PER_FWHM: int = 100
# Central-difference step of the parameter derivatives, as a fraction of the parameter's bound width (the convention
# of :func:`nvision.models.fisher_information.numerical_gradient_vector`).
_GRAD_REL_STEP: float = 1e-4


@dataclass(frozen=True, slots=True)
class OptimalSweepSpacing:
    """Result of :func:`optimal_sweep_spacing`.

    ``k_opt``: requested grid points per prior FWHM (the grid is ``uniform_grid_phys(lo, hi, FWHM / k_opt)``);
    ``spacing_phys``: its actual step (Hz); ``n_points``: points per pass; ``passes_pred``: predicted whole passes
    (mean over dip offsets); ``cost_pred``: predicted measurements ``P * n_points`` (mean over offsets);
    ``passes_precision`` / ``passes_detection``: the two requirements behind ``passes_pred`` (mean over offsets);
    ``within_budget``: ``cost_pred <= max_measurements`` (False: the stopping rules are predicted unreachable, the
    sweep will run out of budget; ``k_opt`` then minimises ``error_ratio_at_cap``); ``error_ratio_at_cap``: in that
    case the mean binding relative uncertainty at the cap (< 1 would be converged), else NaN.
    """

    k_opt: float
    spacing_phys: float
    n_points: int
    passes_pred: float
    cost_pred: float
    passes_precision: float
    passes_detection: float
    within_budget: bool
    error_ratio_at_cap: float = float("nan")


@dataclass(frozen=True, slots=True)
class SignalSensitivity:
    """The belief's signal and its parameter derivatives tabulated on a dense uniform drive-frequency grid.

    ``xs_phys``: shape (n_dense,), Hz, ascending; ``y``: shape (n_dense,), signal value; ``grads``: shape
    (n_dense, n_params), d(signal)/d(parameter) in physical units, columns in ``names`` order.
    """

    names: tuple[str, ...]
    xs_phys: NDArray[np.float64]
    y: NDArray[np.float64]
    grads: NDArray[np.float64]


def tabulate_sensitivity(
    inner_model,
    params_phys: dict[str, float],
    bounds_phys: dict[str, tuple[float, float]],
    xs_phys: NDArray[np.float64],
) -> SignalSensitivity:
    """Signal and central-difference parameter derivatives of a physical model at ``xs_phys`` (Hz).

    ``params_phys`` maps every parameter name of ``inner_model.parameter_names()`` to its physical value;
    ``bounds_phys`` clamps the difference steps (one-sided at a bound, with the realised denominator).
    ``xs_phys`` shape: (n_x,). Fails fast for drive-controlled models (their signal needs a drive).
    """
    if getattr(inner_model, "controls_drive", False):
        raise NotImplementedError("The optimal sweep spacing is not implemented for drive-controlled models.")
    names = tuple(inner_model.parameter_names())
    missing = [n for n in names if n not in params_phys]
    if missing:
        raise ValueError(f"tabulate_sensitivity: no value for parameters {missing}")
    base = [float(params_phys[n]) for n in names]
    xs = np.asarray(xs_phys, dtype=np.float64)
    y = inner_model.compute_many_float64(xs, inner_model.spec.unpack_params(base))
    grads = np.zeros((xs.size, len(names)), dtype=np.float64)
    for i, name in enumerate(names):
        lo, hi = bounds_phys[name]
        h = _GRAD_REL_STEP * (hi - lo) if hi > lo else _GRAD_REL_STEP * max(abs(base[i]), 1.0)
        up, dn = list(base), list(base)
        up[i] = min(base[i] + h, hi) if hi > lo else base[i] + h
        dn[i] = max(base[i] - h, lo) if hi > lo else base[i] - h
        denom = up[i] - dn[i]
        if denom <= 0:
            raise ValueError(f"tabulate_sensitivity: parameter {name!r} has no room for a difference step ({lo}, {hi})")
        y_up = inner_model.compute_many_float64(xs, inner_model.spec.unpack_params(up))
        y_dn = inner_model.compute_many_float64(xs, inner_model.spec.unpack_params(dn))
        grads[:, i] = (y_up - y_dn) / denom
    return SignalSensitivity(names=names, xs_phys=xs, y=np.asarray(y, dtype=np.float64), grads=grads)


def grid_points_phys(lo_phys: float, hi_phys: float, fwhm_phys: float, k: float) -> NDArray[np.float64]:
    """The sweep grid of ``k`` points per FWHM: ``uniform_grid_phys(lo, hi, FWHM / k)``. shape: (n_points,)."""
    return uniform_grid_phys(lo_phys, hi_phys, fwhm_phys / k)


def pass_fisher_phys(
    sens: SignalSensitivity, grid_phys: NDArray[np.float64], offset_phys: float, sigma_phys: float
) -> NDArray[np.float64]:
    """Fisher information matrix of ONE pass over ``grid_phys`` when the dip sits ``offset_phys`` off the grid.

    The signal is sampled at ``grid_phys + offset_phys`` (equivalent to shifting the dip by ``-offset_phys``) with
    Gaussian noise ``sigma_phys``: ``F = G^T G / sigma^2``, ``G`` the (n_points, n_params) derivative table.

    Returns shape: (n_params, n_params), physical units, parameters in ``sens.names`` order.
    """
    if not (sigma_phys > 0.0) or not math.isfinite(sigma_phys):
        raise ValueError(f"pass_fisher_phys: sigma must be positive and finite, got {sigma_phys!r}")
    g = _interp_rows(sens.xs_phys, sens.grads, grid_phys + offset_phys)
    return g.T @ g / (sigma_phys * sigma_phys)


def _interp_rows(xs: NDArray[np.float64], table: NDArray[np.float64], at: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear interpolation of the columns of ``table`` (shape (n_dense, n_cols) or (n_dense,)) at ``at`` (Hz).

    Fails fast outside the tabulated range. Returns shape: (len(at), n_cols) (or (len(at),)).
    """
    if at.min() < xs[0] or at.max() > xs[-1]:
        raise ValueError("_interp_rows: a sample lies outside the tabulated range.")
    idx = np.clip(np.searchsorted(xs, at, side="right") - 1, 0, xs.size - 2)
    w = (at - xs[idx]) / (xs[idx + 1] - xs[idx])
    if table.ndim == 1:
        return (1.0 - w) * table[idx] + w * table[idx + 1]
    return (1.0 - w)[:, None] * table[idx] + w[:, None] * table[idx + 1]


def _passes_for_precision(
    fim_pass_phys: NDArray[np.float64],
    names: tuple[str, ...],
    bounds_phys: dict[str, tuple[float, float]],
    estimates_phys: dict[str, float],
    convergence_threshold: float,
    max_passes: int = _P_UNBOUNDED,
) -> int:
    """Smallest ``P`` in ``[1, max_passes]`` for which ``P`` passes meet the target-parameter convergence rule.

    ``max_passes`` is returned when even that many do not (an unidentifiable direction). The CRLB std of every
    parameter is evaluated through :func:`target_params_converged` (monotone in ``P``: gallop, then bisect).
    """
    ranges = ranges_from_bounds(list(names), bounds_phys)
    unit_fim = fim_pass_phys * np.outer(ranges, ranges)

    def converged(p: int) -> bool:
        stds_phys = single_shot_marginal_stds_from_fim(p * unit_fim, len(names)) * ranges
        return target_params_converged(
            names, bounds_phys, dict(zip(names, stds_phys.tolist(), strict=True)), estimates_phys, convergence_threshold
        )

    if converged(1):
        return 1
    hi = 2
    while hi < max_passes and not converged(hi):
        hi *= 2
    hi = min(hi, max_passes)
    if not converged(hi):
        return max_passes
    lo = hi // 2  # not converged (or 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if converged(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _relative_error_ratio(
    fim_phys: NDArray[np.float64],
    names: tuple[str, ...],
    bounds_phys: dict[str, tuple[float, float]],
    estimates_phys: dict[str, float],
    convergence_threshold: float,
) -> float:
    """Binding relative uncertainty of the CRLB of ``fim_phys``: the ``s`` where ``target_params_converged`` flips.

    The stds are divided by ``s``; ``target_params_converged`` is linear in them, so bisection (log scale) on ``s``
    finds the largest ratio of any parameter's (or the RMS) uncertainty to its ceiling. < 1 means converged.
    """
    ranges = ranges_from_bounds(list(names), bounds_phys)
    stds = single_shot_marginal_stds_from_fim(fim_phys * np.outer(ranges, ranges), len(names)) * ranges

    def converged(scale: float) -> bool:
        return target_params_converged(
            names,
            bounds_phys,
            dict(zip(names, (stds / scale).tolist(), strict=True)),
            estimates_phys,
            convergence_threshold,
        )

    lo, hi = -6.0, 12.0  # log10 of the scale
    if converged(10.0**lo):
        return 10.0**lo
    if not converged(10.0**hi):
        return 10.0**hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if converged(10.0**mid):
            hi = mid
        else:
            lo = mid
    return 10.0**hi


def _detection_window_scores(
    grid_phys: NDArray[np.float64],
    depth_over_sigma: NDArray[np.float64],
    width_lo_hz: float,
    width_hi_hz: float,
) -> tuple[float, int]:
    """Best one-pass boxcar GLRT score and the number of windows :func:`find_dips` tests on this uniform grid.

    ``depth_over_sigma`` shape: (n_points,) -- (baseline - signal) / sigma at each grid point. Returns
    ``(best_score, n_windows)``; ``n_windows`` is 0 when no window fits the allowed widths.
    """
    n = grid_phys.size
    first = np.arange(n)
    score, _, _ = _window_scores(grid_phys, first, first + 1, depth_over_sigma, width_lo_hz, width_hi_hz)
    if score.size == 0:
        return 0.0, 0
    return float(score.max()), int(score.size)


_SCAN_CACHE: dict[bytes, OptimalSweepSpacing] = {}
_SCAN_CACHE_MAX = 64


def optimal_sweep_spacing(
    belief,
    *,
    noise_sigma_phys: float,
    lo_phys: float,
    hi_phys: float,
    max_steps: int = 0,
    max_measurements: int,
    convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
    p_fa: float = NVISION_DIP_P_FA,
) -> OptimalSweepSpacing:
    """The step of the plain sweep that needs the fewest measurements, per the module docstring.

    Args:
        belief: SMC belief whose model, physical bounds and priors define the signal (prior-mean parameters).
        noise_sigma_phys: TRUE Gaussian noise sigma of one measurement (oracle; > 0).
        lo_phys, hi_phys: Sweep window (Hz); the grid spans it, both ends included.
        max_steps: Longest pass allowed (0 = no limit); scanned ``k`` needing more points are skipped.
        max_measurements: Measurement budget of the run (reported through ``within_budget``).
        convergence_threshold: Relative convergence threshold of the stop rule.
        p_fa: Family-wise false-alarm probability of the dip detection.

    Raises:
        ValueError: on a non-positive or non-finite sigma, or when not even ``K_MIN`` fits ``max_steps``.
        NotImplementedError: for drive-controlled models.
    """
    if not (noise_sigma_phys > 0.0) or not math.isfinite(noise_sigma_phys):
        raise ValueError(f"optimal_sweep_spacing: noise sigma must be positive and finite, got {noise_sigma_phys!r}")
    if max_measurements < 1:
        raise ValueError(f"optimal_sweep_spacing: max_measurements must be >= 1, got {max_measurements!r}")
    inner = getattr(belief.model, "inner", belief.model)
    bounds_phys = dict(belief.physical_param_bounds)
    means = prior_mean_params_phys(belief)
    fwhm = prior_fwhm_phys(belief)

    # Dense tabulation of the signal and its derivatives, padded by the largest grid step (FWHM / K_MIN) each side.
    pad = fwhm / K_MIN
    xs_dense = uniform_grid_phys(lo_phys - pad, hi_phys + pad, fwhm / _DENSE_POINTS_PER_FWHM)
    sens = tabulate_sensitivity(inner, means, bounds_phys, xs_dense)
    in_window = (xs_dense >= lo_phys) & (xs_dense <= hi_phys)
    baseline = float(np.percentile(sens.y[in_window], 70.0))  # find_dips' baseline percentile, noise-free

    gamma_mean, gamma_std, gamma_min, gamma_max = hwhm_fwhm_prior_hz(bounds_phys)
    width_lo, width_hi = window_width_range_hz(gamma_mean, gamma_std, gamma_min, gamma_max)

    key = hashlib.blake2b(
        b"|".join(
            [
                np.ascontiguousarray(sens.xs_phys[[0, -1]]).tobytes(),
                np.ascontiguousarray(sens.y).tobytes(),
                np.ascontiguousarray(sens.grads).tobytes(),
                repr(
                    (
                        sens.names,
                        sorted(bounds_phys.items()),
                        sorted(means.items()),
                        noise_sigma_phys,
                        lo_phys,
                        hi_phys,
                        max_steps,
                        max_measurements,
                        convergence_threshold,
                        p_fa,
                        width_lo,
                        width_hi,
                        fwhm,
                        K_MIN,
                        K_MAX,
                        K_STEP,
                        N_OFFSETS,
                    )
                ).encode(),
            ]
        ),
        digest_size=16,
    ).digest()
    cached = _SCAN_CACHE.get(key)
    if cached is not None:
        return cached

    ks = np.arange(K_MIN, K_MAX + 0.5 * K_STEP, K_STEP)
    cost = np.full(ks.size, np.inf)
    detail: list[tuple[int, float, float, float, float] | None] = [None] * ks.size
    fims_by_k: list[list[NDArray[np.float64]]] = [[] for _ in ks]
    for ik, k in enumerate(ks):
        grid = grid_points_phys(lo_phys, hi_phys, fwhm, float(k))
        n_points = grid.size
        if max_steps > 0 and n_points > max_steps:
            continue
        spacing = float(grid[1] - grid[0])
        p_prec_all, p_det_all, p_all = [], [], []
        for offset in np.arange(N_OFFSETS) * (spacing / N_OFFSETS):
            fim = pass_fisher_phys(sens, grid, float(offset), noise_sigma_phys)
            fims_by_k[ik].append(fim)
            p_prec = _passes_for_precision(fim, sens.names, bounds_phys, means, convergence_threshold)
            depth = baseline - _interp_rows(sens.xs_phys, sens.y, grid + offset)
            best, n_windows = _detection_window_scores(grid, depth / noise_sigma_phys, width_lo, width_hi)
            if n_windows == 0 or best <= 0.0:
                p_det = _P_UNBOUNDED
            else:
                threshold = float(norm.isf(p_fa / n_windows))
                p_det = min(_P_UNBOUNDED, max(1, math.ceil((threshold / best) ** 2)))
            p_prec_all.append(p_prec)
            p_det_all.append(p_det)
            p_all.append(max(p_prec, p_det))
        cost[ik] = float(np.mean(p_all)) * n_points
        detail[ik] = (n_points, spacing, float(np.mean(p_all)), float(np.mean(p_prec_all)), float(np.mean(p_det_all)))

    if not np.isfinite(cost).any():
        raise ValueError(
            f"optimal_sweep_spacing: even {K_MIN:g} points per FWHM needs more than max_steps={max_steps} points."
        )
    best_i = int(np.argmin(cost))  # first minimum: the largest step among equal costs, deterministic
    n_points, spacing, passes, p_prec_mean, p_det_mean = detail[best_i]  # type: ignore[misc]
    error_ratio = float("nan")
    if cost[best_i] > max_measurements:
        # No k reaches the stop rule within the budget: minimise the expected error at the cap instead.
        err = np.full(ks.size, np.inf)
        for ik in np.flatnonzero(np.isfinite(cost)):
            p_cap = max(1, max_measurements // detail[ik][0])  # type: ignore[index]
            err[ik] = float(
                np.mean(
                    [
                        _relative_error_ratio(p_cap * f, sens.names, bounds_phys, means, convergence_threshold)
                        for f in fims_by_k[ik]
                    ]
                )
            )
        best_i = int(np.argmin(err))
        n_points, spacing = detail[best_i][0], detail[best_i][1]  # type: ignore[index]
        passes = float(max(1, max_measurements // n_points))
        cost[best_i] = passes * n_points
        error_ratio = float(err[best_i])
    result = OptimalSweepSpacing(
        k_opt=float(ks[best_i]),
        spacing_phys=spacing,
        n_points=n_points,
        passes_pred=passes,
        cost_pred=float(cost[best_i]),
        passes_precision=p_prec_mean,
        passes_detection=p_det_mean,
        within_budget=math.isnan(error_ratio),
        error_ratio_at_cap=error_ratio,
    )
    if len(_SCAN_CACHE) >= _SCAN_CACHE_MAX:
        _SCAN_CACHE.pop(next(iter(_SCAN_CACHE)))
    _SCAN_CACHE[key] = result
    return result
