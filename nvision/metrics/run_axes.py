"""Per-run axes and markers every locator is judged on, read off the recorded run identically for every locator.

The article reads a run through four regimes (search, localize, converge, refine) whose boundaries are the run's own
dip detection and the onset of a Gaussian posterior (``docs/evaluation_metrics.md``). This module stores what those
analyses need per run and nothing else:

* ``detection_step``: the first step the locator's dip test (``NVISION_DIP_P_FA``, the one it narrows its focus
  with) shows a dip and keeps showing one for ``BOUNDARY_HOLD_STEPS`` steps. It ends the search; van Trees restarts
  there (:mod:`nvision.metrics.detection_aligned`).
* ``i_max`` = ``I_max``: the Fisher information about the primary parameter one measurement carries when
  taken at the steepest point of the true dip (shots x ``max g^2 / sigma^2``, Hz^-2 per step, the mean over steps).
* the axes: the stop target ``tau``, the true and prior FWHM, the drive window and the prior variance of the primary
  parameter (``sigma0_phys`` = the prior's standard deviation, ``W / sqrt(12)`` for the uniform global-search prior,
  article Eq. prior). The prior's *information* is not stored: van Trees does not apply to the uniform window.

Fisher information
------------------
Measurement ``t`` at drive frequency ``x_t`` (and drive ``d_t``) with ``k_t`` averaged shots carries, about the
primary parameter ``theta`` at its true value, ``I(x_t) = k_t g(x_t, d_t)^2 / sigma^2`` with
``g = d signal / d theta``. ``g`` is taken on the true signal, so these are simulation-only ground truths that grade
sweeps and SBED identically. For a drive-controlled model the best point also picks the drive amplitude (and, when
pulsed, the pulse duration, from 0.1x to 10x the true pi-pulse).

Steps are 1-indexed measurement counts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from nvision.metrics.milestones import resolve_primary_param
from nvision.models.observer import RunResult
from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT, DriveSetting
from nvision.spectra.nv_center_drive import RABI_PER_AMP_BOUNDS

# Consecutive steps the dip test must keep showing a dip, so a single noise fluctuation does not count as detection.
BOUNDARY_HOLD_STEPS = 3
_EFFICIENCY_GRID_POINTS = 4001
_GRID_MAX_TOLERANCE = 0.01
_ORACLE_DRIVE_AMPS = 48
# Pulsed oracle: coarser amplitudes x pulse durations (as multiples of the true pi-pulse, which is 1.0). The
# multiples span true/estimated Rabi frequency over the prior bounds, the most a calibration can be off.
_ORACLE_PULSED_AMPS = 16
_ORACLE_DURATION_RATIOS = np.geomspace(
    RABI_PER_AMP_BOUNDS[0] / RABI_PER_AMP_BOUNDS[1], RABI_PER_AMP_BOUNDS[1] / RABI_PER_AMP_BOUNDS[0], 17
)

# Every key compute_run_axes returns (the results-row columns).
RUN_AXIS_KEYS: tuple[str, ...] = (
    "primary_param",
    "detection_step",
    "i_max",
    "sigma_target_phys",
    "fwhm_phys",
    "prior_fwhm_phys",
    "window_phys",
    "window_over_fwhm",
    "sigma0_phys",
    "prior_kind",
    "target_cost_step",
    "line_walk_step_phys",
)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _oracle_drives(true_signal: Any) -> list[DriveSetting | None]:
    """Drive settings the oracle may choose: ``[None]`` unless the model is drive-controlled."""
    model = true_signal.model
    if not model.controls_drive:
        return [None]
    from nvision.spectra.nv_center_drive import pi_pulse_duration_s

    amps = np.geomspace(DRIVE_AMP_MIN_UNIT, 1.0, _ORACLE_DRIVE_AMPS)
    if not model.is_pulsed:
        return [DriveSetting(float(a)) for a in amps]
    rabi_per_amp = true_signal.get_param_value("rabi_per_amp")
    # Pulses of any length, not just the true pi-pulse: a locator calibrates its pulse from its *estimated* Rabi
    # frequency, so it can legitimately pick a longer or shorter pulse, and a longer pulse is often far more
    # sensitive (fringes) than the pi-pulse. Durations span the reach of that calibration error.
    pulsed_amps = np.geomspace(DRIVE_AMP_MIN_UNIT, 1.0, _ORACLE_PULSED_AMPS)
    return [
        DriveSetting(float(a), float(m * pi_pulse_duration_s(rabi_per_amp * float(a))))
        for a in pulsed_amps
        for m in _ORACLE_DURATION_RATIOS
    ]


class _Sensitivity(NamedTuple):
    """Squared sensitivity of each measurement and of the best possible one."""

    measured: np.ndarray  # shape: (n_steps,), (signal / Hz)^2 at each step's drive frequency and drive
    best: float  # max over the drive-frequency range (and drive settings) of g^2, (signal / Hz)^2


def _grad_sq_fn(
    true_signal: Any, primary: str, base: list[float]
) -> Callable[[DriveSetting | None, np.ndarray], np.ndarray]:
    """Squared sensitivity ``(d signal / d primary)^2`` of the model at parameter values ``base``.

    ``base``: the model's packed parameter values (``model.spec`` order). Returns ``grad_sq(drive, xs_phys)`` with
    ``xs_phys`` shape (n,) in Hz and result shape (n,) in (signal / Hz)^2, by a central difference on ``primary``.
    """
    names = list(true_signal.parameter_names)
    if primary not in names:
        raise ValueError(f"Primary parameter {primary!r} is not a parameter of the true signal {names}.")
    idx = names.index(primary)
    lo_bound, hi_bound = true_signal.get_param_bounds(primary)
    spec = true_signal.model.spec
    h = 1e-4 * (hi_bound - lo_bound)
    up, dn = list(base), list(base)
    up[idx], dn[idx] = min(base[idx] + h, hi_bound), max(base[idx] - h, lo_bound)
    if up[idx] <= dn[idx]:
        raise ValueError(f"Degenerate bounds for {primary!r}: {(lo_bound, hi_bound)}.")
    params_up, params_dn = spec.unpack_params(up), spec.unpack_params(dn)

    def grad_sq(drive: DriveSetting | None, xs_phys: np.ndarray) -> np.ndarray:
        model = true_signal.model_at(drive)
        y_up = model.compute_many_float64(xs_phys, params_up)
        y_dn = model.compute_many_float64(xs_phys, params_dn)
        return ((y_up - y_dn) / (up[idx] - dn[idx])) ** 2

    return grad_sq


def _primary_sensitivity(run_result: RunResult, primary: str) -> _Sensitivity:
    """:class:`_Sensitivity`: squared sensitivity of each step's measurement and of the best possible one.

    ``g = d signal / d primary`` on the true signal, in signal / Hz; ``g_t^2`` has shape (n_steps,), taken at the
    step's physical drive frequency (Hz) and drive. The max is over a dense grid of the full drive-frequency
    range (and, for drive-controlled models, of drive amplitudes).
    """
    snapshots = run_result.snapshots
    x_lo_phys, x_hi_phys = snapshots[0].belief.drive_freq_bounds_phys
    xs_all = np.array(
        [x_lo_phys + s.obs.drive_freq_unit * (x_hi_phys - x_lo_phys) for s in snapshots],
        dtype=np.float64,
    )
    return primary_sensitivity_at(
        run_result.true_signal, primary, x_lo_phys, x_hi_phys, xs_all, [s.obs.drive for s in snapshots]
    )


def primary_sensitivity_curve(
    true_signal: Any, primary: str, x_lo_phys: float, x_hi_phys: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(grid_phys, g2)``: squared sensitivity of a fixed-drive ``true_signal`` on an even grid of its drive window.

    Both shape (``_EFFICIENCY_GRID_POINTS``,): drive frequency (Hz) and ``(d signal / d primary)^2`` in
    (signal / Hz)^2, the curve whose maximum is ``I*``'s ``best`` and whose mean is what a uniform sweep takes.
    Raises for a drive-controlled model (its sensitivity also depends on the drive setting).
    """
    if true_signal.model.controls_drive:
        raise ValueError(
            "primary_sensitivity_curve is for fixed-drive models; a drive-controlled one has no single curve."
        )
    grad_sq = _grad_sq_fn(
        true_signal, primary, [float(v) for v in true_signal.model.spec.pack_params(true_signal.typed_parameters)]
    )
    grid_phys = np.linspace(x_lo_phys, x_hi_phys, _EFFICIENCY_GRID_POINTS)
    return grid_phys, grad_sq(None, grid_phys)


def primary_sensitivity_at(
    true_signal: Any,
    primary: str,
    x_lo_phys: float,
    x_hi_phys: float,
    xs_phys: np.ndarray,
    drives: Sequence[DriveSetting | None],
) -> _Sensitivity:
    """:class:`_Sensitivity` of ``true_signal`` for measurements at ``xs_phys`` (the array-in core of the above).

    ``xs_phys``: shape (n_steps,), drive frequency (Hz) of each measurement; ``drives``: length n_steps, the drive
    setting of each (all ``None`` unless the model is drive-controlled). ``[x_lo_phys, x_hi_phys]`` is the drive
    range the best point is searched on. Also what :mod:`nvision.metrics.refine_rates` calls for cached runs,
    which have no belief snapshots.
    """
    spec = true_signal.model.spec
    grad_sq = _grad_sq_fn(true_signal, primary, [float(v) for v in spec.pack_params(true_signal.typed_parameters)])
    grid_phys = np.linspace(x_lo_phys, x_hi_phys, _EFFICIENCY_GRID_POINTS)

    best, best_g2 = 0.0, None
    for d in _oracle_drives(true_signal):
        g2 = grad_sq(d, grid_phys)
        if float(np.max(g2)) > best:
            best, best_g2 = float(np.max(g2)), g2
    if best_g2 is None or not best > 0.0:
        raise ValueError(f"The true signal does not depend on {primary!r} anywhere on the drive range.")

    xs_all = np.asarray(xs_phys, dtype=np.float64)
    first_drive = drives[0]
    if all(d == first_drive for d in drives):
        measured = grad_sq(first_drive, xs_all)
    else:
        measured = np.empty(len(drives), dtype=np.float64)
        from collections import defaultdict

        by_drive: dict[DriveSetting | None, list[int]] = defaultdict(list)
        for i, d in enumerate(drives):
            by_drive[d].append(i)
        for d, idxs in by_drive.items():
            idx_arr = np.array(idxs, dtype=np.intp)
            measured[idx_arr] = grad_sq(d, xs_all[idx_arr])

    if measured.max() > (1.0 + _GRID_MAX_TOLERANCE) * best:
        # The grid max is second-order accurate; missing it by more than this means the grid is too coarse.
        raise ValueError(
            f"A measurement beat the grid optimum by {measured.max() / best - 1:.2%}; increase _EFFICIENCY_GRID_POINTS."
        )
    return _Sensitivity(measured, best)


def fisher_information_trace(run_result: RunResult, primary: str) -> np.ndarray:
    """Per-step Fisher information about ``primary`` at its true value, shape: (n_steps,), in Hz^-2.

    ``I_t = k_t g(x_t, d_t)^2 / sigma_t^2`` with ``k_t`` the step's averaged shots and ``sigma_t`` the per-shot
    noise standard deviation of its observation. Informations add over independent measurements, so
    ``(sum_t I_t)^(-1/2)`` is the Cramer-Rao bound (nuisance parameters known) on ``primary``.
    """
    measured = _primary_sensitivity(run_result, primary).measured
    shots = np.array([float(s.obs.n_shots) for s in run_result.snapshots])
    noise_std = np.array([float(s.obs.noise_std) for s in run_result.snapshots])
    return shots * measured / noise_std**2


def _first_held(holds: Callable[[int], bool], first: int, n_steps: int) -> int | None:
    """Earliest index ``n >= first`` with ``holds(i)`` true for the ``BOUNDARY_HOLD_STEPS`` steps ``n..n+hold-1``.

    The window must lie inside the run (``n + hold <= n_steps``). ``holds`` is evaluated lazily and only on the
    steps needed, since a dip scan per step is not free.
    """
    for n in range(first, n_steps - BOUNDARY_HOLD_STEPS + 1):
        if all(holds(i) for i in range(n, n + BOUNDARY_HOLD_STEPS)):
            return n
    return None


def compute_run_axes(run_result: RunResult) -> dict[str, Any]:
    """The detection marker, ``I_max`` and the axes of one run (all JSON-serialisable scalars or ``None``).

    * ``primary_param``: the parameter the run is judged on (``None``, and every other field ``None`` too, when
      the model has no primary parameter -- see ``resolve_primary_param``).
    * ``detection_step``: 1-indexed step the locator's dip test first shows a dip and keeps showing one.
    * ``i_max``: ``I_max`` (Hz^-2 per step), the mean over steps.
    * ``sigma_target_phys``: the stop target tau (Hz), the claimed-sigma threshold the run stops on.
    * ``fwhm_phys`` (``None`` for drive-controlled models, whose width depends on the drive),
      ``prior_fwhm_phys`` (the prior's mean FWHM), ``window_phys`` (the drive-frequency range) and
      ``window_over_fwhm`` = window over true FWHM.
    * ``sigma0_phys`` = sigma_0, the standard deviation (Hz) of the primary parameter's prior, from the locator's prior;
      ``prior_kind`` its shape (``uniform``: global search; ``gaussian``: tracking, van Trees from n = 0).
    * ``line_walk_step_phys``: std (Hz) of the true line's random-walk step per measurement (0 = still line).
    * ``target_cost_step``: first step from which the true error stays ``<= tau`` for ``BOUNDARY_HOLD_STEPS``
      steps (the cost to the stop target; ``None`` if never).
    """
    snapshots = run_result.snapshots
    n_steps = len(snapshots)
    out: dict[str, Any] = {key: None for key in RUN_AXIS_KEYS}
    if n_steps == 0:
        return out
    primary = resolve_primary_param(run_result.true_signal.parameter_values().keys())
    if primary is None:
        # No zeeman_shift/hyperfine_shift and no free zero_field_freq (e.g. a fixed-center, no-Zeeman model).
        return out
    out["primary_param"] = primary

    dip_cache: dict[int, bool] = {}

    def shows_dip(i: int) -> bool:
        if i not in dip_cache:
            dip_cache[i] = bool(snapshots[i].belief.current_dips())
        return dip_cache[i]

    dip_idx = _first_held(shows_dip, 0, n_steps)
    out["detection_step"] = None if dip_idx is None else dip_idx + 1

    truth = run_result.truth_trace(primary)  # shape: (n_steps,), Hz
    estimates = np.array([float(s.belief.estimates()[primary]) for s in snapshots])  # shape: (n_steps,), Hz
    err_trace = np.abs(estimates - truth)  # Hz
    out["line_walk_step_phys"] = float(run_result.line_walk_step_phys)
    _add_axes(run_result, primary, err_trace, out)
    return out


def prior_variance_phys(
    prior_entry: Any,
    bounds_phys: tuple[float, float],
    broad_fraction: float = 0.0,
) -> float:
    """Exact variance (Hz^2) of the prior the SMC belief draws one parameter's particles from.

    ``prior_entry`` is that parameter's entry of ``SMCMarginalDistribution.priors`` (unit-cube form), ``bounds_phys``
    its ``physical_param_bounds`` (lo, hi) in Hz and ``broad_fraction`` the share of particles drawn uniformly over
    the bounds on top of a Gaussian prior (``n_broad / num_particles``). The three shapes the belief samples are:

    * ``None``: uniform over the bounds, ``W^2 / 12``.
    * ``(mu_unit, std_unit)``: ``N(mu, std)`` truncated to the unit interval, mixed with ``broad_fraction`` of
      ``U(0, 1)``; law of total variance, scaled by ``W^2``.
    * ``("sin^2", k)``: ``p(f) ~ sin^2(k (f - lo))`` over the physical bounds (``k`` in rad/Hz), closed form.

    Anything else, and the truncated-normal cases the sampler does not draw from a truncated normal (``std <= 0`` or
    the far-tail exponential branch), raise ``ValueError``: there is no approximate fallback.
    """
    from nvision.belief.smc_marginal import _TRUNC_NORM_TAIL_SIGMAS

    lo_phys, hi_phys = float(bounds_phys[0]), float(bounds_phys[1])
    width_phys = hi_phys - lo_phys
    if not (width_phys > 0 and math.isfinite(width_phys)):
        raise ValueError(f"Prior bounds must be a finite interval with hi > lo, got {bounds_phys}")
    if prior_entry is None:
        return width_phys**2 / 12.0
    if isinstance(prior_entry, tuple) and len(prior_entry) == 2 and prior_entry[0] == "sin^2":
        return _sin2_prior_variance_phys(float(prior_entry[1]), width_phys)
    if isinstance(prior_entry, tuple) and len(prior_entry) == 2:
        from scipy.stats import truncnorm

        mu_unit, std_unit = float(prior_entry[0]), float(prior_entry[1])
        if not (std_unit > 0 and math.isfinite(std_unit) and math.isfinite(mu_unit)):
            raise ValueError(f"Gaussian prior needs a finite std > 0, got {prior_entry}")
        a, b = (0.0 - mu_unit) / std_unit, (1.0 - mu_unit) / std_unit
        if a >= _TRUNC_NORM_TAIL_SIGMAS or b <= -_TRUNC_NORM_TAIL_SIGMAS:
            raise ValueError(f"Gaussian prior {prior_entry} lies in the sampler's far-tail branch; no exact variance")
        if not 0.0 <= broad_fraction <= 1.0:
            raise ValueError(f"broad_fraction must be in [0, 1], got {broad_fraction}")
        mean_g, var_g = (float(v) for v in truncnorm.stats(a, b, loc=mu_unit, scale=std_unit, moments="mv"))
        mean_u, var_u = 0.5, 1.0 / 12.0
        f = broad_fraction
        mean = (1.0 - f) * mean_g + f * mean_u
        var_unit = (1.0 - f) * (var_g + mean_g**2) + f * (var_u + mean_u**2) - mean**2
        return var_unit * width_phys**2
    raise ValueError(f"Unsupported prior entry {prior_entry!r}: cannot compute its variance")


def _sin2_prior_variance_phys(k: float, width_phys: float) -> float:
    """Variance (Hz^2) of ``p(u) ~ sin^2(k u)`` on ``[0, W]``, from the closed-form moments.

    ``sin^2(k u) = (1 - cos(c u)) / 2`` with ``c = 2k``, so ``I_n = W^(n+1) / (2(n+1)) - C_n / 2`` with
    ``C_n = int_0^W u^n cos(c u) du``.
    """
    if not (k > 0 and math.isfinite(k)):
        raise ValueError(f"sin^2 prior needs a finite k > 0, got {k}")
    c, w = 2.0 * k, width_phys
    s, co = math.sin(c * w), math.cos(c * w)
    c0 = s / c
    c1 = w * s / c + (co - 1.0) / c**2
    c2 = w**2 * s / c + 2.0 * w * co / c**2 - 2.0 * s / c**3
    i0 = w / 2.0 - c0 / 2.0
    i1 = w**2 / 4.0 - c1 / 2.0
    i2 = w**3 / 6.0 - c2 / 2.0
    mean = i1 / i0
    return i2 / i0 - mean**2


def belief_prior_kind(belief: Any, name: str) -> str:
    """``"uniform"`` (global search over the prior window), ``"gaussian"`` (tracking: van Trees holds from n = 0 with
    ``J0 = 1 / sigma0^2``) or ``"sin^2"``: the shape of the prior ``belief`` started from on parameter ``name``."""
    from nvision.belief.grid_marginal import GridMarginalDistribution
    from nvision.belief.smc_marginal import SMCMarginalDistribution

    if isinstance(belief, GridMarginalDistribution):
        return "uniform"
    if not isinstance(belief, SMCMarginalDistribution):
        raise TypeError(f"Cannot read the prior of a {type(belief).__name__}: needs an SMC or grid belief")
    entry = (belief.priors or {}).get(name)
    if entry is None:
        return "uniform"
    return "sin^2" if entry[0] == "sin^2" else "gaussian"


def belief_prior_variance_phys(belief: Any, name: str) -> float:
    """Variance (Hz^2) of the prior ``belief`` started from on parameter ``name`` (see :func:`prior_variance_phys`).

    Read from the belief's configuration (bounds and prior entry), never from its particles, so it is the same for a
    snapshot taken after any number of measurements. ``SMCMarginalDistribution`` and ``GridMarginalDistribution``
    (the sweep belief: a flat grid over its bounds, so uniform; the continuous ``W^2 / 12`` is used, the grid being
    a discretisation of it) are supported; any other belief raises ``TypeError``.
    """
    from nvision.belief.grid_marginal import GridMarginalDistribution
    from nvision.belief.smc_marginal import SMCMarginalDistribution

    if isinstance(belief, SMCMarginalDistribution):
        entry = (belief.priors or {}).get(name)
        broad = 0.0
        if entry is not None and entry[0] != "sin^2":
            broad = round(belief.prior_broad_fraction * belief.num_particles) / belief.num_particles
        return prior_variance_phys(entry, belief.physical_param_bounds[name], broad)
    if isinstance(belief, GridMarginalDistribution):
        return prior_variance_phys(None, belief.physical_param_bounds[name])
    raise TypeError(f"Cannot read the prior of a {type(belief).__name__}: J0 needs an SMC or grid belief")


def _true_fwhm_phys(true_signal: Any) -> float | None:
    """Effective FWHM (Hz) of the true dip, ``2 * effective_hwhm``; ``None`` when the width depends on the drive."""
    from nvision.spectra.nv_center import effective_hwhm

    if true_signal.model.controls_drive:
        return None
    return 2.0 * float(effective_hwhm(true_signal.parameter_values()))


def _add_axes(run_result: RunResult, primary: str, err_trace: np.ndarray, out: dict[str, Any]) -> None:
    """Fill ``I_max`` and the axes of :func:`compute_run_axes`.

    ``err_trace``: shape (n_steps,), true error ``|estimate - truth|`` (Hz) of the primary parameter after each
    measurement.
    """
    from nvision.belief.dip_detection import hwhm_fwhm_prior_hz
    from nvision.metrics.series import effective_convergence_threshold

    snapshots = run_result.snapshots
    sens = _primary_sensitivity(run_result, primary)
    shots = np.array([float(s.obs.n_shots) for s in snapshots])
    noise_std = np.array([float(s.obs.noise_std) for s in snapshots])
    fisher_star = float(np.mean(shots * sens.best / noise_std**2))  # Hz^-2 per step
    belief = snapshots[0].belief
    x_lo_phys, x_hi_phys = belief.drive_freq_bounds_phys
    fwhm_true = _true_fwhm_phys(run_result.true_signal)
    tau = effective_convergence_threshold(run_result, primary)
    out["i_max"] = _finite_or_none(fisher_star)
    out["sigma_target_phys"] = None if tau is None else _finite_or_none(float(tau))
    out["fwhm_phys"] = None if fwhm_true is None else _finite_or_none(fwhm_true)
    out["prior_fwhm_phys"] = _finite_or_none(float(hwhm_fwhm_prior_hz(belief.physical_param_bounds)[0]))
    out["window_phys"] = _finite_or_none(float(x_hi_phys - x_lo_phys))
    out["window_over_fwhm"] = None if not fwhm_true else _finite_or_none(float(x_hi_phys - x_lo_phys) / fwhm_true)
    out["sigma0_phys"] = _finite_or_none(math.sqrt(belief_prior_variance_phys(belief, primary)))
    out["prior_kind"] = belief_prior_kind(belief, primary)
    if tau is not None and float(tau) > 0:
        passed = _first_held(lambda i: err_trace[i] <= float(tau), 0, len(err_trace))
        if passed is not None:
            out["target_cost_step"] = passed + 1
