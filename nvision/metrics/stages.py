"""Three-stage decomposition of a locator run, detected identically for every locator.

A search for a resonance whose other parameters are already known passes through three stages:

1. **search** -- until the measurements first show a dip (the *focus window* exists).
2. **focus** -- from then until every measurement is taken where it carries the most information about the
   primary parameter (the *max-information* point).
3. **refine** -- from there to the end of the run, each measurement gaining the maximum (Cramer-Rao) information.

The boundaries are read off the run's per-step snapshots by one function, never from a locator's own
bookkeeping, so a sweep and an SBED run are compared by the same rule.

* ``focus_found_step``: first step whose belief has a significant dip, per
  :meth:`~nvision.belief.smc_marginal.SMCMarginalDistribution.current_dips` (the deterministic
  ``find_dips`` of the measurements and the conjugate noise estimate).
* ``max_info_step``: first step ``n`` after ``focus_found_step`` from which the measurements, taken
  together, gather at least ``1 - REFINE_INFO_LOSS`` of the Fisher information an oracle would:

  .. math:: \\sum_{t \\ge n} I(x_t) \\ge (1 - \\delta) \\sum_{t \\ge n} I^*_t

Fisher information
------------------
Measurement ``t`` at drive frequency ``x_t`` (and drive ``d_t``) with ``k_t`` averaged shots carries, about the
primary parameter ``theta`` at its true value, ``I(x_t) = k_t g(x_t, d_t)^2 / sigma^2`` with
``g = d signal / d theta`` (Gaussian noise of standard deviation ``sigma`` per shot). The oracle knows
``theta`` and spends the same shots at the best point, ``I^*_t = k_t max_{x, d} g(x, d)^2 / sigma^2``.
``sigma`` cancels, so the per-step **efficiency** is ``eta_t = g(x_t, d_t)^2 / max g^2`` in ``[0, 1]`` and the
rule above is a shot-weighted tail mean, ``sum k_t eta_t / sum k_t >= 1 - delta``.

Why this rule: measurements are independent given ``theta``, so their Fisher information adds and the
Cramer-Rao bound on ``theta`` from steps ``n..N`` is ``1 / sum_{t>=n} I(x_t)``. The rule therefore guarantees
that the refine stage's CRLB is within a factor ``1 / (1 - delta)`` of the best any design could reach with
the same measurements -- i.e. in stage 3 the locator gains (nearly) the CRLB-optimal information at every
step. A steady information rate alone is not enough: a locator stuck sampling a weak point is steady too, at
a low ``eta``. ``eta`` uses the true signal, not the locator's belief (whose own precision is overconfident),
so it is a simulation-only ground truth that grades sweeps and SBED identically.

The locator never measures *exactly* at the optimum -- it aims at the optimum of its estimate, off by its
estimation error ``eps``. That costs only second order: the optimum is a maximum of ``eta``, so
``eta(x* + eps) ~ 1 - (kappa / 2) eps^2``. For a Lorentzian (HWHM ``gamma``) the optimum is the steepest slope
point ``|x - x0| = gamma / sqrt(3)`` and ``eta ~ 1 - 4.5 (eps / gamma)^2``: ``delta = 0.1`` accepts measuring
within about ``+-0.15 gamma`` of it -- the locally-linear regime where the estimate is good enough that the
design no longer depends on it.

Assumptions: additive noise whose ``sigma`` does not depend on ``x`` or the drive, and nuisance parameters
known (a model whose other parameters were classified before). With free nuisance parameters a single
measurement's information about ``theta`` alone is the diagonal term, an upper bound on the efficient
(Schur-complement) information. For a drive-controlled model the oracle also picks the drive amplitude
(and, when pulsed, the pulse duration, from 0.1x to 10x the true pi-pulse).

Oracle per stage
----------------
Each stage is also compared with the fewest measurements any locator could need, from the same truth-side Fisher
information: ``I* = max g^2 / sigma^2`` per step. Focus ends at the precision ``eps`` where eta stays >= ``1 - delta``
(half-width of that region around the best point), so its bound is ``1 / (eps^2 I*)``; refine ends at the stop target
``tau``, so its bound is ``(tau^-2 - u3^-2) / I*``. Both fall as 1 / SNR^2. The efficiency (bound / steps spent) and the
axes (true / prior FWHM, window) are stored with the stage metrics; see :func:`compute_stage_metrics` and
``docs/evaluation_metrics.md``.

Steps are 1-indexed measurement counts. A stage that was never reached has ``None`` metrics -- for
a plain sweep, which does not stop when it has converged, a short or absent *refine* stage is a real
result, not missing data.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from nvision.metrics.milestones import resolve_primary_param
from nvision.models.observer import RunResult
from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT, DriveSetting
from nvision.spectra.nv_center_drive import RABI_PER_AMP_BOUNDS

# delta: the refine stage may lose at most this fraction of the oracle's Fisher information (CRLB within 1/(1-delta)).
REFINE_INFO_LOSS = 0.1
# Family-wise false-alarm level of the dip scan that marks the start of the focus stage. Looser than the
# locator's NVISION_DIP_P_FA: the uncertainty already drops like a focus stage before a dip passes the
# locator's test, so the metric declares the focus stage earlier than the locator acts on it.
FOCUS_FOUND_P_FA = 0.5
_EFFICIENCY_GRID_POINTS = 4001
_GRID_MAX_TOLERANCE = 0.01
_ORACLE_DRIVE_AMPS = 48
# Pulsed oracle: coarser amplitudes x pulse durations (as multiples of the true pi-pulse, which is 1.0). The
# multiples span true/estimated Rabi frequency over the prior bounds, the most a calibration can be off.
_ORACLE_PULSED_AMPS = 16
_ORACLE_DURATION_RATIOS = np.geomspace(
    RABI_PER_AMP_BOUNDS[0] / RABI_PER_AMP_BOUNDS[1], RABI_PER_AMP_BOUNDS[1] / RABI_PER_AMP_BOUNDS[0], 17
)

STAGE_NAMES: tuple[str, str, str] = ("search", "focus", "refine")

# Every key compute_stage_metrics returns (the results-row columns).
STAGE_METRIC_KEYS: tuple[str, ...] = (
    "stage_primary_param",
    "focus_found_step",
    "max_info_step",
    *(
        f"stage{k}_{field}"
        for k in (1, 2, 3)
        for field in ("steps", "unc_start", "unc_end", "unc_ratio", "err_end", "sens_eff")
    ),
    # The oracle the stages are judged against and the axes that drive them (see compute_stage_metrics).
    "stage_fisher_star",
    "stage_eps_target_phys",
    "stage_tau_phys",
    "stage_fwhm_true_phys",
    "stage_prior_fwhm_phys",
    "stage_window_phys",
    "stage_window_fwhm",
    "stage2_oracle_steps",
    "stage2_efficiency",
    "stage2_wasted_share",
    "stage3_oracle_steps",
    "stage3_efficiency",
    "stage3_wasted_share",
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
    """Squared sensitivity of each measurement, of the best possible one, and how far from it eta stays >= 1 - delta."""

    measured: np.ndarray  # shape: (n_steps,), (signal / Hz)^2 at each step's drive frequency and drive
    best: float  # max over the drive-frequency range (and drive settings) of g^2, (signal / Hz)^2
    # Half-width (Hz) of the drive-frequency region around the best point where g^2 >= (1 - delta) best; None for a
    # drive-controlled model, whose best point is a drive setting (pulse fringes can be narrower than the grid).
    eps_phys: float | None


def _optimum_tolerance_phys(grid_phys: np.ndarray, g2: np.ndarray, best: float) -> float:
    """Half-width (Hz) of the contiguous region around ``argmax g2`` where ``g2 >= (1 - delta) best``.

    ``grid_phys``, ``g2``: shape (n_grid,), drive frequency (Hz) and squared sensitivity on it. This is how far the
    locator may aim from the optimum and still measure at eta >= 1 - delta (0.15 gamma for a Lorentzian HWHM gamma).
    A locator error eps in the primary parameter moves the aim point by eps (the dip shifts one to one), so it is
    also the primary-parameter precision at which focus ends.
    """
    floor = (1.0 - REFINE_INFO_LOSS) * best
    i = int(np.argmax(g2))
    lo = hi = i
    while lo > 0 and g2[lo - 1] >= floor:
        lo -= 1
    while hi < len(g2) - 1 and g2[hi + 1] >= floor:
        hi += 1
    eps = 0.5 * float(grid_phys[hi] - grid_phys[lo])
    if not eps > 0.0:
        raise ValueError("The efficiency grid is too coarse to resolve the region of near-optimal sensitivity.")
    return eps


def _primary_sensitivity(run_result: RunResult, primary: str) -> _Sensitivity:
    """:class:`_Sensitivity`: squared sensitivity of each step's measurement and of the best possible one.

    ``g = d signal / d primary`` on the true signal, in signal / Hz; ``g_t^2`` has shape (n_steps,), taken at the
    step's physical drive frequency (Hz) and drive. The max is over a dense grid of the full drive-frequency
    range (and, for drive-controlled models, of drive amplitudes).
    """
    true_signal = run_result.true_signal
    names = list(true_signal.parameter_names)
    if primary not in names:
        raise ValueError(f"Primary parameter {primary!r} is not a parameter of the true signal {names}.")
    idx = names.index(primary)
    lo_bound, hi_bound = true_signal.get_param_bounds(primary)
    spec = true_signal.model.spec
    base = [float(v) for v in spec.pack_params(true_signal.typed_parameters)]
    h = 1e-4 * (hi_bound - lo_bound)
    up, dn = list(base), list(base)
    up[idx], dn[idx] = min(base[idx] + h, hi_bound), max(base[idx] - h, lo_bound)
    if up[idx] <= dn[idx]:
        raise ValueError(f"Degenerate bounds for {primary!r}: {(lo_bound, hi_bound)}.")
    params_up, params_dn = spec.unpack_params(up), spec.unpack_params(dn)

    snapshots = run_result.snapshots
    x_lo_phys, x_hi_phys = snapshots[0].belief.drive_freq_bounds_phys
    grid_phys = np.linspace(x_lo_phys, x_hi_phys, _EFFICIENCY_GRID_POINTS)

    def grad_sq(drive: DriveSetting | None, xs_phys: np.ndarray) -> np.ndarray:
        """Squared sensitivity d signal / d primary at ``xs_phys`` (Hz) under ``drive``; shape: (len(xs_phys),)."""
        model = true_signal.model_at(drive)
        y_up = model.compute_many_float64(xs_phys, params_up)
        y_dn = model.compute_many_float64(xs_phys, params_dn)
        return ((y_up - y_dn) / (up[idx] - dn[idx])) ** 2

    best, best_g2 = 0.0, None
    for d in _oracle_drives(true_signal):
        g2 = grad_sq(d, grid_phys)
        if float(np.max(g2)) > best:
            best, best_g2 = float(np.max(g2)), g2
    if best_g2 is None or not best > 0.0:
        raise ValueError(f"The true signal does not depend on {primary!r} anywhere on the drive range.")

    xs_all = np.array(
        [x_lo_phys + s.obs.drive_freq_unit * (x_hi_phys - x_lo_phys) for s in snapshots],
        dtype=np.float64,
    )
    first_drive = snapshots[0].obs.drive
    if all(s.obs.drive == first_drive for s in snapshots):
        measured = grad_sq(first_drive, xs_all)
    else:
        measured = np.empty(len(snapshots), dtype=np.float64)
        from collections import defaultdict

        by_drive: dict[DriveSetting | None, list[int]] = defaultdict(list)
        for i, s in enumerate(snapshots):
            by_drive[s.obs.drive].append(i)
        for d, idxs in by_drive.items():
            idx_arr = np.array(idxs, dtype=np.intp)
            measured[idx_arr] = grad_sq(d, xs_all[idx_arr])

    if measured.max() > (1.0 + _GRID_MAX_TOLERANCE) * best:
        # The grid max is second-order accurate; missing it by more than this means the grid is too coarse.
        raise ValueError(
            f"A measurement beat the grid optimum by {measured.max() / best - 1:.2%}; increase _EFFICIENCY_GRID_POINTS."
        )
    eps = None if true_signal.model.controls_drive else _optimum_tolerance_phys(grid_phys, best_g2, best)
    return _Sensitivity(measured, best, eps)


def sensitivity_efficiency_trace(run_result: RunResult, primary: str) -> np.ndarray:
    """Per-step Fisher-information efficiency about ``primary``, shape: (n_steps,), in ``[0, 1]``.

    ``eta_t = g(x_t, d_t)^2 / max_{x, d} g(x, d)^2`` (see :func:`_primary_sensitivity`). See the module
    docstring for why this is the information relative to the CRLB-optimal measurement.
    """
    return _efficiency(_primary_sensitivity(run_result, primary))


def _efficiency(sens: _Sensitivity) -> np.ndarray:
    """``eta_t = g_t^2 / max g^2``, shape: (n_steps,), in ``[0, 1]`` (the grid max is second-order accurate)."""
    return sens.measured / max(sens.best, float(sens.measured.max()))


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


def _refine_start(eta: np.ndarray, shots: np.ndarray, first: int) -> int | None:
    """Earliest index ``n >= first`` whose tail ``n..end`` has shot-weighted mean efficiency >= ``1 - delta``.

    ``eta``, ``shots``: shape (n_steps,) -- per-step efficiency and number of averaged shots.
    """
    tail_info = np.cumsum((shots * eta)[::-1])[::-1]
    tail_shots = np.cumsum(shots[::-1])[::-1]
    for n in range(first, len(eta)):
        if tail_info[n] >= (1.0 - REFINE_INFO_LOSS) * tail_shots[n]:
            return n
    return None


def compute_stage_metrics(run_result: RunResult) -> dict[str, Any]:
    """Stage boundaries, per-stage durations and primary-parameter uncertainty change for one run.

    Returns a flat dict (all values JSON-serialisable scalars or ``None``):

    * ``stage_primary_param``: the parameter whose sensitivity defines stage 3 (``None``, and every other
      field ``None`` too, when the model has no primary parameter -- see ``resolve_primary_param``).
    * ``focus_found_step`` / ``max_info_step``: 1-indexed boundaries (``None`` if not reached).
    * ``stage{1,2,3}_steps``: measurements spent in the stage (``None`` if the stage was not reached).
    * ``stage{1,2,3}_unc_start`` / ``_unc_end`` / ``_unc_ratio``: primary-parameter uncertainty (Hz) at the
      stage's first and last step, and ``end / start`` (< 1 means the stage shrank the uncertainty).
    * ``stage{1,2,3}_err_end``: true error |estimate - truth| (Hz) of the primary parameter after the stage's last
      measurement (the accuracy reached by the end of the stage).
    * ``stage{1,2,3}_sens_eff``: mean sensitivity efficiency of the stage's measurements (module docstring).

    The oracle and the axes each stage is judged on (see ``docs/evaluation_metrics.md``). With ``I*`` the Fisher
    information about the primary parameter one measurement carries when taken at the best point (shots times
    ``max g^2 / sigma^2``, Hz^-2 per step), a lower bound on the measurements a stage needs is
    ``(precision to reach)^-2 / I*``; the efficiency is that bound over the steps actually spent:

    * ``stage_fisher_star``: ``I*`` (Hz^-2 per step), the mean over steps.
    * ``stage_eps_target_phys``: ``eps`` (Hz), how far from the best point a measurement may sit and still have
      eta >= 1 - delta; the precision at which focus ends (``None`` for drive-controlled models).
    * ``stage_tau_phys``: the stop target tau (Hz), the claimed-sigma threshold the run stops on.
    * ``stage_fwhm_true_phys`` (``None`` for drive-controlled models, whose width depends on the drive),
      ``stage_prior_fwhm_phys`` (the prior's mean FWHM), ``stage_window_phys`` (the drive-frequency range) and
      ``stage_window_fwhm`` = window over true FWHM: the axes the stages scale with.
    * ``stage2_oracle_steps`` = ``1 / (eps^2 I*)``: measurements to reach ``eps`` from scratch at the best point.
      ``stage2_efficiency`` = that over ``stage2_steps``.
    * ``stage3_oracle_steps`` = ``max(0, tau^-2 - u3^-2) / I*`` with ``u3`` the claimed sigma at refine start.
      ``stage3_efficiency`` = that over ``stage3_steps``.
    * ``stage{2,3}_wasted_share``: share of the stage's measurements with eta < 1 - delta.
    """
    snapshots = run_result.snapshots
    n_steps = len(snapshots)
    out: dict[str, Any] = {"stage_primary_param": None, "focus_found_step": None, "max_info_step": None}
    for k in (1, 2, 3):
        out[f"stage{k}_steps"] = None
        out[f"stage{k}_unc_start"] = None
        out[f"stage{k}_unc_end"] = None
        out[f"stage{k}_unc_ratio"] = None
        out[f"stage{k}_err_end"] = None
        out[f"stage{k}_sens_eff"] = None
    out.update({key: None for key in STAGE_METRIC_KEYS if key not in out})
    if n_steps == 0:
        return out

    primary = resolve_primary_param(run_result.true_signal.parameter_values().keys())
    if primary is None:
        # No zeeman_shift/hyperfine_shift and no free zero_field_freq (e.g. a fixed-center, no-Zeeman model):
        # the stages are undefined,
        # same convention as the primary milestone (see resolve_primary_param).
        return out
    out["stage_primary_param"] = primary
    sens = _primary_sensitivity(run_result, primary)
    eta = _efficiency(sens)
    shots = np.array([float(s.obs.n_shots) for s in snapshots])

    unc_trace = [float(s.belief.uncertainty().get(primary, math.nan)) for s in snapshots]
    true_value = float(run_result.true_signal.get_param_value(primary))
    err_trace = [abs(float(s.belief.estimates()[primary]) - true_value) for s in snapshots]  # Hz, (n_steps,)

    # Measurement indices (0-based). The measurement that reveals the dip ends the search; refine is the
    # efficient tail, which starts after that measurement.
    focus_idx: int | None = next(
        (i for i, s in enumerate(snapshots) if s.belief.current_dips(p_fa=FOCUS_FOUND_P_FA)), None
    )
    refine_idx: int | None = None
    if focus_idx is not None:
        refine_idx = _refine_start(eta, shots, focus_idx + 1)

    out["focus_found_step"] = None if focus_idx is None else focus_idx + 1
    out["max_info_step"] = None if refine_idx is None else refine_idx + 1

    # Each stage's measurements as [first, last] indices (None = not reached). Its uncertainty runs from the
    # belief before its first measurement (snapshot 0 for the search) to the belief after its last one.
    last = n_steps - 1
    stage_measurements: list[tuple[int, int] | None] = [
        (0, focus_idx if focus_idx is not None else last),
        None if focus_idx is None else (focus_idx + 1, refine_idx - 1 if refine_idx is not None else last),
        None if refine_idx is None else (refine_idx, last),
    ]
    for k, span in enumerate(stage_measurements, start=1):
        if span is None:
            continue
        first, final = span
        n_in_stage = final - first + 1  # 0 when refine starts right after the focus
        start, end = unc_trace[max(first - 1, 0)], unc_trace[max(final, first - 1, 0)]
        out[f"stage{k}_steps"] = n_in_stage
        out[f"stage{k}_unc_start"] = _finite_or_none(start)
        out[f"stage{k}_unc_end"] = _finite_or_none(end)
        out[f"stage{k}_unc_ratio"] = _finite_or_none(end / start) if start > 0 else None
        out[f"stage{k}_err_end"] = _finite_or_none(err_trace[max(final, first - 1, 0)])
        if n_in_stage > 0:
            w = shots[first : final + 1]
            out[f"stage{k}_sens_eff"] = float(np.sum(w * eta[first : final + 1]) / np.sum(w))
            if k > 1:
                out[f"stage{k}_wasted_share"] = float(np.mean(eta[first : final + 1] < 1.0 - REFINE_INFO_LOSS))
    _add_oracle_metrics(run_result, primary, sens, shots, out)
    return out


def _true_fwhm_phys(true_signal: Any) -> float | None:
    """Effective FWHM (Hz) of the true dip, ``2 * effective_hwhm``; ``None`` when the width depends on the drive."""
    from nvision.spectra.nv_center import effective_hwhm

    if true_signal.model.controls_drive:
        return None
    return 2.0 * float(effective_hwhm(true_signal.parameter_values()))


def _add_oracle_metrics(
    run_result: RunResult, primary: str, sens: _Sensitivity, shots: np.ndarray, out: dict[str, Any]
) -> None:
    """Fill the oracle / axis fields of :func:`compute_stage_metrics` (``out`` already holds the stage metrics).

    ``shots``: shape (n_steps,), averaged shots per step.
    """
    from nvision.belief.dip_detection import hwhm_fwhm_prior_hz
    from nvision.metrics.series import effective_convergence_threshold

    snapshots = run_result.snapshots
    noise_std = np.array([float(s.obs.noise_std) for s in snapshots])
    fisher_star = float(np.mean(shots * sens.best / noise_std**2))  # Hz^-2 per step
    belief = snapshots[0].belief
    x_lo_phys, x_hi_phys = belief.drive_freq_bounds_phys
    fwhm_true = _true_fwhm_phys(run_result.true_signal)
    tau = effective_convergence_threshold(run_result, primary)
    out["stage_fisher_star"] = _finite_or_none(fisher_star)
    out["stage_eps_target_phys"] = None if sens.eps_phys is None else _finite_or_none(sens.eps_phys)
    out["stage_tau_phys"] = None if tau is None else _finite_or_none(float(tau))
    out["stage_fwhm_true_phys"] = None if fwhm_true is None else _finite_or_none(fwhm_true)
    out["stage_prior_fwhm_phys"] = _finite_or_none(float(hwhm_fwhm_prior_hz(belief.physical_param_bounds)[0]))
    out["stage_window_phys"] = _finite_or_none(float(x_hi_phys - x_lo_phys))
    out["stage_window_fwhm"] = None if not fwhm_true else _finite_or_none(float(x_hi_phys - x_lo_phys) / fwhm_true)

    # Lower bounds: measurements at the best point, nuisance parameters known (module docstring).
    if sens.eps_phys is not None:
        oracle2 = 1.0 / (sens.eps_phys**2 * fisher_star)
        out["stage2_oracle_steps"] = _finite_or_none(oracle2)
        if out["stage2_steps"]:
            out["stage2_efficiency"] = _finite_or_none(oracle2 / out["stage2_steps"])
    u3 = out["stage3_unc_start"]
    if tau is not None and u3 is not None:
        oracle3 = max(0.0, float(tau) ** -2 - u3**-2) / fisher_star
        out["stage3_oracle_steps"] = _finite_or_none(oracle3)
        if out["stage3_steps"]:
            out["stage3_efficiency"] = _finite_or_none(oracle3 / out["stage3_steps"])
