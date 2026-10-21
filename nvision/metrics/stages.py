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

Steps are 1-indexed measurement counts. A stage that was never reached has ``None`` metrics -- for
a plain sweep, which does not stop when it has converged, a short or absent *refine* stage is a real
result, not missing data.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nvision.metrics.milestones import resolve_primary_param
from nvision.models.observer import RunResult
from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT, DriveSetting
from nvision.spectra.nv_center_drive import RABI_PER_AMP_BOUNDS

# delta: the refine stage may lose at most this fraction of the oracle's Fisher information (CRLB within 1/(1-delta)).
REFINE_INFO_LOSS = 0.1
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


def _primary_sensitivity(run_result: RunResult, primary: str) -> tuple[np.ndarray, float]:
    """``(g_t^2, max g^2)``: squared sensitivity of each step's measurement and of the best possible one.

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

    best = max(float(np.max(grad_sq(d, grid_phys))) for d in _oracle_drives(true_signal))
    if not best > 0.0:
        raise ValueError(f"The true signal does not depend on {primary!r} anywhere on the drive range.")

    measured = np.array(
        [
            grad_sq(s.obs.drive, np.array([x_lo_phys + s.obs.drive_freq_unit * (x_hi_phys - x_lo_phys)]))[0]
            for s in snapshots
        ]
    )
    if measured.max() > (1.0 + _GRID_MAX_TOLERANCE) * best:
        # The grid max is second-order accurate; missing it by more than this means the grid is too coarse.
        raise ValueError(
            f"A measurement beat the grid optimum by {measured.max() / best - 1:.2%}; increase _EFFICIENCY_GRID_POINTS."
        )
    return measured, best


def sensitivity_efficiency_trace(run_result: RunResult, primary: str) -> np.ndarray:
    """Per-step Fisher-information efficiency about ``primary``, shape: (n_steps,), in ``[0, 1]``.

    ``eta_t = g(x_t, d_t)^2 / max_{x, d} g(x, d)^2`` (see :func:`_primary_sensitivity`). See the module
    docstring for why this is the information relative to the CRLB-optimal measurement.
    """
    measured, best = _primary_sensitivity(run_result, primary)
    return measured / max(best, float(measured.max()))


def fisher_information_trace(run_result: RunResult, primary: str) -> np.ndarray:
    """Per-step Fisher information about ``primary`` at its true value, shape: (n_steps,), in Hz^-2.

    ``I_t = k_t g(x_t, d_t)^2 / sigma_t^2`` with ``k_t`` the step's averaged shots and ``sigma_t`` the per-shot
    noise standard deviation of its observation. Informations add over independent measurements, so
    ``(sum_t I_t)^(-1/2)`` is the Cramer-Rao bound (nuisance parameters known) on ``primary``.
    """
    measured, _ = _primary_sensitivity(run_result, primary)
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
    if n_steps == 0:
        return out

    primary = resolve_primary_param(run_result.true_signal.parameter_values().keys())
    if primary is None:
        # No zeeman_shift/hyperfine_shift and no free zero_field_freq (e.g. a fixed-center, no-Zeeman model):
        # the stages are undefined,
        # same convention as the primary milestone (see resolve_primary_param).
        return out
    out["stage_primary_param"] = primary
    eta = sensitivity_efficiency_trace(run_result, primary)
    shots = np.array([float(s.obs.n_shots) for s in snapshots])

    unc_trace = [float(s.belief.uncertainty().get(primary, math.nan)) for s in snapshots]
    true_value = float(run_result.true_signal.get_param_value(primary))
    err_trace = [abs(float(s.belief.estimates()[primary]) - true_value) for s in snapshots]  # Hz, (n_steps,)

    # Measurement indices (0-based). The measurement that reveals the dip ends the search; refine is the
    # efficient tail, which starts after that measurement.
    focus_idx: int | None = next((i for i, s in enumerate(snapshots) if s.belief.current_dips()), None)
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
    return out
