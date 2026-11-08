"""Per-run columns the article's claims need and the cache does not keep (they need the belief or the true signal).

Computed at run time next to :func:`nvision.metrics.stages.compute_stage_metrics` (``nvision/runner/metrics.py``) and
stored in the results row; the aggregation over runs is in :mod:`nvision.metrics.claims`. The cache holds no particle
snapshots, so a run recorded before these columns existed shows a dash: **re-run** to get them (nothing is backfilled).

* ``stage_fisher_sweep`` (Hz^-2 per measurement): the information a sweep gains per step, the window average of
  ``k g(x)^2 / sigma^2`` over the drive window (article Eq. sweep-info). ``stage_fisher_star / stage_fisher_sweep`` is
  the measured ``R_inf`` of Eq. speedup for any window (the article's ``27 W / (8 pi Gamma)`` is its large-window form).
* ``stage_pit_end`` / ``stage_pit_refine_start``: the probability integral transform of the primary parameter, the
  particle posterior's CDF at the true value, after the last measurement and after the measurement that starts refine.
  Uniform over runs when the posterior is calibrated (``claims.pit_coverage``); this is the "particle-distribution
  quantiles" coverage of Sec. protocol.
* ``stage2_far_share`` / ``stage2_far_mass``: over the localize stage, the share of measurements placed farther than
  ``FAR_FWHM`` true FWHM from the dip, and the mean posterior mass (before each of those measurements) farther than
  ``FAR_FWHM`` FWHM from the true value. Sec. eig says the variance-form EIG keeps spending measurements on residual
  mass far from the dip; the claim holds if the share exceeds the mass. A measurement is "far" when it is more than
  ``FAR_FWHM * FWHM`` from every point of the noise-free true curve whose dip depth is at least half the maximum, so
  with resolved hyperfine lines a measurement on any line is near. The snapshots do not mark SBED's deliberate
  falsification probes, so the share is an upper bound for the EIG's own placement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nvision.metrics.stages import _EFFICIENCY_GRID_POINTS, _finite_or_none, _grad_sq_fn
from nvision.models.observer import RunResult

# "A few FWHM": farther than this many true FWHM from the dip is far.
FAR_FWHM = 3.0

CLAIM_COLUMN_KEYS: tuple[str, ...] = (
    "stage_fisher_sweep",
    "stage_pit_end",
    "stage_pit_refine_start",
    "stage2_far_share",
    "stage2_far_mass",
)


def _true_params(run_result: RunResult) -> list[float]:
    ts = run_result.true_signal
    return [float(v) for v in ts.model.spec.pack_params(ts.typed_parameters)]


def _pit(run_result: RunResult, primary: str, snapshot_index: int) -> float:
    """Posterior CDF of ``primary`` at its true value after snapshot ``snapshot_index`` (0-based)."""
    truth = float(run_result.true_signal.get_param_value(primary))
    cdf = run_result.snapshots[snapshot_index].belief.marginal_cdf(primary, np.array([truth], dtype=np.float64))
    return float(np.asarray(cdf, dtype=np.float64).reshape(-1)[0])


def _sweep_information(run_result: RunResult, primary: str, grid_phys: np.ndarray) -> float:
    """``mean_x [k g(x)^2 / sigma^2]`` (Hz^-2 per measurement) over the drive window (run's own shots, noise)."""
    snapshots = run_result.snapshots
    grad_sq = _grad_sq_fn(run_result.true_signal, primary, _true_params(run_result))
    g2_mean = float(np.mean(grad_sq(snapshots[0].obs.drive, grid_phys)))
    shots = np.array([float(s.obs.n_shots) for s in snapshots])
    noise_std = np.array([float(s.obs.noise_std) for s in snapshots])
    return g2_mean * float(np.mean(shots / noise_std**2))


def _far_columns(
    run_result: RunResult, primary: str, grid_phys: np.ndarray, fwhm_phys: float, first: int, count: int
) -> tuple[float, float]:
    """``(far share of measurements, mean far posterior mass)`` over snapshots ``first .. first + count - 1`` (0-based).

    ``grid_phys``: shape (n_grid,), the drive-frequency grid in Hz. ``first >= 1``: the belief before a measurement is
    the previous snapshot.
    """
    ts = run_result.true_signal
    snapshots = run_result.snapshots
    model = ts.model_at(snapshots[0].obs.drive)
    spec = ts.model.spec
    curve = np.asarray(model.compute_many_float64(grid_phys, spec.unpack_params(_true_params(run_result))), dtype=float)
    depth = curve.max() - curve
    core = grid_phys[depth >= 0.5 * depth.max()]  # shape (n_core,), Hz
    x_lo, x_hi = snapshots[0].belief.drive_freq_bounds_phys
    idx = range(first, first + count)
    xs = np.array([x_lo + snapshots[i].obs.drive_freq_unit * (x_hi - x_lo) for i in idx])  # shape (count,), Hz
    dist = np.abs(xs[:, None] - core[None, :]).min(axis=1)
    share = float(np.mean(dist > FAR_FWHM * fwhm_phys))
    truth = float(ts.get_param_value(primary))
    edges = np.array([truth - FAR_FWHM * fwhm_phys, truth + FAR_FWHM * fwhm_phys])
    mass = []
    for i in idx:
        cdf = np.asarray(snapshots[i - 1].belief.marginal_cdf(primary, edges), dtype=float)
        mass.append(1.0 - float(cdf[1] - cdf[0]))
    return share, float(np.mean(mass))


def compute_claim_columns(run_result: RunResult, stage: dict[str, Any]) -> dict[str, Any]:
    """The :data:`CLAIM_COLUMN_KEYS` of one run; ``stage`` is the output of ``compute_stage_metrics`` for it.

    Every value is ``None`` when it is undefined for the run (no primary parameter, a drive-controlled model with no
    single dip, a stage never reached): that is a property of the run, not a fallback.
    """
    out: dict[str, Any] = dict.fromkeys(CLAIM_COLUMN_KEYS)
    primary = stage.get("stage_primary_param")
    snapshots = run_result.snapshots
    if primary is None or not snapshots:
        return out
    out["stage_pit_end"] = _finite_or_none(_pit(run_result, primary, len(snapshots) - 1))
    refine = stage.get("refine_start_step")
    if refine is not None:
        out["stage_pit_refine_start"] = _finite_or_none(_pit(run_result, primary, int(refine) - 1))
    if run_result.true_signal.model.controls_drive:
        return out
    x_lo, x_hi = snapshots[0].belief.drive_freq_bounds_phys
    grid_phys = np.linspace(x_lo, x_hi, _EFFICIENCY_GRID_POINTS)
    out["stage_fisher_sweep"] = _finite_or_none(_sweep_information(run_result, primary, grid_phys))
    focus, n2, fwhm = stage.get("focus_found_step"), stage.get("stage2_steps"), stage.get("stage_fwhm_true_phys")
    if focus is not None and n2 and fwhm:
        share, mass = _far_columns(run_result, primary, grid_phys, float(fwhm), int(focus), int(n2))
        out["stage2_far_share"], out["stage2_far_mass"] = _finite_or_none(share), _finite_or_none(mass)
    return out
