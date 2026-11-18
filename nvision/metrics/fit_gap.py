"""Truth-free check that a particle-filter belief is not stuck: compare it with a least-squares fit of the same data.

A sequential Monte Carlo belief can lock onto a wrong line shape (e.g. the wrong split between the Lorentzian and
Gaussian widths) and stay there, because its Liu-West rejuvenation only moves particles by a fraction of the cloud's
own spread. Its claimed uncertainty keeps shrinking while the error stays put, so the claimed sigma cannot reveal it.

The check needs no truth, so it also applies to lab data. The final belief's own model is fitted to every observation
it absorbed, by bounded least squares on the unit cube from several starts, and the belief's point estimate (weighted
mean) is compared with the best fit:

    D = (SS_belief - SS_fit) / sigma_fit^2,      sigma_fit^2 = SS_fit / (n - k),

with SS the shot-weighted residual sum of squares (an observation averaging ``m`` shots has variance sigma^2 / m),
``n`` observations and ``k`` parameters. For a belief that has converged on the data, the posterior mean sits within
statistical noise of the best fit and D is of order k (a few). A stuck belief leaves a much better fit unclaimed and D
is large: tens to thousands in the 2026-11-17 study (``docs/evaluation_metrics.md``). ``STUCK_FIT_GAP`` is the flag.

Starts: the belief mean, the best particle, and a one-at-a-time ladder over every non-primary parameter (each set to
``FIT_START_LADDER_UNIT`` in turn, the rest at the belief mean). The ladder reaches the pure line-shape limits (a width
at 0) that random starts miss; it names no parameter, so it applies to any model. A fit is a local search, so D is a
lower bound on how far the belief is from the best fit: a stuck run can still score low.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

# Per-run scalar keys (results-row columns).
FIT_GAP_METRIC_KEYS: tuple[str, ...] = ("belief_fit_gap", "belief_fit_gap_shift_phys")
# D above this flags a stuck belief.
STUCK_FIT_GAP = 20.0
# Unit-cube values each non-primary parameter is started at, one parameter at a time.
FIT_START_LADDER_UNIT: tuple[float, ...] = (0.0, 0.05, 0.15, 0.4)
# Evaluation budget of one least-squares start.
_MAX_NFEV = 400
# Relative finite-difference step of the fit's Jacobian. The signal models compute in float32 (spectra/dtypes.py), so
# scipy's default step (~1.5e-8) changes the prediction by less than float32 resolution: the Jacobian reads zero and
# every start stops where it began.
_DIFF_STEP = 1e-3


def _has_fit_interface(belief: Any) -> bool:
    return all(
        callable(getattr(belief, name, None))
        for name in ("predict_at_observations", "observation_arrays", "observation_shots", "particles_unit")
    )


def fit_starts_unit(mean_unit: np.ndarray, best_particle_unit: np.ndarray, primary_index: int) -> np.ndarray:
    """Start vectors of the fit on the unit cube. shape: (n_starts, n_parameters)

    Args:
        mean_unit: The belief's weighted-mean estimate. shape: (n_parameters,)
        best_particle_unit: The particle with the smallest residual sum of squares. shape: (n_parameters,)
        primary_index: Column of the primary parameter, which the ladder leaves at the belief mean.
    """
    starts = [mean_unit, best_particle_unit]
    for j in range(mean_unit.size):
        if j == primary_index:
            continue
        for value in FIT_START_LADDER_UNIT:
            start = mean_unit.copy()
            start[j] = value
            starts.append(start)
    return np.clip(np.array(starts, dtype=np.float64), 0.0, 1.0)


def compute_fit_gap(belief: Any, primary: str | None) -> dict[str, float | None]:
    """``belief_fit_gap`` (D) and ``belief_fit_gap_shift_phys`` of a run's final belief.

    ``belief_fit_gap_shift_phys``: |primary of the best fit - primary of the belief mean|, in the parameter's physical
    unit (Hz): how far the estimate would move if the fit were trusted instead. Both are ``None`` when there is no
    ``primary``, when the belief is not a particle belief with stored observations (nothing to fit), or when there are
    no more observations than parameters (sigma_fit undefined).
    """
    out: dict[str, float | None] = dict.fromkeys(FIT_GAP_METRIC_KEYS)
    if primary is None or not _has_fit_interface(belief):
        return out
    names = belief.parameter_names
    if primary not in names:
        raise ValueError(f"Primary parameter {primary!r} is not a belief parameter {names}.")
    _, ys = belief.observation_arrays()
    ys = np.asarray(ys, dtype=np.float64)
    k = len(names)
    if ys.size <= k:
        return out
    sqrt_shots = np.sqrt(np.asarray(belief.observation_shots(), dtype=np.float64))  # shape: (n_observations,)
    if sqrt_shots.shape != ys.shape:
        raise ValueError(f"Shots {sqrt_shots.shape} and observations {ys.shape} are not aligned.")

    estimates = belief.estimates_unit()
    mean_unit = np.array([estimates[name] for name in names], dtype=np.float64)
    particles = belief.particles_unit()  # shape: (n_particles, n_parameters)
    weighted = (ys[:, None] - belief.predict_at_observations(particles)) * sqrt_shots[:, None]
    best_particle = particles[int(np.argmin(np.sum(weighted**2, axis=0)))]

    def residuals(u: np.ndarray) -> np.ndarray:
        return (ys - belief.predict_at_observations(u[None, :])[:, 0]) * sqrt_shots

    ss_mean = float(np.sum(residuals(mean_unit) ** 2))
    best_ss, best_u = ss_mean, mean_unit
    for start in fit_starts_unit(mean_unit, best_particle, names.index(primary)):
        fit = least_squares(
            residuals, start, bounds=(0.0, 1.0), x_scale="jac", max_nfev=_MAX_NFEV, diff_step=_DIFF_STEP
        )
        ss = float(np.sum(fit.fun**2))
        if ss < best_ss:
            best_ss, best_u = ss, fit.x
    sigma2_fit = best_ss / (ys.size - k)
    if sigma2_fit <= 0.0:
        raise ValueError("The fit reproduces every observation exactly; the noise level is undefined.")
    lo, hi = belief.physical_param_bounds[primary]
    j = names.index(primary)
    out["belief_fit_gap"] = (ss_mean - best_ss) / sigma2_fit
    out["belief_fit_gap_shift_phys"] = abs(float(best_u[j] - mean_unit[j])) * (hi - lo)
    return out
