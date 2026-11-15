"""Localize phase: from the run's own dip detection to the point where it has locked onto the line.

The article splits a run into search (until the dip test fires, ``dip_detected_step`` = tau), localize (tau until the
estimate has locked onto the line) and refine (afterwards, judged by the information rate q of
``refine_rates.py``). The claim tested here is that the Bayesian strategy gains in localize as well: from the same
starting point, its own detection, it reaches the line in fewer measurements than a sweep.

* **Lock step**: the first measurement from which a per-step error stays ``<= eps`` until the end of the run, with
  ``eps = LOCK_TOLERANCE_HWHM`` times the line's HWHM. The functions take an error array and a tolerance, never the
  true signal: in simulation the caller passes ``|estimate - truth|`` (``lock_step_true``) or the claimed sigma
  (``lock_step_claimed``); in the lab, the error against the holdout centre or the claimed sigma.
* **Localize cost** ``L = lock - tau`` (measurements), per run; ``None`` when the run never detected or never locked.
* **Bound**: van Trees restarted at tau with the posterior as the prior, ``E[err^2] >= (J_tau + n I_max)^-1``, so
  reaching ``eps^2`` takes at least ``max(0, eps^-2 - J_tau) / I_max`` measurements after tau
  (:func:`localize_bound_steps`). It bounds the mean square, not one run's first passage, and J_tau is good to about a
  factor 2 (``posterior_J_tau_2h_phys`` brackets it), so it is a reference line, not a per-run test.

Every run measures its whole budget, so "stays until the end" means until the budget, the same for every locator.
"""

from __future__ import annotations

import math

import numpy as np

# eps of the lock step, in units of the line's HWHM: the estimate is then inside the line, where the signal is
# locally linear in theta and the steepest-slope point (gamma / sqrt 3 from the centre) is within reach.
LOCK_TOLERANCE_HWHM = 0.5


def lock_step(errors: np.ndarray, tolerance: float) -> int | None:
    """1-indexed first step from which ``errors <= tolerance`` holds to the end; ``None`` if the last step fails.

    ``errors``: shape (n_steps,), a per-step error (true or claimed), one value per measurement. NaN counts as a
    failure. Raises on a non-positive or non-finite tolerance.
    """
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be positive and finite, got {tolerance!r}")
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 1:
        raise ValueError(f"errors must be 1-D, got shape {errors.shape}")
    ok = errors <= tolerance  # NaN compares False
    if errors.size == 0 or not ok[-1]:
        return None
    failures = np.flatnonzero(~ok)
    return 1 if failures.size == 0 else int(failures[-1]) + 2


def localize_cost(lock: int | None, dip_step: int | None) -> int | None:
    """Measurements from the detection to the lock, ``lock - dip_step`` (0 if it locked before detecting)."""
    if lock is None or dip_step is None:
        return None
    return max(0, lock - dip_step)


def localize_bound_steps(tolerance: float, j_tau: float, i_max: float) -> float:
    """Fewest measurements after tau for the restarted van Trees bound to reach ``tolerance^2``.

    ``max(0, tolerance^-2 - j_tau) / i_max`` with ``j_tau`` the posterior's Fisher information at tau (Hz^-2) and
    ``i_max`` the information of one measurement at the steepest slope (Hz^-2 per measurement).
    """
    for name, value in (("tolerance", tolerance), ("j_tau", j_tau), ("i_max", i_max)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if tolerance <= 0 or i_max <= 0 or j_tau < 0:
        raise ValueError(f"need tolerance > 0, i_max > 0, j_tau >= 0; got {tolerance}, {i_max}, {j_tau}")
    return max(0.0, tolerance**-2 - j_tau) / i_max
