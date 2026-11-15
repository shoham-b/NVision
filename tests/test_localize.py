"""Localize phase metrics (nvision/metrics/localize.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nvision.metrics.localize import localize_bound_steps, localize_cost, lock_step


def test_lock_step_is_the_first_step_that_stays_within_tolerance():
    errs = np.array([5.0, 0.5, 2.0, 0.9, 0.4, 0.2])
    assert lock_step(errs, 1.0) == 4  # 0.5 at step 2 does not hold: step 3 fails again
    assert lock_step(np.array([0.1, 0.2]), 1.0) == 1
    assert lock_step(np.array([0.1, 2.0]), 1.0) is None  # the last step fails
    assert lock_step(np.array([]), 1.0) is None


def test_lock_step_treats_nan_as_a_failure_and_rejects_bad_input():
    assert lock_step(np.array([0.1, math.nan, 0.2]), 1.0) == 3
    with pytest.raises(ValueError, match="tolerance"):
        lock_step(np.array([0.1]), 0.0)
    with pytest.raises(ValueError, match="1-D"):
        lock_step(np.zeros((2, 2)), 1.0)


def test_localize_cost_counts_from_the_detection():
    assert localize_cost(40, 25) == 15
    assert localize_cost(10, 25) == 0  # locked before the dip test fired
    assert localize_cost(None, 25) is None
    assert localize_cost(40, None) is None


def test_localize_bound_is_the_restarted_van_trees_count():
    # eps = 1, J_tau = 0.25 -> needs 0.75 more information, at I_max = 0.05 per measurement: 15 measurements.
    assert localize_bound_steps(1.0, 0.25, 0.05) == pytest.approx(15.0)
    assert localize_bound_steps(1.0, 4.0, 0.05) == 0.0  # the posterior at detection is already that precise
    with pytest.raises(ValueError, match="finite"):
        localize_bound_steps(1.0, math.nan, 0.05)
    with pytest.raises(ValueError, match="i_max > 0"):
        localize_bound_steps(1.0, 0.1, 0.0)
