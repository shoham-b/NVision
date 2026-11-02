"""SBED acquisition is exactly: a gap-filling falsification probe, otherwise the EIG maximiser. No other path."""

from __future__ import annotations

import numpy as np
import pytest

from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import (
    EXPLORE_PROBABILITY_FLOOR,
    EXPLORE_PROBABILITY_MAX,
    SequentialBayesianExperimentDesignLocator,
    exploration_probability,
    largest_gap_midpoint_unit,
)
from tests.noise import gaussian_noise


def _locator(seed: int = 0) -> SequentialBayesianExperimentDesignLocator:
    belief = nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=200, seed=seed)
    return SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=50)


class _FixedRng:
    """Stand-in for the belief's RNG: ``random()`` returns a fixed value, ``uniform`` the midpoint."""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value

    def uniform(self, lo: float, hi: float) -> float:
        return 0.5 * (lo + hi)


def test_acquire_probes_the_middle_of_the_widest_unmeasured_stretch_when_the_draw_is_low():
    loc = _locator()
    loc.belief._rng = _FixedRng(0.0)
    loc._eig_acquire = lambda: pytest.fail("EIG must be skipped on a probe step")
    lo, hi = loc.belief.drive_freq_bounds_phys
    assert loc._acquire() == 0.5 * (lo + hi)  # nothing measured: the whole axis is one gap


def test_acquire_is_eig_when_the_draw_is_above_the_probe_probability():
    loc = _locator()
    sentinel = 123.0
    loc._eig_acquire = lambda: sentinel
    loc.belief._rng = _FixedRng(0.99)
    assert loc._acquire() == sentinel


def test_largest_gap_midpoint_unit_cuts_the_axis_at_the_measured_positions():
    mid, gap = largest_gap_midpoint_unit(np.array([0.5, 0.1, 0.2]))
    assert (mid, gap) == pytest.approx((0.75, 0.5))
    mid, gap = largest_gap_midpoint_unit(np.array([0.1, 0.9]))
    assert (mid, gap) == pytest.approx((0.5, 0.8))
    assert largest_gap_midpoint_unit(np.array([]))[0] == 0.5
    assert largest_gap_midpoint_unit(np.array([0.5, 0.5]))[1] == pytest.approx(0.5)  # unmeasured edge is a gap


def test_largest_gap_midpoint_unit_rejects_positions_off_the_axis():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        largest_gap_midpoint_unit(np.array([0.5, 1.2]))


def test_exploration_probability_falls_with_coverage_to_a_floor_and_not_below():
    assert exploration_probability(1.0) == pytest.approx(EXPLORE_PROBABILITY_MAX)
    assert exploration_probability(0.0) == pytest.approx(EXPLORE_PROBABILITY_FLOOR)
    assert exploration_probability(0.5) == pytest.approx(0.5 * (EXPLORE_PROBABILITY_MAX + EXPLORE_PROBABILITY_FLOOR))
    with pytest.raises(ValueError, match="gap must lie"):
        exploration_probability(1.5)


def test_probes_fill_the_axis_instead_of_clustering():
    """Consecutive probes always take the widest remaining gap, so the largest gap shrinks every time."""
    xs = np.array([0.5])
    gaps = []
    for _ in range(6):
        mid, gap = largest_gap_midpoint_unit(xs)
        gaps.append(gap)
        xs = np.append(xs, mid)
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < gaps[0]


def test_select_max_information_gain_fails_fast_on_no_candidates():
    belief = _locator().belief
    with pytest.raises(ValueError, match="no candidates"):
        belief.select_max_information_gain(np.array([]), 1)


def test_resample_reflects_particles_instead_of_clipping_them_onto_the_bound():
    belief = nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=2000, seed=0)
    belief._particles[:, :] = 0.0  # every particle sits exactly on the lower bound
    belief._resample()
    assert np.all((belief._particles >= 0.0) & (belief._particles <= 1.0))
    # A clip would leave ~half the particles exactly on 0.0; a reflection leaves (almost) none there.
    assert np.mean(belief._particles == 0.0) < 0.01
