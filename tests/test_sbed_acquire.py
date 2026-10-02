"""SBED acquisition is exactly: a decaying uniform probe, otherwise the EIG maximiser. No other path."""

from __future__ import annotations

import numpy as np
import pytest

from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
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


def test_acquire_explores_uniformly_over_the_full_probe_axis_when_the_draw_is_low():
    loc = _locator()
    loc.inference_step_count = 0
    loc.belief._rng = _FixedRng(0.0)
    loc._eig_acquire = lambda: pytest.fail("EIG must be skipped on an explore step")
    lo, hi = loc.belief.physical_x_bounds
    assert loc._acquire() == 0.5 * (lo + hi)


def test_acquire_is_eig_otherwise_even_where_the_old_dip_branch_used_to_fire():
    loc = _locator()
    loc.inference_step_count = 100  # exploration probability ~ 0.002
    sentinel = 123.0
    loc._eig_acquire = lambda: sentinel
    # 0.15 sat inside the removed dip-biased band [0.1*decay, 0.2): it must now be a plain EIG step.
    loc.belief._rng = _FixedRng(0.15)
    assert loc._acquire() == sentinel


def test_exploration_probability_decays_with_the_step_count():
    loc = _locator()
    loc._eig_acquire = lambda: -1.0
    loc.belief._rng = _FixedRng(0.05)
    loc.inference_step_count = 0
    assert loc._acquire() != -1.0  # 0.05 < 0.1 * exp(0)
    loc.inference_step_count = 200
    assert loc._acquire() == -1.0  # 0.05 > 0.1 * exp(-8)


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
