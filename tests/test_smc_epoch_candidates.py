"""The epoch candidates are one uniform grid over the drive window, step = fraction x posterior-mean HWHM."""

import numpy as np
import pytest

from nvision.belief.smc_marginal import NVISION_SBED_CANDIDATE_STEP_HWHM_FRACTION, uniform_candidate_grid_phys
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise


def _collapse_particles(belief, physical_values: dict[str, float]) -> None:
    """Put every particle exactly at ``physical_values`` (others stay at their current unit position)."""
    for name, value in physical_values.items():
        lo, hi = belief.physical_param_bounds[name]
        belief._particles[:, belief._param_names.index(name)] = (value - lo) / (hi - lo)
    belief._belief_version += 1


def test_grid_is_uniform_with_step_a_fraction_of_the_hwhm():
    belief = nv_center_smc_belief(num_particles=200, noise_model=gaussian_noise(), with_zeeman_splitting=False)
    hwhm = 5.0e6
    _collapse_particles(belief, {"hwhm": hwhm})
    belief._generate_epoch_candidate_drive_freq()

    grid = belief.get_candidate_drive_freq_phys()
    lo, hi = belief.drive_freq_bounds_phys
    steps = np.diff(grid)
    assert grid[0] == pytest.approx(lo, abs=1.0)
    assert grid[-1] == pytest.approx(hi, abs=1.0)
    assert steps.max() <= NVISION_SBED_CANDIDATE_STEP_HWHM_FRACTION * hwhm * (1 + 1e-4)
    # float32 unit storage: uniform to within the Hz-level rounding of the unit coordinate
    assert np.ptp(steps) < 1e-3 * steps.mean() + 1e3


def test_grid_step_follows_the_hwhm():
    belief = nv_center_smc_belief(num_particles=200, noise_model=gaussian_noise(), with_zeeman_splitting=False)
    _collapse_particles(belief, {"hwhm": 4.0e6})
    belief._generate_epoch_candidate_drive_freq()
    n_wide = belief.get_candidate_drive_freq_phys().size
    _collapse_particles(belief, {"hwhm": 8.0e6})
    belief._generate_epoch_candidate_drive_freq()
    assert belief.get_candidate_drive_freq_phys().size < n_wide


def test_uniform_grid_helper_fails_fast():
    with pytest.raises(ValueError, match="uniform_candidate_grid_phys"):
        uniform_candidate_grid_phys(1.0, 1.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="uniform_candidate_grid_phys"):
        uniform_candidate_grid_phys(0.0, 10.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="uniform_candidate_grid_phys"):
        uniform_candidate_grid_phys(0.0, 10.0, -1.0, 0.1)
    with pytest.raises(ValueError, match="uniform_candidate_grid_phys"):
        uniform_candidate_grid_phys(0.0, 10.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="uniform_candidate_grid_phys"):
        uniform_candidate_grid_phys(0.0, 1.0e9, 1.0, 0.1)  # > 200 000 candidates


def test_uniform_grid_helper_includes_both_ends():
    grid = uniform_candidate_grid_phys(0.0, 1.0, 1.0, 0.3)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert np.diff(grid).max() <= 0.3 + 1e-12
