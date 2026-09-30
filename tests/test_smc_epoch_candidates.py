"""The epoch candidate grid targets the dips' steepest slopes at the *physical* estimated positions."""

import numpy as np

from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise


def _collapse_particles(belief, physical_values: dict[str, float]) -> None:
    """Put every particle exactly at ``physical_values`` (others stay at their current unit position)."""
    for name, value in physical_values.items():
        lo, hi = belief.physical_param_bounds[name]
        belief._particles[:, belief._param_names.index(name)] = (value - lo) / (hi - lo)
    belief._belief_version += 1


def test_slope_candidates_sit_at_the_estimated_physical_linewidth():
    """Linewidth bounds start at 200 kHz, so a unit-space delta would misplace the slope by that much."""
    belief = nv_center_smc_belief(num_particles=200, noise_model=gaussian_noise(), with_zeeman_splitting=False)
    centre = 2.87e9  # fixed frequency: the zero-field splitting, the lower edge of the half window
    linewidth = 5.0e6
    _collapse_particles(belief, {"linewidth": linewidth})
    belief._generate_epoch_candidates()

    # c - omega lies below the half window, so it is measured at its mirror image c + omega.
    slope = centre + linewidth
    assert np.min(np.abs(belief.get_candidates() - slope)) < 20e3


def test_plain_voigt_slope_includes_the_gaussian_width():
    belief = nv_center_smc_belief(
        num_particles=200, noise_model=gaussian_noise(), with_zeeman_splitting=False, lineshape="voigt"
    )
    centre = 2.87e9
    homogeneous, sigma_inhom = 3.0e6, 1.0e6
    _collapse_particles(belief, {"homogeneous_linewidth": homogeneous, "sigma_inhom": sigma_inhom})
    belief._generate_epoch_candidates()

    slope = centre + homogeneous + np.sqrt(2.0 * np.log(2.0)) * sigma_inhom
    assert np.min(np.abs(belief.get_candidates() - slope)) < 20e3
