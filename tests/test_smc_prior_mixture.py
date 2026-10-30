"""``prior_broad_fraction`` mixes a uniform component into a Gaussian prior's initial particles."""

import numpy as np

from nvision.belief.smc_marginal import PRIOR_BROAD_FRACTION
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise

_N = 2000


def _belief(fraction: float | None):
    base = nv_center_smc_belief(num_particles=_N, noise_model=gaussian_noise(), seed=3)
    lo, hi = base.physical_param_bounds["zeeman_shift"]
    std = 0.02 * (hi - lo)  # a tight prior whose mean sits 20 stds below the upper end
    bounds = {**base.physical_param_bounds, "_priors": {"zeeman_shift": (lo + 0.1 * (hi - lo), std)}}
    kwargs = {} if fraction is None else {"prior_broad_fraction": fraction}
    belief = nv_center_smc_belief(bounds, num_particles=_N, noise_model=gaussian_noise(), seed=3, **kwargs)
    return belief, belief._particles[:, belief._param_names.index("zeeman_shift")].copy()


def test_default_is_the_mixture():
    belief, col = _belief(None)
    assert PRIOR_BROAD_FRACTION == 0.1
    assert belief.prior_broad_fraction == PRIOR_BROAD_FRACTION
    assert col.max() > 0.4  # uniform draws reach the far end of the bound


def test_zero_fraction_is_the_pure_gaussian_prior():
    _, zero = _belief(0.0)
    assert zero.max() < 0.4  # nothing near the far end of the bound


def test_fraction_puts_about_that_share_of_particles_across_the_bound():
    belief, col = _belief(0.1)
    assert np.all((col >= 0.0) & (col <= 1.0))
    far = np.mean(col > 0.4)  # the Gaussian part never reaches this far
    assert 0.05 < far < 0.12  # ~ 0.1 * (1 - 0.4) = 0.06 uniform draws land there
    assert belief.prior_broad_fraction == 0.1


def test_copy_keeps_the_fraction():
    belief, _ = _belief(0.1)
    assert belief.copy().prior_broad_fraction == 0.1
