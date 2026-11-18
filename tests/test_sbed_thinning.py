import numpy as np

from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel
from tests.noise import gaussian_noise


def test_sbed_candidates_are_the_uniform_grid_inside_the_focus():
    # Setup standard NV center model
    model = NVCenterLorentzianModel()
    phys_bounds = {
        "zero_field_freq": (2.86e9, 2.88e9),
        "hwhm": (5e6, 15e6),
        "hyperfine_shift": (1e6, 5e6),
        "k_np": (0.5, 1.5),
        "contrast_total": (0.05, 0.2),
        "background": (0.0, 0.1),
    }
    x_bounds = phys_bounds["zero_field_freq"]
    wrapped_model = UnitCubeSignalModel(model, phys_bounds, x_bounds)

    # Create flat prior SMC belief
    param_bounds = {name: (0.0, 1.0) for name in phys_bounds}
    belief = SMCMarginalDistribution(
        model=wrapped_model,
        parameter_bounds=param_bounds,
        num_particles=50,
        physical_param_bounds=phys_bounds,
        drive_freq_bounds_phys=x_bounds,
        noise_model=gaussian_noise(),
    )

    # Initialize locator (the belief's uniform grid is used as is, no extra thinning)
    locator = SequentialBayesianExperimentDesignLocator(
        belief=belief,
        max_steps=10,
    )

    # Mock belief.select_max_information_gain to inspect candidates passed to it
    original_select = belief.select_max_information_gain
    passed_candidates = []

    def mock_select(candidates, n):
        passed_candidates.append(candidates)
        return original_select(candidates, n)

    belief.select_max_information_gain = mock_select

    # Seed the belief's stream so the acquisition takes the EIG path (first random() = 0.637 >= 0.2),
    # not the exploration/dip branches that skip the EIG grid search entirely.
    belief._rng = np.random.default_rng(0)

    # Run locator._acquire()
    locator.next()

    # The EIG candidates are exactly the belief's uniform grid restricted to the focus.
    assert len(passed_candidates) == 1
    grid = belief.get_candidate_drive_freq_phys()
    inside = grid[locator._focus.contains(grid)]
    np.testing.assert_array_equal(passed_candidates[0], inside)
    steps = np.diff(passed_candidates[0])
    assert np.ptp(steps) < 1e-3 * steps.mean() + 1e3


if __name__ == "__main__":
    test_sbed_candidates_are_the_uniform_grid_inside_the_focus()
