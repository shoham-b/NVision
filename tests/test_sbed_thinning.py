import numpy as np

from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


def test_sbed_candidate_thinning():
    # Setup standard NV center model
    model = NVCenterLorentzianModel()
    phys_bounds = {
        "frequency": (2.86e9, 2.88e9),
        "linewidth": (5e6, 15e6),
        "split": (1e6, 5e6),
        "k_np": (0.5, 1.5),
        "c_total": (0.05, 0.2),
        "background": (0.0, 0.1),
    }
    x_bounds = phys_bounds["frequency"]
    wrapped_model = UnitCubeSignalModel(model, phys_bounds, x_bounds)

    # Create flat prior SMC belief
    param_bounds = {name: (0.0, 1.0) for name in phys_bounds}
    belief = UnitCubeSMCMarginalDistribution(
        model=wrapped_model,
        parameter_bounds=param_bounds,
        num_particles=50,
        physical_param_bounds=phys_bounds,
        physical_x_bounds=x_bounds,
    )

    # Thinning is governed by a minimum physical spacing (candidate_step_hz),
    # not a hard candidate-count cap. Use a coarse step so the merged
    # slope-targeted + global grid (spanning the full 20 MHz range) thins
    # down well below 100 points.
    candidate_step_hz = 300_000.0
    locator = SequentialBayesianExperimentDesignLocator(
        belief=belief,
        max_steps=10,
        candidate_step_hz=candidate_step_hz,
    )

    # Mock belief.select_max_information_gain to inspect candidates passed to it
    original_select = belief.select_max_information_gain
    passed_candidates = []

    def mock_select(candidates, n, noise_std=0.02):
        passed_candidates.append(candidates)
        return original_select(candidates, n, noise_std=noise_std)

    belief.select_max_information_gain = mock_select

    # Seed so the acquisition takes the EIG path (first rand() = 0.3745 >= 0.2),
    # not the exploration/dip branches that skip the EIG grid search entirely.
    np.random.seed(42)

    # Run locator._acquire()
    locator.next()

    # The thinned candidates length should be at most 100, and consecutive
    # kept candidates must respect the minimum physical spacing (the final
    # gap is exempt since the last candidate is always kept to preserve the
    # full acquisition range, per _thin_by_step_indices).
    assert len(passed_candidates) == 1
    thinned = np.sort(passed_candidates[0])
    assert len(thinned) <= 100
    assert np.all(np.diff(thinned)[:-1] >= candidate_step_hz - 1e-6)
    print(f"Thinned candidates count: {len(thinned)}")


if __name__ == "__main__":
    test_sbed_candidate_thinning()
