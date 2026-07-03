import math

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

    # Locator thins the belief's epoch candidate grid down to a minimum physical
    # spacing of `candidate_step_hz` (default NVISION_SMC_CANDIDATE_STEP_HZ) instead
    # of a fixed candidate count.
    locator = SequentialBayesianExperimentDesignLocator(
        belief=belief,
        max_steps=10,
    )

    # Capture the untouched epoch grid for comparison before it's mutated by acquisition.
    raw_candidates = belief.get_candidates()

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

    assert len(passed_candidates) == 1
    thinned = passed_candidates[0]

    # Thinning must have collapsed the dense ~12.6k-point epoch grid down to
    # something close to domain_width / candidate_step_hz (the full 20 MHz
    # acquisition range hasn't narrowed yet, so this is the pre-convergence
    # maximum candidate count).
    domain_width = phys_bounds["frequency"][1] - phys_bounds["frequency"][0]
    max_expected_candidates = math.ceil(domain_width / locator.candidate_step_hz) + 2
    assert len(thinned) < len(raw_candidates)
    assert len(thinned) <= max_expected_candidates
    # Consecutive kept candidates must respect the minimum physical spacing, except
    # the final gap: the last raw candidate is always force-kept (regardless of
    # spacing) so the acquisition range's upper edge stays represented.
    assert np.all(np.diff(thinned)[:-1] >= locator.candidate_step_hz - 1.0)
    print(f"Thinned candidates count: {len(thinned)} (raw: {len(raw_candidates)})")


if __name__ == "__main__":
    test_sbed_candidate_thinning()
