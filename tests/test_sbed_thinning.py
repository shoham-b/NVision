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

    # SequentialBayesianExperimentDesignLocator no longer takes a hard
    # `n_candidates` cap — candidates are thinned to a minimum physical
    # spacing (`candidate_step_hz`) instead, so the count adapts to the
    # acquisition range and shrinks automatically as the posterior narrows.
    locator = SequentialBayesianExperimentDesignLocator(
        belief=belief,
        max_steps=10,
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

    raw_candidate_count = len(belief.get_candidates())

    # Run locator._acquire()
    locator.next()

    assert len(passed_candidates) == 1
    thinned = passed_candidates[0]

    # Thinning must actually reduce the raw (fine) epoch grid.
    assert len(thinned) < raw_candidate_count

    # Consecutive thinned candidates must respect the minimum physical spacing
    # (the very last gap may be smaller since the final candidate is always
    # kept so the full acquisition range stays represented).
    spacings = np.diff(np.sort(thinned))
    assert np.all(spacings[:-1] >= locator.candidate_step_hz - 1.0), "Thinned candidates must be spaced apart"

    # The count should match the acquisition range divided by the step size.
    lo, hi = phys_bounds["frequency"]
    expected_max = int(np.ceil((hi - lo) / locator.candidate_step_hz)) + 2
    assert len(thinned) <= expected_max
    print(f"Thinned candidates count: {len(thinned)} (raw grid: {raw_candidate_count})")


if __name__ == "__main__":
    test_sbed_candidate_thinning()
