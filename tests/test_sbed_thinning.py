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

    # Run locator._acquire()
    locator.next()

    # Thinning enforces a minimum spacing of candidate_step_hz between consecutive
    # candidates, so the thinned count is bounded by the frequency span divided by
    # that step (+2 for the unconditionally-kept first/last candidates).
    assert len(passed_candidates) == 1
    thinned = passed_candidates[0]
    lo, hi = phys_bounds["frequency"]
    max_expected = int(np.ceil((hi - lo) / locator.candidate_step_hz)) + 2
    assert len(thinned) <= max_expected, f"Expected at most {max_expected} candidates, got {len(thinned)}"
    # Thinning must actually reduce the dense epoch grid, not just pass it through.
    raw_candidates = belief.get_candidates()
    assert len(thinned) < len(raw_candidates)
    # Consecutive thinned candidates (aside from the last, unconditionally-kept one)
    # must respect the minimum spacing.
    spacings = np.diff(np.sort(thinned))
    assert np.all(spacings[:-1] >= locator.candidate_step_hz - 1.0)
    print(f"Thinned candidates count: {len(thinned)} (raw: {len(raw_candidates)}, max expected: {max_expected})")


if __name__ == "__main__":
    test_sbed_candidate_thinning()
