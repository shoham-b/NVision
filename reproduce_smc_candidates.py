"""Reproduction script for checking SMC candidate values in UnitCube.

Verifies if UnitCubeSMCMarginalDistribution incorrectly re-normalizes unit-space
candidates, leading to collapsed grids.
"""

import numpy as np
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.spectra.nv_center import NVCenterLorentzianModel

def test_candidate_normalization():
    # 1. Setup
    inner_model = NVCenterLorentzianModel()
    phys_bounds = {
        "frequency": (2.86e9, 2.88e9),
        "linewidth": (1e3, 10e6),
        "split": (0.1e6, 10e6),
        "k_np": (2.0, 4.0),
        "dip_depth": (0.01, 1.0)
    }
    x_bounds_phys = (2.86e9, 2.88e9)
    model = UnitCubeSignalModel(inner_model, phys_bounds, x_bounds_phys)

    param_names = model.parameter_names()
    parameter_bounds = {name: (0.0, 1.0) for name in param_names}

    belief = UnitCubeSMCMarginalDistribution(
        model=model,
        num_particles=100,
        parameter_bounds=parameter_bounds,
        physical_param_bounds=phys_bounds,
        physical_x_bounds=x_bounds_phys
    )

    # 2. Inspect candidates
    # __post_init__ already called _generate_epoch_candidates
    candidates = belief.get_candidates()
    print(f"Number of candidates: {len(candidates)}")
    if len(candidates) > 0:
        unique = np.unique(candidates)
        print(f"Unique candidate values (subset): {unique[:10]}")
        print(f"Mean candidate value: {np.mean(candidates):.6f}")

        if np.all(candidates == 0.0):
            print("FAIL: All candidates collapsed to 0.0!")
        elif np.mean(candidates) < 0.1:
             print("FAIL: Candidates are heavily biased towards 0 (likely incorrect normalization).")
        else:
            print("SUCCESS: Candidates are distributed across [0, 1].")

if __name__ == "__main__":
    test_candidate_normalization()
