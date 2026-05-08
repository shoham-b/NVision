
import numpy as np
import pytest
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel

def test_select_max_information_gain_diversity():
    """Test that the locator doesn't get stuck at a single point for a flat/sampled prior."""
    # Setup a standard NV center model
    model = NVCenterLorentzianModel()
    phys_bounds = {
        "frequency": (2.86e9, 2.88e9),
        "linewidth": (5e6, 15e6),
        "split": (1e6, 5e6),
        "k_np": (0.5, 1.5),
        "dip_depth": (0.05, 0.2),
        "background": (0.0, 0.1),
    }
    x_bounds = phys_bounds["frequency"]
    wrapped_model = UnitCubeSignalModel(model, phys_bounds, x_bounds)
    
    # Create a flat prior SMC belief
    param_bounds = {name: (0.0, 1.0) for name in phys_bounds}
    belief = UnitCubeSMCMarginalDistribution(
        model=wrapped_model,
        parameter_bounds=param_bounds,
        num_particles=1000,
        physical_param_bounds=phys_bounds,
        physical_x_bounds=x_bounds
    )
    
    # Generate candidates
    candidates = np.linspace(x_bounds[0], x_bounds[1], 1024)
    
    # Call select_max_information_gain multiple times
    results = []
    for _ in range(50):
        best = belief.select_max_information_gain(candidates, 1, noise_std=0.02)
        results.append(best[0])
    
    results = np.array(results)
    unique_counts = len(np.unique(results))
    
    # We expect at least some diversity (e.g. > 5 unique points) 
    # even if there's sampling noise, thanks to tie-breaking/jitter.
    # If it's 1, it's definitely stuck.
    assert unique_counts > 5, f"Locator is stuck! Only {unique_counts} unique points picked out of 50."

if __name__ == "__main__":
    test_select_max_information_gain_diversity()
