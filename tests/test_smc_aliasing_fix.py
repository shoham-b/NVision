import numpy as np
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.spectra.nv_center import NVCenterLorentzianModel

def test_smc_stable_update_prevents_uniform_reset():
    """Test that a highly unlikely observation doesn't cause raw weights to underflow to 0 
    and reset to uniform, but instead properly re-weights using log-likelihoods."""
    model = NVCenterLorentzianModel()
    bounds = {
        "frequency": (2.7e9, 2.8e9),
        "linewidth": (1e6, 3e6),
        "split": (4e6, 6e6),
        "k_np": (1.0, 5.0),
        "dip_depth": (0.1, 0.9),
    }
    
    # Initialize SMC
    smc = SMCMarginalDistribution(
        model=model,
        parameter_bounds=bounds,
        num_particles=100,
        auto_resample=False, # Disable auto-resample to inspect weights
    )
    
    x_obs = 2.75e9
    
    # We observe an intensity of -100.0 (forces massive underflow in raw probabilities)
    # The true predictions will be ~0.5 to 1.0. Residual will be > 100.
    # With noise_std = 0.01, residual / sigma = 10000.
    # log_lik = -0.5 * 10000^2 = -50,000,000.
    # In raw probabilities, exp(-50,000,000) is exactly 0.0.
    target_y = -100.0
    
    obs = Observation(x=x_obs, signal_value=target_y, noise_std=0.01)
    smc.update(obs)
    
    # If the old bug was present, ALL raw likelihoods would be 0.0, weight_sum=0,
    # and weights would be reset to uniform (1/N = 0.01).
    # With the fix, the best particle should get a weight close to 1.0!
    max_weight = np.max(smc._weights)
    assert max_weight > 0.9, (
        f"Max weight of particles should be near 1.0, not reset to uniform. "
        f"Got max weight: {max_weight}. "
        f"This indicates float32 underflow in likelihood computation!"
    )
    
    # Verify weights are normalized
    assert np.isclose(np.sum(smc._weights), 1.0)
