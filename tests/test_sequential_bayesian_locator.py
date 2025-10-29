"""
Tests for Sequential Bayesian Experiment Design Locator

This module tests the SequentialBayesianLocator implementation to ensure
it correctly implements the Sequential Bayesian Experiment Design methodology
for ODMR measurements of NV centers.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.nvision.sim.locs.sequential_bayesian_locator import SequentialBayesianLocator
from src.nvision.sim.locs.obs import Obs


class TestSequentialBayesianLocator:
    """Test suite for Sequential Bayesian Experiment Design locator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.locator = SequentialBayesianLocator(
            max_evals=20,
            prior_bounds=(2.7e9, 3.0e9),
            convergence_threshold=1e6,
            grid_resolution=100  # Smaller for faster tests
        )
        self.domain = (2.7e9, 3.0e9)
        
    def test_initialization(self):
        """Test proper initialization of the locator."""
        assert self.locator.max_evals == 20
        assert self.locator.prior_bounds == (2.7e9, 3.0e9)
        assert self.locator.noise_model == "gaussian"
        assert self.locator.acquisition_function == "expected_information_gain"
        
        # Check that posterior is initialized
        assert hasattr(self.locator, 'freq_grid')
        assert hasattr(self.locator, 'freq_posterior')
        assert len(self.locator.freq_grid) == 100
        assert np.allclose(np.sum(self.locator.freq_posterior), 1.0)  # Normalized
        
    def test_odmr_model(self):
        """Test ODMR lineshape model."""
        params = {
            'frequency': 2.87e9,
            'linewidth': 10e6,
            'amplitude': 0.1,
            'background': 1.0
        }
        
        # Test on-resonance (should show dip)
        on_resonance = self.locator.odmr_model(2.87e9, params)
        assert on_resonance < params['background']
        
        # Test off-resonance (should be closer to background)
        off_resonance = self.locator.odmr_model(2.9e9, params)
        assert off_resonance > on_resonance
        assert off_resonance < params['background']  # Still slightly below due to Lorentzian tails
        
        # Test far off-resonance (should approach background)
        far_off = self.locator.odmr_model(3.2e9, params)
        assert abs(far_off - params['background']) < 0.01
        
    def test_likelihood_gaussian(self):
        """Test Gaussian likelihood calculation."""
        measurement = {
            'frequency': 2.87e9,
            'intensity': 0.95,
            'uncertainty': 0.05
        }
        
        params = {
            'frequency': 2.87e9,
            'linewidth': 10e6,
            'amplitude': 0.1,
            'background': 1.0
        }
        
        likelihood = self.locator.likelihood(measurement, params)
        assert isinstance(likelihood, float)
        assert not np.isnan(likelihood)
        assert not np.isinf(likelihood)
        
    def test_likelihood_poisson(self):
        """Test Poisson likelihood calculation."""
        self.locator.noise_model = "poisson"
        
        measurement = {
            'frequency': 2.87e9,
            'intensity': 100,  # Count data
            'uncertainty': 10
        }
        
        params = {
            'frequency': 2.87e9,
            'linewidth': 10e6,
            'amplitude': 10,
            'background': 110
        }
        
        likelihood = self.locator.likelihood(measurement, params)
        assert isinstance(likelihood, float)
        assert not np.isnan(likelihood)
        assert not np.isinf(likelihood)
        
    def test_posterior_update(self):
        """Test Bayesian posterior updating."""
        # Initial posterior should be uniform
        initial_posterior = self.locator.freq_posterior.copy()
        assert np.allclose(initial_posterior, 1.0/len(initial_posterior))
        
        # Add a measurement
        measurement = {
            'frequency': 2.85e9,
            'intensity': 0.95,
            'uncertainty': 0.05
        }
        
        self.locator.update_posterior(measurement)
        
        # Posterior should have changed
        updated_posterior = self.locator.freq_posterior
        assert not np.allclose(initial_posterior, updated_posterior)
        
        # Should still be normalized
        assert np.allclose(np.sum(updated_posterior), 1.0)
        
        # Uncertainty should have decreased
        assert self.locator.current_estimates['uncertainty'] < np.inf
        
    def test_information_gain_calculation(self):
        """Test expected information gain calculation."""
        # Add some measurements to establish posterior
        measurements = [
            {'frequency': 2.85e9, 'intensity': 0.98, 'uncertainty': 0.05},
            {'frequency': 2.87e9, 'intensity': 0.92, 'uncertainty': 0.05},
            {'frequency': 2.89e9, 'intensity': 0.97, 'uncertainty': 0.05}
        ]
        
        for meas in measurements:
            self.locator.update_posterior(meas)
        
        # Test information gain calculation
        info_gain = self.locator.expected_information_gain(2.86e9)
        
        assert isinstance(info_gain, float)
        assert info_gain >= 0  # Information gain should be non-negative
        assert not np.isnan(info_gain)
        
    def test_mutual_information_criterion(self):
        """Test mutual information utility function."""
        # Add some measurements
        measurement = {
            'frequency': 2.87e9,
            'intensity': 0.95,
            'uncertainty': 0.05
        }
        self.locator.update_posterior(measurement)
        
        mi = self.locator.mutual_information_criterion(2.86e9)
        
        assert isinstance(mi, float)
        assert mi >= 0
        assert not np.isnan(mi)
        
    def test_propose_next_early_measurements(self):
        """Test initial measurement proposals."""
        history = []
        
        # First few measurements should explore domain
        next_freq = self.locator.propose_next(history, self.domain)
        
        assert self.domain[0] <= next_freq <= self.domain[1]
        
    def test_propose_next_with_history(self):
        """Test measurement proposal with existing history."""
        # Create some measurement history
        history = [
            Obs(x=2.85e9, intensity=0.98, uncertainty=0.05),
            Obs(x=2.87e9, intensity=0.92, uncertainty=0.05),
            Obs(x=2.89e9, intensity=0.97, uncertainty=0.05)
        ]
        
        next_freq = self.locator.propose_next(history, self.domain)
        
        assert self.domain[0] <= next_freq <= self.domain[1]
        assert isinstance(next_freq, float)
        
    def test_should_stop_max_evals(self):
        """Test stopping criterion based on maximum evaluations."""
        # Create history with max evaluations
        history = [Obs(x=2.87e9, intensity=0.95, uncertainty=0.05) for _ in range(20)]
        
        assert self.locator.should_stop(history)
        
    def test_should_stop_convergence(self):
        """Test stopping criterion based on convergence."""
        # Set very low uncertainty to trigger convergence
        self.locator.current_estimates['uncertainty'] = 1e5  # Below threshold
        
        history = [Obs(x=2.87e9, intensity=0.95, uncertainty=0.05)]
        
        assert self.locator.should_stop(history)
        
    def test_should_stop_low_utility(self):
        """Test stopping criterion based on low utility."""
        # Add low utility history
        self.locator.utility_history = [0.005, 0.005, 0.005]  # Below threshold
        
        history = [Obs(x=2.87e9, intensity=0.95, uncertainty=0.05)]
        
        assert self.locator.should_stop(history)
        
    def test_finalize_single_peak(self):
        """Test finalization with single peak detection."""
        # Create measurements around a single peak
        history = [
            Obs(x=2.85e9, intensity=0.98, uncertainty=0.05),
            Obs(x=2.87e9, intensity=0.90, uncertainty=0.05),  # Peak (dip)
            Obs(x=2.89e9, intensity=0.97, uncertainty=0.05)
        ]
        
        result = self.locator.finalize(history)
        
        assert 'n_peaks' in result
        assert 'x1' in result
        assert 'uncert' in result
        
        assert result['n_peaks'] == 1.0
        assert self.domain[0] <= result['x1'] <= self.domain[1]
        assert result['uncert'] > 0
        
    def test_finalize_two_peaks(self):
        """Test finalization with two peak detection."""
        # This is a simplified test - in practice would need measurements
        # that clearly show two distinct peaks in the posterior
        history = [
            Obs(x=2.75e9, intensity=0.90, uncertainty=0.05),  # Peak 1
            Obs(x=2.80e9, intensity=0.98, uncertainty=0.05),
            Obs(x=2.85e9, intensity=0.90, uncertainty=0.05),  # Peak 2
            Obs(x=2.90e9, intensity=0.98, uncertainty=0.05)
        ]
        
        result = self.locator.finalize(history)
        
        assert 'n_peaks' in result
        assert result['n_peaks'] >= 1.0  # May detect 1 or 2 peaks depending on posterior
        
    def test_different_acquisition_functions(self):
        """Test different acquisition functions."""
        # Test mutual information acquisition
        self.locator.acquisition_function = "mutual_information"
        
        history = [Obs(x=2.87e9, intensity=0.95, uncertainty=0.05)]
        next_freq = self.locator.propose_next(history, self.domain)
        
        assert self.domain[0] <= next_freq <= self.domain[1]
        
    def test_reset_posterior(self):
        """Test posterior reset functionality."""
        # Modify posterior
        measurement = {
            'frequency': 2.87e9,
            'intensity': 0.95,
            'uncertainty': 0.05
        }
        self.locator.update_posterior(measurement)
        
        # Store modified state
        modified_posterior = self.locator.freq_posterior.copy()
        
        # Reset
        self.locator.reset_posterior()
        
        # Should be back to uniform
        assert not np.allclose(modified_posterior, self.locator.freq_posterior)
        assert np.allclose(self.locator.freq_posterior, 1.0/len(self.locator.freq_posterior))
        
    def test_error_handling(self):
        """Test error handling in utility calculations."""
        # Test with invalid noise model
        with pytest.raises(ValueError):
            self.locator.noise_model = "invalid"
            measurement = {'frequency': 2.87e9, 'intensity': 0.95, 'uncertainty': 0.05}
            params = {'frequency': 2.87e9, 'linewidth': 10e6, 'amplitude': 0.1, 'background': 1.0}
            self.locator.likelihood(measurement, params)
            
    def test_protocol_compliance(self):
        """Test that the locator follows the LocatorStrategy protocol."""
        # Test that all required methods exist and work
        history = [Obs(x=2.87e9, intensity=0.95, uncertainty=0.05)]
        
        # Test propose_next
        next_freq = self.locator.propose_next(history, self.domain)
        assert isinstance(next_freq, float)
        
        # Test should_stop
        should_stop = self.locator.should_stop(history)
        assert isinstance(should_stop, bool)
        
        # Test finalize
        result = self.locator.finalize(history)
        assert isinstance(result, dict)
        assert all(key in result for key in ['n_peaks', 'x1', 'uncert'])
        

if __name__ == "__main__":
    pytest.main([__file__])