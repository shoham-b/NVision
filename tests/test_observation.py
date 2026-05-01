from nvision.models.observation import (
    DEFAULT_MEASUREMENT_NOISE_STD,
    Observation,
    gaussian_likelihood_std,
)


def test_gaussian_likelihood_std_none_obs():
    """Test that None observation returns the default noise standard deviation."""
    assert gaussian_likelihood_std(None) == DEFAULT_MEASUREMENT_NOISE_STD


def test_gaussian_likelihood_std_positive_noise():
    """Test that observation with positive noise_std returns its own noise_std."""
    obs = Observation(x=0.0, signal_value=1.0, noise_std=0.1)
    assert gaussian_likelihood_std(obs) == 0.1


def test_gaussian_likelihood_std_zero_noise():
    """Test that observation with zero noise_std returns the default noise standard deviation."""
    obs = Observation(x=0.0, signal_value=1.0, noise_std=0.0)
    assert gaussian_likelihood_std(obs) == DEFAULT_MEASUREMENT_NOISE_STD


def test_gaussian_likelihood_std_negative_noise():
    """Test that observation with negative noise_std returns the default noise standard deviation."""
    obs = Observation(x=0.0, signal_value=1.0, noise_std=-0.1)
    assert gaussian_likelihood_std(obs) == DEFAULT_MEASUREMENT_NOISE_STD
