from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.noises.over_frequency.gaussian_noise import OverFrequencyGaussianNoise


def test_estimated_noise_std_no_model_configured_uses_unknown_default():
    """No noise model at all: true noise level is unknown, so the coarse default applies."""
    noise = CompositeNoise(over_frequency_noise=None)
    assert noise.estimated_noise_std() == 0.05


def test_estimated_noise_std_configured_zero_uses_negligible_floor_not_unknown_default():
    """Gauss(0.0) means the true noise is known to be ~0, not unknown.

    Regression test: previously this silently fell back to the same 0.05
    used for "no noise model configured", inflating the Bayesian likelihood's
    sigma ~500x above the true noise and starving inference of information
    it should have had (see nvision/models/noise.py CompositeNoise docstring).
    """
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.0)]))
    std = noise.estimated_noise_std()
    assert std < 0.05
    assert std > 0.0


def test_estimated_noise_std_configured_nonzero_passes_through_unaffected():
    """A real, non-negligible configured noise std is unaffected by the floor change."""
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(0.02)]))
    assert noise.estimated_noise_std() == 0.02
