"""Shared noise model for tests that build a Bayesian belief (which always infers noise)."""

from __future__ import annotations

from nvision.spectra.noise_model import GaussianNoiseSignalModel


def gaussian_noise(lo: float = 0.01, hi: float = 0.1) -> GaussianNoiseSignalModel:
    """Gaussian noise whose sigma prior spans ``[lo, hi]``."""
    return GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (lo, hi)})
