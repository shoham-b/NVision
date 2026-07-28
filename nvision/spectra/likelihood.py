"""Likelihood helpers for Bayesian belief updates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit


@njit(cache=True)
def _gaussian_likelihood_jit(obs_y: float, predicted: np.ndarray, sigma: float) -> np.ndarray:
    n = predicted.shape[0]
    out = np.empty(n, dtype=np.float64)
    sigma = max(float(sigma), 1e-9)
    inv_sigma = 1.0 / sigma
    for i in range(n):
        z = (obs_y - predicted[i]) * inv_sigma
        out[i] = math.exp(-0.5 * z * z)
    return out


def _gaussian_likelihood(obs_y: float, predicted: np.ndarray, sigma: float) -> np.ndarray:
    pred = np.asarray(predicted, dtype=np.float64)
    return _gaussian_likelihood_jit(float(obs_y), pred, float(sigma))


def likelihood_from_observation_model(
    *,
    obs_y: float,
    predicted: np.ndarray,
    noise_std: float,
    frequency_noise_model: tuple[dict[str, Any], ...] | None,
    tempering_factor: float = 10.0,
) -> np.ndarray:
    """Compute per-prediction likelihoods using observation noise metadata.

    Gaussian approximation with ``noise_std``.

    A tempering factor > 1.0 slows down Bayesian concentration by increasing effective noise.
    """
    return _gaussian_likelihood(obs_y, predicted, noise_std * np.sqrt(tempering_factor))
