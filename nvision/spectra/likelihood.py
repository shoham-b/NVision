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


@njit(cache=True)
def _poisson_likelihood_jit(k: int, predicted: np.ndarray, scale: float) -> np.ndarray:
    n = predicted.shape[0]
    log_p = np.empty(n, dtype=np.float64)
    max_log_p = -1e300
    lgamma_k = math.lgamma(k + 1.0)
    min_lam = 1e-12

    for i in range(n):
        lam = predicted[i] * scale
        if lam < min_lam:
            lam = min_lam
        lp = k * math.log(lam) - lam - lgamma_k
        log_p[i] = lp
        if lp > max_log_p:
            max_log_p = lp

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = math.exp(log_p[i] - max_log_p)
    return out


def _gaussian_likelihood(obs_y: float, predicted: np.ndarray, sigma: float) -> np.ndarray:
    pred = np.asarray(predicted, dtype=np.float64)
    return _gaussian_likelihood_jit(float(obs_y), pred, float(sigma))


def _poisson_likelihood_from_scaled_observation(obs_y: float, predicted: np.ndarray, scale: float) -> np.ndarray:
    """Poisson likelihood where observation is stored as k/scale.

    Noise pipeline for OverFrequencyPoissonNoise stores the measured value as
    ``obs_y = k / scale``. We recover integer ``k`` and evaluate ``P(k | lambda)``
    with ``lambda = predicted * scale``.
    """
    scale = max(float(scale), 1e-12)
    k = max(round(float(obs_y) * scale), 0)
    pred = np.asarray(predicted, dtype=np.float64)
    return _poisson_likelihood_jit(k, pred, scale)


def likelihood_from_observation_model(
    *,
    obs_y: float,
    predicted: np.ndarray,
    noise_std: float,
    frequency_noise_model: tuple[dict[str, Any], ...] | None,
    tempering_factor: float = 10.0,
) -> np.ndarray:
    """Compute per-prediction likelihoods using observation noise metadata.

    Supported exact path:
    - single-component Poisson over-frequency noise

    Fallback:
    - Gaussian approximation with ``noise_std``

    A tempering factor > 1.0 slows down Bayesian concentration by increasing effective noise.
    """
    if not frequency_noise_model:
        return _gaussian_likelihood(obs_y, predicted, noise_std * np.sqrt(tempering_factor))

    if len(frequency_noise_model) == 1 and frequency_noise_model[0].get("type") == "poisson":
        scale = float(frequency_noise_model[0].get("scale", 0.0))
        if scale > 0:
            return _poisson_likelihood_from_scaled_observation(obs_y, predicted, scale / tempering_factor)

    return _gaussian_likelihood(obs_y, predicted, noise_std * np.sqrt(tempering_factor))
