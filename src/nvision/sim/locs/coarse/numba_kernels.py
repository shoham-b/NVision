"""Shared Numba kernels for coarse locators."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def gaussian_peak_posterior_update(
    grid: np.ndarray,
    posterior: np.ndarray,
    obs_x: float,
    obs_signal: float,
    sigma: float = 0.1,
    eps: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """One-step Bayesian posterior update used by coarse peak locators."""
    n = grid.shape[0]
    updated = np.empty(n, dtype=np.float64)

    inv_sigma = 1.0 / sigma
    signal_scale = obs_signal + 1.0
    total = 0.0

    for i in range(n):
        d = (grid[i] - obs_x) * inv_sigma
        likelihood = math.exp(-0.5 * d * d) * signal_scale
        v = posterior[i] * (likelihood + eps)
        updated[i] = v
        total += v

    if total > eps:
        inv_total = 1.0 / total
        for i in range(n):
            updated[i] *= inv_total

    return updated, total
