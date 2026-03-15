"""Lineshape parameter optimisation via scipy L-BFGS-B for Bayesian locators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from nvision.sim.locs.nv_center._jit_kernels import (
    _calculate_total_log_likelihood_jit,
)
from nvision.sim.locs.nv_center._odmr_model import odmr_model


def get_optim_config(
    distribution: str,
    linewidth_prior: tuple[float, float],
    gaussian_width_prior: tuple[float, float],
    split_prior: tuple[float, float],
    k_np_prior: tuple[float, float],
    amplitude_prior: tuple[float, float],
    background_prior: tuple[float, float],
) -> tuple[list[str], list[tuple[float, float]]]:
    """Return ``(param_keys, bounds)`` for the given *distribution*.

    Args:
        distribution: ODMR lineshape name.
        linewidth_prior: ``(low, high)`` for Lorentzian half-width.
        gaussian_width_prior: ``(low, high)`` for Gaussian sigma.
        split_prior: ``(low, high)`` for Zeeman split.
        k_np_prior: ``(low, high)`` for k_np parameter.
        amplitude_prior: ``(low, high)`` for amplitude.
        background_prior: ``(low, high)`` for background.

    Returns:
        A 2-tuple of ``(param_keys, bounds)`` lists.
    """
    if distribution == "lorentzian":
        param_keys = ["linewidth", "amplitude", "background"]
        bounds = [linewidth_prior, amplitude_prior, background_prior]
    elif distribution == "voigt":
        param_keys = ["linewidth", "gaussian_width", "amplitude", "background"]
        bounds = [linewidth_prior, gaussian_width_prior, amplitude_prior, background_prior]
    elif distribution == "voigt-zeeman":
        param_keys = ["linewidth", "split", "k_np", "amplitude", "background"]
        bounds = [linewidth_prior, split_prior, k_np_prior, amplitude_prior, background_prior]
    else:
        param_keys = []
        bounds = []
    return param_keys, bounds


def _objective_func(
    param_values: np.ndarray,
    param_keys: list[str],
    current_estimates: dict[str, Any],
    measurement_history: list[dict[str, float]],
    noise_model: str,
    distribution: str,
) -> float:
    """Negative total log-likelihood used as the minimisation objective.

    Args:
        param_values: Current parameter vector from the optimiser.
        param_keys: Names corresponding to each element of *param_values*.
        current_estimates: Full estimates dict (frequency is pinned).
        measurement_history: List of coerced measurement dicts.
        noise_model: ``"gaussian"`` or ``"poisson"``.
        distribution: ODMR lineshape name.

    Returns:
        Negative total log-likelihood (lower is better).
    """
    params = current_estimates.copy()
    for key, value in zip(param_keys, param_values, strict=False):
        params[key] = value

    params["frequency"] = current_estimates["frequency"]

    if not measurement_history:
        return 0.0

    measurements_x = np.array([m["x"] for m in measurement_history])
    measurements_y = np.array([m["signal_values"] for m in measurement_history])

    params_array = np.array(
        [
            params["frequency"],
            params["linewidth"],
            params["amplitude"],
            params["background"],
            params.get("gaussian_width", 0.0),
            params.get("split", 0.0),
            params.get("k_np", 0.0),
        ],
        dtype=np.float64,
    )

    noise_model_code = 0 if noise_model == "gaussian" else 1
    dist_code_map = {"lorentzian": 0, "voigt": 1, "voigt-zeeman": 2}
    dist_code = dist_code_map.get(distribution, -1)

    if dist_code != -1:
        total_log_likelihood = _calculate_total_log_likelihood_jit(
            measurements_x, measurements_y, params_array, noise_model_code, dist_code
        )
        if np.isneginf(total_log_likelihood):
            return np.inf
        return -total_log_likelihood

    # Fallback for other distributions
    predicted = odmr_model(measurements_x, params, distribution)
    observed = measurements_y
    sigma = 0.05

    if noise_model == "gaussian":
        log_lik_array = -0.5 * ((observed - predicted) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma**2)
    elif noise_model == "poisson":
        predicted[predicted <= 0] = 1e-9
        lgamma_observed = np.vectorize(math.lgamma)(observed + 1)
        log_lik_array = observed * np.log(predicted) - predicted - lgamma_observed
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    total_log_likelihood = np.sum(log_lik_array)
    if np.isneginf(total_log_likelihood):
        return np.inf
    return -total_log_likelihood


def optimize_lineshape_params(
    measurement_history: list[dict[str, float]],
    current_estimates: dict[str, Any],
    distribution: str,
    noise_model: str,
    param_keys: list[str],
    bounds: list[tuple[float, float]],
) -> dict[str, Any]:
    """Run L-BFGS-B optimisation over lineshape parameters.

    Args:
        measurement_history: Accumulated coerced measurements.
        current_estimates: Full parameter estimates dict.
        distribution: ODMR lineshape name.
        noise_model: ``"gaussian"`` or ``"poisson"``.
        param_keys: Parameter names to optimise.
        bounds: Corresponding ``(low, high)`` bounds.

    Returns:
        Updated *current_estimates* dict (modified in-place **and** returned).
    """
    if len(measurement_history) < 5:
        return current_estimates

    if not param_keys:
        return current_estimates

    initial_guess = [current_estimates[key] for key in param_keys]

    sanitized_bounds = []
    for b in bounds:
        low, high = b
        if low == high:
            high = low + 1e-9
        sanitized_bounds.append((low, high))

    result = minimize(
        lambda p: _objective_func(p, param_keys, current_estimates, measurement_history, noise_model, distribution),
        initial_guess,
        bounds=sanitized_bounds,
        method="L-BFGS-B",
    )

    if result.success:
        for key, value in zip(param_keys, result.x, strict=False):
            current_estimates[key] = value

        try:
            hess_inv = result.hess_inv
            if hasattr(hess_inv, "todense"):
                hess_inv = hess_inv.todense()

            variances = np.diag(hess_inv)
            uncertainties = np.sqrt(np.maximum(variances, 0))

            for key, uncert in zip(param_keys, uncertainties, strict=False):
                current_estimates[f"{key}_uncertainty"] = uncert
        except Exception:
            pass

    return current_estimates
