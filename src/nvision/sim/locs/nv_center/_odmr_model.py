"""ODMR model dispatch and likelihood calculations for NV center Bayesian locators."""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.nv_center._jit_kernels import (
    _calculate_log_likelihoods_grid_jit,
    _gaussian_log_likelihood,
    _lorentzian_model,
    _poisson_log_likelihood,
    _poisson_log_likelihood_scalar,
    _poisson_log_likelihood_scalar_obs,
    _voigt_model,
    _voigt_zeeman_model,
)


def odmr_model(
    frequency: float | np.ndarray,
    params: dict[str, float | np.ndarray],
    distribution: str,
) -> np.ndarray:
    """Compute the ODMR signal for the given distribution type.

    Args:
        frequency: Probe frequency (scalar or array).
        params: Dict with keys ``frequency``, ``linewidth``, ``amplitude``,
            ``background`` (and ``gaussian_width``, ``split``, ``k_np`` for
            Voigt / Voigt-Zeeman).
        distribution: One of ``"lorentzian"``, ``"voigt"``, ``"voigt-zeeman"``.

    Returns:
        Predicted signal value(s).

    Raises:
        ValueError: If *distribution* is not recognised.
    """
    f0 = params["frequency"]
    gamma = params["linewidth"]
    amplitude = params["amplitude"]
    bg = params["background"]

    if distribution == "lorentzian":
        return _lorentzian_model(frequency, f0, gamma, amplitude, bg)

    if distribution == "voigt":
        sigma = params["gaussian_width"]
        return _voigt_model(frequency, f0, gamma, sigma, amplitude, bg)

    if distribution == "voigt-zeeman":
        split = params["split"]
        k_np = params["k_np"]
        return _voigt_zeeman_model(frequency, f0, gamma, split, k_np, amplitude, bg)

    raise ValueError(f"Unknown distribution: {distribution}")


def coerce_measurement(measurement: dict[str, float]) -> dict[str, float]:
    """Normalise a measurement dict to ``{x, signal_values, uncertainty}``."""
    if "frequency" in measurement:
        return {
            "x": measurement.get("x", measurement["frequency"]),
            "signal_values": measurement.get("signal_values", measurement["intensity"]),
            "uncertainty": measurement.get("uncertainty", 0.05),
        }
    if "x" in measurement and "signal_values" in measurement:
        return measurement
    raise KeyError("measurement must include either frequency/intensity or x/signal_values")


def compute_likelihood(
    measurement: dict[str, float],
    params: dict[str, float | np.ndarray],
    distribution: str,
    noise_model: str,
) -> np.ndarray:
    """Compute log-likelihood of *measurement* given *params*.

    Args:
        measurement: Raw or coerced measurement dict.
        params: Model parameters (must include ``frequency``).
        distribution: ODMR lineshape name.
        noise_model: ``"gaussian"`` or ``"poisson"``.

    Returns:
        Log-likelihood (scalar or array).

    Raises:
        ValueError: If *noise_model* is not recognised.
    """
    measurement = coerce_measurement(measurement)
    predicted = odmr_model(measurement["x"], params, distribution)
    observed = np.array(measurement["signal_values"])
    if observed.ndim == 1:
        observed = observed[:, np.newaxis]

    sigma = 0.05  # Placeholder

    if noise_model == "gaussian":
        log_const = -4.153244906  # log(2 * pi * 0.05^2)
        return _gaussian_log_likelihood(observed, predicted, sigma, log_const)

    if noise_model == "poisson":
        obs_is_scalar = False
        if isinstance(observed, (int, float)):
            obs_is_scalar = True
            obs_val = float(observed)
        elif isinstance(observed, np.ndarray) and observed.ndim == 0:
            obs_is_scalar = True
            obs_val = float(observed.item())
        else:
            obs_val = observed

        pred_is_array = isinstance(predicted, np.ndarray) and predicted.ndim > 0

        if obs_is_scalar:
            if pred_is_array:
                return _poisson_log_likelihood_scalar_obs(obs_val, predicted)
            return _poisson_log_likelihood_scalar(obs_val, float(predicted))
        return _poisson_log_likelihood(obs_val, np.atleast_1d(predicted))

    raise ValueError(f"Unknown noise model: {noise_model}")


def calculate_log_likelihoods_grid(
    freq_grid: np.ndarray,
    measurement: dict[str, float],
    base_params: dict[str, float],
    distribution: str,
    noise_model: str,
) -> np.ndarray:
    """Calculate log-likelihoods over the frequency grid using vectorised JIT kernels.

    Args:
        freq_grid: 1-D array of candidate frequencies.
        measurement: Already-coerced measurement dict.
        base_params: Current parameter estimates (without ``frequency``).
        distribution: ODMR lineshape name.
        noise_model: ``"gaussian"`` or ``"poisson"``.

    Returns:
        1-D array of log-likelihoods aligned with *freq_grid*.
    """
    params_array = np.array(
        [
            base_params["linewidth"],
            base_params["amplitude"],
            base_params["background"],
            base_params.get("gaussian_width", 0.0),
            base_params.get("split", 0.0),
            base_params.get("k_np", 0.0),
        ],
        dtype=np.float64,
    )

    dist_code_map = {"lorentzian": 0, "voigt": 1, "voigt-zeeman": 2}
    dist_code = dist_code_map.get(distribution, 0)

    noise_model_code = 0 if noise_model == "gaussian" else 1

    mx = float(measurement["x"])
    my = float(measurement["signal_values"])
    uncert = float(measurement.get("uncertainty", 0.05))

    return _calculate_log_likelihoods_grid_jit(freq_grid, mx, my, uncert, params_array, dist_code, noise_model_code)
