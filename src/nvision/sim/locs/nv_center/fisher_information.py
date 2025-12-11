"""
Fisher Information and Cramér-Rao Bound calculations for NV Center ODMR.
"""

from __future__ import annotations

import numpy as np


def _lorentzian_deriv_f0(
    frequencies: np.ndarray,
    f0: float,
    linewidth: float,
    amplitude: float,
) -> np.ndarray:
    """
    Calculate derivative of Lorentzian signal w.r.t resonance frequency f0.

    Model: S(f) = Background - Amplitude * (gamma/2)^2 / ((f - f0)^2 + (gamma/2)^2)
    Derivative dS/df0:
    Let hwhm = gamma/2
    denom = (f - f0)^2 + hwhm^2
    dS/df0 = -Amplitude * hwhm^2 * (-1) * denom^(-2) * 2(f - f0) * (-1)
           = -Amplitude * hwhm^2 * 2(f - f0) / denom^2
    """
    hwhm = linewidth / 2.0
    diff = frequencies - f0
    denom = diff**2 + hwhm**2
    return -amplitude * (hwhm**2) * 2 * diff / (denom**2)


def calculate_fisher_information(
    measurement_x: list[float] | np.ndarray,
    true_params: dict[str, float],
    noise_model: str = "gaussian",
    noise_params: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculate the cumulative Fisher Information for the frequency parameter f0.

    Args:
        measurement_x: Sequence of measurement frequencies.
        true_params: Dictionary of true parameters (frequency, linewidth, amplitude, background).
        noise_model: "gaussian" or "poisson".
        noise_params: Extra noise parameters (e.g., {'sigma': 0.05} for gaussian).

    Returns:
        Cumulative Fisher Information array corresponding to each step.
    """
    x = np.array(measurement_x)
    f0 = true_params["frequency"]
    gamma = true_params["linewidth"]
    amp = true_params["amplitude"]
    bg = true_params.get("background", 1.0)

    # Calculate derivative of the mean signal w.r.t f0
    # dmu/df0
    deriv = _lorentzian_deriv_f0(x, f0, gamma, amp)

    # Calculate Fisher Information for each measurement
    # I(f0) = (dmu/df0)^2 * Weight

    if noise_model == "gaussian":
        sigma = 0.05
        if noise_params and "sigma" in noise_params:
            sigma = noise_params["sigma"]
        # Weight = 1/sigma^2
        # I = (dmu/df0)^2 / sigma^2
        fi_per_step = (deriv**2) / (sigma**2)

    elif noise_model == "poisson":
        # Weight = 1/mu
        # I = (dmu/df0)^2 / mu

        # Recalculate mean signal mu for the weight
        hwhm = gamma / 2.0
        denom = (x - f0) ** 2 + hwhm**2
        mu = bg - (amp * hwhm**2) / denom

        # Avoid division by zero or negative mu (though physics says mu > 0)
        mu = np.maximum(mu, 1e-9)

        fi_per_step = (deriv**2) / mu

    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    return np.cumsum(fi_per_step)


def calculate_crb(
    measurement_x: list[float] | np.ndarray,
    true_params: dict[str, float],
    noise_model: str = "gaussian",
    noise_params: dict[str, float] | None = None,
) -> list[float]:
    """
    Calculate the Cramér-Rao Bound (Standard Deviation limit) history.
    CRB_std = sqrt(1 / Cumulative_FI)
    """
    fi_cumulative = calculate_fisher_information(measurement_x, true_params, noise_model, noise_params)

    # Avoid division by zero for the first step(s) if FI is 0 (unlikely but possible)
    # or handle initial state where information is effectively prior information?
    # Usually CRB is for the estimator variance.
    # If FI is 0, CRB is inf.

    crb_var = np.zeros_like(fi_cumulative)
    mask = fi_cumulative > 1e-15
    crb_var[mask] = 1.0 / fi_cumulative[mask]
    crb_var[~mask] = np.inf

    return np.sqrt(crb_var).tolist()
