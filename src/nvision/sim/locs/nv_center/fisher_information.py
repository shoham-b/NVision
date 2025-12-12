"""
Fisher Information and Cramér-Rao Bound calculations for NV Center ODMR.
"""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.nv_center._jit_kernels import _calculate_fisher_info_jit


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
    x = np.array(measurement_x, dtype=np.float64)

    # Prepare params array [f0, linewidth, amplitude, background]
    params_array = np.array(
        [
            true_params["frequency"],
            true_params["linewidth"],
            true_params["amplitude"],
            true_params.get("background", 1.0),
        ],
        dtype=np.float64,
    )

    noise_model_code = 0
    noise_val = 0.05

    if noise_model == "gaussian":
        noise_model_code = 0
        if noise_params and "sigma" in noise_params:
            noise_val = noise_params["sigma"]
    elif noise_model == "poisson":
        noise_model_code = 1
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    fi_cumulative = _calculate_fisher_info_jit(x, params_array, noise_model_code, noise_val)

    return fi_cumulative


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
