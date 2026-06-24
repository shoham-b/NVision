"""Simulation default constants — single source of truth for env-driven sim config.

All hardcoded defaults for grid resolution, sweep parameters, noise levels,
and refocus strategies are defined here via environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv import load_dotenv

# Load environment variables from .env so they're available
# regardless of where this module is imported from.
load_dotenv()

# --- Core Locator Defaults -------------------------------------------------

NVISION_DEFAULT_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEFAULT_LOC_MAX_STEPS", "1500"))

# Fraction of SimpleSweep max_steps allocated to SBED and Sobol baseline locators.
NVISION_SBED_STEPS_FRACTION: float = float(os.getenv("NVISION_SBED_STEPS_FRACTION", "0.32"))
NVISION_SOBOL_STEPS_FRACTION: float = float(os.getenv("NVISION_SOBOL_STEPS_FRACTION", "0.5"))

# --- Robust Dip Detection Defaults -----------------------------------------

NVISION_DIP_N_SIGMA: float = float(os.getenv("NVISION_DIP_N_SIGMA", "3.0"))
NVISION_DIP_MIN_CLUSTER: int = int(os.getenv("NVISION_DIP_MIN_CLUSTER", "2"))
NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD: float = float(os.getenv("NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD", "0.15"))
NVISION_DIP_CONFIDENCE: float = float(os.getenv("NVISION_DIP_CONFIDENCE", "0.99"))


# --- Grid Resolution Defaults (belief_builders.py) -------------------------

NVISION_GRID_FREQ: int = int(os.getenv("NVISION_GRID_FREQ", "96"))
NVISION_GRID_WIDTH: int = int(os.getenv("NVISION_GRID_WIDTH", "64"))
NVISION_GRID_DEPTH: int = int(os.getenv("NVISION_GRID_DEPTH", "48"))
NVISION_GRID_BACKGROUND: int = int(os.getenv("NVISION_GRID_BACKGROUND", "48"))

# NV-center specific grid defaults
NVISION_NV_GRID_FREQ: int = int(os.getenv("NVISION_NV_GRID_FREQ", "500"))
NVISION_NV_GRID_LINEWIDTH: int = int(os.getenv("NVISION_NV_GRID_LINEWIDTH", "80"))
NVISION_NV_GRID_FWHM_TOTAL: int = int(os.getenv("NVISION_NV_GRID_FWHM_TOTAL", "80"))
NVISION_NV_GRID_LORENTZ_FRAC: int = int(os.getenv("NVISION_NV_GRID_LORENTZ_FRAC", "60"))
NVISION_NV_GRID_SPLIT: int = int(os.getenv("NVISION_NV_GRID_SPLIT", "80"))
NVISION_NV_GRID_K_NP: int = int(os.getenv("NVISION_NV_GRID_K_NP", "60"))
NVISION_NV_GRID_DEPTH: int = int(os.getenv("NVISION_NV_GRID_DEPTH", "100"))
NVISION_NV_GRID_BACKGROUND: int = int(os.getenv("NVISION_NV_GRID_BACKGROUND", "60"))

# --- Sobol Sweep Defaults (sobol_locator.py) ---------------------------------

NVISION_SOBOL_MIN_POINTS: int = int(os.getenv("NVISION_SOBOL_MIN_POINTS", "255"))
NVISION_SOBOL_MAX_POINTS: int = int(os.getenv("NVISION_SOBOL_MAX_POINTS", "511"))
NVISION_SOBOL_CHECK_INTERVAL: int = int(os.getenv("NVISION_SOBOL_CHECK_INTERVAL", "32"))
NVISION_SOBOL_MIN_DEPTH_SIGMA: float = float(os.getenv("NVISION_SOBOL_MIN_DEPTH_SIGMA", "2.5"))
NVISION_SOBOL_DEPTH_FRACTION: float = float(os.getenv("NVISION_SOBOL_DEPTH_FRACTION", "0.5"))
NVISION_SOBOL_PAD_FRACTION: float = float(os.getenv("NVISION_SOBOL_PAD_FRACTION", "0.005"))

# --- Sweep Steps Defaults (sweep_steps.py) -----------------------------------

NVISION_SWEEP_COVERAGE_FACTOR: float = float(os.getenv("NVISION_SWEEP_COVERAGE_FACTOR", "3.0"))
NVISION_SWEEP_MIN_STEPS: int = int(os.getenv("NVISION_SWEEP_MIN_STEPS", "50"))
NVISION_SWEEP_MAX_STEPS: int = int(os.getenv("NVISION_SWEEP_MAX_STEPS", "500"))

# --- Noise Preset Defaults (presets.py) --------------------------------------

NVISION_NOISE_GAUSS: float = float(os.getenv("NVISION_NOISE_GAUSS", "0.01"))
NVISION_NOISE_POISSON: float = float(os.getenv("NVISION_NOISE_POISSON", "3000.0"))
NVISION_NOISE_OVER_PROBE: float = float(os.getenv("NVISION_NOISE_OVER_PROBE", "0.001"))
NVISION_NOISE_MAX_GAUSS: float = float(os.getenv("NVISION_NOISE_MAX_GAUSS", "0.2"))
NVISION_NOISE_GAUSS_STEPS: int = int(os.getenv("NVISION_NOISE_GAUSS_STEPS", "5"))

# --- Window/Refocus Defaults (window.py) -------------------------------------

NVISION_WINDOW_PADDING_FRAC: float = float(os.getenv("NVISION_WINDOW_PADDING_FRAC", "0.05"))
NVISION_WINDOW_MIN_PADDING_FRAC: float = float(os.getenv("NVISION_WINDOW_MIN_PADDING_FRAC", "0.01"))

# --- NV Center Physical Parameter Bounds (nv_center.py) ----------------------

NVISION_MIN_K_NP: float = float(os.getenv("NVISION_MIN_K_NP", "1.0"))
NVISION_MAX_K_NP: float = float(os.getenv("NVISION_MAX_K_NP", "5.0"))
NVISION_MIN_LINEWIDTH: float = float(os.getenv("NVISION_MIN_LINEWIDTH", "200e3"))
NVISION_MAX_LINEWIDTH: float = float(os.getenv("NVISION_MAX_LINEWIDTH", "5.0e6"))
NVISION_MIN_SPLIT: float = float(os.getenv("NVISION_MIN_SPLIT", "3.0e6"))
NVISION_MAX_SPLIT: float = float(os.getenv("NVISION_MAX_SPLIT", "8.5e6"))
NVISION_NV_CENTER_FREQ_X_MIN: float = float(os.getenv("NVISION_NV_CENTER_FREQ_X_MIN", "2.6e9"))
NVISION_NV_CENTER_FREQ_X_MAX: float = float(os.getenv("NVISION_NV_CENTER_FREQ_X_MAX", "3.1e9"))
# --- Convergence Defaults ----------------------------------------------------

# Default relative convergence threshold (fraction of parameter bound width; 0.01 = 1%).
NVISION_CONVERGENCE_THRESHOLD: float = float(os.getenv("NVISION_CONVERGENCE_THRESHOLD", "0.01"))

# Absolute convergence ceilings for specific parameters (physical units).
# Unset optional vars fall back to relative NVISION_CONVERGENCE_THRESHOLD × bound width.
NVISION_FREQ_CONVERGENCE_THRESHOLD: float = float(os.getenv("NVISION_FREQ_CONVERGENCE_THRESHOLD", "100000.0"))

# Safety factor K applied as a measurement budget multiplier. 
# If the Cramér–Rao Lower Bound (CRLB) dictates that N measurements are 
# theoretically required to reach the frequency convergence threshold, the locator 
# allocates a maximum budget of K × N measurements to account for non-ideal 
# sampling and SMC inefficiencies before failing fast.
NVISION_FREQ_CRLB_SAFETY_FACTOR: float = float(os.getenv("NVISION_FREQ_CRLB_SAFETY_FACTOR", "4.0"))

# Minimum physical step (Hz) between consecutive EIG-evaluated candidates.
# Candidate count from the epoch grid ≈ 6·σ_f / NVISION_SMC_CANDIDATE_STEP_HZ,
# so the budget shrinks automatically as the posterior tightens.
# Defaults to NVISION_FREQ_CONVERGENCE_THRESHOLD (100 kHz).
NVISION_SMC_CANDIDATE_STEP_HZ: float = float(
    os.getenv("NVISION_SMC_CANDIDATE_STEP_HZ", str(NVISION_FREQ_CONVERGENCE_THRESHOLD))
)


def _optional_env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _param_absolute_convergence_thresholds() -> dict[str, float]:
    thresholds: dict[str, float] = {"frequency": NVISION_FREQ_CONVERGENCE_THRESHOLD}
    for param_name, env_name in (
        ("k_np", "NVISION_K_NP_CONVERGENCE_THRESHOLD"),
        ("linewidth", "NVISION_LINEWIDTH_CONVERGENCE_THRESHOLD"),
        ("split", "NVISION_SPLIT_CONVERGENCE_THRESHOLD"),
        ("dip_depth", "NVISION_DIP_DEPTH_CONVERGENCE_THRESHOLD"),
        ("fwhm_total", "NVISION_FWHM_TOTAL_CONVERGENCE_THRESHOLD"),
        ("lorentz_frac", "NVISION_LORENTZ_FRAC_CONVERGENCE_THRESHOLD"),
    ):
        value = _optional_env_float(env_name)
        if value is not None:
            thresholds[param_name] = value
    return thresholds


PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS: dict[str, float] = _param_absolute_convergence_thresholds()


def param_convergence_bound_width(
    param_name: str,
    relative_threshold: float,
    physical_bounds: Mapping[str, tuple[float, float]],
) -> float:
    """Effective bound width for relative convergence comparison.

    Parameters with an entry in ``PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS`` use
    ``absolute_threshold / relative_threshold`` so that
    ``unc / bound_width < relative_threshold`` iff ``unc < absolute_threshold``.
    """
    absolute = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS.get(param_name)
    if absolute is not None:
        return absolute / relative_threshold
    lo, hi = physical_bounds.get(param_name, (0.0, 0.0))
    return hi - lo


def param_converged(
    param_name: str,
    uncertainty: float,
    relative_threshold: float,
    physical_bounds: Mapping[str, tuple[float, float]],
) -> bool:
    """Return whether ``uncertainty`` meets convergence for ``param_name``."""
    absolute = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS.get(param_name)
    if absolute is not None:
        return uncertainty < absolute
    bound_width = param_convergence_bound_width(param_name, relative_threshold, physical_bounds)
    if bound_width <= 0:
        return False
    return uncertainty / bound_width < relative_threshold
