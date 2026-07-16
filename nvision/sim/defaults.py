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

# Actual step count SimpleSweep itself runs with. Kept independent of the
# domain/min_linewidth figure (e.g. ~2500) that SBED/Sobol fractions are still
# computed against — only SimpleSweep's own budget is capped here.
NVISION_SIMPLESWEEP_MAX_STEPS: int = int(os.getenv("NVISION_SIMPLESWEEP_MAX_STEPS", "400"))

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
NVISION_MIN_SPLIT: float = float(os.getenv("NVISION_MIN_SPLIT", "2.0e6"))
NVISION_MAX_SPLIT: float = float(os.getenv("NVISION_MAX_SPLIT", "8.5e6"))
# NV center frequency domain is configured via NVISION_NV_ZERO_FIELD_SPLITTING_HZ /
# NVISION_NV_CENTER_FREQ_DELTA_HZ, read directly in nvision/spectra/nv_center.py
# (DEFAULT_NV_CENTER_FREQ_X_MIN/MAX) so x_min/x_max always stay symmetric around
# the physical zero-field center and can't drift out of sync.
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


# --- CRLB feasibility gate & background-noise estimation --------------------

# Pre-run fail gate: infeasible if any param CRLB > threshold × this margin.
# 1.0 = require CRLB strictly below threshold; increase to allow slight infeasibility.
NVISION_CRLB_FEASIBILITY_MARGIN: float = float(os.getenv("NVISION_CRLB_FEASIBILITY_MARGIN", "1.0"))

# Minimum number of out-of-span (background) observations before trusting σ̂.
# If fewer exist, forced background calibration is triggered in SBED.
NVISION_NOISE_MIN_BG_POINTS: int = int(os.getenv("NVISION_NOISE_MIN_BG_POINTS", "15"))

# Half-span radius (in linewidths) used to classify a measurement as background.
# Points with |x - f̂| > k * linewidth_hat are considered background.
NVISION_NOISE_BG_SPAN_FACTOR: float = float(os.getenv("NVISION_NOISE_BG_SPAN_FACTOR", "3.0"))

# Permissive multiplier on the theoretical step count n_theory for SBED's backstop limit.
# n_theory = 2σ̂²·lw·bandwidth / (π·c²·T²); stops when inference_step_count > K × n_theory.
# Large value (20) so this only fires when something has genuinely gone wrong.
NVISION_SBED_STEPS_THEORY_FACTOR: float = float(os.getenv("NVISION_SBED_STEPS_THEORY_FACTOR", "20"))


def _optional_env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


# --- Signal Parameter Fixing / Sweeping (nv_center_generator.py, presets.py) --

# Fixed linewidth (HWHM, Hz) / contrast (c_total). None = randomize as before.
NVISION_SIGNAL_LINEWIDTH: float | None = _optional_env_float("NVISION_SIGNAL_LINEWIDTH")
NVISION_SIGNAL_CONTRAST: float | None = _optional_env_float("NVISION_SIGNAL_CONTRAST")

# When set to "width" or "contrast", generators_basic() sweeps that parameter over
# NVISION_SIGNAL_SWEEP_STEPS values in [NVISION_SIGNAL_SWEEP_MIN, NVISION_SIGNAL_SWEEP_MAX],
# holding the other signal parameter fixed at its NVISION_SIGNAL_* value (or randomizing it
# per repeat if unset). "noise" or None disables signal-parameter sweeping.
NVISION_SIGNAL_SWEEP_PARAM: str | None = os.getenv("NVISION_SIGNAL_SWEEP_PARAM") or None
NVISION_SIGNAL_SWEEP_MIN: float | None = _optional_env_float("NVISION_SIGNAL_SWEEP_MIN")
NVISION_SIGNAL_SWEEP_MAX: float | None = _optional_env_float("NVISION_SIGNAL_SWEEP_MAX")
NVISION_SIGNAL_SWEEP_STEPS: int = int(os.getenv("NVISION_SIGNAL_SWEEP_STEPS", "5"))

# Width x contrast grid for the (lorentzian) NVCenterCoreGenerator — used by
# sim.presets.param_grid_generators(). Not the SBED run-groups' default anymore
# (see the saturation-Voigt grid below); kept for direct/manual lorentzian studies.
NVISION_SBED_WIDTH_MIN: float = float(os.getenv("NVISION_SBED_WIDTH_MIN", "5e5"))
NVISION_SBED_WIDTH_MAX: float = float(os.getenv("NVISION_SBED_WIDTH_MAX", "5e6"))
NVISION_SBED_WIDTH_STEPS: int = int(os.getenv("NVISION_SBED_WIDTH_STEPS", "5"))
NVISION_SBED_CONTRAST_MIN: float = float(os.getenv("NVISION_SBED_CONTRAST_MIN", "0.1"))
NVISION_SBED_CONTRAST_MAX: float = float(os.getenv("NVISION_SBED_CONTRAST_MAX", "0.4"))
NVISION_SBED_CONTRAST_STEPS: int = int(os.getenv("NVISION_SBED_CONTRAST_STEPS", "5"))

# Default grid for the SBED run-groups (lorentzian-sbed and variants) in
# run_groups.py: saturation-Voigt lineshape, swept over target contrast (the
# same NVISION_SBED_CONTRAST_* range as the lorentzian/voigt width x contrast
# grids above, for direct cross-lineshape comparability) and sigma_inhom
# (independent inhomogeneous/Gaussian width). saturation is *solved* per grid
# point from the target contrast via C = c_max*s/(1+s) => s = C/(c_max-C) --
# see saturation_voigt_param_grid_generators() in presets.py. c_max is held
# fixed, comfortably above NVISION_SBED_CONTRAST_MAX (contrast can only
# approach c_max asymptotically, never reach it) so the solved saturation
# stays moderate (0.25-4.0 across the default contrast range, well inside the
# model's own 0.02-30 prior bounds). A previous design swept saturation
# linearly instead; since s/(1+s) saturates fast, most of that grid barely
# moved contrast at all, while sweeping contrast directly makes every point
# move the signal by an equal, intended amount.
NVISION_SBED_SIGMA_INHOM_MIN: float = float(os.getenv("NVISION_SBED_SIGMA_INHOM_MIN", "0.0"))
NVISION_SBED_SIGMA_INHOM_MAX: float = float(os.getenv("NVISION_SBED_SIGMA_INHOM_MAX", "1.2e6"))
NVISION_SBED_SIGMA_INHOM_STEPS: int = int(os.getenv("NVISION_SBED_SIGMA_INHOM_STEPS", "5"))
# Must equal nvision.spectra.nv_center.NV_SATURATION_C_MAX (the model's fixed,
# non-inferred saturated-contrast constant) -- reads the identical env var/
# default so the two can never drift apart.
NVISION_SBED_C_MAX: float = float(os.getenv("NVISION_SBED_C_MAX", "0.5"))

# NVISION_SBED_SIGMA_INHOM_MIN/MAX/STEPS (above) is also reused by
# sim.presets.voigt_sigma_inhom_param_grid_generators() for the plain-Voigt
# width x contrast x sigma_inhom grid, so the inhomogeneous-broadening axis is
# directly comparable between the voigt and saturation_voigt lineshape studies.

# Noise is swept the same way as width/contrast above — its own dedicated range,
# independent of the generic NVISION_NOISE_MAX_GAUSS/NVISION_NOISE_GAUSS_STEPS
# (which are shared by every other run path, e.g. `nvision run` without a group,
# and often tuned via .env for unrelated purposes).
NVISION_SBED_NOISE_MIN: float = float(os.getenv("NVISION_SBED_NOISE_MIN", "0.0"))
# Capped at 0.01: the lowest contrast any grid point targets is
# NVISION_SBED_CONTRAST_MIN (0.1 default, shared across all three lineshape study
# grids -- see above), so a noise ceiling of 0.1 would leave the worst grid point
# at SNR ~= 1 (undetectable in practice). 0.01 keeps SNR >= ~10 even there.
NVISION_SBED_NOISE_MAX: float = float(os.getenv("NVISION_SBED_NOISE_MAX", "0.01"))
# Denser than the minimum needed, so metrics-vs-noise plots show a clear trend
# as noise grows across the feasible range rather than just a few points.
NVISION_SBED_NOISE_STEPS: int = int(os.getenv("NVISION_SBED_NOISE_STEPS", "6"))


def _param_absolute_convergence_thresholds() -> dict[str, float]:
    thresholds: dict[str, float] = {"frequency": NVISION_FREQ_CONVERGENCE_THRESHOLD}
    for param_name, env_name in (
        ("k_np", "NVISION_K_NP_CONVERGENCE_THRESHOLD"),
        ("linewidth", "NVISION_LINEWIDTH_CONVERGENCE_THRESHOLD"),
        ("split", "NVISION_SPLIT_CONVERGENCE_THRESHOLD"),
        ("dip_depth", "NVISION_DIP_DEPTH_CONVERGENCE_THRESHOLD"),
        ("homogeneous_linewidth", "NVISION_HOMOGENEOUS_LINEWIDTH_CONVERGENCE_THRESHOLD"),
        ("sigma_inhom", "NVISION_SIGMA_INHOM_CONVERGENCE_THRESHOLD"),
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
