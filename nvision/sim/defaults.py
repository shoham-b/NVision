"""Simulation default constants — single source of truth for env-driven sim config.

All hardcoded defaults for grid resolution, sweep parameters, noise levels,
and refocus strategies are defined here via environment variables.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load environment variables from .env so they're available
# regardless of where this module is imported from.
load_dotenv()

# --- Core Locator Defaults -------------------------------------------------

NVISION_DEFAULT_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEFAULT_LOC_MAX_STEPS", "1500"))

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

# --- Window/Refocus Defaults (window.py) -------------------------------------

NVISION_WINDOW_PADDING_FRAC: float = float(os.getenv("NVISION_WINDOW_PADDING_FRAC", "0.05"))
NVISION_WINDOW_MIN_PADDING_FRAC: float = float(os.getenv("NVISION_WINDOW_MIN_PADDING_FRAC", "0.01"))
