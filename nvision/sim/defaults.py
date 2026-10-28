"""Simulation default constants — single source of truth for env-driven sim config.

All hardcoded defaults for grid resolution, sweep parameters, noise levels,
and refocus strategies are defined here via environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv import load_dotenv

from nvision.tools.renamed_env import reject_renamed_env_vars

# Load environment variables from .env so they're available
# regardless of where this module is imported from.
load_dotenv()
reject_renamed_env_vars()

# --- Core Locator Defaults -------------------------------------------------

NVISION_DEFAULT_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEFAULT_LOC_MAX_STEPS", "1500"))

# Actual step count SimpleSweep itself runs with. Kept independent of the
# domain/min_hwhm figure (e.g. ~2500) that the SBED fraction is still
# computed against — only SimpleSweep's own budget is capped here.
NVISION_SIMPLESWEEP_MAX_STEPS: int = int(os.getenv("NVISION_SIMPLESWEEP_MAX_STEPS", "400"))

# Build the per-repeat plots (scan figure, posterior/Fisher/convergence extras) for the
# SimpleSweep strategies. Off by default: these tasks still run and keep
# their data and metrics (results rows, per-step `series`, and the baselines Bayesian-SBED
# compares against), but the figures cost more than the locator run itself and are rarely
# opened. Bayesian-SBED plots are unaffected. Set to 1 to build them again.
NVISION_PLOT_SWEEP_STRATEGIES: bool = os.getenv("NVISION_PLOT_SWEEP_STRATEGIES", "0").lower() in ("1", "true", "yes")

# Build graphs only for the first N repeats of each combination (repeat indices 0..N-1); later
# repeats keep their metrics, per-step series and results row but get no figures. Graphs are
# a small part of a repeat's compute but a lot of stored bytes, and only a few repeats per
# combination are ever inspected. 0 (or negative) builds graphs for every repeat.
NVISION_GRAPH_REPEATS: int = int(os.getenv("NVISION_GRAPH_REPEATS", "10"))

# Sweep curve-fit early stop: once a fit start converges to the noise floor -- reduced
# chi-square <= 1 + K * sqrt(2 / dof), i.e. within K standard deviations of the ~1 a
# correct fit gives -- the remaining starting points are skipped instead of all being
# raced. Only applied when the peak SNR is high enough that the smart seeds are trusted
# (below that the residual gap between basins is within the noise band, so the full
# multi-start race stays). 0 disables early stop (race every start, as before).
NVISION_SWEEP_FIT_EARLY_STOP_SIGMAS: float = float(os.getenv("NVISION_SWEEP_FIT_EARLY_STOP_SIGMAS", "3.0"))

# Sweep inference without an SMC belief (GenericSweepLocator ``fit_only``): the dip scan and a least-squares fit
# replace the per-step particle filter. The first dip triggers one full multi-start fit; later fits start from the
# previous one. Off by default; the SMC sweep is unchanged. Results differ from the SMC sweep's (the uncertainty
# is the fit covariance), so they are cached apart from it.
NVISION_SWEEP_FIT_ONLY: bool = os.getenv("NVISION_SWEEP_FIT_ONLY", "0").lower() in ("1", "true", "yes")

# With ``fit_only``, after the first dip a warm-started fit runs every this many measurements (and at the end of
# every pass over the grid).
NVISION_SWEEP_FIT_EVERY: int = max(1, int(os.getenv("NVISION_SWEEP_FIT_EVERY", "20")))

# --- Robust Dip Detection Defaults -----------------------------------------

# Family-wise false-alarm probability of the boxcar dip scan (Bonferroni over every window
# tested), see nvision.belief.dip_detection.find_dips. 0.1 declares the dip (the focus stage start) a few to ~10
# measurements earlier than 0.01 with no off-dip detections on the single-dip SNR(10) probe; 0.2 and above start
# producing them.
NVISION_DIP_P_FA: float = float(os.getenv("NVISION_DIP_P_FA", "0.1"))

# The stopping rules re-run the dip scan at most once per this many measurements (the scan grows with the number
# of measurements, and a dip already seen does not vanish between two consecutive measurements).
NVISION_DIP_SCAN_EVERY: int = max(1, int(os.getenv("NVISION_DIP_SCAN_EVERY", "10")))


# --- Sweep Steps Defaults (sweep_steps.py) -----------------------------------

NVISION_SWEEP_COVERAGE_FACTOR: float = float(os.getenv("NVISION_SWEEP_COVERAGE_FACTOR", "3.0"))
NVISION_SWEEP_MIN_STEPS: int = int(os.getenv("NVISION_SWEEP_MIN_STEPS", "50"))
NVISION_SWEEP_MAX_STEPS: int = int(os.getenv("NVISION_SWEEP_MAX_STEPS", "500"))
# Measurement budget of every locator in a run (the SBED family's max_steps, the plain sweeps' total measurements,
# capped in whole passes): one shared budget, so no locator is compared on a longer leash than another.
NVISION_MAX_MEASUREMENTS: int = int(os.getenv("NVISION_MAX_MEASUREMENTS", "2500"))

# Step budget for the Bayesian-SBED locator (nvision/sim/combinations.py).
# Read through the module (``sim_defaults.NVISION_SBED_MAX_STEPS``) rather than
# from-imported, so callers that want a cheaper run -- `nv demo`/`nv beta` -- can
# lower it at runtime the same way `nv run` already overrides the noise grid.
NVISION_SBED_MAX_STEPS: int = int(os.getenv("NVISION_SBED_MAX_STEPS", "200"))

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
NVISION_MIN_HWHM: float = float(os.getenv("NVISION_MIN_HWHM", "200e3"))
NVISION_MAX_HWHM: float = float(os.getenv("NVISION_MAX_HWHM", "5.0e6"))
NVISION_MIN_HYPERFINE_SHIFT: float = float(os.getenv("NVISION_MIN_HYPERFINE_SHIFT", "2.0e6"))
NVISION_MAX_HYPERFINE_SHIFT: float = float(os.getenv("NVISION_MAX_HYPERFINE_SHIFT", "8.5e6"))
# NV center drive-frequency domain is configured via NVISION_NV_ZERO_FIELD_SPLITTING_HZ /
# NVISION_NV_DRIVE_FREQ_DELTA_HZ, read directly in nvision/spectra/nv_center.py
# (DEFAULT_NV_DRIVE_FREQ_MIN_PHYS/MAX): the window is the upper half [D, D + delta] of the
# mirror-symmetric spectrum around the physical zero-field center D.
# --- Convergence Defaults ----------------------------------------------------

# Default relative convergence threshold (fraction of parameter bound width; 0.01 = 1%).
NVISION_CONVERGENCE_THRESHOLD: float = float(os.getenv("NVISION_CONVERGENCE_THRESHOLD", "0.01"))

# Absolute convergence ceilings for specific parameters (physical units).
# Unset optional vars fall back to relative NVISION_CONVERGENCE_THRESHOLD × bound width.
NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD: float = float(
    os.getenv("NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD", "100000.0")
)

# Safety factor K applied as a measurement budget multiplier.
# If the Cramér–Rao Lower Bound (CRLB) dictates that N measurements are
# theoretically required to reach the zero_field_freq convergence threshold, the locator
# allocates a maximum budget of K × N measurements to account for non-ideal
# sampling and SMC inefficiencies before failing fast.
NVISION_ZERO_FIELD_FREQ_CRLB_SAFETY_FACTOR: float = float(
    os.getenv("NVISION_ZERO_FIELD_FREQ_CRLB_SAFETY_FACTOR", "4.0")
)

# Minimum physical step (Hz) between consecutive EIG-evaluated candidates.
# Candidate count from the epoch grid ≈ 6·σ_f / NVISION_SMC_CANDIDATE_STEP_HZ,
# so the budget shrinks automatically as the posterior tightens.
# Defaults to NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD (100 kHz).
NVISION_SMC_CANDIDATE_STEP_HZ: float = float(
    os.getenv("NVISION_SMC_CANDIDATE_STEP_HZ", str(NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD))
)


# --- Background-noise estimation ---------------------------------------------


def _optional_env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


# --- Signal Parameter Fixing / Sweeping (nv_center_generator.py, presets.py) --

# Fixed hwhm (HWHM, Hz) / contrast (contrast_total). None = randomize as before.
NVISION_SIGNAL_HWHM: float | None = _optional_env_float("NVISION_SIGNAL_HWHM")
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

# Grid for the saturation-Voigt study generators (sim.presets; no longer a run-group
# in run_groups.py, but historical results still resolve): saturation-Voigt lineshape, swept over target contrast (the
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

# Nitrogen isotope values swept as a fourth axis by
# sim.presets.voigt_sigma_inhom_param_grid_generators() (see NVCenterCoreGenerator's
# `hyperfine` field for what each value means). Order matters only for display;
# CombinationGrid.strategies_for() reads the isotope back out of the generator
# name to keep the locator's belief in sync with whichever value produced the
# true signal -- see nv_center_smc_belief's hyperfine/infer_hyperfine docstring
# for why a mismatch there silently ruins a run instead of erroring.
NVISION_SBED_HYPERFINE_VALUES: tuple[str, ...] = tuple(
    v.strip() for v in os.getenv("NVISION_SBED_HYPERFINE_VALUES", "unresolved,n14,n15").split(",") if v.strip()
)

# --- SNR x window study (sim.presets.voigt_snr_window_param_grid_generators) --------------------
#
# The grid is posed in the dimensionless numbers that decide how hard a search is, instead of in Hz and
# absolute noise, and varies one thing per axis:
#   * SNR = contrast_total / noise sigma. Every generator carries the same contrast_total
#     (NVISION_SBED_SNR_CONTRAST); a run is named SNR(x) and its sigma is contrast / x. 1 is the smallest SNR
#     worth running (every locator hits its budget there) and 20 is past where the locators converge, so the
#     levels are spaced geometrically between them.
#   * window = N x the signal's total FWHM, with ONE total FWHM for every signal (NVISION_SBED_FWHM), so the
#     window is the same absolute 10 / 20 / 40 MHz whatever the line shape. 5-20 FWHM is what the study needs:
#     the search is hard when the dips are a sizeable part of the window, not only when it is orders of
#     magnitude wider than the line.
#   * Gaussian share of that FWHM (NVISION_SBED_GAUSSIAN_SHARES): 0 = pure Lorentzian, larger = more
#     inhomogeneous broadening, at the SAME total FWHM, so it changes the line shape and nothing else.
#   * the isotope is held fixed (NVISION_SBED_STUDY_ISOTOPE): it rescales the per-dip depth, i.e. the SNR.
NVISION_SBED_SNR_VALUES: tuple[float, ...] = tuple(
    float(v) for v in os.getenv("NVISION_SBED_SNR_VALUES", "1,2,5,10,20").split(",") if v.strip()
)
# Share of the window one dip group covers (its hyperfine lines' FWHM intervals, overlap counted once). The window
# of a signal is that covered width / occupancy, so the same occupancy is a different absolute window per isotope.
NVISION_SBED_SIGNAL_OCCUPANCY_VALUES: tuple[float, ...] = tuple(
    float(v) for v in os.getenv("NVISION_SBED_SIGNAL_OCCUPANCY_VALUES", "0.5,0.2,0.1,0.05").split(",") if v.strip()
)
NVISION_SBED_FWHM: float = float(os.getenv("NVISION_SBED_FWHM", "2e6"))
NVISION_SBED_GAUSSIAN_SHARES: tuple[float, ...] = tuple(
    float(v) for v in os.getenv("NVISION_SBED_GAUSSIAN_SHARES", "0,0.15,0.3").split(",") if v.strip()
)
NVISION_SBED_STUDY_ISOTOPE: str = os.getenv("NVISION_SBED_STUDY_ISOTOPE", "unresolved").strip()
NVISION_SBED_SNR_CONTRAST: float = float(os.getenv("NVISION_SBED_SNR_CONTRAST", "0.25"))
# The noise axes of the other studies are SNR levels against the same contrast: drive-control uses
# NVISION_SBED_SNR_VALUES (a nominal SNR there: its contrast comes from the drive, not a fixed value).

# SNR (against NVISION_SBED_SNR_CONTRAST) every drift scenario in the drift run-groups is paired with
# (sim.presets.drift_study_noises); the Gaussian noise is NVISION_SBED_SNR_CONTRAST / this.
NVISION_DRIFT_SNR: float = float(os.getenv("NVISION_DRIFT_SNR", "40"))


# Convergence targets for the NV shape parameters, in the units of each parameter.
#
# These replace the old fall-back of ``(bound_hi - bound_lo) * convergence_threshold``,
# which set the target from how wide a parameter's *prior box* happens to be rather than
# from anything achievable. Measured consequences of that fallback on the voigt grid
# (64 runs, median final uncertainty vs the threshold it had to clear):
#
#   param                  old target   achieved unc   blocked convergence in
#   zeeman_shift             600 kHz       26 kHz        0% of runs  <- never bound
#   hwhm_lorentz    148 kHz      184 kHz       65% of runs
#   sigma_inhom               60 kHz      175 kHz       85% of runs  <- main blocker
#   contrast_total                    0.003       0.0029       45% of runs
#
# So the run length was set by `sigma_inhom`'s 60 kHz target -- a number that comes from
# its 6 MHz bound range, is below the noise floor for these signals, and is asking for
# something physically unavailable: `hwhm_lorentz` and `sigma_inhom` are the
# degenerate width pair, individually unidentifiable below the dip-resolution threshold
# (only their combined width is determined). Meanwhile `zeeman_shift` -- the parameter
# the study actually cares about -- had a target 23x looser than its own achieved
# uncertainty and never gated anything.
#
# The widths and contrast are therefore loosened to values they can actually reach, and
# `zeeman_shift` is made the binding constraint at the same 100 kHz used for zero_field_freq.
# `max_steps` remains the backstop for signals where even that is unreachable (the
# genuinely degenerate ones), which is the correct outcome rather than a convergence claim.
NVISION_ZEEMAN_SHIFT_CONVERGENCE_THRESHOLD: float = float(
    os.getenv("NVISION_ZEEMAN_SHIFT_CONVERGENCE_THRESHOLD", "100000.0")
)
NVISION_WIDTH_CONVERGENCE_THRESHOLD: float = float(os.getenv("NVISION_WIDTH_CONVERGENCE_THRESHOLD", "500000.0"))
NVISION_CONTRAST_TOTAL_CONVERGENCE_THRESHOLD: float = float(
    os.getenv("NVISION_CONTRAST_TOTAL_CONVERGENCE_THRESHOLD", "0.01")
)
NVISION_NOISE_SIGMA_CONVERGENCE_THRESHOLD: float = float(
    os.getenv("NVISION_NOISE_SIGMA_CONVERGENCE_THRESHOLD", "0.001")
)


def _param_absolute_convergence_thresholds() -> dict[str, float]:
    thresholds: dict[str, float] = {
        "zero_field_freq": NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD,
        "zeeman_shift": NVISION_ZEEMAN_SHIFT_CONVERGENCE_THRESHOLD,
        "hwhm_lorentz": NVISION_WIDTH_CONVERGENCE_THRESHOLD,
        "sigma_inhom": NVISION_WIDTH_CONVERGENCE_THRESHOLD,
        "contrast_total": NVISION_CONTRAST_TOTAL_CONVERGENCE_THRESHOLD,
        "noise_sigma": NVISION_NOISE_SIGMA_CONVERGENCE_THRESHOLD,
    }
    for param_name, env_name in (
        ("k_np", "NVISION_K_NP_CONVERGENCE_THRESHOLD"),
        ("hwhm", "NVISION_HWHM_CONVERGENCE_THRESHOLD"),
        ("hyperfine_shift", "NVISION_SPLIT_CONVERGENCE_THRESHOLD"),
        ("contrast", "NVISION_CONTRAST_CONVERGENCE_THRESHOLD"),
        ("hwhm_lorentz", "NVISION_HWHM_LORENTZ_CONVERGENCE_THRESHOLD"),
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
