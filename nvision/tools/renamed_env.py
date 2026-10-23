"""Fail-fast guard for environment variables renamed in the zero_field_freq / drive_freq / hwhm renames.

An old name that is still set (shell or ``.env``) would otherwise be silently ignored and the default
used instead, so a tuned value would vanish without any visible error. Every module that reads one of
the renamed variables calls :func:`reject_renamed_env_vars` before reading it.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# old name -> new name. Rename in ``.env`` / the shell; values and units are unchanged.
RENAMED_ENV_VARS: dict[str, str] = {
    "NVISION_FREQ_CONVERGENCE_THRESHOLD": "NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD",
    "NVISION_FREQ_CRLB_SAFETY_FACTOR": "NVISION_ZERO_FIELD_FREQ_CRLB_SAFETY_FACTOR",
    "NVISION_NV_CENTER_FREQ_DELTA_HZ": "NVISION_NV_DRIVE_FREQ_DELTA_HZ",
    "NVISION_NV_PROBE_DELTA_HZ": "NVISION_NV_DRIVE_FREQ_DELTA_HZ",
    "NVISION_SWEEP_FIT_FREQ_STARTS": "NVISION_SWEEP_FIT_ZERO_FIELD_FREQ_STARTS",
    "NVISION_CENTER_FREQ_CONVERGENCE_THRESHOLD": "NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD",
    "NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR": "NVISION_ZERO_FIELD_FREQ_CRLB_SAFETY_FACTOR",
    "NVISION_SWEEP_FIT_CENTER_FREQ_STARTS": "NVISION_SWEEP_FIT_ZERO_FIELD_FREQ_STARTS",
    "NVISION_LINEWIDTH_CONVERGENCE_THRESHOLD": "NVISION_HWHM_CONVERGENCE_THRESHOLD",
    "NVISION_HOMOGENEOUS_LINEWIDTH_CONVERGENCE_THRESHOLD": "NVISION_HWHM_LORENTZ_CONVERGENCE_THRESHOLD",
    "NVISION_ZEEMAN_SPLIT_CONVERGENCE_THRESHOLD": "NVISION_ZEEMAN_SHIFT_CONVERGENCE_THRESHOLD",
    "NVISION_C_TOTAL_CONVERGENCE_THRESHOLD": "NVISION_CONTRAST_TOTAL_CONVERGENCE_THRESHOLD",
    "NVISION_DIP_DEPTH_CONVERGENCE_THRESHOLD": "NVISION_CONTRAST_CONVERGENCE_THRESHOLD",
    "NVISION_MIN_LINEWIDTH": "NVISION_MIN_HWHM",
    "NVISION_MAX_LINEWIDTH": "NVISION_MAX_HWHM",
    "NVISION_MIN_SPLIT": "NVISION_MIN_HYPERFINE_SHIFT",
    "NVISION_MAX_SPLIT": "NVISION_MAX_HYPERFINE_SHIFT",
    "NVISION_NV_GRID_LINEWIDTH": "NVISION_NV_GRID_HWHM",
    "NVISION_SIGNAL_LINEWIDTH": "NVISION_SIGNAL_HWHM",
    "NVISION_SWEEP_MAX_MEASUREMENTS": "NVISION_MAX_MEASUREMENTS",
}

# Removed variables, with what replaced them; still set, they would be silently ignored.
REMOVED_ENV_VARS: dict[str, str] = {
    "NVISION_SBED_STEPS_FRACTION": "every locator now shares the absolute budget NVISION_MAX_MEASUREMENTS",
    "NVISION_SBED_NOISE_MIN": "the study noise axis is SNR levels: NVISION_SBED_SNR_VALUES (sigma = contrast / SNR)",
    "NVISION_SBED_NOISE_MAX": "the study noise axis is SNR levels: NVISION_SBED_SNR_VALUES (sigma = contrast / SNR)",
    "NVISION_SBED_NOISE_STEPS": "the study noise axis is SNR levels: NVISION_SBED_SNR_VALUES (sigma = contrast / SNR)",
    "NVISION_DRIFT_GAUSS_SIGMA": "the drift study noise is posed as an SNR: NVISION_DRIFT_SNR (sigma = contrast / SNR)",
    "NVISION_MEASUREMENTS_PER_FWHM": "the budget is the same for every window: NVISION_MAX_MEASUREMENTS",
}


def reject_renamed_env_vars() -> None:
    """Raise if any renamed environment variable is still set under its old name."""
    load_dotenv()
    removed = {name: why for name, why in REMOVED_ENV_VARS.items() if name in os.environ}
    if removed:
        listing = "; ".join(f"{name} ({why})" for name, why in removed.items())
        raise RuntimeError(f"Removed environment variable(s) still set: {listing}. Remove them from your .env / shell.")
    stale = {old: new for old, new in RENAMED_ENV_VARS.items() if old in os.environ}
    if stale:
        listing = ", ".join(f"{old} -> {new}" for old, new in stale.items())
        raise RuntimeError(
            f"Renamed environment variable(s) still set under the old name: {listing}. "
            "Rename them in your .env / shell (values are unchanged); the old names are no longer read."
        )
