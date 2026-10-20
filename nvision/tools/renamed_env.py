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
}


def reject_renamed_env_vars() -> None:
    """Raise if any renamed environment variable is still set under its old name."""
    load_dotenv()
    stale = {old: new for old, new in RENAMED_ENV_VARS.items() if old in os.environ}
    if stale:
        listing = ", ".join(f"{old} -> {new}" for old, new in stale.items())
        raise RuntimeError(
            f"Renamed environment variable(s) still set under the old name: {listing}. "
            "Rename them in your .env / shell (values are unchanged); the old names are no longer read."
        )
