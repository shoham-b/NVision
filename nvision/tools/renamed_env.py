"""Fail-fast guard for environment variables that were renamed in the center_freq / drive_freq disambiguation.

An old name that is still set (shell or ``.env``) would otherwise be silently ignored and the default
used instead, so a tuned value would vanish without any visible error. Every module that reads one of
the renamed variables calls :func:`reject_renamed_env_vars` before reading it.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# old name -> new name. Rename in ``.env`` / the shell; values and units are unchanged.
RENAMED_ENV_VARS: dict[str, str] = {
    "NVISION_FREQ_CONVERGENCE_THRESHOLD": "NVISION_CENTER_FREQ_CONVERGENCE_THRESHOLD",
    "NVISION_FREQ_CRLB_SAFETY_FACTOR": "NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR",
    "NVISION_NV_CENTER_FREQ_DELTA_HZ": "NVISION_NV_DRIVE_FREQ_DELTA_HZ",
    "NVISION_NV_PROBE_DELTA_HZ": "NVISION_NV_DRIVE_FREQ_DELTA_HZ",
    "NVISION_SWEEP_FIT_FREQ_STARTS": "NVISION_SWEEP_FIT_CENTER_FREQ_STARTS",
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
