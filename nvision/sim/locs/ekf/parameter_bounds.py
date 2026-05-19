"""Map experiment parameter bounds to the EKF triple-Lorentzian state."""

from __future__ import annotations

import math
from typing import Any


def is_voigt_parameter_bounds(bounds: dict[str, Any]) -> bool:
    """True when bounds use Voigt linewidth parameterization."""
    return "fwhm_total" in bounds or "lorentz_frac" in bounds


def linewidth_hwhm_bounds(bounds: dict[str, Any]) -> tuple[float, float]:
    """Lorentzian HWHM bounds for EKF ``Omega`` from experiment bounds."""
    if "linewidth" in bounds:
        lo, hi = bounds["linewidth"]
        return float(lo), float(hi)
    if "fwhm_total" in bounds:
        fwhm_lo, fwhm_hi = bounds["fwhm_total"]
        frac_lo, frac_hi = bounds.get("lorentz_frac", (0.05, 0.98))
        return (
            0.5 * float(fwhm_lo) * float(frac_lo),
            0.5 * float(fwhm_hi) * float(frac_hi),
        )
    return (50e3, 400e3)


def _linewidth_prior_from_voigt(
    fwhm_prior: tuple[float, float],
    lorentz_frac_prior: tuple[float, float] | None,
) -> tuple[float, float]:
    fwhm_m, fwhm_s = fwhm_prior
    if lorentz_frac_prior is not None:
        frac_m, frac_s = lorentz_frac_prior
    else:
        frac_m, frac_s = 0.5, 0.0
    lw_m = 0.5 * fwhm_m * frac_m
    lw_s = math.hypot(0.5 * fwhm_s * frac_m, 0.5 * fwhm_m * frac_s)
    return (lw_m, max(lw_s, 1.0))


def prepare_ekf_parameter_bounds(
    parameter_bounds: dict[str, Any] | None,
    *,
    x_min: float = 2.6e9,
    x_max: float = 3.1e9,
) -> dict[str, Any]:
    """Normalize Lorentzian/Voigt experiment bounds for EKF locators.

    Voigt experiments keep their native keys; a derived ``linewidth`` (Lorentzian
    HWHM) is added so the triple-Lorentzian EKF state can be initialized.
    """
    from nvision.spectra.nv_center import nv_center_lorentzian_bounds_for_domain

    if parameter_bounds is None:
        return dict(nv_center_lorentzian_bounds_for_domain(x_min, x_max))

    bounds: dict[str, Any] = dict(parameter_bounds)
    if not is_voigt_parameter_bounds(bounds):
        return bounds

    bounds["linewidth"] = linewidth_hwhm_bounds(bounds)
    priors = bounds.get("_priors")
    if isinstance(priors, dict) and "linewidth" not in priors and "fwhm_total" in priors:
        updated = dict(priors)
        updated["linewidth"] = _linewidth_prior_from_voigt(
            priors["fwhm_total"],
            priors.get("lorentz_frac"),
        )
        bounds["_priors"] = updated
    return bounds
