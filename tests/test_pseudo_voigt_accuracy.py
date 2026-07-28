"""Quantify the approximation error of the pseudo-Voigt kernel vs. a true Voigt profile.

``nv_center_pseudo_voigt_eval`` evaluates both the Lorentzian and Gaussian components at the
shared combined width (``fwhm_total``), matching the standard Thompson-Cox-Hastings formulation
that the ``eta`` mixing polynomial is calibrated against. (An earlier version evaluated each
component at its own split width instead — ``lorentz_frac * fwhm_total`` and
``(1 - lorentz_frac) * fwhm_total`` — which pinned the Lorentzian core to the raw homogeneous
linewidth regardless of ``fwhm_total`` and diverged from a true Voigt profile by up to ~68% of
peak height at extreme ``lorentz_frac``; it also meant a hyperfine triplet's sub-dips could never
merge no matter how large the inhomogeneous broadening got.) This is a regression guard recording
the current approximation's magnitude, not an assertion that it is textbook-accurate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import voigt_profile

from nvision.spectra.numba_kernels import nv_center_pseudo_voigt_eval
from nvision.spectra.nv_center import _voigt_reparam_scalar, nv_center_voigt_bounds_for_domain

_SQRT2LOG2 = math.sqrt(2.0 * math.log(2.0))


def _true_voigt_shape(dx: np.ndarray, fwhm_l: float, fwhm_g: float) -> np.ndarray:
    """True Voigt profile shape, normalized to peak height 1 at dx=0."""
    sigma = fwhm_g / (2.0 * _SQRT2LOG2) if fwhm_g > 0 else 1e-12
    gamma = fwhm_l / 2.0
    peak = voigt_profile(0.0, sigma, gamma)
    return voigt_profile(dx, sigma, gamma) / peak


def _pseudo_voigt_shape(dx: np.ndarray, fwhm_total: float, lorentz_frac: float) -> np.ndarray:
    """Two-width pseudo-Voigt shape (peak-normalized by construction)."""
    return np.array(
        [-nv_center_pseudo_voigt_eval(float(x), 0.0, fwhm_total, lorentz_frac, 0.0, 1.0, 1.0, 0.0) for x in dx]
    )


def test_pseudo_voigt_max_absolute_error_within_recorded_tolerance() -> None:
    """Max |approx - true| over peak-normalized shapes, swept across the Voigt prior range.

    Absolute (not relative) error is used because both shapes are normalized to peak height 1:
    relative error blows up in the far tails where the true profile decays toward zero, which is
    a metric artifact rather than a meaningful accuracy signal for the likelihood evaluation.
    """
    bounds = nv_center_voigt_bounds_for_domain(0.0, 1.0)
    hl_lo, hl_hi = bounds["homogeneous_linewidth"]
    si_lo, si_hi = bounds["sigma_inhom"]
    # Sweep the kernel-native fwhm_total across what the model's own
    # (homogeneous_linewidth, sigma_inhom) prior range can produce.
    fwhm_lo, _ = _voigt_reparam_scalar(hl_lo, si_lo)
    fwhm_hi, _ = _voigt_reparam_scalar(hl_hi, si_hi)

    fwhm_values = np.linspace(fwhm_lo, fwhm_hi, 4)
    lorentz_frac_values = np.array([0.05, 0.3, 0.5, 0.7, 0.98])

    max_abs_err = 0.0
    for fwhm_total in fwhm_values:
        for lorentz_frac in lorentz_frac_values:
            fwhm_l = lorentz_frac * fwhm_total
            fwhm_g = (1.0 - lorentz_frac) * fwhm_total
            dx = np.linspace(-3.0 * fwhm_total, 3.0 * fwhm_total, 401)

            true_shape = _true_voigt_shape(dx, fwhm_l, fwhm_g)
            approx_shape = _pseudo_voigt_shape(dx, fwhm_total, lorentz_frac)

            max_abs_err = max(max_abs_err, float(np.max(np.abs(approx_shape - true_shape))))

    # Recorded regression tolerance (measured max ~0.124 at fwhm_total near the prior's upper
    # bound and lorentz_frac=0.5) after re-deriving the mixing at the combined FWHM: this
    # documents the shared-width pseudo-Voigt's observed accuracy, not a design target.
    assert max_abs_err < 0.2, f"pseudo-Voigt max absolute error {max_abs_err:.3f} exceeded recorded tolerance"
