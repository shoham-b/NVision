"""Tests for Voigt → EKF bound mapping."""

from __future__ import annotations

from nvision.sim.locs.ekf.parameter_bounds import (
    is_voigt_parameter_bounds,
    linewidth_hwhm_bounds,
    prepare_ekf_parameter_bounds,
)


def test_voigt_bounds_gain_derived_linewidth() -> None:
    voigt = {
        "frequency": (2.6e9, 3.1e9),
        "fwhm_total": (100e3, 200e3),
        "lorentz_frac": (0.2, 0.8),
        "split": (1.0e6, 2.0e6),
        "k_np": (2.0, 4.0),
        "dip_depth": (0.1, 1.0),
    }
    assert is_voigt_parameter_bounds(voigt)
    assert linewidth_hwhm_bounds(voigt) == (10e3, 80e3)

    prepared = prepare_ekf_parameter_bounds(voigt)
    assert prepared["linewidth"] == (10e3, 80e3)


def test_voigt_priors_map_to_linewidth() -> None:
    voigt = {
        "frequency": (2.6e9, 3.1e9),
        "fwhm_total": (100e3, 200e3),
        "lorentz_frac": (0.5, 0.1),
        "split": (2.0e6, 1.0e4),
        "k_np": (3.0, 0.1),
        "dip_depth": (0.5, 0.05),
        "_priors": {
            "fwhm_total": (120e3, 5e3),
            "lorentz_frac": (0.4, 0.05),
            "split": (2.1e6, 1.0e4),
            "k_np": (3.0, 0.1),
            "dip_depth": (0.5, 0.05),
        },
    }
    prepared = prepare_ekf_parameter_bounds(voigt)
    assert "linewidth" in prepared["_priors"]
    lw_m, lw_s = prepared["_priors"]["linewidth"]
    assert lw_m == 0.5 * 120e3 * 0.4
    assert lw_s > 0
