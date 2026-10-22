"""Metrics updates for the parameter-grid execution approach.

Covers: derived-quantity helpers (Jacobian propagation), derived convergence gating,
parse_gauss_sigma and milestone shift_param selection.
"""

from __future__ import annotations

import math

import pytest

from nvision.sim.combinations import parse_gauss_sigma
from nvision.sim.locs.bayesian.sequential_bayesian_locator import (
    saturation_voigt_derived_bounds,
    saturation_voigt_derived_sigmas,
)
from nvision.spectra.nv_center import (
    NV_NATURAL_HWHM_HZ,
    _saturation_voigt_reparam_scalar,
    saturation_voigt_effective_hwhm_and_unc,
    saturation_voigt_realized_contrast_and_unc,
)

# ---------------------------------------------------------------------------
# Derived-quantity helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("s", "si"), [(0.02, 0.0), (1.0, 1e5), (5.0, 3e5), (30.0, 1.2e6)])
def test_effective_hwhm_matches_reparam(s, si):
    fwhm_total, _, _ = _saturation_voigt_reparam_scalar(s, si, 1.0)
    omega, _ = saturation_voigt_effective_hwhm_and_unc(s, si)
    assert omega == pytest.approx(fwhm_total / 2.0, rel=1e-9)


def test_effective_hwhm_unc_matches_finite_difference():
    s, si, sigma_s, sigma_si = 5.0, 3e5, 0.4, 2e4
    eps = 1e-6
    omega0, sigma_omega = saturation_voigt_effective_hwhm_and_unc(s, si, sigma_s, sigma_si)
    d_ds = (saturation_voigt_effective_hwhm_and_unc(s + eps, si)[0] - omega0) / eps
    d_dsi = (saturation_voigt_effective_hwhm_and_unc(s, si + eps)[0] - omega0) / eps
    expected = math.sqrt((d_ds * sigma_s) ** 2 + (d_dsi * sigma_si) ** 2)
    assert sigma_omega == pytest.approx(expected, rel=1e-4)


def test_effective_hwhm_lorentzian_limit():
    """sigma_inhom -> 0: omega collapses to the homogeneous HWHM gamma0*sqrt(1+s)."""
    omega, _ = saturation_voigt_effective_hwhm_and_unc(3.0, 0.0)
    assert omega == pytest.approx(NV_NATURAL_HWHM_HZ * math.sqrt(4.0), rel=1e-9)


def test_realized_contrast_law_and_unc():
    s, c_max, sigma_s, sigma_cm = 5.0, 0.1, 0.4, 0.01
    c, sigma_c = saturation_voigt_realized_contrast_and_unc(s, c_max, sigma_s, sigma_cm)
    assert c == pytest.approx(c_max * s / (1 + s), rel=1e-12)
    eps = 1e-7
    dc_ds = (saturation_voigt_realized_contrast_and_unc(s + eps, c_max)[0] - c) / eps
    dc_dcm = (saturation_voigt_realized_contrast_and_unc(s, c_max + eps)[0] - c) / eps
    expected = math.sqrt((dc_ds * sigma_s) ** 2 + (dc_dcm * sigma_cm) ** 2)
    assert sigma_c == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Derived convergence gating
# ---------------------------------------------------------------------------

_EST = {"zero_field_freq": 2.8e9, "saturation": 5.0, "sigma_inhom": 3e5, "c_max": 0.1, "zeeman_shift": 1e7}
_BOUNDS = {
    "zero_field_freq": (2.6e9, 3.1e9),
    "saturation": (0.02, 30.0),
    "sigma_inhom": (0.0, 1.2e6),
    "c_max": (0.1, 0.4),
    "zeeman_shift": (0.0, 1e8),
}


def test_derived_sigmas_present_for_saturation_voigt():
    sigmas = {"saturation": 0.4, "sigma_inhom": 2e4, "c_max": 0.01}
    derived = saturation_voigt_derived_sigmas(_EST, sigmas)
    assert derived is not None
    assert set(derived) == {"hwhm", "contrast_total"}
    assert derived["hwhm"] > 0
    assert derived["contrast_total"] > 0


def test_derived_sigmas_none_for_lorentzian_params():
    assert saturation_voigt_derived_sigmas({"zero_field_freq": 1.0, "hwhm": 1e6}, {"hwhm": 1e4}) is None
    assert saturation_voigt_derived_sigmas(_EST, {"hwhm": 1e4}) is None  # sigmas lack raw params


def test_derived_sigmas_propagate_inf_crlbs():
    sigmas = {"saturation": math.inf, "sigma_inhom": 2e4, "c_max": 0.01}
    derived = saturation_voigt_derived_sigmas(_EST, sigmas)
    assert derived is not None
    assert math.isinf(derived["hwhm"])


def test_derived_bounds_monotone_ranges():
    from nvision.spectra.nv_center import NV_SATURATION_C_MAX

    db = saturation_voigt_derived_bounds(_BOUNDS)
    lw_lo, lw_hi = db["hwhm"]
    c_lo, c_hi = db["contrast_total"]
    assert 0 < lw_lo < lw_hi
    # c_max is fixed (not read from _BOUNDS), so realized contrast only ever
    # approaches NV_SATURATION_C_MAX asymptotically as saturation -> infinity.
    assert 0 < c_lo < c_hi < NV_SATURATION_C_MAX
    # Effective HWHM range endpoints come from the bound corners
    assert lw_lo == pytest.approx(saturation_voigt_effective_hwhm_and_unc(0.02, 0.0)[0])
    assert lw_hi == pytest.approx(saturation_voigt_effective_hwhm_and_unc(30.0, 1.2e6)[0])


class _StubModel:
    def parameter_names(self):
        return list(_EST.keys())


class _StubBelief:
    """Minimal belief exposing what _target_params_converged touches."""

    model = _StubModel()
    physical_param_bounds = _BOUNDS

    def __init__(self, uncertainties):
        self._unc = uncertainties

    def uncertainty(self):
        return self._unc

    def estimates(self):
        return dict(_EST)


def _make_locator(uncertainties):
    from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

    loc = SequentialBayesianLocator.__new__(SequentialBayesianLocator)
    loc.belief = _StubBelief(uncertainties)
    loc.convergence_threshold = 0.01
    return loc


def test_target_params_converged_uses_derived_gating():
    # Tight zero_field_freq + tight derived quantities -> converged, even though the RAW
    # saturation uncertainty (0.25) is large in absolute s-units.
    unc = {
        "zero_field_freq": 1e4,  # << 100 kHz threshold
        "saturation": 0.25,  # raw 1%-of-bound gate would need < 0.2998 — borderline
        "sigma_inhom": 1e3,
        "c_max": 1e-4,
        "zeeman_shift": 5e4,  # << 100 kHz absolute threshold (NVISION_ZEEMAN_SHIFT_CONVERGENCE_THRESHOLD)
    }
    assert _make_locator(unc)._target_params_converged(unc) is True


def test_target_params_converged_fails_on_wide_derived_width():
    # Huge sigma_inhom uncertainty -> derived effective-HWHM unc dominates -> not converged.
    unc = {
        "zero_field_freq": 1e4,
        "saturation": 0.25,
        "sigma_inhom": 5e5,  # sqrt(2ln2)*5e5 ~ 590 kHz >> 1% of the ~2.1 MHz effective range
        "c_max": 1e-4,
        "zeeman_shift": 1e5,
    }
    assert _make_locator(unc)._target_params_converged(unc) is False


def test_target_params_converged_lorentzian_path_unchanged():
    class _LorModel:
        def parameter_names(self):
            return ["zero_field_freq", "hwhm", "contrast_total"]

    from typing import ClassVar

    class _LorBelief:
        model = _LorModel()
        physical_param_bounds: ClassVar[dict[str, tuple[float, float]]] = {
            "zero_field_freq": (2.6e9, 3.1e9),
            "hwhm": (2e5, 5e6),
            "contrast_total": (0.1, 0.4),
        }

        def __init__(self, unc):
            self._unc = unc

        def uncertainty(self):
            return self._unc

        def estimates(self):  # must never be needed on this path
            raise AssertionError("estimates() should not be called for Lorentzian gating")

    from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

    loc = SequentialBayesianLocator.__new__(SequentialBayesianLocator)
    unc = {"zero_field_freq": 1e4, "hwhm": 1e4, "contrast_total": 1e-4}
    loc.belief = _LorBelief(unc)
    loc.convergence_threshold = 0.01
    assert loc._target_params_converged(unc) is True


# ---------------------------------------------------------------------------
# parse_gauss_sigma
# ---------------------------------------------------------------------------


def test_parse_gauss_sigma_roundtrips_preset_names():
    from nvision.sim import presets

    for name, _ in presets.sbed_study_noises():
        sigma = parse_gauss_sigma(name)
        assert sigma is not None
        assert name == f"Gauss({sigma})"


def test_parse_gauss_sigma_non_gauss():
    assert parse_gauss_sigma("NoNoise") is None
    assert parse_gauss_sigma("Poisson(5000)") is None


# ---------------------------------------------------------------------------
# Milestone shift_param selection
# ---------------------------------------------------------------------------


def test_default_shift_param_prefers_zeeman_shift():
    from nvision.metrics.milestones import default_shift_param

    class _Sig:
        def __init__(self, params):
            self._p = params

        def parameter_values(self):
            return self._p

    class _RR:
        def __init__(self, params):
            self.true_signal = _Sig(params)

    assert default_shift_param(_RR({"zero_field_freq": 1.0, "zeeman_shift": 2.0})) == "zeeman_shift"
    assert default_shift_param(_RR({"zero_field_freq": 1.0, "hyperfine_shift": 2.0})) == "hyperfine_shift"
    assert default_shift_param(_RR({"zero_field_freq": 1.0})) == "hyperfine_shift"
