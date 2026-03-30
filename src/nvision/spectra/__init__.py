"""Concrete signal model implementations."""

from nvision.spectra.composite import CompositePeakModel
from nvision.spectra.exponential_decay import ExponentialDecayModel
from nvision.spectra.gaussian import GaussianModel
from nvision.spectra.lorentzian import LorentzianModel
from nvision.spectra.nv_center import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    MAX_K_NP,
    MAX_NV_CENTER_DELTA,
    MAX_NV_CENTER_OMEGA,
    MIN_K_NP,
    MIN_NV_CENTER_DELTA,
    MIN_NV_CENTER_OMEGA,
    NVCenterLorentzianModel,
    NVCenterVoigtModel,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
)
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.spectra.voigt_zeeman import VoigtZeemanModel

__all__ = [
    "A_PARAM",
    "MAX_K_NP",
    "MAX_NV_CENTER_DELTA",
    "MAX_NV_CENTER_OMEGA",
    "MIN_K_NP",
    "MIN_NV_CENTER_DELTA",
    "MIN_NV_CENTER_OMEGA",
    "CompositePeakModel",
    "ExponentialDecayModel",
    "GaussianModel",
    "LorentzianModel",
    "NVCenterLorentzianModel",
    "NVCenterVoigtModel",
    "UnitCubeSignalModel",
    "VoigtZeemanModel",
]
