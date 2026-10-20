"""NV center signal generator."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from nvision.spectra.nv_center import (
    DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
    DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
    MAX_K_NP,
    MAX_LINEWIDTH,
    MAX_SPLIT,
    MAX_ZEEMAN_SPLIT,
    MIN_K_NP,
    MIN_LINEWIDTH,
    MIN_SPLIT,
    MIN_ZEEMAN_SPLIT,
    NV_ZERO_FIELD_SPLITTING_HZ,
    PRIOR_MEAN_OFFSET_SIGMAS,
    PRIOR_STD_FRACTION,
    NVCenterLorentzianModel,
    NVCenterLorentzianSingleDipSpectrum,
    NVCenterLorentzianSpectrum,
    NVCenterLorentzianZeemanSpectrum,
    NVCenterSaturationVoigtModel,
    NVCenterSaturationVoigtSingleDipSpectrum,
    NVCenterSaturationVoigtSpectrum,
    NVCenterSaturationVoigtZeemanHyperfineSpectrum,
    NVCenterSaturationVoigtZeemanSpectrum,
    NVCenterVoigtModel,
    NVCenterVoigtSingleDipSpectrum,
    NVCenterVoigtSpectrum,
    NVCenterVoigtZeemanSpectrum,
    hyperfine_geometry,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_saturation_voigt_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
)

from .peak_spec import _true_signal_from_typed

# Gaussian FWHM = 2*sqrt(2 ln2)*sigma; used to convert a physical sigma_inhom
# (Hz) into the voigt variant's lorentz_frac ratio.
_VOIGT_SQRT2LOG2 = math.sqrt(2.0 * math.log(2.0))


# Resonance-search scenario: the previously classified nuisance parameters keep a prior whose std is this
# fraction of its usual width (the spectrum was characterised, the field only moved).
RESONANCE_SEARCH_NUISANCE_STD_SCALE: float = 0.25


def _widened_prior_mean(rng: random.Random, true_value: float, std: float) -> float:
    """Draw a parameter's prior mean, deliberately offset from ``true_value``.

    See ``PRIOR_MEAN_OFFSET_SIGMAS``'s docstring (nvision/spectra/nv_center.py):
    widening this draw beyond the prior's own std (rather than reusing it) keeps
    the SBED locator's starting belief from being suspiciously close to the
    answer before a single measurement.
    """
    return rng.gauss(true_value, std * PRIOR_MEAN_OFFSET_SIGMAS)


@dataclass
class NVCenterCoreGenerator:
    """Generates NV center ODMR signals using core architecture.

    Produces TrueSignal with physically accurate NV center signal.

    ``hyperfine`` picks how much nitrogen hyperfine structure the simulated
    measurement resolves -- ``"unresolved"`` (the default: the lines merge into a
    single dip per Zeeman group, which is what a typical NV linewidth actually
    gives you), ``"n14"`` (¹⁴N, I=1, a resolved triplet) or ``"n15"`` (¹⁵N,
    I=1/2, a resolved doublet). ``infer_hyperfine=True`` additionally randomizes
    ``split``/``k_np`` per repeat and exposes them as free parameters instead of
    pinning them to the isotope's physical coupling.

    ``center_freq`` (the zero-field center) is fixed at the midpoint of the safe
    range for every draw -- like a known, calibrated instrument constant -- so
    only zeeman_split/hyperfine/linewidth/contrast vary between repeats.
    """

    drive_freq_min_phys: float = DEFAULT_NV_DRIVE_FREQ_MIN_PHYS  # 2.6 GHz
    drive_freq_max_phys: float = DEFAULT_NV_DRIVE_FREQ_MAX_PHYS  # 3.1 GHz
    variant: str = "lorentzian"  # "lorentzian", "voigt", "saturation_voigt", "cw_drive" or "pulsed_drive"
    hyperfine: str = "unresolved"  # resolved hyperfine structure: "unresolved" (default) / "n14" / "n15"
    infer_hyperfine: bool = False  # randomize + infer split/k_np instead of using the isotope constant
    with_zeeman_splitting: bool = True  # default: Zeeman-split two-dip model
    linewidth: float | None = None  # if set, fix linewidth (HWHM, Hz) instead of randomizing (lorentzian, voigt)
    c_total: float | None = None  # if set, fix contrast instead of randomizing (lorentzian, voigt)
    # if set, fix Lorentzian share of broadening instead of randomizing (voigt only);
    # 1.0 = pure Lorentzian (no inhomogeneous/Gaussian broadening)
    lorentz_frac: float | None = None
    saturation: float | None = None  # if set, fix drive saturation (homogeneous channel; saturation_voigt only)
    # if set, fix inhomogeneous width, Hz (saturation_voigt; also voigt, converted to
    # lorentz_frac via linewidth -- ignored if lorentz_frac is also set)
    sigma_inhom: float | None = None
    # Resonance search after a field change: every parameter except ``zeeman_split`` (the resonance
    # position) was classified before, so its prior is tight (std scaled by RESONANCE_SEARCH_NUISANCE_STD_SCALE,
    # mean offset from the truth by the usual PRIOR_MEAN_OFFSET_SIGMAS of that smaller std), while
    # ``zeeman_split`` has no prior at all (flat over its bounds): the new resonance could be anywhere.
    resonance_search: bool = False
    # c_max (saturated contrast scale) is not a per-generator field: it's a fixed
    # physical constant (NV_SATURATION_C_MAX in nv_center.py), not something that
    # varies between repeats or studies.

    def generate(self, rng: random.Random):
        """Generate NV center ODMR signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            NV center signal with realistic parameters
        """
        # Random linewidth (HWHM for Lorentzian), or fixed if provided
        linewidth = self.linewidth if self.linewidth is not None else rng.uniform(MIN_LINEWIDTH, MAX_LINEWIDTH)

        zeeman_split = 0.0
        # Hyperfine line offset: drawn per repeat only when it's a free parameter,
        # otherwise the isotope's own physical coupling -- exactly 0 when the
        # structure is unresolved, which collapses the multiplet to one dip.
        split = hyperfine_geometry(self.hyperfine)[0]
        if self.with_zeeman_splitting:
            zeeman_split = rng.uniform(MIN_ZEEMAN_SPLIT, MAX_ZEEMAN_SPLIT)
        if self.infer_hyperfine:
            split = rng.uniform(MIN_SPLIT, MAX_SPLIT)

        # The zero-field center is a fixed, known reference (like a calibrated instrument
        # constant) -- only zeeman_split/hyperfine/linewidth/contrast vary between draws.
        # The drive-frequency window is the upper half [D, D + delta] of the mirror-symmetric spectrum
        # (see DEFAULT_NV_DRIVE_FREQ_MIN_PHYS), so the center is D itself, on its lower edge.
        center_freq = NV_ZERO_FIELD_SPLITTING_HZ

        if self.variant == "lorentzian":
            c_total = self.c_total if self.c_total is not None else rng.uniform(0.1, 0.4)
            linewidth_std = (MAX_LINEWIDTH - MIN_LINEWIDTH) * PRIOR_STD_FRACTION
            c_total_std = 0.3 * PRIOR_STD_FRACTION  # c_total range is roughly [0.1, 0.4]

            if self.with_zeeman_splitting and self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterLorentzianModel(
                    with_zeeman_splitting=True, hyperfine=self.hyperfine, infer_hyperfine=True
                )
                from nvision.spectra.nv_center import NVCenterLorentzianZeemanHyperfineSpectrum

                typed_params = NVCenterLorentzianZeemanHyperfineSpectrum(
                    center_freq=center_freq,
                    linewidth=linewidth,
                    zeeman_split=zeeman_split,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    infer_hyperfine=True,
                    with_zeeman_splitting=True,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (_widened_prior_mean(rng, zeeman_split, zeeman_std), zeeman_std),
                    "split": (_widened_prior_mean(rng, split, split_std), split_std),
                    "linewidth": (_widened_prior_mean(rng, linewidth, linewidth_std), linewidth_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.with_zeeman_splitting:
                model = NVCenterLorentzianModel(with_zeeman_splitting=True, hyperfine=self.hyperfine)
                typed_params = NVCenterLorentzianZeemanSpectrum(
                    center_freq=center_freq,
                    linewidth=linewidth,
                    zeeman_split=zeeman_split,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    with_zeeman_splitting=True,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (_widened_prior_mean(rng, zeeman_split, zeeman_std), zeeman_std),
                    "linewidth": (_widened_prior_mean(rng, linewidth, linewidth_std), linewidth_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterLorentzianModel(hyperfine=self.hyperfine, infer_hyperfine=True)
                typed_params = NVCenterLorentzianSpectrum(
                    center_freq=center_freq,
                    linewidth=linewidth,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys, self.drive_freq_max_phys, hyperfine=self.hyperfine, infer_hyperfine=True
                )
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "split": (_widened_prior_mean(rng, split, split_std), split_std),
                    "linewidth": (_widened_prior_mean(rng, linewidth, linewidth_std), linewidth_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            else:
                model = NVCenterLorentzianModel(hyperfine=self.hyperfine)
                typed_params = NVCenterLorentzianSingleDipSpectrum(
                    center_freq=center_freq,
                    linewidth=linewidth,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys, self.drive_freq_max_phys, hyperfine=self.hyperfine
                )
                bounds["_priors"] = {
                    "linewidth": (_widened_prior_mean(rng, linewidth, linewidth_std), linewidth_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
        elif self.variant in ("cw_drive", "pulsed_drive"):
            return self._generate_drive_controlled(rng, center_freq, zeeman_split)
        elif self.variant == "saturation_voigt":
            sv_bounds = nv_center_saturation_voigt_bounds_for_domain(
                self.drive_freq_min_phys,
                self.drive_freq_max_phys,
                hyperfine=self.hyperfine,
                infer_hyperfine=self.infer_hyperfine,
                with_zeeman_splitting=self.with_zeeman_splitting,
            )
            saturation_bounds = sv_bounds["saturation"]
            sigma_inhom_bounds = sv_bounds["sigma_inhom"]

            saturation = self.saturation if self.saturation is not None else rng.uniform(*saturation_bounds)
            sigma_inhom = self.sigma_inhom if self.sigma_inhom is not None else rng.uniform(*sigma_inhom_bounds)

            saturation_std = (saturation_bounds[1] - saturation_bounds[0]) * PRIOR_STD_FRACTION
            sigma_inhom_std = (sigma_inhom_bounds[1] - sigma_inhom_bounds[0]) * PRIOR_STD_FRACTION

            model = NVCenterSaturationVoigtModel(
                hyperfine=self.hyperfine,
                infer_hyperfine=self.infer_hyperfine,
                with_zeeman_splitting=self.with_zeeman_splitting,
            )
            priors = {
                "saturation": (_widened_prior_mean(rng, saturation, saturation_std), saturation_std),
                "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
            }

            if self.with_zeeman_splitting and self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                typed_params = NVCenterSaturationVoigtZeemanHyperfineSpectrum(
                    center_freq=center_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                    split=split,
                    k_np=k_np,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                priors["zeeman_split"] = (_widened_prior_mean(rng, zeeman_split, zeeman_std), zeeman_std)
                priors["split"] = (_widened_prior_mean(rng, split, split_std), split_std)
                priors["k_np"] = (_widened_prior_mean(rng, k_np, k_np_std), k_np_std)
            elif self.with_zeeman_splitting:
                typed_params = NVCenterSaturationVoigtZeemanSpectrum(
                    center_freq=center_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                priors["zeeman_split"] = (_widened_prior_mean(rng, zeeman_split, zeeman_std), zeeman_std)
            elif self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                typed_params = NVCenterSaturationVoigtSpectrum(
                    center_freq=center_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    split=split,
                    k_np=k_np,
                )
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                priors["split"] = (_widened_prior_mean(rng, split, split_std), split_std)
                priors["k_np"] = (_widened_prior_mean(rng, k_np, k_np_std), k_np_std)
            else:
                typed_params = NVCenterSaturationVoigtSingleDipSpectrum(
                    center_freq=center_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                )

            bounds = sv_bounds
            bounds["_priors"] = priors
        else:  # voigt
            homogeneous_linewidth = linewidth
            if self.sigma_inhom is not None:
                sigma_inhom = self.sigma_inhom
            elif self.lorentz_frac is not None:
                # Backward-compat: convert a pinned lorentz_frac into the equivalent
                # sigma_inhom (inverse of _voigt_reparam_scalar's fwhm_l/(fwhm_l+fwhm_g)
                # -> lorentz_frac, evaluated at this draw's homogeneous_linewidth).
                lorentz_ratio = 1.0 / self.lorentz_frac - 1.0 if self.lorentz_frac > 0 else 0.0
                sigma_inhom = homogeneous_linewidth * lorentz_ratio / _VOIGT_SQRT2LOG2
            else:
                lorentz_ratio = rng.uniform(0.1, 0.3)  # fwhm_gauss / fwhm_lorentz
                sigma_inhom = homogeneous_linewidth * lorentz_ratio / _VOIGT_SQRT2LOG2

            c_total = self.c_total if self.c_total is not None else rng.uniform(0.1, 0.4)
            linewidth_std = (MAX_LINEWIDTH - MIN_LINEWIDTH) * PRIOR_STD_FRACTION
            # sigma_inhom's own bound range (mirrors nv_center_voigt_bounds_for_domain's
            # sigma_inhom_hi, which doesn't depend on the hyperfine/zeeman flags) — not
            # linewidth's range, which this previously (incorrectly) copied.
            voigt_sigma_inhom_hi = max(1.2e6, (self.drive_freq_max_phys - self.drive_freq_min_phys) * 0.02)
            sigma_inhom_std = voigt_sigma_inhom_hi * PRIOR_STD_FRACTION
            c_total_std = 0.3 * PRIOR_STD_FRACTION  # c_total range is roughly [0.1, 0.4]

            if self.with_zeeman_splitting and self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterVoigtModel(with_zeeman_splitting=True, hyperfine=self.hyperfine, infer_hyperfine=True)
                from nvision.spectra.nv_center import NVCenterVoigtZeemanHyperfineSpectrum

                typed_params = NVCenterVoigtZeemanHyperfineSpectrum(
                    center_freq=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    infer_hyperfine=True,
                    with_zeeman_splitting=True,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (_widened_prior_mean(rng, zeeman_split, zeeman_std), zeeman_std),
                    "split": (_widened_prior_mean(rng, split, split_std), split_std),
                    "homogeneous_linewidth": (
                        _widened_prior_mean(rng, homogeneous_linewidth, linewidth_std),
                        linewidth_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.with_zeeman_splitting:
                model = NVCenterVoigtModel(with_zeeman_splitting=True, hyperfine=self.hyperfine)
                typed_params = NVCenterVoigtZeemanSpectrum(
                    center_freq=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    with_zeeman_splitting=True,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (_widened_prior_mean(rng, zeeman_split, zeeman_std), zeeman_std),
                    "homogeneous_linewidth": (
                        _widened_prior_mean(rng, homogeneous_linewidth, linewidth_std),
                        linewidth_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterVoigtModel(hyperfine=self.hyperfine, infer_hyperfine=True)
                typed_params = NVCenterVoigtSpectrum(
                    center_freq=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys, self.drive_freq_max_phys, hyperfine=self.hyperfine, infer_hyperfine=True
                )
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "split": (_widened_prior_mean(rng, split, split_std), split_std),
                    "homogeneous_linewidth": (
                        _widened_prior_mean(rng, homogeneous_linewidth, linewidth_std),
                        linewidth_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            else:
                model = NVCenterVoigtModel(hyperfine=self.hyperfine)
                typed_params = NVCenterVoigtSingleDipSpectrum(
                    center_freq=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys, self.drive_freq_max_phys, hyperfine=self.hyperfine
                )
                bounds["_priors"] = {
                    "homogeneous_linewidth": (
                        _widened_prior_mean(rng, homogeneous_linewidth, linewidth_std),
                        linewidth_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "c_total": (_widened_prior_mean(rng, c_total, c_total_std), c_total_std),
                    "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }

        if self.resonance_search:
            bounds["_priors"] = self._resonance_search_priors(rng, typed_params, bounds["_priors"])
        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)

    def _generate_drive_controlled(self, rng: random.Random, center_freq: float, zeeman_split: float):
        """Drive-controlled CW (``variant="cw_drive"``) or pulsed (``"pulsed_drive"``) NV signal.

        The microwave drive is a per-measurement setting (see nvision/spectra/nv_center_drive.py); the
        repeat draws the sample's *response* to it (``drive_amp_sat`` for CW, ``rabi_per_amp`` and
        ``c_total`` for pulsed) plus ``sigma_inhom`` and ``zeeman_split``. The baseline drive -- used by
        locators that do not choose the drive -- is ``BASELINE_DRIVE_AMP_UNIT``; for pulsed ODMR its
        pulse is the π-pulse calibrated from the repeat's *prior mean* Rabi frequency (an experimenter's
        rough calibration, not the true value), so fixed-drive baselines get no oracle calibration.
        """
        from nvision.spectra.drive import DriveSetting
        from nvision.spectra.nv_center_drive import (
            BASELINE_DRIVE_AMP_UNIT,
            NVCenterCWDriveModel,
            NVCenterCWDriveSpectrum,
            NVCenterCWDriveZeemanSpectrum,
            NVCenterPulsedDriveModel,
            NVCenterPulsedSpectrum,
            NVCenterPulsedZeemanSpectrum,
            nv_center_drive_bounds_for_domain,
            pi_pulse_duration_s,
        )

        if self.infer_hyperfine:
            raise ValueError(f"variant {self.variant!r} does not support infer_hyperfine=True.")
        mode = "cw" if self.variant == "cw_drive" else "pulsed"
        bounds = nv_center_drive_bounds_for_domain(
            mode,
            self.drive_freq_min_phys,
            self.drive_freq_max_phys,
            hyperfine=self.hyperfine,
            with_zeeman_splitting=self.with_zeeman_splitting,
        )

        def _std(name: str) -> float:
            lo, hi = bounds[name]
            return (hi - lo) * PRIOR_STD_FRACTION

        sigma_inhom = self.sigma_inhom if self.sigma_inhom is not None else rng.uniform(*bounds["sigma_inhom"])
        priors: dict[str, tuple] = {
            "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, _std("sigma_inhom")), _std("sigma_inhom")),
            "center_freq": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
        }
        if self.with_zeeman_splitting:
            priors["zeeman_split"] = (
                _widened_prior_mean(rng, zeeman_split, _std("zeeman_split")),
                _std("zeeman_split"),
            )

        if mode == "cw":
            # Log-uniform: the response scale spans a decade.
            a_lo, a_hi = bounds["drive_amp_sat"]
            drive_amp_sat = math.exp(rng.uniform(math.log(a_lo), math.log(a_hi)))
            priors["drive_amp_sat"] = (
                _widened_prior_mean(rng, drive_amp_sat, _std("drive_amp_sat")),
                _std("drive_amp_sat"),
            )
            model = NVCenterCWDriveModel(hyperfine=self.hyperfine, with_zeeman_splitting=self.with_zeeman_splitting)
            common = {"center_freq": center_freq, "drive_amp_sat": drive_amp_sat, "sigma_inhom": sigma_inhom}
            typed_params = (
                NVCenterCWDriveZeemanSpectrum(**common, zeeman_split=zeeman_split)
                if self.with_zeeman_splitting
                else NVCenterCWDriveSpectrum(**common)
            )
            baseline_drive = DriveSetting(BASELINE_DRIVE_AMP_UNIT)
        else:
            rabi_per_amp = rng.uniform(*bounds["rabi_per_amp"])
            c_total = self.c_total if self.c_total is not None else rng.uniform(*bounds["c_total"])
            priors["rabi_per_amp"] = (
                _widened_prior_mean(rng, rabi_per_amp, _std("rabi_per_amp")),
                _std("rabi_per_amp"),
            )
            priors["c_total"] = (_widened_prior_mean(rng, c_total, _std("c_total")), _std("c_total"))
            model = NVCenterPulsedDriveModel(hyperfine=self.hyperfine, with_zeeman_splitting=self.with_zeeman_splitting)
            common = {
                "center_freq": center_freq,
                "rabi_per_amp": rabi_per_amp,
                "c_total": c_total,
                "sigma_inhom": sigma_inhom,
            }
            typed_params = (
                NVCenterPulsedZeemanSpectrum(**common, zeeman_split=zeeman_split)
                if self.with_zeeman_splitting
                else NVCenterPulsedSpectrum(**common)
            )
            r_lo, r_hi = bounds["rabi_per_amp"]
            calibrated_rabi = min(max(priors["rabi_per_amp"][0], r_lo), r_hi) * BASELINE_DRIVE_AMP_UNIT
            baseline_drive = DriveSetting(BASELINE_DRIVE_AMP_UNIT, pi_pulse_duration_s(calibrated_rabi))

        bounds["_priors"] = priors
        return _true_signal_from_typed(
            model=model, typed_params=typed_params, bounds=bounds, baseline_drive=baseline_drive
        )

    @staticmethod
    def _resonance_search_priors(rng: random.Random, typed_params, priors: dict) -> dict:
        """Tight priors on the known nuisance parameters, none on ``zeeman_split`` (see ``resonance_search``)."""
        tight: dict = {}
        for name, prior in priors.items():
            if name == "zeeman_split":
                continue  # the resonance moved: flat prior
            if isinstance(prior[0], str):
                tight[name] = prior  # ("sin^2", ...) center_freq prior: untouched
                continue
            std = prior[1] * RESONANCE_SEARCH_NUISANCE_STD_SCALE
            tight[name] = (_widened_prior_mean(rng, float(getattr(typed_params, name)), std), std)
        return tight
