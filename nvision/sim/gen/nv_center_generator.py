"""NV center signal generator."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from nvision.spectra.nv_center import (
    DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
    DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
    MAX_HWHM,
    MAX_HYPERFINE_SHIFT,
    MAX_K_NP,
    MAX_ZEEMAN_SHIFT,
    MIN_HWHM,
    MIN_HYPERFINE_SHIFT,
    MIN_K_NP,
    MIN_ZEEMAN_SHIFT,
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
    _voigt_reparam_scalar,
    hyperfine_geometry,
    max_zeeman_shift_for_window,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_saturation_voigt_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
    zeeman_shift_edge_pad,
)

from .peak_spec import _true_signal_from_typed

# Gaussian FWHM = 2*sqrt(2 ln2)*sigma; used to convert a physical sigma_inhom
# (Hz) into the voigt variant's lorentz_frac ratio.
_VOIGT_SQRT2LOG2 = math.sqrt(2.0 * math.log(2.0))


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
    single dip per Zeeman group, which is what a typical NV hwhm actually
    gives you), ``"n14"`` (¹⁴N, I=1, a resolved triplet) or ``"n15"`` (¹⁵N,
    I=1/2, a resolved doublet). ``infer_hyperfine=True`` additionally randomizes
    ``hyperfine_shift``/``k_np`` per repeat and exposes them as free parameters instead of
    pinning them to the isotope's physical coupling.

    ``zero_field_freq`` (the zero-field center) is fixed at the midpoint of the safe
    range for every draw -- like a known, calibrated instrument constant -- so
    only zeeman_shift/hyperfine/hwhm/contrast vary between repeats.
    """

    drive_freq_min_phys: float = DEFAULT_NV_DRIVE_FREQ_MIN_PHYS  # 2.6 GHz
    drive_freq_max_phys: float = DEFAULT_NV_DRIVE_FREQ_MAX_PHYS  # 3.1 GHz
    variant: str = "lorentzian"  # "lorentzian", "voigt", "saturation_voigt", "cw_drive" or "pulsed_drive"
    hyperfine: str = "unresolved"  # resolved hyperfine structure: "unresolved" (default) / "n14" / "n15"
    infer_hyperfine: bool = False  # randomize + infer hyperfine_shift/k_np instead of using the isotope constant
    with_zeeman_splitting: bool = True  # default: Zeeman-split two-dip model
    hwhm: float | None = None  # if set, fix hwhm (HWHM, Hz) instead of randomizing (lorentzian, voigt)
    contrast_total: float | None = None  # if set, fix contrast instead of randomizing (lorentzian, voigt)
    # if set, fix Lorentzian share of broadening instead of randomizing (voigt only);
    # 1.0 = pure Lorentzian (no inhomogeneous/Gaussian broadening)
    lorentz_frac: float | None = None
    saturation: float | None = None  # if set, fix drive saturation (homogeneous channel; saturation_voigt only)
    # if set, fix inhomogeneous width, Hz (saturation_voigt; also voigt, converted to
    # lorentz_frac via hwhm -- ignored if lorentz_frac is also set)
    sigma_inhom: float | None = None
    # c_max (saturated contrast scale) is not a per-generator field: it's a fixed
    # physical constant (NV_SATURATION_C_MAX in nv_center.py), not something that
    # varies between repeats or studies.
    # Drive-frequency window as a multiple of the signal's total FWHM, recorded as a grid coordinate by
    # the SNR x window study (sim.presets.voigt_snr_window_param_grid_generators, which also sets
    # drive_freq_min/max_phys to match). Its only effect here: the Zeeman range follows the window
    # (see ``zeeman_max_phys``). None = the default window and MAX_ZEEMAN_SHIFT. Voigt variant only.
    window_fwhm: float | None = None

    @property
    def gaussian_share(self) -> float | None:
        """Gaussian part of the total FWHM (0 = pure Lorentzian) of an SNR x window signal; else ``None``."""
        if self.window_fwhm is None or self.hwhm is None or self.sigma_inhom is None:
            return None
        _, lorentz_frac = _voigt_reparam_scalar(self.hwhm, self.sigma_inhom)
        return 1.0 - lorentz_frac

    @property
    def width_bounds_reference_phys(self) -> float:
        """Window width (Hz) the width priors' bounds are sized off (see ``nv_center_voigt_bounds_for_domain``).

        The default window when ``window_fwhm`` stretches the window: the linewidth is held fixed there, so how
        well it is known must not change with the window. The actual window otherwise.
        """
        if self.window_fwhm is None:
            return self.drive_freq_max_phys - self.drive_freq_min_phys
        return DEFAULT_NV_DRIVE_FREQ_MAX_PHYS - DEFAULT_NV_DRIVE_FREQ_MIN_PHYS

    @property
    def zeeman_max_phys(self) -> float:
        """Largest ``zeeman_shift`` (Hz) this generator draws and its belief bounds allow."""
        if self.window_fwhm is None:
            return MAX_ZEEMAN_SHIFT
        if self.variant != "voigt":
            raise ValueError(f"window_fwhm is only supported for variant='voigt', got {self.variant!r}.")
        return max_zeeman_shift_for_window(self.drive_freq_max_phys - self.drive_freq_min_phys)

    @property
    def zeeman_min_phys(self) -> float:
        """Smallest ``zeeman_shift`` (Hz) this generator draws.

        ``MIN_ZEEMAN_SHIFT`` except for window-grid signals (``window_fwhm`` set): the window is ``[D, D + W]``, so
        a dip drawn near D would have its left flank cut off. They draw no closer than
        :func:`~nvision.spectra.nv_center.zeeman_shift_edge_pad` (the total FWHM plus the hyperfine offset), which
        needs ``hwhm`` and ``sigma_inhom`` fixed so the width is known before the draw.
        """
        if self.window_fwhm is None:
            return MIN_ZEEMAN_SHIFT
        if self.hwhm is None or self.sigma_inhom is None:
            raise ValueError("window_fwhm needs a fixed hwhm and sigma_inhom: the Zeeman padding is sized off them.")
        fwhm_total, _ = _voigt_reparam_scalar(self.hwhm, self.sigma_inhom)
        pad = zeeman_shift_edge_pad(fwhm_total, hyperfine_geometry(self.hyperfine)[0])
        if not pad < self.zeeman_max_phys:
            raise ValueError(
                f"window_fwhm={self.window_fwhm:g}: padding the dip by {pad:g} Hz leaves no room below the largest "
                f"zeeman_shift {self.zeeman_max_phys:g} Hz; widen the window."
            )
        return max(MIN_ZEEMAN_SHIFT, pad)

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
        # Random hwhm (HWHM for Lorentzian), or fixed if provided
        hwhm = self.hwhm if self.hwhm is not None else rng.uniform(MIN_HWHM, MAX_HWHM)

        zeeman_max = self.zeeman_max_phys
        zeeman_shift = 0.0
        # Hyperfine line offset: drawn per repeat only when it's a free parameter,
        # otherwise the isotope's own physical coupling -- exactly 0 when the
        # structure is unresolved, which collapses the multiplet to one dip.
        hyperfine_shift = hyperfine_geometry(self.hyperfine)[0]
        if self.with_zeeman_splitting:
            zeeman_shift = rng.uniform(self.zeeman_min_phys, zeeman_max)
        if self.infer_hyperfine:
            hyperfine_shift = rng.uniform(MIN_HYPERFINE_SHIFT, MAX_HYPERFINE_SHIFT)

        # The zero-field center is a fixed, known reference (like a calibrated instrument
        # constant) -- only zeeman_shift/hyperfine/hwhm/contrast vary between draws.
        # The drive-frequency window is the upper half [D, D + delta] of the mirror-symmetric spectrum
        # (see DEFAULT_NV_DRIVE_FREQ_MIN_PHYS), so the center is D itself, on its lower edge.
        zero_field_freq = NV_ZERO_FIELD_SPLITTING_HZ

        if self.variant == "lorentzian":
            contrast_total = self.contrast_total if self.contrast_total is not None else rng.uniform(0.1, 0.4)
            hwhm_std = (MAX_HWHM - MIN_HWHM) * PRIOR_STD_FRACTION
            contrast_total_std = 0.3 * PRIOR_STD_FRACTION  # contrast_total range is roughly [0.1, 0.4]

            if self.with_zeeman_splitting and self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterLorentzianModel(
                    with_zeeman_splitting=True, hyperfine=self.hyperfine, infer_hyperfine=True
                )
                from nvision.spectra.nv_center import NVCenterLorentzianZeemanHyperfineSpectrum

                typed_params = NVCenterLorentzianZeemanHyperfineSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm=hwhm,
                    zeeman_shift=zeeman_shift,
                    hyperfine_shift=hyperfine_shift,
                    k_np=k_np,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    infer_hyperfine=True,
                    with_zeeman_splitting=True,
                )
                zeeman_std = (MAX_ZEEMAN_SHIFT - MIN_ZEEMAN_SHIFT) * PRIOR_STD_FRACTION
                hyperfine_shift_std = (MAX_HYPERFINE_SHIFT - MIN_HYPERFINE_SHIFT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_shift": (_widened_prior_mean(rng, zeeman_shift, zeeman_std), zeeman_std),
                    "hyperfine_shift": (
                        _widened_prior_mean(rng, hyperfine_shift, hyperfine_shift_std),
                        hyperfine_shift_std,
                    ),
                    "hwhm": (_widened_prior_mean(rng, hwhm, hwhm_std), hwhm_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }
            elif self.with_zeeman_splitting:
                model = NVCenterLorentzianModel(with_zeeman_splitting=True, hyperfine=self.hyperfine)
                typed_params = NVCenterLorentzianZeemanSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm=hwhm,
                    zeeman_shift=zeeman_shift,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    with_zeeman_splitting=True,
                )
                zeeman_std = (MAX_ZEEMAN_SHIFT - MIN_ZEEMAN_SHIFT) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_shift": (_widened_prior_mean(rng, zeeman_shift, zeeman_std), zeeman_std),
                    "hwhm": (_widened_prior_mean(rng, hwhm, hwhm_std), hwhm_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }
            elif self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterLorentzianModel(hyperfine=self.hyperfine, infer_hyperfine=True)
                typed_params = NVCenterLorentzianSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm=hwhm,
                    hyperfine_shift=hyperfine_shift,
                    k_np=k_np,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys, self.drive_freq_max_phys, hyperfine=self.hyperfine, infer_hyperfine=True
                )
                hyperfine_shift_std = (MAX_HYPERFINE_SHIFT - MIN_HYPERFINE_SHIFT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "hyperfine_shift": (
                        _widened_prior_mean(rng, hyperfine_shift, hyperfine_shift_std),
                        hyperfine_shift_std,
                    ),
                    "hwhm": (_widened_prior_mean(rng, hwhm, hwhm_std), hwhm_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }
            else:
                model = NVCenterLorentzianModel(hyperfine=self.hyperfine)
                typed_params = NVCenterLorentzianSingleDipSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm=hwhm,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.drive_freq_min_phys, self.drive_freq_max_phys, hyperfine=self.hyperfine
                )
                bounds["_priors"] = {
                    "hwhm": (_widened_prior_mean(rng, hwhm, hwhm_std), hwhm_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }
        elif self.variant in ("cw_drive", "pulsed_drive"):
            return self._generate_drive_controlled(rng, zero_field_freq, zeeman_shift)
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
                "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
            }

            if self.with_zeeman_splitting and self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                typed_params = NVCenterSaturationVoigtZeemanHyperfineSpectrum(
                    zero_field_freq=zero_field_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    zeeman_shift=zeeman_shift,
                    hyperfine_shift=hyperfine_shift,
                    k_np=k_np,
                )
                zeeman_std = (MAX_ZEEMAN_SHIFT - MIN_ZEEMAN_SHIFT) * PRIOR_STD_FRACTION
                hyperfine_shift_std = (MAX_HYPERFINE_SHIFT - MIN_HYPERFINE_SHIFT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                priors["zeeman_shift"] = (_widened_prior_mean(rng, zeeman_shift, zeeman_std), zeeman_std)
                priors["hyperfine_shift"] = (
                    _widened_prior_mean(rng, hyperfine_shift, hyperfine_shift_std),
                    hyperfine_shift_std,
                )
                priors["k_np"] = (_widened_prior_mean(rng, k_np, k_np_std), k_np_std)
            elif self.with_zeeman_splitting:
                typed_params = NVCenterSaturationVoigtZeemanSpectrum(
                    zero_field_freq=zero_field_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    zeeman_shift=zeeman_shift,
                )
                zeeman_std = (MAX_ZEEMAN_SHIFT - MIN_ZEEMAN_SHIFT) * PRIOR_STD_FRACTION
                priors["zeeman_shift"] = (_widened_prior_mean(rng, zeeman_shift, zeeman_std), zeeman_std)
            elif self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                typed_params = NVCenterSaturationVoigtSpectrum(
                    zero_field_freq=zero_field_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    hyperfine_shift=hyperfine_shift,
                    k_np=k_np,
                )
                hyperfine_shift_std = (MAX_HYPERFINE_SHIFT - MIN_HYPERFINE_SHIFT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                priors["hyperfine_shift"] = (
                    _widened_prior_mean(rng, hyperfine_shift, hyperfine_shift_std),
                    hyperfine_shift_std,
                )
                priors["k_np"] = (_widened_prior_mean(rng, k_np, k_np_std), k_np_std)
            else:
                typed_params = NVCenterSaturationVoigtSingleDipSpectrum(
                    zero_field_freq=zero_field_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                )

            bounds = sv_bounds
            bounds["_priors"] = priors
        else:  # voigt
            hwhm_lorentz = hwhm
            if self.sigma_inhom is not None:
                sigma_inhom = self.sigma_inhom
            elif self.lorentz_frac is not None:
                # Backward-compat: convert a pinned lorentz_frac into the equivalent
                # sigma_inhom (inverse of _voigt_reparam_scalar's fwhm_l/(fwhm_l+fwhm_g)
                # -> lorentz_frac, evaluated at this draw's hwhm_lorentz).
                lorentz_ratio = 1.0 / self.lorentz_frac - 1.0 if self.lorentz_frac > 0 else 0.0
                sigma_inhom = hwhm_lorentz * lorentz_ratio / _VOIGT_SQRT2LOG2
            else:
                lorentz_ratio = rng.uniform(0.1, 0.3)  # fwhm_gauss / fwhm_lorentz
                sigma_inhom = hwhm_lorentz * lorentz_ratio / _VOIGT_SQRT2LOG2

            contrast_total = self.contrast_total if self.contrast_total is not None else rng.uniform(0.1, 0.4)
            hwhm_std = (MAX_HWHM - MIN_HWHM) * PRIOR_STD_FRACTION
            # sigma_inhom's own bound range (mirrors nv_center_voigt_bounds_for_domain's
            # sigma_inhom_hi, which doesn't depend on the hyperfine/zeeman flags) — not
            # hwhm's range, which this previously (incorrectly) copied.
            voigt_sigma_inhom_hi = max(1.2e6, self.width_bounds_reference_phys * 0.02)
            sigma_inhom_std = voigt_sigma_inhom_hi * PRIOR_STD_FRACTION
            contrast_total_std = 0.3 * PRIOR_STD_FRACTION  # contrast_total range is roughly [0.1, 0.4]

            if self.with_zeeman_splitting and self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterVoigtModel(with_zeeman_splitting=True, hyperfine=self.hyperfine, infer_hyperfine=True)
                from nvision.spectra.nv_center import NVCenterVoigtZeemanHyperfineSpectrum

                typed_params = NVCenterVoigtZeemanHyperfineSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm_lorentz=hwhm_lorentz,
                    sigma_inhom=sigma_inhom,
                    zeeman_shift=zeeman_shift,
                    hyperfine_shift=hyperfine_shift,
                    k_np=k_np,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    infer_hyperfine=True,
                    with_zeeman_splitting=True,
                    max_zeeman_shift=zeeman_max,
                    reference_width=self.width_bounds_reference_phys,
                )
                zeeman_std = (zeeman_max - MIN_ZEEMAN_SHIFT) * PRIOR_STD_FRACTION
                hyperfine_shift_std = (MAX_HYPERFINE_SHIFT - MIN_HYPERFINE_SHIFT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_shift": (_widened_prior_mean(rng, zeeman_shift, zeeman_std), zeeman_std),
                    "hyperfine_shift": (
                        _widened_prior_mean(rng, hyperfine_shift, hyperfine_shift_std),
                        hyperfine_shift_std,
                    ),
                    "hwhm_lorentz": (
                        _widened_prior_mean(rng, hwhm_lorentz, hwhm_std),
                        hwhm_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }
            elif self.with_zeeman_splitting:
                model = NVCenterVoigtModel(with_zeeman_splitting=True, hyperfine=self.hyperfine)
                typed_params = NVCenterVoigtZeemanSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm_lorentz=hwhm_lorentz,
                    sigma_inhom=sigma_inhom,
                    zeeman_shift=zeeman_shift,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    with_zeeman_splitting=True,
                    max_zeeman_shift=zeeman_max,
                    reference_width=self.width_bounds_reference_phys,
                )
                zeeman_std = (zeeman_max - MIN_ZEEMAN_SHIFT) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_shift": (_widened_prior_mean(rng, zeeman_shift, zeeman_std), zeeman_std),
                    "hwhm_lorentz": (
                        _widened_prior_mean(rng, hwhm_lorentz, hwhm_std),
                        hwhm_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }
            elif self.infer_hyperfine:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterVoigtModel(hyperfine=self.hyperfine, infer_hyperfine=True)
                typed_params = NVCenterVoigtSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm_lorentz=hwhm_lorentz,
                    sigma_inhom=sigma_inhom,
                    hyperfine_shift=hyperfine_shift,
                    k_np=k_np,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    infer_hyperfine=True,
                    reference_width=self.width_bounds_reference_phys,
                )
                hyperfine_shift_std = (MAX_HYPERFINE_SHIFT - MIN_HYPERFINE_SHIFT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "hyperfine_shift": (
                        _widened_prior_mean(rng, hyperfine_shift, hyperfine_shift_std),
                        hyperfine_shift_std,
                    ),
                    "hwhm_lorentz": (
                        _widened_prior_mean(rng, hwhm_lorentz, hwhm_std),
                        hwhm_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "k_np": (_widened_prior_mean(rng, k_np, k_np_std), k_np_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }
            else:
                model = NVCenterVoigtModel(hyperfine=self.hyperfine)
                typed_params = NVCenterVoigtSingleDipSpectrum(
                    zero_field_freq=zero_field_freq,
                    hwhm_lorentz=hwhm_lorentz,
                    sigma_inhom=sigma_inhom,
                    contrast_total=contrast_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.drive_freq_min_phys,
                    self.drive_freq_max_phys,
                    hyperfine=self.hyperfine,
                    reference_width=self.width_bounds_reference_phys,
                )
                bounds["_priors"] = {
                    "hwhm_lorentz": (
                        _widened_prior_mean(rng, hwhm_lorentz, hwhm_std),
                        hwhm_std,
                    ),
                    "sigma_inhom": (_widened_prior_mean(rng, sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "contrast_total": (
                        _widened_prior_mean(rng, contrast_total, contrast_total_std),
                        contrast_total_std,
                    ),
                    "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
                }

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)

    def _generate_drive_controlled(self, rng: random.Random, zero_field_freq: float, zeeman_shift: float):
        """Drive-controlled CW (``variant="cw_drive"``) or pulsed (``"pulsed_drive"``) NV signal.

        The microwave drive is a per-measurement setting (see nvision/spectra/nv_center_drive.py); the
        repeat draws the sample's *response* to it (``drive_amp_sat`` for CW, ``rabi_per_amp`` and
        ``contrast_total`` for pulsed) plus ``sigma_inhom`` and ``zeeman_shift``. The baseline drive -- used by
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
            "zero_field_freq": ("sin^2", np.pi / (2.0 * MIN_HWHM)),
        }
        if self.with_zeeman_splitting:
            priors["zeeman_shift"] = (
                _widened_prior_mean(rng, zeeman_shift, _std("zeeman_shift")),
                _std("zeeman_shift"),
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
            common = {"zero_field_freq": zero_field_freq, "drive_amp_sat": drive_amp_sat, "sigma_inhom": sigma_inhom}
            typed_params = (
                NVCenterCWDriveZeemanSpectrum(**common, zeeman_shift=zeeman_shift)
                if self.with_zeeman_splitting
                else NVCenterCWDriveSpectrum(**common)
            )
            baseline_drive = DriveSetting(BASELINE_DRIVE_AMP_UNIT)
        else:
            rabi_per_amp = rng.uniform(*bounds["rabi_per_amp"])
            contrast_total = (
                self.contrast_total if self.contrast_total is not None else rng.uniform(*bounds["contrast_total"])
            )
            priors["rabi_per_amp"] = (
                _widened_prior_mean(rng, rabi_per_amp, _std("rabi_per_amp")),
                _std("rabi_per_amp"),
            )
            priors["contrast_total"] = (
                _widened_prior_mean(rng, contrast_total, _std("contrast_total")),
                _std("contrast_total"),
            )
            model = NVCenterPulsedDriveModel(hyperfine=self.hyperfine, with_zeeman_splitting=self.with_zeeman_splitting)
            common = {
                "zero_field_freq": zero_field_freq,
                "rabi_per_amp": rabi_per_amp,
                "contrast_total": contrast_total,
                "sigma_inhom": sigma_inhom,
            }
            typed_params = (
                NVCenterPulsedZeemanSpectrum(**common, zeeman_shift=zeeman_shift)
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
