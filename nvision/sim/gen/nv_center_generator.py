"""NV center signal generator."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from nvision.spectra.nv_center import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    MAX_K_NP,
    MAX_LINEWIDTH,
    MAX_SPLIT,
    MAX_ZEEMAN_SPLIT,
    MIN_K_NP,
    MIN_LINEWIDTH,
    MIN_SPLIT,
    MIN_ZEEMAN_SPLIT,
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
    nv_center_lorentzian_bounds_for_domain,
    nv_center_saturation_voigt_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
)

from .peak_spec import _true_signal_from_typed

# Gaussian FWHM = 2*sqrt(2 ln2)*sigma; used to convert a physical sigma_inhom
# (Hz) into the voigt variant's lorentz_frac ratio.
_VOIGT_SQRT2LOG2 = math.sqrt(2.0 * math.log(2.0))


@dataclass
class NVCenterCoreGenerator:
    """Generates NV center ODMR signals using core architecture.

    Produces TrueSignal with physically accurate NV center signal.
    By default generates a single-dip signal (no hyperfine splitting).
    Set ``with_hyperfine_splitting=True`` to generate a triple-dip signal.

    ``frequency`` (the zero-field center) is fixed at the midpoint of the safe
    range for every draw -- like a known, calibrated instrument constant -- so
    only zeeman_split/hyperfine/linewidth/contrast vary between repeats.
    """

    x_min: float = DEFAULT_NV_CENTER_FREQ_X_MIN  # 2.6 GHz
    x_max: float = DEFAULT_NV_CENTER_FREQ_X_MAX  # 3.1 GHz
    variant: str = "lorentzian"  # "lorentzian", "voigt", or "saturation_voigt"
    with_hyperfine_splitting: bool = False  # default: single-dip (no split/k_np)
    with_zeeman_splitting: bool = True  # default: Zeeman-split two-dip model
    linewidth: float | None = None  # if set, fix linewidth (HWHM, Hz) instead of randomizing (lorentzian, voigt)
    c_total: float | None = None  # if set, fix contrast instead of randomizing (lorentzian, voigt)
    lorentz_frac: float | None = None  # if set, fix Lorentzian share of broadening instead of randomizing (voigt only); 1.0 = pure Lorentzian (no inhomogeneous/Gaussian broadening)
    saturation: float | None = None  # if set, fix drive saturation (homogeneous channel; saturation_voigt only)
    sigma_inhom: float | None = None  # if set, fix inhomogeneous width, Hz (saturation_voigt; also voigt, converted to lorentz_frac via linewidth -- ignored if lorentz_frac is also set)
    # c_max (saturated contrast scale) is not a per-generator field: it's a fixed
    # physical constant (NV_SATURATION_C_MAX in nv_center.py), not something that
    # varies between repeats or studies.

    def generate(self, rng: random.Random):  # noqa: C901
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
        width = self.x_max - self.x_min

        # Random linewidth (HWHM for Lorentzian), or fixed if provided
        linewidth = self.linewidth if self.linewidth is not None else rng.uniform(MIN_LINEWIDTH, MAX_LINEWIDTH)

        zeeman_split = 0.0
        split = 0.0
        if self.with_zeeman_splitting:
            zeeman_split = rng.uniform(MIN_ZEEMAN_SPLIT, MAX_ZEEMAN_SPLIT)
        if self.with_hyperfine_splitting:
            split = rng.uniform(MIN_SPLIT, MAX_SPLIT)

        # Margin needed to keep the full spectrum -- not just its center -- inside
        # [x_min, x_max]. Uses each model's own signal_max_span() (the worst-case
        # center-to-outermost-dip-edge extent across every value this generator could
        # draw for linewidth/split/saturation/etc.), not this repeat's actual drawn
        # values, so center_freq is safely bounded regardless of what gets drawn below.
        if self.variant == "saturation_voigt":
            _margin_model = NVCenterSaturationVoigtModel(
                with_hyperfine_splitting=self.with_hyperfine_splitting,
                with_zeeman_splitting=self.with_zeeman_splitting,
            )
        elif self.variant == "voigt":
            _margin_model = NVCenterVoigtModel(
                with_hyperfine_splitting=self.with_hyperfine_splitting,
                with_zeeman_splitting=self.with_zeeman_splitting,
            )
        else:
            _margin_model = NVCenterLorentzianModel(
                with_hyperfine_splitting=self.with_hyperfine_splitting,
                with_zeeman_splitting=self.with_zeeman_splitting,
            )
        max_span = _margin_model.signal_max_span(width)
        margin = max_span / 2.0 if max_span is not None else zeeman_split + split

        usable_lo = self.x_min + margin + 0.05 * width
        usable_hi = self.x_max - margin - 0.05 * width
        if usable_lo >= usable_hi:
            usable_lo = self.x_min + 0.05 * width
            usable_hi = self.x_max - 0.05 * width

        # Zero-field center is a fixed, known reference (like a calibrated instrument
        # constant) -- only zeeman_split/hyperfine/linewidth/contrast vary between
        # draws. Fixed at the midpoint of the safe (margin-adjusted) range so the
        # full dip cluster fits in [x_min, x_max] regardless of what gets drawn above.
        center_freq = (usable_lo + usable_hi) / 2.0

        if self.variant == "lorentzian":
            c_total = self.c_total if self.c_total is not None else rng.uniform(0.1, 0.4)
            linewidth_std = (MAX_LINEWIDTH - MIN_LINEWIDTH) * PRIOR_STD_FRACTION
            c_total_std = 0.3 * PRIOR_STD_FRACTION  # c_total range is roughly [0.1, 0.4]

            if self.with_zeeman_splitting and self.with_hyperfine_splitting:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterLorentzianModel(with_zeeman_splitting=True, with_hyperfine_splitting=True)
                from nvision.spectra.nv_center import NVCenterLorentzianZeemanHyperfineSpectrum
                typed_params = NVCenterLorentzianZeemanHyperfineSpectrum(
                    frequency=center_freq,
                    linewidth=linewidth,
                    zeeman_split=zeeman_split,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.x_min, self.x_max, with_hyperfine_splitting=True, with_zeeman_splitting=True
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (rng.gauss(zeeman_split, zeeman_std), zeeman_std),
                    "split": (rng.gauss(split, split_std), split_std),
                    "linewidth": (rng.gauss(linewidth, linewidth_std), linewidth_std),
                    "k_np": (rng.gauss(k_np, k_np_std), k_np_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.with_zeeman_splitting:
                model = NVCenterLorentzianModel(with_zeeman_splitting=True, with_hyperfine_splitting=False)
                typed_params = NVCenterLorentzianZeemanSpectrum(
                    frequency=center_freq,
                    linewidth=linewidth,
                    zeeman_split=zeeman_split,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(
                    self.x_min, self.x_max, with_hyperfine_splitting=False, with_zeeman_splitting=True
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (rng.gauss(zeeman_split, zeeman_std), zeeman_std),
                    "linewidth": (rng.gauss(linewidth, linewidth_std), linewidth_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.with_hyperfine_splitting:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterLorentzianModel(with_hyperfine_splitting=True)
                typed_params = NVCenterLorentzianSpectrum(
                    frequency=center_freq,
                    linewidth=linewidth,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(self.x_min, self.x_max, with_hyperfine_splitting=True)
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "split": (rng.gauss(split, split_std), split_std),
                    "linewidth": (rng.gauss(linewidth, linewidth_std), linewidth_std),
                    "k_np": (rng.gauss(k_np, k_np_std), k_np_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            else:
                model = NVCenterLorentzianModel(with_hyperfine_splitting=False)
                typed_params = NVCenterLorentzianSingleDipSpectrum(
                    frequency=center_freq,
                    linewidth=linewidth,
                    c_total=c_total,
                )
                bounds = nv_center_lorentzian_bounds_for_domain(self.x_min, self.x_max, with_hyperfine_splitting=False)
                bounds["_priors"] = {
                    "linewidth": (rng.gauss(linewidth, linewidth_std), linewidth_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
        elif self.variant == "saturation_voigt":
            sv_bounds = nv_center_saturation_voigt_bounds_for_domain(
                self.x_min,
                self.x_max,
                with_hyperfine_splitting=self.with_hyperfine_splitting,
                with_zeeman_splitting=self.with_zeeman_splitting,
            )
            saturation_bounds = sv_bounds["saturation"]
            sigma_inhom_bounds = sv_bounds["sigma_inhom"]

            saturation = self.saturation if self.saturation is not None else rng.uniform(*saturation_bounds)
            sigma_inhom = self.sigma_inhom if self.sigma_inhom is not None else rng.uniform(*sigma_inhom_bounds)

            saturation_std = (saturation_bounds[1] - saturation_bounds[0]) * PRIOR_STD_FRACTION
            sigma_inhom_std = (sigma_inhom_bounds[1] - sigma_inhom_bounds[0]) * PRIOR_STD_FRACTION

            model = NVCenterSaturationVoigtModel(
                with_hyperfine_splitting=self.with_hyperfine_splitting,
                with_zeeman_splitting=self.with_zeeman_splitting,
            )
            priors = {
                "saturation": (rng.gauss(saturation, saturation_std), saturation_std),
                "sigma_inhom": (rng.gauss(sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
            }

            if self.with_zeeman_splitting and self.with_hyperfine_splitting:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                typed_params = NVCenterSaturationVoigtZeemanHyperfineSpectrum(
                    frequency=center_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                    split=split,
                    k_np=k_np,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                priors["zeeman_split"] = (rng.gauss(zeeman_split, zeeman_std), zeeman_std)
                priors["split"] = (rng.gauss(split, split_std), split_std)
                priors["k_np"] = (rng.gauss(k_np, k_np_std), k_np_std)
            elif self.with_zeeman_splitting:
                typed_params = NVCenterSaturationVoigtZeemanSpectrum(
                    frequency=center_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                priors["zeeman_split"] = (rng.gauss(zeeman_split, zeeman_std), zeeman_std)
            elif self.with_hyperfine_splitting:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                typed_params = NVCenterSaturationVoigtSpectrum(
                    frequency=center_freq,
                    saturation=saturation,
                    sigma_inhom=sigma_inhom,
                    split=split,
                    k_np=k_np,
                )
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                priors["split"] = (rng.gauss(split, split_std), split_std)
                priors["k_np"] = (rng.gauss(k_np, k_np_std), k_np_std)
            else:
                typed_params = NVCenterSaturationVoigtSingleDipSpectrum(
                    frequency=center_freq,
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
            voigt_sigma_inhom_hi = max(1.2e6, (self.x_max - self.x_min) * 0.02)
            sigma_inhom_std = voigt_sigma_inhom_hi * PRIOR_STD_FRACTION
            c_total_std = 0.3 * PRIOR_STD_FRACTION  # c_total range is roughly [0.1, 0.4]

            if self.with_zeeman_splitting and self.with_hyperfine_splitting:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterVoigtModel(with_zeeman_splitting=True, with_hyperfine_splitting=True)
                from nvision.spectra.nv_center import NVCenterVoigtZeemanHyperfineSpectrum

                typed_params = NVCenterVoigtZeemanHyperfineSpectrum(
                    frequency=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.x_min, self.x_max, with_hyperfine_splitting=True, with_zeeman_splitting=True
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (rng.gauss(zeeman_split, zeeman_std), zeeman_std),
                    "split": (rng.gauss(split, split_std), split_std),
                    "homogeneous_linewidth": (rng.gauss(homogeneous_linewidth, linewidth_std), linewidth_std),
                    "sigma_inhom": (rng.gauss(sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "k_np": (rng.gauss(k_np, k_np_std), k_np_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.with_zeeman_splitting:
                model = NVCenterVoigtModel(with_zeeman_splitting=True, with_hyperfine_splitting=False)
                typed_params = NVCenterVoigtZeemanSpectrum(
                    frequency=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    zeeman_split=zeeman_split,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(
                    self.x_min, self.x_max, with_hyperfine_splitting=False, with_zeeman_splitting=True
                )
                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (rng.gauss(zeeman_split, zeeman_std), zeeman_std),
                    "homogeneous_linewidth": (rng.gauss(homogeneous_linewidth, linewidth_std), linewidth_std),
                    "sigma_inhom": (rng.gauss(sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            elif self.with_hyperfine_splitting:
                k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
                model = NVCenterVoigtModel(with_hyperfine_splitting=True)
                typed_params = NVCenterVoigtSpectrum(
                    frequency=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    split=split,
                    k_np=k_np,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(self.x_min, self.x_max, with_hyperfine_splitting=True)
                split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
                k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "split": (rng.gauss(split, split_std), split_std),
                    "homogeneous_linewidth": (rng.gauss(homogeneous_linewidth, linewidth_std), linewidth_std),
                    "sigma_inhom": (rng.gauss(sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "k_np": (rng.gauss(k_np, k_np_std), k_np_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            else:
                model = NVCenterVoigtModel(with_hyperfine_splitting=False)
                typed_params = NVCenterVoigtSingleDipSpectrum(
                    frequency=center_freq,
                    homogeneous_linewidth=homogeneous_linewidth,
                    sigma_inhom=sigma_inhom,
                    c_total=c_total,
                )
                bounds = nv_center_voigt_bounds_for_domain(self.x_min, self.x_max, with_hyperfine_splitting=False)
                bounds["_priors"] = {
                    "homogeneous_linewidth": (rng.gauss(homogeneous_linewidth, linewidth_std), linewidth_std),
                    "sigma_inhom": (rng.gauss(sigma_inhom, sigma_inhom_std), sigma_inhom_std),
                    "c_total": (rng.gauss(c_total, c_total_std), c_total_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)
