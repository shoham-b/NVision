"""NV center signal generator."""

from __future__ import annotations

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
    NV_N14_HYPERFINE_SPLIT_HZ,
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
    NVCenterVoigtSpectrum,
    NVCenterVoigtSpectrumSamples,
    NVCenterVoigtZeemanSpectrum,
    NVCenterVoigtZeemanSpectrumSamples,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_saturation_voigt_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
)

from .peak_spec import _true_signal_from_typed


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
    sigma_inhom: float | None = None  # if set, fix inhomogeneous width, Hz (saturation_voigt only)
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
            _margin_model = NVCenterVoigtModel(with_zeeman_splitting=self.with_zeeman_splitting)
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
        else:  # voigt — always uses split and k_np; zeeman_split too when enabled
            voigt_k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
            if self.lorentz_frac is not None:
                lorentz_frac = self.lorentz_frac
                lorentz_ratio = 1.0 / lorentz_frac - 1.0  # fwhm_gauss / fwhm_lorentz
            else:
                lorentz_ratio = rng.uniform(0.1, 0.3)  # fwhm_gauss / fwhm_lorentz
                lorentz_frac = 1.0 / (1.0 + lorentz_ratio)

            if self.with_zeeman_splitting:
                voigt_split = rng.uniform(MIN_SPLIT, MAX_SPLIT)
                fwhm_total = 2 * linewidth * (1.0 + lorentz_ratio)
            else:
                # No Zeeman splitting -> hyperfine structure alone is on screen. Fix the
                # split to the physical N-14 constant (like NVCenterLorentzianModel's
                # with_hyperfine_splitting=False) and, unless the caller pins ``linewidth``
                # explicitly, draw fwhm_total as a large multiple of the hyperfine spacing
                # (>= 4x(2*split), empirically the threshold past which the pseudo-Voigt
                # triplet always collapses to one dip regardless of lorentz_frac/k_np — see
                # tests/spectra/test_pseudo_voigt_accuracy.py) so the Gaussian/Lorentzian
                # broadening reliably washes the triplet out into a single unresolved dip,
                # matching the "no field -> one blob" physical expectation.
                voigt_split = NV_N14_HYPERFINE_SPLIT_HZ
                merge_ratio_min = 4.5  # fwhm_total / (2 * voigt_split); empirical merge threshold is ~4.0
                if self.linewidth is not None:
                    # A caller-pinned linewidth must still land in the merge-safe zone --
                    # promote it to the floor implied by merge_ratio_min instead of letting
                    # a too-small pinned value silently resolve the triplet.
                    min_fwhm_total = merge_ratio_min * 2.0 * voigt_split
                    linewidth_floor = min_fwhm_total / (2.0 * (1.0 + lorentz_ratio))
                    effective_linewidth = max(linewidth, linewidth_floor)
                    fwhm_total = 2 * effective_linewidth * (1.0 + lorentz_ratio)
                else:
                    merge_ratio = rng.uniform(merge_ratio_min, 6.0)
                    fwhm_total = merge_ratio * 2.0 * voigt_split

            model = NVCenterVoigtModel(with_zeeman_splitting=self.with_zeeman_splitting)
            # Scale a desired contrast onto the true peak-shape maximum
            unit_dip_depth = self.c_total if self.c_total is not None else rng.uniform(0.3, 0.95)

            split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
            fwhm_total_std = (MAX_LINEWIDTH * 2 - MIN_LINEWIDTH * 2) * PRIOR_STD_FRACTION
            lorentz_frac_std = 0.2 * PRIOR_STD_FRACTION
            k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
            dip_depth_std = 0.65 * PRIOR_STD_FRACTION

            if self.with_zeeman_splitting:
                half_domain = zeeman_split + voigt_split
                xs = np.linspace(center_freq - half_domain, center_freq + half_domain, 400)
                single = NVCenterVoigtZeemanSpectrumSamples(
                    frequency=np.array([center_freq]),
                    fwhm_total=np.array([fwhm_total]),
                    lorentz_frac=np.array([lorentz_frac]),
                    zeeman_split=np.array([zeeman_split]),
                    split=np.array([voigt_split]),
                    k_np=np.array([voigt_k_np]),
                    dip_depth=np.array([1.0]),
                )
                g_max = float(1.0 - np.min(model.compute_vectorized_many(xs, single)))
                dip_depth = unit_dip_depth / g_max if g_max > 1e-12 else unit_dip_depth

                typed_params = NVCenterVoigtZeemanSpectrum(
                    frequency=center_freq,
                    fwhm_total=fwhm_total,
                    lorentz_frac=lorentz_frac,
                    zeeman_split=zeeman_split,
                    split=voigt_split,
                    k_np=voigt_k_np,
                    dip_depth=dip_depth,
                )
                bounds = nv_center_voigt_bounds_for_domain(self.x_min, self.x_max, with_zeeman_splitting=True)

                zeeman_std = (MAX_ZEEMAN_SPLIT - MIN_ZEEMAN_SPLIT) * PRIOR_STD_FRACTION
                bounds["_priors"] = {
                    "zeeman_split": (rng.gauss(zeeman_split, zeeman_std), zeeman_std),
                    "split": (rng.gauss(voigt_split, split_std), split_std),
                    "fwhm_total": (rng.gauss(fwhm_total, fwhm_total_std), fwhm_total_std),
                    "lorentz_frac": (rng.gauss(lorentz_frac, lorentz_frac_std), lorentz_frac_std),
                    "k_np": (rng.gauss(voigt_k_np, k_np_std), k_np_std),
                    "dip_depth": (rng.gauss(dip_depth, dip_depth_std), dip_depth_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }
            else:
                xs = np.linspace(center_freq - voigt_split, center_freq + voigt_split, 200)
                single = NVCenterVoigtSpectrumSamples(
                    frequency=np.array([center_freq]),
                    fwhm_total=np.array([fwhm_total]),
                    lorentz_frac=np.array([lorentz_frac]),
                    split=np.array([voigt_split]),
                    k_np=np.array([voigt_k_np]),
                    dip_depth=np.array([1.0]),
                )
                g_max = float(1.0 - np.min(model.compute_vectorized_many(xs, single)))
                dip_depth = unit_dip_depth / g_max if g_max > 1e-12 else unit_dip_depth

                typed_params = NVCenterVoigtSpectrum(
                    frequency=center_freq,
                    fwhm_total=fwhm_total,
                    lorentz_frac=lorentz_frac,
                    split=voigt_split,
                    k_np=voigt_k_np,
                    dip_depth=dip_depth,
                )
                bounds = nv_center_voigt_bounds_for_domain(self.x_min, self.x_max)

                bounds["_priors"] = {
                    "split": (rng.gauss(voigt_split, split_std), split_std),
                    "fwhm_total": (rng.gauss(fwhm_total, fwhm_total_std), fwhm_total_std),
                    "lorentz_frac": (rng.gauss(lorentz_frac, lorentz_frac_std), lorentz_frac_std),
                    "k_np": (rng.gauss(voigt_k_np, k_np_std), k_np_std),
                    "dip_depth": (rng.gauss(dip_depth, dip_depth_std), dip_depth_std),
                    "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
                }

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)
