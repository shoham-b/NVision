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
    MIN_K_NP,
    MIN_LINEWIDTH,
    MIN_SPLIT,
    PRIOR_STD_FRACTION,
    NVCenterLorentzianModel,
    NVCenterLorentzianSingleDipSpectrum,
    NVCenterLorentzianSpectrum,
    NVCenterVoigtModel,
    NVCenterVoigtSpectrum,
    NVCenterVoigtSpectrumSamples,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
)

from .peak_spec import _true_signal_from_typed


@dataclass
class NVCenterCoreGenerator:
    """Generates NV center ODMR signals using core architecture.

    Produces TrueSignal with physically accurate NV center signal.
    By default generates a single-dip signal (no hyperfine splitting).
    Set ``with_hyperfine_splitting=True`` to generate a triple-dip signal.
    """

    x_min: float = DEFAULT_NV_CENTER_FREQ_X_MIN  # 2.6 GHz
    x_max: float = DEFAULT_NV_CENTER_FREQ_X_MAX  # 3.1 GHz
    variant: str = "lorentzian"  # "lorentzian" or "voigt"
    center_freq_fraction: float | None = None  # if set, constrain center_freq to middle fraction of domain
    with_hyperfine_splitting: bool = False  # default: single-dip (no split/k_np)

    def generate(self, rng: random.Random):  # TrueSignal
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

        # Random linewidth (HWHM for Lorentzian)
        linewidth = rng.uniform(MIN_LINEWIDTH, MAX_LINEWIDTH)

        if self.with_hyperfine_splitting:
            split = rng.uniform(MIN_SPLIT, MAX_SPLIT)
            usable_lo = self.x_min + split + 0.05 * width
            usable_hi = self.x_max - split - 0.05 * width
        else:
            usable_lo = self.x_min + 0.05 * width
            usable_hi = self.x_max - 0.05 * width

        if self.center_freq_fraction is not None:
            frac = max(0.0, min(1.0, self.center_freq_fraction))
            mid = (usable_lo + usable_hi) / 2.0
            half_span = (usable_hi - usable_lo) * frac / 2.0
            center_freq = rng.uniform(mid - half_span, mid + half_span)
        else:
            center_freq = rng.uniform(usable_lo, usable_hi)

        if self.variant == "lorentzian":
            c_total = rng.uniform(0.1, 0.4)
            linewidth_std = (MAX_LINEWIDTH - MIN_LINEWIDTH) * PRIOR_STD_FRACTION
            c_total_std = 0.3 * PRIOR_STD_FRACTION  # c_total range is roughly [0.1, 0.4]

            if self.with_hyperfine_splitting:
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
        else:  # voigt — always uses split and k_np
            voigt_split = rng.uniform(MIN_SPLIT, MAX_SPLIT)
            voigt_k_np = rng.uniform(MIN_K_NP, MAX_K_NP)
            lorentz_ratio = rng.uniform(0.1, 0.3)  # fwhm_gauss / fwhm_lorentz
            lorentz_frac = 1.0 / (1.0 + lorentz_ratio)
            fwhm_total = 2 * linewidth * (1.0 + lorentz_ratio)

            model = NVCenterVoigtModel()
            # Scale a desired contrast onto the true peak-shape maximum
            unit_dip_depth = rng.uniform(0.3, 0.95)
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

            split_std = (MAX_SPLIT - MIN_SPLIT) * PRIOR_STD_FRACTION
            fwhm_total_std = (MAX_LINEWIDTH * 2 - MIN_LINEWIDTH * 2) * PRIOR_STD_FRACTION
            lorentz_frac_std = 0.2 * PRIOR_STD_FRACTION
            k_np_std = (MAX_K_NP - MIN_K_NP) * PRIOR_STD_FRACTION
            dip_depth_std = 0.65 * PRIOR_STD_FRACTION

            bounds["_priors"] = {
                "split": (rng.gauss(voigt_split, split_std), split_std),
                "fwhm_total": (rng.gauss(fwhm_total, fwhm_total_std), fwhm_total_std),
                "lorentz_frac": (rng.gauss(lorentz_frac, lorentz_frac_std), lorentz_frac_std),
                "k_np": (rng.gauss(voigt_k_np, k_np_std), k_np_std),
                "dip_depth": (rng.gauss(dip_depth, dip_depth_std), dip_depth_std),
                "frequency": ("sin^2", np.pi / (2.0 * MIN_LINEWIDTH)),
            }

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)
