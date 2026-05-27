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
    NVCenterLorentzianModel,
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

    Produces TrueSignal with physically accurate NV center triplet signal.
    """

    x_min: float = DEFAULT_NV_CENTER_FREQ_X_MIN  # 2.6 GHz
    x_max: float = DEFAULT_NV_CENTER_FREQ_X_MAX  # 3.1 GHz
    variant: str = "lorentzian"  # "lorentzian" or "voigt"
    center_freq_fraction: float | None = None  # if set, constrain center_freq to middle fraction of domain

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

        # Generate split and linewidth from the shared physical constants
        split = rng.uniform(MIN_SPLIT, MAX_SPLIT)

        # Random linewidth (HWHM for Lorentzian)
        linewidth = rng.uniform(MIN_LINEWIDTH, MAX_LINEWIDTH)

        usable_lo = self.x_min + split + 0.05 * width
        usable_hi = self.x_max - split - 0.05 * width
        if self.center_freq_fraction is not None:
            frac = max(0.0, min(1.0, self.center_freq_fraction))
            mid = (usable_lo + usable_hi) / 2.0
            half_span = (usable_hi - usable_lo) * frac / 2.0
            center_freq = rng.uniform(mid - half_span, mid + half_span)
        else:
            center_freq = rng.uniform(usable_lo, usable_hi)

        # Random k_np (non-polarization factor)
        k_np = rng.uniform(MIN_K_NP, MAX_K_NP)

        # Normalize NV Center ODMR directly to [0, 1] bounds using exactly 1.0 maximum dip

        if self.variant == "lorentzian":
            c_total = rng.uniform(0.1, 0.4)
            model = NVCenterLorentzianModel()
            
            typed_params = NVCenterLorentzianSpectrum(
                frequency=center_freq,
                linewidth=linewidth,
                split=split,
                k_np=k_np,
                c_total=c_total,
            )
            bounds = nv_center_lorentzian_bounds_for_domain(self.x_min, self.x_max)
            
            # Generate Gaussian priors for all parameters except frequency
            prior_split = rng.gauss(split, 0.1e6)
            prior_linewidth = rng.gauss(linewidth, 10e3)
            prior_k_np = rng.gauss(k_np, 0.1)
            prior_c_total = rng.gauss(c_total, 0.002)
            
            bounds["_priors"] = {
                "split": (prior_split, 0.1e6),
                "linewidth": (prior_linewidth, 10e3),
                "k_np": (prior_k_np, 0.1),
                "c_total": (prior_c_total, 0.002),
                "frequency": ("sin^2", np.pi / width),
            }
        else:  # voigt
            lorentz_ratio = rng.uniform(0.1, 0.3)  # fwhm_gauss / fwhm_lorentz
            lorentz_frac = 1.0 / (1.0 + lorentz_ratio)
            fwhm_total = 2 * linewidth * (1.0 + lorentz_ratio)

            model = NVCenterVoigtModel()
            # Scale a desired contrast onto the true peak-shape maximum
            unit_dip_depth = rng.uniform(0.3, 0.95)
            xs = np.linspace(center_freq - split, center_freq + split, 200)
            single = NVCenterVoigtSpectrumSamples(
                frequency=np.array([center_freq]),
                fwhm_total=np.array([fwhm_total]),
                lorentz_frac=np.array([lorentz_frac]),
                split=np.array([split]),
                k_np=np.array([k_np]),
                dip_depth=np.array([1.0]),
            )
            g_max = float(1.0 - np.min(model.compute_vectorized_many(xs, single)))
            dip_depth = unit_dip_depth / g_max if g_max > 1e-12 else unit_dip_depth

            typed_params = NVCenterVoigtSpectrum(
                frequency=center_freq,
                fwhm_total=fwhm_total,
                lorentz_frac=lorentz_frac,
                split=split,
                k_np=k_np,
                dip_depth=dip_depth,
            )
            bounds = nv_center_voigt_bounds_for_domain(self.x_min, self.x_max)
            
            # Generate Gaussian priors for all parameters except frequency
            prior_split = rng.gauss(split, 0.1e6)
            prior_fwhm_total = rng.gauss(fwhm_total, 20e3)
            prior_lorentz_frac = rng.gauss(lorentz_frac, 0.05)
            prior_k_np = rng.gauss(k_np, 0.1)
            prior_dip_depth = rng.gauss(dip_depth, 0.01)
            
            bounds["_priors"] = {
                "split": (prior_split, 0.1e6),
                "fwhm_total": (prior_fwhm_total, 20e3),
                "lorentz_frac": (prior_lorentz_frac, 0.05),
                "k_np": (prior_k_np, 0.1),
                "dip_depth": (prior_dip_depth, 0.01),
                "frequency": ("sin^2", np.pi / width),
            }

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)
