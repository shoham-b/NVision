"""NV center signal generator."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from nvision.spectra.nv_center import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    MAX_K_NP,
    MIN_K_NP,
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
    narrow_signal: bool = False  # if True, use exceptionally narrow linewidths and splitting

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

        # For hyperfine-split case, need room for side peaks
        # Generate something roughly centered around the physical values for 14N and 15N (2.16 MHz and 3.03 MHz)
        if self.narrow_signal:
            split = rng.uniform(0.5e6, 1.2e6)
        else:
            split = rng.uniform(2.0e6, 3.5e6)

        usable_lo = self.x_min + split + 0.05 * width
        usable_hi = self.x_max - split - 0.05 * width
        if self.center_freq_fraction is not None:
            frac = max(0.0, min(1.0, self.center_freq_fraction))
            mid = (usable_lo + usable_hi) / 2.0
            half_span = (usable_hi - usable_lo) * frac / 2.0
            center_freq = rng.uniform(mid - half_span, mid + half_span)
        else:
            center_freq = rng.uniform(usable_lo, usable_hi)

        # Random linewidth (HWHM for Lorentzian)
        if self.narrow_signal:
            # Extremely sharp lines for testing Bayesian limits (10 kHz to 50 kHz HWHM)
            linewidth = rng.uniform(0.01e6, 0.05e6)
        else:
            # Standard sharp lines (50 kHz to 400 kHz HWHM).
            linewidth = rng.uniform(0.05e6, 0.4e6)

        # Random k_np (non-polarization factor)
        k_np = rng.uniform(MIN_K_NP, MAX_K_NP)

        # Normalize NV Center ODMR directly to [0, 1] bounds using exactly 1.0 maximum dip
        background = 1.0

        if self.variant == "lorentzian":
            unit_dip_depth = rng.uniform(0.3, 0.95)
            lw2 = linewidth**2
            model = NVCenterLorentzianModel()
            # Scale a desired contrast onto the true peak-shape maximum
            xs = np.linspace(center_freq - split, center_freq + split, 200)
            g = (
                (lw2 / k_np**2) / ((xs - (center_freq - split)) ** 2 + lw2)
                + (lw2 / k_np) / ((xs - center_freq) ** 2 + lw2)
                + lw2 / ((xs - (center_freq + split)) ** 2 + lw2)
            )
            dip_depth = unit_dip_depth / float(g.max())
            typed_params = NVCenterLorentzianSpectrum(
                frequency=center_freq,
                linewidth=linewidth,
                split=split,
                k_np=k_np,
                dip_depth=dip_depth,
                background=background,
            )
            bounds = nv_center_lorentzian_bounds_for_domain(
                self.x_min, self.x_max, narrow=self.narrow_signal, true_params=typed_params
            )
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
                background=np.array([0.0]),
            )
            g_max = float(-np.min(model.compute_vectorized_many(xs, single)))
            dip_depth = unit_dip_depth / g_max if g_max > 1e-12 else unit_dip_depth

            typed_params = NVCenterVoigtSpectrum(
                frequency=center_freq,
                fwhm_total=fwhm_total,
                lorentz_frac=lorentz_frac,
                split=split,
                k_np=k_np,
                dip_depth=dip_depth,
                background=background,
            )
            bounds = nv_center_voigt_bounds_for_domain(
                self.x_min, self.x_max, narrow=self.narrow_signal, true_params=typed_params
            )

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)
