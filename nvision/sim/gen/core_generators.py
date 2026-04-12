"""Signal generators that produce typed TrueSignal objects directly."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from nvision.spectra import (
    CompositePeakModel,
    ExponentialDecayModel,
    GaussianModel,
    LorentzianModel,
    NVCenterLorentzianModel,
    NVCenterVoigtModel,
)
from nvision.spectra.composite import CompositeSpectrum
from nvision.spectra.exponential_decay import ExponentialDecaySpectrum
from nvision.spectra.gaussian import GaussianSpectrum
from nvision.spectra.lorentzian import LorentzianSpectrum
from nvision.spectra.nv_center import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    MAX_K_NP,
    MIN_K_NP,
    NVCenterLorentzianSpectrum,
    NVCenterVoigtSpectrum,
    NVCenterVoigtSpectrumSamples,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
)
from nvision.spectra.signal import TrueSignal


def _lorentzian_depth_bounds() -> tuple[float, float]:
    """Upper bound for ``dip_depth`` in :class:`~nvision.spectra.lorentzian.LorentzianModel`."""
    return (0.0, 1.5)


def _lorentzian_depth_draw(rng: random.Random, dip_lo: float, dip_hi: float) -> float:
    """Sample ``dip_depth`` directly."""
    return rng.uniform(dip_lo, dip_hi)


def _true_signal_from_typed(model, typed_params, bounds: dict[str, tuple[float, float]]) -> TrueSignal:
    """Create a backward-compatible TrueSignal from typed model params."""
    return TrueSignal.from_typed(model=model, params=typed_params, bounds=bounds)


@dataclass
class OnePeakCoreGenerator:
    """Generates single-peak signals using core architecture.

    Produces TrueSignal with either Gaussian or Lorentzian model.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    peak_type: str = "gaussian"  # "gaussian" or "lorentzian"

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate a single-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with single peak at random location
        """
        width = self.x_max - self.x_min

        # Random peak position (avoid edges)
        peak_pos = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)

        # Random peak width (5-10% of domain)
        peak_width = rng.uniform(0.05 * width, 0.10 * width)

        # To normalize the signal, we fix amplitudes to exactly 1.0 depth and background appropriately
        if self.peak_type == "gaussian":
            # Gaussian is a bump from 0 to 1
            background = 0.0
            dip_depth = 1.0
            model = GaussianModel()
            typed_params = GaussianSpectrum(
                frequency=peak_pos,
                sigma=peak_width,
                dip_depth=dip_depth,
                background=background,
            )
            bounds = {
                "frequency": (self.x_min, self.x_max),
                "sigma": (width * 0.01, width * 0.2),
                "dip_depth": (0.0, 1.5),
                "background": (0.0, 0.5),
            }
        else:  # lorentzian
            # Lorentzian is a dip from 1 to 0; peak height = dip_depth
            background = 1.0
            dip_depth = 1.0  # This enforces exactly a dip depth of 1.0
            model = LorentzianModel()
            typed_params = LorentzianSpectrum(
                frequency=peak_pos,
                linewidth=peak_width,
                dip_depth=dip_depth,
                background=background,
            )
            bounds = {
                "frequency": (self.x_min, self.x_max),
                "linewidth": (width * 0.01, width * 0.2),
                "dip_depth": (0.0, 1.5),
                "background": (0.5, 1.2),
            }

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)


@dataclass
class NVCenterCoreGenerator:
    """Generates NV center ODMR signals using core architecture.

    Produces TrueSignal with physically accurate NV center signal.
    """

    x_min: float = DEFAULT_NV_CENTER_FREQ_X_MIN  # 2.6 GHz
    x_max: float = DEFAULT_NV_CENTER_FREQ_X_MAX  # 3.1 GHz
    variant: str = "lorentzian"  # "lorentzian" or "voigt"
    zero_field: bool = False  # If True, generate with delta_f_hf = 0 (single peak)

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

        # Random central frequency (avoiding edges)
        if self.zero_field:
            # For zero-field case, can be anywhere
            center_freq = rng.uniform(self.x_min + 0.1 * width, self.x_max - 0.1 * width)
            split = 0.0
        else:
            # For hyperfine-split case, need room for side peaks
            # Generate something roughly centered around the physical values for 14N and 15N (2.16 MHz and 3.03 MHz)
            split = rng.uniform(2.0e6, 3.5e6)
            center_freq = rng.uniform(self.x_min + split + 0.05 * width, self.x_max - split - 0.05 * width)

        # Random linewidth (HWHM for Lorentzian)
        # To ensure the dip strongly returns before the next hyperfine peak,
        # we generate exceptionally sharp lines (50 kHz to 400 kHz HWHM).
        linewidth = rng.uniform(0.05e6, 0.4e6)

        # Random k_np (non-polarization factor)
        k_np = rng.uniform(MIN_K_NP, MAX_K_NP)

        # Normalize NV Center ODMR directly to [0, 1] bounds using exactly 1.0 maximum dip
        background = 1.0

        if self.variant == "lorentzian":
            model = NVCenterLorentzianModel()
            # Scale a desired contrast onto the true peak-shape maximum
            unit_dip_depth = rng.uniform(0.3, 0.95)
            lw2 = linewidth**2
            xs = np.linspace(center_freq - split, center_freq + split, 200)
            g = (
                (lw2 / k_np) / ((xs - (center_freq - split)) ** 2 + lw2)
                + lw2 / ((xs - center_freq) ** 2 + lw2)
                + (lw2 * k_np) / ((xs - (center_freq + split)) ** 2 + lw2)
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
            bounds = nv_center_lorentzian_bounds_for_domain(self.x_min, self.x_max)
        else:  # voigt
            fwhm_lorentz = 2 * linewidth
            fwhm_gauss = fwhm_lorentz * rng.uniform(0.1, 0.3)

            model = NVCenterVoigtModel()
            # Scale a desired contrast onto the true peak-shape maximum
            unit_dip_depth = rng.uniform(0.3, 0.95)
            xs = np.linspace(center_freq - split, center_freq + split, 200)
            single = NVCenterVoigtSpectrumSamples(
                frequency=np.array([center_freq]),
                fwhm_lorentz=np.array([fwhm_lorentz]),
                fwhm_gauss=np.array([fwhm_gauss]),
                split=np.array([split]),
                k_np=np.array([k_np]),
                dip_depth=np.array([1.0]),
                background=np.array([0.0]),
            )
            g_max = float(-np.min(model.compute_vectorized_many(xs, single)))
            dip_depth = unit_dip_depth / g_max if g_max > 1e-12 else unit_dip_depth

            typed_params = NVCenterVoigtSpectrum(
                frequency=center_freq,
                fwhm_lorentz=fwhm_lorentz,
                fwhm_gauss=fwhm_gauss,
                split=split,
                k_np=k_np,
                dip_depth=dip_depth,
                background=background,
            )
            bounds = nv_center_voigt_bounds_for_domain(self.x_min, self.x_max)

        # Lock split to 0 for zero-field case (not a sought parameter)
        if self.zero_field:
            bounds["split"] = (0.0, 0.0)

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)


@dataclass
class TwoPeakCoreGenerator:
    """Generates two-peak signals using core architecture.

    Creates a signal with two separate peaks by combining two single-peak signals.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    peak_type_left: str = "gaussian"
    peak_type_right: str = "gaussian"
    min_separation: float = 0.2  # Minimum separation as fraction of domain

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate two-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with two separated peaks using CompositePeakModel
        """
        width = self.x_max - self.x_min
        min_sep = self.min_separation * width

        # Generate two well-separated peak positions
        peak1_pos = rng.uniform(self.x_min + 0.1 * width, self.x_max - 0.5 * width)
        peak2_pos = rng.uniform(peak1_pos + min_sep, self.x_max - 0.1 * width)

        # Random peak parameters
        peak1_width = rng.uniform(0.02 * width, 0.08 * width)
        peak2_width = rng.uniform(0.02 * width, 0.08 * width)
        peak1_depth = 1.0
        peak2_depth = 1.0
        background = 1.0 if self.peak_type_left == "lorentzian" or self.peak_type_right == "lorentzian" else 0.0

        # Create signal for each peak
        if self.peak_type_left == "gaussian":
            model1 = GaussianModel()
        elif self.peak_type_left == "lorentzian":
            model1 = LorentzianModel()
        else:  # exponential
            model1 = ExponentialDecayModel()

        if self.peak_type_right == "gaussian":
            model2 = GaussianModel()
        elif self.peak_type_right == "lorentzian":
            model2 = LorentzianModel()
        else:  # exponential
            model2 = ExponentialDecayModel()

        # Create composite model
        composite_model = CompositePeakModel(
            [
                ("peak1", model1),
                ("peak2", model2),
            ]
        )

        if self.peak_type_left == "gaussian":
            peak1_typed = GaussianSpectrum(peak1_pos, peak1_width, peak1_depth, background / 2)
            left_width_key = "peak1_sigma"
        elif self.peak_type_left == "lorentzian":
            peak1_typed = LorentzianSpectrum(peak1_pos, peak1_width, peak1_depth, background / 2)
            left_width_key = "peak1_linewidth"
        else:
            peak1_typed = ExponentialDecaySpectrum(peak1_width * 5, peak1_depth, background / 2)
            left_width_key = "peak1_decay_rate"

        if self.peak_type_right == "gaussian":
            peak2_typed = GaussianSpectrum(peak2_pos, peak2_width, peak2_depth, background / 2)
            right_width_key = "peak2_sigma"
        elif self.peak_type_right == "lorentzian":
            peak2_typed = LorentzianSpectrum(peak2_pos, peak2_width, peak2_depth, background / 2)
            right_width_key = "peak2_linewidth"
        else:
            peak2_typed = ExponentialDecaySpectrum(peak2_width * 5, peak2_depth, background / 2)
            right_width_key = "peak2_decay_rate"

        left_width_bounds = (
            (width * 0.1, width * 2.0) if left_width_key.endswith("decay_rate") else (width * 0.01, width * 0.2)
        )
        right_width_bounds = (
            (width * 0.1, width * 2.0) if right_width_key.endswith("decay_rate") else (width * 0.01, width * 0.2)
        )
        bounds = {
            "peak1_frequency": (self.x_min, self.x_max),
            left_width_key: left_width_bounds,
            "peak1_dip_depth": (0.0, 1.5),
            "peak1_background": (0.0, 0.5),
            "peak2_frequency": (self.x_min, self.x_max),
            right_width_key: right_width_bounds,
            "peak2_dip_depth": (0.0, 1.5),
            "peak2_background": (0.0, 0.5),
        }

        typed_params = CompositeSpectrum(peaks=(peak1_typed, peak2_typed))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)


@dataclass
class MultiPeakCoreGenerator:
    """Generates multi-peak signals using core architecture.

    Creates a signal with N separate peaks.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    count: int = 3
    peak_types: list[str] | None = None  # ["gaussian", "lorentzian", ...]
    min_separation: float = 0.1  # Minimum separation as fraction of domain

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate multi-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with N peaks using CompositePeakModel
        """
        width = self.x_max - self.x_min
        min_sep = self.min_separation * width

        # Determine peak types
        if self.peak_types is None:
            peak_types = ["gaussian"] * self.count
        else:
            peak_types = self.peak_types[: self.count]
            # Pad with gaussian if not enough
            while len(peak_types) < self.count:
                peak_types.append("gaussian")

        # Generate well-separated peak positions
        positions = []
        max_attempts = 1000
        for _ in range(self.count):
            for _ in range(max_attempts):
                pos = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
                if not positions or all(abs(pos - p) >= min_sep for p in positions):
                    positions.append(pos)
                    break

        # Sort positions
        positions.sort()

        # Create signal and typed parameters
        models = []
        typed_peak_params: list[object] = []
        bounds: dict[str, tuple[float, float]] = {}
        background = 1.0 if any(pt == "lorentzian" for pt in peak_types) else 0.0

        for i, (pos, peak_type) in enumerate(zip(positions, peak_types, strict=False)):
            prefix = f"peak{i + 1}"

            # Random parameters
            peak_width = rng.uniform(0.02 * width, 0.08 * width)
            dip_depth = 1.0  # Normalized depth

            # Create model
            if peak_type == "gaussian":
                model = GaussianModel()
                typed_peak_params.append(
                    GaussianSpectrum(
                        frequency=pos,
                        sigma=peak_width,
                        dip_depth=dip_depth,
                        background=background / self.count,
                    )
                )
                bounds[f"{prefix}_frequency"] = (self.x_min, self.x_max)
                bounds[f"{prefix}_sigma"] = (width * 0.01, width * 0.2)
                bounds[f"{prefix}_dip_depth"] = (0.0, 1.5)
                bounds[f"{prefix}_background"] = (0.0, 0.5)
            elif peak_type == "lorentzian":
                model = LorentzianModel()
                typed_peak_params.append(
                    LorentzianSpectrum(
                        frequency=pos,
                        linewidth=peak_width,
                        dip_depth=dip_depth,
                        background=1.0 - background / self.count,
                    )
                )
                bounds[f"{prefix}_frequency"] = (self.x_min, self.x_max)
                bounds[f"{prefix}_linewidth"] = (width * 0.01, width * 0.2)
                bounds[f"{prefix}_dip_depth"] = (0.0, 1.5)
                bounds[f"{prefix}_background"] = (0.0, 0.5)
            else:  # exponential
                model = ExponentialDecayModel()
                typed_peak_params.append(
                    ExponentialDecaySpectrum(
                        decay_rate=peak_width * 5,
                        dip_depth=dip_depth,
                        background=background / self.count,
                    )
                )
                bounds[f"{prefix}_decay_rate"] = (width * 0.1, width * 2.0)
                bounds[f"{prefix}_dip_depth"] = (0.0, 1.5)
                bounds[f"{prefix}_background"] = (0.0, 0.5)

            models.append((prefix, model))

        composite_model = CompositePeakModel(models)
        typed_params = CompositeSpectrum(peaks=tuple(typed_peak_params))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)


@dataclass
class SymmetricTwoPeakCoreGenerator:
    """Generates symmetric two-peak signals using core architecture.

    Creates two peaks symmetrically placed around a center point.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    center: float = 0.5
    sep_frac: float = 0.2  # Separation as fraction of domain
    peak_type: str = "gaussian"

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate symmetric two-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with two symmetric peaks
        """
        width = self.x_max - self.x_min
        delta = 0.5 * self.sep_frac * width

        # Symmetric positions
        left_pos = self.center - delta
        right_pos = self.center + delta

        # Same parameters for both peaks (symmetric)
        peak_width = rng.uniform(0.02 * width, 0.06 * width)
        dip_depth = 1.0
        background = 1.0 if self.peak_type == "lorentzian" else 0.0

        # Create model
        if self.peak_type == "gaussian":
            model_left = GaussianModel()
            model_right = GaussianModel()
            param_key = "sigma"
        elif self.peak_type == "lorentzian":
            model_left = LorentzianModel()
            model_right = LorentzianModel()
            param_key = "linewidth"
        else:  # exponential
            model_left = ExponentialDecayModel()
            model_right = ExponentialDecayModel()
            param_key = "decay_rate"

        composite_model = CompositePeakModel(
            [
                ("peak1", model_left),
                ("peak2", model_right),
            ]
        )

        # Build typed parameters

        if self.peak_type in ["gaussian", "lorentzian"]:
            if self.peak_type == "gaussian":
                peak1_typed = GaussianSpectrum(left_pos, peak_width, dip_depth, background / 2)
                peak2_typed = GaussianSpectrum(right_pos, peak_width, dip_depth, background / 2)
            else:
                peak1_typed = LorentzianSpectrum(left_pos, peak_width, dip_depth, background / 2)
                peak2_typed = LorentzianSpectrum(right_pos, peak_width, dip_depth, background / 2)
            bounds = {
                "peak1_frequency": (self.x_min, self.x_max),
                f"peak1_{param_key}": (width * 0.01, width * 0.2),
                "peak1_dip_depth": (0.0, 1.5),
                "peak1_background": (0.0, 0.5),
                "peak2_frequency": (self.x_min, self.x_max),
                f"peak2_{param_key}": (width * 0.01, width * 0.2),
                "peak2_dip_depth": (0.0, 1.5),
                "peak2_background": (0.0, 0.5),
            }
        else:  # exponential
            peak1_typed = ExponentialDecaySpectrum(peak_width * 5, dip_depth, background / 2)
            peak2_typed = ExponentialDecaySpectrum(peak_width * 5, dip_depth, background / 2)
            bounds = {
                "peak1_decay_rate": (width * 0.1, width * 2.0),
                "peak1_dip_depth": (0.0, 1.5),
                "peak1_background": (0.0, 0.5),
                "peak2_decay_rate": (width * 0.1, width * 2.0),
                "peak2_dip_depth": (0.0, 1.5),
                "peak2_background": (0.0, 0.5),
            }

        typed_params = CompositeSpectrum(peaks=(peak1_typed, peak2_typed))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)
