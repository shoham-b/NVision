"""Signal generators that produce typed TrueSignal objects directly."""

from __future__ import annotations

import random
from dataclasses import dataclass

from nvision.signal import (
    MAX_K_NP,
    MAX_NV_CENTER_DELTA,
    MAX_NV_CENTER_OMEGA,
    MIN_K_NP,
    MIN_NV_CENTER_DELTA,
    MIN_NV_CENTER_OMEGA,
    CompositePeakModel,
    ExponentialDecayModel,
    GaussianModel,
    LorentzianModel,
    NVCenterLorentzianModel,
    NVCenterVoigtModel,
)
from nvision.signal.composite import CompositeParams
from nvision.signal.exponential_decay import ExponentialDecayParams
from nvision.signal.gaussian import GaussianParams
from nvision.signal.lorentzian import LorentzianParams
from nvision.signal.nv_center import NVCenterLorentzianParams, NVCenterVoigtParams
from nvision.signal.signal import TrueSignal


def _lorentzian_amplitude_bounds(domain_width: float) -> tuple[float, float]:
    """Upper bound for ``amplitude`` in :class:`~nvision.signal.lorentzian.LorentzianModel`.

    There ``dip_depth = amplitude / linewidth²`` with linewidth up to ~0.2·domain_width
    in our generators, and ``dip_depth`` ≤ 1.
    """
    w_hi = domain_width * 0.2
    return (0.0, w_hi * w_hi)


def _lorentzian_amplitude_from_dip(rng: random.Random, linewidth: float, dip_lo: float, dip_hi: float) -> float:
    """Sample ``amplitude = dip_depth * linewidth²`` (see LorentzianModel docstring)."""
    dip = rng.uniform(dip_lo, dip_hi)
    return dip * linewidth**2


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
            amplitude = 1.0
            model = GaussianModel()
            typed_params = GaussianParams(
                frequency=peak_pos,
                sigma=peak_width,
                amplitude=amplitude,
                background=background,
            )
            bounds = {
                "frequency": (self.x_min, self.x_max),
                "sigma": (width * 0.01, width * 0.2),
                "amplitude": (0.0, 1.5),
                "background": (0.0, 0.5),
            }
        else:  # lorentzian
            # Lorentzian is a dip from 1 to 0; dip depth = amplitude / linewidth²
            background = 1.0
            amp_hi = _lorentzian_amplitude_bounds(width)[1]
            amplitude = peak_width**2  # This enforces exactly a dip depth of 1.0
            model = LorentzianModel()
            typed_params = LorentzianParams(
                frequency=peak_pos,
                linewidth=peak_width,
                amplitude=amplitude,
                background=background,
            )
            bounds = {
                "frequency": (self.x_min, self.x_max),
                "linewidth": (width * 0.01, width * 0.2),
                "amplitude": (0.0, amp_hi),
                "background": (0.5, 1.2),
            }

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)


@dataclass
class NVCenterCoreGenerator:
    """Generates NV center ODMR signals using core architecture.

    Produces TrueSignal with physically accurate NV center signal.
    """

    x_min: float = 2.6e9  # 2.6 GHz
    x_max: float = 3.1e9  # 3.1 GHz
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
            split = rng.uniform(MIN_NV_CENTER_DELTA * width, MAX_NV_CENTER_DELTA * width)
            center_freq = rng.uniform(self.x_min + split + 0.05 * width, self.x_max - split - 0.05 * width)

        # Random linewidth (HWHM for Lorentzian)
        linewidth = rng.uniform(MIN_NV_CENTER_OMEGA * width, MAX_NV_CENTER_OMEGA * width)

        # Random k_np (non-polarization factor)
        k_np = rng.uniform(MIN_K_NP, MAX_K_NP)

        # Normalize NV Center ODMR directly to [0, 1] bounds using exactly 1.0 maximum dip
        # We calculate exact depth per unit amplitude by checking the three specific peak locations.
        background = 1.0

        if self.variant == "lorentzian":
            model = NVCenterLorentzianModel()
            base_amp = linewidth**2
            temp_params = NVCenterLorentzianParams(
                frequency=center_freq,
                linewidth=linewidth,
                split=split,
                k_np=k_np,
                amplitude=base_amp,
                background=0.0,
            )
            tv = model.spec.pack_params(temp_params)
            min_val = min(
                NVCenterLorentzianModel.compute_nvcenter_lorentzian_model(center_freq - split, *tv),
                NVCenterLorentzianModel.compute_nvcenter_lorentzian_model(center_freq, *tv),
                NVCenterLorentzianModel.compute_nvcenter_lorentzian_model(center_freq + split, *tv),
            )
            amplitude = base_amp / abs(min_val) if abs(min_val) > 1e-30 else base_amp
            amp_hi = max(0.6 * (MAX_NV_CENTER_OMEGA * width) ** 2, amplitude * 2.0)

            typed_params = NVCenterLorentzianParams(
                frequency=center_freq,
                linewidth=linewidth,
                split=split,
                k_np=k_np,
                amplitude=amplitude,
                background=background,
            )
            bounds = {
                "frequency": (self.x_min, self.x_max),
                "linewidth": (width * 0.001, width * 0.05),
                "split": (0.0, width * 0.5),
                "k_np": (MIN_K_NP, MAX_K_NP),
                "amplitude": (0.0, amp_hi),
                "background": (0.95, 1.05),
            }
        else:  # voigt
            fwhm_lorentz = 2 * linewidth
            fwhm_gauss = fwhm_lorentz * rng.uniform(0.1, 0.3)

            model = NVCenterVoigtModel()
            base_amp = linewidth**2
            temp_params = NVCenterVoigtParams(
                frequency=center_freq,
                fwhm_lorentz=fwhm_lorentz,
                fwhm_gauss=fwhm_gauss,
                split=split,
                k_np=k_np,
                amplitude=base_amp,
                background=0.0,
            )
            tv = model.spec.pack_params(temp_params)
            min_val = min(
                model.compute_nvcenter_voigt_model(center_freq - split, *tv),
                model.compute_nvcenter_voigt_model(center_freq, *tv),
                model.compute_nvcenter_voigt_model(center_freq + split, *tv),
            )
            voigt_amplitude = base_amp / abs(min_val) if abs(min_val) > 1e-30 else base_amp
            voigt_amp_hi = max(0.6 * (MAX_NV_CENTER_OMEGA * width) ** 2, voigt_amplitude * 2.0)

            typed_params = NVCenterVoigtParams(
                frequency=center_freq,
                fwhm_lorentz=fwhm_lorentz,
                fwhm_gauss=fwhm_gauss,
                split=split,
                k_np=k_np,
                amplitude=voigt_amplitude,
                background=background,
            )
            bounds = {
                "frequency": (self.x_min, self.x_max),
                "fwhm_lorentz": (width * 0.001, width * 0.1),
                "fwhm_gauss": (width * 0.0001, width * 0.05),
                "split": (0.0, width * 0.5),
                "k_np": (MIN_K_NP, MAX_K_NP),
                "amplitude": (0.0, voigt_amp_hi),
                "background": (0.95, 1.05),
            }

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
        peak1_amp = peak1_width**2 if self.peak_type_left == "lorentzian" else 1.0
        peak2_amp = peak2_width**2 if self.peak_type_right == "lorentzian" else 1.0
        background = 1.0 if self.peak_type_left == "lorentzian" or self.peak_type_right == "lorentzian" else 0.0
        lorentz_amp_hi = _lorentzian_amplitude_bounds(width)[1] * 2.0

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
            peak1_typed = GaussianParams(peak1_pos, peak1_width, peak1_amp, background / 2)
            left_width_key = "peak1_sigma"
        elif self.peak_type_left == "lorentzian":
            peak1_typed = LorentzianParams(peak1_pos, peak1_width, peak1_amp, background / 2)
            left_width_key = "peak1_linewidth"
        else:
            peak1_typed = ExponentialDecayParams(peak1_width * 5, peak1_amp, background / 2)
            left_width_key = "peak1_decay_rate"

        if self.peak_type_right == "gaussian":
            peak2_typed = GaussianParams(peak2_pos, peak2_width, peak2_amp, background / 2)
            right_width_key = "peak2_sigma"
        elif self.peak_type_right == "lorentzian":
            peak2_typed = LorentzianParams(peak2_pos, peak2_width, peak2_amp, background / 2)
            right_width_key = "peak2_linewidth"
        else:
            peak2_typed = ExponentialDecayParams(peak2_width * 5, peak2_amp, background / 2)
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
            "peak1_amplitude": (0.0, lorentz_amp_hi) if self.peak_type_left == "lorentzian" else (0.0, 1.5),
            "peak1_background": (0.0, 0.5),
            "peak2_frequency": (self.x_min, self.x_max),
            right_width_key: right_width_bounds,
            "peak2_amplitude": (0.0, lorentz_amp_hi) if self.peak_type_right == "lorentzian" else (0.0, 1.5),
            "peak2_background": (0.0, 0.5),
        }

        typed_params = CompositeParams(peaks=(peak1_typed, peak2_typed))
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
        lorentz_amp_hi = _lorentzian_amplitude_bounds(width)[1] * float(self.count)

        for i, (pos, peak_type) in enumerate(zip(positions, peak_types, strict=False)):
            prefix = f"peak{i + 1}"

            # Random parameters
            peak_width = rng.uniform(0.02 * width, 0.08 * width)
            amplitude = peak_width**2 if peak_type == "lorentzian" else 1.0

            # Create model
            if peak_type == "gaussian":
                model = GaussianModel()
                typed_peak_params.append(
                    GaussianParams(
                        frequency=pos,
                        sigma=peak_width,
                        amplitude=amplitude,
                        background=background / self.count,
                    )
                )
                bounds[f"{prefix}_frequency"] = (self.x_min, self.x_max)
                bounds[f"{prefix}_sigma"] = (width * 0.01, width * 0.2)
                bounds[f"{prefix}_amplitude"] = (0.0, 1.5)
                bounds[f"{prefix}_background"] = (0.0, 0.5)
            elif peak_type == "lorentzian":
                model = LorentzianModel()
                typed_peak_params.append(
                    LorentzianParams(
                        frequency=pos,
                        linewidth=peak_width,
                        amplitude=amplitude,
                        background=1.0 - background / self.count,
                    )
                )
                bounds[f"{prefix}_frequency"] = (self.x_min, self.x_max)
                bounds[f"{prefix}_linewidth"] = (width * 0.01, width * 0.2)
                bounds[f"{prefix}_amplitude"] = (0.0, lorentz_amp_hi)
                bounds[f"{prefix}_background"] = (0.0, 0.5)
            else:  # exponential
                model = ExponentialDecayModel()
                typed_peak_params.append(
                    ExponentialDecayParams(
                        decay_rate=peak_width * 5,
                        amplitude=amplitude,
                        background=background / self.count,
                    )
                )
                bounds[f"{prefix}_decay_rate"] = (width * 0.1, width * 2.0)
                bounds[f"{prefix}_amplitude"] = (0.0, 1.5)
                bounds[f"{prefix}_background"] = (0.0, 0.5)

            models.append((prefix, model))

        composite_model = CompositePeakModel(models)
        typed_params = CompositeParams(peaks=tuple(typed_peak_params))
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
        amplitude = peak_width**2 if self.peak_type == "lorentzian" else 1.0
        background = 1.0 if self.peak_type == "lorentzian" else 0.0
        lorentz_amp_hi = _lorentzian_amplitude_bounds(width)[1] * 2.0

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
            amp_bounds = (0.0, lorentz_amp_hi) if self.peak_type == "lorentzian" else (0.0, 1.5)
            if self.peak_type == "gaussian":
                peak1_typed = GaussianParams(left_pos, peak_width, amplitude, background / 2)
                peak2_typed = GaussianParams(right_pos, peak_width, amplitude, background / 2)
            else:
                peak1_typed = LorentzianParams(left_pos, peak_width, amplitude, background / 2)
                peak2_typed = LorentzianParams(right_pos, peak_width, amplitude, background / 2)
            bounds = {
                "peak1_frequency": (self.x_min, self.x_max),
                f"peak1_{param_key}": (width * 0.01, width * 0.2),
                "peak1_amplitude": amp_bounds,
                "peak1_background": (0.0, 0.5),
                "peak2_frequency": (self.x_min, self.x_max),
                f"peak2_{param_key}": (width * 0.01, width * 0.2),
                "peak2_amplitude": amp_bounds,
                "peak2_background": (0.0, 0.5),
            }
        else:  # exponential
            peak1_typed = ExponentialDecayParams(peak_width * 5, amplitude, background / 2)
            peak2_typed = ExponentialDecayParams(peak_width * 5, amplitude, background / 2)
            bounds = {
                "peak1_decay_rate": (width * 0.1, width * 2.0),
                "peak1_amplitude": (0.0, 1.5),
                "peak1_background": (0.0, 0.5),
                "peak2_decay_rate": (width * 0.1, width * 2.0),
                "peak2_amplitude": (0.0, 1.5),
                "peak2_background": (0.0, 0.5),
            }

        typed_params = CompositeParams(peaks=(peak1_typed, peak2_typed))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)
