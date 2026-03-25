"""Signal generators that produce TrueSignal directly.

These generators create TrueSignal objects with proper SignalModel implementations
and Parameter objects, integrating deeply with the core architecture.
"""

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
from nvision.signal.signal import Parameter, TrueSignal


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
            parameters = [
                Parameter(name="frequency", bounds=(self.x_min, self.x_max), value=peak_pos),
                Parameter(name="sigma", bounds=(width * 0.01, width * 0.2), value=peak_width),
                Parameter(name="amplitude", bounds=(0.0, 1.5), value=amplitude),
                Parameter(name="background", bounds=(0.0, 0.5), value=background),
            ]
        else:  # lorentzian
            # Lorentzian is a dip from 1 to 0; dip depth = amplitude / linewidth²
            background = 1.0
            amp_hi = _lorentzian_amplitude_bounds(width)[1]
            amplitude = peak_width**2  # This enforces exactly a dip depth of 1.0
            model = LorentzianModel()
            parameters = [
                Parameter(name="frequency", bounds=(self.x_min, self.x_max), value=peak_pos),
                Parameter(name="linewidth", bounds=(width * 0.01, width * 0.2), value=peak_width),
                Parameter(name="amplitude", bounds=(0.0, amp_hi), value=amplitude),
                Parameter(name="background", bounds=(0.5, 1.2), value=background),
            ]

        return TrueSignal(model=model, parameters=parameters)


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
            temp_params = [
                Parameter(name="frequency", bounds=(self.x_min, self.x_max), value=center_freq),
                Parameter(name="linewidth", bounds=(width * 0.001, width * 0.05), value=linewidth),
                Parameter(name="split", bounds=(0.0, width * 0.5), value=split),
                Parameter(name="k_np", bounds=(MIN_K_NP, MAX_K_NP), value=k_np),
                Parameter(name="amplitude", bounds=(0.0, max(1.0, base_amp * 2.0)), value=base_amp),
                Parameter(name="background", bounds=(0.0, 1.0), value=0.0),
            ]
            tv = tuple(p.value for p in temp_params)
            min_val = min(
                NVCenterLorentzianModel.compute_nvcenter_lorentzian_model(center_freq - split, *tv),
                NVCenterLorentzianModel.compute_nvcenter_lorentzian_model(center_freq, *tv),
                NVCenterLorentzianModel.compute_nvcenter_lorentzian_model(center_freq + split, *tv),
            )
            amplitude = base_amp / abs(min_val) if abs(min_val) > 1e-30 else base_amp
            amp_hi = max(0.6 * (MAX_NV_CENTER_OMEGA * width) ** 2, amplitude * 2.0)

            parameters = [
                Parameter(name="frequency", bounds=(self.x_min, self.x_max), value=center_freq),
                Parameter(name="linewidth", bounds=(width * 0.001, width * 0.05), value=linewidth),
                Parameter(name="split", bounds=(0.0, width * 0.5), value=split),
                Parameter(name="k_np", bounds=(MIN_K_NP, MAX_K_NP), value=k_np),
                Parameter(name="amplitude", bounds=(0.0, amp_hi), value=amplitude),
                Parameter(name="background", bounds=(0.95, 1.05), value=background),
            ]
        else:  # voigt
            fwhm_lorentz = 2 * linewidth
            fwhm_gauss = fwhm_lorentz * rng.uniform(0.1, 0.3)

            model = NVCenterVoigtModel()
            base_amp = linewidth**2
            temp_params = [
                Parameter(name="frequency", bounds=(self.x_min, self.x_max), value=center_freq),
                Parameter(name="fwhm_lorentz", bounds=(width * 0.001, width * 0.1), value=fwhm_lorentz),
                Parameter(name="fwhm_gauss", bounds=(width * 0.0001, width * 0.05), value=fwhm_gauss),
                Parameter(name="split", bounds=(0.0, width * 0.5), value=split),
                Parameter(name="k_np", bounds=(MIN_K_NP, MAX_K_NP), value=k_np),
                Parameter(name="amplitude", bounds=(0.0, max(1.0, base_amp * 2.0)), value=base_amp),
                Parameter(name="background", bounds=(0.0, 1.0), value=0.0),
            ]
            tv = tuple(p.value for p in temp_params)
            min_val = min(
                model.compute_nvcenter_voigt_model(center_freq - split, *tv),
                model.compute_nvcenter_voigt_model(center_freq, *tv),
                model.compute_nvcenter_voigt_model(center_freq + split, *tv),
            )
            voigt_amplitude = base_amp / abs(min_val) if abs(min_val) > 1e-30 else base_amp
            voigt_amp_hi = max(0.6 * (MAX_NV_CENTER_OMEGA * width) ** 2, voigt_amplitude * 2.0)

            parameters = [
                Parameter(name="frequency", bounds=(self.x_min, self.x_max), value=center_freq),
                Parameter(name="fwhm_lorentz", bounds=(width * 0.001, width * 0.1), value=fwhm_lorentz),
                Parameter(name="fwhm_gauss", bounds=(width * 0.0001, width * 0.05), value=fwhm_gauss),
                Parameter(name="split", bounds=(0.0, width * 0.5), value=split),
                Parameter(name="k_np", bounds=(MIN_K_NP, MAX_K_NP), value=k_np),
                Parameter(name="amplitude", bounds=(0.0, voigt_amp_hi), value=voigt_amplitude),
                Parameter(name="background", bounds=(0.95, 1.05), value=background),
            ]

        return TrueSignal(model=model, parameters=parameters)


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
            param_names1 = ["frequency", "sigma", "amplitude", "background"]
            widths1 = [peak1_width]
        elif self.peak_type_left == "lorentzian":
            model1 = LorentzianModel()
            param_names1 = ["frequency", "linewidth", "amplitude", "background"]
            widths1 = [peak1_width]
        else:  # exponential
            model1 = ExponentialDecayModel()
            param_names1 = ["decay_rate", "amplitude", "background"]
            widths1 = []

        if self.peak_type_right == "gaussian":
            model2 = GaussianModel()
            param_names2 = ["frequency", "sigma", "amplitude", "background"]
            widths2 = [peak2_width]
        elif self.peak_type_right == "lorentzian":
            model2 = LorentzianModel()
            param_names2 = ["frequency", "linewidth", "amplitude", "background"]
            widths2 = [peak2_width]
        else:  # exponential
            model2 = ExponentialDecayModel()
            param_names2 = ["decay_rate", "amplitude", "background"]
            widths2 = []

        # Create composite model
        composite_model = CompositePeakModel(
            [
                ("peak1", model1),
                ("peak2", model2),
            ]
        )

        # Build parameters for both peaks
        parameters = []

        # Peak 1 parameters
        parameters.append(Parameter(name="peak1_frequency", bounds=(self.x_min, self.x_max), value=peak1_pos))
        if widths1:
            parameters.append(
                Parameter(
                    name="peak1_sigma" if "sigma" in param_names1 else "peak1_linewidth",
                    bounds=(width * 0.01, width * 0.2),
                    value=peak1_width,
                )
            )
        parameters.append(
            Parameter(
                name="peak1_amplitude",
                bounds=(0.0, lorentz_amp_hi) if self.peak_type_left == "lorentzian" else (0.0, 1.5),
                value=peak1_amp,
            )
        )
        parameters.append(Parameter(name="peak1_background", bounds=(0.0, 0.5), value=background / 2))

        # Peak 2 parameters
        parameters.append(Parameter(name="peak2_frequency", bounds=(self.x_min, self.x_max), value=peak2_pos))
        if widths2:
            parameters.append(
                Parameter(
                    name="peak2_sigma" if "sigma" in param_names2 else "peak2_linewidth",
                    bounds=(width * 0.01, width * 0.2),
                    value=peak2_width,
                )
            )
        parameters.append(
            Parameter(
                name="peak2_amplitude",
                bounds=(0.0, lorentz_amp_hi) if self.peak_type_right == "lorentzian" else (0.0, 1.5),
                value=peak2_amp,
            )
        )
        parameters.append(Parameter(name="peak2_background", bounds=(0.0, 0.5), value=background / 2))

        return TrueSignal(model=composite_model, parameters=parameters)


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

        # Create signal and parameters
        models = []
        parameters = []
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
                parameters.extend(
                    [
                        Parameter(name=f"{prefix}_frequency", bounds=(self.x_min, self.x_max), value=pos),
                        Parameter(name=f"{prefix}_sigma", bounds=(width * 0.01, width * 0.2), value=peak_width),
                        Parameter(name=f"{prefix}_amplitude", bounds=(0.0, 1.5), value=amplitude),
                        Parameter(name=f"{prefix}_background", bounds=(0.0, 0.5), value=background / self.count),
                    ]
                )
            elif peak_type == "lorentzian":
                model = LorentzianModel()
                parameters.extend(
                    [
                        Parameter(name=f"{prefix}_frequency", bounds=(self.x_min, self.x_max), value=pos),
                        Parameter(name=f"{prefix}_linewidth", bounds=(width * 0.01, width * 0.2), value=peak_width),
                        Parameter(name=f"{prefix}_amplitude", bounds=(0.0, lorentz_amp_hi), value=amplitude),
                        Parameter(name=f"{prefix}_background", bounds=(0.0, 0.5), value=1.0 - background / self.count),
                    ]
                )
            else:  # exponential
                model = ExponentialDecayModel()
                parameters.extend(
                    [
                        Parameter(name=f"{prefix}_decay_rate", bounds=(width * 0.1, width * 2.0), value=peak_width * 5),
                        Parameter(name=f"{prefix}_amplitude", bounds=(0.0, 1.5), value=amplitude),
                        Parameter(name=f"{prefix}_background", bounds=(0.0, 0.5), value=background / self.count),
                    ]
                )

            models.append((prefix, model))

        composite_model = CompositePeakModel(models)
        return TrueSignal(model=composite_model, parameters=parameters)


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

        # Build parameters
        parameters = []

        if self.peak_type in ["gaussian", "lorentzian"]:
            amp_bounds = (0.0, lorentz_amp_hi) if self.peak_type == "lorentzian" else (0.0, 1.5)
            # Peak 1 (left)
            parameters.extend(
                [
                    Parameter(name="peak1_frequency", bounds=(self.x_min, self.x_max), value=left_pos),
                    Parameter(name=f"peak1_{param_key}", bounds=(width * 0.01, width * 0.2), value=peak_width),
                    Parameter(name="peak1_amplitude", bounds=amp_bounds, value=amplitude),
                    Parameter(name="peak1_background", bounds=(0.0, 0.5), value=background / 2),
                ]
            )

            # Peak 2 (right)
            parameters.extend(
                [
                    Parameter(name="peak2_frequency", bounds=(self.x_min, self.x_max), value=right_pos),
                    Parameter(name=f"peak2_{param_key}", bounds=(width * 0.01, width * 0.2), value=peak_width),
                    Parameter(name="peak2_amplitude", bounds=amp_bounds, value=amplitude),
                    Parameter(name="peak2_background", bounds=(0.0, 0.5), value=background / 2),
                ]
            )
        else:  # exponential
            parameters.extend(
                [
                    Parameter(name="peak1_decay_rate", bounds=(width * 0.1, width * 2.0), value=peak_width * 5),
                    Parameter(name="peak1_amplitude", bounds=(0.0, 1.5), value=amplitude),
                    Parameter(name="peak1_background", bounds=(0.0, 0.5), value=background / 2),
                    Parameter(name="peak2_decay_rate", bounds=(width * 0.1, width * 2.0), value=peak_width * 5),
                    Parameter(name="peak2_amplitude", bounds=(0.0, 1.5), value=amplitude),
                    Parameter(name="peak2_background", bounds=(0.0, 0.5), value=background / 2),
                ]
            )

        return TrueSignal(model=composite_model, parameters=parameters)
