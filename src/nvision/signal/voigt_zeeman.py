"""Voigt-broadened NV center model with Zeeman splitting."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.parameter import Parameter
from nvision.signal.dtypes import FLOAT_DTYPE
from nvision.signal.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class VoigtZeemanParams:
    frequency: float
    linewidth: float
    fwhm_gauss: float
    split: float
    k_np: float
    amplitude: float
    background: float


@dataclass(frozen=True)
class VoigtZeemanSampleParams:
    frequency: np.ndarray
    linewidth: np.ndarray
    fwhm_gauss: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    amplitude: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class VoigtZeemanUncertaintyParams:
    frequency: float
    linewidth: float
    fwhm_gauss: float
    split: float
    k_np: float
    amplitude: float
    background: float


class _VoigtZeemanSpec(ParamSpec[VoigtZeemanParams, VoigtZeemanSampleParams, VoigtZeemanUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "linewidth", "fwhm_gauss", "split", "k_np", "amplitude", "background")

    @property
    def dim(self) -> int:
        return 7

    def unpack_params(self, values) -> VoigtZeemanParams:
        f, lw, fg, s, k, a, b = values
        return VoigtZeemanParams(float(f), float(lw), float(fg), float(s), float(k), float(a), float(b))

    def pack_params(self, params: VoigtZeemanParams) -> tuple[float, ...]:
        return (
            float(params.frequency),
            float(params.linewidth),
            float(params.fwhm_gauss),
            float(params.split),
            float(params.k_np),
            float(params.amplitude),
            float(params.background),
        )

    def unpack_uncertainty(self, values) -> VoigtZeemanUncertaintyParams:
        f, lw, fg, s, k, a, b = values
        return VoigtZeemanUncertaintyParams(float(f), float(lw), float(fg), float(s), float(k), float(a), float(b))

    def pack_uncertainty(self, u: VoigtZeemanUncertaintyParams) -> tuple[float, ...]:
        return (
            float(u.frequency),
            float(u.linewidth),
            float(u.fwhm_gauss),
            float(u.split),
            float(u.k_np),
            float(u.amplitude),
            float(u.background),
        )

    def unpack_samples(self, arrays_in_order) -> VoigtZeemanSampleParams:
        f, lw, fg, s, k, a, b = arrays_in_order
        return VoigtZeemanSampleParams(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            linewidth=np.asarray(lw, dtype=FLOAT_DTYPE),
            fwhm_gauss=np.asarray(fg, dtype=FLOAT_DTYPE),
            split=np.asarray(s, dtype=FLOAT_DTYPE),
            k_np=np.asarray(k, dtype=FLOAT_DTYPE),
            amplitude=np.asarray(a, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: VoigtZeemanSampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
            np.asarray(samples.fwhm_gauss, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.amplitude, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class VoigtZeemanModel(SignalModel[VoigtZeemanParams, VoigtZeemanSampleParams, VoigtZeemanUncertaintyParams]):
    """Voigt-broadened NV center model with Zeeman splitting.

    Models an NV center with three Lorentzian dips (hyperfine splitting)
    with optional Gaussian broadening (Voigt profile).

    Signal form:
        f(x) = background - (
            (amplitude / k_np) * Voigt(x, frequency - split, linewidth, fwhm_gauss) +
            amplitude * Voigt(x, frequency, linewidth, fwhm_gauss) +
            (amplitude * k_np) * Voigt(x, frequency + split, linewidth, fwhm_gauss)
        )

    The Lorentzian component is defined by ``linewidth`` (HWHM), and Gaussian by ``fwhm_gauss``.
    Use ``sample_params`` to get parameters that keep the signal in [0, 1].

    Parameters
    ----------
    frequency : float
        Central frequency (f_B)
    linewidth : float
        Lorentzian linewidth (HWHM) in Hz
    fwhm_gauss : float
        Gaussian FWHM in Hz
    split : float
        Hyperfine splitting (delta_f_HF)
    k_np : float
        Non-polarization factor (amplitude ratio between peaks)
    amplitude : float
        Peak amplitude scaling factor
    background : float
        Background level
    """

    def __init__(self):
        self._backend = "pseudo_voigt"
        try:
            from scipy.special import wofz

            self.wofz = wofz
            self._backend = "scipy.special.wofz"
        except ImportError:
            self.wofz = None

    def _voigt_profile(self, x: float, center: float, linewidth: float, fwhm_gauss: float) -> float:
        """Compute Voigt profile at x."""
        if self.wofz is not None:
            sigma = fwhm_gauss / (2 * np.sqrt(2 * np.log(2)))
            gamma = linewidth / 2
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            w = self.wofz(z)
            return w.real / (sigma * np.sqrt(2 * np.pi))
        else:
            phi = linewidth / (linewidth + fwhm_gauss)
            eta = 1.36603 * phi - 0.47719 * phi**2 + 0.11116 * phi**3
            gamma = linewidth / 2
            lorentz = gamma / ((x - center) ** 2 + gamma**2)
            sigma = fwhm_gauss / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            return eta * lorentz + (1 - eta) * gauss

    @staticmethod
    def compute_voigt_zeeman_model(
        x: float,
        frequency: float,
        linewidth: float,
        fwhm_gauss: float,
        split: float,
        k_np: float,
        amplitude: float,
        background: float,
        *,
        model_instance: VoigtZeemanModel | None = None,
    ) -> float:
        """Triple Voigt Zeeman; parameter order matches :meth:`parameter_names`."""
        if model_instance is not None:
            v = model_instance._voigt_profile
        else:
            # Fallback for static calls if needed (usually called via instance)
            def v(x_val, c, lw, fg):
                sigma = fg / (2 * np.sqrt(2 * np.log(2)))
                gamma = lw / 2
                lorentz = gamma / ((x_val - c) ** 2 + gamma**2)
                gauss = np.exp(-0.5 * ((x_val - c) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                return 0.5 * lorentz + 0.5 * gauss  # simple approx

        left_peak = (amplitude / k_np) * v(x, frequency - split, linewidth, fwhm_gauss)
        center_peak = amplitude * v(x, frequency, linewidth, fwhm_gauss)
        right_peak = (amplitude * k_np) * v(x, frequency + split, linewidth, fwhm_gauss)

        return background - (left_peak + center_peak + right_peak)

    _SPEC = _VoigtZeemanSpec()

    @property
    def spec(self) -> _VoigtZeemanSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "fwhm_gauss", "amplitude")

    def compute(self, x: float, params: VoigtZeemanParams) -> float:
        return self.compute_voigt_zeeman_model(
            float(x),
            params.frequency,
            params.linewidth,
            params.fwhm_gauss,
            params.split,
            params.k_np,
            params.amplitude,
            params.background,
            model_instance=self,
        )

    def compute_vectorized_samples(self, x: float, samples: VoigtZeemanSampleParams) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        fg = np.asarray(samples.fwhm_gauss, dtype=FLOAT_DTYPE)
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.amplitude, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        sigma = fg / (2 * np.sqrt(2 * np.log(2)))
        gamma = lw / 2

        def profile_at(center: np.ndarray) -> np.ndarray:
            if self.wofz is not None:
                z = ((x_f - center) + 1j * gamma) / (sigma * np.sqrt(2))
                w = self.wofz(z)
                return w.real / (sigma * np.sqrt(2 * np.pi))
            phi = lw / (lw + fg)
            eta = 1.36603 * phi - 0.47719 * phi**2 + 0.11116 * phi**3
            lorentz = gamma / ((x_f - center) ** 2 + gamma**2)
            gauss = np.exp(-0.5 * ((x_f - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            return eta * lorentz + (1 - eta) * gauss

        l_peak = (amp / k_np) * profile_at(freq - split)
        c_peak = amp * profile_at(freq)
        r_peak = (amp * k_np) * profile_at(freq + split)

        return (bg - (l_peak + c_peak + r_peak)).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: VoigtZeemanSampleParams) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        fg = np.asarray(samples.fwhm_gauss, dtype=FLOAT_DTYPE)
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        amp = np.asarray(samples.amplitude, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        sigma = fg / (2 * np.sqrt(2 * np.log(2)))
        gamma = lw / 2
        x2d = xs[:, None]

        def profile_at(center: np.ndarray) -> np.ndarray:
            c2d = center[None, :]
            if self.wofz is not None:
                z = ((x2d - c2d) + 1j * gamma[None, :]) / (sigma[None, :] * np.sqrt(2))
                w = self.wofz(z)
                return w.real / (sigma[None, :] * np.sqrt(2 * np.pi))
            phi = lw / (lw + fg)
            eta = 1.36603 * phi - 0.47719 * phi**2 + 0.11116 * phi**3
            lorentz = gamma[None, :] / ((x2d - c2d) ** 2 + gamma[None, :] ** 2)
            gauss = np.exp(-0.5 * ((x2d - c2d) / sigma[None, :]) ** 2) / (sigma[None, :] * np.sqrt(2 * np.pi))
            return eta[None, :] * lorentz + (1 - eta)[None, :] * gauss

        l_peak = (amp[None, :] / k_np[None, :]) * profile_at(freq - split)
        c_peak = amp[None, :] * profile_at(freq)
        r_peak = (amp[None, :] * k_np[None, :]) * profile_at(freq + split)

        return (bg[None, :] - (l_peak + c_peak + r_peak)).astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> list[Parameter]:
        """Sample parameters that keep the signal within [0, 1]."""
        linewidth = rng.uniform(0.03, 0.08)
        fwhm_gauss = linewidth * rng.uniform(0.1, 0.4)
        split = rng.uniform(0.05, 0.12)
        k_np = rng.uniform(2.0, 4.0)
        frequency = rng.uniform(split + 0.1, 1.0 - split - 0.1)
        background = 1.0

        temp_params = [
            Parameter(name="frequency", bounds=(0.0, 1.0), value=frequency),
            Parameter(name="linewidth", bounds=(0.001, 0.3), value=linewidth),
            Parameter(name="fwhm_gauss", bounds=(0.0001, 0.1), value=fwhm_gauss),
            Parameter(name="split", bounds=(0.0, 0.3), value=split),
            Parameter(name="k_np", bounds=(1.0, 6.0), value=k_np),
            Parameter(name="amplitude", bounds=(0.0, 1.0), value=1.0),
            Parameter(name="background", bounds=(0.0, 1.0), value=0.0),
        ]

        # Use exactly 1.0 maximum dip depth scaling
        tv = tuple(p.value for p in temp_params)
        min_val = min(
            self.compute_voigt_zeeman_model(frequency - split, *tv, model_instance=self),
            self.compute_voigt_zeeman_model(frequency, *tv, model_instance=self),
            self.compute_voigt_zeeman_model(frequency + split, *tv, model_instance=self),
        )

        amplitude = 1.0 / abs(min_val) if abs(min_val) > 1e-12 else 1.0

        return [
            Parameter(name="frequency", bounds=(0.0, 1.0), value=frequency),
            Parameter(name="linewidth", bounds=(0.001, 0.3), value=linewidth),
            Parameter(name="fwhm_gauss", bounds=(0.0001, 0.1), value=fwhm_gauss),
            Parameter(name="split", bounds=(0.0, 0.3), value=split),
            Parameter(name="k_np", bounds=(1.0, 6.0), value=k_np),
            Parameter(name="amplitude", bounds=(0.0, amplitude * 2.0), value=amplitude),
            Parameter(name="background", bounds=(0.5, 1.5), value=background),
        ]
