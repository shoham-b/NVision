"""Voigt-broadened NV center model with Zeeman splitting."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.parameter import Parameter
from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class VoigtZeemanParams:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): dip_depth * linewidth²."""
        return self.dip_depth * self.linewidth**2


@dataclass(frozen=True)
class VoigtZeemanSampleParams:
    frequency: np.ndarray
    linewidth: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class VoigtZeemanUncertaintyParams:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float
    background: float


class _VoigtZeemanSpec(ParamSpec[VoigtZeemanParams, VoigtZeemanSampleParams, VoigtZeemanUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "linewidth", "split", "k_np", "dip_depth", "background")

    @property
    def dim(self) -> int:
        return 6

    def unpack_params(self, values) -> VoigtZeemanParams:
        f, w, s, k, d, b = values
        return VoigtZeemanParams(float(f), float(w), float(s), float(k), float(d), float(b))

    def pack_params(self, params: VoigtZeemanParams) -> tuple[float, ...]:
        return (
            float(params.frequency),
            float(params.linewidth),
            float(params.split),
            float(params.k_np),
            float(params.dip_depth),
            float(params.background),
        )

    def unpack_uncertainty(self, values) -> VoigtZeemanUncertaintyParams:
        f, w, s, k, d, b = values
        return VoigtZeemanUncertaintyParams(float(f), float(w), float(s), float(k), float(d), float(b))

    def pack_uncertainty(self, u: VoigtZeemanUncertaintyParams) -> tuple[float, ...]:
        return (
            float(u.frequency),
            float(u.linewidth),
            float(u.split),
            float(u.k_np),
            float(u.dip_depth),
            float(u.background),
        )

    def unpack_samples(self, arrays_in_order) -> VoigtZeemanSampleParams:
        f, w, s, k, d, b = arrays_in_order
        return VoigtZeemanSampleParams(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            linewidth=np.asarray(w, dtype=FLOAT_DTYPE),
            split=np.asarray(s, dtype=FLOAT_DTYPE),
            k_np=np.asarray(k, dtype=FLOAT_DTYPE),
            dip_depth=np.asarray(d, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: VoigtZeemanSampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class VoigtZeemanModel(SignalModel[VoigtZeemanParams, VoigtZeemanSampleParams, VoigtZeemanUncertaintyParams]):
    """Voigt-broadened NV center model with Zeeman splitting.

    Models an NV center with three Lorentzian dips (hyperfine splitting)
    with optional Gaussian broadening (Voigt profile).

    Parameters
    ----------
    frequency : float
        Central frequency (f_B)
    linewidth : float
        Lorentzian linewidth (omega, HWHM)
    split : float
        Hyperfine splitting (delta_f_HF)
    k_np : float
        Non-polarization factor (amplitude ratio between peaks)
    dip_depth : float
        Peak dip depth (contrast) factor. Peak height = dip_depth.
    background : float
        Background level
    """

    @staticmethod
    def compute_voigt_zeeman_model(
        x: float,
        frequency: float,
        linewidth: float,
        split: float,
        k_np: float,
        dip_depth: float,
        background: float,
    ) -> float:
        """Simplified Zeeman Lorentzian triple; parameter order matches :meth:`parameter_names`."""
        lw2 = linewidth**2
        left_denom = (x - (frequency - split)) ** 2 + lw2
        left_peak = (dip_depth * lw2 / k_np) / left_denom

        center_denom = (x - frequency) ** 2 + lw2
        center_peak = (dip_depth * lw2) / center_denom

        right_denom = (x - (frequency + split)) ** 2 + lw2
        right_peak = (dip_depth * lw2 * k_np) / right_denom

        return background - (left_peak + center_peak + right_peak)

    _SPEC = _VoigtZeemanSpec()

    @property
    def spec(self) -> _VoigtZeemanSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "dip_depth")

    def compute(self, x: float, params: VoigtZeemanParams) -> float:
        return self.compute_voigt_zeeman_model(
            float(x),
            params.frequency,
            params.linewidth,
            params.split,
            params.k_np,
            params.dip_depth,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: VoigtZeemanSampleParams) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        dip_depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        lw2 = lw**2
        left_denom = (x_f - (freq - split)) ** 2 + lw2
        left_peak = (dip_depth * lw2 / k_np) / left_denom

        center_denom = (x_f - freq) ** 2 + lw2
        center_peak = (dip_depth * lw2) / center_denom

        right_denom = (x_f - (freq + split)) ** 2 + lw2
        right_peak = (dip_depth * lw2 * k_np) / right_denom

        return (bg - (left_peak + center_peak + right_peak)).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: VoigtZeemanSampleParams) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        dip_depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        x2d = xs[:, None]
        lw2 = lw[None, :] ** 2
        denom_r = (x2d - (freq[None, :] + split[None, :])) ** 2 + lw2

        amp_l = (dip_depth[None, :] * lw2) / k_np[None, :]
        amp_c = dip_depth[None, :] * lw2
        amp_r = dip_depth[None, :] * lw2 * k_np[None, :]

        denom_l = (x2d - (freq[None, :] - split[None, :])) ** 2 + lw2
        denom_c = (x2d - freq[None, :]) ** 2 + lw2
        denom_r = (x2d - (freq[None, :] + split[None, :])) ** 2 + lw2

        out = bg[None, :] - (amp_l / denom_l + amp_c / denom_c + amp_r / denom_r)
        return out.astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> list[Parameter]:
        """Sample parameters that keep the signal within [0, 1].

        Evaluates the exact minimum of the generated parameters and
        bounds the peak to an exact dip of 1.0 from the background.
        """
        linewidth = rng.uniform(0.03, 0.08)
        split = rng.uniform(0.05, 0.12)
        k_np = rng.uniform(2.0, 4.0)
        frequency = rng.uniform(split + 0.1, 1.0 - split - 0.1)
        background = 1.0

        temp_params = [
            Parameter(name="frequency", bounds=(0.0, 1.0), value=frequency),
            Parameter(name="linewidth", bounds=(0.001, 0.3), value=linewidth),
            Parameter(name="split", bounds=(0.0, 0.3), value=split),
            Parameter(name="k_np", bounds=(1.0, 6.0), value=k_np),
            Parameter(name="dip_depth", bounds=(0.0, 1.5), value=1.0),
            Parameter(name="background", bounds=(0.0, 1.0), value=0.0),
        ]

        # The maximum dip will occur at one of the three peaks
        tv = tuple(p.value for p in temp_params)
        min_val = min(
            VoigtZeemanModel.compute_voigt_zeeman_model(frequency - split, *tv),
            VoigtZeemanModel.compute_voigt_zeeman_model(frequency, *tv),
            VoigtZeemanModel.compute_voigt_zeeman_model(frequency + split, *tv),
        )

        # Min_val is effectively the negative depth per unit dip_depth
        dip_depth = 1.0 / abs(min_val) if abs(min_val) > 1e-12 else 1.0

        return [
            Parameter(name="frequency", bounds=(0.0, 1.0), value=frequency),
            Parameter(name="linewidth", bounds=(0.001, 0.3), value=linewidth),
            Parameter(name="split", bounds=(0.0, 0.3), value=split),
            Parameter(name="k_np", bounds=(1.0, 6.0), value=k_np),
            Parameter(name="dip_depth", bounds=(0.0, 2.0), value=dip_depth),
            Parameter(name="background", bounds=(0.5, 1.5), value=background),
        ]
