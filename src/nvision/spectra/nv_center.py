"""NV center signal signal based on physics.

These signal implement the actual ODMR signal equations for NV centers in diamond.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import nv_center_lorentzian_eval
from nvision.spectra.signal import ParamSpec, SignalModel

# Legacy scale factor (no longer used by :class:`~nvision.sim.gen.core_generators.NVCenterCoreGenerator`;
# Lorentzian NV uses ``amplitude ≈ dip_depth * linewidth²`` in Hz², matching :class:`LorentzianModel`).
A_PARAM = 0.0003
MIN_K_NP = 2.0
MAX_K_NP = 4.0
MIN_NV_CENTER_DELTA = 0.01
MAX_NV_CENTER_DELTA = 0.15
MIN_NV_CENTER_OMEGA = 0.008
MAX_NV_CENTER_OMEGA = 0.01

DEFAULT_NV_CENTER_FREQ_X_MIN = 2.6e9
DEFAULT_NV_CENTER_FREQ_X_MAX = 3.1e9


@dataclass(frozen=True)
class NVCenterLorentzianParams:
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
class NVCenterLorentzianSampleParams:
    frequency: np.ndarray
    linewidth: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class NVCenterLorentzianUncertaintyParams:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float
    background: float


class _NVCenterLorentzianSpec(
    ParamSpec[
        NVCenterLorentzianParams,
        NVCenterLorentzianSampleParams,
        NVCenterLorentzianUncertaintyParams,
    ]
):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "linewidth", "split", "k_np", "dip_depth", "background")

    @property
    def dim(self) -> int:
        return 6

    def unpack_params(self, values) -> NVCenterLorentzianParams:
        f, w, s, k, d, b = values
        return NVCenterLorentzianParams(float(f), float(w), float(s), float(k), float(d), float(b))

    def pack_params(self, params: NVCenterLorentzianParams) -> tuple[float, ...]:
        return (
            float(params.frequency),
            float(params.linewidth),
            float(params.split),
            float(params.k_np),
            float(params.dip_depth),
            float(params.background),
        )

    def unpack_uncertainty(self, values) -> NVCenterLorentzianUncertaintyParams:
        f, w, s, k, d, b = values
        return NVCenterLorentzianUncertaintyParams(float(f), float(w), float(s), float(k), float(d), float(b))

    def pack_uncertainty(self, u: NVCenterLorentzianUncertaintyParams) -> tuple[float, ...]:
        return (
            float(u.frequency),
            float(u.linewidth),
            float(u.split),
            float(u.k_np),
            float(u.dip_depth),
            float(u.background),
        )

    def unpack_samples(self, arrays_in_order) -> NVCenterLorentzianSampleParams:
        f, w, s, k, d, b = arrays_in_order
        return NVCenterLorentzianSampleParams(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            linewidth=np.asarray(w, dtype=FLOAT_DTYPE),
            split=np.asarray(s, dtype=FLOAT_DTYPE),
            k_np=np.asarray(k, dtype=FLOAT_DTYPE),
            dip_depth=np.asarray(d, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: NVCenterLorentzianSampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class NVCenterLorentzianModel(
    SignalModel[
        NVCenterLorentzianParams,
        NVCenterLorentzianSampleParams,
        NVCenterLorentzianUncertaintyParams,
    ]
):
    """NV center ODMR signal model with three Lorentzian dips.

    Prefer :meth:`compute_nvcenter_lorentzian_model` when you already have floats.

    Models the optically detected magnetic resonance (ODMR) spectrum of an
    NV center in diamond. The signal has three Lorentzian dips from a baseline
    of 1.0, corresponding to the ms=±1 and ms=0 spin states with hyperfine splitting.

    Signal form:
        S(f) = 1 - L_left - L_center - L_right

    Where each Lorentzian dip is:
        L(f, f_0, A, ω) = A / ((f - f_0)^2 + ω^2)

    Parameters
    ----------
    frequency : float
        Central frequency f_B (center of main dip) in Hz
    linewidth : float
        Lorentzian linewidth ω (HWHM) in Hz
    split : float
        Hyperfine splitting Δf_HF in Hz (distance from center to outer peaks)
    k_np : float
        Non-polarization factor (amplitude ratio between peaks)
        Left peak amplitude: a/k_np, Center: a, Right: a*k_np
    dip_depth : float
        Normalized peak depth (0 to 1). Peak height = dip_depth.
    background : float
        Background level (typically 1.0 for normalized signal)
    """

    @staticmethod
    def compute_nvcenter_lorentzian_model(
        x: float,
        frequency: float,
        linewidth: float,
        split: float,
        k_np: float,
        dip_depth: float,
        background: float,
    ) -> float:
        """Triple Lorentzian NV ODMR; parameter order matches :meth:`parameter_names`."""
        return nv_center_lorentzian_eval(float(x), frequency, linewidth, split, k_np, dip_depth, background)

    def compute_nvcenter_lorentzian_model_vectorized(
        self,
        x: float,
        frequency: np.ndarray,
        linewidth: np.ndarray,
        split: np.ndarray,
        k_np: np.ndarray,
        dip_depth: np.ndarray,
        background: np.ndarray,
    ) -> np.ndarray:
        """Vectorized triple-Lorentzian NV evaluation for one probe location."""
        x_f = float(x)
        freq = np.asarray(frequency, dtype=np.float64)
        linewidth_arr = np.asarray(linewidth, dtype=np.float64)
        split_arr = np.asarray(split, dtype=np.float64)
        k_np_arr = np.asarray(k_np, dtype=np.float64)
        depth = np.asarray(dip_depth, dtype=np.float64)
        bg = np.asarray(background, dtype=np.float64)

        denom_l = (x_f - (freq - split_arr)) * (x_f - (freq - split_arr)) + linewidth_arr * linewidth_arr
        denom_c = (x_f - freq) * (x_f - freq) + linewidth_arr * linewidth_arr
        denom_r = (x_f - (freq + split_arr)) * (x_f - (freq + split_arr)) + linewidth_arr * linewidth_arr
        return bg - (
            ((depth * linewidth_arr**2) / k_np_arr) / denom_l
            + (depth * linewidth_arr**2) / denom_c
            + (depth * linewidth_arr**2 * k_np_arr) / denom_r
        )

    _SPEC = _NVCenterLorentzianSpec()

    @property
    def spec(self) -> _NVCenterLorentzianSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "dip_depth")

    def compute(self, x: float, params: NVCenterLorentzianParams) -> float:
        return self.compute_nvcenter_lorentzian_model(
            float(x),
            params.frequency,
            params.linewidth,
            params.split,
            params.k_np,
            params.dip_depth,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: NVCenterLorentzianSampleParams) -> np.ndarray:
        out = self.compute_nvcenter_lorentzian_model_vectorized(
            x,
            samples.frequency,
            samples.linewidth,
            samples.split,
            samples.k_np,
            samples.dip_depth,
            samples.background,
        )
        return np.asarray(out, dtype=FLOAT_DTYPE)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: NVCenterLorentzianSampleParams) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=np.float64)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=np.float64)
        linewidth_arr = np.asarray(samples.linewidth, dtype=np.float64)
        split_arr = np.asarray(samples.split, dtype=np.float64)
        k_np_arr = np.asarray(samples.k_np, dtype=np.float64)
        depth = np.asarray(samples.dip_depth, dtype=np.float64)
        bg = np.asarray(samples.background, dtype=np.float64)

        x2d = xs[:, None]
        lw2 = linewidth_arr[None, :] ** 2
        denom_l = (x2d - (freq[None, :] - split_arr[None, :])) ** 2 + lw2
        denom_c = (x2d - freq[None, :]) ** 2 + lw2
        denom_r = (x2d - (freq[None, :] + split_arr[None, :])) ** 2 + lw2

        amp_l = (depth[None, :] * lw2) / k_np_arr[None, :]
        amp_c = depth[None, :] * lw2
        amp_r = depth[None, :] * lw2 * k_np_arr[None, :]

        out = bg[None, :] - (amp_l / denom_l + amp_c / denom_c + amp_r / denom_r)
        return np.asarray(out, dtype=FLOAT_DTYPE)


@dataclass(frozen=True)
class NVCenterVoigtParams:
    frequency: float
    fwhm_lorentz: float
    fwhm_gauss: float
    split: float
    k_np: float
    dip_depth: float
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): approximate Lorentzian-equivalent amplitude."""
        # For Voigt, dip_depth * gamma_L^2 is a reasonable scale factor.
        return self.dip_depth * (self.fwhm_lorentz / 2) ** 2


@dataclass(frozen=True)
class NVCenterVoigtSampleParams:
    frequency: np.ndarray
    fwhm_lorentz: np.ndarray
    fwhm_gauss: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class NVCenterVoigtUncertaintyParams:
    frequency: float
    fwhm_lorentz: float
    fwhm_gauss: float
    split: float
    k_np: float
    dip_depth: float
    background: float


class _NVCenterVoigtSpec(
    ParamSpec[
        NVCenterVoigtParams,
        NVCenterVoigtSampleParams,
        NVCenterVoigtUncertaintyParams,
    ]
):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "fwhm_lorentz", "fwhm_gauss", "split", "k_np", "dip_depth", "background")

    @property
    def dim(self) -> int:
        return 7

    def unpack_params(self, values) -> NVCenterVoigtParams:
        f, fl, fg, s, k, d, b = values
        return NVCenterVoigtParams(float(f), float(fl), float(fg), float(s), float(k), float(d), float(b))

    def pack_params(self, params: NVCenterVoigtParams) -> tuple[float, ...]:
        return (
            float(params.frequency),
            float(params.fwhm_lorentz),
            float(params.fwhm_gauss),
            float(params.split),
            float(params.k_np),
            float(params.dip_depth),
            float(params.background),
        )

    def unpack_uncertainty(self, values) -> NVCenterVoigtUncertaintyParams:
        f, fl, fg, s, k, d, b = values
        return NVCenterVoigtUncertaintyParams(float(f), float(fl), float(fg), float(s), float(k), float(d), float(b))

    def pack_uncertainty(self, u: NVCenterVoigtUncertaintyParams) -> tuple[float, ...]:
        return (
            float(u.frequency),
            float(u.fwhm_lorentz),
            float(u.fwhm_gauss),
            float(u.split),
            float(u.k_np),
            float(u.dip_depth),
            float(u.background),
        )

    def unpack_samples(self, arrays_in_order) -> NVCenterVoigtSampleParams:
        f, fl, fg, s, k, d, b = arrays_in_order
        return NVCenterVoigtSampleParams(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            fwhm_lorentz=np.asarray(fl, dtype=FLOAT_DTYPE),
            fwhm_gauss=np.asarray(fg, dtype=FLOAT_DTYPE),
            split=np.asarray(s, dtype=FLOAT_DTYPE),
            k_np=np.asarray(k, dtype=FLOAT_DTYPE),
            dip_depth=np.asarray(d, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: NVCenterVoigtSampleParams) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.fwhm_lorentz, dtype=FLOAT_DTYPE),
            np.asarray(samples.fwhm_gauss, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class NVCenterVoigtModel(SignalModel[NVCenterVoigtParams, NVCenterVoigtSampleParams, NVCenterVoigtUncertaintyParams]):
    """NV center with Gaussian broadening (Voigt profile).

    Not njit-accelerated: evaluation uses SciPy/JAX ``wofz`` or a pseudo-Voigt fallback.

    Models an NV center where each Lorentzian dip is convolved with a Gaussian,
    resulting in a Voigt profile. This accounts for inhomogeneous broadening
    due to strain, temperature variations, etc.

    Parameters
    ----------
    frequency : float
        Central frequency f_B in Hz
    fwhm_lorentz : float
        Lorentzian FWHM (full width at half maximum) in Hz
    fwhm_gauss : float
        Gaussian FWHM in Hz
    split : float
        Hyperfine splitting in Hz
    k_np : float
        Non-polarization factor
    dip_depth : float
        Amplitude scaling factor (directly proportional to peak depth)
    background : float
        Background level
    """

    def __init__(self):
        self._backend = "pseudo_voigt"
        try:
            # Prefer JAX implementation when available.
            from jax.scipy.special import wofz

            self.wofz = wofz
            self._backend = "jax.scipy.special.wofz"
        except ImportError:
            try:
                from scipy.special import wofz

                self.wofz = wofz
                self._backend = "scipy.special.wofz"
            except ImportError:
                self.wofz = None
        except Exception:
            # Any runtime issue in JAX path falls back to SciPy/pseudo-Voigt.
            try:
                from scipy.special import wofz

                self.wofz = wofz
                self._backend = "scipy.special.wofz"
            except ImportError:
                self.wofz = None
        if self.wofz is None:
            self._backend = "pseudo_voigt"

    def _voigt_profile(self, x: float, center: float, fwhm_l: float, fwhm_g: float) -> float:
        """Compute Voigt profile at x.

        Uses Faddeeva function for exact computation if JAX/SciPy are available,
        otherwise falls back to pseudo-Voigt approximation.
        """
        if self.wofz is not None:
            sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
            gamma = fwhm_l / 2
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            w = self.wofz(z)
            return w.real / (sigma * np.sqrt(2 * np.pi))
        else:
            eta = (
                1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
            )
            gamma = fwhm_l / 2
            lorentz = gamma / ((x - center) ** 2 + gamma**2)
            sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            return eta * lorentz + (1 - eta) * gauss

    def _voigt_center_height(self, fwhm_l: float, fwhm_g: float) -> float:
        """Return Voigt profile value at its own center."""
        return self._voigt_profile(0.0, 0.0, fwhm_l, fwhm_g)

    def _voigt_profile_unit_peak(self, x: float, center: float, fwhm_l: float, fwhm_g: float) -> float:
        """Voigt profile normalized so value at center is 1.

        ``_voigt_profile`` returns an area-normalized PDF-like profile, whose peak
        height changes with width parameters. Normalizing by the center value keeps
        dip_depth semantics stable across linewidth/fwhm settings.
        """
        center_val = self._voigt_center_height(fwhm_l, fwhm_g)
        if abs(center_val) < 1e-12:
            return 0.0
        return self._voigt_profile(x, center, fwhm_l, fwhm_g) / center_val

    def compute_nvcenter_voigt_model(
        self,
        x: float,
        frequency: float,
        fwhm_lorentz: float,
        fwhm_gauss: float,
        split: float,
        k_np: float,
        dip_depth: float,
        background: float,
    ) -> float:
        """Triple Voigt NV ODMR; parameter order matches :meth:`parameter_names`."""
        if split < 1e-10:
            combined_amplitude = dip_depth * (k_np + 1 + 1 / k_np)
            return background - combined_amplitude * self._voigt_profile_unit_peak(x, frequency, fwhm_lorentz, fwhm_gauss)

        left_dip = (dip_depth / k_np) * self._voigt_profile_unit_peak(x, frequency - split, fwhm_lorentz, fwhm_gauss)
        center_dip = dip_depth * self._voigt_profile_unit_peak(x, frequency, fwhm_lorentz, fwhm_gauss)
        right_dip = (dip_depth * k_np) * self._voigt_profile_unit_peak(x, frequency + split, fwhm_lorentz, fwhm_gauss)

        return background - (left_dip + center_dip + right_dip)

    _SPEC = _NVCenterVoigtSpec()

    @property
    def spec(self) -> _NVCenterVoigtSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("fwhm_lorentz", "fwhm_gauss", "dip_depth")

    def compute(self, x: float, params: NVCenterVoigtParams) -> float:
        return self.compute_nvcenter_voigt_model(
            float(x),
            params.frequency,
            params.fwhm_lorentz,
            params.fwhm_gauss,
            params.split,
            params.k_np,
            params.dip_depth,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: NVCenterVoigtSampleParams) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=np.float64)
        fwhm_l = np.asarray(samples.fwhm_lorentz, dtype=np.float64)
        fwhm_g = np.asarray(samples.fwhm_gauss, dtype=np.float64)
        split = np.asarray(samples.split, dtype=np.float64)
        k_np = np.asarray(samples.k_np, dtype=np.float64)
        depth = np.asarray(samples.dip_depth, dtype=np.float64)
        bg = np.asarray(samples.background, dtype=np.float64)

        sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l / 2

        # Compute center height (peak value) for normalization to unit peak
        if self.wofz is not None:
            z_center = (0.0 + 1j * gamma) / (sigma * np.sqrt(2))
            w_center = self.wofz(z_center)
            center_height = w_center.real / (sigma * np.sqrt(2 * np.pi))
        else:
            eta = (
                1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
            )
            lorentz_center = 1.0 / gamma
            gauss_center = 1.0 / (sigma * np.sqrt(2 * np.pi))
            center_height = eta * lorentz_center + (1 - eta) * gauss_center

        # Profile at center(s) - normalized to unit peak
        def profile_at(center: np.ndarray) -> np.ndarray:
            if self.wofz is not None:
                z = ((x_f - center) + 1j * gamma) / (sigma * np.sqrt(2))
                w = self.wofz(z)
                profile = w.real / (sigma * np.sqrt(2 * np.pi))
            else:
                eta = (
                    1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                    - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                    + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
                )
                lorentz = gamma / ((x_f - center) ** 2 + gamma**2)
                gauss = np.exp(-0.5 * ((x_f - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                profile = eta * lorentz + (1 - eta) * gauss
            return profile / center_height

        profile_c = profile_at(freq)
        split_mask = split < 1e-10

        # Split case: left/right/center dips
        profile_l = profile_at(freq - split)
        profile_r = profile_at(freq + split)
        center_dip = depth * profile_c
        left_dip = (depth / k_np) * profile_l
        right_dip = (depth * k_np) * profile_r
        pred_split = bg - (left_dip + center_dip + right_dip)

        # No-split case: combined dip at center
        combined_amp = depth * (k_np + 1.0 + 1.0 / k_np)
        pred_nosplit = bg - combined_amp * profile_c

        return np.where(split_mask, pred_nosplit, pred_split).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: NVCenterVoigtSampleParams) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=np.float64)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=np.float64)
        fwhm_l = np.asarray(samples.fwhm_lorentz, dtype=np.float64)
        fwhm_g = np.asarray(samples.fwhm_gauss, dtype=np.float64)
        split = np.asarray(samples.split, dtype=np.float64)
        k_np = np.asarray(samples.k_np, dtype=np.float64)
        depth = np.asarray(samples.dip_depth, dtype=np.float64)
        bg = np.asarray(samples.background, dtype=np.float64)

        sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l / 2
        x2d = xs[:, None]

        # Compute center height (peak value) for normalization to unit peak
        if self.wofz is not None:
            z_center = (0.0 + 1j * gamma) / (sigma * np.sqrt(2))
            w_center = self.wofz(z_center)
            center_height = w_center.real / (sigma * np.sqrt(2 * np.pi))
        else:
            eta = (
                1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
            )
            lorentz_center = 1.0 / gamma
            gauss_center = 1.0 / (sigma * np.sqrt(2 * np.pi))
            center_height = eta * lorentz_center + (1 - eta) * gauss_center

        # Profile at center(s) - normalized to unit peak
        def profile_at(center: np.ndarray) -> np.ndarray:
            center2d = center[None, :]
            if self.wofz is not None:
                z = ((x2d - center2d) + 1j * gamma[None, :]) / (sigma[None, :] * np.sqrt(2))
                w = self.wofz(z)
                profile = w.real / (sigma[None, :] * np.sqrt(2 * np.pi))
            else:
                eta = (
                    1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                    - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                    + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
                )
                lorentz = gamma[None, :] / ((x2d - center2d) ** 2 + gamma[None, :] ** 2)
                gauss = np.exp(-0.5 * ((x2d - center2d) / sigma[None, :]) ** 2) / (sigma[None, :] * np.sqrt(2 * np.pi))
                profile = eta[None, :] * lorentz + (1 - eta)[None, :] * gauss
            return profile / center_height[None, :]

        profile_c = profile_at(freq)
        split_mask = split < 1e-10
        profile_l = profile_at(freq - split)
        profile_r = profile_at(freq + split)

        center_dip = depth[None, :] * profile_c
        left_dip = (depth[None, :] / k_np[None, :]) * profile_l
        right_dip = (depth[None, :] * k_np[None, :]) * profile_r
        pred_split = bg[None, :] - (left_dip + center_dip + right_dip)

        combined_amp = depth * (k_np + 1.0 + 1.0 / k_np)
        pred_nosplit = bg[None, :] - combined_amp[None, :] * profile_c
        return np.where(split_mask[None, :], pred_nosplit, pred_split).astype(FLOAT_DTYPE, copy=False)


def nv_center_lorentzian_bounds_for_domain(
    x_min: float,
    x_max: float,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV Lorentzian signals over ``[x_min, x_max]``."""
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

    return {
        "frequency": (float(x_min), float(x_max)),
        "linewidth": (width * 0.001, width * 0.05),
        "split": (0.0, width * 0.5),
        "k_np": (MIN_K_NP, MAX_K_NP),
        "dip_depth": (0.05, 1.5),
        "background": (0.95, 1.05),
    }


def nv_center_voigt_bounds_for_domain(
    x_min: float,
    x_max: float,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV Voigt signals over ``[x_min, x_max]``."""
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

    return {
        "frequency": (float(x_min), float(x_max)),
        "fwhm_lorentz": (width * 0.001, width * 0.1),
        "fwhm_gauss": (width * 0.0001, width * 0.05),
        "split": (0.0, width * 0.5),
        "k_np": (MIN_K_NP, MAX_K_NP),
        "dip_depth": (0.05, 1.5),
        "background": (0.95, 1.05),
    }
