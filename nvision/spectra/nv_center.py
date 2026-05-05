"""NV center signal signal based on physics.

These signal implement the actual ODMR signal equations for NV centers in diamond.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import (
    nv_center_lorentzian_eval,
    nv_center_lorentzian_vectorized_many,
    nv_center_pseudo_voigt_vectorized_many,
)
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec

# Legacy scale factor (no longer used by :class:`~nvision.sim.gen.nv_center_generator.NVCenterCoreGenerator`;
# Lorentzian NV uses ``amplitude ≈ dip_depth * linewidth²`` in Hz², matching :class:`LorentzianModel`).
A_PARAM = 0.0003
MIN_K_NP = 2.0
MAX_K_NP = 4.0

DEFAULT_NV_CENTER_FREQ_X_MIN = 2.6e9
DEFAULT_NV_CENTER_FREQ_X_MAX = 3.1e9


@dataclass(frozen=True)
class NVCenterLorentzianSpectrum:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): right peak depth × linewidth²."""
        return (self.dip_depth / self.k_np) * self.linewidth**2


@dataclass(frozen=True)
class NVCenterLorentzianSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class NVCenterLorentzianSpectrumUncertainty:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float
    background: float


class _NVCenterLorentzianSpec(
    GenericParamSpec[
        NVCenterLorentzianSpectrum,
        NVCenterLorentzianSpectrumSamples,
        NVCenterLorentzianSpectrumUncertainty,
    ]
):
    params_cls = NVCenterLorentzianSpectrum
    samples_cls = NVCenterLorentzianSpectrumSamples
    uncertainty_cls = NVCenterLorentzianSpectrumUncertainty


class NVCenterLorentzianModel(
    SignalModel[
        NVCenterLorentzianSpectrum,
        NVCenterLorentzianSpectrumSamples,
        NVCenterLorentzianSpectrumUncertainty,
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
        Right (deepest) peak depth in [0, 1]. Center depth = dip_depth / k_np.
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
        freq = np.asarray(frequency, dtype=FLOAT_DTYPE)
        linewidth_arr = np.asarray(linewidth, dtype=FLOAT_DTYPE)
        split_arr = np.asarray(split, dtype=FLOAT_DTYPE)
        k_np_arr = np.asarray(k_np, dtype=FLOAT_DTYPE)
        depth = np.asarray(dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(background, dtype=FLOAT_DTYPE)

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

    def parameter_weights(self) -> dict[str, float]:
        return {"frequency": 2.0, "linewidth": 1.0, "split": 1.0, "k_np": 1.0, "dip_depth": 1.0, "background": 1.0}

    def signal_min_span(self, domain_width: float) -> float | None:
        linewidth_lo = domain_width * 0.0001
        return 4.0 * linewidth_lo

    def signal_max_span(self, domain_width: float) -> float | None:
        split_hi = 5.0e6
        linewidth_hi = domain_width * 0.05
        return 2.0 * split_hi + 4.0 * linewidth_hi

    def expected_dip_count(self) -> int:
        """Doublet (two dips) when strain-split; model supports ms=+1/-1 transitions."""
        return 3

    def compute(self, x: float, params: NVCenterLorentzianSpectrum) -> float:
        return self.compute_nvcenter_lorentzian_model(
            float(x),
            params.frequency,
            params.linewidth,
            params.split,
            params.k_np,
            params.dip_depth / params.k_np,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: NVCenterLorentzianSpectrumSamples) -> np.ndarray:
        actual_depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE) / np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        out = self.compute_nvcenter_lorentzian_model_vectorized(
            x,
            samples.frequency,
            samples.linewidth,
            samples.split,
            samples.k_np,
            actual_depth,
            samples.background,
        )
        return np.asarray(out, dtype=FLOAT_DTYPE)

    def compute_vectorized_many(
        self, x_array: Sequence[float], samples: NVCenterLorentzianSpectrumSamples
    ) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], freq.shape[0]), dtype=FLOAT_DTYPE)

        nv_center_lorentzian_vectorized_many(
            xs,
            freq,
            np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
            out,
        )
        return out


@dataclass(frozen=True)
class NVCenterVoigtSpectrum:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    split: float
    k_np: float
    dip_depth: float
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): approximate Lorentzian-equivalent amplitude."""
        gamma_l = self.lorentz_frac * self.fwhm_total / 2
        return (self.dip_depth / self.k_np) * gamma_l**2


@dataclass(frozen=True)
class NVCenterVoigtSpectrumSamples:
    frequency: np.ndarray
    fwhm_total: np.ndarray
    lorentz_frac: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class NVCenterVoigtSpectrumUncertainty:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    split: float
    k_np: float
    dip_depth: float
    background: float


class _NVCenterVoigtSpec(
    GenericParamSpec[
        NVCenterVoigtSpectrum,
        NVCenterVoigtSpectrumSamples,
        NVCenterVoigtSpectrumUncertainty,
    ]
):
    params_cls = NVCenterVoigtSpectrum
    samples_cls = NVCenterVoigtSpectrumSamples
    uncertainty_cls = NVCenterVoigtSpectrumUncertainty


class NVCenterVoigtModel(
    SignalModel[NVCenterVoigtSpectrum, NVCenterVoigtSpectrumSamples, NVCenterVoigtSpectrumUncertainty]
):
    """NV center with Gaussian broadening (Voigt profile).

    Not njit-accelerated: evaluation uses SciPy/JAX ``wofz`` or a pseudo-Voigt fallback.

    Models an NV center where each Lorentzian dip is convolved with a Gaussian,
    resulting in a Voigt profile. This accounts for inhomogeneous broadening
    due to strain, temperature variations, etc.

    Parameters
    ----------
    frequency : float
        Central frequency f_B in Hz
    fwhm_total : float
        Total effective linewidth (Lorentzian + Gaussian) in Hz
    lorentz_frac : float
        Lorentzian share of broadening in [0, 1] (0 = pure Gaussian, 1 = pure Lorentzian)
    split : float
        Hyperfine splitting in Hz
    k_np : float
        Non-polarization factor
    dip_depth : float
        Right (deepest) peak depth in [0, 1]. Center depth = dip_depth / k_np.
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
        fwhm_total: float,
        lorentz_frac: float,
        split: float,
        k_np: float,
        dip_depth: float,
        background: float,
    ) -> float:
        """Triple Voigt NV ODMR; parameter order matches :meth:`parameter_names`."""
        fwhm_l = lorentz_frac * fwhm_total
        fwhm_g = (1.0 - lorentz_frac) * fwhm_total
        actual_depth = dip_depth / k_np
        if split < 1e-10:
            # Zero-field: single combined dip with amplitude = actual_depth
            return background - actual_depth * self._voigt_profile_unit_peak(x, frequency, fwhm_l, fwhm_g)

        left_dip = (actual_depth / k_np) * self._voigt_profile_unit_peak(x, frequency - split, fwhm_l, fwhm_g)
        center_dip = actual_depth * self._voigt_profile_unit_peak(x, frequency, fwhm_l, fwhm_g)
        right_dip = (actual_depth * k_np) * self._voigt_profile_unit_peak(x, frequency + split, fwhm_l, fwhm_g)

        return background - (left_dip + center_dip + right_dip)

    _SPEC = _NVCenterVoigtSpec()

    @property
    def spec(self) -> _NVCenterVoigtSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("fwhm_total", "dip_depth")

    def parameter_weights(self) -> dict[str, float]:
        return {
            "frequency": 2.0,
            "fwhm_total": 1.0,
            "lorentz_frac": 1.0,
            "split": 1.0,
            "k_np": 1.0,
            "dip_depth": 1.0,
            "background": 1.0,
        }

    def signal_min_span(self, domain_width: float) -> float | None:
        fwhm_total_lo = 70e3
        return 2.0 * fwhm_total_lo

    def signal_max_span(self, domain_width: float) -> float | None:
        split_hi = 5.0e6
        fwhm_total_hi = 2.8e6
        return 2.0 * split_hi + 2.0 * fwhm_total_hi

    def expected_dip_count(self) -> int:
        """Triplet (three dips) when split>0: ms=-1, 0, +1 transitions."""
        return 3

    def compute(self, x: float, params: NVCenterVoigtSpectrum) -> float:
        return self.compute_nvcenter_voigt_model(
            float(x),
            params.frequency,
            params.fwhm_total,
            params.lorentz_frac,
            params.split,
            params.k_np,
            params.dip_depth,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: NVCenterVoigtSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        fwhm_total = np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE)
        lorentz_frac = np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE)
        fwhm_l = lorentz_frac * fwhm_total
        fwhm_g = (1.0 - lorentz_frac) * fwhm_total
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        actual_depth = depth / k_np
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

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
        center_dip = actual_depth * profile_c
        left_dip = (actual_depth / k_np) * profile_l
        right_dip = (actual_depth * k_np) * profile_r
        pred_split = bg - (left_dip + center_dip + right_dip)

        # No-split case: combined dip at center
        pred_nosplit = bg - actual_depth * profile_c

        return np.where(split_mask, pred_nosplit, pred_split).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: NVCenterVoigtSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            # Accept raw arrays / sample containers via the generic base fallback.
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)

        if self.wofz is None:
            # Fast pseudo-Voigt path via fused Numba kernel.
            out = np.empty((xs.shape[0], freq.shape[0]), dtype=FLOAT_DTYPE)
            nv_center_pseudo_voigt_vectorized_many(
                xs,
                freq,
                np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
                np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
                np.asarray(samples.split, dtype=FLOAT_DTYPE),
                np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
                np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
                np.asarray(samples.background, dtype=FLOAT_DTYPE),
                out,
            )
            return out

        # Exact wofz path – keep vectorised SciPy/JAX calls but reduce temporaries.
        fwhm_total = np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE)
        lorentz_frac = np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE)
        fwhm_l = lorentz_frac * fwhm_total
        fwhm_g = (1.0 - lorentz_frac) * fwhm_total
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        actual_depth = depth / k_np
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l / 2
        x2d = xs[:, None]

        # Center height (peak value) for normalization
        z_center = (0.0 + 1j * gamma) / (sigma * np.sqrt(2))
        w_center = self.wofz(z_center)
        center_height = w_center.real / (sigma * np.sqrt(2 * np.pi))

        def _wofz_profile_at(center: np.ndarray) -> np.ndarray:
            center2d = center[None, :]
            z = ((x2d - center2d) + 1j * gamma[None, :]) / (sigma[None, :] * np.sqrt(2))
            w = self.wofz(z)
            profile = w.real / (sigma[None, :] * np.sqrt(2 * np.pi))
            return profile / center_height[None, :]

        profile_c = _wofz_profile_at(freq)
        split_mask = split < 1e-10
        profile_l = _wofz_profile_at(freq - split)
        profile_r = _wofz_profile_at(freq + split)

        center_dip = actual_depth[None, :] * profile_c
        left_dip = (actual_depth[None, :] / k_np[None, :]) * profile_l
        right_dip = (actual_depth[None, :] * k_np[None, :]) * profile_r
        pred_split = bg[None, :] - (left_dip + center_dip + right_dip)

        pred_nosplit = bg[None, :] - actual_depth[None, :] * profile_c
        return np.where(split_mask[None, :], pred_nosplit, pred_split).astype(FLOAT_DTYPE, copy=False)


def nv_center_lorentzian_bounds_for_domain(
    x_min: float,
    x_max: float,
    narrow: bool = False,
    true_params: NVCenterLorentzianSpectrum | dict | None = None,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV Lorentzian signals over ``[x_min, x_max]``."""
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

    if narrow:
        # Tighter bounds for "narrow" variant
        linewidth_bounds = (1e3, 0.2e6)  # 1 kHz to 200 kHz
        split_bounds = (0.1e6, 2.0e6)  # 0.1 MHz to 2 MHz
        max_span = 5.0e6

        if true_params is not None:

            def _pm10(v, lo=None, hi=None):
                left, right = float(v) * 0.9, float(v) * 1.1
                if lo is not None:
                    left = max(left, float(lo))
                if hi is not None:
                    right = min(right, float(hi))
                return (float(left), float(right))

            # Helper to get attribute or dict key
            def _val(k):
                if isinstance(true_params, dict):
                    return true_params.get(k)
                return getattr(true_params, k, None)

            f_true = _val("frequency")
            # For frequency, use +-10% of domain width if available
            f_win = 0.1 * width
            f_bounds = (float(f_true - f_win), float(f_true + f_win))

            return {
                "frequency": f_bounds,
                "linewidth": _pm10(_val("linewidth")),
                "split": _pm10(_val("split")),
                "k_np": _pm10(_val("k_np"), lo=MIN_K_NP, hi=MAX_K_NP),
                "dip_depth": _pm10(_val("dip_depth"), lo=0.01, hi=1.0),
                "background": _pm10(_val("background"), lo=0.5, hi=1.5),
                "_signal_max_span": (0.0, max_span),
            }
    else:
        linewidth_bounds = (width * 0.001, width * 0.05)
        split_bounds = (width * 0.005, width * 0.02)
        max_span = width * 0.1

    return {
        "frequency": (float(x_min), float(x_max)),
        "linewidth": linewidth_bounds,
        "split": split_bounds,
        "k_np": (MIN_K_NP, MAX_K_NP),
        "dip_depth": (0.1, 1.0),
        "background": (0.5, 1.5),
        "_signal_max_span": (0.0, max_span),
    }


# ---------------------------------------------------------------------------
# One-peak (zero-field) NV Lorentzian model — split and k_np are fixed to 0/1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVCenterOnePeakLorentzianSpectrum:
    frequency: float
    linewidth: float
    dip_depth: float
    background: float


@dataclass(frozen=True)
class NVCenterOnePeakLorentzianSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class NVCenterOnePeakLorentzianSpectrumUncertainty:
    frequency: float
    linewidth: float
    dip_depth: float
    background: float


class _NVCenterOnePeakLorentzianSpec(
    GenericParamSpec[
        NVCenterOnePeakLorentzianSpectrum,
        NVCenterOnePeakLorentzianSpectrumSamples,
        NVCenterOnePeakLorentzianSpectrumUncertainty,
    ]
):
    params_cls = NVCenterOnePeakLorentzianSpectrum
    samples_cls = NVCenterOnePeakLorentzianSpectrumSamples
    uncertainty_cls = NVCenterOnePeakLorentzianSpectrumUncertainty


class NVCenterOnePeakLorentzianModel(
    SignalModel[
        NVCenterOnePeakLorentzianSpectrum,
        NVCenterOnePeakLorentzianSpectrumSamples,
        NVCenterOnePeakLorentzianSpectrumUncertainty,
    ]
):
    """NV center ODMR signal — single Lorentzian dip (zero-field / no hyperfine splitting).

    split is fixed to 0 and k_np is fixed to 1, so only 4 parameters are inferred:
    frequency, linewidth, dip_depth, background.

    Signal form:
        S(f) = background - dip_depth * linewidth² / ((f - frequency)² + linewidth²)
    """

    _SPEC = _NVCenterOnePeakLorentzianSpec()

    @property
    def spec(self) -> _NVCenterOnePeakLorentzianSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "dip_depth")

    def parameter_weights(self) -> dict[str, float]:
        return {"frequency": 2.0, "linewidth": 1.0, "dip_depth": 1.0, "background": 1.0}

    def signal_min_span(self, domain_width: float) -> float | None:
        linewidth_lo = domain_width * 0.0001
        return 4.0 * linewidth_lo

    def signal_max_span(self, domain_width: float) -> float | None:
        linewidth_hi = domain_width * 0.05
        return 4.0 * linewidth_hi

    def expected_dip_count(self) -> int:
        """Single dip (no splitting); single Lorentzian lineshape."""
        return 1

    def compute(self, x: float, params: NVCenterOnePeakLorentzianSpectrum) -> float:
        lw2 = params.linewidth**2
        denom = (float(x) - params.frequency) ** 2 + lw2
        return float(params.background - (params.dip_depth * lw2) / denom)

    def compute_vectorized_samples(self, x: float, samples: NVCenterOnePeakLorentzianSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        lw2 = lw**2
        denom = (x_f - freq) ** 2 + lw2
        return (bg - depth * lw2 / denom).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(
        self, x_array: Sequence[float], samples: NVCenterOnePeakLorentzianSpectrumSamples
    ) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]
        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)
        x2d = xs[:, None]
        lw2 = lw[None, :] ** 2
        denom = (x2d - freq[None, :]) ** 2 + lw2
        return (bg[None, :] - depth[None, :] * lw2 / denom).astype(FLOAT_DTYPE, copy=False)


def nv_center_one_peak_lorentzian_bounds_for_domain(
    x_min: float,
    x_max: float,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV single-peak (zero-field) Lorentzian over ``[x_min, x_max]``."""
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")
    linewidth_hi = width * 0.05
    return {
        "frequency": (float(x_min), float(x_max)),
        "linewidth": (width * 0.0001, linewidth_hi),
        "dip_depth": (0.01, 1.0),
        "background": (0.95, 1.05),
        "_signal_max_span": (0.0, 4.0 * linewidth_hi),
    }


def nv_center_voigt_bounds_for_domain(
    x_min: float,
    x_max: float,
    narrow: bool = False,
    true_params: NVCenterVoigtSpectrum | dict | None = None,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV Voigt signals over ``[x_min, x_max]``."""
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

    if narrow:
        # Tighter bounds for "narrow" variant
        fwhm_total_bounds = (2e3, 0.4e6)  # 2 kHz to 400 kHz
        split_bounds = (0.1e6, 2.0e6)  # 0.1 MHz to 2 MHz
        max_span = 5.0e6

        if true_params is not None:

            def _pm10(v, lo=None, hi=None):
                left, right = float(v) * 0.9, float(v) * 1.1
                if lo is not None:
                    left = max(left, float(lo))
                if hi is not None:
                    right = min(right, float(hi))
                return (float(left), float(right))

            # Helper to get attribute or dict key
            def _val(k):
                if isinstance(true_params, dict):
                    return true_params.get(k)
                return getattr(true_params, k, None)

            f_true = _val("frequency")
            # For frequency, use +-10% of domain width if available
            f_win = 0.1 * width
            f_bounds = (float(f_true - f_win), float(f_true + f_win))

            return {
                "frequency": f_bounds,
                "fwhm_total": _pm10(_val("fwhm_total")),
                "lorentz_frac": _pm10(_val("lorentz_frac"), lo=0.01, hi=0.99),
                "split": _pm10(_val("split")),
                "k_np": _pm10(_val("k_np"), lo=MIN_K_NP, hi=MAX_K_NP),
                "dip_depth": _pm10(_val("dip_depth"), lo=0.01, hi=1.0),
                "background": _pm10(_val("background"), lo=0.5, hi=1.5),
                "_signal_max_span": (0.0, max_span),
            }
    else:
        split_hi = 5.0e6
        fwhm_total_hi = 2.8e6
        fwhm_total_bounds = (70e3, fwhm_total_hi)
        split_bounds = (0.0, split_hi)
        max_span = 2.0 * split_hi + 2.0 * fwhm_total_hi

    return {
        "frequency": (float(x_min), float(x_max)),
        "fwhm_total": fwhm_total_bounds,
        "lorentz_frac": (0.05, 0.98),
        "split": split_bounds,
        "k_np": (MIN_K_NP, MAX_K_NP),
        "dip_depth": (0.001, 1.0),
        "background": (0.95, 1.05),
        "_signal_max_span": (0.0, max_span),
    }
