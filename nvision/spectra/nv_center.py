"""NV center signal signal based on physics.

These signal implement the actual ODMR signal equations for NV centers in diamond.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.numba_kernels import (
    get_background_ones,
    nv_center_lorentzian_eval,
    nv_center_lorentzian_vectorized_many,
    nv_center_lorentzian_vectorized_many_fast,
    nv_center_lorentzian_vectorized_one_serial,
    nv_center_pseudo_voigt_eval,
    nv_center_pseudo_voigt_vectorized_many,
    nv_center_pseudo_voigt_vectorized_many_fast,
    nv_center_pseudo_voigt_vectorized_one_serial,
    nv_center_zeeman_lorentzian_eval,
    nv_center_zeeman_lorentzian_vectorized_many,
    nv_center_zeeman_lorentzian_vectorized_many_fast,
    nv_center_zeeman_lorentzian_vectorized_one_serial,
)
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec

MIN_K_NP: float = 1.0  # Captures reverse polarization regimes
MAX_K_NP: float = 5.0  # Captures high asymmetric polarization regimes

# Zeeman splitting bounds (half-separation between the two main dips).
# At γ_NV ≈ 28 GHz/T, 100 MHz ≈ 3.6 mT — a common weak-field lab range.
MIN_ZEEMAN_SPLIT: float = 0.0    # dips fully overlap at zero field
MAX_ZEEMAN_SPLIT: float = 100e6  # 100 MHz → ~3.6 mT

# N-14 parallel hyperfine coupling constant for NV⁻ in diamond.
# Fixed physical constant used when with_hyperfine_splitting=False:
# the triplet structure is still modeled but with split locked to this value
# and k_np locked to 1.0 (symmetric lines), removing both as free parameters.
NV_N14_HYPERFINE_SPLIT_HZ: float = 2.16e6  # ~2.16 MHz

# Physical ranges for linewidth (HWHM) and hyperfine splitting.
# These constants are shared between the signal generator and the inference
# prior bounds so they can never drift apart.
MIN_LINEWIDTH: float = 200e3  # 200 kHz — lower bound for broader lines
MAX_LINEWIDTH: float = 5.0e6  # 5.0 MHz — handles heavy power broadening and strong dipole dephasing
MIN_SPLIT: float = 3.0e6  # 2.0 MHz — minimum split generated / searched
MAX_SPLIT: float = 8.5e6  # 3.5 MHz — maximum split generated / searched

DEFAULT_NV_CENTER_FREQ_X_MIN = 2.6e9
DEFAULT_NV_CENTER_FREQ_X_MAX = 3.1e9

# Gaussian prior std as a fraction of parameter range (1/10 of range by default)
# Can be configured via NVISION_PRIOR_STD_FRACTION environment variable
import os

from dotenv import load_dotenv

_load_env = load_dotenv()  # Ensure .env is loaded
PRIOR_STD_FRACTION: float = float(os.getenv("NVISION_PRIOR_STD_FRACTION", "0.1"))


@dataclass(frozen=True)
class NVCenterLorentzianSpectrum:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    c_total: float

    @property
    def physical_amplitude(self) -> float:
        """Physical Hz² amplitude (numerator): right peak depth × linewidth²."""
        return (self.c_total / self.k_np) * self.linewidth**2


@dataclass(frozen=True)
class NVCenterLorentzianSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterLorentzianSpectrumUncertainty:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    c_total: float


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


# ---------------------------------------------------------------------------
# Single-dip (no hyperfine splitting) parameter bundle — split and k_np absent.
# Used when NVCenterLorentzianModel(with_hyperfine_splitting=False).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVCenterLorentzianSingleDipSpectrum:
    frequency: float
    linewidth: float
    c_total: float


@dataclass(frozen=True)
class NVCenterLorentzianSingleDipSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterLorentzianSingleDipSpectrumUncertainty:
    frequency: float
    linewidth: float
    c_total: float


class _NVCenterLorentzianSingleDipSpec(
    GenericParamSpec[
        NVCenterLorentzianSingleDipSpectrum,
        NVCenterLorentzianSingleDipSpectrumSamples,
        NVCenterLorentzianSingleDipSpectrumUncertainty,
    ]
):
    params_cls = NVCenterLorentzianSingleDipSpectrum
    samples_cls = NVCenterLorentzianSingleDipSpectrumSamples
    uncertainty_cls = NVCenterLorentzianSingleDipSpectrumUncertainty


# ---------------------------------------------------------------------------
# Zeeman-split parameter bundles (no hyperfine inference) — 4 params.
# Used when NVCenterLorentzianModel(with_zeeman_splitting=True, with_hyperfine_splitting=False).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVCenterLorentzianZeemanSpectrum:
    frequency: float
    linewidth: float
    zeeman_split: float
    c_total: float


@dataclass(frozen=True)
class NVCenterLorentzianZeemanSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    zeeman_split: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterLorentzianZeemanSpectrumUncertainty:
    frequency: float
    linewidth: float
    zeeman_split: float
    c_total: float


class _NVCenterLorentzianZeemanSpec(
    GenericParamSpec[
        NVCenterLorentzianZeemanSpectrum,
        NVCenterLorentzianZeemanSpectrumSamples,
        NVCenterLorentzianZeemanSpectrumUncertainty,
    ]
):
    params_cls = NVCenterLorentzianZeemanSpectrum
    samples_cls = NVCenterLorentzianZeemanSpectrumSamples
    uncertainty_cls = NVCenterLorentzianZeemanSpectrumUncertainty


# ---------------------------------------------------------------------------
# Zeeman + hyperfine parameter bundle — 6 params.
# Used when NVCenterLorentzianModel(with_zeeman_splitting=True, with_hyperfine_splitting=True).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVCenterLorentzianZeemanHyperfineSpectrum:
    frequency: float
    linewidth: float
    zeeman_split: float
    split: float
    k_np: float
    c_total: float


@dataclass(frozen=True)
class NVCenterLorentzianZeemanHyperfineSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    zeeman_split: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterLorentzianZeemanHyperfineSpectrumUncertainty:
    frequency: float
    linewidth: float
    zeeman_split: float
    split: float
    k_np: float
    c_total: float


class _NVCenterLorentzianZeemanHyperfineSpec(
    GenericParamSpec[
        NVCenterLorentzianZeemanHyperfineSpectrum,
        NVCenterLorentzianZeemanHyperfineSpectrumSamples,
        NVCenterLorentzianZeemanHyperfineSpectrumUncertainty,
    ]
):
    params_cls = NVCenterLorentzianZeemanHyperfineSpectrum
    samples_cls = NVCenterLorentzianZeemanHyperfineSpectrumSamples
    uncertainty_cls = NVCenterLorentzianZeemanHyperfineSpectrumUncertainty


class NVCenterLorentzianModel(
    SignalModel[
        NVCenterLorentzianSpectrum,
        NVCenterLorentzianSpectrumSamples,
        NVCenterLorentzianSpectrumUncertainty,
    ]
):
    """NV center ODMR signal model — single dip by default, triple dips with hyperfine splitting.

    Pass ``with_hyperfine_splitting=True`` to infer split and k_np as free parameters.
    The default (no splitting) models the ODMR spectrum as a single Lorentzian dip,
    parameterised by frequency, linewidth, and c_total.

    With hyperfine splitting enabled the model has three Lorentzian dips:
        S(f) = 1 - L_left - L_center - L_right
    where the outer peaks are displaced by ±split from the centre frequency.
    """

    _SPEC_FULL = _NVCenterLorentzianSpec()
    _SPEC_SINGLE = _NVCenterLorentzianSingleDipSpec()
    _SPEC_ZEEMAN = _NVCenterLorentzianZeemanSpec()
    _SPEC_ZEEMAN_HF = _NVCenterLorentzianZeemanHyperfineSpec()

    def __init__(self, with_hyperfine_splitting: bool = True, with_zeeman_splitting: bool = False) -> None:
        self._with_hyperfine_splitting = with_hyperfine_splitting
        self._with_zeeman_splitting = with_zeeman_splitting

    @staticmethod
    def compute_nvcenter_lorentzian_model(
        x: float,
        frequency: float,
        linewidth: float,
        split: float,
        k_np: float,
        c_total: float,
    ) -> float:
        """Triple Lorentzian NV ODMR; parameter order matches :meth:`parameter_names`."""
        return nv_center_lorentzian_eval(
            float(x),
            float(frequency),
            float(linewidth),
            float(split),
            float(k_np),
            float(c_total),
            1.0,
        )

    def compute_nvcenter_lorentzian_model_vectorized(
        self,
        x: float,
        frequency: np.ndarray,
        linewidth: np.ndarray,
        split: np.ndarray,
        k_np: np.ndarray,
        c_total: np.ndarray,
    ) -> np.ndarray:
        """Vectorized triple-Lorentzian NV evaluation for one probe location."""
        freq = np.asarray(frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        out = np.empty(n, dtype=FLOAT_DTYPE)
        nv_center_lorentzian_vectorized_one_serial(
            float(x),
            freq,
            np.asarray(linewidth, dtype=FLOAT_DTYPE),
            np.asarray(split, dtype=FLOAT_DTYPE),
            np.asarray(k_np, dtype=FLOAT_DTYPE),
            np.asarray(c_total, dtype=FLOAT_DTYPE),
            get_background_ones(n),
            out,
        )
        return out

    @property
    def spec(self):
        if self._with_zeeman_splitting:
            return self._SPEC_ZEEMAN_HF if self._with_hyperfine_splitting else self._SPEC_ZEEMAN
        return self._SPEC_FULL if self._with_hyperfine_splitting else self._SPEC_SINGLE

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "c_total")

    def parameter_weights(self) -> dict[str, float]:
        if self._with_zeeman_splitting and self._with_hyperfine_splitting:
            return {"frequency": 2.0, "linewidth": 1.0, "zeeman_split": 1.5, "split": 1.0, "k_np": 1.0, "c_total": 1.0}
        if self._with_zeeman_splitting:
            return {"frequency": 2.0, "linewidth": 1.0, "zeeman_split": 1.5, "c_total": 1.0}
        if self._with_hyperfine_splitting:
            return {"frequency": 2.0, "linewidth": 1.0, "split": 1.0, "k_np": 1.0, "c_total": 1.0}
        return {"frequency": 2.0, "linewidth": 1.0, "c_total": 1.0}

    def signal_min_span(self, domain_width: float) -> float | None:
        return 4.0 * domain_width * 0.0001

    def signal_max_span(self, domain_width: float) -> float | None:
        linewidth_hi = domain_width * 0.05
        if self._with_zeeman_splitting:
            hf_hi = MAX_SPLIT if self._with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
            return 2.0 * MAX_ZEEMAN_SPLIT + 2.0 * hf_hi + 4.0 * linewidth_hi
        if self._with_hyperfine_splitting:
            return 2.0 * MAX_SPLIT + 4.0 * linewidth_hi
        return 2.0 * NV_N14_HYPERFINE_SPLIT_HZ + 4.0 * linewidth_hi

    def expected_dip_count(self) -> int:
        return 2 if self._with_zeeman_splitting else 3

    def _hf_arrays(self, n: int, samples=None) -> tuple[np.ndarray, np.ndarray]:
        """Return (hf_split_arr, k_np_arr) for n particles."""
        if self._with_hyperfine_splitting and samples is not None:
            return (
                np.asarray(samples.split, dtype=FLOAT_DTYPE),
                np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            )
        return (
            np.full(n, NV_N14_HYPERFINE_SPLIT_HZ, dtype=FLOAT_DTYPE),
            np.ones(n, dtype=FLOAT_DTYPE),
        )

    def compute(self, x: float, params) -> float:
        hf_split = params.split if self._with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        k_np = params.k_np if self._with_hyperfine_splitting else 1.0
        if self._with_zeeman_splitting:
            return nv_center_zeeman_lorentzian_eval(
                float(x), params.frequency, params.linewidth,
                params.zeeman_split, hf_split, k_np, params.c_total, 1.0,
            )
        return self.compute_nvcenter_lorentzian_model(
            float(x), params.frequency, params.linewidth, hf_split, k_np, params.c_total
        )

    def compute_vectorized_samples(self, x: float, samples) -> np.ndarray:
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples)
        out = np.empty(n, dtype=FLOAT_DTYPE)
        if self._with_zeeman_splitting:
            nv_center_zeeman_lorentzian_vectorized_one_serial(
                float(x), freq,
                np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
                np.asarray(samples.zeeman_split, dtype=FLOAT_DTYPE),
                hf_arr, k_arr,
                np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
                get_background_ones(n), out,
            )
        else:
            nv_center_lorentzian_vectorized_one_serial(
                float(x), freq,
                np.asarray(samples.linewidth, dtype=FLOAT_DTYPE),
                hf_arr, k_arr,
                np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
                get_background_ones(n), out,
            )
        return out

    def compute_vectorized_many(self, x_phys_array: Sequence[float], samples_phys) -> np.ndarray:
        if isinstance(samples_phys, list | tuple):
            samples_phys = self.spec.unpack_samples(samples_phys)  # type: ignore[arg-type]
        elif not hasattr(samples_phys, "frequency"):
            return super().compute_vectorized_many(x_phys_array, samples_phys)  # type: ignore[arg-type]

        xs = np.asarray(x_phys_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_phys_array must be one-dimensional")
        freq = np.asarray(samples_phys.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples_phys)
        out = np.empty((xs.shape[0], n), dtype=FLOAT_DTYPE)

        if self._with_zeeman_splitting:
            nv_center_zeeman_lorentzian_vectorized_many(
                xs, freq,
                np.asarray(samples_phys.linewidth, dtype=FLOAT_DTYPE),
                np.asarray(samples_phys.zeeman_split, dtype=FLOAT_DTYPE),
                hf_arr, k_arr,
                np.asarray(samples_phys.c_total, dtype=FLOAT_DTYPE),
                get_background_ones(n), out,
            )
        else:
            nv_center_lorentzian_vectorized_many(
                xs, freq,
                np.asarray(samples_phys.linewidth, dtype=FLOAT_DTYPE),
                hf_arr, k_arr,
                np.asarray(samples_phys.c_total, dtype=FLOAT_DTYPE),
                get_background_ones(n), out,
            )
        return out

    def compute_vectorized_many_fast(self, x_phys_array: Sequence[float], samples_phys) -> np.ndarray:
        if isinstance(samples_phys, list | tuple):
            samples_phys = self.spec.unpack_samples(samples_phys)  # type: ignore[arg-type]
        elif not hasattr(samples_phys, "frequency"):
            return super().compute_vectorized_many_fast(x_phys_array, samples_phys)  # type: ignore[arg-type]

        xs = np.asarray(x_phys_array, dtype=FLOAT_DTYPE)
        freq = np.asarray(samples_phys.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples_phys)
        out = np.empty((xs.shape[0], n), dtype=FLOAT_DTYPE)

        if self._with_zeeman_splitting:
            nv_center_zeeman_lorentzian_vectorized_many_fast(
                xs, freq,
                np.asarray(samples_phys.linewidth, dtype=FLOAT_DTYPE),
                np.asarray(samples_phys.zeeman_split, dtype=FLOAT_DTYPE),
                hf_arr, k_arr,
                np.asarray(samples_phys.c_total, dtype=FLOAT_DTYPE),
                get_background_ones(n), out,
            )
        else:
            nv_center_lorentzian_vectorized_many_fast(
                xs, freq,
                np.asarray(samples_phys.linewidth, dtype=FLOAT_DTYPE),
                hf_arr, k_arr,
                np.asarray(samples_phys.c_total, dtype=FLOAT_DTYPE),
                get_background_ones(n), out,
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


@dataclass(frozen=True)
class NVCenterVoigtSpectrumUncertainty:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    split: float
    k_np: float
    dip_depth: float


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
        Background level (fixed to 1.0)
    """

    def compute_nvcenter_voigt_model(
        self,
        x: float,
        frequency: float,
        fwhm_total: float,
        lorentz_frac: float,
        split: float,
        k_np: float,
        dip_depth: float,
    ) -> float:
        """Triple Voigt NV ODMR; parameter order matches :meth:`parameter_names`."""
        return nv_center_pseudo_voigt_eval(
            float(x),
            float(frequency),
            float(fwhm_total),
            float(lorentz_frac),
            float(split),
            float(k_np),
            float(dip_depth),
            1.0,
        )

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
        )

    def compute_vectorized_samples(self, x: float, samples: NVCenterVoigtSpectrumSamples) -> np.ndarray:
        """Vectorized Voigt evaluation for a single probe position across all particles."""
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        out = np.empty(n, dtype=FLOAT_DTYPE)
        nv_center_pseudo_voigt_vectorized_one_serial(
            float(x),
            freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            get_background_ones(n),
            out,
        )
        return out

    def compute_vectorized_many_fast(
        self, x_array: Sequence[float], samples: NVCenterVoigtSpectrumSamples
    ) -> np.ndarray:
        """Acquisition-only fast variant: uses the fastmath pseudo-Voigt kernel."""
        if isinstance(samples, list | tuple):
            samples = self.spec.unpack_samples(samples)  # type: ignore[arg-type]
        elif not hasattr(samples, "frequency"):
            return super().compute_vectorized_many_fast(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], freq.shape[0]), dtype=FLOAT_DTYPE)
        nv_center_pseudo_voigt_vectorized_many_fast(
            xs,
            freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            get_background_ones(freq.shape[0]),
            out,
        )
        return out

    def compute_vectorized_many(self, x_array: Sequence[float], samples: NVCenterVoigtSpectrumSamples) -> np.ndarray:
        if isinstance(samples, list | tuple):
            # Raw list of parameter arrays from SMC batch_update — unpack to typed samples.
            samples = self.spec.unpack_samples(samples)  # type: ignore[arg-type]
        elif not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        out = np.empty((xs.shape[0], freq.shape[0]), dtype=FLOAT_DTYPE)
        nv_center_pseudo_voigt_vectorized_many(
            xs,
            freq,
            np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE),
            np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE),
            np.asarray(samples.split, dtype=FLOAT_DTYPE),
            np.asarray(samples.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            get_background_ones(freq.shape[0]),
            out,
        )
        return out


def nv_center_lorentzian_bounds_for_domain(
    x_min: float,
    x_max: float,
    with_hyperfine_splitting: bool = True,
    with_zeeman_splitting: bool = False,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV Lorentzian signals over ``[x_min, x_max]``.

    ``with_zeeman_splitting=True`` adds ``zeeman_split`` and narrows the frequency
    range so the two Zeeman dips always land within the domain.
    ``with_hyperfine_splitting=False`` (default for builders) fixes split/k_np
    to N-14 constants and omits them from the returned dict.
    """
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

    linewidth_bounds = (MIN_LINEWIDTH, max(MAX_LINEWIDTH, width * 0.05))
    linewidth_hi = linewidth_bounds[1]

    if with_zeeman_splitting:
        # Center frequency must stay MAX_ZEEMAN_SPLIT inside each edge.
        zeeman_margin = MAX_ZEEMAN_SPLIT
        f_lo = float(x_min) + zeeman_margin
        f_hi = float(x_max) - zeeman_margin
        zeeman_bounds = (MIN_ZEEMAN_SPLIT, MAX_ZEEMAN_SPLIT)
        hf_hi = MAX_SPLIT if with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        max_span = 2.0 * MAX_ZEEMAN_SPLIT + 2.0 * hf_hi + 4.0 * linewidth_hi

        if with_hyperfine_splitting:
            split_bounds = (MIN_SPLIT, max(MAX_SPLIT, width * 0.02))
            return {
                "frequency": (f_lo, f_hi),
                "linewidth": linewidth_bounds,
                "zeeman_split": zeeman_bounds,
                "split": split_bounds,
                "k_np": (MIN_K_NP, MAX_K_NP),
                "c_total": (0.1, 0.4),
                "_signal_max_span": (0.0, max_span),
            }
        return {
            "frequency": (f_lo, f_hi),
            "linewidth": linewidth_bounds,
            "zeeman_split": zeeman_bounds,
            "c_total": (0.1, 0.4),
            "_signal_max_span": (0.0, max_span),
        }

    # Non-Zeeman cases (existing behaviour preserved).
    if with_hyperfine_splitting:
        split_bounds = (MIN_SPLIT, max(MAX_SPLIT, width * 0.02))
        return {
            "frequency": (float(x_min), float(x_max)),
            "linewidth": linewidth_bounds,
            "split": split_bounds,
            "k_np": (MIN_K_NP, MAX_K_NP),
            "c_total": (0.1, 0.4),
            "_signal_max_span": (0.0, width * 0.1),
        }

    return {
        "frequency": (float(x_min), float(x_max)),
        "linewidth": linewidth_bounds,
        "c_total": (0.1, 0.4),
        "_signal_max_span": (0.0, 2.0 * NV_N14_HYPERFINE_SPLIT_HZ + 4.0 * linewidth_hi),
    }


# ---------------------------------------------------------------------------
# One-peak (zero-field) NV Lorentzian model — split and k_np are fixed to 0/1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVCenterOnePeakLorentzianSpectrum:
    frequency: float
    linewidth: float
    dip_depth: float


@dataclass(frozen=True)
class NVCenterOnePeakLorentzianSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    dip_depth: np.ndarray


@dataclass(frozen=True)
class NVCenterOnePeakLorentzianSpectrumUncertainty:
    frequency: float
    linewidth: float
    dip_depth: float


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
    frequency, linewidth, dip_depth.

    Signal form:
        S(f) = 1.0 - dip_depth * linewidth² / ((f - frequency)² + linewidth²)
    """

    _SPEC = _NVCenterOnePeakLorentzianSpec()

    @property
    def spec(self) -> _NVCenterOnePeakLorentzianSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "dip_depth")

    def parameter_weights(self) -> dict[str, float]:
        return {"frequency": 2.0, "linewidth": 1.0, "dip_depth": 1.0}

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
        return float(1.0 - (params.dip_depth * lw2) / denom)

    def compute_vectorized_samples(self, x: float, samples: NVCenterOnePeakLorentzianSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        lw = np.asarray(samples.linewidth, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        lw2 = lw**2
        denom = (x_f - freq) ** 2 + lw2
        return (1.0 - depth * lw2 / denom).astype(FLOAT_DTYPE, copy=False)

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
        x2d = xs[:, None]
        lw2 = lw[None, :] ** 2
        denom = (x2d - freq[None, :]) ** 2 + lw2
        return (1.0 - depth[None, :] * lw2 / denom).astype(FLOAT_DTYPE, copy=False)


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
        "_signal_max_span": (0.0, 4.0 * linewidth_hi),
    }


def nv_center_voigt_bounds_for_domain(
    x_min: float,
    x_max: float,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV Voigt signals over ``[x_min, x_max]``."""
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

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
        "_signal_max_span": (0.0, max_span),
    }
