"""NV center signal signal based on physics.

These signal implement the actual ODMR signal equations for NV centers in diamond.
"""

from __future__ import annotations

import math
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
    nv_center_zeeman_lorentzian_eval,
    nv_center_zeeman_lorentzian_vectorized_many,
    nv_center_zeeman_lorentzian_vectorized_many_fast,
    nv_center_zeeman_lorentzian_vectorized_one_serial,
    nv_center_zeeman_pseudo_voigt_eval,
    nv_center_zeeman_pseudo_voigt_vectorized_many,
    nv_center_zeeman_pseudo_voigt_vectorized_many_fast,
    nv_center_zeeman_pseudo_voigt_vectorized_one_serial,
)
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec

MIN_K_NP: float = 1.0  # Captures reverse polarization regimes
MAX_K_NP: float = 5.0  # Captures high asymmetric polarization regimes

# Zeeman splitting bounds (half-separation between the two main dips).
# At γ_NV ≈ 28 GHz/T, 60 MHz ≈ 2.1 mT — a common weak-field lab range. Kept small
# relative to the (narrow) NV center frequency domain below so the two Zeeman
# groups stay a visually significant fraction of the plotted domain rather than
# being lost in a much wider empty range.
MIN_ZEEMAN_SPLIT: float = 0.0   # dips fully overlap at zero field
MAX_ZEEMAN_SPLIT: float = 60e6  # 60 MHz → ~2.1 mT

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
MIN_SPLIT: float = 2.0e6  # 2.0 MHz — minimum split generated / searched
MAX_SPLIT: float = 8.5e6  # 8.5 MHz — maximum split generated / searched

# Ceiling for the plain-Voigt fwhm_total axis (Hz), shared by
# nv_center_voigt_bounds_for_domain and NVCenterVoigtModel.signal_max_span so the
# prior bounds and the generator's placement margin can never drift apart. Must
# cover everything NVCenterCoreGenerator's voigt branch can produce:
# fwhm_total = 2*linewidth*(1 + fwhm_gauss/fwhm_lorentz), with linewidth up to
# MAX_LINEWIDTH and the Gaussian share up to lorentz_frac=0.55 in the registered
# presets (ratio ~0.82), i.e. ~2*5MHz*1.82 = 18.2 MHz worst case. 4x MAX_LINEWIDTH
# gives that headroom. (An earlier hardcoded 2.8e6 here silently made 4 of the 5
# width-grid rows unrepresentable, biasing every voigt frequency fit by ~3 MHz.)
VOIGT_FWHM_TOTAL_HI: float = 4.0 * MAX_LINEWIDTH

# Natural (zero-power) homogeneous HWHM for the saturation-coupled Voigt model.
# Fixed rather than inferred: the saturation parameter (drive power) is only
# identifiable from a single spectrum if the unbroadened linewidth is a known
# constant, since gamma_hom = NV_NATURAL_HWHM_HZ * sqrt(1 + saturation) is the
# one free equation relating the observed homogeneous width to the drive.
NV_NATURAL_HWHM_HZ: float = 150e3  # 150 kHz — typical NV T2*-limited natural HWHM

# Gaussian prior std as a fraction of parameter range (1/10 of range by default)
# Can be configured via NVISION_PRIOR_STD_FRACTION environment variable
import os

from dotenv import load_dotenv

_load_env = load_dotenv()  # Ensure .env is loaded
PRIOR_STD_FRACTION: float = float(os.getenv("NVISION_PRIOR_STD_FRACTION", "0.1"))

# Saturated (drive -> infinity) ODMR contrast ceiling for the saturation-coupled
# Voigt model. Fixed rather than inferred: like NV_NATURAL_HWHM_HZ, this is a
# property of the specific NV ensemble/detection setup, not something that
# varies experiment-to-experiment for a given setup, so it is not identifiable
# (nor should it be searched) from a single spectrum -- only the realized
# contrast C = c_max * s/(1+s) is observable, and c_max/saturation are nearly
# degenerate at low saturation (C ~ c_max*s) if both were left free.
# Shares NVISION_SBED_C_MAX with sim/presets.py's saturation-solving formula
# (C = c_max*s/(1+s) => s = C/(c_max-C)), which must agree with the actual
# fixed contrast the model uses -- same env var/default, single physical value.
NV_SATURATION_C_MAX: float = float(os.getenv("NVISION_SBED_C_MAX", "0.5"))

# Real NV⁻ ground-state zero-field splitting (D), ~2.87 GHz -- the physical
# center every Zeeman/hyperfine structure splits around at zero/weak field.
# Fixed (not inferred): like a calibrated instrument constant, not something a
# locator needs to discover from scratch.
NV_ZERO_FIELD_SPLITTING_HZ: float = float(os.getenv("NVISION_NV_ZERO_FIELD_SPLITTING_HZ", "2.87e9"))

# Domain half-width around NV_ZERO_FIELD_SPLITTING_HZ. Only this delta is
# configurable -- x_min/x_max are always symmetric around the physical center,
# so the two can never drift out of sync. Sized off MAX_ZEEMAN_SPLIT (the same
# ~2.1 mT reasonable-experiment field above) rather than an arbitrary round
# number: 2.5x comfortably clears MAX_ZEEMAN_SPLIT plus hyperfine/linewidth
# margin, so signal generation never degrades to a single, non-varying
# center_freq (see NVCenterCoreGenerator).
NV_CENTER_FREQ_DELTA_HZ: float = float(os.getenv("NVISION_NV_CENTER_FREQ_DELTA_HZ", str(2.5 * MAX_ZEEMAN_SPLIT)))

DEFAULT_NV_CENTER_FREQ_X_MIN = NV_ZERO_FIELD_SPLITTING_HZ - NV_CENTER_FREQ_DELTA_HZ
DEFAULT_NV_CENTER_FREQ_X_MAX = NV_ZERO_FIELD_SPLITTING_HZ + NV_CENTER_FREQ_DELTA_HZ


def physics_config_fingerprint() -> str:
    """Short hash of every physical bound/constant that affects what
    :class:`~nvision.sim.gen.nv_center_generator.NVCenterCoreGenerator` draws.

    Stamped into every generated ``true_params`` block (see ``runner/plots.py``)
    so cached scan/plot artifacts on disk can self-report when they were built
    under different physics config (env vars or these module constants changed)
    than what's currently running -- e.g. ``nvision cache clean-manifest
    --stale-physics`` diffs this against a stored fingerprint to find artifacts
    that silently drifted out of sync with the code instead of staying wrong
    forever because no run group happened to touch them again.
    """
    import hashlib

    values = (
        MIN_K_NP,
        MAX_K_NP,
        MIN_ZEEMAN_SPLIT,
        MAX_ZEEMAN_SPLIT,
        NV_N14_HYPERFINE_SPLIT_HZ,
        MIN_LINEWIDTH,
        MAX_LINEWIDTH,
        MIN_SPLIT,
        MAX_SPLIT,
        NV_NATURAL_HWHM_HZ,
        PRIOR_STD_FRACTION,
        NV_ZERO_FIELD_SPLITTING_HZ,
        NV_CENTER_FREQ_DELTA_HZ,
    )
    digest = hashlib.sha256(repr(values).encode()).hexdigest()
    return digest[:12]


PHYSICS_CONFIG_FINGERPRINT: str = physics_config_fingerprint()


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
    _SPEC_FULL_FIXED_FREQ = _NVCenterLorentzianSpec(fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ})
    _SPEC_SINGLE_FIXED_FREQ = _NVCenterLorentzianSingleDipSpec(fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ})
    _SPEC_ZEEMAN_FIXED_FREQ = _NVCenterLorentzianZeemanSpec(fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ})
    _SPEC_ZEEMAN_HF_FIXED_FREQ = _NVCenterLorentzianZeemanHyperfineSpec(
        fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ}
    )

    def __init__(
        self,
        with_hyperfine_splitting: bool = True,
        with_zeeman_splitting: bool = False,
        with_fixed_frequency: bool = True,
    ) -> None:
        self._with_hyperfine_splitting = with_hyperfine_splitting
        self._with_zeeman_splitting = with_zeeman_splitting
        self._with_fixed_frequency = with_fixed_frequency

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
            base = self._SPEC_ZEEMAN_HF if self._with_hyperfine_splitting else self._SPEC_ZEEMAN
            fixed = self._SPEC_ZEEMAN_HF_FIXED_FREQ if self._with_hyperfine_splitting else self._SPEC_ZEEMAN_FIXED_FREQ
        else:
            base = self._SPEC_FULL if self._with_hyperfine_splitting else self._SPEC_SINGLE
            fixed = self._SPEC_FULL_FIXED_FREQ if self._with_hyperfine_splitting else self._SPEC_SINGLE_FIXED_FREQ
        return fixed if self._with_fixed_frequency else base

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "c_total")

    def parameter_weights(self) -> dict[str, float]:
        freq_w = {} if self._with_fixed_frequency else {"frequency": 2.0}
        if self._with_zeeman_splitting and self._with_hyperfine_splitting:
            return {**freq_w, "linewidth": 1.0, "zeeman_split": 1.5, "split": 1.0, "k_np": 1.0, "c_total": 1.0}
        if self._with_zeeman_splitting:
            return {**freq_w, "linewidth": 1.0, "zeeman_split": 1.5, "c_total": 1.0}
        if self._with_hyperfine_splitting:
            return {**freq_w, "linewidth": 1.0, "split": 1.0, "k_np": 1.0, "c_total": 1.0}
        return {**freq_w, "linewidth": 1.0, "c_total": 1.0}

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
        """Each dip is individually resolved (no broadening merges lines), so
        hyperfine splitting multiplies the Zeeman-group count by 3."""
        if self._with_zeeman_splitting:
            return 6 if self._with_hyperfine_splitting else 2
        return 3 if self._with_hyperfine_splitting else 1

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


# ---------------------------------------------------------------------------
# Voigt parameter bundles — physically-decomposed width, mirroring
# NVCenterLorentzianModel's own (with_zeeman_splitting, with_hyperfine_splitting)
# four-way split exactly:
#   homogeneous_linewidth (Hz, HWHM) + sigma_inhom (Hz, Gaussian inhomogeneous
#   width) replace the old kernel-native (fwhm_total, lorentz_frac) pair —
#   reparameterized via _voigt_reparam_scalar/_voigt_reparam right before the
#   shared pseudo-Voigt kernel, the same pattern NVCenterSaturationVoigtModel
#   already uses (minus that model's saturation-law amplitude coupling).
#   c_total (population-normalized, cannot go negative) replaces dip_depth as
#   the amplitude parameter, matching Lorentzian/saturation-Voigt.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVCenterVoigtSingleDipSpectrum:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    c_total: float


@dataclass(frozen=True)
class NVCenterVoigtSingleDipSpectrumSamples:
    frequency: np.ndarray
    homogeneous_linewidth: np.ndarray
    sigma_inhom: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterVoigtSingleDipSpectrumUncertainty:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    c_total: float


class _NVCenterVoigtSingleDipSpec(
    GenericParamSpec[
        NVCenterVoigtSingleDipSpectrum,
        NVCenterVoigtSingleDipSpectrumSamples,
        NVCenterVoigtSingleDipSpectrumUncertainty,
    ]
):
    params_cls = NVCenterVoigtSingleDipSpectrum
    samples_cls = NVCenterVoigtSingleDipSpectrumSamples
    uncertainty_cls = NVCenterVoigtSingleDipSpectrumUncertainty


@dataclass(frozen=True)
class NVCenterVoigtSpectrum:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    split: float
    k_np: float
    c_total: float


@dataclass(frozen=True)
class NVCenterVoigtSpectrumSamples:
    frequency: np.ndarray
    homogeneous_linewidth: np.ndarray
    sigma_inhom: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterVoigtSpectrumUncertainty:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    split: float
    k_np: float
    c_total: float


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


@dataclass(frozen=True)
class NVCenterVoigtZeemanSpectrum:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    zeeman_split: float
    c_total: float


@dataclass(frozen=True)
class NVCenterVoigtZeemanSpectrumSamples:
    frequency: np.ndarray
    homogeneous_linewidth: np.ndarray
    sigma_inhom: np.ndarray
    zeeman_split: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterVoigtZeemanSpectrumUncertainty:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    zeeman_split: float
    c_total: float


class _NVCenterVoigtZeemanSpec(
    GenericParamSpec[
        NVCenterVoigtZeemanSpectrum,
        NVCenterVoigtZeemanSpectrumSamples,
        NVCenterVoigtZeemanSpectrumUncertainty,
    ]
):
    params_cls = NVCenterVoigtZeemanSpectrum
    samples_cls = NVCenterVoigtZeemanSpectrumSamples
    uncertainty_cls = NVCenterVoigtZeemanSpectrumUncertainty


@dataclass(frozen=True)
class NVCenterVoigtZeemanHyperfineSpectrum:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    zeeman_split: float
    split: float
    k_np: float
    c_total: float


@dataclass(frozen=True)
class NVCenterVoigtZeemanHyperfineSpectrumSamples:
    frequency: np.ndarray
    homogeneous_linewidth: np.ndarray
    sigma_inhom: np.ndarray
    zeeman_split: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    c_total: np.ndarray


@dataclass(frozen=True)
class NVCenterVoigtZeemanHyperfineSpectrumUncertainty:
    frequency: float
    homogeneous_linewidth: float
    sigma_inhom: float
    zeeman_split: float
    split: float
    k_np: float
    c_total: float


class _NVCenterVoigtZeemanHyperfineSpec(
    GenericParamSpec[
        NVCenterVoigtZeemanHyperfineSpectrum,
        NVCenterVoigtZeemanHyperfineSpectrumSamples,
        NVCenterVoigtZeemanHyperfineSpectrumUncertainty,
    ]
):
    params_cls = NVCenterVoigtZeemanHyperfineSpectrum
    samples_cls = NVCenterVoigtZeemanHyperfineSpectrumSamples
    uncertainty_cls = NVCenterVoigtZeemanHyperfineSpectrumUncertainty


class NVCenterVoigtModel(
    SignalModel[NVCenterVoigtSpectrum, NVCenterVoigtSpectrumSamples, NVCenterVoigtSpectrumUncertainty]
):
    """NV center with Gaussian broadening (Voigt profile).

    Uses a two-width pseudo-Voigt approximation (see the shape-only accuracy check in
    ``tests/spectra/test_pseudo_voigt_accuracy.py``, which exercises
    :func:`~nvision.spectra.numba_kernels.nv_center_pseudo_voigt_eval` directly), not a true
    Voigt profile (no ``wofz``/error-function evaluation). The Lorentzian and Gaussian
    components are evaluated at the shared combined ``fwhm_total``, matching the standard
    Thompson-Cox-Hastings mixing weight's calibration; see
    ``tests/spectra/test_pseudo_voigt_accuracy.py`` for the residual approximation error.

    Mirrors :class:`NVCenterSaturationVoigtModel` structurally: always evaluated via the
    population-normalized Zeeman pseudo-Voigt kernel (``zeeman_split=0`` when Zeeman splitting
    is disabled), reparameterizing the physically-decomposed width
    ``(homogeneous_linewidth, sigma_inhom)`` into the kernel-native ``(fwhm_total, lorentz_frac)``
    via :func:`_voigt_reparam_scalar`/:func:`_voigt_reparam` right before evaluation — the same
    pattern :func:`_saturation_voigt_reparam_scalar` uses, but without that model's
    saturation-law amplitude coupling: ``c_total`` here is a directly free parameter, not derived.

    Parameters (``with_zeeman_splitting=True, with_hyperfine_splitting=True``)
    ----------------------------------------------------------------------
    frequency : float
        Central frequency f_B in Hz.
    homogeneous_linewidth : float
        Lorentzian (homogeneous) HWHM in Hz.
    sigma_inhom : float
        Inhomogeneous (Gaussian) broadening width in Hz. ``sigma_inhom -> 0`` is the pure
        Lorentzian limit (``lorentz_frac -> 1``).
    zeeman_split : float
        Half-separation between the two Zeeman groups in Hz (0 when
        ``with_zeeman_splitting=False``).
    split : float
        Hyperfine splitting in Hz (fixed to the N-14 constant when
        ``with_hyperfine_splitting=False``).
    k_np : float
        Non-polarization factor (fixed to 1.0 when hyperfine is disabled).
    c_total : float
        Population-normalized total contrast (same convention as
        :class:`NVCenterLorentzianModel`/:class:`NVCenterSaturationVoigtModel`; cannot go negative).
    background : float
        Background level (fixed to 1.0).
    """

    _SPEC_FULL = _NVCenterVoigtSpec()
    _SPEC_SINGLE = _NVCenterVoigtSingleDipSpec()
    _SPEC_ZEEMAN = _NVCenterVoigtZeemanSpec()
    _SPEC_ZEEMAN_HF = _NVCenterVoigtZeemanHyperfineSpec()
    _SPEC_FULL_FIXED_FREQ = _NVCenterVoigtSpec(fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ})
    _SPEC_SINGLE_FIXED_FREQ = _NVCenterVoigtSingleDipSpec(fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ})
    _SPEC_ZEEMAN_FIXED_FREQ = _NVCenterVoigtZeemanSpec(fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ})
    _SPEC_ZEEMAN_HF_FIXED_FREQ = _NVCenterVoigtZeemanHyperfineSpec(
        fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ}
    )

    def __init__(
        self,
        with_hyperfine_splitting: bool = False,
        with_zeeman_splitting: bool = False,
        with_fixed_frequency: bool = True,
    ) -> None:
        self._with_hyperfine_splitting = with_hyperfine_splitting
        self._with_zeeman_splitting = with_zeeman_splitting
        self._with_fixed_frequency = with_fixed_frequency

    @property
    def spec(self):
        if self._with_zeeman_splitting:
            base = self._SPEC_ZEEMAN_HF if self._with_hyperfine_splitting else self._SPEC_ZEEMAN
            fixed = self._SPEC_ZEEMAN_HF_FIXED_FREQ if self._with_hyperfine_splitting else self._SPEC_ZEEMAN_FIXED_FREQ
        else:
            base = self._SPEC_FULL if self._with_hyperfine_splitting else self._SPEC_SINGLE
            fixed = self._SPEC_FULL_FIXED_FREQ if self._with_hyperfine_splitting else self._SPEC_SINGLE_FIXED_FREQ
        return fixed if self._with_fixed_frequency else base

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("homogeneous_linewidth", "sigma_inhom", "c_total")

    def parameter_weights(self) -> dict[str, float]:
        freq_w = {} if self._with_fixed_frequency else {"frequency": 2.0}
        if self._with_zeeman_splitting and self._with_hyperfine_splitting:
            return {
                **freq_w,
                "homogeneous_linewidth": 1.0,
                "sigma_inhom": 1.0,
                "zeeman_split": 1.5,
                "split": 1.0,
                "k_np": 1.0,
                "c_total": 1.0,
            }
        if self._with_zeeman_splitting:
            return {
                **freq_w,
                "homogeneous_linewidth": 1.0,
                "sigma_inhom": 1.0,
                "zeeman_split": 1.5,
                "c_total": 1.0,
            }
        if self._with_hyperfine_splitting:
            return {
                **freq_w,
                "homogeneous_linewidth": 1.0,
                "sigma_inhom": 1.0,
                "split": 1.0,
                "k_np": 1.0,
                "c_total": 1.0,
            }
        return {**freq_w, "homogeneous_linewidth": 1.0, "sigma_inhom": 1.0, "c_total": 1.0}

    def signal_min_span(self, domain_width: float) -> float | None:
        fwhm_total_lo = 70e3
        return 2.0 * fwhm_total_lo

    def signal_max_span(self, domain_width: float) -> float | None:
        hf_hi = MAX_SPLIT if self._with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        if self._with_zeeman_splitting:
            # Mirror nv_center_voigt_bounds_for_domain exactly or the placement margin
            # stops covering what the prior/fit is allowed to represent.
            return 2.0 * MAX_ZEEMAN_SPLIT + 2.0 * hf_hi + 4.0 * VOIGT_FWHM_TOTAL_HI
        return 2.0 * hf_hi + 4.0 * VOIGT_FWHM_TOTAL_HI

    def expected_dip_count(self) -> int:
        """Zeeman splitting produces two resolvable groups; otherwise one hyperfine triplet."""
        return 2 if self._with_zeeman_splitting else 1

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

    def _zeeman_array(self, n: int, samples=None) -> np.ndarray:
        """Return the zeeman_split array (zeros when Zeeman splitting is disabled)."""
        if self._with_zeeman_splitting and samples is not None:
            return np.asarray(samples.zeeman_split, dtype=FLOAT_DTYPE)
        return np.zeros(n, dtype=FLOAT_DTYPE)

    def compute(self, x: float, params) -> float:
        hf_split = params.split if self._with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        k_np = params.k_np if self._with_hyperfine_splitting else 1.0
        zeeman_split = params.zeeman_split if self._with_zeeman_splitting else 0.0
        fwhm_total, lorentz_frac = _voigt_reparam_scalar(params.homogeneous_linewidth, params.sigma_inhom)
        return nv_center_zeeman_pseudo_voigt_eval(
            float(x),
            params.frequency,
            fwhm_total,
            lorentz_frac,
            zeeman_split,
            hf_split,
            k_np,
            params.c_total,
            1.0,
        )

    def compute_vectorized_samples(self, x: float, samples) -> np.ndarray:
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples)
        zeeman_arr = self._zeeman_array(n, samples)
        fwhm_total, lorentz_frac = _voigt_reparam(samples.homogeneous_linewidth, samples.sigma_inhom)
        out = np.empty(n, dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_one_serial(
            float(x),
            freq,
            fwhm_total,
            lorentz_frac,
            zeeman_arr,
            hf_arr,
            k_arr,
            np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
            get_background_ones(n),
            out,
        )
        return out

    def compute_vectorized_many(self, x_array: Sequence[float], samples) -> np.ndarray:
        if isinstance(samples, list | tuple):
            samples = self.spec.unpack_samples(samples)  # type: ignore[arg-type]
        elif not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples)
        zeeman_arr = self._zeeman_array(n, samples)
        fwhm_total, lorentz_frac = _voigt_reparam(samples.homogeneous_linewidth, samples.sigma_inhom)
        out = np.empty((xs.shape[0], n), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many(
            xs,
            freq,
            fwhm_total,
            lorentz_frac,
            zeeman_arr,
            hf_arr,
            k_arr,
            np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
            get_background_ones(n),
            out,
        )
        return out

    def compute_vectorized_many_fast(self, x_array: Sequence[float], samples) -> np.ndarray:
        """Acquisition-only fast variant: uses the fastmath pseudo-Voigt kernel."""
        if isinstance(samples, list | tuple):
            samples = self.spec.unpack_samples(samples)  # type: ignore[arg-type]
        elif not hasattr(samples, "frequency"):
            return super().compute_vectorized_many_fast(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples)
        zeeman_arr = self._zeeman_array(n, samples)
        fwhm_total, lorentz_frac = _voigt_reparam(samples.homogeneous_linewidth, samples.sigma_inhom)
        out = np.empty((xs.shape[0], n), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many_fast(
            xs,
            freq,
            fwhm_total,
            lorentz_frac,
            zeeman_arr,
            hf_arr,
            k_arr,
            np.asarray(samples.c_total, dtype=FLOAT_DTYPE),
            get_background_ones(n),
            out,
        )
        return out


# ---------------------------------------------------------------------------
# Saturation-coupled Voigt parameter bundles — power broadening and
# inhomogeneous broadening as separate, physically distinct parameters.
#
# Power broadening is a *homogeneous* effect coupled to the same drive that
# sets the contrast: at saturation parameter s (drive power / half-saturation
# power), the homogeneous (Lorentzian) HWHM broadens as
#     gamma_hom = NV_NATURAL_HWHM_HZ * sqrt(1 + s)
# and the realized population contrast saturates as
#     C = c_max * s / (1 + s)
# so a single ``saturation`` parameter moves width and amplitude together, as
# physically required (they share the same drive). ``sigma_inhom`` is the
# independent inhomogeneous (Gaussian) width — strain, unresolved hyperfine or
# field ensemble spread — with no such coupling.
#
# The lineshape is always a Voigt: reparameterize (s, sigma_inhom, c_max) into
# the existing pseudo-Voigt kernel inputs (fwhm_total, lorentz_frac, c_total)
# via :func:`_saturation_voigt_reparam` / :func:`_saturation_voigt_reparam_scalar`
# and reuse the population-normalized Zeeman pseudo-Voigt kernels — a pure
# Lorentzian is just the sigma_inhom -> 0 limit (lorentz_frac -> 1).
#
# Four variants mirror :class:`NVCenterLorentzianModel` exactly:
#   (with_zeeman_splitting, with_hyperfine_splitting) in
#   {(F,F), (F,T), (T,F), (T,T)}.
# ---------------------------------------------------------------------------

_SATURATION_VOIGT_SQRT2LOG2 = math.sqrt(2.0 * math.log(2.0))


def _saturation_voigt_reparam_scalar(saturation: float, sigma_inhom: float, c_max: float) -> tuple[float, float, float]:
    """Scalar ``(saturation, sigma_inhom, c_max) -> (fwhm_total, lorentz_frac, c_total)``."""
    gamma_hom = NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + saturation)
    fwhm_l = 2.0 * gamma_hom
    fwhm_g = 2.0 * _SATURATION_VOIGT_SQRT2LOG2 * sigma_inhom
    fwhm_total = fwhm_l + fwhm_g
    lorentz_frac = fwhm_l / fwhm_total if fwhm_total > 1e-12 else 1.0
    c_total = c_max * saturation / (1.0 + saturation)
    return fwhm_total, lorentz_frac, c_total


def _saturation_voigt_reparam(
    saturation: np.ndarray, sigma_inhom: np.ndarray, c_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``(saturation, sigma_inhom, c_max) -> (fwhm_total, lorentz_frac, c_total)``."""
    s = np.asarray(saturation, dtype=FLOAT_DTYPE)
    sig = np.asarray(sigma_inhom, dtype=FLOAT_DTYPE)
    cmax = np.asarray(c_max, dtype=FLOAT_DTYPE)
    gamma_hom = NV_NATURAL_HWHM_HZ * np.sqrt(1.0 + s)
    fwhm_l = 2.0 * gamma_hom
    fwhm_g = (2.0 * _SATURATION_VOIGT_SQRT2LOG2) * sig
    fwhm_total = fwhm_l + fwhm_g
    lorentz_frac = np.where(fwhm_total > 1e-12, fwhm_l / np.maximum(fwhm_total, 1e-12), 1.0)
    c_total = cmax * s / (1.0 + s)
    return (
        fwhm_total.astype(FLOAT_DTYPE, copy=False),
        lorentz_frac.astype(FLOAT_DTYPE, copy=False),
        c_total.astype(FLOAT_DTYPE, copy=False),
    )


def saturation_voigt_effective_hwhm_and_unc(
    saturation: float,
    sigma_inhom: float,
    sigma_saturation: float = 0.0,
    sigma_sigma_inhom: float = 0.0,
) -> tuple[float, float]:
    """Effective HWHM ``omega`` (Hz) and its propagated uncertainty for the saturation-Voigt model.

    ``omega(s, sigma_inhom) = gamma0*sqrt(1+s) + sqrt(2 ln2)*sigma_inhom`` is nonlinear in
    ``s``, so the two independent input uncertainties are combined in quadrature via the
    local Jacobian (``domega/ds = gamma0/(2*sqrt(1+s))``, ``domega/dsigma = sqrt(2 ln2)``)
    rather than a direct bound-range rescale.
    """
    s = max(saturation, 0.0)
    omega = NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + s) + _SATURATION_VOIGT_SQRT2LOG2 * sigma_inhom
    domega_ds = NV_NATURAL_HWHM_HZ / (2.0 * math.sqrt(1.0 + s))
    sigma_omega = math.sqrt(
        (domega_ds * sigma_saturation) ** 2 + (_SATURATION_VOIGT_SQRT2LOG2 * sigma_sigma_inhom) ** 2
    )
    return omega, sigma_omega


def saturation_voigt_realized_contrast_and_unc(
    saturation: float,
    c_max: float,
    sigma_saturation: float = 0.0,
    sigma_c_max: float = 0.0,
) -> tuple[float, float]:
    """Realized contrast ``C = c_max * s/(1+s)`` (saturation law) and its propagated uncertainty.

    Jacobian quadrature with ``dC/dc_max = s/(1+s)`` and ``dC/ds = c_max/(1+s)^2``.
    """
    s = max(saturation, 0.0)
    c_total = c_max * s / (1.0 + s)
    dc_dcmax = s / (1.0 + s)
    dc_ds = c_max / (1.0 + s) ** 2
    sigma_c = math.sqrt((dc_dcmax * sigma_c_max) ** 2 + (dc_ds * sigma_saturation) ** 2)
    return c_total, sigma_c


def _voigt_reparam_scalar(homogeneous_linewidth: float, sigma_inhom: float) -> tuple[float, float]:
    """Scalar ``(homogeneous_linewidth, sigma_inhom) -> (fwhm_total, lorentz_frac)``.

    Same Voigt-width algebra as :func:`_saturation_voigt_reparam_scalar`, but without that
    function's saturation-law amplitude coupling: ``c_total`` is a directly free parameter
    for plain Voigt, not derived from ``homogeneous_linewidth``.
    """
    fwhm_l = 2.0 * homogeneous_linewidth
    fwhm_g = 2.0 * _SATURATION_VOIGT_SQRT2LOG2 * sigma_inhom
    fwhm_total = fwhm_l + fwhm_g
    lorentz_frac = fwhm_l / fwhm_total if fwhm_total > 1e-12 else 1.0
    return fwhm_total, lorentz_frac


def _voigt_reparam(homogeneous_linewidth: np.ndarray, sigma_inhom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``(homogeneous_linewidth, sigma_inhom) -> (fwhm_total, lorentz_frac)``."""
    hl = np.asarray(homogeneous_linewidth, dtype=FLOAT_DTYPE)
    sig = np.asarray(sigma_inhom, dtype=FLOAT_DTYPE)
    fwhm_l = 2.0 * hl
    fwhm_g = (2.0 * _SATURATION_VOIGT_SQRT2LOG2) * sig
    fwhm_total = fwhm_l + fwhm_g
    lorentz_frac = np.where(fwhm_total > 1e-12, fwhm_l / np.maximum(fwhm_total, 1e-12), 1.0)
    return (
        fwhm_total.astype(FLOAT_DTYPE, copy=False),
        lorentz_frac.astype(FLOAT_DTYPE, copy=False),
    )


@dataclass(frozen=True)
class NVCenterSaturationVoigtSingleDipSpectrum:
    frequency: float
    saturation: float
    sigma_inhom: float


@dataclass(frozen=True)
class NVCenterSaturationVoigtSingleDipSpectrumSamples:
    frequency: np.ndarray
    saturation: np.ndarray
    sigma_inhom: np.ndarray


@dataclass(frozen=True)
class NVCenterSaturationVoigtSingleDipSpectrumUncertainty:
    frequency: float
    saturation: float
    sigma_inhom: float


class _NVCenterSaturationVoigtSingleDipSpec(
    GenericParamSpec[
        NVCenterSaturationVoigtSingleDipSpectrum,
        NVCenterSaturationVoigtSingleDipSpectrumSamples,
        NVCenterSaturationVoigtSingleDipSpectrumUncertainty,
    ]
):
    params_cls = NVCenterSaturationVoigtSingleDipSpectrum
    samples_cls = NVCenterSaturationVoigtSingleDipSpectrumSamples
    uncertainty_cls = NVCenterSaturationVoigtSingleDipSpectrumUncertainty


@dataclass(frozen=True)
class NVCenterSaturationVoigtSpectrum:
    frequency: float
    saturation: float
    sigma_inhom: float
    split: float
    k_np: float


@dataclass(frozen=True)
class NVCenterSaturationVoigtSpectrumSamples:
    frequency: np.ndarray
    saturation: np.ndarray
    sigma_inhom: np.ndarray
    split: np.ndarray
    k_np: np.ndarray


@dataclass(frozen=True)
class NVCenterSaturationVoigtSpectrumUncertainty:
    frequency: float
    saturation: float
    sigma_inhom: float
    split: float
    k_np: float


class _NVCenterSaturationVoigtSpec(
    GenericParamSpec[
        NVCenterSaturationVoigtSpectrum,
        NVCenterSaturationVoigtSpectrumSamples,
        NVCenterSaturationVoigtSpectrumUncertainty,
    ]
):
    params_cls = NVCenterSaturationVoigtSpectrum
    samples_cls = NVCenterSaturationVoigtSpectrumSamples
    uncertainty_cls = NVCenterSaturationVoigtSpectrumUncertainty


@dataclass(frozen=True)
class NVCenterSaturationVoigtZeemanSpectrum:
    frequency: float
    saturation: float
    sigma_inhom: float
    zeeman_split: float


@dataclass(frozen=True)
class NVCenterSaturationVoigtZeemanSpectrumSamples:
    frequency: np.ndarray
    saturation: np.ndarray
    sigma_inhom: np.ndarray
    zeeman_split: np.ndarray


@dataclass(frozen=True)
class NVCenterSaturationVoigtZeemanSpectrumUncertainty:
    frequency: float
    saturation: float
    sigma_inhom: float
    zeeman_split: float


class _NVCenterSaturationVoigtZeemanSpec(
    GenericParamSpec[
        NVCenterSaturationVoigtZeemanSpectrum,
        NVCenterSaturationVoigtZeemanSpectrumSamples,
        NVCenterSaturationVoigtZeemanSpectrumUncertainty,
    ]
):
    params_cls = NVCenterSaturationVoigtZeemanSpectrum
    samples_cls = NVCenterSaturationVoigtZeemanSpectrumSamples
    uncertainty_cls = NVCenterSaturationVoigtZeemanSpectrumUncertainty


@dataclass(frozen=True)
class NVCenterSaturationVoigtZeemanHyperfineSpectrum:
    frequency: float
    saturation: float
    sigma_inhom: float
    zeeman_split: float
    split: float
    k_np: float


@dataclass(frozen=True)
class NVCenterSaturationVoigtZeemanHyperfineSpectrumSamples:
    frequency: np.ndarray
    saturation: np.ndarray
    sigma_inhom: np.ndarray
    zeeman_split: np.ndarray
    split: np.ndarray
    k_np: np.ndarray


@dataclass(frozen=True)
class NVCenterSaturationVoigtZeemanHyperfineSpectrumUncertainty:
    frequency: float
    saturation: float
    sigma_inhom: float
    zeeman_split: float
    split: float
    k_np: float


class _NVCenterSaturationVoigtZeemanHyperfineSpec(
    GenericParamSpec[
        NVCenterSaturationVoigtZeemanHyperfineSpectrum,
        NVCenterSaturationVoigtZeemanHyperfineSpectrumSamples,
        NVCenterSaturationVoigtZeemanHyperfineSpectrumUncertainty,
    ]
):
    params_cls = NVCenterSaturationVoigtZeemanHyperfineSpectrum
    samples_cls = NVCenterSaturationVoigtZeemanHyperfineSpectrumSamples
    uncertainty_cls = NVCenterSaturationVoigtZeemanHyperfineSpectrumUncertainty


class NVCenterSaturationVoigtModel(
    SignalModel[
        NVCenterSaturationVoigtSpectrum,
        NVCenterSaturationVoigtSpectrumSamples,
        NVCenterSaturationVoigtSpectrumUncertainty,
    ]
):
    """NV center Voigt model with explicit, physically distinct broadening channels.

    Replaces the single lumped linewidth with two parameters:

    * ``saturation`` (s) — drive power relative to half-saturation. Sets the
      **homogeneous** (Lorentzian) HWHM via ``gamma_hom = NV_NATURAL_HWHM_HZ *
      sqrt(1 + s)`` *and* the realized contrast via ``C = NV_SATURATION_C_MAX *
      s/(1+s)``, since power broadening and the ODMR contrast are driven by the
      same microwave field and must move together.
    * ``sigma_inhom`` — the **inhomogeneous** (Gaussian) width from strain and
      the unresolved hyperfine/field ensemble, independent of drive power.

    The lineshape is always a pseudo-Voigt; a pure Lorentzian is the
    ``sigma_inhom -> 0`` limit. Population is partitioned across dips exactly
    as in :class:`NVCenterLorentzianModel` (``k_np`` across the hyperfine
    triplet, evenly across the two Zeeman groups), using
    :data:`NV_SATURATION_C_MAX` — the saturated (drive -> infinity) contrast —
    as the amplitude scale.

    ``NV_SATURATION_C_MAX`` is a **fixed constant, not an inferred parameter**
    (see its definition): only the realized contrast ``C = c_max*s/(1+s)`` is
    observable from a spectrum, and at low saturation ``C ~ c_max*s`` makes
    ``c_max``/``saturation`` nearly degenerate if both were left free. Treating
    it like a calibrated instrument constant (the same reasoning as
    ``NV_NATURAL_HWHM_HZ``) keeps the model identifiable.

    Parameters (``with_zeeman_splitting=True, with_hyperfine_splitting=True``)
    ----------------------------------------------------------------------
    frequency : float
        Central (zero-field) frequency f_B in Hz.
    saturation : float
        Drive power relative to half-saturation power (dimensionless, > 0).
    sigma_inhom : float
        Inhomogeneous (Gaussian) broadening width in Hz.
    zeeman_split : float
        Half-separation between the two Zeeman groups in Hz (0 when
        ``with_zeeman_splitting=False``).
    split : float
        Hyperfine splitting in Hz (fixed to the N-14 constant when
        ``with_hyperfine_splitting=False``).
    k_np : float
        Non-polarization factor (fixed to 1.0 when hyperfine is disabled).
    """

    _SPEC_FULL = _NVCenterSaturationVoigtSpec()
    _SPEC_SINGLE = _NVCenterSaturationVoigtSingleDipSpec()
    _SPEC_ZEEMAN = _NVCenterSaturationVoigtZeemanSpec()
    _SPEC_ZEEMAN_HF = _NVCenterSaturationVoigtZeemanHyperfineSpec()
    _SPEC_FULL_FIXED_FREQ = _NVCenterSaturationVoigtSpec(fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ})
    _SPEC_SINGLE_FIXED_FREQ = _NVCenterSaturationVoigtSingleDipSpec(
        fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ}
    )
    _SPEC_ZEEMAN_FIXED_FREQ = _NVCenterSaturationVoigtZeemanSpec(
        fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ}
    )
    _SPEC_ZEEMAN_HF_FIXED_FREQ = _NVCenterSaturationVoigtZeemanHyperfineSpec(
        fixed_values={"frequency": NV_ZERO_FIELD_SPLITTING_HZ}
    )

    def __init__(
        self,
        with_hyperfine_splitting: bool = False,
        with_zeeman_splitting: bool = False,
        with_fixed_frequency: bool = True,
    ) -> None:
        self._with_hyperfine_splitting = with_hyperfine_splitting
        self._with_zeeman_splitting = with_zeeman_splitting
        self._with_fixed_frequency = with_fixed_frequency

    @property
    def spec(self):
        if self._with_zeeman_splitting:
            base = self._SPEC_ZEEMAN_HF if self._with_hyperfine_splitting else self._SPEC_ZEEMAN
            fixed = self._SPEC_ZEEMAN_HF_FIXED_FREQ if self._with_hyperfine_splitting else self._SPEC_ZEEMAN_FIXED_FREQ
        else:
            base = self._SPEC_FULL if self._with_hyperfine_splitting else self._SPEC_SINGLE
            fixed = self._SPEC_FULL_FIXED_FREQ if self._with_hyperfine_splitting else self._SPEC_SINGLE_FIXED_FREQ
        return fixed if self._with_fixed_frequency else base

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("saturation", "sigma_inhom")

    def parameter_weights(self) -> dict[str, float]:
        freq_w = {} if self._with_fixed_frequency else {"frequency": 2.0}
        if self._with_zeeman_splitting and self._with_hyperfine_splitting:
            return {
                **freq_w,
                "saturation": 1.0,
                "sigma_inhom": 1.0,
                "zeeman_split": 1.5,
                "split": 1.0,
                "k_np": 1.0,
            }
        if self._with_zeeman_splitting:
            return {
                **freq_w,
                "saturation": 1.0,
                "sigma_inhom": 1.0,
                "zeeman_split": 1.5,
            }
        if self._with_hyperfine_splitting:
            return {
                **freq_w,
                "saturation": 1.0,
                "sigma_inhom": 1.0,
                "split": 1.0,
                "k_np": 1.0,
            }
        return {**freq_w, "saturation": 1.0, "sigma_inhom": 1.0}

    def signal_min_span(self, domain_width: float) -> float | None:
        fwhm_total_lo = 2.0 * NV_NATURAL_HWHM_HZ
        return 2.0 * fwhm_total_lo

    def signal_max_span(self, domain_width: float) -> float | None:
        fwhm_total_hi = 2.8e6
        hf_hi = MAX_SPLIT if self._with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        if self._with_zeeman_splitting:
            return 2.0 * MAX_ZEEMAN_SPLIT + 2.0 * hf_hi + 4.0 * fwhm_total_hi
        return 2.0 * hf_hi + 4.0 * fwhm_total_hi

    def expected_dip_count(self) -> int:
        """Zeeman splitting produces two resolvable groups; otherwise one hyperfine triplet."""
        return 2 if self._with_zeeman_splitting else 1

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

    def _zeeman_array(self, n: int, samples=None) -> np.ndarray:
        """Return the zeeman_split array (zeros when Zeeman splitting is disabled)."""
        if self._with_zeeman_splitting and samples is not None:
            return np.asarray(samples.zeeman_split, dtype=FLOAT_DTYPE)
        return np.zeros(n, dtype=FLOAT_DTYPE)

    def compute(self, x: float, params) -> float:
        hf_split = params.split if self._with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        k_np = params.k_np if self._with_hyperfine_splitting else 1.0
        zeeman_split = params.zeeman_split if self._with_zeeman_splitting else 0.0
        fwhm_total, lorentz_frac, c_total = _saturation_voigt_reparam_scalar(
            params.saturation, params.sigma_inhom, NV_SATURATION_C_MAX
        )
        return nv_center_zeeman_pseudo_voigt_eval(
            float(x),
            params.frequency,
            fwhm_total,
            lorentz_frac,
            zeeman_split,
            hf_split,
            k_np,
            c_total,
            1.0,
        )

    def compute_vectorized_samples(self, x: float, samples) -> np.ndarray:
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples)
        zeeman_arr = self._zeeman_array(n, samples)
        fwhm_total, lorentz_frac, c_total = _saturation_voigt_reparam(
            samples.saturation, samples.sigma_inhom, np.full(n, NV_SATURATION_C_MAX, dtype=FLOAT_DTYPE)
        )
        out = np.empty(n, dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_one_serial(
            float(x),
            freq,
            fwhm_total,
            lorentz_frac,
            zeeman_arr,
            hf_arr,
            k_arr,
            c_total,
            get_background_ones(n),
            out,
        )
        return out

    def compute_vectorized_many(self, x_array: Sequence[float], samples) -> np.ndarray:
        if isinstance(samples, list | tuple):
            samples = self.spec.unpack_samples(samples)  # type: ignore[arg-type]
        elif not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples)
        zeeman_arr = self._zeeman_array(n, samples)
        fwhm_total, lorentz_frac, c_total = _saturation_voigt_reparam(
            samples.saturation, samples.sigma_inhom, np.full(n, NV_SATURATION_C_MAX, dtype=FLOAT_DTYPE)
        )
        out = np.empty((xs.shape[0], n), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many(
            xs,
            freq,
            fwhm_total,
            lorentz_frac,
            zeeman_arr,
            hf_arr,
            k_arr,
            c_total,
            get_background_ones(n),
            out,
        )
        return out

    def compute_vectorized_many_fast(self, x_array: Sequence[float], samples) -> np.ndarray:
        if isinstance(samples, list | tuple):
            samples = self.spec.unpack_samples(samples)  # type: ignore[arg-type]
        elif not hasattr(samples, "frequency"):
            return super().compute_vectorized_many_fast(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        n = freq.shape[0]
        hf_arr, k_arr = self._hf_arrays(n, samples)
        zeeman_arr = self._zeeman_array(n, samples)
        fwhm_total, lorentz_frac, c_total = _saturation_voigt_reparam(
            samples.saturation, samples.sigma_inhom, np.full(n, NV_SATURATION_C_MAX, dtype=FLOAT_DTYPE)
        )
        out = np.empty((xs.shape[0], n), dtype=FLOAT_DTYPE)
        nv_center_zeeman_pseudo_voigt_vectorized_many_fast(
            xs,
            freq,
            fwhm_total,
            lorentz_frac,
            zeeman_arr,
            hf_arr,
            k_arr,
            c_total,
            get_background_ones(n),
            out,
        )
        return out


def nv_center_saturation_voigt_bounds_for_domain(
    x_min: float,
    x_max: float,
    with_hyperfine_splitting: bool = False,
    with_zeeman_splitting: bool = False,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for the saturation-coupled Voigt NV signal.

    ``saturation`` spans a wide dynamic range (near-zero to strongly saturated
    drive) so its bounds are asymmetric-but-linear; ``sigma_inhom`` spans zero
    (pure Lorentzian) up to a generous inhomogeneous width. ``c_max`` (the
    saturated-contrast scale) is not a free parameter here — see
    :data:`NV_SATURATION_C_MAX`.
    """
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

    saturation_bounds = (0.02, 30.0)
    sigma_inhom_hi = max(1.2e6, width * 0.02)
    sigma_inhom_bounds = (0.0, sigma_inhom_hi)

    if with_zeeman_splitting:
        zeeman_margin = MAX_ZEEMAN_SPLIT
        f_lo = float(x_min) + zeeman_margin
        f_hi = float(x_max) - zeeman_margin
        zeeman_bounds = (MIN_ZEEMAN_SPLIT, MAX_ZEEMAN_SPLIT)
        hf_hi = MAX_SPLIT if with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        fwhm_total_hi = 2.0 * NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + saturation_bounds[1]) + 2.0 * (
            2.0 * _SATURATION_VOIGT_SQRT2LOG2 * sigma_inhom_hi
        )
        max_span = 2.0 * MAX_ZEEMAN_SPLIT + 2.0 * hf_hi + 4.0 * fwhm_total_hi

        if with_hyperfine_splitting:
            split_bounds = (MIN_SPLIT, max(MAX_SPLIT, width * 0.02))
            return {
                "frequency": (f_lo, f_hi),
                "saturation": saturation_bounds,
                "sigma_inhom": sigma_inhom_bounds,
                "zeeman_split": zeeman_bounds,
                "split": split_bounds,
                "k_np": (MIN_K_NP, MAX_K_NP),
                "_signal_max_span": (0.0, max_span),
            }
        return {
            "frequency": (f_lo, f_hi),
            "saturation": saturation_bounds,
            "sigma_inhom": sigma_inhom_bounds,
            "zeeman_split": zeeman_bounds,
            "_signal_max_span": (0.0, max_span),
        }

    fwhm_total_hi = 2.0 * NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + saturation_bounds[1]) + 2.0 * (
        2.0 * _SATURATION_VOIGT_SQRT2LOG2 * sigma_inhom_hi
    )
    if with_hyperfine_splitting:
        split_bounds = (MIN_SPLIT, max(MAX_SPLIT, width * 0.02))
        return {
            "frequency": (float(x_min), float(x_max)),
            "saturation": saturation_bounds,
            "sigma_inhom": sigma_inhom_bounds,
            "split": split_bounds,
            "k_np": (MIN_K_NP, MAX_K_NP),
            "_signal_max_span": (0.0, 2.0 * MAX_SPLIT + 4.0 * fwhm_total_hi),
        }
    return {
        "frequency": (float(x_min), float(x_max)),
        "saturation": saturation_bounds,
        "sigma_inhom": sigma_inhom_bounds,
        "_signal_max_span": (0.0, 2.0 * NV_N14_HYPERFINE_SPLIT_HZ + 4.0 * fwhm_total_hi),
    }


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
    with_hyperfine_splitting: bool = False,
    with_zeeman_splitting: bool = False,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for NV Voigt signals over ``[x_min, x_max]``.

    Mirrors ``nv_center_lorentzian_bounds_for_domain``'s structure exactly:
    ``with_zeeman_splitting=True`` adds ``zeeman_split`` and narrows the frequency range so
    the two Zeeman groups always land within the domain; ``with_hyperfine_splitting=False``
    (default) fixes ``split``/``k_np`` to N-14 constants and omits them from the returned dict.
    ``homogeneous_linewidth`` reuses the same bounds as Lorentzian's ``linewidth`` (same
    physical quantity); ``sigma_inhom`` reuses saturation-Voigt's inhomogeneous-width bound.
    """
    width = float(x_max - x_min)
    if width <= 0:
        raise ValueError("x_max must exceed x_min")

    linewidth_bounds = (MIN_LINEWIDTH, max(MAX_LINEWIDTH, width * 0.05))
    sigma_inhom_hi = max(1.2e6, width * 0.02)
    sigma_inhom_bounds = (0.0, sigma_inhom_hi)

    if with_zeeman_splitting:
        # Bounds must cover everything NVCenterCoreGenerator's voigt Zeeman branch can draw:
        # split ~ U(MIN_SPLIT, MAX_SPLIT). These previously capped homogeneous width at 5.0/2.8
        # MHz — below the generated range — which made the true signal unrepresentable for most
        # repeats: curve_fit and the SMC belief pinned split/width at the ceiling and compensated
        # by shifting frequency, a systematic ~3 MHz bias on every voigt fit.
        zeeman_margin = MAX_ZEEMAN_SPLIT
        f_lo = float(x_min) + zeeman_margin
        f_hi = float(x_max) - zeeman_margin
        zeeman_bounds = (MIN_ZEEMAN_SPLIT, MAX_ZEEMAN_SPLIT)
        hf_hi = MAX_SPLIT if with_hyperfine_splitting else NV_N14_HYPERFINE_SPLIT_HZ
        max_span = 2.0 * MAX_ZEEMAN_SPLIT + 2.0 * hf_hi + 4.0 * VOIGT_FWHM_TOTAL_HI

        if with_hyperfine_splitting:
            split_bounds = (MIN_SPLIT, max(MAX_SPLIT, width * 0.02))
            return {
                "frequency": (f_lo, f_hi),
                "homogeneous_linewidth": linewidth_bounds,
                "sigma_inhom": sigma_inhom_bounds,
                "zeeman_split": zeeman_bounds,
                "split": split_bounds,
                "k_np": (MIN_K_NP, MAX_K_NP),
                "c_total": (0.1, 0.4),
                "_signal_max_span": (0.0, max_span),
            }
        return {
            "frequency": (f_lo, f_hi),
            "homogeneous_linewidth": linewidth_bounds,
            "sigma_inhom": sigma_inhom_bounds,
            "zeeman_split": zeeman_bounds,
            "c_total": (0.1, 0.4),
            "_signal_max_span": (0.0, max_span),
        }

    # Non-Zeeman cases (existing behaviour preserved).
    if with_hyperfine_splitting:
        split_bounds = (MIN_SPLIT, max(MAX_SPLIT, width * 0.02))
        return {
            "frequency": (float(x_min), float(x_max)),
            "homogeneous_linewidth": linewidth_bounds,
            "sigma_inhom": sigma_inhom_bounds,
            "split": split_bounds,
            "k_np": (MIN_K_NP, MAX_K_NP),
            "c_total": (0.1, 0.4),
            "_signal_max_span": (0.0, width * 0.1),
        }

    return {
        "frequency": (float(x_min), float(x_max)),
        "homogeneous_linewidth": linewidth_bounds,
        "sigma_inhom": sigma_inhom_bounds,
        "c_total": (0.1, 0.4),
        "_signal_max_span": (0.0, 2.0 * NV_N14_HYPERFINE_SPLIT_HZ + 4.0 * VOIGT_FWHM_TOTAL_HI),
    }
