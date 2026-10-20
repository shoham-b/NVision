"""Drive-controlled NV-center ODMR models: the microwave drive is a per-measurement setting.

Two experiment modes, each a :class:`~nvision.spectra.signal.SignalModel` with ``controls_drive``:

* :class:`NVCenterCWDriveModel` -- **CW ODMR**. The drive amplitude ``a`` (``drive_amp_unit``, a
  fraction of the source's maximum field) sets the saturation parameter
  ``s = (a / drive_amp_sat)**2`` (power ∝ field²), and the existing saturation-Voigt physics follows:
  homogeneous HWHM ``Γ0·sqrt(1+s)`` (power broadening) and contrast ``c_max·s/(1+s)``. High drive
  gives wide, deep dips -- easy to *find* -- low drive gives narrow ones -- better *precision*.
  ``drive_amp_sat`` (the amplitude at which ``s = 1``) is inferred.
* :class:`NVCenterPulsedDriveModel` -- **pulsed ODMR** with a single π-pulse. The drive amplitude sets
  the Rabi frequency ``Ω = rabi_per_amp · a``; the pulse duration ``t`` is the locator's calibration
  (``t = 1/(2·Ω_est)``) carried in the :class:`~nvision.spectra.drive.DriveSetting`. Each line's
  spin-flip probability is the Rabi formula
  ``P(Δ) = Ω²/(Ω²+Δ²) · sin²(π·t·sqrt(Ω²+Δ²))`` (Ω, Δ in Hz), averaged over a Gaussian
  inhomogeneous detuning spread ``sigma_inhom``; signal ``1 - c_total · Σ_lines w_line·P``.
  ``rabi_per_amp`` (Rabi frequency at full amplitude) and ``c_total`` are inferred. Note the
  on-resonance depth ``sin²(π/2 · Ω/Ω_est)`` is periodic in the calibration ratio, so the
  ``rabi_per_amp`` posterior can be multimodal (aliased); the off-resonance lineshape separates the
  branches (see docs/amplitude_control.md).

Neither model is evaluated directly: :meth:`at_drive` returns a fixed-drive view, an ordinary
frequency-only model over the *same* parameters, which is what fixed-drive locators, plots and the
per-drive-group likelihood use. Hyperfine structure is the fixed isotope geometry
(``hyperfine``); inferring ``split``/``k_np`` is not supported for these models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT, DriveSetting
from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.nv_center import (
    _SATURATION_VOIGT_SQRT2LOG2,
    MAX_ZEEMAN_SPLIT,
    MIN_ZEEMAN_SPLIT,
    NV_NATURAL_HWHM_HZ,
    NV_ZERO_FIELD_SPLITTING_HZ,
    NVCenterSaturationVoigtModel,
    _check_hyperfine,
)
from nvision.spectra.signal import SignalModel
from nvision.spectra.spec import GenericParamSpec

# ---------------------------------------------------------------------------
# Physical ranges
# ---------------------------------------------------------------------------

# CW: drive amplitude (fraction of max field) at which s = 1. Chosen so the saturation at full
# amplitude, s(1) = 1/drive_amp_sat**2, spans 4..400 -- always past the CW slope optimum s ~ 2, so
# the amplitude is a genuine design trade-off rather than "always drive at maximum".
DRIVE_AMP_SAT_BOUNDS: tuple[float, float] = (0.05, 0.5)
# Pulsed: Rabi frequency at full amplitude (Hz). The π-pulse line HWHM is ~0.8 * Omega, so the
# drive spans ~40 kHz (minimum amplitude, weakest coupling) to ~8 MHz (full amplitude, strongest).
RABI_PER_AMP_BOUNDS: tuple[float, float] = (1.0e6, 10.0e6)
PULSED_C_TOTAL_BOUNDS: tuple[float, float] = (0.05, 0.4)
CW_SIGMA_INHOM_HI: float = 1.2e6
PULSED_SIGMA_INHOM_HI: float = 0.6e6

# Baseline drive amplitude used by locators that do not choose the drive (fixed-drive baselines).
BASELINE_DRIVE_AMP_UNIT: float = 0.3

# HWHM of the π-pulse Rabi line in units of Omega: P(Δ) = 1/2 at Δ ≈ 0.7988·Ω.
PI_PULSE_HWHM_PER_RABI: float = 0.7988

# Gauss-Hermite nodes/weights (physicists') for the inhomogeneous detuning average.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(7)
_GH_SHIFT_PER_SIGMA = (math.sqrt(2.0) * _GH_NODES).astype(np.float64)
_GH_PROB = (_GH_WEIGHTS / math.sqrt(math.pi)).astype(np.float64)


def pi_pulse_duration_s(rabi_hz: float) -> float:
    """Duration (s) of a π-pulse at Rabi frequency ``rabi_hz`` (Hz): ``1 / (2·Ω)``."""
    if not rabi_hz > 0.0:
        raise ValueError(f"pi_pulse_duration_s needs a positive Rabi frequency; got {rabi_hz!r}")
    return 1.0 / (2.0 * rabi_hz)


def cw_saturation(drive_amp_unit, drive_amp_sat):
    """Saturation parameter ``s = (a / drive_amp_sat)**2``; scalars or arrays (power ∝ field²)."""
    return (np.asarray(drive_amp_unit) / np.asarray(drive_amp_sat)) ** 2


def drive_effective_hwhm(params, drive_amp_unit: float):
    """Effective dip HWHM (Hz) of a drive-controlled model's parameters at drive amplitude ``drive_amp_unit``.

    ``params`` maps names to physical values (scalars or per-particle arrays). CW:
    ``Γ0·sqrt(1+s) + sqrt(2 ln2)·sigma_inhom``; pulsed: ``0.8·Ω + sqrt(2 ln2)·sigma_inhom``.
    """
    sigma_hwhm = _SATURATION_VOIGT_SQRT2LOG2 * np.asarray(params["sigma_inhom"])
    if "drive_amp_sat" in params:
        s = cw_saturation(drive_amp_unit, params["drive_amp_sat"])
        return NV_NATURAL_HWHM_HZ * np.sqrt(1.0 + s) + sigma_hwhm
    if "rabi_per_amp" in params:
        return PI_PULSE_HWHM_PER_RABI * np.asarray(params["rabi_per_amp"]) * drive_amp_unit + sigma_hwhm
    raise ValueError(f"drive_effective_hwhm: no drive parameter ('drive_amp_sat'/'rabi_per_amp') in {sorted(params)}")


def is_drive_param_dict(params) -> bool:
    """True when a parameter mapping belongs to a drive-controlled model."""
    return "drive_amp_sat" in params or "rabi_per_amp" in params


# ---------------------------------------------------------------------------
# Parameter bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVCenterCWDriveSpectrum:
    center_freq: float
    drive_amp_sat: float
    sigma_inhom: float


@dataclass(frozen=True)
class NVCenterCWDriveSpectrumSamples:
    center_freq: np.ndarray
    drive_amp_sat: np.ndarray
    sigma_inhom: np.ndarray


@dataclass(frozen=True)
class NVCenterCWDriveZeemanSpectrum:
    center_freq: float
    drive_amp_sat: float
    sigma_inhom: float
    zeeman_split: float


@dataclass(frozen=True)
class NVCenterCWDriveZeemanSpectrumSamples:
    center_freq: np.ndarray
    drive_amp_sat: np.ndarray
    sigma_inhom: np.ndarray
    zeeman_split: np.ndarray


@dataclass(frozen=True)
class NVCenterPulsedSpectrum:
    center_freq: float
    rabi_per_amp: float
    c_total: float
    sigma_inhom: float


@dataclass(frozen=True)
class NVCenterPulsedSpectrumSamples:
    center_freq: np.ndarray
    rabi_per_amp: np.ndarray
    c_total: np.ndarray
    sigma_inhom: np.ndarray


@dataclass(frozen=True)
class NVCenterPulsedZeemanSpectrum:
    center_freq: float
    rabi_per_amp: float
    c_total: float
    sigma_inhom: float
    zeeman_split: float


@dataclass(frozen=True)
class NVCenterPulsedZeemanSpectrumSamples:
    center_freq: np.ndarray
    rabi_per_amp: np.ndarray
    c_total: np.ndarray
    sigma_inhom: np.ndarray
    zeeman_split: np.ndarray


def _spec(params_cls, samples_cls, fixed_center: bool):
    """A GenericParamSpec over ``params_cls`` (uncertainty bundle = params bundle shape)."""
    spec_cls = type(
        f"_{params_cls.__name__}Spec",
        (GenericParamSpec,),
        {"params_cls": params_cls, "samples_cls": samples_cls, "uncertainty_cls": params_cls},
    )
    return spec_cls(fixed_values={"center_freq": NV_ZERO_FIELD_SPLITTING_HZ} if fixed_center else None)


_CW_SPECS = {
    (zeeman, fixed): _spec(
        NVCenterCWDriveZeemanSpectrum if zeeman else NVCenterCWDriveSpectrum,
        NVCenterCWDriveZeemanSpectrumSamples if zeeman else NVCenterCWDriveSpectrumSamples,
        fixed,
    )
    for zeeman in (False, True)
    for fixed in (False, True)
}
_PULSED_SPECS = {
    (zeeman, fixed): _spec(
        NVCenterPulsedZeemanSpectrum if zeeman else NVCenterPulsedSpectrum,
        NVCenterPulsedZeemanSpectrumSamples if zeeman else NVCenterPulsedSpectrumSamples,
        fixed,
    )
    for zeeman in (False, True)
    for fixed in (False, True)
}


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _NVCenterDriveModelBase(SignalModel):
    """Common configuration, spec selection and the "needs a drive" guard."""

    _SPECS: ClassVar[dict[tuple[bool, bool], GenericParamSpec]] = {}
    _DRIVE_PARAM: str = ""

    def __init__(
        self,
        hyperfine: str = "unresolved",
        with_zeeman_splitting: bool = True,
        with_fixed_center_freq: bool = True,
    ) -> None:
        self._hf_offset, self._w_center, self._hf_lines = _check_hyperfine(hyperfine, False)
        self._hyperfine = hyperfine
        self._with_zeeman_splitting = with_zeeman_splitting
        self._with_fixed_center_freq = with_fixed_center_freq
        self._views: dict[DriveSetting, SignalModel] = {}

    @property
    def hyperfine(self) -> str:
        return self._hyperfine

    @property
    def with_zeeman_splitting(self) -> bool:
        return self._with_zeeman_splitting

    @property
    def spec(self):
        return self._SPECS[(self._with_zeeman_splitting, self._with_fixed_center_freq)]

    @property
    def controls_drive(self) -> bool:
        return True

    def at_drive(self, drive: DriveSetting) -> SignalModel:
        self._check_drive(drive)
        view = self._views.get(drive)
        if view is None:
            view = self._make_view(drive)
            self._views[drive] = view
        return view

    def _check_drive(self, drive: DriveSetting) -> None:
        if not isinstance(drive, DriveSetting):
            raise TypeError(f"at_drive expects a DriveSetting; got {type(drive).__name__}")
        if self.is_pulsed and drive.pulse_duration_s is None:
            raise ValueError(f"{type(self).__name__} is pulsed: its drive needs a pulse_duration_s.")
        if not self.is_pulsed and drive.pulse_duration_s is not None:
            raise ValueError(f"{type(self).__name__} is CW: its drive must not carry a pulse_duration_s.")

    def _make_view(self, drive: DriveSetting) -> SignalModel:
        raise NotImplementedError

    def _needs_drive(self, *_args, **_kwargs):
        raise ValueError(
            f"{type(self).__name__} is drive-controlled: evaluate it through at_drive(DriveSetting) "
            "or compute_vectorized_many_driven, never directly."
        )

    compute = _needs_drive
    compute_vectorized_samples = _needs_drive
    compute_vectorized_many = _needs_drive
    compute_vectorized_many_fast = _needs_drive
    compute_many_float64 = _needs_drive

    def is_scale_parameter(self, name: str) -> bool:
        return name in (self._DRIVE_PARAM, "sigma_inhom")

    def parameter_weights(self) -> dict[str, float]:
        weights = {name: 1.0 for name in self.parameter_names()}
        if "zeeman_split" in weights:
            weights["zeeman_split"] = 1.5
        if "center_freq" in weights:
            weights["center_freq"] = 2.0
        return weights

    def expected_dip_count(self) -> int:
        return self._hf_lines * (2 if self._with_zeeman_splitting else 1)

    def signal_min_span(self, domain_width: float) -> float | None:
        return 4.0 * NV_NATURAL_HWHM_HZ

    def signal_max_span(self, domain_width: float) -> float | None:
        fwhm_total_hi = 2.0 * self._max_hwhm()
        zeeman = 2.0 * MAX_ZEEMAN_SPLIT if self._with_zeeman_splitting else 0.0
        return zeeman + 2.0 * self._hf_offset + 4.0 * fwhm_total_hi

    def _max_hwhm(self) -> float:
        raise NotImplementedError


class _FixedDriveViewBase(SignalModel):
    """A drive-controlled model frozen at one :class:`DriveSetting` -- an ordinary frequency-only model."""

    def __init__(self, parent: _NVCenterDriveModelBase, drive: DriveSetting) -> None:
        self._parent = parent
        self._drive = drive

    @property
    def spec(self):
        return self._parent.spec

    @property
    def fixed_drive(self) -> DriveSetting:
        return self._drive

    @property
    def is_pulsed(self) -> bool:
        return self._parent.is_pulsed

    @property
    def drive_model(self) -> _NVCenterDriveModelBase:
        """The drive-controlled model this view was taken from."""
        return self._parent

    def is_scale_parameter(self, name: str) -> bool:
        return self._parent.is_scale_parameter(name)

    def parameter_weights(self) -> dict[str, float]:
        return self._parent.parameter_weights()

    def expected_dip_count(self) -> int:
        return self._parent.expected_dip_count()

    def signal_min_span(self, domain_width: float) -> float | None:
        return self._parent.signal_min_span(domain_width)

    def signal_max_span(self, domain_width: float) -> float | None:
        return self._parent.signal_max_span(domain_width)

    def _typed_samples(self, samples):
        if isinstance(samples, list | tuple):
            return self.spec.unpack_samples(samples)
        try:
            return self.spec.unpack_samples(samples.arrays_in_order())
        except AttributeError:
            return samples


# ---------------------------------------------------------------------------
# CW: saturation-coupled Voigt driven at amplitude a
# ---------------------------------------------------------------------------


class NVCenterCWDriveModel(_NVCenterDriveModelBase):
    """CW ODMR with a controllable drive amplitude (see module docstring).

    Parameters (``with_zeeman_splitting=True``): ``center_freq`` (fixed to D by default),
    ``drive_amp_sat`` (amplitude at s = 1, fraction of max field), ``sigma_inhom`` (Hz),
    ``zeeman_split`` (Hz).
    """

    _SPECS = _CW_SPECS
    _DRIVE_PARAM = "drive_amp_sat"

    def __init__(
        self,
        hyperfine: str = "unresolved",
        with_zeeman_splitting: bool = True,
        with_fixed_center_freq: bool = True,
    ) -> None:
        super().__init__(hyperfine, with_zeeman_splitting, with_fixed_center_freq)
        self._saturation_model = NVCenterSaturationVoigtModel(
            hyperfine=hyperfine,
            infer_hyperfine=False,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_center_freq=with_fixed_center_freq,
        )

    def _make_view(self, drive: DriveSetting) -> SignalModel:
        return _CWFixedDriveView(self, drive)

    def _max_hwhm(self) -> float:
        s_hi = (1.0 / DRIVE_AMP_SAT_BOUNDS[0]) ** 2
        return NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + s_hi) + _SATURATION_VOIGT_SQRT2LOG2 * CW_SIGMA_INHOM_HI


class _CWFixedDriveView(_FixedDriveViewBase):
    """CW model at one amplitude: maps ``drive_amp_sat`` to ``saturation`` and reuses the saturation-Voigt kernels."""

    def _as_saturation(self, params):
        zeeman = getattr(params, "zeeman_split", 0.0)
        return SimpleNamespace(
            center_freq=params.center_freq,
            saturation=float(cw_saturation(self._drive.drive_amp_unit, params.drive_amp_sat)),
            sigma_inhom=params.sigma_inhom,
            zeeman_split=zeeman,
        )

    def _as_saturation_samples(self, samples):
        typed = self._typed_samples(samples)
        out = SimpleNamespace(
            center_freq=np.asarray(typed.center_freq, dtype=FLOAT_DTYPE),
            saturation=cw_saturation(
                self._drive.drive_amp_unit, np.asarray(typed.drive_amp_sat, dtype=FLOAT_DTYPE)
            ).astype(FLOAT_DTYPE, copy=False),
            sigma_inhom=np.asarray(typed.sigma_inhom, dtype=FLOAT_DTYPE),
        )
        if hasattr(typed, "zeeman_split"):
            out.zeeman_split = np.asarray(typed.zeeman_split, dtype=FLOAT_DTYPE)
        return out

    @property
    def _sv(self) -> NVCenterSaturationVoigtModel:
        return self._parent._saturation_model

    def compute(self, x: float, params) -> float:
        return self._sv.compute(x, self._as_saturation(params))

    def compute_many_float64(self, xs: np.ndarray, params) -> np.ndarray:
        return self._sv.compute_many_float64(xs, self._as_saturation(params))

    def compute_vectorized_samples(self, x: float, samples) -> np.ndarray:
        return self._sv.compute_vectorized_samples(x, self._as_saturation_samples(samples))

    def compute_vectorized_many(self, x_array: Sequence[float], samples) -> np.ndarray:
        return self._sv.compute_vectorized_many(x_array, self._as_saturation_samples(samples))

    def compute_vectorized_many_fast(self, x_array: Sequence[float], samples) -> np.ndarray:
        return self._sv.compute_vectorized_many_fast(x_array, self._as_saturation_samples(samples))


# ---------------------------------------------------------------------------
# Pulsed: single π-pulse Rabi line
# ---------------------------------------------------------------------------


class NVCenterPulsedDriveModel(_NVCenterDriveModelBase):
    """Pulsed ODMR with a controllable drive amplitude and a calibrated π-pulse (see module docstring).

    Parameters (``with_zeeman_splitting=True``): ``center_freq`` (fixed to D by default),
    ``rabi_per_amp`` (Rabi frequency at full amplitude, Hz), ``c_total`` (readout contrast of a full
    spin flip), ``sigma_inhom`` (Hz), ``zeeman_split`` (Hz).
    """

    _SPECS = _PULSED_SPECS
    _DRIVE_PARAM = "rabi_per_amp"

    @property
    def is_pulsed(self) -> bool:
        return True

    def _make_view(self, drive: DriveSetting) -> SignalModel:
        return _PulsedFixedDriveView(self, drive)

    def _max_hwhm(self) -> float:
        return PI_PULSE_HWHM_PER_RABI * RABI_PER_AMP_BOUNDS[1] + _SATURATION_VOIGT_SQRT2LOG2 * PULSED_SIGMA_INHOM_HI

    def line_offsets_and_weights(self) -> list[tuple[float, float]]:
        """Hyperfine lines of one Zeeman group: ``(offset_hz, weight)``, weights summing to 1."""
        hf, w = self._hf_offset, self._w_center
        if hf == 0.0:
            return [(0.0, 1.0)]
        norm = 2.0 + w
        lines = [(-hf, 1.0 / norm), (hf, 1.0 / norm)]
        if w > 0.0:
            lines.append((0.0, w / norm))
        return lines


def rabi_transition_probability(detuning_hz, rabi_hz, pulse_duration_s: float):
    """Spin-flip probability of a square pulse: ``Ω²/(Ω²+Δ²) · sin²(π·t·sqrt(Ω²+Δ²))`` (Hz units).

    ``detuning_hz`` and ``rabi_hz`` broadcast against each other (e.g. shape (n_x, 1) and (1, n)).
    """
    om2 = np.square(rabi_hz)
    eff2 = om2 + np.square(detuning_hz)
    return om2 / eff2 * np.square(np.sin(np.pi * pulse_duration_s * np.sqrt(eff2)))


class _PulsedFixedDriveView(_FixedDriveViewBase):
    """Pulsed model at one (amplitude, pulse duration): Rabi lines averaged over the inhomogeneous spread."""

    def _predict(self, xs: np.ndarray, center, rabi_per_amp, c_total, sigma_inhom, zeeman) -> np.ndarray:
        """Signal at drive frequencies ``xs`` (shape (n_x,)) for parameter arrays (shape (n,)); returns (n_x, n)."""
        drive = self._drive
        rabi = (np.asarray(rabi_per_amp, dtype=np.float64) * drive.drive_amp_unit)[None, :]
        sigma = np.asarray(sigma_inhom, dtype=np.float64)[None, :]
        center = np.asarray(center, dtype=np.float64)[None, :]
        zeeman = np.asarray(zeeman, dtype=np.float64)[None, :]
        x = np.asarray(xs, dtype=np.float64)[:, None]
        t = float(drive.pulse_duration_s)
        depth = np.zeros((x.shape[0], rabi.shape[1]), dtype=np.float64)
        groups = (-1.0, 1.0) if self._parent.with_zeeman_splitting else (0.0,)
        group_weight = 1.0 / len(groups)
        lines = self._parent.line_offsets_and_weights()
        for sign in groups:
            # Mirror the hyperfine order in the lower group (matches the CW kernels' convention).
            for offset, w_line in lines:
                line_center = center + sign * zeeman + (offset if sign >= 0.0 else -offset)
                detuning = x - line_center
                for shift, p in zip(_GH_SHIFT_PER_SIGMA, _GH_PROB, strict=True):
                    depth += (group_weight * w_line * p) * rabi_transition_probability(
                        detuning + shift * sigma, rabi, t
                    )
        return 1.0 - np.asarray(c_total, dtype=np.float64)[None, :] * depth

    def _arrays(self, typed):
        zeeman = getattr(typed, "zeeman_split", np.zeros_like(np.asarray(typed.center_freq)))
        return typed.center_freq, typed.rabi_per_amp, typed.c_total, typed.sigma_inhom, zeeman

    def compute(self, x: float, params) -> float:
        zeeman = getattr(params, "zeeman_split", 0.0)
        return float(
            self._predict(
                np.array([x]),
                np.array([params.center_freq]),
                np.array([params.rabi_per_amp]),
                np.array([params.c_total]),
                np.array([params.sigma_inhom]),
                np.array([zeeman]),
            )[0, 0]
        )

    def compute_many_float64(self, xs: np.ndarray, params) -> np.ndarray:
        zeeman = getattr(params, "zeeman_split", 0.0)
        return self._predict(
            np.asarray(xs, dtype=np.float64),
            np.array([params.center_freq]),
            np.array([params.rabi_per_amp]),
            np.array([params.c_total]),
            np.array([params.sigma_inhom]),
            np.array([zeeman]),
        )[:, 0]

    def compute_vectorized_samples(self, x: float, samples) -> np.ndarray:
        typed = self._typed_samples(samples)
        return self._predict(np.array([x]), *self._arrays(typed))[0].astype(FLOAT_DTYPE)

    def compute_vectorized_many(self, x_array: Sequence[float], samples) -> np.ndarray:
        xs = np.asarray(x_array, dtype=np.float64)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        typed = self._typed_samples(samples)
        return self._predict(xs, *self._arrays(typed)).astype(FLOAT_DTYPE)

    def compute_vectorized_many_fast(self, x_array: Sequence[float], samples) -> np.ndarray:
        return self.compute_vectorized_many(x_array, samples)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def nv_center_drive_bounds_for_domain(
    mode: str,
    drive_freq_min_phys: float,
    drive_freq_max_phys: float,
    hyperfine: str = "unresolved",
    with_zeeman_splitting: bool = True,
) -> dict[str, tuple[float, float]]:
    """Physical parameter bounds for a drive-controlled NV model (``mode`` is ``"cw"`` or ``"pulsed"``)."""
    if not drive_freq_max_phys > drive_freq_min_phys:
        raise ValueError("drive_freq_max_phys must exceed drive_freq_min_phys")
    if mode == "cw":
        model: _NVCenterDriveModelBase = NVCenterCWDriveModel(hyperfine, with_zeeman_splitting)
        bounds: dict[str, tuple[float, float]] = {
            "center_freq": (float(drive_freq_min_phys), float(drive_freq_max_phys)),
            "drive_amp_sat": DRIVE_AMP_SAT_BOUNDS,
            "sigma_inhom": (0.0, CW_SIGMA_INHOM_HI),
        }
    elif mode == "pulsed":
        model = NVCenterPulsedDriveModel(hyperfine, with_zeeman_splitting)
        bounds = {
            "center_freq": (float(drive_freq_min_phys), float(drive_freq_max_phys)),
            "rabi_per_amp": RABI_PER_AMP_BOUNDS,
            "c_total": PULSED_C_TOTAL_BOUNDS,
            "sigma_inhom": (0.0, PULSED_SIGMA_INHOM_HI),
        }
    else:
        raise ValueError(f"unknown drive mode {mode!r}; expected 'cw' or 'pulsed'")
    if with_zeeman_splitting:
        bounds["zeeman_split"] = (MIN_ZEEMAN_SPLIT, MAX_ZEEMAN_SPLIT)
    bounds["_signal_max_span"] = (0.0, float(model.signal_max_span(drive_freq_max_phys - drive_freq_min_phys)))
    return bounds


__all__ = [
    "BASELINE_DRIVE_AMP_UNIT",
    "DRIVE_AMP_MIN_UNIT",
    "DRIVE_AMP_SAT_BOUNDS",
    "RABI_PER_AMP_BOUNDS",
    "NVCenterCWDriveModel",
    "NVCenterPulsedDriveModel",
    "cw_saturation",
    "drive_effective_hwhm",
    "is_drive_param_dict",
    "nv_center_drive_bounds_for_domain",
    "pi_pulse_duration_s",
    "rabi_transition_probability",
]
