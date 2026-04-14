"""Single Voigt peak model (convolution of Lorentzian and Gaussian)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import ParamSpec, SignalModel


@dataclass(frozen=True)
class OnePeakVoigtSpectrum:
    frequency: float
    fwhm_lorentz: float
    fwhm_gauss: float
    dip_depth: float
    background: float


@dataclass(frozen=True)
class OnePeakVoigtSpectrumSamples:
    frequency: np.ndarray
    fwhm_lorentz: np.ndarray
    fwhm_gauss: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class OnePeakVoigtSpectrumUncertainty:
    frequency: float
    fwhm_lorentz: float
    fwhm_gauss: float
    dip_depth: float
    background: float


class _OnePeakVoigtSpec(ParamSpec[OnePeakVoigtSpectrum, OnePeakVoigtSpectrumSamples, OnePeakVoigtSpectrumUncertainty]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("frequency", "fwhm_lorentz", "fwhm_gauss", "dip_depth", "background")

    @property
    def dim(self) -> int:
        return 5

    def unpack_params(self, values) -> OnePeakVoigtSpectrum:
        f, wl, wg, d, b = values
        return OnePeakVoigtSpectrum(float(f), float(wl), float(wg), float(d), float(b))

    def pack_params(self, params: OnePeakVoigtSpectrum) -> tuple[float, ...]:
        return (
            float(params.frequency),
            float(params.fwhm_lorentz),
            float(params.fwhm_gauss),
            float(params.dip_depth),
            float(params.background),
        )

    def unpack_uncertainty(self, values) -> OnePeakVoigtSpectrumUncertainty:
        f, wl, wg, d, b = values
        return OnePeakVoigtSpectrumUncertainty(float(f), float(wl), float(wg), float(d), float(b))

    def pack_uncertainty(self, u: OnePeakVoigtSpectrumUncertainty) -> tuple[float, ...]:
        return (
            float(u.frequency),
            float(u.fwhm_lorentz),
            float(u.fwhm_gauss),
            float(u.dip_depth),
            float(u.background),
        )

    def unpack_samples(self, arrays_in_order) -> OnePeakVoigtSpectrumSamples:
        f, wl, wg, d, b = arrays_in_order
        return OnePeakVoigtSpectrumSamples(
            frequency=np.asarray(f, dtype=FLOAT_DTYPE),
            fwhm_lorentz=np.asarray(wl, dtype=FLOAT_DTYPE),
            fwhm_gauss=np.asarray(wg, dtype=FLOAT_DTYPE),
            dip_depth=np.asarray(d, dtype=FLOAT_DTYPE),
            background=np.asarray(b, dtype=FLOAT_DTYPE),
        )

    def pack_samples(self, samples: OnePeakVoigtSpectrumSamples) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(samples.frequency, dtype=FLOAT_DTYPE),
            np.asarray(samples.fwhm_lorentz, dtype=FLOAT_DTYPE),
            np.asarray(samples.fwhm_gauss, dtype=FLOAT_DTYPE),
            np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE),
            np.asarray(samples.background, dtype=FLOAT_DTYPE),
        )


class OnePeakVoigtModel(SignalModel[OnePeakVoigtSpectrum, OnePeakVoigtSpectrumSamples, OnePeakVoigtSpectrumUncertainty]):
    """Single Voigt peak model (convolution of Lorentzian and Gaussian).

    Not njit-accelerated: evaluation uses SciPy/JAX ``wofz`` or a pseudo-Voigt fallback.

    The Voigt profile is the convolution of a Lorentzian and a Gaussian,
    accounting for both homogeneous (Lorentzian) and inhomogeneous (Gaussian)
    broadening mechanisms.

    Parameters
    ----------
    frequency : float
        Peak center, in [0, 1] normalized units
    fwhm_lorentz : float
        Lorentzian FWHM (full width at half maximum)
    fwhm_gauss : float
        Gaussian FWHM (inhomogeneous broadening)
    dip_depth : float
        Normalized peak depth (0 to 1). Peak height = dip_depth.
    background : float
        Baseline level (max signal)
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
        """Voigt profile normalized so value at center is 1."""
        center_val = self._voigt_center_height(fwhm_l, fwhm_g)
        if abs(center_val) < 1e-12:
            return 0.0
        return self._voigt_profile(x, center, fwhm_l, fwhm_g) / center_val

    _SPEC = _OnePeakVoigtSpec()

    @property
    def spec(self) -> _OnePeakVoigtSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("fwhm_lorentz", "fwhm_gauss", "dip_depth")

    def compute(self, x: float, params: OnePeakVoigtSpectrum) -> float:
        return float(
            params.background
            - params.dip_depth * self._voigt_profile_unit_peak(x, params.frequency, params.fwhm_lorentz, params.fwhm_gauss)
        )

    def compute_vectorized_samples(self, x: float, samples: OnePeakVoigtSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        fwhm_l = np.asarray(samples.fwhm_lorentz, dtype=FLOAT_DTYPE)
        fwhm_g = np.asarray(samples.fwhm_gauss, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        # Compute center heights for normalization
        sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l / 2
        z_center = (0.0 + 1j * gamma) / (sigma * np.sqrt(2))
        w_center = self.wofz(z_center) if self.wofz is not None else None

        if w_center is not None:
            center_height = w_center.real / (sigma * np.sqrt(2 * np.pi))
        else:
            # Pseudo-Voigt fallback center height
            eta = (
                1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
            )
            gamma = fwhm_l / 2
            lorentz_center = gamma / (gamma**2)
            sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
            gauss_center = 1.0 / (sigma * np.sqrt(2 * np.pi))
            center_height = eta * lorentz_center + (1 - eta) * gauss_center

        # Compute profile values
        if self.wofz is not None:
            z = ((x_f - freq) + 1j * gamma) / (sigma * np.sqrt(2))
            w = self.wofz(z)
            profile = w.real / (sigma * np.sqrt(2 * np.pi))
        else:
            # Pseudo-Voigt fallback
            eta = (
                1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
            )
            gamma = fwhm_l / 2
            lorentz = gamma / ((x_f - freq) ** 2 + gamma**2)
            sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x_f - freq) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            profile = eta * lorentz + (1 - eta) * gauss

        normalized = profile / center_height
        return (bg - depth * normalized).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: OnePeakVoigtSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        fwhm_l = np.asarray(samples.fwhm_lorentz, dtype=FLOAT_DTYPE)
        fwhm_g = np.asarray(samples.fwhm_gauss, dtype=FLOAT_DTYPE)
        depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        x2d = xs[:, None]
        freq2d = freq[None, :]
        fwhm_l2d = fwhm_l[None, :]
        fwhm_g2d = fwhm_g[None, :]

        # Compute center heights
        sigma = fwhm_g2d / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l2d / 2
        z_center = (0.0 + 1j * gamma) / (sigma * np.sqrt(2))

        if self.wofz is not None:
            w_center = self.wofz(z_center)
            center_height = w_center.real / (sigma * np.sqrt(2 * np.pi))
        else:
            # Pseudo-Voigt fallback
            eta = (
                1.36603 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d))
                - 0.47719 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 2
                + 0.11116 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 3
            )
            lorentz_center = gamma / (gamma**2)
            gauss_center = 1.0 / (sigma * np.sqrt(2 * np.pi))
            center_height = eta * lorentz_center + (1 - eta) * gauss_center

        # Compute profiles
        if self.wofz is not None:
            z = ((x2d - freq2d) + 1j * gamma) / (sigma * np.sqrt(2))
            w = self.wofz(z)
            profile = w.real / (sigma * np.sqrt(2 * np.pi))
        else:
            # Pseudo-Voigt fallback
            eta = (
                1.36603 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d))
                - 0.47719 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 2
                + 0.11116 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 3
            )
            lorentz = gamma / ((x2d - freq2d) ** 2 + gamma**2)
            gauss = np.exp(-0.5 * ((x2d - freq2d) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            profile = eta * lorentz + (1 - eta) * gauss

        normalized = profile / center_height
        return (bg[None, :] - depth[None, :] * normalized).astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> OnePeakVoigtSpectrum:
        """Sample parameters that keep the signal within [0, 1]."""
        frequency = rng.uniform(0.1, 0.9)
        fwhm_lorentz = rng.uniform(0.03, 0.08)
        fwhm_gauss = rng.uniform(0.01, 0.05)
        dip_depth = rng.uniform(0.3, 0.85)
        background = 1.0
        return OnePeakVoigtSpectrum(
            frequency=frequency,
            fwhm_lorentz=fwhm_lorentz,
            fwhm_gauss=fwhm_gauss,
            dip_depth=dip_depth,
            background=background,
        )
