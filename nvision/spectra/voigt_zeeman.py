"""Voigt-broadened NV center model with Zeeman splitting."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import GenericParamSpec, SignalModel


@dataclass(frozen=True)
class VoigtZeemanSpectrum:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    split: float
    k_np: float
    dip_depth: float
    background: float


@dataclass(frozen=True)
class VoigtZeemanSpectrumSamples:
    frequency: np.ndarray
    fwhm_total: np.ndarray
    lorentz_frac: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class VoigtZeemanSpectrumUncertainty:
    frequency: float
    fwhm_total: float
    lorentz_frac: float
    split: float
    k_np: float
    dip_depth: float
    background: float


class _VoigtZeemanSpec(
    GenericParamSpec[
        VoigtZeemanSpectrum,
        VoigtZeemanSpectrumSamples,
        VoigtZeemanSpectrumUncertainty,
    ]
):
    params_cls = VoigtZeemanSpectrum
    samples_cls = VoigtZeemanSpectrumSamples
    uncertainty_cls = VoigtZeemanSpectrumUncertainty


class VoigtZeemanModel(SignalModel[VoigtZeemanSpectrum, VoigtZeemanSpectrumSamples, VoigtZeemanSpectrumUncertainty]):
    """Voigt-broadened NV center model with Zeeman splitting.

    Not njit-accelerated: evaluation uses SciPy/JAX ``wofz`` or a pseudo-Voigt fallback.

    Models an NV center with three Voigt profile dips (hyperfine splitting),
    where each Lorentzian dip is convolved with a Gaussian. This accounts for
    both homogeneous (Lorentzian) and inhomogeneous (Gaussian) broadening.

    Parameters
    ----------
    frequency : float
        Central frequency (f_B)
    fwhm_total : float
        Total effective linewidth (Lorentzian + Gaussian)
    lorentz_frac : float
        Lorentzian share of broadening in [0, 1]
    split : float
        Hyperfine splitting (delta_f_HF)
    k_np : float
        Non-polarization factor (amplitude ratio between peaks)
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
        """Voigt profile normalized so value at center is 1."""
        center_val = self._voigt_center_height(fwhm_l, fwhm_g)
        if abs(center_val) < 1e-12:
            return 0.0
        return self._voigt_profile(x, center, fwhm_l, fwhm_g) / center_val

    def compute_voigt_zeeman_model(
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
        """Triple Voigt NV model; parameter order matches :meth:`parameter_names`."""
        fwhm_l = lorentz_frac * fwhm_total
        fwhm_g = (1.0 - lorentz_frac) * fwhm_total
        actual_depth = dip_depth / k_np
        left_dip = (actual_depth / k_np) * self._voigt_profile_unit_peak(x, frequency - split, fwhm_l, fwhm_g)
        center_dip = actual_depth * self._voigt_profile_unit_peak(x, frequency, fwhm_l, fwhm_g)
        right_dip = (actual_depth * k_np) * self._voigt_profile_unit_peak(x, frequency + split, fwhm_l, fwhm_g)
        return background - (left_dip + center_dip + right_dip)

    _SPEC = _VoigtZeemanSpec()

    @property
    def spec(self) -> _VoigtZeemanSpec:
        return self._SPEC

    def is_scale_parameter(self, name: str) -> bool:
        return name in ("fwhm_total", "dip_depth")

    def expected_dip_count(self) -> int:
        """Zeeman splitting produces 3 dips: ms=-1, 0, +1 transitions."""
        return 3

    def compute(self, x: float, params: VoigtZeemanSpectrum) -> float:
        return self.compute_voigt_zeeman_model(
            float(x),
            params.frequency,
            params.fwhm_total,
            params.lorentz_frac,
            params.split,
            params.k_np,
            params.dip_depth,
            params.background,
        )

    def compute_vectorized_samples(self, x: float, samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        x_f = float(x)
        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        fwhm_total = np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE)
        lorentz_frac = np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE)
        fwhm_l = lorentz_frac * fwhm_total
        fwhm_g = (1.0 - lorentz_frac) * fwhm_total
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        dip_depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        actual_depth = dip_depth / k_np
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        # Compute center heights for normalization
        sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l / 2
        z_center = (0.0 + 1j * gamma) / (sigma * np.sqrt(2))

        if self.wofz is not None:
            w_center = self.wofz(z_center)
            center_height = w_center.real / (sigma * np.sqrt(2 * np.pi))
        else:
            # Pseudo-Voigt fallback center height
            eta = (
                1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
            )
            lorentz_center = gamma / (gamma**2)
            gauss_center = 1.0 / (sigma * np.sqrt(2 * np.pi))
            center_height = eta * lorentz_center + (1 - eta) * gauss_center

        def profile_at(center: np.ndarray) -> np.ndarray:
            if self.wofz is not None:
                z = ((x_f - center) + 1j * gamma) / (sigma * np.sqrt(2))
                w = self.wofz(z)
                return w.real / (sigma * np.sqrt(2 * np.pi))
            else:
                eta = (
                    1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                    - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                    + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
                )
                lorentz = gamma / ((x_f - center) ** 2 + gamma**2)
                gauss = np.exp(-0.5 * ((x_f - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                return eta * lorentz + (1 - eta) * gauss

        left_profile = profile_at(freq - split) / center_height
        center_profile = profile_at(freq) / center_height
        right_profile = profile_at(freq + split) / center_height

        left_dip = (actual_depth / k_np) * left_profile
        center_dip = actual_depth * center_profile
        right_dip = (actual_depth * k_np) * right_profile

        return (bg - (left_dip + center_dip + right_dip)).astype(FLOAT_DTYPE, copy=False)

    def compute_vectorized_many(self, x_array: Sequence[float], samples: VoigtZeemanSpectrumSamples) -> np.ndarray:
        if not hasattr(samples, "frequency"):
            return super().compute_vectorized_many(x_array, samples)  # type: ignore[arg-type]

        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")

        freq = np.asarray(samples.frequency, dtype=FLOAT_DTYPE)
        fwhm_total = np.asarray(samples.fwhm_total, dtype=FLOAT_DTYPE)
        lorentz_frac = np.asarray(samples.lorentz_frac, dtype=FLOAT_DTYPE)
        fwhm_l = lorentz_frac * fwhm_total
        fwhm_g = (1.0 - lorentz_frac) * fwhm_total
        split = np.asarray(samples.split, dtype=FLOAT_DTYPE)
        k_np = np.asarray(samples.k_np, dtype=FLOAT_DTYPE)
        dip_depth = np.asarray(samples.dip_depth, dtype=FLOAT_DTYPE)
        actual_depth = dip_depth / k_np
        bg = np.asarray(samples.background, dtype=FLOAT_DTYPE)

        x2d = xs[:, None]
        freq2d = freq[None, :]
        split2d = split[None, :]
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
            eta = (
                1.36603 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d))
                - 0.47719 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 2
                + 0.11116 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 3
            )
            lorentz_center = gamma / (gamma**2)
            gauss_center = 1.0 / (sigma * np.sqrt(2 * np.pi))
            center_height = eta * lorentz_center + (1 - eta) * gauss_center

        def profile_at(center: np.ndarray) -> np.ndarray:
            if self.wofz is not None:
                z = ((x2d - center) + 1j * gamma) / (sigma * np.sqrt(2))
                w = self.wofz(z)
                return w.real / (sigma * np.sqrt(2 * np.pi))
            else:
                eta = (
                    1.36603 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d))
                    - 0.47719 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 2
                    + 0.11116 * (fwhm_l2d / (fwhm_l2d + fwhm_g2d)) ** 3
                )
                lorentz = gamma / ((x2d - center) ** 2 + gamma**2)
                gauss = np.exp(-0.5 * ((x2d - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                return eta * lorentz + (1 - eta) * gauss

        left_profile = profile_at(freq2d - split2d) / center_height
        center_profile = profile_at(freq2d) / center_height
        right_profile = profile_at(freq2d + split2d) / center_height

        amp_l = actual_depth[None, :] / k_np[None, :]
        amp_c = actual_depth[None, :]
        amp_r = actual_depth[None, :] * k_np[None, :]

        out = bg[None, :] - (amp_l * left_profile + amp_c * center_profile + amp_r * right_profile)
        return out.astype(FLOAT_DTYPE, copy=False)

    def sample_params(self, rng: random.Random) -> VoigtZeemanSpectrum:
        """Sample parameters that keep the signal within [0, 1]."""
        fwhm_total = rng.uniform(0.04, 0.13)
        lorentz_frac = rng.uniform(0.23, 0.89)
        fwhm_l = lorentz_frac * fwhm_total
        fwhm_g = (1.0 - lorentz_frac) * fwhm_total
        split = rng.uniform(0.05, 0.12)
        k_np = rng.uniform(2.0, 4.0)
        frequency = rng.uniform(split + 0.1, 1.0 - split - 0.1)
        background = 1.0

        # Estimate dip_depth (right peak depth) using a coarse grid with actual_depth=1/k_np
        xs = np.linspace(frequency - split, frequency + split, 200)
        left_vals = self._voigt_profile_unit_peak(xs, frequency - split, fwhm_l, fwhm_g)
        center_vals = self._voigt_profile_unit_peak(xs, frequency, fwhm_l, fwhm_g)
        right_vals = self._voigt_profile_unit_peak(xs, frequency + split, fwhm_l, fwhm_g)
        g = (left_vals / k_np**2) + (center_vals / k_np) + right_vals
        dip_depth = 1.0 / float(g.max())

        return VoigtZeemanSpectrum(
            frequency=frequency,
            fwhm_total=fwhm_total,
            lorentz_frac=lorentz_frac,
            split=split,
            k_np=k_np,
            dip_depth=dip_depth,
            background=background,
        )
