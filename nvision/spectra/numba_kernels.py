"""Shared scalar kernels for signal models.

Lorentzian pieces are compiled once with Numba and reused by :class:`~nvision.spectra.lorentzian.LorentzianModel`
and :class:`~nvision.spectra.nv_center.NVCenterLorentzianModel`.

The single-Gaussian peak uses :func:`math.exp` only (no Numba): for one scalar per call, the stdlib C
implementation is simpler and avoids an extra compilation unit versus a trivial ``@njit`` wrapper.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def lorentzian_dip_term(x: float, center: float, linewidth: float, dip_depth: float) -> float:
    """``dip_depth * linewidth² / ((x - center)² + linewidth²)`` — one Lorentzian dip contribution."""
    d = (x - center) * (x - center) + linewidth * linewidth
    return (dip_depth * linewidth * linewidth) / d


@njit(cache=True)
def lorentzian_peak_value(
    x: float,
    freq: float,
    linewidth: float,
    dip_depth: float,
    background: float,
) -> float:
    """Single dip: ``background - dip_depth * linewidth² / ((x - freq)² + linewidth²)``."""
    return background - lorentzian_dip_term(x, freq, linewidth, dip_depth)


@njit(cache=True)
def nv_center_lorentzian_eval(
    x: float,
    freq: float,
    linewidth: float,
    split: float,
    k_np: float,
    dip_depth: float,
    background: float,
) -> float:
    """NV triple-Lorentzian ODMR contrast using decoupled dip_depth."""
    if split < 1e-10:
        combined_depth = dip_depth * (k_np + 1.0 + 1.0 / k_np)
        return background - lorentzian_dip_term(x, freq, linewidth, combined_depth)
    left = lorentzian_dip_term(x, freq - split, linewidth, dip_depth / k_np)
    center = lorentzian_dip_term(x, freq, linewidth, dip_depth)
    right = lorentzian_dip_term(x, freq + split, linewidth, dip_depth * k_np)
    return background - (left + center + right)


def gaussian_peak_value(
    x: float,
    freq: float,
    sigma: float,
    amplitude: float,
    background: float,
) -> float:
    """``background + amplitude * exp(-0.5 * ((x - freq) / sigma)²)`` — scalar ``math.exp``."""
    z = (x - freq) / sigma
    return background + amplitude * math.exp(-0.5 * z * z)


@njit(cache=True, parallel=True)
def nv_center_lorentzian_vectorized_many(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    dip_depth: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Triple-Lorentzian ODMR for many probe positions and many particles.

    Writes into ``out`` which must have shape ``(len(xs), len(freq))``.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in prange(m):
        x = xs[i]
        for j in range(n):
            lw = linewidth[j]
            lw2 = lw * lw
            f = freq[j]
            s = split[j]
            k = k_np[j]
            d = dip_depth[j]
            bg = background[j]

            actual_depth = d / k
            denom_l = (x - (f - s)) * (x - (f - s)) + lw2
            denom_c = (x - f) * (x - f) + lw2
            denom_r = (x - (f + s)) * (x - (f + s)) + lw2

            amp_l = actual_depth * lw2 / k
            amp_c = actual_depth * lw2
            amp_r = actual_depth * lw2 * k

            out[i, j] = bg - (amp_l / denom_l + amp_c / denom_c + amp_r / denom_r)


_SQRT2PI = math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_SQRT2LOG2 = math.sqrt(2.0 * math.log(2.0))


@njit(cache=True, parallel=True)
def nv_center_pseudo_voigt_vectorized_many(
    xs: np.ndarray,
    freq: np.ndarray,
    fwhm_total: np.ndarray,
    lorentz_frac: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    dip_depth: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Triple pseudo-Voigt ODMR for many probe positions and many particles.

    Writes into ``out`` which must have shape ``(len(xs), len(freq))``.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in prange(m):
        x = xs[i]
        for j in range(n):
            fwhm = fwhm_total[j]
            lf = lorentz_frac[j]
            fwhm_l = lf * fwhm
            fwhm_g = (1.0 - lf) * fwhm
            f = freq[j]
            s = split[j]
            k = k_np[j]
            d = dip_depth[j]
            bg = background[j]

            sigma = fwhm_g / (2.0 * _SQRT2LOG2)
            gamma = fwhm_l / 2.0
            ratio = fwhm_l / (fwhm_l + fwhm_g)
            eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio
            lorentz_center = 1.0 / gamma if abs(gamma) > 1e-12 else 0.0
            gauss_center = 1.0 / (sigma * _SQRT2PI) if abs(sigma) > 1e-12 else 0.0
            center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
            inv_center_height = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

            actual_depth = d / k

            # Helper to compute pseudo-Voigt profile at x, centered at c
            def _profile(
                xv: float,
                c: float,
                gamma: float = gamma,
                sigma: float = sigma,
                eta: float = eta,
                inv_center_height: float = inv_center_height,
            ) -> float:
                dx = xv - c
                lorentz_v = gamma / (dx * dx + gamma * gamma) if abs(gamma) > 1e-12 else 0.0
                if abs(sigma) > 1e-12:
                    z_v = dx / sigma
                    gauss_v = math.exp(-0.5 * z_v * z_v) / (sigma * _SQRT2PI)
                else:
                    gauss_v = 0.0
                return (eta * lorentz_v + (1.0 - eta) * gauss_v) * inv_center_height

            pc = _profile(x, f)
            if s < 1e-10:
                out[i, j] = bg - actual_depth * pc
            else:
                pl = _profile(x, f - s)
                pr = _profile(x, f + s)
                left_dip = (actual_depth / k) * pl
                center_dip = actual_depth * pc
                right_dip = (actual_depth * k) * pr
                out[i, j] = bg - (left_dip + center_dip + right_dip)
