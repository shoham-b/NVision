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
    for j in prange(n):
        lw = linewidth[j]
        lw2 = lw * lw
        f = freq[j]
        s = split[j]
        k = k_np[j]
        d = dip_depth[j]
        bg = background[j]

        # Precompute particle-specific amplitude scaling
        amp_c = (d / k) * lw2
        amp_l = amp_c / k
        amp_r = amp_c * k

        for i in range(m):
            x = xs[i]
            dx_c = x - f
            dx_l = dx_c + s
            dx_r = dx_c - s

            denom_l = dx_l * dx_l + lw2
            denom_c = dx_c * dx_c + lw2
            denom_r = dx_r * dx_r + lw2

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
    for j in prange(n):
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
        
        gamma2 = gamma * gamma
        has_gamma = abs(gamma) > 1e-12
        lorentz_center = 1.0 / gamma if has_gamma else 0.0
        
        has_sigma = abs(sigma) > 1e-12
        if has_sigma:
            gauss_center = 1.0 / (sigma * _SQRT2PI)
            neg_half_inv_sigma2 = -0.5 / (sigma * sigma)
            eta_gauss_factor = (1.0 - eta) * gauss_center
        else:
            gauss_center = 0.0
            neg_half_inv_sigma2 = 0.0
            eta_gauss_factor = 0.0
            
        center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
        inv_center_height = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

        actual_depth = d / k
        eta_lorentz_factor = eta * gamma * inv_center_height if has_gamma else 0.0
        eta_gauss_factor = eta_gauss_factor * inv_center_height

        amp_c = actual_depth
        amp_l = amp_c / k
        amp_r = amp_c * k
        
        has_split = s >= 1e-10

        if not has_split:
            for i in range(m):
                x = xs[i]
                dx_c = x - f
                dx_c2 = dx_c * dx_c
                
                lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
                gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
                pc = lorentz_c + gauss_c
                
                out[i, j] = bg - amp_c * pc
        else:
            for i in range(m):
                x = xs[i]
                dx_c = x - f
                dx_c2 = dx_c * dx_c
                
                lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
                gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
                pc = lorentz_c + gauss_c

                dx_l = dx_c + s
                dx_l2 = dx_l * dx_l
                lorentz_l = eta_lorentz_factor / (dx_l2 + gamma2) if has_gamma else 0.0
                gauss_l = eta_gauss_factor * math.exp(dx_l2 * neg_half_inv_sigma2) if has_sigma else 0.0
                pl = lorentz_l + gauss_l
                
                dx_r = dx_c - s
                dx_r2 = dx_r * dx_r
                lorentz_r = eta_lorentz_factor / (dx_r2 + gamma2) if has_gamma else 0.0
                gauss_r = eta_gauss_factor * math.exp(dx_r2 * neg_half_inv_sigma2) if has_sigma else 0.0
                pr = lorentz_r + gauss_r
                
                out[i, j] = bg - (amp_l * pl + amp_c * pc + amp_r * pr)
@njit(cache=True)
def nv_center_pseudo_voigt_eval(
    x: float,
    freq: float,
    fwhm_total: float,
    lorentz_frac: float,
    split: float,
    k_np: float,
    dip_depth: float,
    background: float,
) -> float:
    """NV triple pseudo-Voigt ODMR implementation."""
    fwhm_l = lorentz_frac * fwhm_total
    fwhm_g = (1.0 - lorentz_frac) * fwhm_total
    sigma = fwhm_g / (2.0 * _SQRT2LOG2)
    gamma = fwhm_l / 2.0
    
    ratio = fwhm_l / (fwhm_l + fwhm_g)
    eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio
    
    def _profile(xv, center):
        dx = xv - center
        dx2 = dx * dx
        
        # Heights for normalization
        has_gamma = abs(gamma) > 1e-12
        lorentz_peak = 1.0 / gamma if has_gamma else 0.0
        
        has_sigma = abs(sigma) > 1e-12
        if has_sigma:
            gauss_peak = 1.0 / (sigma * _SQRT2PI)
            gauss = math.exp(-0.5 * (dx2 / (sigma * sigma))) / (sigma * _SQRT2PI)
        else:
            gauss_peak = 0.0
            gauss = 0.0
            
        lorentz = gamma / (dx2 + gamma * gamma) if has_gamma else 0.0
        peak = eta * lorentz_peak + (1.0 - eta) * gauss_peak
        profile = eta * lorentz + (1.0 - eta) * gauss
        return profile / peak if abs(peak) > 1e-12 else 0.0

    pc = _profile(x, freq)
    actual_depth = dip_depth / k_np
    
    if split < 1e-10:
        return background - actual_depth * pc

    pl = _profile(x, freq - split)
    pr = _profile(x, freq + split)
    return background - (actual_depth / k_np * pl + actual_depth * pc + actual_depth * k_np * pr)
