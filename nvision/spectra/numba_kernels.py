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
    c_total: float,
    background: float,
) -> float:
    """NV triple-Lorentzian ODMR contrast using Population-Normalized Geometric Reparameterization."""
    omega = linewidth if linewidth > 1e-10 else 1e-10
    x_dim = (x - freq) / omega
    alpha = split / omega

    k = k_np if k_np > 1e-10 else 1e-10
    p_sum = (1.0 / k) + 1.0 + k

    p_0 = c_total / p_sum
    p_L = c_total * ((1.0 / k) / p_sum)
    p_R = c_total * (k / p_sum)

    return background - (
        p_L / ((x_dim + alpha) ** 2 + 1.0)
        + p_0 / (x_dim ** 2 + 1.0)
        + p_R / ((x_dim - alpha) ** 2 + 1.0)
    )


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
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Triple-Lorentzian ODMR for many probe positions and many particles.

    Writes into ``out`` which must have shape ``(len(xs), len(freq))``.

    Parallelises over probe positions (rows) — ``out[i, :]`` is contiguous
    in row-major (C) order, so each thread writes a full cache line at a time.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in prange(m):
        x = xs[i]
        for j in range(n):
            lw = linewidth[j]
            f = freq[j]
            s = split[j]
            k = k_np[j]
            c = c_total[j]
            bg = background[j]

            omega = lw if lw > 1e-10 else 1e-10
            x_dim = (x - f) / omega
            alpha = s / omega

            k_safe = k if k > 1e-10 else 1e-10
            p_sum = (1.0 / k_safe) + 1.0 + k_safe

            p_0 = c / p_sum
            p_L = c * ((1.0 / k_safe) / p_sum)
            p_R = c * (k_safe / p_sum)

            out[i, j] = bg - (
                p_L / ((x_dim + alpha) ** 2 + 1.0)
                + p_0 / (x_dim ** 2 + 1.0)
                + p_R / ((x_dim - alpha) ** 2 + 1.0)
            )


@njit(cache=True, parallel=True)
def nv_center_lorentzian_vectorized_one(
    x: float,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Triple-Lorentzian ODMR for a SINGLE probe position across many particles.

    Writes into ``out`` which must have shape ``(len(freq),)``.

    Parallelises over particles — the correct layout when m=1 (used in
    :meth:`SMCMarginalDistribution.update` which evaluates one x at a time).
    """
    n = freq.shape[0]
    for j in prange(n):
        lw = linewidth[j]
        f = freq[j]
        s = split[j]
        k = k_np[j]
        c = c_total[j]
        bg = background[j]

        omega = lw if lw > 1e-10 else 1e-10
        x_dim = (x - f) / omega
        alpha = s / omega

        k_safe = k if k > 1e-10 else 1e-10
        p_sum = (1.0 / k_safe) + 1.0 + k_safe

        p_0 = c / p_sum
        p_L = c * ((1.0 / k_safe) / p_sum)
        p_R = c * (k_safe / p_sum)

        out[j] = bg - (
            p_L / ((x_dim + alpha) ** 2 + 1.0)
            + p_0 / (x_dim ** 2 + 1.0)
            + p_R / ((x_dim - alpha) ** 2 + 1.0)
        )


@njit(cache=True, parallel=True)
def nv_center_pseudo_voigt_vectorized_one(
    x: float,
    freq: np.ndarray,
    fwhm_total: np.ndarray,
    lorentz_frac: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    dip_depth: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Triple pseudo-Voigt ODMR for a SINGLE probe position across many particles.

    Writes into ``out`` which must have shape ``(len(freq),)``.

    Parallelises over particles — the correct layout for the update() hot path.
    """
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

        dx_c = x - f
        dx_c2 = dx_c * dx_c
        lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
        gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
        pc = lorentz_c + gauss_c

        if s < 1e-10:
            out[j] = bg - amp_c * pc
        else:
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

            out[j] = bg - (amp_l * pl + amp_c * pc + amp_r * pr)





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

    Parallelises over probe positions (rows) — ``out[i, :]`` is contiguous
    in row-major (C) order, so each thread writes a full cache line at a time.
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

            dx_c = x - f
            dx_c2 = dx_c * dx_c

            lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
            gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
            pc = lorentz_c + gauss_c

            if s < 1e-10:
                out[i, j] = bg - amp_c * pc
            else:
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


# ---------------------------------------------------------------------------
# Fast (fastmath=True) acquisition-only variants of the _many kernels.
# Use ONLY in EIG / information-gain scoring where approximate arithmetic is
# acceptable. Never use in weight updates or uncertainty computation.
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True, fastmath=True)
def nv_center_lorentzian_vectorized_many_fast(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fast (fastmath) triple-Lorentzian ODMR — acquisition / EIG path only.

    Identical body to :func:`nv_center_lorentzian_vectorized_many` but compiled
    with ``fastmath=True``.  Do **not** use this for Bayesian weight updates or
    uncertainty estimation.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in prange(m):
        x = xs[i]
        for j in range(n):
            lw = linewidth[j]
            f = freq[j]
            s = split[j]
            k = k_np[j]
            c = c_total[j]
            bg = background[j]

            omega = lw if lw > 1e-10 else 1e-10
            x_dim = (x - f) / omega
            alpha = s / omega

            k_safe = k if k > 1e-10 else 1e-10
            p_sum = (1.0 / k_safe) + 1.0 + k_safe

            p_0 = c / p_sum
            p_L = c * ((1.0 / k_safe) / p_sum)
            p_R = c * (k_safe / p_sum)

            out[i, j] = bg - (
                p_L / ((x_dim + alpha) ** 2 + 1.0)
                + p_0 / (x_dim ** 2 + 1.0)
                + p_R / ((x_dim - alpha) ** 2 + 1.0)
            )


@njit(cache=True, parallel=True, fastmath=True)
def nv_center_pseudo_voigt_vectorized_many_fast(
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
    """Fast (fastmath) triple pseudo-Voigt ODMR — acquisition / EIG path only.

    Identical body to :func:`nv_center_pseudo_voigt_vectorized_many` but compiled
    with ``fastmath=True``.  Do **not** use this for Bayesian weight updates or
    uncertainty estimation.
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

            dx_c = x - f
            dx_c2 = dx_c * dx_c

            lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
            gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
            pc = lorentz_c + gauss_c

            if s < 1e-10:
                out[i, j] = bg - amp_c * pc
            else:
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
