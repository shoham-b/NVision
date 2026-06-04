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

from nvision.spectra.dtypes import FLOAT_DTYPE

# ---------------------------------------------------------------------------
# Global background array — background is always 1.0 for NV-center models.
# Avoids allocating a fresh np.ones(n) on every update() call (920+ times
# per run).  The array grows on demand and is read-only by convention.
# Thread safety: the array contains only 1.0s, so concurrent reads are safe;
# the single write (resize) replaces the module reference atomically.
# ---------------------------------------------------------------------------

_BG_ONES: np.ndarray = np.ones(0, dtype=FLOAT_DTYPE)


def get_background_ones(n: int) -> np.ndarray:
    """Return a float32 array of ones of length >= n (may be longer).

    The returned slice is a view into a module-level cached array, so no
    heap allocation occurs once the cache is large enough.
    """
    global _BG_ONES
    if len(_BG_ONES) < n:
        _BG_ONES = np.ones(n, dtype=FLOAT_DTYPE)
    return _BG_ONES[:n]


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

    Parallelises over particles.  **Do not use for practical particle counts** —
    thread-coordination overhead (~7 ms) dwarfs the arithmetic at N ≤ ~1 M.
    Use :func:`nv_center_lorentzian_vectorized_one_serial` instead.
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


@njit(cache=True)
def nv_center_lorentzian_vectorized_one_serial(
    x: float,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Serial triple-Lorentzian ODMR for a SINGLE probe position across many particles.

    Identical arithmetic to :func:`nv_center_lorentzian_vectorized_one` but
    compiled without ``parallel=True``.  At practical particle counts (≤ ~1 M)
    this is **26× faster** because Numba's thread-pool coordination costs ~7 ms
    per call — far more than the ~0.03 µs/particle arithmetic.

    Uses ``inv_omega`` to replace two divisions with one division + two
    multiplications in the inner loop.
    """
    n = freq.shape[0]
    for j in range(n):
        lw = linewidth[j]
        f = freq[j]
        s = split[j]
        k = k_np[j]
        c = c_total[j]
        bg = background[j]

        omega = lw if lw > 1e-10 else 1e-10
        inv_omega = 1.0 / omega
        x_dim = (x - f) * inv_omega
        alpha = s * inv_omega

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

    Parallelises over particles.  **Do not use for practical particle counts** —
    see :func:`nv_center_pseudo_voigt_vectorized_one_serial`.
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


@njit(cache=True)
def nv_center_pseudo_voigt_vectorized_one_serial(
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
    """Serial triple pseudo-Voigt ODMR for a SINGLE probe position across many particles.

    Identical arithmetic to :func:`nv_center_pseudo_voigt_vectorized_one` but
    without ``parallel=True``.  Thread overhead dominates at practical particle
    counts; serial is 26x+ faster.
    """
    n = freq.shape[0]
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


@njit(cache=True, fastmath=True)
def nv_center_pseudo_voigt_vectorized_many_fast_serial(
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
    """Serial fast pseudo-Voigt ODMR — EIG path when matrix is too small to justify threads.

    Same arithmetic as :func:`nv_center_pseudo_voigt_vectorized_many_fast` but
    without ``parallel=True``.  Faster than the parallel variant when
    ``n_candidates × n_particles < ~500 000`` because thread-coordination
    overhead (~7 ms) exceeds the compute time for small matrices.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in range(m):
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

    Uses ``inv_omega`` to replace 2 divisions with 1 division + 2 multiplications.
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
            inv_omega = 1.0 / omega
            x_dim = (x - f) * inv_omega
            alpha = s * inv_omega

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


@njit(cache=True, fastmath=True)
def nv_center_lorentzian_vectorized_many_fast_serial(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Serial fast Lorentzian — EIG path when matrix is too small to justify threads.

    Same arithmetic as :func:`nv_center_lorentzian_vectorized_many_fast` but
    without ``parallel=True``.  Faster than the parallel variant when
    ``n_candidates × n_particles < ~500 000`` because thread-coordination
    overhead (~7 ms) exceeds the compute time for small matrices.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in range(m):
        x = xs[i]
        for j in range(n):
            lw = linewidth[j]
            f = freq[j]
            s = split[j]
            k = k_np[j]
            c = c_total[j]
            bg = background[j]

            omega = lw if lw > 1e-10 else 1e-10
            inv_omega = 1.0 / omega
            x_dim = (x - f) * inv_omega
            alpha = s * inv_omega

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


# ---------------------------------------------------------------------------
# Fused EIG-variance kernels -- compute weighted prediction variance per
# candidate without ever materialising the (n_candidates x n_particles) matrix.
# Used exclusively in the EIG / acquisition-scoring path.
#
# Replaces the two-kernel sequence:
#   nv_center_*_vectorized_many_fast  ->  writes 4 MB matrix
#   _weighted_variance_rows           ->  reads  4 MB matrix back
# with a single pass that accumulates weighted mean/mean-square on the fly.
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True, fastmath=True)
def nv_center_lorentzian_eig_variance(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fused: weighted prediction variance per candidate -- Lorentzian EIG path.

    Computes Var_w[f(x, theta)] for each probe position x in xs without
    materialising the (len(xs), len(freq)) predictions matrix.
    Writes one float32 per candidate into out (shape (len(xs),)).

    background is omitted -- for NV-center models it is always 1.0 and
    cancels out of the variance calculation.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in prange(m):
        x = xs[i]
        sum_p  = 0.0
        sum_p2 = 0.0
        for j in range(n):
            lw = linewidth[j]
            f  = freq[j]
            s  = split[j]
            k  = k_np[j]
            c  = c_total[j]
            wi = weights[j]

            omega = lw if lw > 1e-10 else 1e-10
            inv_omega = 1.0 / omega
            x_dim = (x - f) * inv_omega
            alpha = s * inv_omega

            k_safe = k if k > 1e-10 else 1e-10
            inv_k = 1.0 / k_safe                    # 1 div — reused below
            inv_p_sum = 1.0 / (inv_k + 1.0 + k_safe)  # 1 div (was 3)

            p_0 = c * inv_p_sum
            p_L = c * (inv_k * inv_p_sum)
            p_R = c * (k_safe * inv_p_sum)

            pred = 1.0 - (
                p_L / ((x_dim + alpha) ** 2 + 1.0)
                + p_0 / (x_dim ** 2 + 1.0)
                + p_R / ((x_dim - alpha) ** 2 + 1.0)
            )

            sum_p  += wi * pred
            sum_p2 += wi * pred * pred

        v = sum_p2 - sum_p * sum_p
        out[i] = v if v > 0.0 else 0.0


@njit(cache=True, parallel=True, fastmath=True)
def nv_center_pseudo_voigt_eig_variance(
    xs: np.ndarray,
    freq: np.ndarray,
    fwhm_total: np.ndarray,
    lorentz_frac: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    dip_depth: np.ndarray,
    weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fused: weighted prediction variance per candidate -- pseudo-Voigt EIG path.

    Same contract as nv_center_lorentzian_eig_variance.
    """
    m = xs.shape[0]
    n = freq.shape[0]
    for i in prange(m):
        x = xs[i]
        sum_p  = 0.0
        sum_p2 = 0.0
        for j in range(n):
            fwhm = fwhm_total[j]
            lf   = lorentz_frac[j]
            fwhm_l = lf * fwhm
            fwhm_g = (1.0 - lf) * fwhm
            f  = freq[j]
            s  = split[j]
            k  = k_np[j]
            d  = dip_depth[j]
            wi = weights[j]

            sigma = fwhm_g / (2.0 * _SQRT2LOG2)
            gamma = fwhm_l / 2.0
            ratio = fwhm_l / (fwhm_l + fwhm_g)
            eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio

            gamma2 = gamma * gamma
            has_gamma = abs(gamma) > 1e-12
            lorentz_center = 1.0 / gamma if has_gamma else 0.0

            has_sigma = abs(sigma) > 1e-12
            if has_sigma:
                inv_sigma = 1.0 / sigma                      # 1 div — reused twice below
                gauss_center = inv_sigma / _SQRT2PI
                neg_half_inv_sigma2 = -0.5 * inv_sigma * inv_sigma
                eta_gauss_factor = (1.0 - eta) * gauss_center
            else:
                gauss_center = 0.0
                neg_half_inv_sigma2 = 0.0
                eta_gauss_factor = 0.0

            center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
            inv_center_height = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

            inv_k = 1.0 / k                                  # 1 div — reused below
            actual_depth = d * inv_k
            eta_lorentz_factor = eta * gamma * inv_center_height if has_gamma else 0.0
            eta_gauss_factor   = eta_gauss_factor * inv_center_height

            amp_c = actual_depth
            amp_l = amp_c * inv_k                            # was amp_c / k
            amp_r = amp_c * k

            dx_c  = x - f
            dx_c2 = dx_c * dx_c
            lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
            gauss_c   = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
            pc = lorentz_c + gauss_c

            if s < 1e-10:
                pred = 1.0 - amp_c * pc
            else:
                dx_l  = dx_c + s
                dx_l2 = dx_l * dx_l
                lorentz_l = eta_lorentz_factor / (dx_l2 + gamma2) if has_gamma else 0.0
                gauss_l   = eta_gauss_factor * math.exp(dx_l2 * neg_half_inv_sigma2) if has_sigma else 0.0
                pl = lorentz_l + gauss_l

                dx_r  = dx_c - s
                dx_r2 = dx_r * dx_r
                lorentz_r = eta_lorentz_factor / (dx_r2 + gamma2) if has_gamma else 0.0
                gauss_r   = eta_gauss_factor * math.exp(dx_r2 * neg_half_inv_sigma2) if has_sigma else 0.0
                pr = lorentz_r + gauss_r

                pred = 1.0 - (amp_l * pl + amp_c * pc + amp_r * pr)

            sum_p  += wi * pred
            sum_p2 += wi * pred * pred

        v = sum_p2 - sum_p * sum_p
        out[i] = v if v > 0.0 else 0.0
