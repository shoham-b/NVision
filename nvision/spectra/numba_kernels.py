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
        p_L / ((x_dim + alpha) ** 2 + 1.0) + p_0 / (x_dim**2 + 1.0) + p_R / ((x_dim - alpha) ** 2 + 1.0)
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

    # Per-particle precompute: hoists all particle-only math (three divisions
    # per particle) out of the m x n inner loop — see nv_center_lorentzian_eig_variance
    # for the same pattern applied to the fused EIG-variance kernel.
    inv_omega = np.empty(n, dtype=np.float64)
    alpha_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        lw = linewidth[j]
        omega = lw if lw > 1e-10 else 1e-10
        io = 1.0 / omega
        inv_omega[j] = io
        alpha_arr[j] = split[j] * io

        k = k_np[j]
        k_safe = k if k > 1e-10 else 1e-10
        inv_k = 1.0 / k_safe
        inv_p_sum = 1.0 / (inv_k + 1.0 + k_safe)
        c = c_total[j]
        p_0_arr[j] = c * inv_p_sum
        p_l_arr[j] = c * (inv_k * inv_p_sum)
        p_r_arr[j] = c * (k_safe * inv_p_sum)

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            x_dim = (x - freq[j]) * inv_omega[j]
            alpha = alpha_arr[j]
            d_l = x_dim + alpha
            d_r = x_dim - alpha

            out[i, j] = background[j] - (
                p_l_arr[j] / (d_l * d_l + 1.0) + p_0_arr[j] / (x_dim * x_dim + 1.0) + p_r_arr[j] / (d_r * d_r + 1.0)
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

        out[j] = bg - (p_L / ((x_dim + alpha) ** 2 + 1.0) + p_0 / (x_dim**2 + 1.0) + p_R / ((x_dim - alpha) ** 2 + 1.0))


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

        out[j] = bg - (p_L / ((x_dim + alpha) ** 2 + 1.0) + p_0 / (x_dim**2 + 1.0) + p_R / ((x_dim - alpha) ** 2 + 1.0))


# ---------------------------------------------------------------------------
# Zeeman-split Lorentzian kernels — symmetric two-group NV model.
#
# Signal = background − Σ_{±} half_triplet(x, freq ± zeeman_split)
#
# Each group is a standard NV hyperfine triplet with half the total contrast.
# When zeeman_split → 0 the two groups coincide and the formula collapses to
# the standard single-group triple-Lorentzian with full contrast c_total.
# ---------------------------------------------------------------------------


@njit(cache=True)
def nv_center_zeeman_lorentzian_eval(
    x: float,
    freq: float,
    linewidth: float,
    zeeman_split: float,
    hf_split: float,
    k_np: float,
    c_total: float,
    background: float,
) -> float:
    """Zeeman + hyperfine NV ODMR scalar kernel.

    Two triple-Lorentzian groups centered at freq ± zeeman_split, mirrored
    about the true center: the ms=+1/ms=-1 groups are physical mirror images
    of each other, so the sub-line closest to center in one group must carry
    the same population weight as the sub-line closest to center in the
    other, not the weight at the same offset from its own group's center.
    """
    omega = linewidth if linewidth > 1e-10 else 1e-10
    inv_omega = 1.0 / omega
    x_dim = (x - freq) * inv_omega
    alpha = hf_split * inv_omega
    beta = zeeman_split * inv_omega

    k = k_np if k_np > 1e-10 else 1e-10
    p_sum = (1.0 / k) + 1.0 + k

    p_0 = 0.5 * c_total / p_sum
    p_L = 0.5 * c_total * (1.0 / k) / p_sum
    p_R = 0.5 * c_total * k / p_sum

    x_m = x_dim + beta  # (x − (freq − zeeman_split)) / omega
    left = p_L / ((x_m + alpha) ** 2 + 1.0) + p_0 / (x_m ** 2 + 1.0) + p_R / ((x_m - alpha) ** 2 + 1.0)

    x_p = x_dim - beta  # (x − (freq + zeeman_split)) / omega
    # p_L/p_R swapped vs. the left group: x_p - alpha (closest to true center)
    # mirrors x_m + alpha (closest to true center in the left group, p_R).
    right = p_R / ((x_p + alpha) ** 2 + 1.0) + p_0 / (x_p ** 2 + 1.0) + p_L / ((x_p - alpha) ** 2 + 1.0)

    return background - (left + right)


@njit(cache=True)
def nv_center_zeeman_lorentzian_vectorized_one_serial(
    x: float,
    freq: np.ndarray,
    linewidth: np.ndarray,
    zeeman_split: np.ndarray,
    hf_split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zeeman serial kernel: single probe x across many particles."""
    n = freq.shape[0]
    for j in range(n):
        lw = linewidth[j]
        f = freq[j]
        z = zeeman_split[j]
        hf = hf_split[j]
        k = k_np[j]
        c = c_total[j]
        bg = background[j]

        omega = lw if lw > 1e-10 else 1e-10
        inv_omega = 1.0 / omega
        x_dim = (x - f) * inv_omega
        alpha = hf * inv_omega
        beta = z * inv_omega

        k_safe = k if k > 1e-10 else 1e-10
        p_sum = (1.0 / k_safe) + 1.0 + k_safe

        p_0 = 0.5 * c / p_sum
        p_L = 0.5 * c * (1.0 / k_safe) / p_sum
        p_R = 0.5 * c * k_safe / p_sum

        x_m = x_dim + beta
        left = p_L / ((x_m + alpha) ** 2 + 1.0) + p_0 / (x_m ** 2 + 1.0) + p_R / ((x_m - alpha) ** 2 + 1.0)

        # p_L/p_R swapped vs. the left group -- mirror the two Zeeman groups
        # about the true center (see nv_center_zeeman_lorentzian_eval).
        x_p = x_dim - beta
        right = p_R / ((x_p + alpha) ** 2 + 1.0) + p_0 / (x_p ** 2 + 1.0) + p_L / ((x_p - alpha) ** 2 + 1.0)

        out[j] = bg - (left + right)


@njit(cache=True, parallel=True)
def nv_center_zeeman_lorentzian_vectorized_many(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    zeeman_split: np.ndarray,
    hf_split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zeeman parallel kernel: many probes × many particles. Writes into out[m, n]."""
    m = xs.shape[0]
    n = freq.shape[0]

    # Per-particle precompute: hoists all particle-only math out of the m x n
    # inner loop (same pattern as nv_center_lorentzian_eig_variance).
    inv_omega = np.empty(n, dtype=np.float64)
    alpha_arr = np.empty(n, dtype=np.float64)
    beta_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        lw = linewidth[j]
        omega = lw if lw > 1e-10 else 1e-10
        io = 1.0 / omega
        inv_omega[j] = io
        alpha_arr[j] = hf_split[j] * io
        beta_arr[j] = zeeman_split[j] * io

        k = k_np[j]
        k_safe = k if k > 1e-10 else 1e-10
        inv_k = 1.0 / k_safe
        inv_p_sum = 1.0 / (inv_k + 1.0 + k_safe)
        c = c_total[j]
        p_0_arr[j] = 0.5 * c * inv_p_sum
        p_l_arr[j] = 0.5 * c * (inv_k * inv_p_sum)
        p_r_arr[j] = 0.5 * c * (k_safe * inv_p_sum)

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            x_dim = (x - freq[j]) * inv_omega[j]
            alpha = alpha_arr[j]
            beta = beta_arr[j]
            p_0 = p_0_arr[j]
            p_l = p_l_arr[j]
            p_r = p_r_arr[j]

            x_m = x_dim + beta
            left = p_l / ((x_m + alpha) ** 2 + 1.0) + p_0 / (x_m ** 2 + 1.0) + p_r / ((x_m - alpha) ** 2 + 1.0)

            # p_l/p_r swapped vs. the left group -- mirror the two Zeeman groups
            # about the true center (see nv_center_zeeman_lorentzian_eval).
            x_p = x_dim - beta
            right = p_r / ((x_p + alpha) ** 2 + 1.0) + p_0 / (x_p ** 2 + 1.0) + p_l / ((x_p - alpha) ** 2 + 1.0)

            out[i, j] = background[j] - (left + right)


@njit(cache=True, parallel=True, fastmath=True)
def nv_center_zeeman_lorentzian_vectorized_many_fast(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    zeeman_split: np.ndarray,
    hf_split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fast (fastmath) Zeeman kernel — acquisition / EIG path only."""
    m = xs.shape[0]
    n = freq.shape[0]

    # Per-particle precompute: hoists all particle-only math out of the m x n
    # inner loop (same pattern as nv_center_lorentzian_eig_variance).
    inv_omega = np.empty(n, dtype=np.float64)
    alpha_arr = np.empty(n, dtype=np.float64)
    beta_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        lw = linewidth[j]
        omega = lw if lw > 1e-10 else 1e-10
        io = 1.0 / omega
        inv_omega[j] = io
        alpha_arr[j] = hf_split[j] * io
        beta_arr[j] = zeeman_split[j] * io

        k = k_np[j]
        k_safe = k if k > 1e-10 else 1e-10
        inv_k = 1.0 / k_safe
        inv_p_sum = 1.0 / (inv_k + 1.0 + k_safe)
        c = c_total[j]
        p_0_arr[j] = 0.5 * c * inv_p_sum
        p_l_arr[j] = 0.5 * c * (inv_k * inv_p_sum)
        p_r_arr[j] = 0.5 * c * (k_safe * inv_p_sum)

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            x_dim = (x - freq[j]) * inv_omega[j]
            alpha = alpha_arr[j]
            beta = beta_arr[j]
            p_0 = p_0_arr[j]
            p_l = p_l_arr[j]
            p_r = p_r_arr[j]

            x_m = x_dim + beta
            left = p_l / ((x_m + alpha) ** 2 + 1.0) + p_0 / (x_m ** 2 + 1.0) + p_r / ((x_m - alpha) ** 2 + 1.0)

            # p_l/p_r swapped vs. the left group -- mirror the two Zeeman groups
            # about the true center (see nv_center_zeeman_lorentzian_eval).
            x_p = x_dim - beta
            right = p_r / ((x_p + alpha) ** 2 + 1.0) + p_0 / (x_p ** 2 + 1.0) + p_l / ((x_p - alpha) ** 2 + 1.0)

            out[i, j] = background[j] - (left + right)


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
        f = freq[j]
        s = split[j]
        k = k_np[j]
        d = dip_depth[j]
        bg = background[j]

        gamma = fwhm / 2.0
        sigma = fwhm / (2.0 * _SQRT2LOG2)
        ratio = lorentz_frac[j] if fwhm > 1e-30 else 0.0
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
        f = freq[j]
        s = split[j]
        k = k_np[j]
        d = dip_depth[j]
        bg = background[j]

        gamma = fwhm / 2.0
        sigma = fwhm / (2.0 * _SQRT2LOG2)
        ratio = lorentz_frac[j] if fwhm > 1e-30 else 0.0
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

    # Per-particle precompute: hoists the entire pseudo-Voigt parameterisation
    # (eta polynomial, several divisions, factor setup) out of the m x n inner
    # loop — same pattern as nv_center_pseudo_voigt_eig_variance.
    elf_arr = np.empty(n, dtype=np.float64)
    egf_arr = np.empty(n, dtype=np.float64)
    nhs_arr = np.empty(n, dtype=np.float64)
    gamma2_arr = np.empty(n, dtype=np.float64)
    amp_l_arr = np.empty(n, dtype=np.float64)
    amp_c_arr = np.empty(n, dtype=np.float64)
    amp_r_arr = np.empty(n, dtype=np.float64)
    has_gamma_arr = np.empty(n, dtype=np.bool_)
    has_sigma_arr = np.empty(n, dtype=np.bool_)
    for j in range(n):
        fwhm = fwhm_total[j]
        k = k_np[j]
        d = dip_depth[j]

        gamma = fwhm / 2.0
        sigma = fwhm / (2.0 * _SQRT2LOG2)
        ratio = lorentz_frac[j] if fwhm > 1e-30 else 0.0
        eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio

        gamma2 = gamma * gamma
        has_gamma = abs(gamma) > 1e-12
        lorentz_center = 1.0 / gamma if has_gamma else 0.0

        has_sigma = abs(sigma) > 1e-12
        if has_sigma:
            inv_sigma = 1.0 / sigma
            gauss_center = inv_sigma / _SQRT2PI
            neg_half_inv_sigma2 = -0.5 * inv_sigma * inv_sigma
            eta_gauss_factor = (1.0 - eta) * gauss_center
        else:
            gauss_center = 0.0
            neg_half_inv_sigma2 = 0.0
            eta_gauss_factor = 0.0

        center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
        inv_center_height = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

        actual_depth = d / k

        elf_arr[j] = eta * gamma * inv_center_height if has_gamma else 0.0
        egf_arr[j] = eta_gauss_factor * inv_center_height
        nhs_arr[j] = neg_half_inv_sigma2
        gamma2_arr[j] = gamma2
        amp_c_arr[j] = actual_depth
        amp_l_arr[j] = actual_depth / k
        amp_r_arr[j] = actual_depth * k
        has_gamma_arr[j] = has_gamma
        has_sigma_arr[j] = has_sigma

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            eta_lorentz_factor = elf_arr[j]
            eta_gauss_factor = egf_arr[j]
            neg_half_inv_sigma2 = nhs_arr[j]
            gamma2 = gamma2_arr[j]
            has_gamma = has_gamma_arr[j]
            has_sigma = has_sigma_arr[j]
            s = split[j]

            dx_c = x - freq[j]
            dx_c2 = dx_c * dx_c

            lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
            gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
            pc = lorentz_c + gauss_c

            if s < 1e-10:
                out[i, j] = background[j] - amp_c_arr[j] * pc
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

                out[i, j] = background[j] - (amp_l_arr[j] * pl + amp_c_arr[j] * pc + amp_r_arr[j] * pr)


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

    # Per-particle precompute: hoists the entire pseudo-Voigt parameterisation
    # out of the m x n inner loop — same pattern as nv_center_pseudo_voigt_eig_variance.
    elf_arr = np.empty(n, dtype=np.float64)
    egf_arr = np.empty(n, dtype=np.float64)
    nhs_arr = np.empty(n, dtype=np.float64)
    gamma2_arr = np.empty(n, dtype=np.float64)
    amp_l_arr = np.empty(n, dtype=np.float64)
    amp_c_arr = np.empty(n, dtype=np.float64)
    amp_r_arr = np.empty(n, dtype=np.float64)
    has_gamma_arr = np.empty(n, dtype=np.bool_)
    has_sigma_arr = np.empty(n, dtype=np.bool_)
    for j in range(n):
        fwhm = fwhm_total[j]
        k = k_np[j]
        d = dip_depth[j]

        gamma = fwhm / 2.0
        sigma = fwhm / (2.0 * _SQRT2LOG2)
        ratio = lorentz_frac[j] if fwhm > 1e-30 else 0.0
        eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio

        gamma2 = gamma * gamma
        has_gamma = abs(gamma) > 1e-12
        lorentz_center = 1.0 / gamma if has_gamma else 0.0

        has_sigma = abs(sigma) > 1e-12
        if has_sigma:
            inv_sigma = 1.0 / sigma
            gauss_center = inv_sigma / _SQRT2PI
            neg_half_inv_sigma2 = -0.5 * inv_sigma * inv_sigma
            eta_gauss_factor = (1.0 - eta) * gauss_center
        else:
            gauss_center = 0.0
            neg_half_inv_sigma2 = 0.0
            eta_gauss_factor = 0.0

        center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
        inv_center_height = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

        actual_depth = d / k

        elf_arr[j] = eta * gamma * inv_center_height if has_gamma else 0.0
        egf_arr[j] = eta_gauss_factor * inv_center_height
        nhs_arr[j] = neg_half_inv_sigma2
        gamma2_arr[j] = gamma2
        amp_c_arr[j] = actual_depth
        amp_l_arr[j] = actual_depth / k
        amp_r_arr[j] = actual_depth * k
        has_gamma_arr[j] = has_gamma
        has_sigma_arr[j] = has_sigma

    for i in range(m):
        x = xs[i]
        for j in range(n):
            eta_lorentz_factor = elf_arr[j]
            eta_gauss_factor = egf_arr[j]
            neg_half_inv_sigma2 = nhs_arr[j]
            gamma2 = gamma2_arr[j]
            has_gamma = has_gamma_arr[j]
            has_sigma = has_sigma_arr[j]
            s = split[j]

            dx_c = x - freq[j]
            dx_c2 = dx_c * dx_c

            lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
            gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
            pc = lorentz_c + gauss_c

            if s < 1e-10:
                out[i, j] = background[j] - amp_c_arr[j] * pc
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

                out[i, j] = background[j] - (amp_l_arr[j] * pl + amp_c_arr[j] * pc + amp_r_arr[j] * pr)


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
    gamma = fwhm_total / 2.0
    sigma = fwhm_total / (2.0 * _SQRT2LOG2)

    ratio = lorentz_frac if fwhm_total > 1e-30 else 0.0
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

    # Per-particle precompute: hoists all particle-only math out of the m x n
    # inner loop (same pattern as nv_center_lorentzian_eig_variance).
    inv_omega = np.empty(n, dtype=np.float64)
    alpha_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        lw = linewidth[j]
        omega = lw if lw > 1e-10 else 1e-10
        io = 1.0 / omega
        inv_omega[j] = io
        alpha_arr[j] = split[j] * io

        k = k_np[j]
        k_safe = k if k > 1e-10 else 1e-10
        inv_k = 1.0 / k_safe
        inv_p_sum = 1.0 / (inv_k + 1.0 + k_safe)
        c = c_total[j]
        p_0_arr[j] = c * inv_p_sum
        p_l_arr[j] = c * (inv_k * inv_p_sum)
        p_r_arr[j] = c * (k_safe * inv_p_sum)

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            x_dim = (x - freq[j]) * inv_omega[j]
            alpha = alpha_arr[j]
            d_l = x_dim + alpha
            d_r = x_dim - alpha

            out[i, j] = background[j] - (
                p_l_arr[j] / (d_l * d_l + 1.0) + p_0_arr[j] / (x_dim * x_dim + 1.0) + p_r_arr[j] / (d_r * d_r + 1.0)
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

    # Per-particle precompute: hoists all particle-only math out of the m x n
    # inner loop (same pattern as nv_center_lorentzian_eig_variance).
    inv_omega = np.empty(n, dtype=np.float64)
    alpha_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        lw = linewidth[j]
        omega = lw if lw > 1e-10 else 1e-10
        io = 1.0 / omega
        inv_omega[j] = io
        alpha_arr[j] = split[j] * io

        k = k_np[j]
        k_safe = k if k > 1e-10 else 1e-10
        inv_k = 1.0 / k_safe
        inv_p_sum = 1.0 / (inv_k + 1.0 + k_safe)
        c = c_total[j]
        p_0_arr[j] = c * inv_p_sum
        p_l_arr[j] = c * (inv_k * inv_p_sum)
        p_r_arr[j] = c * (k_safe * inv_p_sum)

    for i in range(m):
        x = xs[i]
        for j in range(n):
            x_dim = (x - freq[j]) * inv_omega[j]
            alpha = alpha_arr[j]
            d_l = x_dim + alpha
            d_r = x_dim - alpha

            out[i, j] = background[j] - (
                p_l_arr[j] / (d_l * d_l + 1.0) + p_0_arr[j] / (x_dim * x_dim + 1.0) + p_r_arr[j] / (d_r * d_r + 1.0)
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

    # Per-particle precompute: hoists the entire pseudo-Voigt parameterisation
    # out of the m x n inner loop — same pattern as nv_center_pseudo_voigt_eig_variance.
    elf_arr = np.empty(n, dtype=np.float64)
    egf_arr = np.empty(n, dtype=np.float64)
    nhs_arr = np.empty(n, dtype=np.float64)
    gamma2_arr = np.empty(n, dtype=np.float64)
    amp_l_arr = np.empty(n, dtype=np.float64)
    amp_c_arr = np.empty(n, dtype=np.float64)
    amp_r_arr = np.empty(n, dtype=np.float64)
    has_gamma_arr = np.empty(n, dtype=np.bool_)
    has_sigma_arr = np.empty(n, dtype=np.bool_)
    for j in range(n):
        fwhm = fwhm_total[j]
        k = k_np[j]
        d = dip_depth[j]

        gamma = fwhm / 2.0
        sigma = fwhm / (2.0 * _SQRT2LOG2)
        ratio = lorentz_frac[j] if fwhm > 1e-30 else 0.0
        eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio

        gamma2 = gamma * gamma
        has_gamma = abs(gamma) > 1e-12
        lorentz_center = 1.0 / gamma if has_gamma else 0.0

        has_sigma = abs(sigma) > 1e-12
        if has_sigma:
            inv_sigma = 1.0 / sigma
            gauss_center = inv_sigma / _SQRT2PI
            neg_half_inv_sigma2 = -0.5 * inv_sigma * inv_sigma
            eta_gauss_factor = (1.0 - eta) * gauss_center
        else:
            gauss_center = 0.0
            neg_half_inv_sigma2 = 0.0
            eta_gauss_factor = 0.0

        center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
        inv_center_height = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

        actual_depth = d / k

        elf_arr[j] = eta * gamma * inv_center_height if has_gamma else 0.0
        egf_arr[j] = eta_gauss_factor * inv_center_height
        nhs_arr[j] = neg_half_inv_sigma2
        gamma2_arr[j] = gamma2
        amp_c_arr[j] = actual_depth
        amp_l_arr[j] = actual_depth / k
        amp_r_arr[j] = actual_depth * k
        has_gamma_arr[j] = has_gamma
        has_sigma_arr[j] = has_sigma

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            eta_lorentz_factor = elf_arr[j]
            eta_gauss_factor = egf_arr[j]
            neg_half_inv_sigma2 = nhs_arr[j]
            gamma2 = gamma2_arr[j]
            has_gamma = has_gamma_arr[j]
            has_sigma = has_sigma_arr[j]
            s = split[j]

            dx_c = x - freq[j]
            dx_c2 = dx_c * dx_c

            lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
            gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
            pc = lorentz_c + gauss_c

            if s < 1e-10:
                out[i, j] = background[j] - amp_c_arr[j] * pc
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

                out[i, j] = background[j] - (amp_l_arr[j] * pl + amp_c_arr[j] * pc + amp_r_arr[j] * pr)


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

    # Per-particle precompute: hoists all particle-only math (including three
    # divisions per pair) out of the m x n inner loop. float64 arrays keep the
    # numerics identical to the previous per-pair scalar computation.
    inv_omega = np.empty(n, dtype=np.float64)
    alpha = np.empty(n, dtype=np.float64)
    p_0 = np.empty(n, dtype=np.float64)
    p_l = np.empty(n, dtype=np.float64)
    p_r = np.empty(n, dtype=np.float64)
    for j in range(n):
        lw = linewidth[j]
        omega = lw if lw > 1e-10 else 1e-10
        io = 1.0 / omega
        inv_omega[j] = io
        alpha[j] = split[j] * io

        k = k_np[j]
        k_safe = k if k > 1e-10 else 1e-10
        inv_k = 1.0 / k_safe
        inv_p_sum = 1.0 / (inv_k + 1.0 + k_safe)
        c = c_total[j]
        p_0[j] = c * inv_p_sum
        p_l[j] = c * (inv_k * inv_p_sum)
        p_r[j] = c * (k_safe * inv_p_sum)

    for i in prange(m):
        x = xs[i]
        sum_p = 0.0
        sum_p2 = 0.0
        for j in range(n):
            x_dim = (x - freq[j]) * inv_omega[j]
            a = alpha[j]
            d_l = x_dim + a
            d_r = x_dim - a
            pred = 1.0 - (p_l[j] / (d_l * d_l + 1.0) + p_0[j] / (x_dim * x_dim + 1.0) + p_r[j] / (d_r * d_r + 1.0))

            wi = weights[j]
            sum_p += wi * pred
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

    # Per-particle precompute: hoists the entire pseudo-Voigt parameterisation
    # (eta polynomial, several divisions, factor setup) out of the m x n inner
    # loop. float64 arrays keep the numerics identical to the previous
    # per-pair scalar computation.
    elf_arr = np.empty(n, dtype=np.float64)  # eta * gamma * inv_center_height (0 when no gamma)
    egf_arr = np.empty(n, dtype=np.float64)  # (1-eta) * gauss_center * inv_center_height (0 when no sigma)
    nhs_arr = np.empty(n, dtype=np.float64)  # -0.5 / sigma^2 (0 when no sigma)
    gamma2_arr = np.empty(n, dtype=np.float64)
    amp_l_arr = np.empty(n, dtype=np.float64)
    amp_c_arr = np.empty(n, dtype=np.float64)
    amp_r_arr = np.empty(n, dtype=np.float64)
    has_gamma_arr = np.empty(n, dtype=np.bool_)
    has_sigma_arr = np.empty(n, dtype=np.bool_)
    for j in range(n):
        fwhm = fwhm_total[j]
        k = k_np[j]
        d = dip_depth[j]

        gamma = fwhm / 2.0
        sigma = fwhm / (2.0 * _SQRT2LOG2)
        ratio = lorentz_frac[j] if fwhm > 1e-30 else 0.0
        eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio

        gamma2 = gamma * gamma
        has_gamma = abs(gamma) > 1e-12
        lorentz_center = 1.0 / gamma if has_gamma else 0.0

        has_sigma = abs(sigma) > 1e-12
        if has_sigma:
            inv_sigma = 1.0 / sigma
            gauss_center = inv_sigma / _SQRT2PI
            neg_half_inv_sigma2 = -0.5 * inv_sigma * inv_sigma
            eta_gauss_factor = (1.0 - eta) * gauss_center
        else:
            gauss_center = 0.0
            neg_half_inv_sigma2 = 0.0
            eta_gauss_factor = 0.0

        center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
        inv_center_height = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

        inv_k = 1.0 / k
        actual_depth = d * inv_k

        elf_arr[j] = eta * gamma * inv_center_height if has_gamma else 0.0
        egf_arr[j] = eta_gauss_factor * inv_center_height
        nhs_arr[j] = neg_half_inv_sigma2
        gamma2_arr[j] = gamma2
        amp_c_arr[j] = actual_depth
        amp_l_arr[j] = actual_depth * inv_k
        amp_r_arr[j] = actual_depth * k
        has_gamma_arr[j] = has_gamma
        has_sigma_arr[j] = has_sigma

    for i in prange(m):
        x = xs[i]
        sum_p = 0.0
        sum_p2 = 0.0
        for j in range(n):
            eta_lorentz_factor = elf_arr[j]
            eta_gauss_factor = egf_arr[j]
            neg_half_inv_sigma2 = nhs_arr[j]
            gamma2 = gamma2_arr[j]
            has_gamma = has_gamma_arr[j]
            has_sigma = has_sigma_arr[j]
            s = split[j]
            wi = weights[j]

            dx_c = x - freq[j]
            dx_c2 = dx_c * dx_c
            lorentz_c = eta_lorentz_factor / (dx_c2 + gamma2) if has_gamma else 0.0
            gauss_c = eta_gauss_factor * math.exp(dx_c2 * neg_half_inv_sigma2) if has_sigma else 0.0
            pc = lorentz_c + gauss_c

            if s < 1e-10:
                pred = 1.0 - amp_c_arr[j] * pc
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

                pred = 1.0 - (amp_l_arr[j] * pl + amp_c_arr[j] * pc + amp_r_arr[j] * pr)

            sum_p += wi * pred
            sum_p2 += wi * pred * pred

        v = sum_p2 - sum_p * sum_p
        out[i] = v if v > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Zeeman-split pseudo-Voigt kernels — population-normalized (c_total) framing.
#
# Signal = background − Σ_{±zeeman} Σ_{p∈{L,0,R}} p_weight · V_norm(x − center_p)
#
# This is the Voigt analogue of the Zeeman Lorentzian kernels: it keeps the
# identical *population* reparameterization (total contrast ``c_total`` split
# geometrically by ``k_np`` across the hyperfine triplet, and evenly across the
# two Zeeman groups) but replaces each unit-height Lorentzian ``1/(u²+1)`` with a
# unit-height pseudo-Voigt profile ``V_norm``.  ``V_norm`` combines the
# homogeneous (Lorentzian, ``lorentz_frac`` share of ``fwhm_total``) and
# inhomogeneous (Gaussian) broadening channels, normalized so the peak value is
# 1.0 at its center, leaving the population weights as the sole amplitudes.
#
# When zeeman_split → 0 the two groups coincide and the model collapses to a
# single population-normalized pseudo-Voigt triplet with full contrast c_total.
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _pv_factors(fwhm_total: float, lorentz_frac: float):
    """Precompute the height-normalized pseudo-Voigt factors for one particle.

    Returns ``(elf, egf, nhs, gamma2, has_gamma, has_sigma)`` where
    ``V_norm(dx) = elf/(dx²+gamma2) + egf·exp(nhs·dx²)`` has unit peak height.
    ``elf = eta·gamma/center_height`` and ``egf = (1-eta)·gauss_center/center_height``.
    """
    ratio = lorentz_frac if fwhm_total > 1e-30 else 0.0
    eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio

    gamma = fwhm_total / 2.0
    sigma = fwhm_total / (2.0 * _SQRT2LOG2)
    gamma2 = gamma * gamma

    has_gamma = abs(gamma) > 1e-12
    lorentz_center = 1.0 / gamma if has_gamma else 0.0

    has_sigma = abs(sigma) > 1e-12
    if has_sigma:
        inv_sigma = 1.0 / sigma
        gauss_center = inv_sigma / _SQRT2PI
        nhs = -0.5 * inv_sigma * inv_sigma
        egf0 = (1.0 - eta) * gauss_center
    else:
        gauss_center = 0.0
        nhs = 0.0
        egf0 = 0.0

    center_height = eta * lorentz_center + (1.0 - eta) * gauss_center
    inv_ch = 1.0 / center_height if abs(center_height) > 1e-12 else 0.0

    elf = eta * gamma * inv_ch if has_gamma else 0.0
    egf = egf0 * inv_ch
    return elf, egf, nhs, gamma2, has_gamma, has_sigma


@njit(cache=True, inline="always")
def _pv_norm(dx: float, elf: float, egf: float, nhs: float, gamma2: float, has_gamma: bool, has_sigma: bool) -> float:
    """Unit-height pseudo-Voigt profile value at offset ``dx`` from a peak center."""
    dx2 = dx * dx
    lo = elf / (dx2 + gamma2) if has_gamma else 0.0
    ga = egf * math.exp(dx2 * nhs) if has_sigma else 0.0
    return lo + ga


@njit(cache=True, inline="always")
def _zeeman_pv_populations(k_np: float, c_total: float):
    """Per-group population weights ``(p_l, p_0, p_r)`` summing to ``0.5·c_total``."""
    k = k_np if k_np > 1e-10 else 1e-10
    inv_p_sum = 1.0 / ((1.0 / k) + 1.0 + k)
    half_c = 0.5 * c_total
    p_0 = half_c * inv_p_sum
    p_l = half_c * (1.0 / k) * inv_p_sum
    p_r = half_c * k * inv_p_sum
    return p_l, p_0, p_r


@njit(cache=True, inline="always")
def _zeeman_pv_pred(
    x: float,
    freq: float,
    zeeman_split: float,
    hf_split: float,
    p_l: float,
    p_0: float,
    p_r: float,
    elf: float,
    egf: float,
    nhs: float,
    gamma2: float,
    has_gamma: bool,
    has_sigma: bool,
) -> float:
    """Two-group hyperfine pseudo-Voigt contrast sum (background not subtracted).

    The ms=+1/ms=-1 Zeeman groups are physically mirror images of each other
    about the true center: the hyperfine sub-line closest to center in one
    group corresponds to the sub-line closest to center in the other, not to
    the sub-line at the same absolute offset from its own group center. Using
    the same (p_l, p_0, p_r) left-to-right order in both groups (as if they
    were two independent, non-mirrored triplets) puts the asymmetric outer
    lines on the same side in both groups instead of facing each other, which
    doesn't match real NV ODMR spectra and their expected mirror symmetry.
    """
    cm = freq - zeeman_split
    left = (
        p_l * _pv_norm(x - (cm - hf_split), elf, egf, nhs, gamma2, has_gamma, has_sigma)
        + p_0 * _pv_norm(x - cm, elf, egf, nhs, gamma2, has_gamma, has_sigma)
        + p_r * _pv_norm(x - (cm + hf_split), elf, egf, nhs, gamma2, has_gamma, has_sigma)
    )
    cp = freq + zeeman_split
    # p_l/p_r swapped relative to the left group: the sub-line closest to the
    # true center (here, at cp - hf_split) mirrors the left group's
    # closest-to-center sub-line (at cm + hf_split, weighted p_r).
    right = (
        p_r * _pv_norm(x - (cp - hf_split), elf, egf, nhs, gamma2, has_gamma, has_sigma)
        + p_0 * _pv_norm(x - cp, elf, egf, nhs, gamma2, has_gamma, has_sigma)
        + p_l * _pv_norm(x - (cp + hf_split), elf, egf, nhs, gamma2, has_gamma, has_sigma)
    )
    return left + right


@njit(cache=True)
def nv_center_zeeman_pseudo_voigt_eval(
    x: float,
    freq: float,
    fwhm_total: float,
    lorentz_frac: float,
    zeeman_split: float,
    hf_split: float,
    k_np: float,
    c_total: float,
    background: float,
) -> float:
    """Zeeman + hyperfine NV pseudo-Voigt scalar kernel (population-normalized)."""
    elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total, lorentz_frac)
    p_l, p_0, p_r = _zeeman_pv_populations(k_np, c_total)
    return background - _zeeman_pv_pred(
        x, freq, zeeman_split, hf_split, p_l, p_0, p_r, elf, egf, nhs, gamma2, has_gamma, has_sigma
    )


@njit(cache=True)
def nv_center_zeeman_pseudo_voigt_vectorized_one_serial(
    x: float,
    freq: np.ndarray,
    fwhm_total: np.ndarray,
    lorentz_frac: np.ndarray,
    zeeman_split: np.ndarray,
    hf_split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zeeman pseudo-Voigt serial kernel: single probe ``x`` across many particles."""
    n = freq.shape[0]
    for j in range(n):
        elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total[j], lorentz_frac[j])
        p_l, p_0, p_r = _zeeman_pv_populations(k_np[j], c_total[j])
        out[j] = background[j] - _zeeman_pv_pred(
            x, freq[j], zeeman_split[j], hf_split[j], p_l, p_0, p_r, elf, egf, nhs, gamma2, has_gamma, has_sigma
        )


@njit(cache=True, parallel=True)
def nv_center_zeeman_pseudo_voigt_vectorized_many(
    xs: np.ndarray,
    freq: np.ndarray,
    fwhm_total: np.ndarray,
    lorentz_frac: np.ndarray,
    zeeman_split: np.ndarray,
    hf_split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Zeeman pseudo-Voigt parallel kernel: many probes × many particles → out[m, n]."""
    m = xs.shape[0]
    n = freq.shape[0]

    # Per-particle precompute: hoists the pseudo-Voigt + population setup out of
    # the m x n inner loop — same pattern as nv_center_zeeman_pseudo_voigt_eig_variance.
    elf_arr = np.empty(n, dtype=np.float64)
    egf_arr = np.empty(n, dtype=np.float64)
    nhs_arr = np.empty(n, dtype=np.float64)
    gamma2_arr = np.empty(n, dtype=np.float64)
    has_gamma_arr = np.empty(n, dtype=np.bool_)
    has_sigma_arr = np.empty(n, dtype=np.bool_)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total[j], lorentz_frac[j])
        elf_arr[j] = elf
        egf_arr[j] = egf
        nhs_arr[j] = nhs
        gamma2_arr[j] = gamma2
        has_gamma_arr[j] = has_gamma
        has_sigma_arr[j] = has_sigma
        p_l, p_0, p_r = _zeeman_pv_populations(k_np[j], c_total[j])
        p_l_arr[j] = p_l
        p_0_arr[j] = p_0
        p_r_arr[j] = p_r

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            out[i, j] = background[j] - _zeeman_pv_pred(
                x,
                freq[j],
                zeeman_split[j],
                hf_split[j],
                p_l_arr[j],
                p_0_arr[j],
                p_r_arr[j],
                elf_arr[j],
                egf_arr[j],
                nhs_arr[j],
                gamma2_arr[j],
                has_gamma_arr[j],
                has_sigma_arr[j],
            )


@njit(cache=True, parallel=True, fastmath=True)
def nv_center_zeeman_pseudo_voigt_vectorized_many_fast(
    xs: np.ndarray,
    freq: np.ndarray,
    fwhm_total: np.ndarray,
    lorentz_frac: np.ndarray,
    zeeman_split: np.ndarray,
    hf_split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fast (fastmath) Zeeman pseudo-Voigt kernel — acquisition / EIG path only."""
    m = xs.shape[0]
    n = freq.shape[0]

    # Per-particle precompute (see nv_center_zeeman_pseudo_voigt_vectorized_many).
    elf_arr = np.empty(n, dtype=np.float64)
    egf_arr = np.empty(n, dtype=np.float64)
    nhs_arr = np.empty(n, dtype=np.float64)
    gamma2_arr = np.empty(n, dtype=np.float64)
    has_gamma_arr = np.empty(n, dtype=np.bool_)
    has_sigma_arr = np.empty(n, dtype=np.bool_)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total[j], lorentz_frac[j])
        elf_arr[j] = elf
        egf_arr[j] = egf
        nhs_arr[j] = nhs
        gamma2_arr[j] = gamma2
        has_gamma_arr[j] = has_gamma
        has_sigma_arr[j] = has_sigma
        p_l, p_0, p_r = _zeeman_pv_populations(k_np[j], c_total[j])
        p_l_arr[j] = p_l
        p_0_arr[j] = p_0
        p_r_arr[j] = p_r

    for i in prange(m):
        x = xs[i]
        for j in range(n):
            out[i, j] = background[j] - _zeeman_pv_pred(
                x,
                freq[j],
                zeeman_split[j],
                hf_split[j],
                p_l_arr[j],
                p_0_arr[j],
                p_r_arr[j],
                elf_arr[j],
                egf_arr[j],
                nhs_arr[j],
                gamma2_arr[j],
                has_gamma_arr[j],
                has_sigma_arr[j],
            )


@njit(cache=True, parallel=True, fastmath=True)
def nv_center_zeeman_pseudo_voigt_eig_variance(
    xs: np.ndarray,
    freq: np.ndarray,
    fwhm_total: np.ndarray,
    lorentz_frac: np.ndarray,
    zeeman_split: np.ndarray,
    hf_split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fused: weighted prediction variance per candidate — Zeeman pseudo-Voigt EIG path.

    Same contract as :func:`nv_center_zeeman_lorentzian_eig_variance`: background
    is omitted (always 1.0 for NV models and cancels out of the variance).
    """
    m = xs.shape[0]
    n = freq.shape[0]

    # Per-particle precompute hoists the entire pseudo-Voigt + population setup
    # out of the m x n inner loop. float64 keeps the numerics identical to the
    # per-pair scalar path.
    elf_arr = np.empty(n, dtype=np.float64)
    egf_arr = np.empty(n, dtype=np.float64)
    nhs_arr = np.empty(n, dtype=np.float64)
    gamma2_arr = np.empty(n, dtype=np.float64)
    has_gamma_arr = np.empty(n, dtype=np.bool_)
    has_sigma_arr = np.empty(n, dtype=np.bool_)
    p_l_arr = np.empty(n, dtype=np.float64)
    p_0_arr = np.empty(n, dtype=np.float64)
    p_r_arr = np.empty(n, dtype=np.float64)
    for j in range(n):
        elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total[j], lorentz_frac[j])
        elf_arr[j] = elf
        egf_arr[j] = egf
        nhs_arr[j] = nhs
        gamma2_arr[j] = gamma2
        has_gamma_arr[j] = has_gamma
        has_sigma_arr[j] = has_sigma
        p_l, p_0, p_r = _zeeman_pv_populations(k_np[j], c_total[j])
        p_l_arr[j] = p_l
        p_0_arr[j] = p_0
        p_r_arr[j] = p_r

    for i in prange(m):
        x = xs[i]
        sum_p = 0.0
        sum_p2 = 0.0
        for j in range(n):
            pred = 1.0 - _zeeman_pv_pred(
                x,
                freq[j],
                zeeman_split[j],
                hf_split[j],
                p_l_arr[j],
                p_0_arr[j],
                p_r_arr[j],
                elf_arr[j],
                egf_arr[j],
                nhs_arr[j],
                gamma2_arr[j],
                has_gamma_arr[j],
                has_sigma_arr[j],
            )
            wi = weights[j]
            sum_p += wi * pred
            sum_p2 += wi * pred * pred

        v = sum_p2 - sum_p * sum_p
        out[i] = v if v > 0.0 else 0.0
