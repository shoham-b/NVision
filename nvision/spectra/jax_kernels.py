"""JAX kernels for signal models."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def nv_center_pseudo_voigt_jax(
    x: Any,
    frequency: Any,
    fwhm_total: Any,
    lorentz_frac: Any,
    split: Any,
    k_np: Any,
    dip_depth: Any,
    background: Any,
) -> Any:
    """Triple Voigt NV ODMR implementation in JAX, highly optimized for autodiff."""
    fwhm_l = lorentz_frac * fwhm_total
    fwhm_g = (1.0 - lorentz_frac) * fwhm_total

    # Pre-compute invariant parameters
    sigma = fwhm_g / jnp.float32(2.3548200450309493)  # 2 * sqrt(2 * ln(2))
    gamma = fwhm_l / jnp.float32(2.0)
    gamma2 = gamma * gamma

    ratio = fwhm_l / fwhm_total
    eta = jnp.float32(1.36603) * ratio - jnp.float32(0.47719) * ratio**2 + jnp.float32(0.11116) * ratio**3

    lorentz_peak = jnp.float32(1.0) / gamma
    sigma_sqrt_2pi = sigma * jnp.float32(2.5066282746310002)  # sigma * sqrt(2 * pi)
    gauss_peak = jnp.float32(1.0) / sigma_sqrt_2pi

    # Calculate unified profile peak to normalize to unit height
    inv_peak = jnp.float32(1.0) / (eta * lorentz_peak + (jnp.float32(1.0) - eta) * gauss_peak)

    # Pre-multiply mixture factors with the normalization inverse
    eta_lorentz_factor = eta * gamma * inv_peak
    eta_gauss_factor = (jnp.float32(1.0) - eta) / sigma_sqrt_2pi * inv_peak
    neg_half_inv_sigma2 = jnp.float32(-0.5) / (sigma * sigma)

    actual_depth = dip_depth / k_np
    amp_c = actual_depth
    amp_l = actual_depth / k_np
    amp_r = actual_depth * k_np

    # Profile component evaluation
    dx_c = x - frequency
    dx_c2 = dx_c * dx_c
    pc = (eta_lorentz_factor / (dx_c2 + gamma2)) + eta_gauss_factor * jnp.exp(dx_c2 * neg_half_inv_sigma2)

    dx_l = dx_c + split
    dx_l2 = dx_l * dx_l
    pl = (eta_lorentz_factor / (dx_l2 + gamma2)) + eta_gauss_factor * jnp.exp(dx_l2 * neg_half_inv_sigma2)

    dx_r = dx_c - split
    dx_r2 = dx_r * dx_r
    pr = (eta_lorentz_factor / (dx_r2 + gamma2)) + eta_gauss_factor * jnp.exp(dx_r2 * neg_half_inv_sigma2)

    return background - (amp_l * pl + amp_c * pc + amp_r * pr)


def nv_center_lorentzian_jax(
    x: Any,
    frequency: Any,
    linewidth: Any,
    split: Any,
    k_np: Any,
    dip_depth: Any,
    background: Any,
) -> Any:
    """Triple Lorentzian NV ODMR implementation in JAX, highly optimized for autodiff."""
    gamma2 = (linewidth / jnp.float32(2.0))**2

    actual_depth = dip_depth / k_np
    amp_c = actual_depth
    amp_l = actual_depth / k_np
    amp_r = actual_depth * k_np

    dx_c = x - frequency
    pc = gamma2 / (dx_c**2 + gamma2)

    dx_l = dx_c + split
    pl = gamma2 / (dx_l**2 + gamma2)

    dx_r = dx_c - split
    pr = gamma2 / (dx_r**2 + gamma2)

    return background - (amp_l * pl + amp_c * pc + amp_r * pr)
