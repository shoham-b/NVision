"""Shared scalar kernels for signal models.

Lorentzian pieces are compiled once with Numba and reused by :class:`~nvision.signal.lorentzian.LorentzianModel`
and :class:`~nvision.signal.nv_center.NVCenterLorentzianModel`.

The single-Gaussian peak uses :func:`math.exp` only (no Numba): for one scalar per call, the stdlib C
implementation is simpler and avoids an extra compilation unit versus a trivial ``@njit`` wrapper.
"""

from __future__ import annotations

import math

from numba import njit


@njit(cache=True)
def lorentzian_dip_term(x: float, center: float, linewidth: float, peak_amplitude: float) -> float:
    """``peak_amplitude / ((x - center)² + linewidth²)`` — one Lorentzian dip contribution."""
    d = (x - center) * (x - center) + linewidth * linewidth
    return peak_amplitude / d


@njit(cache=True)
def lorentzian_peak_value(
    x: float,
    freq: float,
    linewidth: float,
    amplitude: float,
    background: float,
) -> float:
    """Single dip: ``background - amplitude / ((x - freq)² + linewidth²)``."""
    return background - lorentzian_dip_term(x, freq, linewidth, amplitude)


@njit(cache=True)
def nv_center_lorentzian_eval(
    x: float,
    freq: float,
    linewidth: float,
    split: float,
    k_np: float,
    amplitude: float,
    background: float,
) -> float:
    """NV triple-Lorentzian ODMR contrast (same formula as the original Python reference)."""
    if split < 1e-10:
        combined_amplitude = amplitude * (k_np + 1.0 + 1.0 / k_np)
        return background - lorentzian_dip_term(x, freq, linewidth, combined_amplitude)
    left = lorentzian_dip_term(x, freq - split, linewidth, amplitude / k_np)
    center = lorentzian_dip_term(x, freq, linewidth, amplitude)
    right = lorentzian_dip_term(x, freq + split, linewidth, amplitude * k_np)
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
