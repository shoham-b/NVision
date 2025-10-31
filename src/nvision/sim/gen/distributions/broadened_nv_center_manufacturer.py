from __future__ import annotations

import math
import random
from collections.abc import Callable

from nvision.sim.gen._protocols import PeakManufacturer


def peak_normalized_pseudo_voigt(
    x: float, center: float, amplitude: float, fwhm_g: float, fwhm_l: float
) -> float:
    """
    Computes a peak-normalized pseudo-Voigt profile.

    Args:
        x: The independent variable.
        center: The center of the peak.
        amplitude: The amplitude of the peak.
        fwhm_g: The Full Width at Half Maximum of the Gaussian component.
        fwhm_l: The Full Width at Half Maximum of the Lorentzian component.

    Returns:
        The value of the pseudo-Voigt profile at x.
    """
    sigma = fwhm_g / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gamma = fwhm_l / 2.0

    # Total FWHM approximation
    fwhm_total = (
        fwhm_g**5
        + 2.69269 * fwhm_g**4 * fwhm_l
        + 2.42843 * fwhm_g**3 * fwhm_l**2
        + 4.47163 * fwhm_g**2 * fwhm_l**3
        + 0.07842 * fwhm_g * fwhm_l**4
        + fwhm_l**5
    ) ** 0.2

    # Mixing parameter eta
    eta = (
        1.36603 * (fwhm_l / fwhm_total)
        - 0.47719 * (fwhm_l / fwhm_total) ** 2
        + 0.11116 * (fwhm_l / fwhm_total) ** 3
    )

    # Peak-normalized Lorentzian and Gaussian components
    lorentzian_part = gamma**2 / ((x - center) ** 2 + gamma**2)
    gaussian_part = math.exp(-((x - center) ** 2) / (2 * sigma**2))

    # Weighted sum for pseudo-Voigt
    return amplitude * (eta * lorentzian_part + (1.0 - eta) * gaussian_part)


class BroadenedNVCenterManufacturer(PeakManufacturer):
    """
    A manufacturer for a signal that models a Gaussian-broadened NV-center spectrum.

    This is the result of convolving the ideal triple-Lorentzian NV center signal
    with a Gaussian. The resulting dips have a Voigt profile, which is approximated
    here using a pseudo-Voigt function.
    """

    def __init__(
        self,
        a: float,
        k_np: float,
        delta_f_hf: float,
        fwhm_l: float,
        fwhm_g: float,
    ):
        self.a = a
        self.k_np = k_np
        self.delta_f_hf = delta_f_hf
        self.fwhm_l = fwhm_l
        self.fwhm_g = fwhm_g

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        f_b = center

        def f(x: float) -> float:
            # The total signal is 1 minus the sum of the three Voigt-profile dips.
            # We use a peak-normalized pseudo-Voigt, so the amplitudes from the
            # NV model can be used directly.

            dip1 = peak_normalized_pseudo_voigt(
                x, f_b + self.delta_f_hf, self.a * self.k_np, self.fwhm_g, self.fwhm_l
            )
            dip2 = peak_normalized_pseudo_voigt(x, f_b, self.a, self.fwhm_g, self.fwhm_l)
            dip3 = peak_normalized_pseudo_voigt(
                x, f_b - self.delta_f_hf, self.a / self.k_np, self.fwhm_g, self.fwhm_l
            )

            total_dip = dip1 + dip2 + dip3

            return 1.0 - total_dip

        params = {
            "f_B": f_b,
            "mode": "nv_center_broadened",
        }
        return f, params
