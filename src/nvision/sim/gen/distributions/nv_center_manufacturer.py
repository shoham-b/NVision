from __future__ import annotations

import math
import random
from collections.abc import Callable

from nvision.sim.gen._protocols import PeakManufacturer
from nvision.sim.gen.distributions.broadened_nv_center_manufacturer import (
    BroadenedNVCenterManufacturer,
)
from nvision.sim.gen.distributions.gaussian_manufacturer import GaussianManufacturer


class NVCenterManufacturer(PeakManufacturer):
    """A manufacturer for a signal distribution that models an NV-center spectrum.

    This class implements the `PeakManufacturer` protocol to generate a signal
    function that simulates the characteristic optically detected magnetic resonance (ODMR)
    spectrum of a Nitrogen-Vacancy (NV) center in diamond.

    This distribution is characterized by three Lorentzian-shaped dips from a baseline of 1.
    The equation for the distribution mu(f) is:

        mu(f) = 1
                - (a * k_NP) / ((f - (f_B + delta_f_HF))^2 + omega^2)
                - a / ((f - f_B)^2 + omega^2)
                - (a / k_NP) / ((f - (f_B - delta_f_HF))^2 + omega^2)

    Where:
        f: The independent variable (e.g., frequency).
        f_B: The central frequency of the main dip.
        a: An amplitude scaling factor for the dips.
        k_NP: A non-polarization factor.
        delta_f_HF: The hyperfine splitting frequency.
        omega: The linewidth/width parameter (HWHM).

    The `build_peak` method uses the `center` argument as `f_B` and ignores the `base`
    argument, as the baseline is fixed at 1.0 by the model's equation.
    """

    def __init__(
        self,
        a: float = 1,  # Amplitude scaling factor for the central dip
        k_np: float = 100,  # Non-polarization factor, typically > 0
        delta_f_hf: float = 0.2,  # Hyperfine splitting, typically > 0
        omega: float | None = 1,  # Linewidth/width parameter (HWHM), typically > 0
        **kwargs,  # For potential future compatibility or alternative parameter names
    ) -> None:
        # Store parameters
        self.a = a
        self.k_np = k_np
        self.delta_f_hf = delta_f_hf
        self.omega = omega

        # Basic validation for parameters
        if self.a <= 0:
            raise ValueError("Parameter 'a' (amplitude scaling) must be positive.")
        if self.k_np <= 0:
            raise ValueError("Parameter 'k_NP' (non-polarization factor) must be positive.")
        if self.delta_f_hf < 0:  # Can be zero if no splitting
            raise ValueError("Parameter 'delta_f_HF' (hyperfine splitting) cannot be negative.")
        if self.omega is not None and self.omega <= 0:
            raise ValueError("Parameter 'omega' (linewidth) must be positive if specified.")

    def build_peak(
        self,
        center: float,  # This corresponds to f_B in the equation
        base: float,  # The baseline for the overall distribution. For this model, the equation
        # inherently defines a baseline of 1.0. The 'base' parameter from the
        # protocol is effectively ignored in the calculation of the NV center shape,
        # as the equation itself is the final function.
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        """
        Builds the NV-center-like peak distribution function.

        Args:
            center: The central frequency (f_B) for the main dip.
            base: The baseline value. For this specific NV-center model, the equation
                  defines its own baseline of 1.0, so this parameter is not used
                  in the calculation of the peak shape itself.
            x_min: Minimum x-value for the domain.
            x_max: Maximum x-value for the domain.
            rng: Random number generator (not used in this deterministic peak shape).

        Returns:
            A tuple containing:
            - A callable function f(x) that computes the distribution value at x.
            - A dictionary of parameters describing the generated peak.
        """
        # Determine the effective omega (linewidth).
        # If self.omega is not provided, use a heuristic based on the range.
        # This heuristic is similar to the CauchyLorentzPeakManufacturer.
        effective_omega = self.omega
        if effective_omega is None:
            # Ensure a positive width, even if x_max - x_min is very small or zero.
            effective_omega = max(0.05 * (x_max - x_min), 1e-6)
            if effective_omega <= 0:  # Fallback for extreme cases
                effective_omega = 1e-6  # A very small positive default

        # The 'center' argument maps to 'f_B' in the equation.
        f_b = center

        # Define the Lorentzian dip term helper function
        # L(f, f0, A_num, Omega) = A_num / ((f - f0)^2 + Omega^2)
        def lorentzian_peak_term(
            f_val: float,
            f0: float,
            amplitude_num: float,
            omega_width: float,
        ) -> float:
            # Denominator can be zero if f_val == f0 and Omega == 0, but Omega is enforced > 0.
            # If Omega is very small, the term can become very large.
            # This is standard for Lorentzians.
            return amplitude_num / ((f_val - f0) ** 2 + omega_width**2)

        def f(x: float) -> float:
            """
            Computes the NV-center distribution value at a given frequency 'x'.
            """
            if self.delta_f_hf == 0:
                # If there is no hyperfine splitting, the three terms collapse into one.
                # We can optimize by calculating a single Lorentzian with a combined amplitude.
                combined_amplitude_num = self.a * (self.k_np + 1 + 1 / self.k_np)
                total_dip = lorentzian_peak_term(x, f_b, combined_amplitude_num, effective_omega)
                return total_dip
            else:
                # Calculate the three separate Lorentzian dip terms for the hyperfine splitting
                term1 = lorentzian_peak_term(
                    x,
                    f_b + self.delta_f_hf,
                    self.a * self.k_np,
                    effective_omega,
                )
                term2 = lorentzian_peak_term(x, f_b, self.a, effective_omega)
                term3 = lorentzian_peak_term(
                    x,
                    f_b - self.delta_f_hf,
                    self.a / self.k_np,
                    effective_omega,
                )
                return term1 + term2 + term3

        # Collect all parameters used to define this specific peak instance
        params = {
            "a": self.a,
            "k_NP": self.k_np,
            "delta_f_HF": self.delta_f_hf,
            "omega": effective_omega,
            "f_B": f_b,
            "mode": "nv_center",  # Identifier for this peak type
        }

        return f, params

    def convolve(self, other: PeakManufacturer) -> PeakManufacturer:
        """Handles convolution, specifically with a Gaussian for broadening."""
        if not isinstance(other, GaussianManufacturer):
            return super().convolve(other)

        # The convolution of the NV center's triple-Lorentzian signal with a
        # Gaussian results in a triple-Voigt signal. We return a new
        # manufacturer that produces this broadened signal directly.

        # FWHM of the Lorentzian component is 2 * omega (HWHM)
        fwhm_l = 2 * self.omega if self.omega is not None else 0.1

        # FWHM of the Gaussian component is 2 * sigma * sqrt(2 * ln(2))
        # other.sigma is sigma.
        fwhm_g = 2 * other.sigma * math.sqrt(2 * math.log(2)) if other.sigma is not None else 0.1

        return BroadenedNVCenterManufacturer(
            a=self.a,
            k_np=self.k_np,
            delta_f_hf=self.delta_f_hf,
            fwhm_l=fwhm_l,
            fwhm_g=fwhm_g,
        )
