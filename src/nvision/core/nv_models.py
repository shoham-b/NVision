"""NV center signal models based on physics.

These models implement the actual ODMR signal equations for NV centers in diamond.
Based on the physics from sim/gen/distributions/nv_center_manufacturer.py.
"""

from __future__ import annotations

import numpy as np

from nvision.core.signal import SignalModel

# Physical constants
A_PARAM = 0.0003
MIN_K_NP = 2.0
MAX_K_NP = 4.0
MIN_NV_CENTER_DELTA = 0.0
MAX_NV_CENTER_DELTA = 0.4
MIN_NV_CENTER_OMEGA = 0.008
MAX_NV_CENTER_OMEGA = 0.01


class NVCenterLorentzianModel(SignalModel):
    """NV center ODMR signal model with three Lorentzian dips.

    Models the optically detected magnetic resonance (ODMR) spectrum of an
    NV center in diamond. The signal has three Lorentzian dips from a baseline
    of 1.0, corresponding to the ms=±1 and ms=0 spin states with hyperfine splitting.

    Signal form:
        S(f) = 1 - L_left - L_center - L_right

    Where each Lorentzian dip is:
        L(f, f_0, A, ω) = A / ((f - f_0)^2 + ω^2)

    Parameters
    ----------
    frequency : float
        Central frequency f_B (center of main dip) in Hz
    linewidth : float
        Lorentzian linewidth ω (HWHM) in Hz
    split : float
        Hyperfine splitting Δf_HF in Hz (distance from center to outer peaks)
    k_np : float
        Non-polarization factor (amplitude ratio between peaks)
        Left peak amplitude: a/k_np, Center: a, Right: a*k_np
    amplitude : float
        Amplitude scaling factor 'a' for the dips
    background : float
        Background level (typically 1.0 for normalized signal)
    """

    def compute(self, x: float, params: list) -> float:
        """Compute NV center signal at frequency x.

        Parameters
        ----------
        x : float
            Frequency in Hz
        params : list[Parameter]
            Must contain: frequency, linewidth, split, k_np, amplitude, background

        Returns
        -------
        float
            Normalized signal value (typically 0.0-1.0)
        """
        p = self._params_to_dict(params)

        freq = p["frequency"]
        linewidth = p["linewidth"]
        split = p["split"]
        k_np = p["k_np"]
        amplitude = p["amplitude"]
        background = p["background"]

        # Handle zero-split case (single peak)
        if split < 1e-10:
            # Three peaks collapse to one with combined amplitude
            combined_amplitude = amplitude * (k_np + 1 + 1 / k_np)
            denom = (x - freq) ** 2 + linewidth**2
            return background - combined_amplitude / denom

        # Three separate Lorentzian dips
        # Left peak at freq - split with amplitude a/k_np
        left_denom = (x - (freq - split)) ** 2 + linewidth**2
        left_dip = (amplitude / k_np) / left_denom

        # Center peak at freq with amplitude a
        center_denom = (x - freq) ** 2 + linewidth**2
        center_dip = amplitude / center_denom

        # Right peak at freq + split with amplitude a*k_np
        right_denom = (x - (freq + split)) ** 2 + linewidth**2
        right_dip = (amplitude * k_np) / right_denom

        # Background minus sum of dips
        return background - (left_dip + center_dip + right_dip)

    def parameter_names(self) -> list[str]:
        """Return ordered parameter names."""
        return ["frequency", "linewidth", "split", "k_np", "amplitude", "background"]


class NVCenterVoigtModel(SignalModel):
    """NV center with Gaussian broadening (Voigt profile).

    Models an NV center where each Lorentzian dip is convolved with a Gaussian,
    resulting in a Voigt profile. This accounts for inhomogeneous broadening
    due to strain, temperature variations, etc.

    Parameters
    ----------
    frequency : float
        Central frequency f_B in Hz
    fwhm_lorentz : float
        Lorentzian FWHM (full width at half maximum) in Hz
    fwhm_gauss : float
        Gaussian FWHM in Hz
    split : float
        Hyperfine splitting in Hz
    k_np : float
        Non-polarization factor
    amplitude : float
        Amplitude scaling factor
    background : float
        Background level
    """

    def __init__(self):
        """Initialize Voigt model."""
        # Import Voigt function if needed
        try:
            from scipy.special import wofz

            self.wofz = wofz
        except ImportError:
            # Fall back to Lorentzian approximation if scipy not available
            self.wofz = None

    def _voigt_profile(self, x: float, center: float, fwhm_l: float, fwhm_g: float) -> float:
        """Compute Voigt profile at x.

        Uses Faddeeva function for exact computation if scipy available,
        otherwise falls back to pseudo-Voigt approximation.
        """
        if self.wofz is not None:
            # Exact Voigt via Faddeeva function
            sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
            gamma = fwhm_l / 2
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            w = self.wofz(z)
            return w.real / (sigma * np.sqrt(2 * np.pi))
        else:
            # Pseudo-Voigt approximation
            # V(x) ≈ η*L(x) + (1-η)*G(x)
            eta = (
                1.36603 * (fwhm_l / (fwhm_l + fwhm_g))
                - 0.47719 * (fwhm_l / (fwhm_l + fwhm_g)) ** 2
                + 0.11116 * (fwhm_l / (fwhm_l + fwhm_g)) ** 3
            )

            # Lorentzian component
            gamma = fwhm_l / 2
            lorentz = gamma / ((x - center) ** 2 + gamma**2)

            # Gaussian component
            sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
            gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

            return eta * lorentz + (1 - eta) * gauss

    def compute(self, x: float, params: list) -> float:
        """Compute Voigt-broadened NV center signal.

        Parameters
        ----------
        x : float
            Frequency in Hz
        params : list[Parameter]
            Must contain: frequency, fwhm_lorentz, fwhm_gauss, split, k_np, amplitude, background

        Returns
        -------
        float
            Signal value
        """
        p = self._params_to_dict(params)

        freq = p["frequency"]
        fwhm_l = p["fwhm_lorentz"]
        fwhm_g = p["fwhm_gauss"]
        split = p["split"]
        k_np = p["k_np"]
        amplitude = p["amplitude"]
        background = p["background"]

        # Three Voigt dips
        if split < 1e-10:
            # Single peak with combined amplitude
            combined_amplitude = amplitude * (k_np + 1 + 1 / k_np)
            return background - combined_amplitude * self._voigt_profile(x, freq, fwhm_l, fwhm_g)

        # Left, center, right peaks
        left_dip = (amplitude / k_np) * self._voigt_profile(x, freq - split, fwhm_l, fwhm_g)
        center_dip = amplitude * self._voigt_profile(x, freq, fwhm_l, fwhm_g)
        right_dip = (amplitude * k_np) * self._voigt_profile(x, freq + split, fwhm_l, fwhm_g)

        return background - (left_dip + center_dip + right_dip)

    def parameter_names(self) -> list[str]:
        """Return ordered parameter names."""
        return ["frequency", "fwhm_lorentz", "fwhm_gauss", "split", "k_np", "amplitude", "background"]
