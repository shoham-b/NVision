"""Concrete signal model implementations."""

from __future__ import annotations

import numpy as np

from nvision.core.signal import Parameter, SignalModel


class LorentzianModel(SignalModel):
    """Single Lorentzian peak model.

    Signal form:
        f(x) = background - amplitude / ((x - frequency)^2 + linewidth^2)

    Parameters
    ----------
    frequency : float
        Peak center frequency
    linewidth : float
        Half-width at half-maximum (HWHM)
    amplitude : float
        Peak amplitude
    background : float
        Background level
    """

    def compute(self, x: float, params: list) -> float:
        """Compute Lorentzian signal at position x.

        Parameters
        ----------
        x : float
            Position to evaluate
        params : list[Parameter]
            Must contain: frequency, linewidth, amplitude, background

        Returns
        -------
        float
            Signal value at x
        """
        p = self._params_to_dict(params)

        freq = p["frequency"]
        linewidth = p["linewidth"]
        amplitude = p["amplitude"]
        background = p["background"]

        denominator = (x - freq) ** 2 + linewidth**2
        return background - amplitude / denominator

    def parameter_names(self) -> list[str]:
        """Return ordered parameter names."""
        return ["frequency", "linewidth", "amplitude", "background"]


class VoigtZeemanModel(SignalModel):
    """Voigt-broadened NV center model with Zeeman splitting.

    Models an NV center with three Lorentzian dips (hyperfine splitting)
    with optional Gaussian broadening (Voigt profile).

    Signal form (simplified Lorentzian version):
        f(x) = background - (
            (amplitude * k_np) / ((x - (frequency + split))^2 + linewidth^2) +
            amplitude / ((x - frequency)^2 + linewidth^2) +
            (amplitude / k_np) / ((x - (frequency - split))^2 + linewidth^2)
        )

    Parameters
    ----------
    frequency : float
        Central frequency (f_B)
    linewidth : float
        Lorentzian linewidth (omega, HWHM)
    split : float
        Hyperfine splitting (delta_f_HF)
    k_np : float
        Non-polarization factor (amplitude ratio between peaks)
    amplitude : float
        Peak amplitude scaling factor
    background : float
        Background level
    """

    def compute(self, x: float, params: list) -> float:
        """Compute Voigt-Zeeman signal at position x.

        Parameters
        ----------
        x : float
            Position to evaluate
        params : list[Parameter]
            Must contain: frequency, linewidth, split, k_np, amplitude, background

        Returns
        -------
        float
            Signal value at x
        """
        p = self._params_to_dict(params)

        freq = p["frequency"]
        linewidth = p["linewidth"]
        split = p["split"]
        k_np = p["k_np"]
        amplitude = p["amplitude"]
        background = p["background"]

        # Three Lorentzian dips with different amplitudes
        # Left peak (lower frequency): amplitude / k_np
        left_denom = (x - (freq - split)) ** 2 + linewidth**2
        left_peak = (amplitude / k_np) / left_denom

        # Center peak: amplitude
        center_denom = (x - freq) ** 2 + linewidth**2
        center_peak = amplitude / center_denom

        # Right peak (higher frequency): amplitude * k_np
        right_denom = (x - (freq + split)) ** 2 + linewidth**2
        right_peak = (amplitude * k_np) / right_denom

        # Return background minus sum of dips
        return background - (left_peak + center_peak + right_peak)

    def parameter_names(self) -> list[str]:
        """Return ordered parameter names."""
        return ["frequency", "linewidth", "split", "k_np", "amplitude", "background"]


class GaussianModel(SignalModel):
    """Single Gaussian peak model.

    Signal form:
        f(x) = background + amplitude * exp(-0.5 * ((x - frequency) / sigma)^2)

    Parameters
    ----------
    frequency : float
        Peak center
    sigma : float
        Standard deviation
    amplitude : float
        Peak amplitude
    background : float
        Background level
    """

    def compute(self, x: float, params: list) -> float:
        """Compute Gaussian signal at position x.

        Parameters
        ----------
        x : float
            Position to evaluate
        params : list[Parameter]
            Must contain: frequency, sigma, amplitude, background

        Returns
        -------
        float
            Signal value at x
        """
        p = self._params_to_dict(params)

        freq = p["frequency"]
        sigma = p["sigma"]
        amplitude = p["amplitude"]
        background = p["background"]

        z = (x - freq) / sigma
        return background + amplitude * np.exp(-0.5 * z**2)

    def parameter_names(self) -> list[str]:
        """Return ordered parameter names."""
        return ["frequency", "sigma", "amplitude", "background"]


class ExponentialDecayModel(SignalModel):
    """Exponential decay model.

    Signal form:
        f(x) = background + amplitude * exp(-x / decay_rate)

    Parameters
    ----------
    decay_rate : float
        Decay rate constant
    amplitude : float
        Peak amplitude
    background : float
        Background level
    """

    def compute(self, x: float, params: list) -> float:
        """Compute exponential decay at position x.

        Parameters
        ----------
        x : float
            Position to evaluate
        params : list[Parameter]
            Must contain: decay_rate, amplitude, background

        Returns
        -------
        float
            Signal value at x
        """
        p = self._params_to_dict(params)

        decay_rate = p["decay_rate"]
        amplitude = p["amplitude"]
        background = p["background"]

        return background + amplitude * np.exp(-x / max(decay_rate, 1e-12))

    def parameter_names(self) -> list[str]:
        """Return ordered parameter names."""
        return ["decay_rate", "amplitude", "background"]


class CompositePeakModel(SignalModel):
    """Multiple independent peaks summed together.

    Combines multiple peak models into a composite signal.
    Parameters are organized as: [peak1_params..., peak2_params..., ...]
    with prefixes like "peak1_", "peak2_" to distinguish them.
    """

    def __init__(self, peak_models: list[tuple[str, SignalModel]]):
        """Initialize composite model.

        Parameters
        ----------
        peak_models : list[tuple[str, SignalModel]]
            List of (prefix, model) pairs. Prefix is used to namespace parameters.
            Example: [("peak1", LorentzianModel()), ("peak2", GaussianModel())]
        """
        self.peak_models = peak_models

    def compute(self, x: float, params: list[Parameter]) -> float:
        """Compute composite signal by summing all peaks.

        Parameters
        ----------
        x : float
            Position to evaluate
        params : list[Parameter]
            All parameters for all peaks, with prefixed names

        Returns
        -------
        float
            Sum of all peak contributions
        """
        total = 0.0

        for prefix, model in self.peak_models:
            # Extract parameters for this peak
            peak_params = [p for p in params if p.name.startswith(f"{prefix}_")]

            # Remove prefix for model evaluation
            unprefixed_params = [
                Parameter(
                    name=p.name[len(prefix) + 1 :],  # Remove "peakN_" prefix
                    bounds=p.bounds,
                    value=p.value,
                )
                for p in peak_params
            ]

            # Add this peak's contribution
            total += model.compute(x, unprefixed_params)

        return total

    def parameter_names(self) -> list[str]:
        """Return flattened parameter names with prefixes.

        Returns
        -------
        list[str]
            ["peak1_frequency", "peak1_amplitude", ...,
             "peak2_frequency", "peak2_amplitude", ...]
        """
        names = []
        for prefix, model in self.peak_models:
            for param_name in model.parameter_names():
                names.append(f"{prefix}_{param_name}")
        return names
