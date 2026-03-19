"""Voigt-broadened NV center model with Zeeman splitting."""

from __future__ import annotations

import random

from nvision.signal.signal import Parameter, SignalModel


class VoigtZeemanModel(SignalModel):
    """Voigt-broadened NV center model with Zeeman splitting.

    Models an NV center with three Lorentzian dips (hyperfine splitting)
    with optional Gaussian broadening (Voigt profile).

    Signal form (simplified Lorentzian version):
        f(x) = background - (
            (amplitude / k_np) / ((x - (frequency - split))^2 + linewidth^2) +
            amplitude / ((x - frequency)^2 + linewidth^2) +
            (amplitude * k_np) / ((x - (frequency + split))^2 + linewidth^2)
        )

    The worst-case total dip depth is (amplitude/linewidth^2) * (1/k_np + 1 + k_np).
    Use ``sample_params`` to get parameters that keep the signal in [0, 1].

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
        p = self._params_to_dict(params)
        freq = p["frequency"]
        linewidth = p["linewidth"]
        split = p["split"]
        k_np = p["k_np"]
        amplitude = p["amplitude"]
        background = p["background"]

        left_denom = (x - (freq - split)) ** 2 + linewidth**2
        left_peak = (amplitude / k_np) / left_denom

        center_denom = (x - freq) ** 2 + linewidth**2
        center_peak = amplitude / center_denom

        right_denom = (x - (freq + split)) ** 2 + linewidth**2
        right_peak = (amplitude * k_np) / right_denom

        return background - (left_peak + center_peak + right_peak)

    def parameter_names(self) -> list[str]:
        return ["frequency", "linewidth", "split", "k_np", "amplitude", "background"]

    def sample_params(self, rng: random.Random) -> list[Parameter]:
        """Sample parameters that keep the signal within [0, 1].

        Constrains amplitude so that even the worst-case total dip
        (all three peaks overlapping) stays within the background level.
        """
        linewidth = rng.uniform(0.03, 0.08)
        split = rng.uniform(0.05, 0.12)
        k_np = rng.uniform(2.0, 4.0)
        frequency = rng.uniform(split + 0.1, 1.0 - split - 0.1)
        background = 1.0
        total_factor = 1.0 / k_np + 1.0 + k_np
        max_depth = rng.uniform(0.3, 0.75)
        amplitude = max_depth * linewidth**2 / total_factor
        return [
            Parameter(name="frequency", bounds=(0.0, 1.0), value=frequency),
            Parameter(name="linewidth", bounds=(0.001, 0.3), value=linewidth),
            Parameter(name="split", bounds=(0.0, 0.3), value=split),
            Parameter(name="k_np", bounds=(1.0, 6.0), value=k_np),
            Parameter(name="amplitude", bounds=(0.0, 0.01), value=amplitude),
            Parameter(name="background", bounds=(0.5, 1.5), value=background),
        ]
