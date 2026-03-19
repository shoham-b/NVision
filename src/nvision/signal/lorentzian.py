"""Single Lorentzian peak model."""

from __future__ import annotations

import random

from nvision.signal.signal import Parameter, SignalModel


class LorentzianModel(SignalModel):
    """Single Lorentzian peak model.

    Signal form:
        f(x) = background - amplitude / ((x - frequency)^2 + linewidth^2)

    The amplitude parameter has units of [signal * frequency^2] so that
    dip depth = amplitude / linewidth².  Use ``sample_params`` to get a set
    of parameters that keeps the signal in [0, 1].

    Parameters
    ----------
    frequency : float
        Peak center, in [0, 1] normalized units
    linewidth : float
        Half-width at half-maximum (HWHM)
    amplitude : float
        = dip_depth * linewidth^2 (not dip depth directly)
    background : float
        Baseline level (max signal)
    """

    def compute(self, x: float, params: list) -> float:
        p = self._params_to_dict(params)
        freq = p["frequency"]
        linewidth = p["linewidth"]
        amplitude = p["amplitude"]
        background = p["background"]
        denominator = (x - freq) ** 2 + linewidth**2
        return background - amplitude / denominator

    def parameter_names(self) -> list[str]:
        return ["frequency", "linewidth", "amplitude", "background"]

    def sample_params(self, rng: random.Random) -> list[Parameter]:
        """Sample parameters that keep the signal within [0, 1]."""
        frequency = rng.uniform(0.1, 0.9)
        linewidth = rng.uniform(0.03, 0.12)
        depth = rng.uniform(0.3, 0.85)
        amplitude = depth * linewidth**2
        background = 1.0
        return [
            Parameter(name="frequency", bounds=(0.0, 1.0), value=frequency),
            Parameter(name="linewidth", bounds=(0.001, 0.5), value=linewidth),
            Parameter(name="amplitude", bounds=(0.0, 0.05), value=amplitude),
            Parameter(name="background", bounds=(0.5, 1.5), value=background),
        ]
