"""Exponential decay model."""

from __future__ import annotations

import numpy as np

from nvision.signal.signal import SignalModel


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
        p = self._params_to_dict(params)
        decay_rate = p["decay_rate"]
        amplitude = p["amplitude"]
        background = p["background"]
        return background + amplitude * np.exp(-x / max(decay_rate, 1e-12))

    def parameter_names(self) -> list[str]:
        return ["decay_rate", "amplitude", "background"]
