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

    @staticmethod
    def eval_exponential_decay_model(x: float, decay_rate: float, amplitude: float, background: float) -> float:
        """Evaluate decay; parameter order matches :meth:`parameter_names`."""
        return background + amplitude * np.exp(-x / max(decay_rate, 1e-12))

    def compute(self, x: float, params: list) -> float:
        v = self._param_floats_canonical(params)
        return self.eval_exponential_decay_model(float(x), *v)

    def parameter_names(self) -> list[str]:
        return ["decay_rate", "amplitude", "background"]
