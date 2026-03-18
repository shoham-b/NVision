"""Single Gaussian peak model."""

from __future__ import annotations

import numpy as np

from nvision.signal.signal import SignalModel


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
        p = self._params_to_dict(params)
        freq = p["frequency"]
        sigma = p["sigma"]
        amplitude = p["amplitude"]
        background = p["background"]
        z = (x - freq) / sigma
        return background + amplitude * np.exp(-0.5 * z**2)

    def parameter_names(self) -> list[str]:
        return ["frequency", "sigma", "amplitude", "background"]
