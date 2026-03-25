"""Single Gaussian peak model."""

from __future__ import annotations

from nvision.signal.numba_kernels import gaussian_peak_value
from nvision.signal.signal import SignalModel


class GaussianModel(SignalModel):
    """Single Gaussian peak model.

    Prefer :meth:`eval_gaussian_model` when arguments are already floats.

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

    @staticmethod
    def eval_gaussian_model(
        x: float,
        frequency: float,
        sigma: float,
        amplitude: float,
        background: float,
    ) -> float:
        """Evaluate Gaussian peak; parameter order matches :meth:`parameter_names`."""
        return gaussian_peak_value(float(x), frequency, sigma, amplitude, background)

    def compute(self, x: float, params: list) -> float:
        v = self._param_floats_canonical(params)
        return self.eval_gaussian_model(float(x), *v)

    def parameter_names(self) -> list[str]:
        return ["frequency", "sigma", "amplitude", "background"]
