"""Map unit-interval parameters and probe position to physical signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from nvision.signal.signal import Parameter, SignalModel


@dataclass
class UnitCubeSignalModel:
    """Wrap a physical-domain :class:`SignalModel` for inference on ``[0, 1]``.

    * ``x_unit`` — normalized probe coordinate in ``[0, 1]`` (same convention as
      :meth:`~nvision.models.experiment.CoreExperiment.measure`).
    * Each parameter value in ``params`` is interpreted as a fraction in ``[0, 1]``
      over the corresponding physical interval in ``param_bounds_phys``.

    The inner model is evaluated at physical frequency and physical parameters, so
    predicted values stay on the same scale as the ground-truth signal and noisy
    measurements (e.g. ODMR contrast near 1.0).
    """

    inner: SignalModel
    param_bounds_phys: dict[str, tuple[float, float]]
    x_bounds_phys: tuple[float, float]

    def compute(self, x_unit: float, params: list[Parameter]) -> float:
        x_lo, x_hi = self.x_bounds_phys
        x_phys = x_lo + float(x_unit) * (x_hi - x_lo)
        phys_params: list[Parameter] = []
        for p in params:
            lo, hi = self.param_bounds_phys[p.name]
            u = float(p.value)
            v = lo + u * (hi - lo)
            phys_params.append(Parameter(name=p.name, bounds=(lo, hi), value=v))
        return self.inner.compute(x_phys, phys_params)

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def gradient(self, x: float, params: list[Parameter]) -> dict[str, float] | None:
        return None
