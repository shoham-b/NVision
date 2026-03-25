"""Map unit-interval parameters and probe position to physical signal evaluation."""

from __future__ import annotations

from sys import float_info

from nvision.signal.signal import Parameter, SignalModel


class UnitCubeSignalModel:
    """Wrap a physical-domain :class:`SignalModel` for inference on ``[0, 1]``.

    * ``x_unit`` — normalized probe coordinate in ``[0, 1]`` (same convention as
      :meth:`~nvision.models.experiment.CoreExperiment.measure`).
    * Each parameter value in ``params`` is interpreted as a fraction in ``[0, 1]``
      over the corresponding physical interval in ``param_bounds_phys``.

    The inner model is evaluated at physical frequency and physical parameters, so
    predicted values stay on the same scale as the ground-truth signal and noisy
    measurements (e.g. ODMR contrast near 1.0).

    **Layout:** :meth:`compute` stays a thin Python shim: map unit inputs to physical
    values, keep a **stateful** list of physical :class:`Parameter` instances
    (created once, updated in place) to avoid per-call allocation, then delegate to
    ``inner.compute``.     Inner :class:`~nvision.signal.signal.SignalModel` subclasses may expose
    ``eval_<class_snake>`` for direct float evaluation; this wrapper still uses
    :meth:`~nvision.signal.signal.SignalModel.compute` with physical
    ``list[Parameter]``. Heavy arithmetic lives in :mod:`nvision.signal.numba_kernels`
    (and similar). ``UnitCubeSignalModel``
    itself is not a Numba ``jitclass`` because of dict bounds and polymorphic ``inner``.
    """

    __slots__ = ("_phys_params", "inner", "param_bounds_phys", "x_bounds_phys")
    _BOUND_TOL = float_info.epsilon

    def __init__(
        self,
        inner: SignalModel,
        param_bounds_phys: dict[str, tuple[float, float]],
        x_bounds_phys: tuple[float, float],
    ) -> None:
        self.inner = inner
        self.param_bounds_phys = param_bounds_phys
        self.x_bounds_phys = x_bounds_phys
        self._phys_params: list[Parameter] = []
        for name in inner.parameter_names():
            lo, hi = param_bounds_phys[name]
            mid = 0.5 * (lo + hi)
            self._phys_params.append(Parameter(name=name, bounds=(lo, hi), value=mid))

    def compute(self, x_unit: float, params: list[Parameter]) -> float:
        x_lo, x_hi = self.x_bounds_phys
        x_phys = x_lo + float(x_unit) * (x_hi - x_lo)
        u_by_name = {p.name: float(p.value) for p in params}
        for p in self._phys_params:
            lo, hi = self.param_bounds_phys[p.name]
            u = u_by_name[p.name]
            v = lo + u * (hi - lo)
            if v < lo - self._BOUND_TOL or v > hi + self._BOUND_TOL:
                raise ValueError(f"Parameter {p.name} value {v} outside bounds {(lo, hi)}")
            # Guard against tiny floating-point endpoint overshoot (e.g. 0.5000000000000001).
            v = min(max(v, lo), hi)
            p.value = v
        return self.inner.compute(x_phys, self._phys_params)

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def gradient(self, x: float, params: list[Parameter]) -> dict[str, float] | None:
        return None
