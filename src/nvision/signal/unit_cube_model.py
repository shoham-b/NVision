"""Map unit-interval parameters and probe position to physical signal evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from sys import float_info

import numpy as np

from nvision.signal.dtypes import FLOAT_DTYPE
from nvision.signal.signal import (
    Parameter,
    ParamSpec,
    ParamsT,
    SampleParamsT,
    SignalModel,
    UncertaintyT,
    VectorizedManySamplesInput,
)


class UnitCubeSignalModel[ParamsT, SampleParamsT, UncertaintyT](SignalModel[ParamsT, SampleParamsT, UncertaintyT]):
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
    ``evaluate_<class_snake>`` for direct float evaluation; this wrapper still uses
    :meth:`~nvision.signal.signal.SignalModel.compute` with physical
    ``list[Parameter]``. Heavy arithmetic lives in :mod:`nvision.signal.numba_kernels`
    (and similar). ``UnitCubeSignalModel``
    itself is not a Numba ``jitclass`` because of dict bounds and polymorphic ``inner``.
    """

    __slots__ = ("_phys_params", "inner", "param_bounds_phys", "x_bounds_phys")
    _BOUND_TOL = float_info.epsilon

    def __init__(
        self,
        inner: SignalModel[ParamsT, SampleParamsT, UncertaintyT],
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

    @property
    def spec(self) -> ParamSpec[ParamsT, SampleParamsT, UncertaintyT]:
        return self.inner.spec

    def compute(self, x: float, params: ParamsT) -> float:
        values = self.spec.pack_params(params)
        unit_params = [
            Parameter(name=n, bounds=(0.0, 1.0), value=float(v))
            for n, v in zip(self.parameter_names(), values, strict=True)
        ]
        return self.compute_from_params(x, unit_params)

    def compute_vectorized_samples(self, x: float, samples: SampleParamsT) -> np.ndarray:
        return self.compute_vectorized(x, *self.spec.pack_samples(samples))

    def compute_from_params(self, x_unit: float, params: list[Parameter]) -> float:
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
        return self.inner.compute_from_params(x_phys, self._phys_params)

    def compute_vectorized(self, x_unit: float, *param_arrays: object) -> np.ndarray:
        """Vectorized one-x evaluation over many unit-cube parameter samples.

        ``param_arrays`` are passed in :meth:`parameter_names` order (same names as the
        wrapped physical inner model), but the values are in the unit-cube ``[0, 1]`` interval.
        """
        x_lo, x_hi = self.x_bounds_phys
        x_phys = x_lo + float(x_unit) * (x_hi - x_lo)

        names = self.parameter_names()
        if len(param_arrays) == 1 and hasattr(param_arrays[0], "arrays_in_order"):
            bundle = param_arrays[0]
            param_arrays = bundle.arrays_in_order()
        if len(param_arrays) != len(names):
            raise ValueError(f"{type(self).__name__}: expected {len(names)} param arrays but got {len(param_arrays)}")

        phys_arrays: list[np.ndarray] = []
        for name, u_arr in zip(names, param_arrays, strict=True):
            lo, hi = self.param_bounds_phys[name]
            u = np.asarray(u_arr, dtype=FLOAT_DTYPE)
            v = lo + u * (hi - lo)
            if np.any(v < lo - self._BOUND_TOL) or np.any(v > hi + self._BOUND_TOL):
                raise ValueError(f"Parameter {name} has values outside bounds {(lo, hi)}")
            phys_arrays.append(np.clip(v, lo, hi).astype(FLOAT_DTYPE, copy=False))

        return self.inner.compute_vectorized(x_phys, *phys_arrays)

    def compute_vectorized_many(
        self,
        x_array: Sequence[float],
        samples: VectorizedManySamplesInput[object],
    ) -> np.ndarray:
        """Vectorized evaluation at many unit x positions over shared samples."""
        xs = np.asarray(x_array, dtype=FLOAT_DTYPE)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        if xs.size == 0:
            return np.empty((0, 0), dtype=FLOAT_DTYPE)

        names = self.parameter_names()
        if hasattr(samples, "arrays_in_order"):
            param_arrays = samples.arrays_in_order()
        elif isinstance(samples, tuple | list):
            param_arrays = samples
        else:
            raise TypeError("samples must provide arrays_in_order() or be parameter arrays")

        if len(param_arrays) != len(names):
            raise ValueError(f"{type(self).__name__}: expected {len(names)} param arrays but got {len(param_arrays)}")

        phys_arrays: list[np.ndarray] = []
        for name, u_arr in zip(names, param_arrays, strict=True):
            lo, hi = self.param_bounds_phys[name]
            u = np.asarray(u_arr, dtype=FLOAT_DTYPE)
            v = lo + u * (hi - lo)
            if np.any(v < lo - self._BOUND_TOL) or np.any(v > hi + self._BOUND_TOL):
                raise ValueError(f"Parameter {name} has values outside bounds {(lo, hi)}")
            phys_arrays.append(np.clip(v, lo, hi).astype(FLOAT_DTYPE, copy=False))

        x_lo, x_hi = self.x_bounds_phys
        x_phys = x_lo + xs * (x_hi - x_lo)

        if hasattr(self.inner, "compute_vectorized_many"):
            typed_samples = self.inner.spec.unpack_samples(tuple(phys_arrays))
            return self.inner.compute_vectorized_many(x_phys, typed_samples)

        return np.stack([self.inner.compute_vectorized(float(xp), *phys_arrays) for xp in x_phys], axis=0)

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def gradient(self, x: float, params: list[Parameter]) -> dict[str, float] | None:
        return None
