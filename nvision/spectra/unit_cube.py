"""Map unit-interval parameters and probe position to physical signal evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import (
    ParamSpec,
    ParamsT,
    SampleParamsT,
    SignalModel,
    UncertaintyT,
    VectorizedManySamplesInput,
)

_FI = np.finfo(np.dtype(FLOAT_DTYPE))
# ~sqrt(machine epsilon): a few ULPs around 0 and 1 before clipping unit parameters.
_UNIT_INTERVAL_SLACK = np.sqrt(np.float64(_FI.eps))
_ONE_PLUS_SLACK = np.float64(1.0) + _UNIT_INTERVAL_SLACK


def _unit_interval_to_physical(u_raw: np.ndarray, lo: float, hi: float, param_name: str) -> np.ndarray:
    """Map unit-cube samples to ``[lo, hi]``, clipping benign float endpoint error."""
    if np.any(u_raw < -_UNIT_INTERVAL_SLACK) or np.any(u_raw > _ONE_PLUS_SLACK):
        raise ValueError(
            f"Parameter {param_name} unit values must lie in [0, 1]; got min {float(np.min(u_raw))}, "
            f"max {float(np.max(u_raw))}"
        )
    u = np.clip(u_raw, np.float64(0.0), np.float64(1.0))
    lo64 = np.float64(lo)
    hi64 = np.float64(hi)
    v = lo64 + u.astype(np.float64, copy=False) * (hi64 - lo64)
    return np.clip(v, lo64, hi64).astype(FLOAT_DTYPE, copy=False)


class UnitCubeSignalModel[ParamsT, SampleParamsT, UncertaintyT](SignalModel[ParamsT, SampleParamsT, UncertaintyT]):
    """Wrap a physical-domain :class:`SignalModel` for inference on ``[0, 1]``.

    * ``x_unit`` — normalized probe coordinate in ``[0, 1]`` (same convention as
      :meth:`~nvision.models.experiment.CoreExperiment.measure`).
    * Each parameter value in ``params`` is interpreted as a fraction in ``[0, 1]``
      over the corresponding physical interval in ``param_bounds_phys``.

    The inner model is evaluated at physical frequency and physical parameters, so
    predicted values stay on the same scale as the ground-truth signal and noisy
    measurements (e.g. ODMR contrast near 1.0).

    **Layout:** :meth:`compute` maps unit inputs to physical values and delegates to
    ``inner.compute``. Heavy arithmetic lives in :mod:`nvision.spectra.numba_kernels`
    (and similar). ``UnitCubeSignalModel`` itself is not a Numba ``jitclass`` because
    of dict bounds and polymorphic ``inner``.
    """

    __slots__ = ("inner", "param_bounds_phys", "x_bounds_phys")
    _BOUND_TOL = float(_FI.eps)

    def __init__(
        self,
        inner: SignalModel[ParamsT, SampleParamsT, UncertaintyT],
        param_bounds_phys: dict[str, tuple[float, float]],
        x_bounds_phys: tuple[float, float],
    ) -> None:
        self.inner = inner
        self.param_bounds_phys = param_bounds_phys
        self.x_bounds_phys = x_bounds_phys

    @property
    def spec(self) -> ParamSpec[ParamsT, SampleParamsT, UncertaintyT]:
        return self.inner.spec

    def compute(self, x: float, params: ParamsT) -> float:
        u_values = self.spec.pack_params(params)
        names = self.parameter_names()
        x_lo, x_hi = self.x_bounds_phys
        x_phys = x_lo + float(x) * (x_hi - x_lo)
        phys_values: list[float] = []
        for name, u in zip(names, u_values, strict=True):
            lo, hi = self.param_bounds_phys[name]
            v = lo + float(u) * (hi - lo)
            if v < lo - self._BOUND_TOL or v > hi + self._BOUND_TOL:
                raise ValueError(f"Parameter {name} value {v} outside bounds {(lo, hi)}")
            phys_values.append(min(max(v, lo), hi))
        phys_typed = self.inner.spec.unpack_params(phys_values)
        return float(self.inner.compute(x_phys, phys_typed))

    def compute_vectorized_samples(self, x: float, samples: SampleParamsT) -> np.ndarray:
        return self.compute_vectorized(x, *self.spec.pack_samples(samples))

    def compute_from_params(self, x: float, params: ParamsT) -> float:
        return self.compute(x, params)

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
            u_raw = np.asarray(u_arr, dtype=FLOAT_DTYPE)
            phys_arrays.append(_unit_interval_to_physical(u_raw, lo, hi, name))

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
            u_raw = np.asarray(u_arr, dtype=FLOAT_DTYPE)
            phys_arrays.append(_unit_interval_to_physical(u_raw, lo, hi, name))

        x_lo, x_hi = self.x_bounds_phys
        x_phys = x_lo + xs * (x_hi - x_lo)

        if hasattr(self.inner, "compute_vectorized_many"):
            typed_samples = self.inner.spec.unpack_samples(tuple(phys_arrays))
            return self.inner.compute_vectorized_many(x_phys, typed_samples)

        return np.stack([self.inner.compute_vectorized(float(xp), *phys_arrays) for xp in x_phys], axis=0)

    def is_scale_parameter(self, name: str) -> bool:
        return self.inner.is_scale_parameter(name)

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def narrow_physical_interval_for_param(
        self,
        param_name: str,
        new_lo: float,
        new_hi: float,
        *,
        update_x_axis: bool = True,
    ) -> tuple[float, float]:
        """Clip ``(new_lo, new_hi)`` to current bounds and update physical ranges in place.

        Used after a coarse sweep to restrict the scan axis and matching parameter
        interval without rebuilding the model. When ``update_x_axis`` is true, probe
        position maps to the same narrowed physical interval as ``param_name``.
        """
        cur_lo, cur_hi = self.param_bounds_phys[param_name]
        nl = float(max(min(new_lo, new_hi), cur_lo))
        nh = float(min(max(new_lo, new_hi), cur_hi))
        if nh <= nl:
            return (cur_lo, cur_hi)

        self.param_bounds_phys[param_name] = (nl, nh)
        if update_x_axis:
            self.x_bounds_phys = (nl, nh)
        return (nl, nh)
