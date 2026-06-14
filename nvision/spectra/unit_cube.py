"""Map unit-interval parameters and probe position to physical signal evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import (
    SignalModel,
    VectorizedManySamplesInput,
)
from nvision.spectra.spec import ParamSpec

_FI = np.finfo(np.dtype(FLOAT_DTYPE))
# ~sqrt(machine epsilon): a few ULPs around 0 and 1 before clipping unit parameters.
_UNIT_INTERVAL_SLACK = np.sqrt(np.float32(_FI.eps))
_ONE_PLUS_SLACK = np.float32(1.0) + _UNIT_INTERVAL_SLACK


def _unit_interval_to_physical(u_raw: np.ndarray, lo: float, hi: float, param_name: str) -> np.ndarray:
    """Map unit-cube samples to ``[lo, hi]``, clipping benign float endpoint error.

    Validates via a single min/max pass (instead of boolean-array comparisons)
    and skips the input clip entirely when all samples already lie in [0, 1] —
    the overwhelmingly common case on the per-step likelihood path.
    """
    lo32 = np.float32(lo)
    hi32 = np.float32(hi)
    if u_raw.size == 0:
        return u_raw * (hi32 - lo32) + lo32
    mn = float(u_raw.min())
    mx = float(u_raw.max())
    if mn < -_UNIT_INTERVAL_SLACK or mx > _ONE_PLUS_SLACK:
        raise ValueError(f"Parameter {param_name} unit values must lie in [0, 1]; got min {mn}, max {mx}")
    u = u_raw if (mn >= 0.0 and mx <= 1.0) else np.clip(u_raw, np.float32(0.0), np.float32(1.0))
    v = lo32 + u * (hi32 - lo32)
    return np.clip(v, lo32, hi32, out=v)


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
        if len(param_arrays) == 1:
            try:  # noqa: SIM105
                param_arrays = param_arrays[0].arrays_in_order()  # type: ignore[union-attr]
            except AttributeError:
                pass
        if len(param_arrays) != len(names):
            raise ValueError(f"{type(self).__name__}: expected {len(names)} param arrays but got {len(param_arrays)}")

        phys_arrays: list[np.ndarray] = []
        for name, u_arr in zip(names, param_arrays, strict=True):
            lo, hi = self.param_bounds_phys[name]
            u_raw = np.asarray(u_arr, dtype=FLOAT_DTYPE)
            phys_arrays.append(_unit_interval_to_physical(u_raw, lo, hi, name))

        return self.inner.compute_vectorized(x_phys, *phys_arrays)

    def _get_param_arrays_norm(self, samples_norm: VectorizedManySamplesInput[object]) -> Sequence[np.ndarray]:
        """Convert samples_norm (dataclass, tuple, or list) into a sequence of parameter arrays."""
        try:
            return samples_norm.arrays_in_order()  # type: ignore[union-attr]
        except AttributeError:
            if not isinstance(samples_norm, tuple | list):
                # If it's a dataclass/spec-unpackable but doesn't have arrays_in_order
                return self.inner.spec.pack_samples(samples_norm)
            return samples_norm  # type: ignore[return-value]

    def compute_vectorized_many(
        self,
        x_norm_array: Sequence[float],
        samples_norm: VectorizedManySamplesInput[object],
    ) -> np.ndarray:
        """Vectorized signal evaluation at many unit x positions over unit samples."""
        xs_norm = np.asarray(x_norm_array, dtype=FLOAT_DTYPE)
        param_arrays_norm = self._get_param_arrays_norm(samples_norm)

        names = self.parameter_names()
        if len(param_arrays_norm) != len(names):
            raise ValueError(f"Expected {len(names)} parameter arrays, got {len(param_arrays_norm)}")

        phys_arrays: list[np.ndarray] = []
        for name, u_arr in zip(names, param_arrays_norm, strict=True):
            lo, hi = self.param_bounds_phys[name]
            u_raw = np.asarray(u_arr, dtype=FLOAT_DTYPE)
            phys_arrays.append(_unit_interval_to_physical(u_raw, lo, hi, name))

        x_lo, x_hi = self.x_bounds_phys
        xs_phys = x_lo + xs_norm * (x_hi - x_lo)

        typed_samples_phys = self.inner.spec.unpack_samples(tuple(phys_arrays))
        return self.inner.compute_vectorized_many(xs_phys, typed_samples_phys)

    def compute_vectorized_many_fast(
        self,
        x_norm_array: Sequence[float],
        samples_norm: VectorizedManySamplesInput[object],
    ) -> np.ndarray:
        """Fast acquisition-only variant — routes to the inner fastmath kernel.

        Without this override the base class falls through to
        :meth:`compute_vectorized_many`, silently bypassing the ``fastmath``
        compiled kernel in the inner model.  Only call from EIG / acquisition
        scoring paths; never from weight updates.
        """
        xs_norm = np.asarray(x_norm_array, dtype=FLOAT_DTYPE)
        param_arrays_norm = self._get_param_arrays_norm(samples_norm)

        names = self.parameter_names()
        if len(param_arrays_norm) != len(names):
            raise ValueError(f"Expected {len(names)} parameter arrays, got {len(param_arrays_norm)}")

        phys_arrays: list[np.ndarray] = []
        for name, u_arr in zip(names, param_arrays_norm, strict=True):
            lo, hi = self.param_bounds_phys[name]
            u_raw = np.asarray(u_arr, dtype=FLOAT_DTYPE)
            phys_arrays.append(_unit_interval_to_physical(u_raw, lo, hi, name))

        x_lo, x_hi = self.x_bounds_phys
        xs_phys = x_lo + xs_norm * (x_hi - x_lo)

        typed_samples_phys = self.inner.spec.unpack_samples(tuple(phys_arrays))
        return self.inner.compute_vectorized_many_fast(xs_phys, typed_samples_phys)

    def is_scale_parameter(self, name: str) -> bool:
        return self.inner.is_scale_parameter(name)

    def parameter_names(self) -> list[str]:
        return self.inner.parameter_names()

    def signal_min_span(self, domain_width: float) -> float | None:
        x_lo, x_hi = self.x_bounds_phys
        phys_width = float(x_hi - x_lo)
        if phys_width <= 0:
            return self.inner.signal_min_span(domain_width)

        res = self.inner.signal_min_span(phys_width)
        if res is None:
            return None
        return res / phys_width

    def signal_max_span(self, domain_width: float) -> float | None:
        x_lo, x_hi = self.x_bounds_phys
        phys_width = float(x_hi - x_lo)
        if phys_width <= 0:
            return self.inner.signal_max_span(domain_width)

        res = self.inner.signal_max_span(phys_width)
        if res is None:
            return None
        return res / phys_width

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
        nl = float(min(new_lo, new_hi))
        nh = float(max(new_lo, new_hi))
        if nh <= nl:
            return self.param_bounds_phys[param_name]

        self.param_bounds_phys[param_name] = (nl, nh)
        if update_x_axis:
            self.x_bounds_phys = (nl, nh)
        return (nl, nh)
