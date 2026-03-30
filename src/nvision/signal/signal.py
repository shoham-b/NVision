"""Signal model abstractions for Bayesian localization.

This module supports the *new generic signal* interface (typed parameter bundles)
while remaining compatible with the existing core simulation/belief machinery,
which still passes around legacy :class:`~nvision.signal.signal.Parameter` objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np

from nvision.parameter import Parameter

ParamsT = TypeVar("ParamsT")
SampleParamsT = TypeVar("SampleParamsT")
UncertaintyT = TypeVar("UncertaintyT")


@runtime_checkable
class ParamSpec(Protocol[ParamsT, SampleParamsT, UncertaintyT]):
    """Adapter that lets generic beliefs work with typed parameter bundles.

    Beliefs operate on numeric vectors / arrays; models operate on typed bundles.
    This spec defines the mapping between those representations.
    """

    @property
    def names(self) -> tuple[str, ...]: ...

    @property
    def dim(self) -> int: ...

    def unpack_params(self, values: Sequence[float]) -> ParamsT: ...
    def pack_params(self, params: ParamsT) -> tuple[float, ...]: ...

    def unpack_uncertainty(self, values: Sequence[float]) -> UncertaintyT: ...
    def pack_uncertainty(self, u: UncertaintyT) -> tuple[float, ...]: ...

    def unpack_samples(self, arrays_in_order: Sequence[np.ndarray]) -> SampleParamsT: ...
    def pack_samples(self, samples: SampleParamsT) -> tuple[np.ndarray, ...]: ...


@runtime_checkable
class ArraysInOrder(Protocol):
    """Container exposing parameter arrays in model order."""

    def arrays_in_order(self) -> Sequence[np.ndarray]: ...


type VectorizedManySamplesInput[T] = T | Sequence[np.ndarray] | ArraysInOrder


class SignalModel[ParamsT, SampleParamsT, UncertaintyT](ABC):
    """Abstract signal model with typed parameter bundles.

    Concrete subclasses must implement:
    - :meth:`compute` for scalar evaluation
    - :meth:`compute_vectorized_samples` for vectorized evaluation over samples

    This base class then provides compatibility wrappers:
    - :meth:`parameter_names`
    - :meth:`compute_from_params` (accepts either legacy ``list[Parameter]`` or a typed params bundle)
    - :meth:`compute_vectorized` (accepts either belief samples or raw parameter arrays)
    """

    @property
    @abstractmethod
    def spec(self) -> ParamSpec[ParamsT, SampleParamsT, UncertaintyT]:
        """Mapping between numeric vectors and typed parameter bundles."""

    @abstractmethod
    def compute(self, x: float, params: ParamsT) -> float:
        """Scalar prediction at probe x."""

    @abstractmethod
    def compute_vectorized_samples(self, x: float, samples: SampleParamsT) -> np.ndarray:
        """Vectorized prediction at one probe x over many parameter samples."""

    def compute_vectorized_many(
        self,
        x_array: Sequence[float],
        samples: VectorizedManySamplesInput[SampleParamsT],
    ) -> np.ndarray:
        """Vectorized prediction at many probe positions over shared samples.

        Default implementation stacks repeated :meth:`compute_vectorized_samples`
        calls along axis 0, returning shape ``(len(x_array), n_samples)``.
        Concrete models may override with a fused implementation.
        """
        xs = np.asarray(x_array, dtype=float)
        if xs.ndim != 1:
            raise ValueError("x_array must be one-dimensional")
        if xs.size == 0:
            return np.empty((0, 0), dtype=float)

        # Allow either typed sample bundles or raw parameter-array sequences.
        if isinstance(samples, ArraysInOrder | tuple | list):
            param_arrays = samples.arrays_in_order() if hasattr(samples, "arrays_in_order") else samples
            return np.stack([self.compute_vectorized(float(x), *param_arrays) for x in xs], axis=0)

        return np.stack([self.compute_vectorized_samples(float(x), samples) for x in xs], axis=0)

    # --------------------------
    # Compatibility / legacy API
    # --------------------------

    def is_scale_parameter(self, name: str) -> bool:
        """Return True if the parameter represents a strictly-positive scale
        (e.g., width, amplitude) that should employ logarithmic spacing.
        """
        return False

    def parameter_names(self) -> list[str]:
        """Return parameter names in the order expected by beliefs/core generators."""

        return list(self.spec.names)

    def gradient(self, x: float, params: list[Parameter]) -> dict[str, float] | None:
        """Optional analytical gradient support (defaults to None)."""

        return None

    def compute_from_params(self, x: float, params: ParamsT | list[Parameter]) -> float:
        """Evaluate using either typed params or legacy ``list[Parameter]``."""

        x_f = float(x)
        if isinstance(params, list):
            if params and not isinstance(params[0], Parameter):
                raise TypeError("params list must contain nvision.signal.signal.Parameter objects")
            by_name: Mapping[str, float] = {p.name: float(p.value) for p in params}
            values = [by_name[name] for name in self.parameter_names()]
            typed = self.spec.unpack_params(values)
            return float(self.compute(x_f, typed))

        # Assume typed bundle
        return float(self.compute(x_f, params))

    def compute_vectorized(self, x: float, *args: object) -> np.ndarray:
        """Vectorized evaluation used by beliefs and acquisition locators.

        Supported call shapes:
        - ``model.compute_vectorized(x, *param_arrays)`` where each array corresponds
          to a parameter in :meth:`parameter_names`.
        - ``model.compute_vectorized(x, samples)`` where ``samples`` is a belief-sample
          container providing ``arrays_in_order()`` (e.g. ``ParameterValues``).
        - ``model.compute_vectorized(x, typed_samples)`` where ``typed_samples`` is the
          model's native ``SampleParamsT`` bundle.
        """

        x_f = float(x)
        if len(args) == 1:
            s = args[0]
            if hasattr(s, "arrays_in_order"):
                arrays = s.arrays_in_order()
                samples = self.spec.unpack_samples(arrays)
                return self.compute_vectorized_samples(x_f, samples)
            # Assume native typed sample bundle
            return self.compute_vectorized_samples(x_f, s)  # type: ignore[arg-type]

        # Treat args as raw parameter arrays in parameter_names order.
        samples = self.spec.unpack_samples(args)  # type: ignore[arg-type]
        return self.compute_vectorized_samples(x_f, samples)


@dataclass
class TrueSignal[ParamsT]:
    """Ground-truth signal with typed model parameters."""

    model: SignalModel[ParamsT, Any, Any]
    typed_parameters: ParamsT
    bounds: Mapping[str, tuple[float, float]]

    def __call__(self, x: float) -> float:
        return float(self.model.compute(float(x), self.typed_parameters))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.model.spec.names)

    def parameter_values(self) -> dict[str, float]:
        values = self.model.spec.pack_params(self.typed_parameters)
        return {name: float(value) for name, value in zip(self.parameter_names, values, strict=True)}

    def get_param_value(self, name: str) -> float:
        values = self.parameter_values()
        if name not in values:
            raise KeyError(name)
        return float(values[name])

    def is_scale_parameter(self, name: str) -> bool:
        """Forward checks to the inner model."""
        return self.model.is_scale_parameter(name)

    def get_param_bounds(self, name: str) -> tuple[float, float]:
        if name not in self.bounds:
            raise KeyError(name)
        lo, hi = self.bounds[name]
        return float(lo), float(hi)

    def all_param_bounds(self) -> dict[str, tuple[float, float]]:
        return {name: self.get_param_bounds(name) for name in self.parameter_names}

    @classmethod
    def from_typed(
        cls,
        model: SignalModel[ParamsT, Any, Any],
        params: ParamsT,
        *,
        bounds: Mapping[str, tuple[float, float]],
    ) -> TrueSignal[ParamsT]:
        """Build a TrueSignal from typed params plus explicit legacy bounds."""

        names = tuple(model.spec.names)
        values = tuple(float(v) for v in model.spec.pack_params(params))
        missing = [name for name in names if name not in bounds]
        if missing:
            raise KeyError(f"Missing bounds for typed TrueSignal parameters: {missing}")
        _ = values  # Keep pack validation coupled to names order.
        return cls(model=model, typed_parameters=params, bounds=dict(bounds))
