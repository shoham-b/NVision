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

ParamsT = TypeVar("ParamsT")
SampleParamsT = TypeVar("SampleParamsT")
UncertaintyT = TypeVar("UncertaintyT")


@dataclass(slots=True)
class Parameter:
    """Legacy numeric parameter with bounds and current value."""

    name: str
    bounds: tuple[float, float]
    value: float

    def __post_init__(self) -> None:
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        if hi <= lo:
            raise ValueError(f"Invalid bounds for {self.name}: {(lo, hi)}")
        self.bounds = (lo, hi)
        self.value = float(self.value)


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
class TrueSignal:
    """Ground truth signal used by the core simulation.

    The core architecture stores parameters as legacy :class:`Parameter` objects.
    """

    model: SignalModel[Any, Any, Any]
    parameters: list[Parameter]

    def __call__(self, x: float) -> float:
        return float(self.model.compute_from_params(float(x), self.parameters))

    @property
    def params(self) -> list[Parameter]:
        """Alias for backwards compatibility (new code uses ``parameters``)."""

        return self.parameters

    def get_param(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)
