"""Signal model abstractions for Bayesian localization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

ParamsT = TypeVar("ParamsT")
SampleParamsT = TypeVar("SampleParamsT")
UncertaintyT = TypeVar("UncertaintyT")


class GenericParamSpec(Generic[ParamsT, SampleParamsT, UncertaintyT]):
    """Auto-implement ParamSpec methods using dataclass field introspection.

    Subclasses must define three class attributes:
        params_cls: type[ParamsT]          # frozen dataclass with fields
        samples_cls: type[SampleParamsT]  # frozen dataclass with np.ndarray fields
        uncertainty_cls: type[UncertaintyT]  # frozen dataclass with fields

    The ``names`` property is derived from ``params_cls`` field names.
    All pack/unpack methods work by iterating fields in declaration order.
    """

    params_cls: type[ParamsT]
    samples_cls: type[SampleParamsT]
    uncertainty_cls: type[UncertaintyT]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self.params_cls))

    @property
    def dim(self) -> int:
        return len(self.names)

    def unpack_params(self, values: Sequence[float]) -> ParamsT:
        return self.params_cls(**dict(zip(self.names, values)))

    def pack_params(self, params: ParamsT) -> tuple[float, ...]:
        return tuple(getattr(params, name) for name in self.names)

    def unpack_uncertainty(self, values: Sequence[float]) -> UncertaintyT:
        return self.uncertainty_cls(**dict(zip(self.names, values)))

    def pack_uncertainty(self, u: UncertaintyT) -> tuple[float, ...]:
        return tuple(getattr(u, name) for name in self.names)

    def unpack_samples(self, arrays_in_order: Sequence[np.ndarray]) -> SampleParamsT:
        from nvision.spectra.dtypes import FLOAT_DTYPE

        return self.samples_cls(
            **{
                name: np.asarray(arr, dtype=FLOAT_DTYPE)
                for name, arr in zip(self.names, arrays_in_order, strict=True)
            }
        )

    def pack_samples(self, samples: SampleParamsT) -> tuple[np.ndarray, ...]:
        from nvision.spectra.dtypes import FLOAT_DTYPE

        return tuple(
            np.asarray(getattr(samples, name), dtype=FLOAT_DTYPE) for name in self.names
        )


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

    @property
    def inner(self) -> SignalModel:
        """Return the inner physical model (for wrapped models like UnitCubeSignalModel).

        By default returns self. Subclasses that wrap other models (e.g.,
        UnitCubeSignalModel) should override to return the wrapped inner model.
        """
        return self  # type: ignore[return-value]

    def is_scale_parameter(self, name: str) -> bool:
        """Return True if the parameter represents a strictly-positive scale
        (e.g., width, amplitude) that should employ logarithmic spacing.
        """
        return False

    def signal_min_span(self, domain_width: float) -> float | None:
        """Minimum possible frequency span of this signal in physical units.

        Determines the maximum sweep step count: the sweep must be dense enough
        to guarantee hits even when the signal is at its narrowest.
        Returns ``None`` if the model cannot estimate this.
        """
        return None

    def signal_max_span(self, domain_width: float) -> float | None:
        """Maximum possible frequency span of this signal in physical units.

        Used to size the mid-sweep refocus window so all dips (including outer
        Zeeman-split dips) fall inside the focus band.
        Returns ``None`` if the model cannot estimate this.
        """
        return None

    def expected_dip_count(self) -> int:
        """Expected number of dips/peaks in this signal's spectrum.

        Used by locators to apply appropriate window narrowing strategies
        after coarse sweeps. For example:
        - 1: Single dip (e.g., unstrained NV center)
        - 2: Doublet (e.g., strain-split NV center ms=+1/-1 transitions)
        - 3: Triplet (e.g., NV center with all three ms transitions visible)

        Returns 1 by default; subclasses with known multi-dip structure should override.
        """
        return 1

    def parameter_weights(self) -> dict[str, float]:
        """Return relative convergence weights for each parameter (default 1.0).

        A weight ``w > 1`` for a parameter means the locator must reach
        ``convergence_threshold / w`` normalized uncertainty for that parameter
        before declaring convergence.  Override in concrete models to tighten
        convergence on parameters whose estimation drives accuracy of the rest.
        """
        return {name: 1.0 for name in self.parameter_names()}

    def parameter_names(self) -> list[str]:
        """Return parameter names in the order expected by beliefs/core generators."""

        return list(self.spec.names)

    def gradient(self, x: float, params: ParamsT) -> dict[str, float] | None:
        """Optional analytical gradient support (defaults to None)."""

        return None

    def gradient_vectorized(
        self, x: float, *param_arrays: object
    ) -> dict[str, np.ndarray] | None:
        """Vectorized gradient computation for all particles at position x.

        Returns a dict mapping parameter names to arrays of gradient values
        (one per particle). Default implementation loops over particles and
        calls gradient() - models with analytical gradients should override.

        Parameters
        ----------
        x : float
            Measurement position.
        *param_arrays : object
            Arrays of parameter values in parameter_names order, each shaped
            (n_particles,).

        Returns
        -------
        dict[str, np.ndarray] | None
            Gradient arrays per parameter, or None if gradients unavailable.
        """
        n_particles = len(param_arrays[0]) if param_arrays else 0
        if n_particles == 0:
            return None

        # Loop over particles and collect gradients (slow fallback)
        param_names = self.parameter_names()
        result: dict[str, list[float]] = {name: [] for name in param_names}

        for i in range(n_particles):
            # Extract single particle params
            particle_values = [float(arr[i]) for arr in param_arrays]
            typed_params = self.spec.unpack_params(particle_values)
            grads = self.gradient(float(x), typed_params)

            if grads is None:
                return None  # Model doesn't support gradients

            for name in param_names:
                result[name].append(float(grads[name]))

        return {name: np.array(values) for name, values in result.items()}

    def compute_from_params(self, x: float, params: ParamsT) -> float:
        """Evaluate the model at ``x`` using a typed parameter bundle."""

        return float(self.compute(float(x), params))

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

    def min_dip_amplitude(self) -> float | None:
        """Return the smallest dip amplitude for multi-dip signals, or None for single-peak.

        For NV center with Zeeman splitting (3 dips), the smallest dip is:
            dip_depth / k_np^2

        This is used to constrain noise so that max_noise < smallest_dip,
        ensuring the signal remains detectable.
        """
        params = self.typed_parameters

        # NV center models with k_np and dip_depth
        if hasattr(params, "k_np") and hasattr(params, "dip_depth"):
            k_np = float(params.k_np)
            dip_depth = float(params.dip_depth)
            # For 3-dip Zeeman splitting, smallest dip is left dip: dip_depth / k_np^2
            return dip_depth / (k_np ** 2)

        return None

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
