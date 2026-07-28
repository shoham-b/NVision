"""Parameter specification protocols for joint inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

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


class BasicParamSpec:
    """Simple ParamSpec that doesn't use typed bundles (uses tuples/dicts)."""

    def __init__(self, names: list[str], bounds: dict[str, tuple[float, float]]):
        self._names = tuple(names)
        self._bounds = bounds

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def dim(self) -> int:
        return len(self._names)

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return self._bounds

    def unpack_params(self, values: Sequence[float]) -> tuple[float, ...]:
        return tuple(values)

    def pack_params(self, params: Sequence[float]) -> tuple[float, ...]:
        return tuple(params)

    def unpack_uncertainty(self, values: Sequence[float]) -> tuple[float, ...]:
        return tuple(values)

    def pack_uncertainty(self, u: Sequence[float]) -> tuple[float, ...]:
        return tuple(u)

    def unpack_samples(self, arrays_in_order: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        return tuple(arrays_in_order)

    def pack_samples(self, samples: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        return tuple(samples)


class GenericParamSpec[ParamsT, SampleParamsT, UncertaintyT]:
    """Auto-implement ParamSpec methods using dataclass field introspection.

    ``fixed_values`` lets a subset of the dataclass's fields be locked to a
    constant instead of exposed as a free/inferred dimension: ``.names``/``.dim``
    (and therefore particle dimensionality) exclude them, while ``unpack_params``/
    ``unpack_samples`` still populate the full dataclass by filling those fields
    in from ``fixed_values`` -- so ``params_cls``/``samples_cls`` instances are
    always fully populated regardless of which fields are free.
    """

    params_cls: type[ParamsT]
    samples_cls: type[SampleParamsT]
    uncertainty_cls: type[UncertaintyT]
    fixed_values: dict[str, float]

    def __init__(self, fixed_values: dict[str, float] | None = None) -> None:
        self.fixed_values = dict(fixed_values) if fixed_values else {}
        # `names` is fully determined by `params_cls` (a class attribute) and
        # `fixed_values` (set once above, never mutated afterward), so it is
        # safe -- and, since every pack/unpack call re-derives it, worthwhile
        # -- to compute once here instead of re-running dataclass field
        # introspection on every access.
        from dataclasses import fields

        self._names: tuple[str, ...] = tuple(f.name for f in fields(self.params_cls) if f.name not in self.fixed_values)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def dim(self) -> int:
        return len(self.names)

    def unpack_params(self, values: Sequence[float]) -> ParamsT:
        d = dict(zip(self.names, values, strict=False))
        d.update(self.fixed_values)
        return self.params_cls(**d)

    def pack_params(self, params: ParamsT) -> tuple[float, ...]:
        return tuple(getattr(params, name) for name in self.names)

    def unpack_uncertainty(self, values: Sequence[float]) -> UncertaintyT:
        d = dict(zip(self.names, values, strict=False))
        d.update(dict.fromkeys(self.fixed_values, 0.0))
        return self.uncertainty_cls(**d)

    def pack_uncertainty(self, u: UncertaintyT) -> tuple[float, ...]:
        return tuple(getattr(u, name) for name in self.names)

    def unpack_samples(self, arrays_in_order: Sequence[np.ndarray]) -> SampleParamsT:
        from nvision.spectra.dtypes import FLOAT_DTYPE

        d = {name: np.asarray(arr, dtype=FLOAT_DTYPE) for name, arr in zip(self.names, arrays_in_order, strict=True)}
        if self.fixed_values:
            n = next(iter(d.values())).shape[0] if d else 0
            for name, val in self.fixed_values.items():
                d[name] = np.full(n, val, dtype=FLOAT_DTYPE)
        return self.samples_cls(**d)

    def pack_samples(self, samples: SampleParamsT) -> tuple[np.ndarray, ...]:
        from nvision.spectra.dtypes import FLOAT_DTYPE

        return tuple(np.asarray(getattr(samples, name), dtype=FLOAT_DTYPE) for name in self.names)


class SignalParamSpec(ParamSpec):
    """Refined protocol for signal models (backward compat)."""

    @property
    def bounds(self) -> dict[str, tuple[float, float]]: ...


@runtime_checkable
class NoiseSignalModel(Protocol):
    """Abstract base for noise models that can be jointly inferred.

    A NoiseSignalModel provides a ParamSpec for its latent parameters
    and a composite log-likelihood kernel that combines aleatoric
    (physical noise) and epistemic (parameter uncertainty) spread.
    """

    @property
    def spec(self) -> ParamSpec:
        """The parameter specification for this noise model."""

    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        """Compute per-particle log-likelihoods."""
