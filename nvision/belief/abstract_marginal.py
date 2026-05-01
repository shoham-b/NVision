"""Abstract belief distribution interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from nvision.models.fisher_information import (
    fisher_information_matrix,
    single_shot_marginal_stds_from_fim,
)
from nvision.models.observation import Observation
from nvision.spectra.signal import SignalModel

T = TypeVar("T")


@dataclass(frozen=True)
class ParameterValues[T](Mapping[str, T]):
    """Parameter values with fixed model order plus mapping-style access."""

    names: tuple[str, ...]
    values_ordered: tuple[T, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values_ordered):
            raise ValueError("names and values_ordered lengths must match")

    @classmethod
    def from_mapping(cls, names: list[str], data: Mapping[str, T]) -> ParameterValues[T]:
        return cls(tuple(names), tuple(data[name] for name in names))

    def __getitem__(self, key: str) -> T:
        try:
            idx = self.names.index(key)
        except ValueError as e:
            raise KeyError(key) from e
        return self.values_ordered[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def arrays_in_order(self) -> tuple[T, ...]:
        return self.values_ordered

    def as_dict(self) -> dict[str, T]:
        return {name: self.values_ordered[i] for i, name in enumerate(self.names)}


@dataclass
class AbstractMarginalDistribution(ABC):
    """Abstract base class for all belief distributions.

    Represents the locator's live belief about the signal parameters.
    Can be implemented via discrete grids, Monte Carlo particles, or
    analytical approximations.

    Attributes
    ----------
    model : SignalModel
        The stateless signal model defining the shape.
    last_obs : Observation | None
        Most recent observation for history tracking.
    """

    model: SignalModel
    last_obs: Observation | None = None

    @abstractmethod
    def update(self, obs: Observation) -> None:
        """Incremental Bayesian update from a new observation."""

    @abstractmethod
    def estimates(self) -> dict[str, float]:
        """Get current parameter estimates (e.g., posterior means)."""

    def uncertainty(self) -> ParameterValues[float]:
        """Marginal standard deviation for each parameter from the belief itself.

        Uses :meth:`_empirical_uncertainty` (grid PMFs, weighted particles, etc.)
        so reported values match the represented posterior. For a separate local
        Fisher diagnostic at the last probe only, see :meth:`single_shot_information_std`.
        """
        return self._empirical_uncertainty()

    def single_shot_information_std(self) -> ParameterValues[float]:
        """Local Fisher diagonal scale at :attr:`last_obs` — **not** posterior uncertainty.

        Uses :func:`~nvision.models.observation.single_shot_marginal_stds_from_fim` with
        :meth:`fisher_information` at the
        most recent probe (Moore-Penrose inverse, no separate singular branch). That
        is a single-shot sensitivity snapshot; it does **not** equal marginal
        posterior std after many updates (use :meth:`uncertainty` for that).

        If there is no last observation, no gradients, or the Fisher matrix shape
        does not match the model parameters, every entry is ``nan``.
        """
        obs = self.last_obs
        names = tuple(self.model.parameter_names())
        fim = self.fisher_information(obs.x) if obs is not None else None
        stds = single_shot_marginal_stds_from_fim(fim, len(names))
        data = {names[i]: float(stds[i]) for i in range(len(names))}
        return ParameterValues.from_mapping(list(names), data)

    @abstractmethod
    def _empirical_uncertainty(self) -> ParameterValues[float]:
        """Compute empirical uncertainty from the underlying grid/particles."""

    @abstractmethod
    def entropy(self) -> float:
        """Compute total entropy across all parameters.

        This could be overridden by future subclasses to compute analytical
        entropy instead of empirical entropy.
        """

    @abstractmethod
    def converged(self, threshold: float) -> bool:
        """Check if all parameters have converged below threshold."""

    @abstractmethod
    def copy(self) -> AbstractMarginalDistribution:
        """Create deep copy of this belief for snapshotting."""

    def expected_information_gain(self, x: float) -> float:
        """Compute expected information gain if we measure at position x.

        By default, this is not implemented. Future analytical models (like a
        LaplaceBeliefDistribution) can implement this mathematically using
        the SignalModel gradients, completely bypassing the need for SMC.
        """
        raise NotImplementedError("Analytical EIG not implemented for this belief type.")

    @abstractmethod
    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        """Draw n joint samples from the posterior distribution.

        Returns
        -------
        ParameterValues[np.ndarray]
            Ordered parameter arrays (and mapping-style lookup) of length n.
        """

    @abstractmethod
    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate the marginal Probability Density Function.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        x : np.ndarray
            Points at which to evaluate the PDF.

        Returns
        -------
        np.ndarray
            PDF values corresponding to x.
        """

    def __call__(self, x: float) -> float:
        """Evaluate belief signal at position x using posterior means."""
        names = self.model.parameter_names()
        est = self.estimates()
        typed = self.model.spec.unpack_params([est[n] for n in names])
        return self.model.compute_from_params(x, typed)

    def fisher_information(self, x: float) -> np.ndarray | None:
        """Delegate to :func:`~nvision.models.fisher_information.fisher_information_matrix`.

        Gaussian sigma comes from :attr:`last_obs` via :func:`~nvision.models.observation.gaussian_likelihood_std`.

        Returns None if the underlying SignalModel does not support analytical gradients.
        """
        names = self.model.parameter_names()
        est = self.estimates()
        typed = self.model.spec.unpack_params([est[n] for n in names])
        return fisher_information_matrix(
            x=x,
            model=self.model,
            parameters=typed,
            last_obs=self.last_obs,
        )

    @abstractmethod
    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate the marginal Cumulative Density Function.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        x : np.ndarray
            Points at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            CDF values corresponding to x.
        """

    from collections.abc import Sequence

    def batch_update(self, observations: Sequence[Observation]) -> None:
        """Update belief from a sequence of observations.

        Default implementation loops over :meth:`update`.  Subclasses may
        override with more efficient batch algorithms.
        """
        for obs in observations:
            self.update(obs)

    @property
    @abstractmethod
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds for each parameter (same as ``parameter_bounds`` for non-unit-cube beliefs)."""

    @abstractmethod
    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Shrink physical bounds for ``param_name`` after a coarse sweep.

        Default is a no-op for beliefs that operate directly in physical space.
        Unit-cube beliefs override to remap their internal normalized coordinates.
        """

    def normalized_uncertainties(self) -> ParameterValues[float]:
        """Return uncertainties in normalized [0, 1] space (for convergence checking).

        For beliefs already in physical space this equals :meth:`_empirical_uncertainty`.
        Unit-cube beliefs override to return the raw unit-cube uncertainties before scaling.
        """
        return self._empirical_uncertainty()
