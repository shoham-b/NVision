"""Abstract belief distribution interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

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

    Subclasses also expose ``physical_param_bounds`` (``dict[str, tuple[float, float]]``, the physical
    range of each parameter) as a dataclass field or a property.
    """

    model: SignalModel
    last_obs: Observation | None = None
    resampled: bool = False  # Track if a resampling or major structural update happened

    @abstractmethod
    def update(self, obs: Observation) -> None:
        """Incremental Bayesian update from a new observation."""

    @abstractmethod
    def estimates(self) -> dict[str, float]:
        """Get current parameter estimates (e.g., posterior means)."""

    @abstractmethod
    def mode_estimates(self) -> dict[str, float]:
        """Get the posterior mode as one internally consistent joint state.

        Unlike :meth:`estimates` (the marginal mean, taken independently per
        parameter), this must return values that could all belong to the same
        underlying state — not an average that can fall in a low-density
        trough between modes of a multimodal posterior, or combine
        independently-averaged parameters into a combination nothing in the
        posterior actually supports.
        """

    def uncertainty(self) -> ParameterValues[float]:
        """Marginal standard deviation for each parameter from the belief itself.

        Uses :meth:`_empirical_uncertainty` (grid PMFs, weighted particles, etc.)
        so reported values match the represented posterior. For a separate local
        Fisher information lives in :mod:`nvision.models.fisher_information`.
        """
        return self._empirical_uncertainty()

    def robust_uncertainty(self) -> ParameterValues[float]:
        """Outlier-insensitive marginal spread, for gating streak/consecutive-
        checks decisions that would otherwise flicker on a transient event (e.g.
        a handful of outlier particles after an SMC resample -- see
        :meth:`~nvision.belief.smc_marginal.SMCMarginalDistribution.
        _robust_uncertainty_unit` for why).

        This is NOT a general-purpose replacement for :meth:`uncertainty`: it can
        underreport genuine multi-modal spread. Use it only for that narrow
        purpose, not for CRLB comparisons or anything reported as the belief's
        claimed precision. Falls back to :meth:`uncertainty` for belief types
        with no such estimate (e.g. grid beliefs).
        """
        return self._empirical_robust_uncertainty()

    def crlb_center_freq(self) -> float:
        """Analytical Cramér-Rao lower bound for center_freq in physical Hz.

        Returns ``math.inf`` unless overridden by a subclass that knows the
        physical signal model (e.g. NV-center Lorentzian).
        """
        return math.inf

    def reported_uncertainty(self) -> ParameterValues[float]:
        """Uncertainty for external reporting.

        Matches :meth:`uncertainty` directly.
        """
        return self.uncertainty()

    @abstractmethod
    def _empirical_uncertainty(self) -> ParameterValues[float]:
        """Compute empirical uncertainty from the underlying grid/particles."""

    def _empirical_robust_uncertainty(self) -> ParameterValues[float]:
        """Outlier-insensitive variant of :meth:`_empirical_uncertainty`.

        Default falls back to the raw value -- belief types with no cheaper/
        more-robust estimate (e.g. grid beliefs, which have no resample-artifact
        equivalent to filter out) just report the same thing under both names.
        Override where a robust estimate is meaningful (see
        :meth:`~nvision.belief.smc_marginal.SMCMarginalDistribution.
        _empirical_robust_uncertainty`).
        """
        return self._empirical_uncertainty()

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

    def accumulate_fim(self, obs: Observation) -> None:  # noqa: B027
        """Fold ``obs`` into the belief's cumulative Fisher information (no-op unless the belief tracks one)."""

    def crlb_per_param(self) -> dict[str, float]:
        """Marginal CRLB per parameter in physical units; empty unless the belief tracks a Fisher information."""
        return {}

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

    def batch_update(self, observations: Sequence[Observation]) -> None:
        """Update belief from a sequence of observations.

        Default implementation loops over :meth:`update`.  Subclasses may
        override with more efficient batch algorithms.
        """
        for obs in observations:
            self.update(obs)

    def get_candidate_x_phys(self) -> np.ndarray:
        """Return candidate measurement positions for acquisition selection.

        Subclasses override this to implement custom grid generation logic (e.g.
        epoch-based grids, slope-targeted grids, etc.). If not overridden,
        returns a default linear grid based on model spans.
        """
        lo, hi = self.physical_param_bounds[self.model.parameter_names()[0]]
        return np.linspace(lo, hi, 100)

    def _to_physical(self, param_name: str, val: float) -> float:
        """Convert an internal coordinate back to physical space.

        Default implementation returns the value unmodified (for beliefs operating in physical space).
        Unit-cube wrappers override this to map [0, 1] back to the physical bounds.
        """
        return float(val)
