"""Abstract locator interface for Bayesian localization."""

from abc import ABC, abstractmethod

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.observation import Observation


class Locator(ABC):
    """Stateful locator for one repeat run.

    Owns a BeliefSignal updated incrementally each observation.
    Created fresh per repeat via classmethod `create()`.
    Knows the signal model shape and acquisition strategy
    (EIG, grid, golden section etc) but not the true parameter values.

    Attributes
    ----------
    belief : AbstractMarginalDistribution
        Current belief state about signal parameters
    """

    def __init__(self, belief: AbstractMarginalDistribution):
        """Initialize the locator with the initial belief.

        Parameters
        ----------
        belief : AbstractMarginalDistribution
            Initial belief (usually uniform prior)
        """
        self.belief = belief

    @classmethod
    @abstractmethod
    def create(cls, **config):
        """Create a fresh locator instance with fresh BeliefSignal.

        This is the factory method that subclasses implement to create
        locators with properly initialized beliefs (uniform priors, etc).
        Called once per repeat.

        Parameters
        ----------
        **config
            Configuration parameters (max_steps, convergence_threshold, etc)
            Specific to each locator subclass.

        Returns
        -------
        Locator
            New locator with initialized belief (uniform prior)

        Examples
        --------
        >>> locator = SimpleSweepLocator.create(max_steps=50)
        >>> locator = BayesianLocator.create(
        ...     acquisition="eig",
        ...     max_steps=150,
        ...     convergence_threshold=0.01
        ... )
        """
        pass

    @abstractmethod
    def next(self) -> float:
        """Propose next measurement position.

        Uses current belief to pick next x based on acquisition strategy.
        Strategy (EIG, grid search, golden section, etc.) is implemented
        by concrete subclasses.

        Returns
        -------
        float
            Next position to measure
        """
        pass

    @abstractmethod
    def done(self) -> bool:
        """Check if localization is complete.

        Uses belief.converged() plus max steps or other stopping criteria.

        Returns
        -------
        bool
            True if no more measurements needed
        """
        pass

    @abstractmethod
    def result(self) -> dict[str, float]:
        """Extract final parameter estimates from belief.

        Returns
        -------
        dict[str, float]
            Final parameter estimates (e.g., {'frequency': 2.87e9, ...})
        """
        pass

    def observe(self, obs: Observation) -> None:
        """Update belief with new observation.

        Incremental Bayesian update — no history replay.
        Runner calls this after each measurement.

        Parameters
        ----------
        obs : Observation
            New measurement to incorporate
        """
        self.belief.update(obs)
