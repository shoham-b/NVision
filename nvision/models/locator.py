"""Abstract locator interface for Bayesian localization."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.observation import Observation


@dataclass
class ConvergenceConfig:
    """Convergence parameters for Locators."""

    threshold: float = 0.01
    patience_steps: int = 8
    params: Sequence[str] | None = None


@dataclass
class LocatorConfig:
    """Configuration bundle for Locators."""

    max_steps: int = 150
    noise_std: float | None = None
    initial_sweep_steps: int | None = None
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)


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
    def create(cls, config: LocatorConfig, **kwargs):
        """Create a fresh locator instance with fresh BeliefSignal.

        This is the factory method that subclasses implement to create
        locators with properly initialized beliefs (uniform priors, etc).
        Called once per repeat.

        Parameters
        ----------
        config : LocatorConfig
            Configuration bundle for the locator.
        **kwargs
            Additional parameters specific to each locator subclass.

        Returns
        -------
        Locator
            New locator with initialized belief (uniform prior)

        Examples
        --------
        >>> cfg = LocatorConfig(max_steps=50)
        >>> locator = SimpleSweepLocator.create(config=cfg)
        >>> cfg2 = LocatorConfig(max_steps=150, convergence=ConvergenceConfig(threshold=0.01))
        >>> locator = BayesianLocator.create(config=cfg2, acquisition="eig")
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
