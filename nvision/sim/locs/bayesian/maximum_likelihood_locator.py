"""Maximum Likelihood Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import LocatorConfig
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class MaximumLikelihoodLocator(SequentialBayesianLocator):
    """Maximum Likelihood (Mode) acquisition.

    Measures where the marginal posterior distribution is maximized.
    """

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        scan_param: str | None = None,
    ) -> None:
        super().__init__(
            belief=belief,
            config=config,
            scan_param=scan_param,
        )

    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        **grid_config: object,
    ) -> MaximumLikelihoodLocator:
        if builder is None:
            raise ValueError("MaximumLikelihoodLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief=belief,
            config=config,
            scan_param=scan_param,
        )
