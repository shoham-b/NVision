"""Maximum Likelihood Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class MaximumLikelihoodLocator(SequentialBayesianLocator):
    """Maximum Likelihood (Mode) acquisition.

    Measures where the marginal posterior distribution is maximized.
    """

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
    ) -> None:
        super().__init__(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
        )

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        **grid_config: object,
    ) -> MaximumLikelihoodLocator:
        if builder is None:
            raise ValueError("MaximumLikelihoodLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
        )

