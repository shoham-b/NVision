"""Utility-sampling Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from nvision.signal.abstract_belief import AbstractBeliefDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class UtilitySamplingLocator(SequentialBayesianLocator):
    """Utility sampling with pickiness.

    ``Utility(x) = Var_params(x) / sigma_noise^2 / cost``

    Next setting sampled with probability ``~ Utility(x)^pickiness``.
    """

    def __init__(
        self,
        belief: AbstractBeliefDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        pickiness: float = 4.0,
        noise_std: float = 0.02,
        cost: float = 1.0,
        n_mc_samples: int = 64,
        n_candidates: int = 64,
    ) -> None:
        super().__init__(belief, max_steps, convergence_threshold, scan_param)
        self.pickiness = float(max(0.0, pickiness))
        self.noise_std = float(max(1e-9, noise_std))
        self.cost = float(max(1e-9, cost))
        self.n_mc_samples = int(max(8, n_mc_samples))
        self.n_candidates = int(max(8, n_candidates))

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractBeliefDistribution],
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        pickiness: float = 4.0,
        noise_std: float = 0.02,
        cost: float = 1.0,
        n_mc_samples: int = 64,
        n_candidates: int = 64,
        **grid_config: object,
    ) -> UtilitySamplingLocator:
        if builder is None:
            raise ValueError("UtilitySamplingLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            pickiness=pickiness,
            noise_std=noise_std,
            cost=cost,
            n_mc_samples=n_mc_samples,
            n_candidates=n_candidates,
        )

    def _acquire(self) -> float:
        candidates = np.linspace(*self.belief.get_param(self._scan_param).bounds, self.n_candidates)
        sampled = self.belief.sample(self.n_mc_samples)

        utilities = np.zeros(len(candidates))
        noise_var = self.noise_std**2

        for i, x_setting in enumerate(candidates):
            y_samples = self.belief.model.compute_vectorized(float(x_setting), sampled)
            utilities[i] = max(float(np.var(y_samples)) / noise_var / self.cost, 0.0)

        utilities += 1e-12
        probs = utilities**self.pickiness
        probs /= probs.sum()

        chosen = float(candidates[int(np.random.choice(len(candidates), p=probs))])
        return chosen
