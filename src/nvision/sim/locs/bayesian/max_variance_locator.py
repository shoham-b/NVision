"""Maximum-variance Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numba import njit

from nvision.belief.abstract_belief import AbstractBeliefDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


@njit(cache=True)
def _argmax_bernoulli_variance(prob: np.ndarray) -> int:
    best_idx = 0
    best_score = -1.0
    for i in range(prob.shape[0]):
        p = prob[i]
        score = p * (1.0 - p)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


class MaxVarianceLocator(SequentialBayesianLocator):
    """Maximum posterior-variance acquisition.

    Measures where Bernoulli-style variance ``p(1-p)`` is largest.
    """

    def __init__(
        self,
        belief: AbstractBeliefDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
    ) -> None:
        super().__init__(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
        )

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractBeliefDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        **grid_config: object,
    ) -> MaxVarianceLocator:
        if builder is None:
            raise ValueError("MaxVarianceLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
        )

    def _acquire(self) -> float:
        candidates = np.linspace(*self._acquisition_bounds(), 100)
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)
        prob = pdf / (np.sum(pdf) + 1e-12)
        best_idx = int(_argmax_bernoulli_variance(prob))
        return float(candidates[best_idx])
