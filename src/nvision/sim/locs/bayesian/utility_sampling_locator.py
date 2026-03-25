"""Utility-sampling Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numba import njit

from nvision.signal.abstract_belief import AbstractBeliefDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


@njit(cache=True)
def _utility_from_mu_preds(mu_preds: np.ndarray, inv_noise_var: float, inv_cost: float) -> np.ndarray:
    """Compute per-candidate utility from row-wise predictive variance."""
    n_candidates, n_samples = mu_preds.shape
    out = np.empty(n_candidates, dtype=np.float64)

    for i in range(n_candidates):
        # Two-pass variance for numerical stability.
        mean = 0.0
        for j in range(n_samples):
            mean += mu_preds[i, j]
        mean /= n_samples

        var = 0.0
        for j in range(n_samples):
            d = mu_preds[i, j] - mean
            var += d * d
        var /= n_samples

        u = var * inv_noise_var * inv_cost
        out[i] = u if u > 0.0 else 0.0

    return out


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

        noise_var = self.noise_std**2
        model = self.belief.model

        try:
            mu_preds = model.compute_vectorized_many(candidates, sampled)
        except AttributeError:
            # Compatibility path for models exposing only scalar-vectorized prediction.
            mu_preds = np.empty((len(candidates), self.n_mc_samples), dtype=float)
            for i, x_setting in enumerate(candidates):
                mu_preds[i, :] = model.compute_vectorized(float(x_setting), sampled)

        utilities = _utility_from_mu_preds(
            mu_preds=np.asarray(mu_preds, dtype=np.float64),
            inv_noise_var=1.0 / noise_var,
            inv_cost=1.0 / self.cost,
        )

        utilities += 1e-12
        probs = utilities**self.pickiness
        probs /= probs.sum()

        chosen = float(candidates[int(np.random.choice(len(candidates), p=probs))])
        return chosen
