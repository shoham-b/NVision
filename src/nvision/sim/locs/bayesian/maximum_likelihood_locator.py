"""Maximum Likelihood Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.belief.abstract_belief import AbstractBeliefDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class MaximumLikelihoodLocator(SequentialBayesianLocator):
    """Maximum Likelihood (Mode) acquisition.

    Measures where the marginal posterior distribution is maximized.
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
        exploration_rate: float = 100.0,
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
        self.exploration_rate = max(0.0, float(exploration_rate))

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
        exploration_rate: float = 8.0,
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
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            exploration_rate=exploration_rate,
        )

    def _acquire(self) -> float:
        """Sample from a softmax over the marginal with decreasing exploration.

        Early in the run, we use a flatter temperature so the locator explores
        multiple plausible modes of the marginal. As `inference_step_count`
        approaches `max_steps`, the temperature sharpens and the policy becomes
        close to greedy argmax over the marginal.
        """
        candidates = np.linspace(*self._acquisition_bounds(), 200)
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)

        # Guard against numerical issues / empty mass.
        pdf = np.asarray(pdf, dtype=float)
        total = float(np.sum(pdf))
        if not np.isfinite(total) or total <= 0.0:
            # Fall back to uniform over candidates.
            return float(np.random.choice(candidates))

        # Normalized discrete marginal.
        base_prob = pdf / total

        # Annealed softmax exponent: tau starts near 1.0 (exploratory) and
        # increases towards (1 + alpha) over the budget, concentrating mass.
        frac = 0.0
        if self.max_steps > 0:
            frac = min(1.0, max(0.0, self.inference_step_count / float(self.max_steps)))
        alpha = self.exploration_rate  # controls how sharp we get by the end
        tau = 1.0 + alpha * frac

        logits = np.power(base_prob, tau)
        logits_sum = float(np.sum(logits))
        # If annealing produced degenerate weights, fall back to base_prob.
        probs = base_prob if (not np.isfinite(logits_sum) or logits_sum <= 0.0) else logits / logits_sum

        idx = int(np.random.choice(len(candidates), p=probs))
        return float(candidates[idx])
