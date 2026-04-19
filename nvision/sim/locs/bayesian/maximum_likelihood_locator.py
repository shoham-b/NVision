"""Maximum Likelihood Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

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
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        exploration_rate: float = 100.0,
        noise_std: float | None = None,
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
            noise_std=noise_std,
        )
        self.exploration_rate = max(0.0, float(exploration_rate))

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        exploration_rate: float = 8.0,
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
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            exploration_rate=exploration_rate,
            noise_std=noise_std,
        )

    def _acquire(self) -> float:
        """Sample from a softmax over the marginal with guaranteed exploration.

        Early in the run, we use a flatter temperature so the locator explores
        multiple plausible modes of the marginal. As `inference_step_count`
        approaches `max_steps`, the temperature sharpens. To prevent getting
        permanently stuck in false noise-induced modes, we maintain a baseline
        epsilon-exploration floor at all times.
        """
        candidates = self._generate_candidates()
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)

        # Guard against numerical issues / empty mass.
        pdf = np.asarray(pdf, dtype=float)
        total = float(np.sum(pdf))
        if not np.isfinite(total) or total <= 0.0:
            # Fall back to uniform over candidates.
            return float(np.random.choice(candidates))

        # Normalized discrete marginal.
        base_prob = pdf / total

        # Apply frequency-specificity bias: boost probabilities for candidates
        # whose signal predictions covary most with high-weight parameters.
        try:
            n_samples = 64
            # Use maximum likelihood particle selection instead of random sampling
            sampled = self.belief.select_maximum_likelihood(n_samples)
            mu_preds = self.belief.model.compute_vectorized_many(candidates, sampled)
            biased = self._apply_parameter_weight_bias(np.asarray(base_prob, dtype=float), np.asarray(mu_preds, dtype=float), sampled, candidates)
            base_prob = np.maximum(biased, 0.0)
            total = float(np.sum(base_prob))
            if total > 0:
                base_prob = base_prob / total
        except Exception:
            pass  # Fall back to unbiased marginal if sampling fails

        # Soften extreme zeroes prior to temperature scaling
        base_prob = base_prob * 0.99 + 0.01 / len(candidates)

        # Annealed softmax exponent: tau starts near 0.1 (highly exploratory/flat)
        # and increases towards alpha over the budget, concentrating mass.
        frac = 0.0
        if self.max_steps > 0:
            frac = min(1.0, max(0.0, self.inference_step_count / float(self.max_steps)))
        alpha = self.exploration_rate  # controls how sharp we get by the end
        tau = 0.1 + alpha * frac

        logits = np.power(base_prob, tau)
        logits_sum = float(np.sum(logits))

        # If annealing produced degenerate weights, fall back to base_prob.
        probs = base_prob if (not np.isfinite(logits_sum) or logits_sum <= 0.0) else logits / logits_sum

        # Add a baseline uniform floor AFTER scaling (epsilon-greedy-like).
        # Ensures that even when the locator sharply focuses on a false mode,
        # it periodically checks other areas to escape and find the true signal.
        # Higher noise defaults to a higher exploration rate (bounded 5% to 50%).
        epsilon = 0.15
        sweep_obs = getattr(self, "_staged_sobol", None)
        if sweep_obs is not None and hasattr(sweep_obs, "_sweep_observations") and sweep_obs._sweep_observations:
            epsilon = float(np.clip(sweep_obs._sweep_observations[-1].noise_std, 0.05, 0.50))

        # Decay exploration as we approach max budget (ends at 20% of starting value)
        # Allows robust early exploration but tight exploitation later.
        epsilon = epsilon * (1.0 - 0.8 * frac)

        probs = probs * (1.0 - epsilon) + epsilon / len(candidates)

        idx = int(np.random.choice(len(candidates), p=probs))
        return float(candidates[idx])
