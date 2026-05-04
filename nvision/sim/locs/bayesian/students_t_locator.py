"""Parametric Student's t Locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class StudentsTLocator(SequentialBayesianLocator):
    """Parametric Bayesian Locator using Student's t approximations.

    Performs fully analytical Bayesian back inference using MAP optimization
    and Laplace approximation (inverse Hessian) to update belief parameters,
    bypassing SMC particles or discrete grids.
    """

    REQUIRES_BELIEF = True

    def __init__(
        self,
        belief: StudentsTMixtureMarginalDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        df: float = 3.0,
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
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )
        self.belief: StudentsTMixtureMarginalDistribution = belief
        self.df = max(1.0, float(df))

    @classmethod
    def create(
        cls,
        signal_model=None,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        df: float = 3.0,
        **grid_config: object,
    ) -> StudentsTLocator:
        # We enforce the parametric belief here, so we extract the model and use it
        model = signal_model
        if model is None:
            if builder is not None:
                # Create a dummy belief just to extract the model
                dummy_belief = builder(parameter_bounds, **grid_config)
                model = dummy_belief.model
            else:
                raise ValueError("StudentsTLocator requires either signal_model or a builder.")

        bounds = dict(parameter_bounds) if parameter_bounds else {}
        belief = StudentsTMixtureMarginalDistribution(model=model, _physical_param_bounds=bounds, dfs=np.array([df]))

        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            df=df,
        )

    def _acquire(self) -> float:
        """Draw a candidate from the Student's t marginal for the scan parameter.

        Returns a physical (not normalized) probe position.
        """
        idx = self.belief._param_names.index(self._scan_param)
        mu = self.belief.means[0, idx]
        sigma = np.sqrt(max(self.belief.covariances[0, idx, idx], 1e-12))

        lo, hi = self.belief.physical_param_bounds.get(self._scan_param, (0.0, 1.0))

        candidate = mu + sigma * np.random.standard_t(self.df)

        # Add a small exploration chance (epsilon-greedy)
        if np.random.random() < 0.05:
            candidate = np.random.uniform(lo, hi)

        return float(np.clip(candidate, lo, hi))

