"""Parametric Student's t Locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation


class StudentsTLocator(Locator):
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
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        df: float = 3.0,
    ) -> None:
        super().__init__(belief)
        self.belief: StudentsTMixtureMarginalDistribution = belief
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold

        # Use first parameter as default scan parameter if none provided
        self._scan_param = scan_param or (
            self.belief.model.parameter_names()[0] if self.belief.model.parameter_names() else "peak_x"
        )
        self.initial_sweep_steps = initial_sweep_steps or 20
        self._initial_sweep_builder = initial_sweep_builder

        self.convergence_params = convergence_params or [self._scan_param]
        self.convergence_patience_steps = convergence_patience_steps
        self._convergence_streak = 0

        self.noise_std = noise_std
        self.df = max(1.0, float(df))

        self.step_count = 0
        self.inference_step_count = 0

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
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
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
        belief = StudentsTMixtureMarginalDistribution(
            model=model,
            _physical_param_bounds=bounds,
            dfs=np.array([df])
        )

        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            df=df,
        )

    def next(self) -> float:
        self.step_count += 1

        # 1. Initial Sweep Phase
        if self.step_count <= self.initial_sweep_steps:
            if self._initial_sweep_builder is not None:
                sweep_points = self._initial_sweep_builder(self.initial_sweep_steps)
                return float(sweep_points[self.step_count - 1])

            # Default fallback sweep: uniform grid
            lo, hi = self.belief.physical_param_bounds.get(self._scan_param, (0.0, 1.0))
            if self.initial_sweep_steps <= 1:
                return (lo + hi) / 2.0
            return float(lo + (hi - lo) * (self.step_count - 1) / (self.initial_sweep_steps - 1))

        # 2. Acquisition Phase (using the parametric belief's uncertainty)
        # "the students t is just for acquire"
        # We sample from the Student's t marginal for the scan parameter
        idx = self.belief._param_names.index(self._scan_param)
        mu = self.belief.means[0, idx]
        sigma = np.sqrt(max(self.belief.covariances[0, idx, idx], 1e-12))

        lo, hi = self.belief.physical_param_bounds.get(self._scan_param, (0.0, 1.0))

        # Draw a candidate from the Student's t distribution
        # In 1D, we can just use standard numpy t distribution
        candidate = mu + sigma * np.random.standard_t(self.df)

        # Add a small exploration chance (epsilon-greedy)
        if np.random.random() < 0.05:
            candidate = np.random.uniform(lo, hi)

        return float(np.clip(candidate, lo, hi))

    def observe(self, obs: Observation) -> None:
        # Buffer observations during the sweep, but we can also just update directly
        # since the parametric belief uses all history anyway
        super().observe(obs)

        if self.step_count > self.initial_sweep_steps:
            self.inference_step_count += 1

    def _target_params_converged(self) -> bool:
        if not self.convergence_params:
            return self.belief.converged(self.convergence_threshold)

        stds = self.belief._empirical_uncertainty()
        param_uncertainties: dict[str, float] = {}
        for p in self.convergence_params:
            if p in stds:
                param_uncertainties[p] = stds[p]

        if not param_uncertainties:
            return self.belief.converged(self.convergence_threshold)

        # Check 1: Each individual parameter must be below threshold
        individual_converged = all(u < self.convergence_threshold for u in param_uncertainties.values())
        if not individual_converged:
            return False

        # Check 2: Overall (RMS) uncertainty must also be below threshold
        uncertainties_array = np.array(list(param_uncertainties.values()))
        rms_uncertainty = float(np.sqrt(np.mean(uncertainties_array**2)))
        overall_converged = rms_uncertainty < self.convergence_threshold

        return overall_converged

    def done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if self.inference_step_count > 0:
            if self._target_params_converged():
                self._convergence_streak += 1
            else:
                self._convergence_streak = 0
            return self._convergence_streak >= self.convergence_patience_steps
        return False

    def effective_initial_sweep_steps(self) -> int:
        return min(self.step_count, self.initial_sweep_steps)

    def result(self) -> dict[str, float]:
        """Return posterior-mean estimates for all parameters."""
        return self.belief.estimates()
