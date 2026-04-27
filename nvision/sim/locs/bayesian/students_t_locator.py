"""Student's t Mixture Bayesian acquisition locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import scipy.special
from numba import njit

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


@njit(cache=True)
def _students_t_log_likelihood(y: float, mu: np.ndarray, sigma: float, df: float) -> np.ndarray:
    """Compute Student's t log-likelihood for a vector of predictions.
    
    L(y | mu, sigma, df) = Gamma((df+1)/2) / (Gamma(df/2) * sqrt(pi * df * sigma^2)) * 
                           (1 + (y - mu)^2 / (df * sigma^2))^(-(df+1)/2)
    """
    n = mu.shape[0]
    out = np.empty(n, dtype=np.float64)
    
    # Precompute constants
    sigma_sq = sigma * sigma
    df_sigma_sq = df * sigma_sq
    power = -0.5 * (df + 1.0)
    
    # Log-constant part (not strictly needed for relative weights but good for rigor)
    # log_const = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(math.pi * df_sigma_sq)
    
    for i in range(n):
        diff = y - mu[i]
        # We only need the kernel part for Bayesian updates as we normalize anyway
        out[i] = power * np.log(1.0 + (diff * diff) / df_sigma_sq)
        
    return out


class StudentsTLocator(SequentialBayesianLocator):
    """Student's t Mixture acquisition.
    
    Uses a Student's t distribution as the likelihood approximation instead of 
    Gaussian. This provides heavier tails, making the acquisition more robust 
    to outliers and promoting broader exploration of the parameter space.
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
        noise_std: float | None = None,
        df: float = 3.0,
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
        self.df = max(1.0, float(df))

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
        noise_std: float | None = None,
        df: float = 3.0,
        **grid_config: object,
    ) -> StudentsTLocator:
        if builder is None:
            raise ValueError("StudentsTLocator requires a builder callable.")
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
            noise_std=noise_std,
            df=df,
        )

    def observe(self, obs: Observation) -> None:
        """Update belief using Student's t likelihood.
        
        Overrides the default Gaussian update by computing the Student's t
        likelihood and manually applying it to the belief.
        """
        if not self._staged_sobol.done():
            # During sweep phase: use base class logic (buffering)
            super().observe(obs)
            return

        # Inference phase: manual update with Student's t likelihood
        self.step_count += 1
        self.inference_step_count += 1
        self.belief.last_obs = obs

        # 1. Generate predictions for the belief (grid or particles)
        if hasattr(self.belief, "parameters"):
            # Grid belief: update each marginal independently
            param_names = self.belief.model.parameter_names()
            param_by_name = {p.name: p for p in self.belief.parameters}
            
            for param in self.belief.parameters:
                grid = np.asarray(param.grid, dtype=np.float64)
                arrays_in_order = []
                for name in param_names:
                    if name == param.name:
                        arrays_in_order.append(grid)
                    else:
                        other = param_by_name[name]
                        arrays_in_order.append(np.full(grid.shape, float(other.value), dtype=np.float64))
                
                predicted = self.belief.model.compute_vectorized(obs.x, *arrays_in_order)
                log_lik = _students_t_log_likelihood(obs.signal_value, predicted, obs.noise_std, self.df)
                
                # Convert log-likelihood to likelihood with stable normalization
                likelihoods = np.exp(log_lik - np.max(log_lik))
                param.apply_likelihood(likelihoods)
        
        elif hasattr(self.belief, "_particles"):
            # SMC belief: update particles
            arrays_in_order = [self.belief._particles[:, j] for j in range(len(self.belief._param_names))]
            predicted = self.belief.model.compute_vectorized(obs.x, *arrays_in_order)
            
            log_lik = _students_t_log_likelihood(obs.signal_value, predicted, obs.noise_std, self.df)
            
            # Combine with information weights if available (SMC-specific)
            info_weights = self.belief._compute_information_weights(
                obs.x, predicted, obs.noise_std, obs.frequency_noise_model
            )
            
            # Stable update
            self.belief._weights *= np.exp(log_lik - np.max(log_lik)) * info_weights
            
            # Normalize
            weight_sum = np.sum(self.belief._weights)
            if weight_sum > 1e-10:
                self.belief._weights /= weight_sum
            else:
                self.belief._weights = np.ones(self.belief.num_particles) / self.belief.num_particles
                
            # Annealed jitter and Resample
            if self.belief.annealed_jitter:
                self.belief._apply_annealed_jitter()
            
            ess = 1.0 / np.sum(self.belief._weights**2)
            if ess < self.belief.ess_threshold * self.belief.num_particles:
                self.belief._resample()
        else:
            # Fallback to standard update if belief structure is unknown
            super().observe(obs)

    def _acquire(self) -> float:
        """Acquire by sampling from the marginal PDF (as in MaximumLikelihoodLocator)."""
        candidates = self._generate_candidates()
        pdf = self.belief.marginal_pdf(self._scan_param, candidates)
        
        pdf = np.asarray(pdf, dtype=float)
        total = float(np.sum(pdf))
        if not np.isfinite(total) or total <= 0.0:
            return float(np.random.choice(candidates))
            
        probs = pdf / total
        
        # Add slight exploration floor
        epsilon = 0.05
        probs = probs * (1.0 - epsilon) + epsilon / len(candidates)
        
        idx = int(np.random.choice(len(candidates), p=probs))
        return float(candidates[idx])
