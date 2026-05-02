"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import abc
import math

import numpy as np
import polars as pl
from numba import njit

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class INoiseModelStrategy(abc.ABC):
    @abc.abstractmethod
    def generate_noise_chunk(self, mu_preds: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate (noise_chunk, inv_noise_std_arr) given model predictions."""
        pass


class GaussianNoiseStrategy(INoiseModelStrategy):
    def __init__(self, noise_std: float):
        self.noise_std = noise_std

    def generate_noise_chunk(self, mu_preds: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        m = mu_preds.shape[0]
        noise_chunk = np.random.normal(0.0, self.noise_std, size=(m, num_samples))
        inv_noise_std_arr = np.full((m, num_samples), 1.0 / self.noise_std)
        return noise_chunk, inv_noise_std_arr


class PoissonNoiseStrategy(INoiseModelStrategy):
    def __init__(self, poisson_scale: float):
        self.poisson_scale = poisson_scale

    def generate_noise_chunk(self, mu_preds: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        m = mu_preds.shape[0]
        lam = np.maximum(mu_preds, 1e-12) * self.poisson_scale
        std_y = np.sqrt(lam) / self.poisson_scale
        noise_chunk = np.random.normal(0.0, 1.0, size=(m, num_samples)) * std_y
        inv_noise_std_arr = 1.0 / std_y
        return noise_chunk, inv_noise_std_arr


class NoiseModelFactory:
    @staticmethod
    def create(last_obs, default_noise_std: float) -> INoiseModelStrategy:
        noise_std = last_obs.noise_std if last_obs is not None else default_noise_std
        freq_model = getattr(last_obs, "frequency_noise_model", None) if last_obs is not None else None

        use_poisson = bool(
            freq_model
            and len(freq_model) == 1
            and freq_model[0].get("type") == "poisson"
            and float(freq_model[0].get("scale", 0.0)) > 0.0
        )

        if use_poisson:
            poisson_scale = float(freq_model[0]["scale"])
            return PoissonNoiseStrategy(poisson_scale)
        return GaussianNoiseStrategy(noise_std)


@njit(cache=True)
def _sbed_eig_utilities_from_mu_and_noise(
    mu_preds: np.ndarray,
    noise_chunk: np.ndarray,
    inv_noise_std: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    Compute the SBED utility for a chunk of candidates.

    ``inv_noise_std`` is a 2-D array (m, n_outcomes) so that heteroscedastic
    noise (e.g. Poisson) can use the correct per-outcome precision.

    Notes
    -----
    This mirrors the original NumPy tensor math but avoids materializing the
    full (m, n_outcomes, n_particles) intermediate via explicit loops.
    """
    m, n_particles = mu_preds.shape
    _, n_outcomes = noise_chunk.shape

    log_eps = math.log(eps)
    utilities = np.empty(m, dtype=np.float64)
    buffer = np.empty(n_particles, dtype=np.float64)

    for i in range(m):
        acc_entropy = 0.0

        # For each hypothetical outcome o, form the discrete posterior over
        # particles and compute its Shannon entropy.
        for o in range(n_outcomes):
            y = mu_preds[i, o] + noise_chunk[i, o]
            inv_sigma = inv_noise_std[i, o]
            inv_var_half = -0.5 * inv_sigma * inv_sigma

            # Stable normalization: subtract max log-likelihood over particles.
            max_log_lik = -1e300
            for p in range(n_particles):
                diff = y - mu_preds[i, p]
                ll = (diff * diff) * inv_var_half
                buffer[p] = ll
                if ll > max_log_lik:
                    max_log_lik = ll

            sum_exp = 0.0
            for p in range(n_particles):
                e = math.exp(buffer[p] - max_log_lik)
                buffer[p] = e
                sum_exp += e

            inv_sum_exp = 1.0 / (sum_exp + 1e-300)
            entropy = 0.0

            for p in range(n_particles):
                w = buffer[p] * inv_sum_exp

                # Match np.log(np.clip(weights, eps, None)) semantics.
                if w < eps:
                    entropy += -w * log_eps
                else:
                    entropy += -w * math.log(w)

            acc_entropy += entropy

        # Utility is negative expected entropy (so argmax picks minimum entropy).
        utilities[i] = -(acc_entropy / n_outcomes)

    return utilities


class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Uses exact Expected Information Gain (posterior Shannon entropy reduction).
    """

    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        noise_std: float = 0.02,
        n_candidates: int = 200,
        n_draws: int = 100,
    ) -> None:
        super().__init__(belief, max_steps, convergence_threshold, scan_param, noise_std=noise_std)
        self.n_candidates = int(n_candidates)
        self.n_draws = int(n_draws)

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds=None,
        noise_std: float | None = None,
        n_candidates: int = 200,
        n_draws: int = 100,
        **grid_config,
    ):
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            noise_std=noise_std,
            n_candidates=n_candidates,
            n_draws=n_draws,
        )

    def _acquire(self) -> float:
        candidates = self._generate_candidates(self.n_candidates)

        num_samples = self.n_draws
        candidates_arr = np.asarray(candidates)
        sampled = self.belief.select_max_information_gain(candidates_arr, num_samples)

        noise_strategy = NoiseModelFactory.create(self.belief.last_obs, 0.05)
        model = self.belief.model

        chunk_size = 64 if len(candidates) > 64 else len(candidates)
        utilities = np.zeros(len(candidates), dtype=float)
        eps = 1e-12

        for start in range(0, len(candidates), chunk_size):
            end = min(len(candidates), start + chunk_size)
            xs = candidates[start:end]
            mu_preds = model.compute_vectorized_many(xs, sampled)

            noise_chunk, inv_noise_std_arr = noise_strategy.generate_noise_chunk(mu_preds, num_samples)

            utilities[start:end] = _sbed_eig_utilities_from_mu_and_noise(
                mu_preds=mu_preds,
                noise_chunk=noise_chunk,
                inv_noise_std=inv_noise_std_arr,
                eps=eps,
            )

        best_idx = int(pl.Series(utilities).arg_max())
        return float(candidates[best_idx])
