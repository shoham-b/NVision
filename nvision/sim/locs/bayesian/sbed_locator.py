"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


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
        initial_sweep_steps: int | None = None,
        noise_std: float = 0.02,
        n_candidates: int | None = None,
        n_draws: int = 100,
        n_restarts: int = 8,
        n_opt_steps: int = 30,
    ) -> None:
        super().__init__(
            belief,
            max_steps,
            convergence_threshold,
            scan_param,
            initial_sweep_steps=initial_sweep_steps,
            noise_std=noise_std,
        )
        self.n_candidates = int(n_candidates) if n_candidates is not None else None
        self.n_draws = int(n_draws)
        self.n_restarts = int(n_restarts)
        self.n_opt_steps = int(n_opt_steps)

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds=None,
        initial_sweep_steps: int | None = None,
        noise_std: float | None = None,
        n_candidates: int | None = None,
        n_draws: int = 100,
        n_restarts: int = 8,
        n_opt_steps: int = 30,
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
            initial_sweep_steps=initial_sweep_steps,
            noise_std=noise_std,
            n_candidates=n_candidates,
            n_draws=n_draws,
            n_restarts=n_restarts,
            n_opt_steps=n_opt_steps,
        )

    def _generate_candidates(self, num_candidates: int | None = None) -> np.ndarray:
        """Generate candidates spanning the whole frequency spectrum.

        When ``num_candidates`` is not provided, compute it dynamically from
        the acquisition bounds so that the step resolution is two decimal
        places finer than the order of magnitude of the range.
        """
        if num_candidates is None:
            lo, hi = self._acquisition_bounds()
            range_val = float(hi - lo)
            if range_val <= 0:
                num_candidates = 1
            else:
                magnitude = math.floor(math.log10(range_val))
                resolution = 10 ** (magnitude - 2)
                num_candidates = max(1, math.ceil(range_val / resolution)) + 1
        return super()._generate_candidates(num_candidates)

    def _acquire(self) -> float:
        debug_timing = os.environ.get("DEBUG_SBED_TIMING") == "1"

        if debug_timing:
            t_start = time.perf_counter()

        result = self._acquire_jax()

        if debug_timing:
            print(f"\nSBED Timing (_acquire JAX): {time.perf_counter() - t_start:.4f}s")

        return result

    def _acquire_jax(self) -> float:
        """Find maximum EIG via JAX multirestart gradient ascent."""
        lo, hi = self._acquisition_bounds()
        if hi <= lo:
            return float(lo)

        n_restarts = max(1, self.n_restarts)
        n_steps = max(0, self.n_opt_steps)
        step_size = (hi - lo) * 0.02
        h = max(1e-12, (hi - lo) * 1e-3)

        belief = self.belief

        def _eig_fn(x):
            # x is scalar, expected_information_gain_jax expects a 1D array of candidates.
            # Returns 1D array of gains, we take [0]
            return belief.expected_information_gain_jax(jnp.array([x]))[0]

        _grad_fn = jax.grad(_eig_fn)

        def _step(carry, _):
            x = carry
            g = _grad_fn(x)
            x_new = jnp.clip(x + step_size * g, lo, hi)
            return x_new, None

        # Coarse utility evaluation to find good starting points
        n_coarse = self.n_candidates if self.n_candidates is not None else max(100, n_restarts * 5)
        coarse_candidates = jnp.array(self._generate_candidates(n_coarse))
        coarse_eigs = jax.vmap(_eig_fn)(coarse_candidates)
        top_indices = jnp.argsort(coarse_eigs)[-n_restarts:]
        x0s = coarse_candidates[top_indices]

        def _optimize(x0):
            return jax.lax.scan(_step, x0, None, length=n_steps)[0]

        x_finals = jax.vmap(_optimize)(x0s)

        # Evaluate EIG at restarts, final points, and bounds
        x_eval = jnp.concatenate([x0s, x_finals, jnp.array([lo, hi])])
        eigs_eval = jax.vmap(_eig_fn)(x_eval)
        best_idx = int(jnp.argmax(eigs_eval))
        return float(x_eval[best_idx])
