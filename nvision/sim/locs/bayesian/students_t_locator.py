"""Parametric Student's t Locator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
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
        n_restarts: int = 8,
        n_opt_steps: int = 30,
        n_mc_samples: int = 64,
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
        self.n_restarts = int(n_restarts)
        self.n_opt_steps = int(n_opt_steps)
        self.n_mc_samples = int(n_mc_samples)

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
        n_restarts: int = 8,
        n_opt_steps: int = 30,
        n_mc_samples: int = 64,
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
            n_restarts=n_restarts,
            n_opt_steps=n_opt_steps,
            n_mc_samples=n_mc_samples,
        )

    def _acquire(self) -> float:
        """Find next probe position via JAX multirestart optimization.

        Maximizes the predictive variance of the model output at *x* under
        the current parametric belief, using float32 throughout.

        Returns a physical (not normalized) probe position.
        """
        lo, hi = self._acquisition_bounds()
        if hi <= lo:
            return float(lo)

        # Sample parameters from the belief (float32) - captured for numpy callback
        sampled = self.belief.sample(self.n_mc_samples)
        sample_arrays = tuple(np.asarray(arr, dtype=np.float32) for arr in sampled.arrays_in_order())
        model = self.belief.model

        def _numpy_pred_variance(x: object) -> object:
            x_arr = np.asarray(x, dtype=np.float32)
            if x_arr.ndim == 0:
                x_arr = np.array([float(x_arr)], dtype=np.float32)
            preds = model.compute_vectorized_many(x_arr, sample_arrays)
            return np.var(preds, axis=1).astype(np.float32)

        def _var_fn(x):
            return jax.pure_callback(
                _numpy_pred_variance,
                jax.ShapeDtypeStruct((), jnp.float32),
                x,
                vmap_method="expand_dims",
            )

        n_restarts = max(1, self.n_restarts)
        n_steps = max(0, self.n_opt_steps)
        step_size = (hi - lo) * 0.02
        h = max(1e-12, (hi - lo) * 1e-3)

        def _grad_fn(x):
            return (_var_fn(x + h) - _var_fn(x - h)) / (2 * h)

        def _step(carry, _):
            x = carry
            g = _grad_fn(x)
            x_new = jnp.clip(x + step_size * g, lo, hi)
            return x_new, None

        def _optimize(x0):
            return jax.lax.scan(_step, x0, None, length=n_steps)[0]

        key = jax.random.PRNGKey(self.inference_step_count)
        x0s = jax.random.uniform(key, shape=(n_restarts,), minval=lo, maxval=hi)

        x_finals = jax.vmap(_optimize)(x0s)

        # Evaluate variance at restarts, final points, and bounds
        x_eval = jnp.concatenate([x0s, x_finals, jnp.array([lo, hi])])
        vars_eval = jax.vmap(_var_fn)(x_eval)
        best_idx = int(jnp.argmax(vars_eval))
        return float(x_eval[best_idx])
