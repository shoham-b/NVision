"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math

import numpy as np

from nvision.belief.smc_marginal import _inverse_sum_squares
from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Uses Expected Information Gain (prediction variance disagreement) to select
    the next measurement point from a fine frequency grid. No JAX gradient ascent
    is performed — the chunked EIG search on the belief is sufficient.
    """

    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        noise_std: float = 0.02,
        n_candidates: int | None = None,
    ) -> None:
        super().__init__(
            belief,
            max_steps,
            convergence_threshold,
            scan_param,
            noise_std=noise_std,
        )
        self.n_candidates = int(n_candidates) if n_candidates is not None else None

        # We handle resampling manually to check convergence at the right moment
        if hasattr(self.belief, "auto_resample"):
            self.belief.auto_resample = False
        self._is_converged = False

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds=None,
        noise_std: float | None = None,
        n_candidates: int | None = None,
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
        """Select the next measurement point by maximizing EIG over a frequency grid."""
        lo, hi = self._acquisition_bounds()
        if hi <= lo:
            return float(lo)

        # Retrieve candidates directly from the belief (slope-targeted epoch grid)
        # But if the locator defines an overriding `_generate_candidates` and specifies
        # `n_candidates`, use it.
        candidates = self._generate_candidates(self.n_candidates)
        best = self.belief.select_max_information_gain(candidates, 1, noise_std=self._noise_std)
        eig_choice = float(best[0]) if len(best) > 0 else float(candidates[len(candidates) // 2])

        # Mix EIG with Thompson sampling to prevent sample impoverishment gaps
        # 20% of the time, explore directly on a sampled particle's frequency
        if hasattr(self.belief, "_particles") and hasattr(self.belief, "_weights") and np.random.rand() < 0.2:
            weights = self.belief._weights
            if np.sum(weights) > 0:
                idx = int(np.random.choice(len(weights), p=weights))
                if "frequency" in getattr(self.belief, "_param_names", []):
                    f_idx = self.belief._param_names.index("frequency")
                    val = float(self.belief._particles[idx, f_idx])
                    if hasattr(self.belief, "_to_physical"):
                        val = float(self.belief._to_physical("frequency", val))
                    return val

        return eig_choice

    def _observe_acquisition(self, obs: Observation) -> None:
        """Handle acquisition observations and manually trigger resample checks."""
        super()._observe_acquisition(obs)
        self._check_and_resample(check_convergence=True)

    def _check_and_resample(self, check_convergence: bool = True) -> None:
        if not hasattr(self.belief, "_weights"):
            return
        ess = _inverse_sum_squares(self.belief._weights)
        ess_threshold = getattr(self.belief, "ess_threshold", 0.0) * getattr(self.belief, "num_particles", 0)
        if ess < ess_threshold:
            if hasattr(self.belief, "_resample"):
                self.belief._resample()

            if check_convergence and self._target_params_converged():
                self._is_converged = True

    def _acquisition_done(self) -> bool:
        if self._is_converged:
            return True
        return super()._acquisition_done()
