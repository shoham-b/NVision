"""Parametric Student's t Locator."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


class StudentsTLocator(SequentialBayesianLocator):
    """Parametric Bayesian Locator using Student's t Mixture.

    Performs online Bayesian updates using a linearized conditionally conjugate 
    mixture-of-experts approach. Acquisition uses analytical EIG from the 
    mixture predictive variance.
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
        n_components: int = 3,
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

        # Ensure we are only running on Lorentzian NV center as requested.
        from nvision.spectra.nv_center import NVCenterLorentzianModel
        if not isinstance(model, NVCenterLorentzianModel):
            raise ValueError(f"StudentsTLocator only supports NVCenterLorentzianModel, got {type(model).__name__}")

        bounds = dict(parameter_bounds) if parameter_bounds else {}
        belief = StudentsTMixtureMarginalDistribution(
            model=model, 
            _physical_param_bounds=bounds, 
            n_components=n_components
        )

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
        )

    def _generate_candidates(self, num_candidates: int | None = None) -> np.ndarray:
        """Generate candidates spanning the acquisition window.
        
        Resolution is chosen to be efficient while still sampling the dips.
        """
        if num_candidates is None:
            lo, hi = self._acquisition_bounds()
            range_val = float(hi - lo)
            if range_val <= 0:
                num_candidates = 1
            else:
                # 250 kHz resolution for grid (max 2000 points)
                resolution = 250000.0 
                num_candidates = min(2000, max(100, math.ceil(range_val / resolution))) + 1
        return super()._generate_candidates(num_candidates)

    def _acquire(self) -> float:
        """Find next probe position by maximizing analytical EIG over a grid."""
        candidates = self._generate_candidates()
        
        # Compute analytical EIG for all candidates in one vectorized call
        eigs = self.belief.expected_information_gain_batch(candidates)
        
        best_idx = np.argmax(eigs)
        return float(candidates[best_idx])
