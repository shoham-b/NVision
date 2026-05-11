"""Extended Kalman Filter (EKF) Locator using Gaussian Mixtures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import os
import numpy as np
from dotenv import load_dotenv

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
from nvision.belief.unit_cube_gaussian_marginal import UnitCubeGaussianMixtureMarginalDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator


# --- Environment-driven defaults ---------------------------------------------

load_dotenv()

NVISION_GAUSSIAN_NUM_EXPERTS: int = int(os.getenv("NVISION_GAUSSIAN_NUM_EXPERTS", "3"))


class EKFLocator(SequentialBayesianLocator):
    """ARCHIVED: Currently not used in the main simulation grid.
    
    Parametric Bayesian Locator using Gaussian Mixture with EKF updates.

    Acquisition uses analytical EIG from the mixture predictive variance.
    """

    REQUIRES_BELIEF = True

    def __init__(
        self,
        belief: GaussianMixtureMarginalDistribution,
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
        self.belief: GaussianMixtureMarginalDistribution = belief

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
        n_components: int = NVISION_GAUSSIAN_NUM_EXPERTS,
        **grid_config: object,
    ) -> EKFLocator:
        model = signal_model
        if model is None:
            if builder is not None:
                dummy_belief = builder(parameter_bounds, **grid_config)
                model = dummy_belief.model
            else:
                raise ValueError("EKFLocator requires either signal_model or a builder.")

        from nvision.spectra.nv_center import NVCenterLorentzianModel
        if not isinstance(model, NVCenterLorentzianModel):
            raise ValueError(f"EKFLocator only supports NVCenterLorentzianModel, got {type(model).__name__}")

        bounds_phys = dict(parameter_bounds) if parameter_bounds else {}
        from nvision.spectra.unit_cube import UnitCubeSignalModel
        
        freq_bounds_phys = bounds_phys.get("frequency", (2.6e9, 3.1e9))
        model_norm = UnitCubeSignalModel(model, bounds_phys, freq_bounds_phys)
        
        belief_norm = UnitCubeGaussianMixtureMarginalDistribution(
            model=model_norm,
            n_components=n_components,
            _physical_param_bounds=bounds_phys,
            _physical_x_bounds=freq_bounds_phys
        )

        return cls(
            belief=belief_norm,
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

    def _generate_candidates_phys(self) -> np.ndarray:
        """Return a slope-targeted candidate grid in Hz."""
        lo_phys, hi_phys = float(self._acquisition_lo), float(self._acquisition_hi)
        xs_phys = np.linspace(lo_phys, hi_phys, 1000)
        
        estimates_phys = self.belief.estimates()
        freq_est_phys = estimates_phys.get("frequency", (lo_phys + hi_phys) / 2.0)
        lw_est_phys = estimates_phys.get("linewidth", 5e6)
        
        resonance_points_phys = np.linspace(freq_est_phys - 2*lw_est_phys, freq_est_phys + 2*lw_est_phys, 200)
        candidates_phys = np.unique(np.sort(np.concatenate([xs_phys, resonance_points_phys])))
        
        mask = (candidates_phys >= lo_phys) & (candidates_phys <= hi_phys)
        return candidates_phys[mask].astype(np.float64)

    def _acquire(self) -> float:
        """Maximum EIG acquisition."""
        candidates_phys = self._generate_candidates_phys()
        eigs = self.belief.expected_information_gain_batch(candidates_phys)
        best_idx = int(np.argmax(eigs))
        return float(candidates_phys[best_idx])
