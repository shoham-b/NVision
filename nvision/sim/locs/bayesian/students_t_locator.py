"""Parametric Student's t Locator."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from dotenv import load_dotenv

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
from nvision.belief.unit_cube_students_t_marginal import UnitCubeStudentsTMixtureMarginalDistribution
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

# --- Environment-driven defaults ---------------------------------------------

load_dotenv()

NVISION_STUDENTS_T_NUM_EXPERTS: int = int(os.getenv("NVISION_STUDENTS_T_NUM_EXPERTS", "3"))


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
        n_components: int = NVISION_STUDENTS_T_NUM_EXPERTS,
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

        bounds_phys = dict(parameter_bounds) if parameter_bounds else {}
        from nvision.spectra.unit_cube import UnitCubeSignalModel

        # Determine frequency bounds for the UnitCube mapping
        freq_bounds_phys = bounds_phys.get("frequency", (2.6e9, 3.1e9))

        # Wrap the physical model in a UnitCubeSignalModel
        model_norm = UnitCubeSignalModel(model, bounds_phys, freq_bounds_phys)

        # Create the UnitCube belief for normalized simulation
        # It will initialize its internal means to 0.5 (normalized) automatically.
        belief_norm = UnitCubeStudentsTMixtureMarginalDistribution(
            model=model_norm,
            n_components=n_components,
            _physical_param_bounds=bounds_phys,
            _physical_x_bounds=freq_bounds_phys,
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

    def _generate_candidates(self, num_candidates: int | None = None) -> np.ndarray:
        """Generate slope-targeted candidates from the mixture component means.

        Builds a dense local grid around each of the 6 signal slopes (center ± linewidth
        for each of the 3 dips) for every mixture component, merged with a global
        background grid spanning the full acquisition window. This mirrors the SBED
        epoch grid strategy and ensures the EIG has informative candidates near all
        signal features, not just the steepest one.
        """
        lo, hi = self._acquisition_bounds()
        if hi <= lo:
            return np.array([lo])

        domain = hi - lo
        estimates = self.belief.estimates()
        uncertainties = self.belief.uncertainty()

        f_b = estimates.get("frequency", (lo + hi) / 2)
        split = estimates.get("split", domain * 0.005)
        lw = estimates.get("linewidth", domain * 0.001)
        sigma_f = float(uncertainties.get("frequency", domain * 0.1))
        sigma_lw = float(uncertainties.get("linewidth", lw * 0.5))

    def _acquisition_bounds_phys(self) -> tuple[float, float]:
        """Return the focus window in physical frequency [Hz]."""
        return float(self._acquisition_lo), float(self._acquisition_hi)

    def _generate_candidates_phys(self) -> np.ndarray:
        """Return a slope-targeted candidate grid in Hz."""
        lo_phys, hi_phys = self._acquisition_bounds_phys()

        # Heuristic: 1000 points over the acquisition window
        xs_phys = np.linspace(lo_phys, hi_phys, 1000)

        # Target regions of high slope for EIG acquisition
        estimates_phys = self.belief.estimates()
        freq_est_phys = estimates_phys.get("frequency", (lo_phys + hi_phys) / 2.0)
        lw_est_phys = estimates_phys.get("linewidth", 5e6)

        # Add high-density points around the resonance(s)
        resonance_points_phys = np.linspace(freq_est_phys - 2 * lw_est_phys, freq_est_phys + 2 * lw_est_phys, 200)
        candidates_phys = np.unique(np.sort(np.concatenate([xs_phys, resonance_points_phys])))

        # Clip to acquisition window
        mask = (candidates_phys >= lo_phys) & (candidates_phys <= hi_phys)
        return candidates_phys[mask].astype(np.float64)

    def next(self) -> float:
        """Propose next measurement in normalized [0, 1] units."""
        return super().next()

    def _acquire(self) -> float:
        """Bayesian acquisition: propose next position in physical Hz via maximum EIG."""
        candidates_phys = self._generate_candidates_phys()

        # expected_information_gain_batch expects physical Hz candidates
        eigs = self.belief.expected_information_gain_batch(candidates_phys)

        best_idx = int(np.argmax(eigs))
        return float(candidates_phys[best_idx])
