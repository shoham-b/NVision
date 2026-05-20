import numpy as np
from collections.abc import Callable, Sequence

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
from nvision.sim.locs.ekf.parameter_bounds import prepare_ekf_parameter_bounds


class GaussianMixtureLocator(SequentialBayesianLocator):
    """Locator using Gaussian Mixture Marginal Distribution for tracking."""

    def __init__(
        self,
        belief: GaussianMixtureMarginalDistribution,
        max_steps: int = 200,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
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
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std or 0.05,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 200,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        n_components: int = 5,
        **config: object,
    ) -> "GaussianMixtureLocator":
        from nvision.spectra.nv_center import NVCenterLorentzianModel

        signal_model = NVCenterLorentzianModel()
        bounds_phys = prepare_ekf_parameter_bounds(
            dict(parameter_bounds) if parameter_bounds else None,
        )

        belief = GaussianMixtureMarginalDistribution(
            model=signal_model,
            n_components=n_components,
            _physical_param_bounds=bounds_phys,
        )

        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )

    def _acquire(self) -> float:
        """Run OED acquisition maximizing expected information gain."""
        lo, hi = self._acquisition_bounds()

        # Generate candidates along physical scan axis
        candidates_hz = np.linspace(lo, hi, 1000)

        # Calculate EIG
        eigs = self.belief.expected_information_gain_batch(candidates_hz)

        # Select maximum EIG
        best_idx = np.argmax(eigs)
        return float(candidates_hz[best_idx])

    def observe(self, obs: Observation) -> None:
        """Route observation, denormalizing the x coordinate."""
        lo, hi = self._acquisition_bounds()
        x_hz = lo + obs.x * (hi - lo)

        obs_physical = Observation(
            x=x_hz,
            signal_value=obs.signal_value,
            noise_std=obs.noise_std,
            frequency_noise_model=obs.frequency_noise_model,
        )

        self.belief.update(obs_physical)

    def _on_sweep_complete(self) -> None:
        """Narrow bounds after initial sweep."""
        lo, hi = self._acquisition_bounds()
        if self._scan_param:
            self.belief.narrow_scan_parameter_physical_bounds(self._scan_param, lo, hi)
