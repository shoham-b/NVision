from collections.abc import Callable, Sequence

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
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
        signal_model=None,
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
        **grid_config: object,
    ) -> "GaussianMixtureLocator":
        # We enforce the parametric belief here, so we extract the model and use it
        model = signal_model
        if model is None:
            if builder is not None:
                # Create a dummy belief just to extract the model
                dummy_belief = builder(parameter_bounds, **grid_config)
                model = dummy_belief.model
            else:
                raise ValueError("GaussianMixtureLocator requires either signal_model or a builder.")

        bounds_phys = prepare_ekf_parameter_bounds(
            dict(parameter_bounds) if parameter_bounds else None,
        )

        from nvision.spectra.unit_cube import UnitCubeSignalModel
        is_unit_cube = isinstance(model, UnitCubeSignalModel) or "UnitCube" in type(model).__name__

        if is_unit_cube:
            from nvision.belief.unit_cube_gaussian_marginal import UnitCubeGaussianMixtureMarginalDistribution
            belief = UnitCubeGaussianMixtureMarginalDistribution(
                model=model,
                n_components=n_components,
                _physical_param_bounds=bounds_phys,
                _physical_x_bounds=(bounds_phys.get("frequency") or (2.6e9, 3.1e9)),
            )
        else:
            belief = GaussianMixtureMarginalDistribution(
                model=model,
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
        """Acquire next probe by maximising EIG over the full acquisition range.

        The previous Thompson-sampling ±2σ window approach caused a systematic
        dead zone: it restricted the search to near each component's mean, where
        the Lorentzian gradient ∂y/∂freq = 0 by symmetry.  As a result the EKF
        mean update δμ ≈ solve(P, J·r/σ²) was also near-zero regardless of the
        residual, so no component ever moved toward the true dip.

        Full-range EIG evaluation fixes this because:

        1. **Gradient-variance term** ``gk @ cov[k] @ gk`` is maximised at the
           Lorentzian inflection points ``freq_k ± Γ_k/√3``, where ∂y/∂freq is
           largest, giving a strong mean-update signal.
        2. **Model-disagreement term** ``(y_pred[k] − y_mix)²`` is maximised
           where components predict different values, naturally driving
           exploration toward ambiguous regions including the true dip.

        Together these terms guide measurement toward the true resonance far
        more reliably than the restricted-window approach.
        """
        lo, hi = self._acquisition_bounds()
        belief: GaussianMixtureMarginalDistribution = self.belief  # type: ignore[assignment]

        # Use enough candidates to resolve narrow Lorentzian inflection peaks.
        # At 1 MHz linewidth over a 500 MHz range we need ~1000+ points.
        candidates_hz = np.linspace(lo, hi, 1000)
        eigs = belief.expected_information_gain_batch(candidates_hz)

        eig_max = float(np.max(eigs))
        if eig_max < 1e-12:
            # Degenerate belief (all components collapsed): explore uniformly.
            return float(np.random.uniform(lo, hi))

        # Softmax sampling with a mild temperature = 10% of the EIG range.
        # Pure argmax always picks the same point, accumulating an identical
        # rank-1 delta_prec each step and causing precision-matrix collapse
        # in the J-direction.  Softmax keeps the probe near the optimum while
        # varying the exact position, maintaining a full-rank information matrix.
        temperature = max(eig_max * 0.10, 1e-12)
        logits = (eigs - eig_max) / temperature  # shift so max logit = 0
        probs = np.exp(logits)
        probs /= probs.sum()
        best_idx = int(np.random.choice(len(candidates_hz), p=probs))
        return float(candidates_hz[best_idx])



    def observe(self, obs: Observation) -> None:
        """Route observation, denormalizing the x coordinate if physical belief."""
        if getattr(self.belief, "_is_unit_cube", False):
            self.belief.update(obs)
            return

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
