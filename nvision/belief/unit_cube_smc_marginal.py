"""SMC belief with particles on ``[0, 1]`` and physical-scale public summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nvision.belief.abstract_marginal import ParameterValues
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.spectra.unit_cube import UnitCubeSignalModel


@dataclass
class UnitCubeSMCMarginalDistribution(SMCMarginalDistribution):
    """Like :class:`SMCMarginalDistribution` but particles live on ``[0, 1]``.

    The ``model`` must be a :class:`UnitCubeSignalModel` mapping unit coordinates to
    the inner physical model. Internal particles and uncertainties are in normalized
    space so acquisition and ``converged()`` thresholds apply uniformly across
    parameters.

    :meth:`estimates` and :meth:`uncertainty` are returned in **physical** units for
    metrics, plotting, and comparison.
    """

    physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    physical_x_bounds: tuple[float, float] = (0.0, 1.0)

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:  # type: ignore[override]  # noqa: F811
        if hasattr(self, "_physical_param_bounds"):
            return self._physical_param_bounds
        return {}

    @physical_param_bounds.setter
    def physical_param_bounds(self, value: dict[str, tuple[float, float]]) -> None:
        self._physical_param_bounds = value

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeSMCMarginalDistribution requires a UnitCubeSignalModel")

        # Ensure all parameters (including noise) are in unit space [0, 1].
        # We must detect noise parameters here because super().__post_init__ 
        # expects them to be in parameter_bounds before it iterates over _param_names.
        all_names = list(self.model.parameter_names())
        if self.noise_model is not None:
            all_names.extend(n for n in self.noise_model.spec.names if n not in all_names)

        self.parameter_bounds = {name: (0.0, 1.0) for name in all_names}
        super().__post_init__()

    def expected_information_gain(self, candidates: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Override to normalize physical candidates to [0, 1] for the UnitCube model."""
        lo, hi = self.physical_x_bounds
        unit_candidates = (candidates - lo) / (hi - lo)
        return super().expected_information_gain(unit_candidates, noise_std=noise_std)

    def get_candidates(self) -> np.ndarray:
        """Return candidates in **physical** frequency space.

        The base class stores candidates in unit [0, 1] space (matching internal
        particles). The locator and ``expected_information_gain`` both operate in
        physical space, so we convert here before returning.
        """
        lo, hi = self.physical_x_bounds
        return lo + self._current_candidates.astype(np.float64) * (hi - lo)

    def _generate_epoch_candidates(self) -> None:
        """Generate candidates in unit space.
        
        The base class now uses internal _estimates_unit() which correctly
        returns [0, 1] values even for this subclass.
        """
        super()._generate_epoch_candidates()

    def estimates(self) -> dict[str, float]:
        raw = super().estimates()
        return {k: self._to_physical(k, v) for k, v in raw.items()}

    def _to_physical(self, name: str, u: float) -> float:
        lo, hi = self.physical_param_bounds[name]
        return lo + float(u) * (hi - lo)

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        raw = super()._empirical_uncertainty()
        data = {
            name: u * (self.physical_param_bounds[name][1] - self.physical_param_bounds[name][0])
            for name, u in raw.items()
        }
        return ParameterValues.from_mapping(list(raw.keys()), data)

    def uncertainty(self) -> ParameterValues[float]:
        return self._empirical_uncertainty()

    def converged(self, threshold: float) -> bool:
        # Check convergence uniformly using inner [0, 1] uncertainties
        raw_uncertainties = super()._empirical_uncertainty()
        return all(u < threshold for u in raw_uncertainties.values())

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        return super().sample(n)

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Shrink physical bounds for ``param_name`` and remap unit particles (see grid variant)."""
        if param_name not in self.physical_param_bounds:
            raise KeyError(param_name)
        old_lo, old_hi = self.physical_param_bounds[param_name]
        w_old = old_hi - old_lo
        if w_old <= 0:
            return

        nl = float(max(min(new_lo, new_hi), old_lo))
        nh = float(min(max(new_lo, new_hi), old_hi))
        if nh <= nl:
            return

        sync_x = self.physical_x_bounds == (old_lo, old_hi)
        w_new = nh - nl

        j = self._param_names.index(param_name)
        u_col = self._particles[:, j]
        f = old_lo + u_col * w_old
        u_new = (f - nl) / w_new
        self._particles[:, j] = np.clip(u_new, 0.0, 1.0)

        self.model.narrow_physical_interval_for_param(param_name, nl, nh, update_x_axis=sync_x)
        self.physical_param_bounds[param_name] = (nl, nh)
        if sync_x:
            self.physical_x_bounds = (nl, nh)

    def copy(self) -> UnitCubeSMCMarginalDistribution:
        dist = UnitCubeSMCMarginalDistribution(
            model=self.model,
            parameter_bounds=self.parameter_bounds.copy(),
            num_particles=self.num_particles,
            ess_threshold=self.ess_threshold,
            a_param=self.a_param,
            last_obs=self.last_obs,
            noise_model=self.noise_model,
            auto_resample=self.auto_resample,
            resample_delay=self.resample_delay,
            physical_param_bounds=dict(self.physical_param_bounds),
            physical_x_bounds=self.physical_x_bounds,
            priors=self.priors,
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy()
        dist._weights = self._weights.copy()
        dist._step_count = self._step_count
        dist.resampled = self.resampled
        return dist
