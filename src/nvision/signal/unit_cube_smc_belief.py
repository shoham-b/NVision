"""SMC belief with particles on ``[0, 1]`` and physical-scale public summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nvision.signal.abstract_belief import ParameterValues
from nvision.signal.signal import Parameter
from nvision.signal.smc_belief import SMCBeliefDistribution
from nvision.signal.unit_cube_model import UnitCubeSignalModel


@dataclass
class UnitCubeSMCBeliefDistribution(SMCBeliefDistribution):
    """Like :class:`SMCBeliefDistribution` but particles live on ``[0, 1]``.

    The ``model`` must be a :class:`UnitCubeSignalModel` mapping unit coordinates to
    the inner physical model. Internal particles and uncertainties are in normalized
    space so acquisition and ``converged()`` thresholds apply uniformly across
    parameters.

    :meth:`estimates` and :meth:`uncertainty` are returned in **physical** units for
    metrics, plotting, and comparison.
    """

    physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    physical_x_bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeSMCBeliefDistribution requires a UnitCubeSignalModel")
        super().__post_init__()

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

    def get_param(self, name: str) -> Parameter:
        p = super().get_param(name)
        lo, hi = self.physical_param_bounds[name]
        return Parameter(name=name, bounds=(lo, hi), value=lo + float(p.value) * (hi - lo))

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        return super().sample(n)

    def copy(self) -> UnitCubeSMCBeliefDistribution:
        dist = UnitCubeSMCBeliefDistribution(
            model=self.model,
            parameter_bounds=self.parameter_bounds.copy(),
            num_particles=self.num_particles,
            jitter_scale=self.jitter_scale,
            ess_threshold=self.ess_threshold,
            last_obs=self.last_obs,
            physical_param_bounds=dict(self.physical_param_bounds),
            physical_x_bounds=self.physical_x_bounds,
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy()
        dist._weights = self._weights.copy()
        dist._init_param_scratch()
        return dist
