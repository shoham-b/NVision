"""Grid belief with marginals on ``[0, 1]`` and physical-scale public summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nvision.signal.abstract_belief import AbstractBeliefDistribution, ParameterValues
from nvision.signal.grid_belief import GridBeliefDistribution, GridParameter
from nvision.signal.signal import Parameter
from nvision.signal.unit_cube_model import UnitCubeSignalModel


@dataclass
class UnitCubeGridBeliefDistribution(GridBeliefDistribution):
    """Like :class:`GridBeliefDistribution` but each marginal lives on ``[0, 1]``.

    The ``model`` must be a :class:`UnitCubeSignalModel` mapping unit coordinates to
    the inner physical model.  Internal grids and uncertainties are in normalized
    space so acquisition and ``converged()`` thresholds apply uniformly across
    parameters (e.g. ``0.01`` ≈ 1% of each parameter's range).

    :meth:`estimates` and :meth:`uncertainty` are returned in **physical** units for
    metrics, plotting, and comparison to :class:`~nvision.signal.signal.TrueSignal`.
    """

    physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    physical_x_bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeGridBeliefDistribution requires a UnitCubeSignalModel")
        super().__post_init__()

    def estimates(self) -> dict[str, float]:
        raw = {p.name: p.mean() for p in self.parameters}
        return {k: self._to_physical(k, v) for k, v in raw.items()}

    def _to_physical(self, name: str, u: float) -> float:
        lo, hi = self.physical_param_bounds[name]
        return lo + float(u) * (hi - lo)

    def uncertainty(self) -> ParameterValues[float]:
        data = {
            p.name: p.uncertainty() * (self.physical_param_bounds[p.name][1] - self.physical_param_bounds[p.name][0])
            for p in self.parameters
        }
        return ParameterValues.from_mapping(self.model.parameter_names(), data)

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        return super().sample(n)

    def converged(self, threshold: float) -> bool:
        return all(p.uncertainty() < threshold for p in self.parameters)

    def physical_param_grid(self, name: str) -> np.ndarray:
        """Posterior support grid for ``name`` in physical units (for plotting)."""
        p = super().get_param(name)
        lo, hi = self.physical_param_bounds[name]
        return lo + p.grid * (hi - lo)

    def get_param(self, name: str) -> Parameter:
        """Return parameter metadata in physical units for acquisition callers."""
        p = super().get_param(name)
        lo, hi = self.physical_param_bounds[name]
        return Parameter(name=name, bounds=(lo, hi), value=self._to_physical(name, p.mean()))

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self.physical_param_bounds[param_name]
        u = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
        # Use the underlying unit-cube GridParameter directly.
        p = GridBeliefDistribution.get_param(self, param_name)
        spacing = p.grid[1] - p.grid[0] if len(p.grid) > 1 else 1.0
        density = p.posterior / spacing
        return np.interp(u, p.grid, density, left=0.0, right=0.0)

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self.physical_param_bounds[param_name]
        u = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
        # Use the underlying unit-cube GridParameter directly.
        p = GridBeliefDistribution.get_param(self, param_name)
        cdf = np.cumsum(p.posterior)
        return np.interp(u, p.grid, cdf, left=0.0, right=1.0)

    def copy(self) -> AbstractBeliefDistribution:
        return UnitCubeGridBeliefDistribution(
            model=self.model,
            parameters=[
                GridParameter(
                    name=p.name,
                    bounds=p.bounds,
                    grid=p.grid.copy(),
                    posterior=p.posterior.copy(),
                )
                for p in self.parameters
            ],
            last_obs=self.last_obs,
            physical_param_bounds=dict(self.physical_param_bounds),
            physical_x_bounds=self.physical_x_bounds,
        )
