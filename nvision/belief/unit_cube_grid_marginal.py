"""Grid belief with marginals on ``[0, 1]`` and physical-scale public summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.spectra.unit_cube import UnitCubeSignalModel


@dataclass
class UnitCubeGridMarginalDistribution(GridMarginalDistribution):
    """Like :class:`GridMarginalDistribution` but each marginal lives on ``[0, 1]``.

    The ``model`` must be a :class:`UnitCubeSignalModel` mapping unit coordinates to
    the inner physical model.  Internal grids and uncertainties are in normalized
    space so acquisition and ``converged()`` thresholds apply uniformly across
    parameters (e.g. ``0.01`` ≈ 1% of each parameter's range).

    :meth:`estimates` and :meth:`uncertainty` are returned in **physical** units for
    metrics, plotting, and comparison to :class:`~nvision.spectra.signal.TrueSignal`.
    """

    physical_x_bounds: tuple[float, float] = (0.0, 1.0)

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:  # type: ignore[override]
        if hasattr(self, "_physical_param_bounds"):
            return self._physical_param_bounds
        # Fallback avoids recursion into super() property which returns self.parameter_bounds
        return {p.name: p.bounds for p in self.parameters}

    @physical_param_bounds.setter
    def physical_param_bounds(self, value: dict[str, tuple[float, float]]) -> None:
        self._physical_param_bounds = value

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeGridMarginalDistribution requires a UnitCubeSignalModel")
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

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self.physical_param_bounds)

    def physical_param_grid(self, name: str) -> np.ndarray:
        """Posterior support grid for ``name`` in physical units (for plotting)."""
        p = super().get_grid_param(name)
        lo, hi = self.physical_param_bounds[name]
        return lo + p.grid * (hi - lo)

    def get_grid_param(self, name: str) -> GridParameter:
        """Return grid parameter in physical units."""
        p = super().get_grid_param(name)
        lo, hi = self.physical_param_bounds[name]
        phys_grid = lo + p.grid * (hi - lo)
        return GridParameter(name=name, bounds=(lo, hi), grid=phys_grid, posterior=p.posterior.copy())

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self.physical_param_bounds[param_name]
        u = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
        # Use the underlying unit-cube GridParameter directly.
        p = GridMarginalDistribution.get_grid_param(self, param_name)
        spacing = p.grid[1] - p.grid[0] if len(p.grid) > 1 else 1.0
        density = p.posterior / spacing
        return np.interp(u, p.grid, density, left=0.0, right=0.0)

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self.physical_param_bounds[param_name]
        u = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
        # Use the underlying unit-cube GridParameter directly.
        p = GridMarginalDistribution.get_grid_param(self, param_name)
        cdf = np.cumsum(p.posterior)
        return np.interp(u, p.grid, cdf, left=0.0, right=1.0)

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """After a coarse sweep, shrink the physical interval for ``param_name`` and remap its unit marginal.

        If ``physical_x_bounds`` matches this parameter's prior interval, the probe axis
        is narrowed the same way (single-peak and multi-peak builders where the scan
        axis equals ``param_name``). Otherwise only the parameter's physical bounds are
        updated (e.g. a secondary peak frequency while the experiment probes another axis).
        """
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

        u_a = (nl - old_lo) / w_old
        u_b = (nh - old_lo) / w_old
        u_a = float(np.clip(u_a, 0.0, 1.0))
        u_b = float(np.clip(u_b, 0.0, 1.0))
        if u_b - u_a < 1e-15:
            return

        new_params: list[GridParameter] = []
        for p in self.parameters:
            if p.name != param_name:
                new_params.append(
                    GridParameter(
                        name=p.name,
                        bounds=p.bounds,
                        grid=p.grid.copy(),
                        posterior=p.posterior.copy(),
                    )
                )
                continue

            n = len(p.grid)
            new_grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
            u_old = u_a + new_grid * (u_b - u_a)
            new_post = np.interp(u_old, p.grid, p.posterior, left=0.0, right=0.0)
            s = float(np.sum(new_post))
            if s > 1e-15:
                new_post /= s
            else:
                new_post = np.ones(n, dtype=np.float64) / n
            new_params.append(
                GridParameter(
                    name=param_name,
                    bounds=(0.0, 1.0),
                    grid=new_grid,
                    posterior=new_post,
                )
            )

        self.model.narrow_physical_interval_for_param(param_name, nl, nh, update_x_axis=sync_x)
        self.physical_param_bounds[param_name] = (nl, nh)
        if sync_x:
            self.physical_x_bounds = (nl, nh)
        self.parameters = new_params

    def copy(self) -> AbstractMarginalDistribution:
        return UnitCubeGridMarginalDistribution(
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
