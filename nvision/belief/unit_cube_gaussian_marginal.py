"""Gaussian Mixture belief with normalized unit-cube parameters and physical-scale public summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nvision.belief.abstract_marginal import ParameterValues
from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
from nvision.models.observation import Observation
from nvision.spectra.unit_cube import UnitCubeSignalModel


@dataclass
class UnitCubeGaussianMixtureMarginalDistribution(GaussianMixtureMarginalDistribution):
    """Like :class:`GaussianMixtureMarginalDistribution` but internal state is on ``[0, 1]``.

    Internal means and uncertainties are in normalized space.
    :meth:`estimates` and :meth:`uncertainty` are returned in **physical** units.
    """

    _physical_x_bounds: tuple[float, float] = (0.0, 1.0)

    @property
    def physical_x_bounds(self) -> tuple[float, float]:
        return self._physical_x_bounds

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeGaussianMixtureMarginalDistribution requires a UnitCubeSignalModel")
        super().__post_init__()

    def update(self, obs_norm: Observation) -> None:
        """Update with a normalized observation."""
        self.last_obs = obs_norm
        self._update_mixtures(obs_norm.x, obs_norm.signal_value, obs_norm.noise_std)
        self._recompute_covariances()

    def batch_update(self, observations_norm: Sequence[Observation]) -> None:
        """Batch update with normalized observations."""
        if not observations_norm:
            return
        for obs in observations_norm:
            self._update_mixtures(obs.x, obs.signal_value, obs.noise_std)
        self.last_obs = observations_norm[-1]
        self._recompute_covariances()

    def expected_information_gain_batch(self, xs_phys: np.ndarray) -> np.ndarray:
        """Override to normalize physical candidates to [0, 1] for the UnitCube model."""
        lo, hi = self._physical_x_bounds
        xs_norm = (xs_phys - lo) / (hi - lo + 1e-12)
        return super().expected_information_gain_batch(xs_norm)

    def estimates(self) -> dict[str, float]:
        raw_norm = super().estimates()
        return {k: self._to_physical(k, v) for k, v in raw_norm.items()}

    def _to_physical(self, name: str, u_norm: float) -> float:
        lo, hi = self.physical_param_bounds[name]
        return lo + float(u_norm) * (hi - lo)

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        raw_norm = super()._empirical_uncertainty()
        data_phys = {
            name: u_norm * (self.physical_param_bounds[name][1] - self.physical_param_bounds[name][0])
            for name, u_norm in raw_norm.items()
        }
        return ParameterValues.from_mapping(list(raw_norm.keys()), data_phys)

    def uncertainty(self) -> ParameterValues[float]:
        return self._empirical_uncertainty()

    def converged(self, threshold: float) -> bool:
        raw_uncertainties_norm = super()._empirical_uncertainty()
        return all(u < threshold for u in raw_uncertainties_norm.values())

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        raw_norm = super().sample(n)
        data_phys = {
            name: self._to_physical(name, u_norm_arr)
            for name, u_norm_arr in raw_norm.items()
        }
        return ParameterValues.from_mapping(list(raw_norm.keys()), data_phys)

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo_phys: float, new_hi_phys: float) -> None:
        """Shrink physical bounds and remap unit particles/means."""
        if param_name not in self.physical_param_bounds:
            raise KeyError(param_name)
        old_lo, old_hi = self.physical_param_bounds[param_name]
        w_old = old_hi - old_lo
        if w_old <= 0:
            return

        nl = float(max(min(new_lo_phys, new_hi_phys), old_lo))
        nh = float(min(max(new_lo_phys, new_hi_phys), old_hi))
        if nh <= nl:
            return

        sync_x = self.physical_x_bounds == (old_lo, old_hi)
        w_new = nh - nl

        if param_name in self._param_names:
            idx = self._param_names.index(param_name)
            for k in range(self.n_components):
                u_norm = self.means[k, idx]
                f_phys = old_lo + u_norm * w_old
                u_new_norm = (f_phys - nl) / w_new
                self.means[k, idx] = np.clip(u_new_norm, 0.0, 1.0)
                self.precisions[k, idx, idx] *= (w_old / w_new)**2

        self.model.narrow_physical_interval_for_param(param_name, nl, nh, update_x_axis=sync_x)
        self.physical_param_bounds[param_name] = (nl, nh)
        if sync_x:
            self._physical_x_bounds = (nl, nh)

        self._recompute_covariances()

    def copy(self) -> UnitCubeGaussianMixtureMarginalDistribution:
        dist = UnitCubeGaussianMixtureMarginalDistribution(
            model=self.model,
            n_components=self.n_components,
            _physical_param_bounds=dict(self.physical_param_bounds),
            _physical_x_bounds=self.physical_x_bounds,
        )
        dist.means, dist.precisions = self.means.copy(), self.precisions.copy()
        dist.weights, dist._covariances = self.weights.copy(), self._covariances.copy()
        dist.last_obs = self.last_obs
        return dist
