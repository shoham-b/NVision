"""Gaussian Mixture Belief Distribution using Extended Kalman Filter updates."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from dotenv import load_dotenv
from scipy.linalg import inv, solve

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.models.observation import Observation
from nvision.spectra.dtypes import FLOAT_DTYPE

# --- Environment-driven defaults ---------------------------------------------

load_dotenv()

NVISION_GAUSSIAN_NUM_EXPERTS: int = int(os.getenv("NVISION_GAUSSIAN_NUM_EXPERTS", "3"))
NVISION_GAUSSIAN_WEIGHT_FLOOR: float = float(os.getenv("NVISION_GAUSSIAN_WEIGHT_FLOOR", "0.05"))
NVISION_GAUSSIAN_WEIGHT_FLOOR_STEPS: int = int(os.getenv("NVISION_GAUSSIAN_WEIGHT_FLOOR_STEPS", "30"))
NVISION_GAUSSIAN_EPSILON: float = float(os.getenv("NVISION_GAUSSIAN_EPSILON", "1e-8"))

NVISION_GAUSSIAN_DEFAULT_LINEWIDTH: float = float(os.getenv("NVISION_GAUSSIAN_DEFAULT_LINEWIDTH", "1e6"))
NVISION_GAUSSIAN_DEFAULT_SPLIT: float = float(os.getenv("NVISION_GAUSSIAN_DEFAULT_SPLIT", "5e6"))
NVISION_GAUSSIAN_DEFAULT_K_NP: float = float(os.getenv("NVISION_GAUSSIAN_DEFAULT_K_NP", "1.5"))
NVISION_GAUSSIAN_DEFAULT_DIP_DEPTH: float = float(os.getenv("NVISION_GAUSSIAN_DEFAULT_DIP_DEPTH", "0.1"))


@dataclass
class GaussianMixtureMarginalDistribution(AbstractMarginalDistribution):
    """Parametric belief tracking the posterior as a Gaussian Mixture.

    Uses a linearized Extended Kalman Filter (EKF) update for each component.
    """

    n_components: int = NVISION_GAUSSIAN_NUM_EXPERTS

    # --- Mode-protection parameters ---
    weight_floor: float = NVISION_GAUSSIAN_WEIGHT_FLOOR
    weight_floor_steps: int = NVISION_GAUSSIAN_WEIGHT_FLOOR_STEPS

    means: np.ndarray = field(init=False)  # (K, D)
    precisions: np.ndarray = field(init=False)  # (K, D, D)
    weights: np.ndarray = field(init=False)  # (K,)
    _covariances: np.ndarray = field(init=False)  # (K, D, D)
    _update_count: int = field(init=False)  # observations seen so far

    _physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    _param_names: list[str] = field(init=False)
    _dim: int = field(init=False)

    _PARAM_DEFAULTS: ClassVar[dict[str, float]] = {
        "linewidth": NVISION_GAUSSIAN_DEFAULT_LINEWIDTH,
        "split": NVISION_GAUSSIAN_DEFAULT_SPLIT,
        "k_np": NVISION_GAUSSIAN_DEFAULT_K_NP,
        "dip_depth": NVISION_GAUSSIAN_DEFAULT_DIP_DEPTH,
    }

    def __post_init__(self) -> None:
        self._param_names = list(self.model.parameter_names())
        self._dim = len(self._param_names)
        K, D = self.n_components, self._dim  # noqa: N806

        self.means = np.zeros((K, D), dtype=FLOAT_DTYPE)
        self.precisions = np.zeros((K, D, D), dtype=FLOAT_DTYPE)
        self.weights = np.ones(K, dtype=FLOAT_DTYPE) / K
        self._covariances = np.zeros((K, D, D), dtype=FLOAT_DTYPE)
        self._update_count = 0

        # Derive a reference frequency range for heuristic scaling.
        freq_bounds = self._physical_param_bounds.get("frequency")
        if freq_bounds is not None:
            freq_lo, freq_hi = freq_bounds
            freq_width = max(freq_hi - freq_lo, 1.0)
        else:
            freq_lo, freq_hi, freq_width = 2.87e9, 2.87e9, 1.0

        from nvision.spectra.unit_cube import UnitCubeSignalModel

        self._is_unit_cube = isinstance(self.model, UnitCubeSignalModel) or "UnitCube" in type(self.model).__name__

        for i, name in enumerate(self._param_names):
            if self._is_unit_cube:
                mid = 0.5
                for k in range(K):
                    self.means[k, i] = mid
                    if name == "frequency" and K > 1:
                        offset = (k - (K - 1) / 2.0) * 0.1
                        self.means[k, i] = float(np.clip(mid + offset, 0.0, 1.0))
                var = (0.25) ** 2
                for k in range(K):
                    self.precisions[k, i, i] = 1.0 / var
            elif name in self._physical_param_bounds:
                lo, hi = self._physical_param_bounds[name]
                mid = (lo + hi) / 2.0
                width = hi - lo
                for k in range(K):
                    self.means[k, i] = mid
                    if name == "frequency" and K > 1:
                        offset = (k - (K - 1) / 2.0) * (width * 0.1)
                        self.means[k, i] = float(np.clip(mid + offset, lo, hi))
                var = (width / 4.0) ** 2
                for k in range(K):
                    self.precisions[k, i, i] = 1.0 / max(var, 1e-20)
            else:
                default = self._PARAM_DEFAULTS.get(name)
                if default is None:
                    default = freq_width * 1e-3
                for k in range(K):
                    self.means[k, i] = float(default)
                var = (default * 2.0) ** 2
                for k in range(K):
                    self.precisions[k, i, i] = 1.0 / max(var, 1e-20)

        if not self._physical_param_bounds:
            for k in range(K):
                self.precisions[k] = np.eye(D)

        self._recompute_covariances()

    def _recompute_covariances(self) -> None:
        epsilon = NVISION_GAUSSIAN_EPSILON
        for k in range(self.n_components):
            reg_prec = self.precisions[k] + epsilon * np.eye(self._dim)
            c = inv(reg_prec)
            self._covariances[k] = (c + c.T) / 2.0

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        if param_name in self._physical_param_bounds:
            old_lo, old_hi = self._physical_param_bounds[param_name]
            lo, hi = max(old_lo, new_lo), min(old_hi, new_hi)
            self._physical_param_bounds[param_name] = (lo, hi)

            if param_name in self._param_names:
                idx = self._param_names.index(param_name)
                width = hi - lo
                mid = (lo + hi) / 2.0
                K = self.n_components  # noqa: N806
                for k in range(K):
                    if K > 1:
                        offset = (k - (K - 1) / 2.0) * (width * 0.1)
                        self.means[k, idx] = float(np.clip(mid + offset, lo, hi))
                    else:
                        self.means[k, idx] = mid

                for k in range(K):
                    var = (width / 4.0) ** 2
                    self.precisions[k, idx, idx] = max(self.precisions[k, idx, idx], 1.0 / max(var, 1e-20))

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:
        return self._physical_param_bounds

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self._physical_param_bounds

    def update(self, obs: Observation) -> None:
        self.last_obs = obs
        self._update_mixtures(obs.x, obs.signal_value, obs.noise_std)
        self._recompute_covariances()

    def batch_update(self, observations: list[Observation]) -> None:
        if not observations:
            return
        for obs in observations:
            self._update_mixtures(obs.x, obs.signal_value, obs.noise_std)
        self.last_obs = observations[-1]
        self._recompute_covariances()

    def _update_mixtures(self, x_probe: float, y_obs: float, sigma_eta: float) -> None:
        """Perform linearized EKF update for each component."""
        K, D = self.n_components, self._dim  # noqa: N806
        sigma2 = sigma_eta**2
        epsilon = NVISION_GAUSSIAN_EPSILON

        new_means = np.zeros_like(self.means)
        new_precisions = np.zeros_like(self.precisions)
        responsibilities = np.zeros(K)

        samples = self.model.spec.unpack_samples(tuple(self.means.T))
        y_preds = self.model.compute_vectorized_samples(x_probe, samples)
        J_all = self.model.gradient_vectorized_many([x_probe], samples)[0]  # noqa: N806

        for k in range(K):
            m = self.means[k]
            y_pred = y_preds[k]
            r = y_obs - y_pred
            J = J_all[k]  # noqa: N806

            # Standard Gaussian Precision update
            delta_prec = (1.0 / sigma2) * np.outer(J, J)
            new_precisions[k] = self.precisions[k] + delta_prec

            # Stable Mean update
            reg_new_prec = new_precisions[k] + epsilon * np.eye(D)
            rhs = (1.0 / sigma2) * J * r
            try:
                delta_mu = solve(reg_new_prec, rhs)
            except np.linalg.LinAlgError:
                delta_mu = np.zeros(D)

            new_means[k] = m + delta_mu

            # Clip to bounds
            for i, name in enumerate(self._param_names):
                if self._is_unit_cube:
                    new_means[k, i] = np.clip(new_means[k, i], 0.0, 1.0)
                elif name in self._physical_param_bounds:
                    lo, hi = self._physical_param_bounds[name]
                    new_means[k, i] = np.clip(new_means[k, i], lo, hi)

            # Weight responsibility (likelihood of observation under this component)
            var_pred = J @ self._covariances[k] @ J.T + sigma2
            responsibilities[k] = np.exp(-0.5 * (r**2) / var_pred) / np.sqrt(2 * np.pi * var_pred)

        self.means = new_means
        self.precisions = new_precisions
        self.weights *= responsibilities
        w_sum = np.sum(self.weights)
        if w_sum > 1e-30:
            self.weights /= w_sum
        else:
            self.weights = np.ones(K) / K

        # Decaying weight floor
        if self.weight_floor > 0.0 and self._update_count < self.weight_floor_steps:
            t = self._update_count / max(self.weight_floor_steps, 1)
            active_floor = self.weight_floor * (1.0 - t)
            self.weights = np.maximum(self.weights, active_floor)
            self.weights /= self.weights.sum()

        self._update_count += 1

    def estimates(self) -> dict[str, float]:
        weighted_mean = np.sum(self.weights[:, None] * self.means, axis=0)
        return {name: float(weighted_mean[i]) for i, name in enumerate(self._param_names)}

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        weighted_mean = np.sum(self.weights[:, None] * self.means, axis=0)
        total_var = np.zeros(self._dim)
        for k in range(self.n_components):
            diff = self.means[k] - weighted_mean
            total_var += self.weights[k] * (np.diag(self._covariances[k]) + diff**2)
        stds = np.sqrt(np.maximum(total_var, 0.0))
        return ParameterValues.from_mapping(
            self._param_names, {name: float(stds[i]) for i, name in enumerate(self._param_names)}
        )

    def converged(self, threshold: float) -> bool:
        stds = self._empirical_uncertainty()
        return all(u < threshold for u in stds.values())

    def entropy(self) -> float:
        total_h = 0.0
        for k in range(self.n_components):
            sign, logdet = np.linalg.slogdet(self._covariances[k])
            h = 0.5 * logdet if sign > 0 else 0.0
            total_h += self.weights[k] * h
        return float(total_h)

    def copy(self) -> GaussianMixtureMarginalDistribution:
        dist = GaussianMixtureMarginalDistribution(
            model=self.model,
            n_components=self.n_components,
            weight_floor=self.weight_floor,
            weight_floor_steps=self.weight_floor_steps,
            _physical_param_bounds=self._physical_param_bounds.copy(),
        )
        dist.means, dist.precisions = self.means.copy(), self.precisions.copy()
        dist.weights, dist._covariances = self.weights.copy(), self._covariances.copy()
        dist._update_count = self._update_count
        dist.last_obs = self.last_obs
        return dist

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        K, D = self.n_components, self._dim  # noqa: N806
        samples = np.zeros((n, D), dtype=FLOAT_DTYPE)
        comp_indices = np.random.choice(K, size=n, p=self.weights)
        for k in range(K):
            mask = comp_indices == k
            nk = np.sum(mask)
            if nk == 0:
                continue
            z = np.random.multivariate_normal(self.means[k], self._covariances[k], size=nk)
            samples[mask] = z
        return ParameterValues.from_mapping(
            self._param_names, {name: samples[:, i] for i, name in enumerate(self._param_names)}
        )

    def expected_information_gain_batch(self, xs_phys: np.ndarray) -> np.ndarray:
        """Calculate EIG in physical space."""
        noise_var = (self.last_obs.noise_std**2) if self.last_obs else 0.05**2

        K, D = self.n_components, self._dim  # noqa: N806, F841
        n_x = xs_phys.shape[0]

        samples_phys = self.model.spec.unpack_samples(tuple(self.means.T))
        y_preds = self.model.compute_vectorized_many(xs_phys, samples_phys)
        J_phys = self.model.gradient_vectorized_many(xs_phys, samples_phys)  # noqa: N806

        y_mix = np.sum(self.weights * y_preds, axis=1)  # (n_x,)

        eigs = np.zeros(n_x)
        for i in range(n_x):
            pred_var = 0.0
            for k in range(K):
                gk = J_phys[i, k]
                ev = gk @ self._covariances[k] @ gk.T
                pred_var += self.weights[k] * (ev + (y_preds[i, k] - y_mix[i]) ** 2)

            eigs[i] = 0.5 * np.log(1.0 + pred_var / (noise_var + 1e-12))

        return eigs

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import norm

        idx = self._param_names.index(param_name)
        pdf_val = np.zeros_like(x, dtype=np.float64)
        for k in range(self.n_components):
            mu, sigma = self.means[k, idx], np.sqrt(max(self._covariances[k, idx, idx], 1e-12))
            pdf_val += self.weights[k] * norm.pdf(x, loc=mu, scale=sigma)
        return pdf_val

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import norm

        idx = self._param_names.index(param_name)
        cdf_val = np.zeros_like(x, dtype=np.float64)
        for k in range(self.n_components):
            mu, sigma = self.means[k, idx], np.sqrt(max(self._covariances[k, idx, idx], 1e-12))
            cdf_val += self.weights[k] * norm.cdf(x, loc=mu, scale=sigma)
        return cdf_val
