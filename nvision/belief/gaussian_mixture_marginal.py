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
    priors: dict[str, tuple[float, float]] | None = None

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

    def __post_init__(self) -> None:  # noqa: C901
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

        # Resolve active priors (mapping physical to unit space if in unit cube mode)
        if self.priors is None and "_priors" in self._physical_param_bounds:
            self.priors = self._physical_param_bounds["_priors"]

        active_priors = {}
        if self.priors:
            for name, (mu, std) in self.priors.items():
                if self._is_unit_cube:
                    if name in self._physical_param_bounds:
                        lo, hi = self._physical_param_bounds[name]
                        width = max(hi - lo, 1e-12)
                        active_priors[name] = (float((mu - lo) / width), float(std / width))
                else:
                    active_priors[name] = (float(mu), float(std))

        for i, name in enumerate(self._param_names):
            if name in active_priors:
                prior_mu, prior_std = active_priors[name]
                lo, hi = (0.0, 1.0) if self._is_unit_cube else self._physical_param_bounds.get(name, (0.0, 1.0))
                sampled_means = np.random.normal(loc=prior_mu, scale=prior_std, size=K)
                for k in range(K):
                    self.means[k, i] = float(np.clip(sampled_means[k], lo, hi))
                var = prior_std**2
                for k in range(K):
                    self.precisions[k, i, i] = 1.0 / max(var, 1e-20)
            elif name == "frequency":
                if self._is_unit_cube:
                    lo, hi, width = 0.0, 1.0, 1.0
                else:
                    lo, hi = self._physical_param_bounds.get("frequency", (2.87e9, 2.87e9))
                    width = max(hi - lo, 1e-12)
                part_width = width / K
                std = part_width / 2.0
                var = std**2
                for k in range(K):
                    center = lo + (k + 0.5) * part_width
                    self.means[k, i] = float(np.clip(center, lo, hi))
                    self.precisions[k, i, i] = 1.0 / max(var, 1e-20)
            elif self._is_unit_cube:
                mid = 0.5
                for k in range(K):
                    self.means[k, i] = mid
                var = (0.15) ** 2
                for k in range(K):
                    self.precisions[k, i, i] = 1.0 / var
            elif name in self._physical_param_bounds:
                lo, hi = self._physical_param_bounds[name]
                mid = (lo + hi) / 2.0
                for k in range(K):
                    self.means[k, i] = mid
                var = ((hi - lo) * 0.15) ** 2
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
        """Invert each component's precision matrix to obtain the covariance.

        Uses a **trace-relative** regularisation so that the epsilon floor
        scales with the actual magnitude of each precision matrix.  This is
        critical because the 5 tracked parameters live on very different
        physical scales: frequency is O(1e9 Hz) → precision ~O(1e-17 Hz⁻²),
        while k_np and dip_depth are dimensionless → precision ~O(1–1000).
        A fixed epsilon = 1e-8 (absolute) is smaller than the frequency
        entries but much smaller than the dimensionless entries, leaving
        the matrix effectively singular after a few EKF updates.
        """
        for k in range(self.n_components):
            P = self.precisions[k]  # noqa: N806
            # Relative epsilon: at least NVISION_GAUSSIAN_EPSILON, but scales
            # with the average diagonal magnitude so every eigenvalue direction
            # is regularised proportionally.
            trace_val = float(np.trace(P))
            eps = max(NVISION_GAUSSIAN_EPSILON, 1e-6 * trace_val / max(self._dim, 1))
            reg_prec = P + eps * np.eye(self._dim)
            try:
                c = inv(reg_prec)
            except np.linalg.LinAlgError:
                # Last-resort: pseudo-inverse (drops near-zero singular values).
                c = np.linalg.pinv(reg_prec)
            cov = (c + c.T) / 2.0
            # Ensure positive semi-definite by clamping the diagonal floor.
            np.fill_diagonal(cov, np.maximum(np.diag(cov), 0.0))
            self._covariances[k] = cov

        # Enforce frequency uncertainty >= linewidth uncertainty
        if "frequency" in self._param_names and "linewidth" in self._param_names:
            freq_idx = self._param_names.index("frequency")
            lw_idx = self._param_names.index("linewidth")
            if self._is_unit_cube:
                freq_lo, freq_hi = self._physical_param_bounds.get("frequency", (0.0, 1.0))
                freq_w = max(freq_hi - freq_lo, 1e-12)
                lw_lo, lw_hi = self._physical_param_bounds.get("linewidth", (0.0, 1.0))
                lw_w = max(lw_hi - lw_lo, 1e-12)
                scale_factor = (lw_w / freq_w) ** 2
            else:
                scale_factor = 1.0

            for k in range(self.n_components):
                cov = self._covariances[k]
                var_lw = cov[lw_idx, lw_idx]
                var_freq_min = var_lw * scale_factor
                if cov[freq_idx, freq_idx] < var_freq_min:
                    cov[freq_idx, freq_idx] = var_freq_min
                    self.precisions[k, freq_idx, freq_idx] = 1.0 / max(var_freq_min, 1e-20)

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
            # Use a forgetting factor to prevent unbounded precision growth (covariance collapse)
            forgetting_factor = 0.95
            new_precisions[k] = self.precisions[k] * forgetting_factor + delta_prec

            # Stable Mean update — use the same trace-relative epsilon as
            # _recompute_covariances to keep the solve well-conditioned across
            # the mixed physical/dimensionless parameter scales.
            trace_val = float(np.trace(new_precisions[k]))
            eps_solve = max(epsilon, 1e-6 * trace_val / max(D, 1))
            reg_new_prec = new_precisions[k] + eps_solve * np.eye(D)
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
            # Using just sigma2 prevents heavily penalizing uncertain components
            # which leads to weight stagnation when var_pred becomes large.
            responsibilities[k] = np.exp(-0.5 * (r**2) / sigma2)

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

        # Enforce frequency uncertainty >= linewidth uncertainty
        if "frequency" in self._param_names and "linewidth" in self._param_names:
            freq_idx = self._param_names.index("frequency")
            lw_idx = self._param_names.index("linewidth")
            if self._is_unit_cube:
                freq_lo, freq_hi = self._physical_param_bounds.get("frequency", (0.0, 1.0))
                freq_w = max(freq_hi - freq_lo, 1e-12)
                lw_lo, lw_hi = self._physical_param_bounds.get("linewidth", (0.0, 1.0))
                lw_w = max(lw_hi - lw_lo, 1e-12)
                scale_factor = (lw_w / freq_w) ** 2
            else:
                scale_factor = 1.0

            total_var[freq_idx] = max(total_var[freq_idx], total_var[lw_idx] * scale_factor)

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
            priors=self.priors.copy() if self.priors else None,
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
        """Calculate EIG in physical space.

        Guards against NaN/inf that can appear when the covariance matrix
        has blown up due to numerical ill-conditioning: clamps ``pred_var``
        to be non-negative and replaces any non-finite EIG value with 0.
        """
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

            # Clamp pred_var >= 0 (can go slightly negative from floating-point noise).
            pred_var = max(pred_var, 0.0)
            eig_val = 0.5 * np.log(1.0 + pred_var / (noise_var + 1e-12))
            # Replace NaN/inf with 0 so argmax / downstream stay well-defined.
            eigs[i] = float(eig_val) if np.isfinite(eig_val) else 0.0

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
