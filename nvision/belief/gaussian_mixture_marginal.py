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
NVISION_GAUSSIAN_MIN_EXPLORATION_FRAC: float = float(os.getenv("NVISION_GAUSSIAN_MIN_EXPLORATION_FRAC", "0.05"))
NVISION_GAUSSIAN_UPDATE_DAMPING: float = float(os.getenv("NVISION_GAUSSIAN_UPDATE_DAMPING", "0.2"))

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
        self._observations: list[Observation] = []

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
            p_mat = self.precisions[k]
            # Guard against non-finite values in precisions to prevent ValueError in linear algebra operations.
            if not np.all(np.isfinite(p_mat)):
                p_mat = np.nan_to_num(p_mat, nan=0.0, posinf=1e14, neginf=-1e14)
                self.precisions[k] = p_mat

            # Relative epsilon: at least NVISION_GAUSSIAN_EPSILON, but scales
            # with the average diagonal magnitude so every eigenvalue direction
            # is regularised proportionally.
            trace_val = float(np.trace(p_mat))
            if not np.isfinite(trace_val):
                trace_val = 0.0
            eps = max(NVISION_GAUSSIAN_EPSILON, 1e-6 * trace_val / max(self._dim, 1))
            reg_prec = p_mat + eps * np.eye(self._dim)
            try:
                c = inv(reg_prec)
            except (np.linalg.LinAlgError, ValueError):
                # Last-resort: pseudo-inverse (drops near-zero singular values).
                try:
                    c = np.linalg.pinv(reg_prec)
                except (np.linalg.LinAlgError, ValueError):
                    c = np.eye(self._dim)

            if not np.all(np.isfinite(c)):
                c = np.nan_to_num(c, nan=0.0, posinf=1e14, neginf=-1e14)
            cov = (c + c.T) / 2.0
            # Ensure positive semi-definite by clamping the diagonal floor.
            np.fill_diagonal(cov, np.maximum(np.diag(cov), 0.0))
            self._covariances[k] = cov

        # Enforce min exploration standard deviation floor for all parameters.
        # Decay the exploration floor over time so it can converge.
        decay = np.exp(-self._update_count / 25.0)
        for k in range(self.n_components):
            cov = self._covariances[k]
            for i, name in enumerate(self._param_names):
                if self._is_unit_cube:
                    width = 1.0
                else:
                    lo, hi = self._physical_param_bounds.get(name, (0.0, 1.0))
                    width = max(hi - lo, 1e-12)
                min_std = NVISION_GAUSSIAN_MIN_EXPLORATION_FRAC * width * decay
                min_var = min_std ** 2
                if cov[i, i] < min_var:
                    cov[i, i] = min_var
                    self.precisions[k, i, i] = 1.0 / min_var

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
        self._observations.append(obs)
        self._update_mixtures(obs.x, obs.signal_value, obs.noise_std)
        self._recompute_covariances()

    def batch_update(self, observations: list[Observation]) -> None:
        if not observations:
            return
        for obs in observations:
            self._observations.append(obs)
            self._update_mixtures(obs.x, obs.signal_value, obs.noise_std)
        self.last_obs = observations[-1]
        self._recompute_covariances()

    def _update_mixtures(self, x_probe: float, y_obs: float, sigma_eta: float) -> None:
        """Perform EKF update for each component."""
        K, D = self.n_components, self._dim  # noqa: N806
        sigma2 = max(sigma_eta**2, 1e-12)
        epsilon = NVISION_GAUSSIAN_EPSILON

        new_means = np.zeros_like(self.means)
        new_precisions = np.zeros_like(self.precisions)
        responsibilities = np.zeros(K)

        samples = self.model.spec.unpack_samples(tuple(self.means.T))
        y_preds = self.model.compute_vectorized_samples(x_probe, samples)
        J_all = self.model.gradient_vectorized_many([x_probe], samples)[0]  # noqa: N806

        damping = NVISION_GAUSSIAN_UPDATE_DAMPING
        for k in range(K):
            m = self.means[k]
            y_pred = y_preds[k]
            r = y_obs - y_pred
            J = J_all[k]  # noqa: N806

            # Guard against non-finite values in observation gradients or residuals.
            if not np.all(np.isfinite(J)) or not np.isfinite(r):
                new_precisions[k] = self.precisions[k]
                new_means[k] = m
                continue

            # Standard Gaussian Precision update
            delta_prec = (damping / sigma2) * np.outer(J, J)
            if not np.all(np.isfinite(delta_prec)):
                delta_prec = np.zeros_like(delta_prec)

            # Use a forgetting factor to prevent unbounded precision growth (covariance collapse)
            forgetting_factor = 0.95
            candidate_prec = self.precisions[k] * forgetting_factor + delta_prec

            # Cap the precision matrix elements to prevent exponential growth and numerical overflow.
            if not np.all(np.isfinite(candidate_prec)) or np.any(np.abs(candidate_prec) > 1e14):
                candidate_prec = np.clip(candidate_prec, -1e14, 1e14)
                for i in range(D):
                    candidate_prec[i, i] = max(candidate_prec[i, i], 1e-8)

            new_precisions[k] = candidate_prec

            # Stable Mean update — use the same trace-relative epsilon as
            # _recompute_covariances to keep the solve well-conditioned across
            # the mixed physical/dimensionless parameter scales.
            trace_val = float(np.trace(new_precisions[k]))
            if not np.isfinite(trace_val):
                trace_val = 0.0
            eps_solve = max(epsilon, 1e-6 * trace_val / max(D, 1))
            reg_new_prec = new_precisions[k] + eps_solve * np.eye(D)
            rhs = (damping / sigma2) * J * r

            if not np.all(np.isfinite(rhs)):
                rhs = np.zeros_like(rhs)

            try:
                if np.all(np.isfinite(reg_new_prec)) and np.all(np.isfinite(rhs)):
                    delta_mu = solve(reg_new_prec, rhs)
                else:
                    delta_mu = np.zeros(D)
            except (np.linalg.LinAlgError, ValueError):
                delta_mu = np.zeros(D)

            if not np.all(np.isfinite(delta_mu)):
                delta_mu = np.zeros(D)

            new_means[k] = m + delta_mu

            # Reflective bounds
            for i, name in enumerate(self._param_names):
                val = new_means[k, i]
                if self._is_unit_cube:
                    lo, hi = 0.0, 1.0
                elif name in self._physical_param_bounds:
                    lo, hi = self._physical_param_bounds[name]
                else:
                    continue

                if val < lo:
                    val = 2.0 * lo - val
                elif val > hi:
                    val = 2.0 * hi - val
                # Safe fallback clip in case of an extremely large step
                new_means[k, i] = np.clip(val, lo, hi)

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

        # Permanent minimum weight floor to prevent weight collapse.
        # We decay this floor exponentially to allow it to fully converge to a single expert if appropriate.
        decay = np.exp(-self._update_count / 25.0)
        curr_floor = 0.01 * decay
        if curr_floor > 0:
            self.weights = np.maximum(self.weights, curr_floor)
            self.weights /= self.weights.sum()

        self._update_count += 1

    def estimates(self) -> dict[str, float]:
        best_k = np.argmax(self.weights)
        best_mean = self.means[best_k]
        return {name: float(best_mean[i]) for i, name in enumerate(self._param_names)}

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
        dist._observations = list(self._observations)
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

    def get_candidates(self) -> np.ndarray:
        """Return candidate measurement frequencies in physical space.

        Combines EIG-weighted mean positions and empirically observed dip
        locations so the locator focuses on regions with true signal, not
        just regions where the EKF posterior has high uncertainty.
        """
        freq_bounds = self._physical_param_bounds.get("frequency")
        if freq_bounds is None:
            return np.linspace(0.0, 1.0, 200)
        f_lo, f_hi = freq_bounds
        domain = f_hi - f_lo

        # Base: a coarse global sweep as fallback
        n_global = max(200, int(domain / max(domain * 0.001, 1e5)))
        candidates = list(np.linspace(f_lo, f_hi, n_global))

        # Add dense grids around each expert's mean frequency
        if "frequency" in self._param_names:
            freq_idx = self._param_names.index("frequency")
            for k in range(self.n_components):
                mu_f = float(self.means[k, freq_idx])
                sigma_f = float(np.sqrt(max(self._covariances[k, freq_idx, freq_idx], 0.0)))
                window = max(sigma_f * 3.0, 5e6)  # at least 5 MHz
                pts = np.linspace(max(mu_f - window, f_lo), min(mu_f + window, f_hi), 60)
                candidates.extend(pts.tolist())

        # Add dense grids around empirically observed dip locations
        if len(self._observations) >= 5:
            obs_xs = np.array([o.x for o in self._observations])
            obs_ys = np.array([o.signal_value for o in self._observations])
            upper_perc = float(np.percentile(obs_ys, 70))
            flat_pts = obs_ys[obs_ys >= float(np.percentile(obs_ys, 40))]
            noise_est = float(np.std(flat_pts)) if len(flat_pts) >= 3 else 0.02
            dip_thresh = upper_perc - max(3.0 * noise_est, 0.015 * upper_perc)
            dip_xs = obs_xs[obs_ys < dip_thresh]
            if len(dip_xs) > 0:
                window = max(10e6, domain * 0.02)  # 10 MHz or 2% of domain
                for dip_x in dip_xs:
                    pts = np.linspace(
                        max(dip_x - window, f_lo),
                        min(dip_x + window, f_hi),
                        50,
                    )
                    candidates.extend(pts.tolist())

        arr = np.unique(np.clip(np.array(candidates, dtype=np.float64), f_lo, f_hi))
        return arr

    def select_max_information_gain(self, candidates: np.ndarray, n: int, noise_std: float = 0.02) -> np.ndarray:
        """Select the top-n candidates by Expected Information Gain (EIG).

        Uses :meth:`expected_information_gain_batch` over the provided candidates
        and returns the top ``n`` by EIG score.

        Args:
            candidates: 1D array of candidate frequencies in physical space.
            n: Number of top candidates to return.
            noise_std: Measurement noise std for EIG computation.

        Returns:
            1D array of up to ``n`` candidates ranked by EIG (highest first).
        """
        if len(candidates) == 0:
            return candidates[:0]

        # Override noise_var using provided noise_std so the caller controls it
        orig_last_obs = self.last_obs

        class _FakeObs:
            def __init__(self, std: float) -> None:
                self.noise_std = std

        self.last_obs = _FakeObs(noise_std)  # type: ignore[assignment]
        eig_scores = self.expected_information_gain_batch(candidates)
        self.last_obs = orig_last_obs

        top_n = min(n, len(candidates))
        top_indices = np.argpartition(eig_scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(eig_scores[top_indices])[::-1]]
        return candidates[top_indices]

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
