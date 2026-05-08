"""Student's t Mixture Belief Distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from scipy.linalg import inv, solve

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.models.observation import Observation
from nvision.spectra.dtypes import FLOAT_DTYPE


@njit(cache=True, fastmath=True)
def _nv_lorentzian_grad_scalar(
    x: float,
    f: float,
    lw: float,
    s: float,
    k: float,
    d: float,
    grad: np.ndarray,
) -> None:
    """Analytical gradient of NV Lorentzian signal model (in-place).

    Order: frequency, linewidth, split, k_np, dip_depth, background
    """
    lw2 = lw * lw

    # Amplitudes
    amp_r = d
    amp_c = d / k
    amp_l = d / (k * k)

    # Dip centers
    fc, fl, fr = f, f - s, f + s

    # Center dip
    dxc = x - fc
    dxc2 = dxc * dxc
    den_c = dxc2 + lw2
    iden_c = 1.0 / den_c
    iden_c2 = iden_c * iden_c
    tc = lw2 * iden_c
    dt_df_c = 2.0 * lw2 * dxc * iden_c2
    dt_dlw_c = 2.0 * lw * dxc2 * iden_c2

    # Left dip
    dxl = x - fl
    dxl2 = dxl * dxl
    den_l = dxl2 + lw2
    iden_l = 1.0 / den_l
    iden_l2 = iden_l * iden_l
    tl = lw2 * iden_l
    dt_df_l = 2.0 * lw2 * dxl * iden_l2
    dt_dlw_l = 2.0 * lw * dxl2 * iden_l2

    # Right dip
    dxr = x - fr
    dxr2 = dxr * dxr
    den_r = dxr2 + lw2
    iden_r = 1.0 / den_r
    iden_r2 = iden_r * iden_r
    tr = lw2 * iden_r
    dt_df_r = 2.0 * lw2 * dxr * iden_r2
    dt_dlw_r = 2.0 * lw * dxr2 * iden_r2

    # dS/df
    grad[0] = -(amp_l * dt_df_l + amp_c * dt_df_c + amp_r * dt_df_r)
    # dS/dlw
    grad[1] = -(amp_l * dt_dlw_l + amp_c * dt_dlw_c + amp_r * dt_dlw_r)
    # dS/ds
    grad[2] = amp_l * dt_df_l - amp_r * dt_df_r
    # dS/dk
    k2 = k * k
    grad[3] = (2.0 * d / (k2 * k)) * tl + (d / k2) * tc
    # dS/dd
    grad[4] = -(tl / k2 + tc / k + tr)
    # dS/dbg
    grad[5] = 1.0


@njit(cache=True, parallel=True, fastmath=True)
def _eig_batch_jit(
    xs: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    noise_var: float,
) -> np.ndarray:
    """Highly optimized Numba-accelerated EIG batch computation."""
    n_x = xs.shape[0]
    n_k = weights.shape[0]
    means.shape[1]
    eigs = np.empty(n_x, dtype=np.float64)

    for i in prange(n_x):
        x = xs[i]

        y_preds = np.empty(n_k)
        # Avoid large allocations; since dim is fixed and small, this is okay
        # but we can also use a flat array if needed.
        grads = np.empty((n_k, 6))

        for k in range(n_k):
            m = means[k]
            f, lw, s, knp, d, bg = m[0], m[1], m[2], m[3], m[4], m[5]

            lw2 = lw * lw
            fl, fc, fr = f - s, f, f + s
            tl = lw2 / ((x - fl) ** 2 + lw2)
            tc = lw2 / ((x - fc) ** 2 + lw2)
            tr = lw2 / ((x - fr) ** 2 + lw2)

            y_preds[k] = bg - (d / (knp * knp) * tl + d / knp * tc + d * tr)

            # Inlined/Efficient gradient call
            _nv_lorentzian_grad_scalar(x, f, lw, s, knp, d, grads[k])

        y_mix = 0.0
        for k in range(n_k):
            y_mix += weights[k] * y_preds[k]

        pred_var = 0.0
        for k in range(n_k):
            gk = grads[k]
            ck = covs[k]

            # Manual dot product for speed (gT @ Σ @ g)
            ev = 0.0
            for r in range(6):
                tmp = 0.0
                for c in range(6):
                    tmp += gk[c] * ck[r, c]
                ev += tmp * gk[r]

            pred_var += weights[k] * (ev + (y_preds[k] - y_mix) ** 2)

        eigs[i] = 0.5 * math.log(1.0 + pred_var / (noise_var + 1e-12))

    return eigs


@dataclass
class StudentsTMixtureMarginalDistribution(AbstractMarginalDistribution):
    """Parametric belief tracking the posterior as a Student's t Mixture.

    Uses a linearized conditionally conjugate Normal-Inverse-Wishart (NIW) update
    with Student's t observation weighting for robustness. Optimized with Numba.
    """

    n_components: int = 3

    means: np.ndarray = field(init=False)  # (k_comps, d_dims)
    precisions: np.ndarray = field(init=False)  # (k_comps, d_dims, d_dims)
    kappas: np.ndarray = field(init=False)  # (k_comps,)
    nus: np.ndarray = field(init=False)  # (k_comps,)
    weights: np.ndarray = field(init=False)  # (k_comps,)
    _covariances: np.ndarray = field(init=False)  # (k_comps, d_dims, d_dims)

    _physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    _param_names: list[str] = field(init=False)
    _dim: int = field(init=False)

    def __post_init__(self) -> None:
        self._param_names = list(self.model.parameter_names())
        self._dim = len(self._param_names)
        k_comps, d_dims = self.n_components, self._dim

        self.means = np.zeros((k_comps, d_dims), dtype=FLOAT_DTYPE)
        self.precisions = np.zeros((k_comps, d_dims, d_dims), dtype=FLOAT_DTYPE)
        self.kappas = np.ones(k_comps, dtype=FLOAT_DTYPE) * 1.0
        self.nus = np.ones(k_comps, dtype=FLOAT_DTYPE) * (d_dims + 2.0)
        self.weights = np.ones(k_comps, dtype=FLOAT_DTYPE) / k_comps
        self._covariances = np.zeros((k_comps, d_dims, d_dims), dtype=FLOAT_DTYPE)

        if self._physical_param_bounds:
            for i, name in enumerate(self._param_names):
                if name in self._physical_param_bounds:
                    lo, hi = self._physical_param_bounds[name]
                    mid = (lo + hi) / 2.0
                    width = hi - lo
                    for k in range(k_comps):
                        self.means[k, i] = mid
                        if name == "frequency" and k_comps > 1:
                            offset = (k - (k_comps - 1) / 2.0) * (width * 0.1)
                            self.means[k, i] = np.clip(mid + offset, lo, hi)
                    var = (width / 4.0) ** 2
                    for k in range(k_comps):
                        self.precisions[k, i, i] = 1.0 / max(var, 1e-20)
        else:
            for k in range(k_comps):
                self.precisions[k] = np.eye(d_dims)

        self._recompute_covariances()

    def _recompute_covariances(self) -> None:
        """Cache Σ = Λ⁻¹ for all components with ridge regularization."""
        epsilon = 1e-8
        for k in range(self.n_components):
            reg_prec = self.precisions[k] + epsilon * np.eye(self._dim)
            c = inv(reg_prec)
            self._covariances[k] = (c + c.T) / 2.0

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
        for obs in observations:
            self._update_mixtures(obs.x, obs.signal_value, obs.noise_std)
        self._recompute_covariances()

    def _update_mixtures(self, x: float, y: float, sigma_eta: float) -> None:
        """Perform linearized update with Student's t weighting."""
        k_comps, d_dims = self.n_components, self._dim
        sigma2 = sigma_eta**2
        df_weight = 3.0
        epsilon = 1e-8

        new_means = np.zeros_like(self.means)
        new_precisions = np.zeros_like(self.precisions)
        responsibilities = np.zeros(k_comps)

        # Pre-allocate gradient workspace
        grad_j = np.zeros(d_dims, dtype=np.float64)

        for k in range(k_comps):
            m = self.means[k]
            f, lw, s, knp, d, bg = m[0], m[1], m[2], m[3], m[4], m[5]

            # Prediction
            y_pred = self.model.compute_nvcenter_lorentzian_model(x, f, lw, s, knp, d, bg)
            r = y - y_pred

            # Gradient ∇_θ S(x; θ) in-place
            _nv_lorentzian_grad_scalar(x, f, lw, s, knp, d, grad_j)

            # Weighting
            w = (1.0 + (r**2) / (df_weight * sigma2)) ** (-(df_weight + 1.0) / 2.0)

            # Precision update
            delta_prec = (w / sigma2) * np.outer(grad_j, grad_j)
            new_precisions[k] = self.precisions[k] + delta_prec

            # Stable Mean update
            reg_new_prec = new_precisions[k] + epsilon * np.eye(d_dims)
            rhs = (w / sigma2) * grad_j * r
            delta_mu = solve(reg_new_prec, rhs)

            new_means[k] = m + delta_mu

            # Clip to bounds
            for i, name in enumerate(self._param_names):
                if name in self._physical_param_bounds:
                    lo, hi = self._physical_param_bounds[name]
                    new_means[k, i] = np.clip(new_means[k, i], lo, hi)

            self.kappas[k] += w
            self.nus[k] += w

            # Weight responsibility
            var_pred = grad_j @ self._covariances[k] @ grad_j.T + sigma2
            responsibilities[k] = np.exp(-0.5 * (r**2) / var_pred) / np.sqrt(2 * np.pi * var_pred)

        self.means = new_means
        self.precisions = new_precisions
        self.weights *= responsibilities
        w_sum = np.sum(self.weights)
        if w_sum > 1e-30:
            self.weights /= w_sum
        else:
            self.weights = np.ones(k_comps) / k_comps

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

    def copy(self) -> StudentsTMixtureMarginalDistribution:
        dist = StudentsTMixtureMarginalDistribution(
            model=self.model,
            n_components=self.n_components,
            _physical_param_bounds=self._physical_param_bounds.copy(),
        )
        dist.means, dist.precisions = self.means.copy(), self.precisions.copy()
        dist.kappas, dist.nus = self.kappas.copy(), self.nus.copy()
        dist.weights, dist._covariances = self.weights.copy(), self._covariances.copy()
        dist.last_obs = self.last_obs
        return dist

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        k_comps, d_dims = self.n_components, self._dim
        samples = np.zeros((n, d_dims), dtype=FLOAT_DTYPE)
        comp_indices = np.random.choice(k_comps, size=n, p=self.weights)
        for k in range(k_comps):
            mask = comp_indices == k
            nk = np.sum(mask)
            if nk == 0:
                continue
            df = max(self.nus[k] - d_dims + 1.0, 1.0)
            u = np.random.chisquare(df, size=nk) / df
            z = np.random.multivariate_normal(np.zeros(d_dims), self._covariances[k], size=nk)
            samples[mask] = self.means[k] + z / np.sqrt(u[:, None])
        return ParameterValues.from_mapping(
            self._param_names, {name: samples[:, i] for i, name in enumerate(self._param_names)}
        )

    def expected_information_gain(self, x: float) -> float:
        return float(self.expected_information_gain_batch(np.array([x]))[0])

    def expected_information_gain_batch(self, xs: np.ndarray) -> np.ndarray:
        noise_var = (self.last_obs.noise_std**2) if self.last_obs else 0.01**2
        return _eig_batch_jit(xs, self.weights, self.means, self._covariances, noise_var)

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import t

        idx = self._param_names.index(param_name)
        pdf_val = np.zeros_like(x, dtype=np.float64)
        for k in range(self.n_components):
            mu, sigma = self.means[k, idx], np.sqrt(max(self._covariances[k, idx, idx], 1e-12))
            df = max(self.nus[k] - self._dim + 1.0, 1.0)
            pdf_val += self.weights[k] * t.pdf(x, df=df, loc=mu, scale=sigma)
        return pdf_val

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import t

        idx = self._param_names.index(param_name)
        cdf_val = np.zeros_like(x, dtype=np.float64)
        for k in range(self.n_components):
            mu, sigma = self.means[k, idx], np.sqrt(max(self._covariances[k, idx, idx], 1e-12))
            df = max(self.nus[k] - self._dim + 1.0, 1.0)
            cdf_val += self.weights[k] * t.cdf(x, df=df, loc=mu, scale=sigma)
        return cdf_val
