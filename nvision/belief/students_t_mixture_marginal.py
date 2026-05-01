"""Student's t Mixture Belief Distribution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.models.observation import Observation


@dataclass
class StudentsTMixtureMarginalDistribution(AbstractMarginalDistribution):
    """Parametric belief tracking the posterior as a Student's t Mixture.

    Performs analytical Bayesian back inference (MAP + Laplace approximation)
    using the observed points directly, bypassing any grids or particles.
    """

    # K components, D parameters
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    means: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))  # (K, D)
    covariances: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1)))  # (K, D, D)
    dfs: np.ndarray = field(default_factory=lambda: np.array([3.0]))  # (K,)

    _physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    _xs: list[float] = field(default_factory=list)
    _ys: list[float] = field(default_factory=list)
    _stds: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._param_names = list(self.model.parameter_names())
        dim = len(self._param_names)
        if self.means.shape == (1, 1) and dim > 1:
            self.means = np.zeros((1, dim))
            self.covariances = np.array([np.eye(dim) * 1e4])

        if self._physical_param_bounds:
            for i, name in enumerate(self._param_names):
                if name in self._physical_param_bounds:
                    lo, hi = self._physical_param_bounds[name]
                    self.means[0, i] = (lo + hi) / 2.0
                    self.covariances[0, i, i] = ((hi - lo) / 4.0) ** 2

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:
        return self._physical_param_bounds

    def update(self, obs: Observation) -> None:
        self.last_obs = obs
        self._xs.append(obs.x)
        self._ys.append(obs.signal_value)
        self._stds.append(obs.noise_std)
        self._refit()

    def batch_update(self, observations: list[Observation]) -> None:
        if not observations:
            return
        self.last_obs = observations[-1]
        for obs in observations:
            self._xs.append(obs.x)
            self._ys.append(obs.signal_value)
            self._stds.append(obs.noise_std)
        self._refit()

    def _refit(self) -> None:
        """Find the MAP parameters and compute inverse Hessian for uncertainty."""
        if len(self._xs) < 1:
            return

        xs = np.array(self._xs)
        ys = np.array(self._ys)
        stds = np.array(self._stds)

        def nll(theta: np.ndarray) -> float:
            try:
                # Bounds penalty
                for i, name in enumerate(self._param_names):
                    if name in self._physical_param_bounds:
                        lo, hi = self._physical_param_bounds[name]
                        if not (lo <= theta[i] <= hi):
                            return 1e9

                params = self.model.spec.unpack_params(theta)
                preds = np.array([self.model.compute_from_params(x, params) for x in xs])

                # Student's t likelihood for data
                df = self.dfs[0]
                sigma_sq = stds**2
                diff = ys - preds

                # Negative log likelihood (kernel only)
                ll = -0.5 * (df + 1.0) * np.sum(np.log(1.0 + (diff**2) / (df * sigma_sq)))

                # Weak Gaussian prior towards center of bounds to stabilize
                prior = 0.0
                for i, name in enumerate(self._param_names):
                    if name in self._physical_param_bounds:
                        lo, hi = self._physical_param_bounds[name]
                        center = (lo + hi) / 2.0
                        scale = (hi - lo) / 2.0
                        z = (theta[i] - center) / scale
                        prior += 0.5 * z**2

                return float(-ll + prior)
            except Exception:
                return 1e9

        x0 = self.means[0]
        res = scipy.optimize.minimize(nll, x0, method="L-BFGS-B")
        self.means[0] = res.x

        try:
            # Requires numdifftools, if not present fallback to identity or previous
            import numdifftools as nd

            hess = nd.Hessian(nll)(res.x)
            eigvals, eigvecs = np.linalg.eigh(hess)
            eigvals = np.maximum(eigvals, 1e-6)
            self.covariances[0] = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        except Exception:
            pass

    def estimates(self) -> dict[str, float]:
        return {name: float(self.means[0, i]) for i, name in enumerate(self._param_names)}

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        stds = np.sqrt(np.maximum(np.diag(self.covariances[0]), 0.0))
        return ParameterValues.from_mapping(
            self._param_names, {name: float(stds[i]) for i, name in enumerate(self._param_names)}
        )

    def converged(self, threshold: float) -> bool:
        stds = self._empirical_uncertainty()
        return all(u < threshold for u in stds.values())

    def entropy(self) -> float:
        sign, logdet = np.linalg.slogdet(self.covariances[0])
        return float(0.5 * logdet) if sign > 0 else 0.0

    def copy(self) -> StudentsTMixtureMarginalDistribution:
        dist = StudentsTMixtureMarginalDistribution(
            model=self.model,
            weights=self.weights.copy(),
            means=self.means.copy(),
            covariances=self.covariances.copy(),
            dfs=self.dfs.copy(),
            _physical_param_bounds=self._physical_param_bounds.copy(),
            last_obs=self.last_obs,
        )
        dist._xs = self._xs.copy()
        dist._ys = self._ys.copy()
        dist._stds = self._stds.copy()
        return dist

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        # Using a Gaussian approx for the parameter posterior sampling
        # (could use multivariate t here)
        samples = np.random.multivariate_normal(self.means[0], self.covariances[0], size=n)
        return ParameterValues.from_mapping(

            self._param_names, {name: samples[:, i] for i, name in enumerate(self._param_names)}
        )

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import t

        idx = self._param_names.index(param_name)
        mu = self.means[0, idx]
        sigma = np.sqrt(max(self.covariances[0, idx, idx], 1e-12))
        return t.pdf(x, df=self.dfs[0], loc=mu, scale=sigma)

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        from scipy.stats import t

        idx = self._param_names.index(param_name)
        mu = self.means[0, idx]
        sigma = np.sqrt(max(self.covariances[0, idx, idx], 1e-12))
        return t.cdf(x, df=self.dfs[0], loc=mu, scale=sigma)
