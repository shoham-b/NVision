"""Gaussian Process Regression acquisition locator."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation, ObservationHistory
from nvision.spectra.signal import SignalModel


class GaussianProcessLocator(Locator):
    """Non-sequential Gaussian Process (GP) Regression locator.

    Performs a sweep phase using Sobol sequences, and sequentially fits a Gaussian
    Process (acting as a sum of Gaussians) over the whole observation history.
    Finds the center frequency analytically by minimizing the GP posterior mean.
    """

    REQUIRES_BELIEF = True
    USES_SWEEP_MAX_STEPS = True

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel | None = None,
        max_steps: int = 150,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ) -> None:
        super().__init__(belief)
        self.signal_model = signal_model
        self.max_steps = max_steps
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi

        self.history = ObservationHistory(self.max_steps)
        self.step_count = 0
        self._sweep_points = self._generate_sobol_points(max_steps)

        self.kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=0.05, length_scale_bounds=(1e-3, 1.0))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=False)
        self._minimum_estimate = (domain_hi + domain_lo) / 2.0
        self._best_params = None

    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel | None = None,
        max_steps: int = 150,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        **kwargs
    ) -> GaussianProcessLocator:
        return cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )

    def _generate_sobol_points(self, n: int) -> np.ndarray:
        def vdc(k: int, base: int = 2) -> float:
            v, denom = 0.0, 1.0
            while k:
                k, remainder = divmod(k, base)
                denom *= base
                v += remainder / denom
            return v
        return np.array([self.domain_lo + vdc(i + 1) * (self.domain_hi - self.domain_lo) for i in range(n)])

    def next(self) -> float:
        if self.step_count < self.max_steps:
            return float(self._sweep_points[self.step_count])
        return self._minimum_estimate

    def observe(self, obs: Observation) -> None:
        self.history.append(obs)
        self.step_count += 1

        # Fit using the known profile of the signal as the prior mean!
        if self.step_count == self.max_steps or (self.step_count > 10 and self.step_count % 30 == 0):
            import scipy.optimize
            xs = self.history.xs
            ys = self.history.ys

            if self.signal_model is not None:
                param_names = self.signal_model.parameter_names()

                def objective(theta):
                    try:
                        params = self.signal_model.spec.unpack_params(theta)
                        preds = np.array([self.signal_model.compute_from_params(x, params) for x in xs])
                        residuals = ys - preds

                        # Fit GP on residuals
                        self.gp.fit(xs.reshape(-1, 1), residuals)

                        # Return the negative log marginal likelihood
                        return -self.gp.log_marginal_likelihood_value_
                    except Exception:
                        return 1e9

                x0 = np.array([self.domain_hi / 2.0] * len(param_names))
                if self._best_params is not None:
                    x0 = self._best_params

                res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B')
                self._best_params = res.x

                try:
                    cf_idx = param_names.index("center_frequency")
                    self._minimum_estimate = float(self._best_params[cf_idx])
                except ValueError:
                    self._minimum_estimate = float(self._best_params[0])
            else:
                # Fallback to zero-mean GP
                self.gp.fit(xs.reshape(-1, 1), ys)
                grid = np.linspace(self.domain_lo, self.domain_hi, 1000).reshape(-1, 1)
                y_pred = self.gp.predict(grid)
                self._minimum_estimate = float(grid[np.argmin(y_pred)][0])

    def done(self) -> bool:
        return self.step_count >= self.max_steps

    def result(self) -> dict[str, float]:
        res = {"center_frequency": self._minimum_estimate}
        if self._best_params is not None and self.signal_model is not None:
            names = self.signal_model.parameter_names()
            for i, name in enumerate(names):
                res[name] = float(self._best_params[i])
        return res
