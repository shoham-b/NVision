"""Laplace Approximation (IEKF) acquisition locator."""

from __future__ import annotations

import numpy as np
import scipy.optimize

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation, ObservationHistory
from nvision.spectra.signal import SignalModel


class LaplaceLocator(Locator):
    """Laplace Approximation + Relinearization Locator.

    Performs an initial Sobol sweep to establish a valid log-posterior,
    then updates the parameters sequentially by finding the MAP via gradient
    descent on the log-posterior, and fitting a Gaussian around it (relinearization).
    """

    REQUIRES_BELIEF = True

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int = 150,
        initial_sweep_steps: int = 50,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ) -> None:
        super().__init__(belief)
        self.signal_model = signal_model
        self.max_steps = max_steps
        self.initial_sweep_steps = initial_sweep_steps
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi

        self.history = ObservationHistory(self.max_steps)
        self.step_count = 0
        self._sweep_points = self._generate_sobol_points(initial_sweep_steps)

        # Gaussian prior parameters
        self.param_names = self.signal_model.parameter_names()
        self.dim = len(self.param_names)
        self.mu_map = np.zeros(self.dim)
        self.sigma_inv = np.zeros((self.dim, self.dim))  # precision matrix

    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel | None = None,
        max_steps: int = 150,
        initial_sweep_steps: int | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        **kwargs
    ) -> LaplaceLocator:
        if signal_model is None:
            # Fallback if signal_model wasn't explicitly passed, though it should be.
            from nvision.spectra.models.lorentzian import NVCenterLorentzianModel
            signal_model = NVCenterLorentzianModel()

        if initial_sweep_steps is None:
            from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
            initial_sweep_steps = SequentialBayesianLocator._sweep_steps_for_signal_coverage(
                belief,
                noise_std=kwargs.get("noise_std"),
                signal_min_span=kwargs.get("signal_min_span")
            )

        return cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            initial_sweep_steps=initial_sweep_steps,
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
        if self.step_count < self.initial_sweep_steps:
            return float(self._sweep_points[self.step_count])

        # Post-sweep: measure at the current MAP estimate of the center frequency!
        # Find index of center_frequency
        try:
            cf_idx = self.param_names.index("center_frequency")
            return float(self.mu_map[cf_idx])
        except ValueError:
            return (self.domain_hi + self.domain_lo) / 2.0

    def observe(self, obs: Observation) -> None:
        self.history.append(obs)
        self.step_count += 1

        if self.step_count == self.initial_sweep_steps:
            # Establish initial MAP globally over all data
            self._update_map(full_history=True)

        elif self.step_count > self.initial_sweep_steps:
            # Sequentially update MAP (relinearization) using new observation + Gaussian prior
            self._update_map(full_history=False)

    def _update_map(self, full_history: bool) -> None:
        """Find the MAP via gradient descent on log-posterior and update Gaussian."""

        def obj_func(theta: np.ndarray) -> float:
            nll = 0.0

            try:
                params = self.signal_model.spec.unpack_params(theta)

                # Check bounds roughly
                if not (0.0 <= params.center_frequency <= 1.0):
                    return 1e9
            except Exception:
                return 1e9

            # Predict
            if full_history:
                preds = [self.signal_model.compute_from_params(x, params) for x in self.history.xs]
                obs_ys = self.history.ys
                nll = 0.5 * np.sum((np.array(preds) - obs_ys)**2 / (0.05**2))
            else:
                diff = theta - self.mu_map
                prior_penalty = 0.5 * diff.T @ self.sigma_inv @ diff

                latest_xs = self.history.xs[-1]
                latest_ys = self.history.ys[-1]
                pred = self.signal_model.compute_from_params(latest_xs, params)
                log_lk = 0.5 * ((pred - latest_ys) / 0.05)**2

                nll = prior_penalty + log_lk

            return float(nll)

        # Initial guess
        if full_history:
            # A rough guess from grid search or bounds
            x0 = np.array([self.domain_hi / 2.0] * self.dim) # Will need proper init!
            # Example heuristic for NV Center
            x0_dict = {
                "center_frequency": self.history.xs[np.argmin(self.history.ys)],
                "dip_depth": 0.05,
                "linewidth": 0.015,
                "splitting": 0.0,
            }
            x0 = np.array([x0_dict.get(n, 0.0) for n in self.param_names])
        else:
            x0 = self.mu_map

        res = scipy.optimize.minimize(
            lambda th: obj_func(th),
            x0,
            method='L-BFGS-B',
            jac=None # Use 2-point finite differences
        )

        self.mu_map = res.x

        # Approximate Hessian (Inverse Covariance)
        try:
            import numdifftools as nd
            hess = nd.Hessian(lambda th: obj_func(th))(self.mu_map)
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(hess)
            eigvals = np.maximum(eigvals, 1e-6)
            self.sigma_inv = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except Exception:
            # Fallback if Hessian fails or numdifftools is not installed
            if full_history:
                self.sigma_inv = np.eye(self.dim) * 1e4

    def done(self) -> bool:
        return self.step_count >= self.max_steps

    def result(self) -> dict[str, float]:
        res = {}
        for idx, name in enumerate(self.param_names):
            res[name] = float(self.mu_map[idx])
        return res
