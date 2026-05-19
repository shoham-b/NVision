"""Extended Kalman Filter belief distribution for ODMR."""

from __future__ import annotations

import math
from collections.abc import Sequence
import numpy as np
from scipy.special import erf

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.models.observation import Observation
from nvision.spectra.signal import SignalModel
from nvision.sim.locs.ekf.models import mu
from nvision.sim.locs.ekf.filter import ekf_update


class EKFBelief(AbstractMarginalDistribution):
    """Belief distribution using a multivariate Gaussian updated via EKF."""

    def __init__(
        self,
        model: SignalModel,
        theta_initial: np.ndarray,
        P_initial: np.ndarray,
        R: float,
        parameter_bounds: dict[str, tuple[float, float]],
        last_obs: Observation | None = None,
    ) -> None:
        """Initialize EKFBelief.

        Args:
            model: Stateless signal model.
            theta_initial: Initial EKF state vector (shape: 5,).
            P_initial: Initial EKF covariance matrix (shape: 5x5).
            R: Measurement noise variance.
            parameter_bounds: Prior parameter bounds.
            last_obs: Optional last observation.
        """
        super().__init__(model, last_obs)
        self.theta_hat = theta_initial.copy()
        self.P = P_initial.copy()
        self.R = R
        self._physical_bounds = dict(parameter_bounds)

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds for each parameter."""
        return self._physical_bounds

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds alias for compatibility."""
        return self.physical_param_bounds

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Shrink physical bounds for a parameter (e.g. after sweep)."""
        self._physical_bounds[param_name] = (new_lo, new_hi)

    def update(self, obs: Observation) -> None:
        """Incremental Bayesian update from a new observation via EKF."""
        self.last_obs = obs
        f_mhz = obs.x / 1e6
        self.theta_hat, self.P = ekf_update(self.theta_hat, self.P, obs.signal_value, f_mhz, self.R)

    def batch_update(self, observations: Sequence[Observation]) -> None:
        """Batch update for EKF belief. We ignore the coarse sweep buffer to prevent covariance explosion."""
        pass

    def estimates(self) -> dict[str, float]:
        """Get current parameter estimates in standard physical units (Hz)."""
        f_B = self.theta_hat[0]  # noqa: N806
        delta_f_HF = self.theta_hat[1]  # noqa: N806
        a = self.theta_hat[2]
        Omega = self.theta_hat[3]  # noqa: N806
        k_NP = self.theta_hat[4]  # noqa: N806

        frequency = f_B * 1e6
        linewidth = Omega * 1e6
        split = delta_f_HF * 1e6
        k_np = 1.0 / k_NP if k_NP != 0 else 1.0
        dip_depth = a / (k_NP * Omega**2) if (k_NP != 0 and Omega != 0) else 0.0

        return {
            "frequency": frequency,
            "linewidth": linewidth,
            "split": split,
            "k_np": k_np,
            "dip_depth": dip_depth,
        }

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        """Compute empirical uncertainty (standard deviations) from covariance."""
        f_B = self.theta_hat[0]  # noqa: N806
        delta_f_HF = self.theta_hat[1]  # noqa: N806
        a = self.theta_hat[2]
        Omega = self.theta_hat[3]  # noqa: N806
        k_NP = self.theta_hat[4]  # noqa: N806

        var_f_B = self.P[0, 0]  # noqa: N806
        var_delta = self.P[1, 1]
        var_a = self.P[2, 2]
        var_Omega = self.P[3, 3]  # noqa: N806
        var_k = self.P[4, 4]

        std_freq = math.sqrt(max(0.0, var_f_B)) * 1e6
        std_lw = math.sqrt(max(0.0, var_Omega)) * 1e6
        std_split = math.sqrt(max(0.0, var_delta)) * 1e6

        if k_NP != 0:
            std_k_np = math.sqrt(max(0.0, var_k)) / (k_NP**2)
        else:
            std_k_np = 0.0

        if k_NP != 0 and Omega != 0:
            std_dip_depth = math.sqrt(max(0.0, var_a)) / (k_NP * Omega**2)
        else:
            std_dip_depth = 0.0

        names = list(self.model.parameter_names())
        data = {
            "frequency": std_freq,
            "linewidth": std_lw,
            "split": std_split,
            "k_np": std_k_np,
            "dip_depth": std_dip_depth,
        }
        return ParameterValues.from_mapping(names, data)

    def entropy(self) -> float:
        """Compute total entropy of the Gaussian belief."""
        try:
            sign, logdet = np.linalg.slogdet(2 * np.pi * np.e * self.P)
            if sign > 0:
                return 0.5 * logdet
        except np.linalg.LinAlgError:
            pass

        diag = np.diagonal(self.P)
        diag_pos = diag[diag > 0]
        if len(diag_pos) > 0:
            return 0.5 * float(np.sum(np.log(2 * np.pi * np.e * diag_pos)))
        return 0.0

    def converged(self, threshold: float) -> bool:
        """Check if all parameters have converged below threshold."""
        uncertainties = self.uncertainty()
        for name in self.model.parameter_names():
            if name not in uncertainties:
                continue
            lo, hi = self._physical_bounds.get(name, (0.0, 0.0))
            width = hi - lo
            if width > 0 and uncertainties[name] / width >= threshold:
                return False
        return True

    def copy(self) -> EKFBelief:
        """Create a deep copy of this belief for snapshotting."""
        copied = EKFBelief(
            model=self.model,
            theta_initial=self.theta_hat,
            P_initial=self.P,
            R=self.R,
            parameter_bounds=self._physical_bounds,
            last_obs=self.last_obs,
        )
        copied.resampled = self.resampled
        return copied

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        """Draw n joint samples from the posterior distribution."""
        P_reg = self.P.copy()  # noqa: N806
        min_eig = np.min(np.linalg.eigvals(P_reg))
        if min_eig < 1e-12:
            P_reg += np.eye(5) * (abs(min_eig) + 1e-10)

        raw_samples = np.random.multivariate_normal(self.theta_hat, P_reg, n)

        f_B_s = raw_samples[:, 0]  # noqa: N806
        delta_f_s = raw_samples[:, 1]
        a_s = raw_samples[:, 2]
        Omega_s = raw_samples[:, 3]  # noqa: N806
        k_NP_s = raw_samples[:, 4]  # noqa: N806

        frequency = f_B_s * 1e6
        linewidth = Omega_s * 1e6
        split = delta_f_s * 1e6
        k_np = np.zeros_like(k_NP_s)
        mask_k = k_NP_s != 0
        k_np[mask_k] = 1.0 / k_NP_s[mask_k]
        k_np[~mask_k] = 1.0

        dip_depth = np.zeros_like(a_s)
        mask_d = (k_NP_s != 0) & (Omega_s != 0)
        dip_depth[mask_d] = a_s[mask_d] / (k_NP_s[mask_d] * Omega_s[mask_d] ** 2)

        names = list(self.model.parameter_names())
        data = {
            "frequency": frequency,
            "linewidth": linewidth,
            "split": split,
            "k_np": k_np,
            "dip_depth": dip_depth,
        }
        return ParameterValues.from_mapping(names, data)

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate marginal PDF of a parameter at points x."""
        est = self.estimates()
        unc = self.uncertainty()
        mean = est.get(param_name, 0.0)
        std = unc.get(param_name, 1.0)
        if std <= 0:
            std = 1e-9
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate marginal CDF of a parameter at points x."""
        est = self.estimates()
        unc = self.uncertainty()
        mean = est.get(param_name, 0.0)
        std = unc.get(param_name, 1.0)
        if std <= 0:
            std = 1e-9
        return 0.5 * (1.0 + erf((x - mean) / (std * math.sqrt(2.0))))

    def __call__(self, x: float) -> float:
        """Evaluate predicted PL signal at physical x (Hz)."""
        f_mhz = float(x) / 1e6
        return float(mu(f_mhz, self.theta_hat))


class EKFParticleFrequencyBelief(AbstractMarginalDistribution):
    """Hybrid belief using particles for frequency and EKF for other parameters.

    This belief maintains:
    - A particle filter for the frequency parameter (index 0)
    - EKF (Gaussian) for the remaining 4 parameters (split, amplitude, linewidth, k_np)
    """

    def __init__(
        self,
        model: SignalModel,
        theta_initial: np.ndarray,
        P_initial: np.ndarray,
        R: float,
        parameter_bounds: dict[str, tuple[float, float]],
        num_particles: int = 1000,
        last_obs: Observation | None = None,
    ) -> None:
        """Initialize EKFParticleFrequencyBelief.

        Args:
            model: Stateless signal model.
            theta_initial: Initial EKF state vector (shape: 5,).
            P_initial: Initial EKF covariance matrix (shape: 5x5).
            R: Measurement noise variance.
            parameter_bounds: Prior parameter bounds.
            num_particles: Number of particles for frequency.
            last_obs: Optional last observation.
        """
        super().__init__(model, last_obs)

        # EKF state for parameters 1-4 (split, amplitude, linewidth, k_np)
        self.theta_hat = theta_initial.copy()
        self.P = P_initial.copy()
        self.R = R
        self._physical_bounds = dict(parameter_bounds)

        # Particle filter for frequency (parameter 0)
        self.num_particles = num_particles
        freq_bounds = parameter_bounds.get("frequency", (2.6e9, 3.1e9))
        self._frequency_particles = np.random.uniform(freq_bounds[0], freq_bounds[1], num_particles)
        self._frequency_weights = np.ones(num_particles) / num_particles

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds for each parameter."""
        return self._physical_bounds

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds alias for compatibility."""
        return self.physical_param_bounds

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Shrink physical bounds for a parameter (e.g. after sweep)."""
        self._physical_bounds[param_name] = (new_lo, new_hi)
        if param_name == "frequency":
            self._frequency_particles = np.clip(self._frequency_particles, new_lo, new_hi)

    def update(self, obs: Observation) -> None:
        """Incremental Bayesian update from a new observation.

        Updates frequency particles via likelihood weighting and
        other parameters via EKF. The EKF operates on the full 5-element
        state vector, with frequency overridden by the particle mean.
        """
        self.last_obs = obs
        f_mhz = obs.x / 1e6

        # Update frequency particles via likelihood
        freq_particles_mhz = self._frequency_particles / 1e6
        # For each frequency particle, compute prediction using current EKF state
        # for other parameters
        predictions = []
        for f_p in freq_particles_mhz:
            theta_temp = self.theta_hat.copy()
            theta_temp[0] = f_p  # Use particle frequency
            predictions.append(mu(f_p, theta_temp))
        predictions = np.array(predictions)

        # Compute likelihoods
        likelihoods = np.exp(-0.5 * ((obs.signal_value - predictions) / np.sqrt(self.R)) ** 2)
        likelihoods /= np.sum(likelihoods)

        # Update frequency weights
        self._frequency_weights *= likelihoods
        weight_sum = np.sum(self._frequency_weights)
        if weight_sum > 1e-100:
            self._frequency_weights /= weight_sum
        else:
            self._frequency_weights = np.ones(self.num_particles) / self.num_particles

        # Resample frequency particles if ESS is low
        ess = 1.0 / np.sum(self._frequency_weights**2)
        if ess < 0.5 * self.num_particles:
            indices = np.random.choice(self.num_particles, self.num_particles, p=self._frequency_weights)
            self._frequency_particles = self._frequency_particles[indices]
            self._frequency_weights = np.ones(self.num_particles) / self.num_particles

        # Update EKF for all parameters using weighted mean frequency
        f_mean_mhz = np.average(freq_particles_mhz, weights=self._frequency_weights)
        self.theta_hat[0] = f_mean_mhz
        self.theta_hat, self.P = ekf_update(self.theta_hat, self.P, obs.signal_value, f_mean_mhz, self.R)
        # Override frequency with particle mean (EKF may have moved it)
        self.theta_hat[0] = f_mean_mhz

    def batch_update(self, observations: Sequence[Observation]) -> None:
        """Batch update - ignore coarse sweep buffer to prevent covariance explosion."""
        pass

    def estimates(self) -> dict[str, float]:
        """Get current parameter estimates in standard physical units (Hz)."""
        f_B = np.average(self._frequency_particles / 1e6, weights=self._frequency_weights)  # noqa: N806
        delta_f_HF = self.theta_hat[1]  # noqa: N806
        a = self.theta_hat[2]
        Omega = self.theta_hat[3]  # noqa: N806
        k_NP = self.theta_hat[4]  # noqa: N806

        frequency = f_B * 1e6
        linewidth = Omega * 1e6
        split = delta_f_HF * 1e6
        k_np = 1.0 / k_NP if k_NP != 0 else 1.0
        dip_depth = a / (k_NP * Omega**2) if (k_NP != 0 and Omega != 0) else 0.0

        return {
            "frequency": frequency,
            "linewidth": linewidth,
            "split": split,
            "k_np": k_np,
            "dip_depth": dip_depth,
        }

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        """Compute empirical uncertainty (standard deviations)."""
        # Frequency uncertainty from particles
        f_B = np.average(self._frequency_particles / 1e6, weights=self._frequency_weights)  # noqa: N806
        var_f = np.average((self._frequency_particles / 1e6 - f_B) ** 2, weights=self._frequency_weights)
        std_freq = np.sqrt(max(0.0, var_f)) * 1e6

        # Other parameters from EKF covariance
        delta_f_HF = self.theta_hat[1]  # noqa: N806
        a = self.theta_hat[2]
        Omega = self.theta_hat[3]  # noqa: N806
        k_NP = self.theta_hat[4]  # noqa: N806

        var_delta = self.P[1, 1]
        var_a = self.P[2, 2]
        var_Omega = self.P[3, 3]  # noqa: N806
        var_k = self.P[4, 4]

        std_lw = math.sqrt(max(0.0, var_Omega)) * 1e6
        std_split = math.sqrt(max(0.0, var_delta)) * 1e6

        if k_NP != 0:
            std_k_np = math.sqrt(max(0.0, var_k)) / (k_NP**2)
        else:
            std_k_np = 0.0

        if k_NP != 0 and Omega != 0:
            std_dip_depth = math.sqrt(max(0.0, var_a)) / (k_NP * Omega**2)
        else:
            std_dip_depth = 0.0

        names = list(self.model.parameter_names())
        data = {
            "frequency": std_freq,
            "linewidth": std_lw,
            "split": std_split,
            "k_np": std_k_np,
            "dip_depth": std_dip_depth,
        }
        return ParameterValues.from_mapping(names, data)

    def entropy(self) -> float:
        """Compute total entropy of the hybrid belief."""
        # Approximate entropy as sum of frequency particle entropy + EKF entropy
        f_B = np.average(self._frequency_particles / 1e6, weights=self._frequency_weights)  # noqa: N806
        var_f = np.average((self._frequency_particles / 1e6 - f_B) ** 2, weights=self._frequency_weights)
        freq_entropy = 0.5 * np.log(2 * np.pi * np.e * max(1e-12, var_f))

        # EKF entropy for parameters 1-4
        try:
            sign, logdet = np.linalg.slogdet(2 * np.pi * np.e * self.P[1:, 1:])
            if sign > 0:
                ekf_entropy = 0.5 * logdet
            else:
                ekf_entropy = 0.0
        except np.linalg.LinAlgError:
            ekf_entropy = 0.0

        return freq_entropy + ekf_entropy

    def converged(self, threshold: float) -> bool:
        """Check if all parameters have converged below threshold."""
        uncertainties = self.uncertainty()
        for name in self.model.parameter_names():
            if name not in uncertainties:
                continue
            lo, hi = self._physical_bounds.get(name, (0.0, 0.0))
            width = hi - lo
            if width > 0 and uncertainties[name] / width >= threshold:
                return False
        return True

    def copy(self) -> EKFParticleFrequencyBelief:
        """Create a deep copy of this belief for snapshotting."""
        copied = EKFParticleFrequencyBelief(
            model=self.model,
            theta_initial=self.theta_hat,
            P_initial=self.P,
            R=self.R,
            parameter_bounds=self._physical_bounds,
            num_particles=self.num_particles,
            last_obs=self.last_obs,
        )
        copied._frequency_particles = self._frequency_particles.copy()
        copied._frequency_weights = self._frequency_weights.copy()
        copied.resampled = self.resampled
        return copied

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        """Draw n joint samples from the posterior distribution."""
        # Sample frequency from particles
        freq_indices = np.random.choice(self.num_particles, n, p=self._frequency_weights)
        frequency_samples = self._frequency_particles[freq_indices]

        # Sample other parameters from EKF Gaussian
        P_reg = self.P[1:, 1:].copy()  # noqa: N806
        min_eig = np.min(np.linalg.eigvals(P_reg))
        if min_eig < 1e-12:
            P_reg += np.eye(4) * (abs(min_eig) + 1e-10)

        raw_samples = np.random.multivariate_normal(self.theta_hat[1:], P_reg, n)

        delta_f_s = raw_samples[:, 0]
        a_s = raw_samples[:, 1]
        Omega_s = raw_samples[:, 2]
        k_NP_s = raw_samples[:, 3]  # noqa: N806

        frequency = frequency_samples
        linewidth = Omega_s * 1e6
        split = delta_f_s * 1e6
        k_np = np.zeros_like(k_NP_s)
        mask_k = k_NP_s != 0
        k_np[mask_k] = 1.0 / k_NP_s[mask_k]
        k_np[~mask_k] = 1.0

        dip_depth = np.zeros_like(a_s)
        mask_d = (k_NP_s != 0) & (Omega_s != 0)
        dip_depth[mask_d] = a_s[mask_d] / (k_NP_s[mask_d] * Omega_s[mask_d] ** 2)

        names = list(self.model.parameter_names())
        data = {
            "frequency": frequency,
            "linewidth": linewidth,
            "split": split,
            "k_np": k_np,
            "dip_depth": dip_depth,
        }
        return ParameterValues.from_mapping(names, data)

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate marginal PDF of a parameter at points x."""
        est = self.estimates()
        unc = self.uncertainty()
        mean = est.get(param_name, 0.0)
        std = unc.get(param_name, 1.0)
        if std <= 0:
            std = 1e-9
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        """Evaluate marginal CDF of a parameter at points x."""
        est = self.estimates()
        unc = self.uncertainty()
        mean = est.get(param_name, 0.0)
        std = unc.get(param_name, 1.0)
        if std <= 0:
            std = 1e-9
        return 0.5 * (1.0 + erf((x - mean) / (std * math.sqrt(2.0))))

    def __call__(self, x: float) -> float:
        """Evaluate predicted PL signal at physical x (Hz)."""
        f_mhz = float(x) / 1e6
        return float(mu(f_mhz, self.theta_hat))
