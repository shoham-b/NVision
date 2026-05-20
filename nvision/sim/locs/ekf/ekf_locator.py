"""Extended Kalman Filter active learning locator for NV Center ODMR."""

# Reformat imports to ensure compliance with style rules.

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.sim.locs.ekf.belief import EKFBelief, EKFParticleFrequencyBelief
from nvision.sim.locs.ekf.parameter_bounds import linewidth_hwhm_bounds, prepare_ekf_parameter_bounds


class EKFLocator(SequentialBayesianLocator):
    """Parametric Bayesian Locator using analytical Triple-Lorentzian EKF and OED."""

    REQUIRES_BELIEF = True

    def __init__(
        self,
        belief: EKFBelief,
        max_steps: int = 200,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        acquisition: str = "d_optimal",
    ) -> None:
        """Initialize EKFLocator.

        Args:
            belief: The EKF belief distribution.
            max_steps: Maximum inference steps.
            convergence_threshold: Convergence threshold.
            scan_param: Optional scan parameter.
            convergence_params: Optional target convergence parameters.
            convergence_patience_steps: Patience count for convergence.
            noise_std: Optional noise standard deviation.
            noise_max_dev: Optional maximum noise deviation.
            signal_max_span: Optional maximum signal span.
            acquisition: Acquisition strategy ("d_optimal" or "a_optimal").
        """
        super().__init__(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std or 0.05,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )

        from nvision.sim.locs.ekf.acquisition import AOptimalAcquisition, DOptimalAcquisition

        if isinstance(acquisition, str):
            acq_lower = acquisition.lower()
            if "a_optimal" in acq_lower or "a-optimal" in acq_lower:
                self.acquisition_strategy = AOptimalAcquisition()
            else:
                self.acquisition_strategy = DOptimalAcquisition()
        else:
            self.acquisition_strategy = acquisition or DOptimalAcquisition()

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 200,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        acquisition: str = "d_optimal",
        **grid_config: object,
    ) -> EKFLocator:
        """Create EKFLocator using priors mapped from bounds."""
        from nvision.spectra.nv_center import NVCenterLorentzianModel

        signal_model = NVCenterLorentzianModel()
        bounds_phys = prepare_ekf_parameter_bounds(
            dict(parameter_bounds) if parameter_bounds else None,
        )

        freq_bounds = bounds_phys.get("frequency", (2.6e9, 3.1e9))
        linewidth_bounds = linewidth_hwhm_bounds(bounds_phys)
        split_bounds = bounds_phys.get("split", (2.0e6, 3.5e6))
        k_np_bounds = bounds_phys.get("k_np", (2.0, 4.0))
        dip_depth_bounds = bounds_phys.get("dip_depth", (0.1, 1.0))

        # Check for prior distributions
        priors = bounds_phys.get("_priors") if isinstance(bounds_phys, dict) else None

        # Midpoint or prior distribution initialization
        if priors:
            # 1. Frequency (always flat prior spanning the physical bounds)
            f_B = (freq_bounds[0] + freq_bounds[1]) / 2.0 / 1e6  # noqa: N806
            P_freq = ((freq_bounds[1] - freq_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806

            # 2. Hyperfine Split
            if "split" in priors:
                split_val, split_std = priors["split"]
                delta_f_HF = split_val / 1e6  # noqa: N806
                P_split = (split_std / 1e6) ** 2  # noqa: N806
            else:
                delta_f_HF = (split_bounds[0] + split_bounds[1]) / 2.0 / 1e6  # noqa: N806
                P_split = ((split_bounds[1] - split_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806

            # 3. Linewidth
            if "linewidth" in priors:
                lw_val, lw_std = priors["linewidth"]
                Omega = lw_val / 1e6  # noqa: N806
                P_lw = (lw_std / 1e6) ** 2  # noqa: N806
            else:
                Omega = (linewidth_bounds[0] + linewidth_bounds[1]) / 2.0 / 1e6  # noqa: N806
                P_lw = ((linewidth_bounds[1] - linewidth_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806

            # 4. Polarization Contrast Ratio k_NP = 1.0 / k_np
            if "k_np" in priors:
                knp_val, knp_std = priors["k_np"]
                k_NP = 1.0 / knp_val if knp_val != 0 else 1.0  # noqa: N806
                P_knp = (knp_std / (knp_val**2)) ** 2 if knp_val != 0 else 0.1  # noqa: N806
            else:
                k_np_init = (k_np_bounds[0] + k_np_bounds[1]) / 2.0
                k_NP = 1.0 / k_np_init  # noqa: N806
                k_NP_hi = 1.0 / k_np_bounds[0]  # noqa: N806
                k_NP_lo = 1.0 / k_np_bounds[1]  # noqa: N806
                P_knp = (k_NP_hi - k_NP_lo) ** 2 / 12.0  # noqa: N806

            # 5. Amplitude / Dip Depth scale factor: a = dip_depth * k_NP * Omega**2
            if "dip_depth" in priors:
                dd_val, dd_std = priors["dip_depth"]
                a = dd_val * k_NP * (Omega**2)
                P_a = (dd_std * k_NP * (Omega**2)) ** 2  # noqa: N806
            else:
                dip_depth_init = (dip_depth_bounds[0] + dip_depth_bounds[1]) / 2.0
                a = dip_depth_init * k_NP * (Omega**2)
                a_lo = dip_depth_bounds[0] * k_NP * (Omega**2)
                a_hi = dip_depth_bounds[1] * k_NP * (Omega**2)
                P_a = (a_hi - a_lo) ** 2 / 12.0  # noqa: N806
        else:
            f_B = (freq_bounds[0] + freq_bounds[1]) / 2.0 / 1e6  # noqa: N806
            delta_f_HF = (split_bounds[0] + split_bounds[1]) / 2.0 / 1e6  # noqa: N806
            Omega = (linewidth_bounds[0] + linewidth_bounds[1]) / 2.0 / 1e6  # noqa: N806
            k_np_init = (k_np_bounds[0] + k_np_bounds[1]) / 2.0
            k_NP = 1.0 / k_np_init  # noqa: N806
            dip_depth_init = (dip_depth_bounds[0] + dip_depth_bounds[1]) / 2.0
            a = dip_depth_init * k_NP * (Omega**2)

            P_freq = ((freq_bounds[1] - freq_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806
            P_split = ((split_bounds[1] - split_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806
            P_lw = ((linewidth_bounds[1] - linewidth_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806
            k_NP_hi = 1.0 / k_np_bounds[0]  # noqa: N806
            k_NP_lo = 1.0 / k_np_bounds[1]  # noqa: N806
            P_knp = (k_NP_hi - k_NP_lo) ** 2 / 12.0  # noqa: N806
            a_lo = dip_depth_bounds[0] * k_NP * (Omega**2)
            a_hi = dip_depth_bounds[1] * k_NP * (Omega**2)
            P_a = (a_hi - a_lo) ** 2 / 12.0  # noqa: N806

        theta_initial = np.array([f_B, delta_f_HF, a, Omega, k_NP])

        # Prior covariance based on uniform or Gaussian variance
        P_initial = np.zeros((5, 5))  # noqa: N806
        P_initial[0, 0] = max(1e-6, P_freq)
        P_initial[1, 1] = max(1e-6, P_split)
        P_initial[2, 2] = max(1e-6, P_a)
        P_initial[3, 3] = max(1e-6, P_lw)
        P_initial[4, 4] = max(1e-6, P_knp)

        # Create the EKF belief distribution
        R = (noise_std**2) if noise_std is not None else (0.05**2)  # noqa: N806
        belief = EKFBelief(
            model=signal_model,
            theta_initial=theta_initial,
            P_initial=P_initial,
            R=R,
            parameter_bounds=bounds_phys,
        )

        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            acquisition=acquisition,
        )

    def _acquisition_bounds(self) -> tuple[float, float]:
        """Stub for acquisition bounds resolution."""
        return self._full_domain_lo, self._full_domain_hi

    def observe(self, obs: Observation) -> None:
        """Route observation, denormalizing the x coordinate for the EKF belief."""
        lo, hi = self._acquisition_bounds()
        x_hz = lo + obs.x * (hi - lo)

        obs_physical = Observation(
            x=x_hz,
            signal_value=obs.signal_value,
            noise_std=obs.noise_std,
            frequency_noise_model=obs.frequency_noise_model,
        )

        self.belief.update(obs_physical)

    def _acquire(self) -> float:
        """Run OED acquisition using the narrowed parameter bounds."""
        lo, hi = self._acquisition_bounds()

        # Generate candidates along physical scan axis
        candidates_hz = np.linspace(lo, hi, 1000)
        candidates_mhz = candidates_hz / 1e6

        priors = self.belief.parameter_bounds.get("_priors")
        if isinstance(priors, dict):
            for key in ("split", "linewidth", "k_np", "dip_depth"):
                assert key in priors, f"Missing key in priors: {key}"
                assert isinstance(priors[key], tuple), f"Invalid value for {key} in priors: not a tuple."
                assert len(priors[key]) == 2, f"Invalid value for {key} in priors: tuple length is not 2."

        # Ensure `parameter_bounds`, `theta_hat`, `P`, and `R` are resolved.
        assert hasattr(self.belief, "parameter_bounds"), "`parameter_bounds` is not defined in EKFBelief."
        assert hasattr(self.belief, "theta_hat"), "`theta_hat` is not defined in EKFBelief."
        assert hasattr(self.belief, "P"), "`P` is not defined in EKFBelief."
        assert hasattr(self.belief, "R"), "`R` is not defined in EKFBelief."

        # Thompson sampling: pick a frequency particle proportional to its weight to drive exploration
        theta_sample = self.belief.theta_hat.copy()
        if hasattr(self.belief, "_frequency_particles") and hasattr(self.belief, "_frequency_weights"):
            idx = np.random.choice(len(self.belief._frequency_particles), p=self.belief._frequency_weights)
            theta_sample[0] = self.belief._frequency_particles[idx] / 1e6

        # Update `select_next_frequency` call to match expected arguments.
        next_f_mhz = self.acquisition_strategy.select_next_frequency(
            theta_hat=theta_sample,
            P=self.belief.P,
            R=self.belief.R,
            f_candidates=candidates_mhz,
        )

        return next_f_mhz * 1e6


class EKFDOptimalLocator(EKFLocator):
    """EKF Locator configured to use D-Optimal acquisition."""

    @classmethod
    def create(cls, **config: object) -> EKFDOptimalLocator:
        """Create EKFDOptimalLocator enforcing d_optimal strategy."""
        config["acquisition"] = "d_optimal"
        return super().create(**config)


class EKFAOptimalLocator(EKFLocator):
    """EKF Locator configured to use A-Optimal acquisition."""

    @classmethod
    def create(cls, **config: object) -> EKFAOptimalLocator:
        """Create EKFAOptimalLocator enforcing a_optimal strategy."""
        config["acquisition"] = "a_optimal"
        return super().create(**config)


class EKFParticleFrequencyLocator(EKFLocator):
    """EKF Locator with particles for frequency and EKF for other parameters."""

    def __init__(
        self,
        belief: EKFParticleFrequencyBelief,
        max_steps: int = 200,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        acquisition: str = "d_optimal",
    ) -> None:
        super().__init__(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            acquisition=acquisition,
        )

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 200,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        acquisition: str = "d_optimal",
        num_particles: int = 1000,
        **grid_config: object,
    ) -> EKFParticleFrequencyLocator:
        """Create EKFParticleFrequencyLocator using the hybrid belief."""
        from nvision.spectra.nv_center import NVCenterLorentzianModel

        signal_model = NVCenterLorentzianModel()
        bounds_phys = prepare_ekf_parameter_bounds(
            dict(parameter_bounds) if parameter_bounds else None,
        )

        freq_bounds = bounds_phys.get("frequency", (2.6e9, 3.1e9))
        linewidth_bounds = linewidth_hwhm_bounds(bounds_phys)
        split_bounds = bounds_phys.get("split", (2.0e6, 3.5e6))
        k_np_bounds = bounds_phys.get("k_np", (2.0, 4.0))
        dip_depth_bounds = bounds_phys.get("dip_depth", (0.1, 1.0))

        # Check for prior distributions
        priors = bounds_phys.get("_priors") if isinstance(bounds_phys, dict) else None

        # Midpoint or prior distribution initialization
        if priors:
            # 1. Frequency (always flat prior spanning the physical bounds)
            f_B = (freq_bounds[0] + freq_bounds[1]) / 2.0 / 1e6  # noqa: N806
            P_freq = ((freq_bounds[1] - freq_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806

            # 2. Hyperfine Split
            if "split" in priors:
                split_val, split_std = priors["split"]
                delta_f_HF = split_val / 1e6  # noqa: N806
                P_split = (split_std / 1e6) ** 2  # noqa: N806
            else:
                delta_f_HF = (split_bounds[0] + split_bounds[1]) / 2.0 / 1e6  # noqa: N806
                P_split = ((split_bounds[1] - split_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806

            # 3. Linewidth
            if "linewidth" in priors:
                lw_val, lw_std = priors["linewidth"]
                Omega = lw_val / 1e6  # noqa: N806
                P_lw = (lw_std / 1e6) ** 2  # noqa: N806
            else:
                Omega = (linewidth_bounds[0] + linewidth_bounds[1]) / 2.0 / 1e6  # noqa: N806
                P_lw = ((linewidth_bounds[1] - linewidth_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806

            # 4. Polarization Contrast Ratio k_NP = 1.0 / k_np
            if "k_np" in priors:
                knp_val, knp_std = priors["k_np"]
                k_NP = 1.0 / knp_val if knp_val != 0 else 1.0  # noqa: N806
                P_knp = (knp_std / (knp_val**2)) ** 2 if knp_val != 0 else 0.1  # noqa: N806
            else:
                k_np_init = (k_np_bounds[0] + k_np_bounds[1]) / 2.0
                k_NP = 1.0 / k_np_init  # noqa: N806
                k_NP_hi = 1.0 / k_np_bounds[0]  # noqa: N806
                k_NP_lo = 1.0 / k_np_bounds[1]  # noqa: N806
                P_knp = (k_NP_hi - k_NP_lo) ** 2 / 12.0  # noqa: N806

            # 5. Amplitude / Dip Depth scale factor: a = dip_depth * k_NP * Omega**2
            if "dip_depth" in priors:
                dd_val, dd_std = priors["dip_depth"]
                a = dd_val * k_NP * (Omega**2)
                P_a = (dd_std * k_NP * (Omega**2)) ** 2  # noqa: N806
            else:
                dip_depth_init = (dip_depth_bounds[0] + dip_depth_bounds[1]) / 2.0
                a = dip_depth_init * k_NP * (Omega**2)
                a_lo = dip_depth_bounds[0] * k_NP * (Omega**2)
                a_hi = dip_depth_bounds[1] * k_NP * (Omega**2)
                P_a = (a_hi - a_lo) ** 2 / 12.0  # noqa: N806
        else:
            f_B = (freq_bounds[0] + freq_bounds[1]) / 2.0 / 1e6  # noqa: N806
            delta_f_HF = (split_bounds[0] + split_bounds[1]) / 2.0 / 1e6  # noqa: N806
            Omega = (linewidth_bounds[0] + linewidth_bounds[1]) / 2.0 / 1e6  # noqa: N806
            k_np_init = (k_np_bounds[0] + k_np_bounds[1]) / 2.0
            k_NP = 1.0 / k_np_init  # noqa: N806
            dip_depth_init = (dip_depth_bounds[0] + dip_depth_bounds[1]) / 2.0
            a = dip_depth_init * k_NP * (Omega**2)

            P_freq = ((freq_bounds[1] - freq_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806
            P_split = ((split_bounds[1] - split_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806
            P_lw = ((linewidth_bounds[1] - linewidth_bounds[0]) / 1e6) ** 2 / 12.0  # noqa: N806
            k_NP_hi = 1.0 / k_np_bounds[0]  # noqa: N806
            k_NP_lo = 1.0 / k_np_bounds[1]  # noqa: N806
            P_knp = (k_NP_hi - k_NP_lo) ** 2 / 12.0  # noqa: N806
            a_lo = dip_depth_bounds[0] * k_NP * (Omega**2)
            a_hi = dip_depth_bounds[1] * k_NP * (Omega**2)
            P_a = (a_hi - a_lo) ** 2 / 12.0  # noqa: N806

        theta_initial = np.array([f_B, delta_f_HF, a, Omega, k_NP])

        # Prior covariance based on uniform or Gaussian variance
        P_initial = np.zeros((5, 5))  # noqa: N806
        P_initial[0, 0] = max(1e-6, P_freq)
        P_initial[1, 1] = max(1e-6, P_split)
        P_initial[2, 2] = max(1e-6, P_a)
        P_initial[3, 3] = max(1e-6, P_lw)
        P_initial[4, 4] = max(1e-6, P_knp)

        # Create the hybrid EKF+particle belief distribution
        R = (noise_std**2) if noise_std is not None else (0.05**2)  # noqa: N806
        belief = EKFParticleFrequencyBelief(
            model=signal_model,
            theta_initial=theta_initial,
            P_initial=P_initial,
            R=R,
            parameter_bounds=bounds_phys,
            num_particles=num_particles,
        )

        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            acquisition=acquisition,
        )
