"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math
import os

import numpy as np

from nvision.belief.smc_marginal import _inverse_sum_squares
from nvision.models.observation import Observation
from nvision.sim.defaults import NVISION_CONVERGENCE_THRESHOLD, NVISION_SMC_CANDIDATE_STEP_HZ
from nvision.sim.locs.bayesian.dip_detection import identify_dip_candidates
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

# Minimum number of consecutive converged checks before declaring convergence.
# Prevents false early stops on the first measurement, especially with no noise.
NVISION_CONVERGENCE_PATIENCE: int = int(os.getenv("NVISION_CONVERGENCE_PATIENCE", "8"))


class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Uses Expected Information Gain (prediction variance disagreement) to select
    the next measurement point from a fine frequency grid. No JAX gradient ascent
    is performed — the chunked EIG search on the belief is sufficient.
    """

    REQUIRES_BELIEF = True
    USES_SWEEP_MAX_STEPS = True

    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        noise_std: float = 0.02,
        candidate_step_hz: float | None = None,
        convergence_patience_steps: int = NVISION_CONVERGENCE_PATIENCE,
    ) -> None:
        super().__init__(
            belief,
            max_steps,
            convergence_threshold,
            scan_param,
            noise_std=noise_std,
            convergence_patience_steps=convergence_patience_steps,
        )
        self.candidate_step_hz: float = (
            float(candidate_step_hz) if candidate_step_hz is not None else NVISION_SMC_CANDIDATE_STEP_HZ
        )

        # We handle resampling manually to check convergence at the right moment
        if hasattr(self.belief, "auto_resample"):
            self.belief.auto_resample = False
        self._is_converged = False

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        parameter_bounds=None,
        noise_std: float | None = None,
        candidate_step_hz: float | None = None,
        convergence_patience_steps: int = NVISION_CONVERGENCE_PATIENCE,
        **grid_config,
    ):
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            noise_std=noise_std,
            candidate_step_hz=candidate_step_hz,
            convergence_patience_steps=convergence_patience_steps,
        )

    def _generate_candidates(self, num_candidates: int | None = None) -> np.ndarray:
        """Generate candidates spanning the whole frequency spectrum.

        When ``num_candidates`` is not provided, compute it dynamically from
        the acquisition bounds so that the step resolution is two decimal
        places finer than the order of magnitude of the range.
        """
        if num_candidates is None:
            lo, hi = self._acquisition_bounds()
            range_val = float(hi - lo)
            if range_val <= 0:
                num_candidates = 1
            else:
                magnitude = math.floor(math.log10(range_val))
                resolution = 10 ** (magnitude - 2)
                num_candidates = max(1, math.ceil(range_val / resolution)) + 1
        return super()._generate_candidates(num_candidates)

    def _thin_candidates_by_step(self, candidates: np.ndarray) -> np.ndarray:
        """Return a subset of *candidates* (physical space) with minimum physical spacing.

        Walks the sorted candidate array once and keeps a candidate only when it
        is at least ``candidate_step_hz`` away from the previously kept one.
        This is O(n) and preserves the first and last candidates so the full
        acquisition range is always represented.
        """
        if len(candidates) <= 1:
            return candidates
        kept: list[int] = [0]
        for i in range(1, len(candidates) - 1):
            if candidates[i] - candidates[kept[-1]] >= self.candidate_step_hz:
                kept.append(i)
        kept.append(len(candidates) - 1)
        return candidates[np.array(kept, dtype=np.intp)]

    def _acquire(self) -> float:
        """Select the next measurement point by maximizing EIG over a frequency grid."""
        lo, hi = self._acquisition_bounds()
        if hi <= lo:
            return float(lo)

        # Retrieve candidates directly from the belief (slope-targeted epoch grid)
        candidates = self.belief.get_candidates()

        # Thin candidates to minimum physical step spacing.
        # The epoch grid window is ±3σ_f, so candidate count ≈ 6σ_f / step_hz:
        # many candidates early (large σ_f), few near convergence (σ_f ≈ step_hz).
        candidates = self._thin_candidates_by_step(candidates)

        best = self.belief.select_max_information_gain(candidates, 1, noise_std=self._noise_std)
        eig_choice = float(best[0]) if len(best) > 0 else float(candidates[len(candidates) // 2])

        # Mix EIG with dip-observation-biased exploration.
        # The uniform exploration probability decays exponentially to focus on EIG as the scan progresses.
        decay = np.exp(-self.inference_step_count / 25.0)
        rand_val = np.random.rand()
        if rand_val < 0.1 * decay:
            # Explore globally uniformly to find missing peaks (probability decays over time)
            return float(np.random.uniform(lo, hi))
        elif rand_val < 0.2:
            # Dip-observation biased sampling: find the empirically lowest measured signal values
            # and draw near one of them. This corrects for a biased posterior that has drifted
            # away from the true dip location.
            obs_list = getattr(self.belief, "_observations", [])
            if len(obs_list) >= 5:
                # Use precomputed/cached dip centers from the belief (calculated only during resampling)
                # to satisfy "dip detection only upon resampling" and avoid massive sorting overhead.
                dip_centers = getattr(self.belief, "_dip_centers", None)
                if dip_centers is None:
                    # Fallback (e.g. if using a belief type that does not precompute them)
                    rescale_maps = self.belief._rescale_maps
                    if "frequency" not in rescale_maps:
                        raise RuntimeError(
                            f"{type(self.belief).__name__} is missing _rescale_maps['frequency']. "
                            "Ensure physical_param_bounds includes 'frequency' at construction."
                        )
                    freq_rescale = rescale_maps["frequency"]
                    obs_xs_phys = freq_rescale.to_phys(np.array([o.x for o in obs_list]))
                    obs_ys = np.array([o.signal_value for o in obs_list])
                    if (
                        hasattr(self.belief, "estimated_noise_std")
                        and getattr(self.belief, "noise_model", None) is not None
                    ):
                        noise_std = self.belief.estimated_noise_std()
                        noise_std_unc = self.belief.noise_std_uncertainty(noise_std)
                    else:
                        noise_std = self._noise_std
                        noise_std_unc = 0.0

                    phys_bounds = getattr(
                        self.belief, "physical_param_bounds", getattr(self.belief, "parameter_bounds", {})
                    )
                    lw_key = "linewidth" if "linewidth" in phys_bounds else "fwhm_total"
                    max_linewidth_hz = phys_bounds[lw_key][1]
                    max_split_hz = phys_bounds["split"][1] if "split" in phys_bounds else None

                    per_particle_sigmas = None
                    particle_weights = None
                    if hasattr(self.belief, "_weights"):
                        particle_weights = self.belief._weights
                        if getattr(self.belief, "_use_rao_blackwell_noise", False):
                            per_particle_sigmas = np.sqrt(self.belief._noise_betas / (self.belief._noise_alphas + 0.5))
                        elif hasattr(self.belief, "_param_names") and "noise_sigma" in self.belief._param_names:
                            idx = self.belief._param_names.index("noise_sigma")
                            raw_sigmas = self.belief._particles[:, idx]
                            if (
                                hasattr(self.belief, "physical_param_bounds")
                                and "noise_sigma" in self.belief.physical_param_bounds
                            ):
                                lo_ns, hi_ns = self.belief.physical_param_bounds["noise_sigma"]
                                per_particle_sigmas = lo_ns + raw_sigmas * (hi_ns - lo_ns)
                            else:
                                per_particle_sigmas = raw_sigmas

                    dip_candidates = identify_dip_candidates(
                        obs_xs_phys,
                        obs_ys,
                        noise_std,
                        max_linewidth_hz,
                        noise_std_unc=noise_std_unc,
                        per_particle_sigmas=per_particle_sigmas,
                        particle_weights=particle_weights,
                        max_split_hz=max_split_hz,
                    )
                    dip_centers = [c.centroid_hz for c in dip_candidates]

                valid_dip_centers = [c for c in dip_centers if lo <= c <= hi]
                if valid_dip_centers:
                    # Pick a random dip centroid and jitter within ±5 MHz around it, keeping it strictly within [lo, hi]
                    center = float(np.random.choice(valid_dip_centers))
                    j_min = max(-5e6, lo - center)
                    j_max = min(5e6, hi - center)
                    if j_max >= j_min:
                        jitter = float(np.random.uniform(j_min, j_max))
                        val = center + jitter
                        if not (lo <= val <= hi):
                            raise ValueError(f"Jittered dip value {val} is outside acquisition bounds {(lo, hi)}")
                        return val
                    else:
                        if not (lo <= center <= hi):
                            raise ValueError(f"Dip center {center} is outside acquisition bounds {(lo, hi)}")
                        return center

            # Fallback: Thompson sampling from posterior particles
            if hasattr(self.belief, "_particles") and hasattr(self.belief, "_weights"):
                weights = self.belief._weights
                if np.sum(weights) > 0:
                    idx = int(np.random.choice(len(weights), p=weights))
                    param_names = getattr(self.belief, "_param_names", [])
                    scan_param = self._scan_param
                    if scan_param in param_names:
                        p_idx = param_names.index(scan_param)
                        val = float(self.belief._particles[idx, p_idx])
                        return self.belief._to_physical(scan_param, val)

        return eig_choice

    def _observe_acquisition(self, obs: Observation) -> None:
        """Handle acquisition observations and manually trigger resample checks."""
        super()._observe_acquisition(obs)
        self._check_and_resample(check_convergence=True)

    def _check_and_resample(self, check_convergence: bool = True) -> None:
        if not hasattr(self.belief, "_weights"):
            return
        ess = _inverse_sum_squares(self.belief._weights)
        ess_threshold = getattr(self.belief, "ess_threshold", 0.0) * getattr(self.belief, "num_particles", 0)
        if ess < ess_threshold:
            if hasattr(self.belief, "_resample"):
                self.belief._resample()

        if check_convergence:
            if self._target_params_converged():
                self._convergence_streak += 1
                if self._convergence_streak >= self._convergence_patience_steps:
                    self._is_converged = True
            else:
                self._convergence_streak = 0
            self._check_convergence_milestones()
