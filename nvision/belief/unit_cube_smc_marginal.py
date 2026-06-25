"""SMC belief with particles on ``[0, 1]`` and physical-scale public summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nvision.belief.abstract_marginal import ParameterValues
from nvision.belief.coordinate import RescaleMap
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.spectra.noise_model import NoiseSignalModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


class UnitCubeNoiseSignalModelWrapper(NoiseSignalModel):
    """Wrap a physical NoiseSignalModel to map unit-cube particles to physical space for likelihood evaluation."""

    def __init__(self, inner: NoiseSignalModel, physical_param_bounds: dict[str, tuple[float, float]]):
        self.inner = inner
        self.physical_param_bounds = physical_param_bounds

    @property
    def spec(self):
        return self.inner.spec

    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        phys_arrays = []
        names = self.inner.spec.names
        for name, arr in zip(names, noise_param_arrays, strict=True):
            lo, hi = self.physical_param_bounds[name]
            phys_arr = lo + arr * (hi - lo)
            phys_arrays.append(phys_arr)
        return self.inner.composite_log_likelihood(predicted, residuals, phys_arrays, sigma_epistemic)


@dataclass
class UnitCubeSMCMarginalDistribution(SMCMarginalDistribution):
    """Like :class:`SMCMarginalDistribution` but particles live on ``[0, 1]``.

    The ``model`` must be a :class:`UnitCubeSignalModel` mapping unit coordinates to
    the inner physical model. Internal particles and uncertainties are in normalized
    space so acquisition and ``converged()`` thresholds apply uniformly across
    parameters.

    :meth:`estimates` and :meth:`uncertainty` are returned in **physical** units for
    metrics, plotting, and comparison.
    """

    physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    physical_x_bounds: tuple[float, float] = (0.0, 1.0)

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:  # type: ignore[override]  # noqa: F811
        if hasattr(self, "_physical_param_bounds"):
            return self._physical_param_bounds
        return {}

    @physical_param_bounds.setter
    def physical_param_bounds(self, value: dict[str, tuple[float, float]]) -> None:
        self._physical_param_bounds = value

    @property
    def _rescale_maps(self) -> dict[str, RescaleMap]:
        """Return a ``RescaleMap`` per parameter using the *original* physical bounds.

        For the frequency axis this uses ``_original_physical_x_bounds`` — the
        full domain set at construction and never narrowed — so that
        ``freq_rescale.to_phys(o.x)`` always converts the stored [0, 1]
        observation coordinate correctly, even after the focus window has
        shrunk ``physical_param_bounds["frequency"]``.

        For all other parameters the current ``physical_param_bounds`` is
        used (those bounds are never narrowed).
        """
        if not hasattr(self, "_original_physical_x_bounds"):
            raise RuntimeError(
                "UnitCubeSMCMarginalDistribution._rescale_maps called before "
                "__post_init__ set _original_physical_x_bounds."
            )
        phys = self.physical_param_bounds
        maps: dict[str, RescaleMap] = {}
        for name in self._param_names:
            if name == "frequency":
                lo, hi = self._original_physical_x_bounds
            elif name in phys:
                lo, hi = phys[name]
            else:
                raise RuntimeError(
                    f"UnitCubeSMCMarginalDistribution is missing physical_param_bounds "
                    f"for parameter '{name}'. All parameters must have physical bounds at construction."
                )
            maps[name] = RescaleMap(lo=float(lo), hi=float(hi))
        return maps

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeSMCMarginalDistribution requires a UnitCubeSignalModel")

        if self.noise_model is not None and not isinstance(self.noise_model, UnitCubeNoiseSignalModelWrapper):
            self.noise_model = UnitCubeNoiseSignalModelWrapper(self.noise_model, self.physical_param_bounds)

        # Ensure all parameters (including noise) are in unit space [0, 1].
        # We must detect noise parameters here because super().__post_init__
        # expects them to be in parameter_bounds before it iterates over _param_names.
        all_names = list(self.model.parameter_names())
        if self.noise_model is not None:
            all_names.extend(n for n in self.noise_model.spec.names if n not in all_names)

        self.parameter_bounds = {name: (0.0, 1.0) for name in all_names}
        super().__post_init__()
        self._original_physical_x_bounds = self.physical_x_bounds

    def expected_information_gain(self, candidates: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Override to normalize physical candidates to [0, 1] for the UnitCube model."""
        lo, hi = self.physical_x_bounds
        unit_candidates = (candidates - lo) / (hi - lo)
        return super().expected_information_gain(unit_candidates, noise_std=noise_std)

    def get_candidates(self) -> np.ndarray:
        """Return candidates in **physical** frequency space.

        The base class stores candidates in unit [0, 1] space (matching internal
        particles). The locator and ``expected_information_gain`` both operate in
        physical space, so we convert here before returning.
        """
        lo, hi = self.physical_x_bounds
        return lo + self._current_candidates.astype(np.float64) * (hi - lo)

    def _generate_epoch_candidates(self) -> None:
        """Generate candidates in unit space.

        The base class now uses internal _estimates_unit() which correctly
        returns [0, 1] values even for this subclass.
        """
        super()._generate_epoch_candidates()

    def estimates(self) -> dict[str, float]:
        raw = super().estimates()
        return {
            k: (
                self._to_physical(k, v)
                if (k != "noise_sigma" or not getattr(self, "_use_rao_blackwell_noise", False))
                else v
            )
            for k, v in raw.items()
        }

    def _to_physical(self, name: str, u: float) -> float:
        lo, hi = self.physical_param_bounds[name]
        return lo + float(u) * (hi - lo)

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        raw = super()._empirical_uncertainty()
        data = {}
        for name, u in raw.items():
            if name == "noise_sigma" and getattr(self, "_use_rao_blackwell_noise", False):
                data[name] = u
            elif name in self.physical_param_bounds:
                data[name] = u * (self.physical_param_bounds[name][1] - self.physical_param_bounds[name][0])
            else:
                data[name] = u
        return ParameterValues.from_mapping(list(raw.keys()), data)

    def uncertainty(self) -> ParameterValues[float]:
        return self._empirical_uncertainty()

    def crlb_frequency(self) -> float:
        """Analytical CRLB for frequency (physical Hz) for the NV Lorentzian model.

        Uses the closed-form Fisher result for uniform sampling of a Lorentzian:
        ``CRLB_f = sqrt(2σ²Ω / (πa²ρ))``
        where σ = noise std, Ω = linewidth (FWHM, Hz), a = contrast (c_total),
        ρ = n_obs / bandwidth (measurements per Hz).

        Derivation: integrating (∂S/∂f)²/σ² over a uniform density ρ gives
        I(f) = πa²ρ / (2σ²Ω), so CRLB_var = 2σ²Ω/(πa²ρ).

        Returns ``math.inf`` for non-Lorentzian models or before any observations.
        """
        from nvision.spectra.nv_center import NVCenterLorentzianModel

        inner = getattr(self.model, "inner", None)
        if not isinstance(inner, NVCenterLorentzianModel):
            return math.inf

        if self._obs_count == 0 or self.last_obs is None:
            return math.inf

        ests = self.estimates()  # physical-space values
        linewidth = ests.get("linewidth")
        c_total = ests.get("c_total")
        if linewidth is None or c_total is None or c_total <= 0 or linewidth <= 0:
            return math.inf

        noise_std = float(self.last_obs.noise_std)
        if noise_std <= 0:
            return math.inf

        lo, hi = self._original_physical_x_bounds
        bandwidth = hi - lo
        if bandwidth <= 0:
            return math.inf

        rho = self._obs_count / bandwidth  # measurements per Hz
        variance = (2.0 * noise_std**2 * linewidth) / (math.pi * c_total**2 * rho)
        return math.sqrt(max(variance, 0.0))

    def reported_uncertainty(self) -> ParameterValues[float]:
        """Physical uncertainty floored at K× the Lorentzian CRLB for frequency.

        K is ``NVISION_FREQ_CRLB_SAFETY_FACTOR`` — the same factor the locator
        uses to raise the frequency convergence threshold, so at convergence the
        reported σ equals the convergence ceiling. Only the frequency entry is
        floored; all other parameters pass through unchanged. Control-flow paths
        must use :meth:`uncertainty` instead.
        """
        from nvision.sim.defaults import NVISION_FREQ_CRLB_SAFETY_FACTOR

        data = dict(self.uncertainty().items())
        crlb = self.crlb_frequency()
        if math.isfinite(crlb) and "frequency" in data:
            data["frequency"] = max(data["frequency"], NVISION_FREQ_CRLB_SAFETY_FACTOR * crlb)
        return ParameterValues.from_mapping(list(data.keys()), data)

    def converged(self, threshold: float) -> bool:
        # Check convergence in unit [0, 1] space. Not used by the SBED/Sobol
        # locators (they gate on _target_params_converged), but kept consistent
        # for any belief-level convergence query.
        raw_uncertainties = super()._empirical_uncertainty()
        return all(u < threshold for u in raw_uncertainties.values())

    def covariance_matrix(self) -> np.ndarray:
        """Return the physical-scale covariance matrix."""
        raw_cov = super().covariance_matrix()
        ranges = np.array(
            [self.physical_param_bounds[name][1] - self.physical_param_bounds[name][0] for name in self._param_names]
        )
        return raw_cov * np.outer(ranges, ranges)

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        return super().sample(n)

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Shrink physical bounds for ``param_name`` and remap unit particles (see grid variant)."""
        if param_name not in self.physical_param_bounds:
            raise KeyError(param_name)
        old_lo, old_hi = self.physical_param_bounds[param_name]
        w_old = old_hi - old_lo
        if w_old <= 0:
            return

        lo_orig, hi_orig = self._original_physical_x_bounds
        nl = float(max(min(new_lo, new_hi), lo_orig))
        nh = float(min(max(new_lo, new_hi), hi_orig))
        if nh <= nl:
            return

        sync_x = self.physical_x_bounds == (old_lo, old_hi)
        w_new = nh - nl

        j = self._param_names.index(param_name)
        u_col = self._particles[:, j]
        f = old_lo + u_col * w_old
        u_new = (f - nl) / w_new

        out_of_bounds = (u_new < 0.0) | (u_new > 1.0)
        n_out = int(np.sum(out_of_bounds))
        if n_out > 0:
            in_bounds_indices = np.where(~out_of_bounds)[0]
            if len(in_bounds_indices) > 0:
                replacement_indices = np.random.choice(in_bounds_indices, size=n_out, replace=True)
                # Copy entire particle rows for out-of-bounds particles to preserve joint parameter dependencies
                self._particles[out_of_bounds] = self._particles[replacement_indices]
                # Re-evaluate u_col for the narrowed scan parameter
                u_col = self._particles[:, j]
                f = old_lo + u_col * w_old
                u_new = (f - nl) / w_new
            else:
                import logging

                logging.getLogger(__name__).warning(
                    "All particles are out of bounds during narrow_scan_parameter_physical_bounds (%s, %s). "
                    "Rejecting narrowing proposal to protect filter stability.",
                    nl,
                    nh,
                )
                return

        # Verify that all particles are now within [0, 1] with a tiny tolerance for numerical precision
        tol = 1e-7
        if np.any((u_new < -tol) | (u_new > 1.0 + tol)):
            raise ValueError(
                f"Particles out of bounds after resampling in narrow_scan_parameter_physical_bounds: min {np.min(u_new)}, max {np.max(u_new)}"
            )
        self._particles[:, j] = np.clip(u_new, 0.0, 1.0)

        self.model.narrow_physical_interval_for_param(param_name, nl, nh, update_x_axis=sync_x)
        self.physical_param_bounds[param_name] = (nl, nh)
        if sync_x:
            self.physical_x_bounds = (nl, nh)

    def _resample(self) -> None:
        """Perform systematic resampling and automatically narrow the frequency bounds.

        Only the scan axis (``frequency``) is narrowed. All other parameters keep
        their full physical range so the posterior can freely re-explore them once
        the true frequency is located — their optimal values may differ from the
        initial posterior mode.
        """
        super()._resample()

        # Identify the scan parameter (almost always "frequency").
        scan_param = "frequency" if "frequency" in self.physical_param_bounds else None
        if scan_param is None:
            for name, bounds in self.physical_param_bounds.items():
                if bounds == self.physical_x_bounds:
                    scan_param = name
                    break
        if scan_param is None:
            return  # Nothing to narrow — no frequency axis found.

        lo_phys, hi_phys = self.physical_param_bounds[scan_param]
        cur_width = hi_phys - lo_phys
        if cur_width <= 0:
            return

        # Setup dip detection parameters
        estimates = self.estimates()
        omega_phys = float(estimates.get("linewidth", estimates.get("fwhm_total", 2.0e6) / 2.0))
        omega_phys = max(omega_phys, 1.0e5)  # at least 100 kHz

        # --- Active Boundary-Escape Guard ----------------------------------
        # If particles are piling up near the unit boundaries [0.0, 1.0], it
        # means the true frequency is outside the currently narrowed window.
        # We proactively expand the physical bounds in that direction.
        j = self._param_names.index(scan_param)
        u_vals = self._particles[:, j]

        left_piling = float(np.mean(u_vals < 0.05)) > 0.15
        right_piling = float(np.mean(u_vals > 0.95)) > 0.15

        lo_orig, hi_orig = self._original_physical_x_bounds
        import logging
        import os

        if left_piling and lo_phys > lo_orig:
            expansion = max(cur_width, 10.0 * omega_phys)
            new_lo = max(lo_phys - expansion, lo_orig)
            new_hi = hi_phys
            logging.getLogger(__name__).debug(
                "SMC particles piling up at left boundary (%.2f%%). Expanding physical scan window to the left: [%.6f, %.6f] GHz",
                float(np.mean(u_vals < 0.05)) * 100,
                new_lo / 1e9,
                new_hi / 1e9,
            )
            self._last_expansion_step = self._step_count
            self.narrow_scan_parameter_physical_bounds(scan_param, new_lo, new_hi)
            self._cached_cov = None
            self._cov_step = -1
            self._generate_epoch_candidates()
            return

        if right_piling and hi_phys < hi_orig:
            expansion = max(cur_width, 10.0 * omega_phys)
            new_lo = lo_phys
            new_hi = min(hi_phys + expansion, hi_orig)
            logging.getLogger(__name__).debug(
                "SMC particles piling up at right boundary (%.2f%%). Expanding physical scan window to the right: [%.6f, %.6f] GHz",
                float(np.mean(u_vals > 0.95)) * 100,
                new_lo / 1e9,
                new_hi / 1e9,
            )
            self._last_expansion_step = self._step_count
            self.narrow_scan_parameter_physical_bounds(scan_param, new_lo, new_hi)
            self._cached_cov = None
            self._cov_step = -1
            self._generate_epoch_candidates()
            return

        # --- Observation-Based Shoulder Narrowing --------------------------
        # If we have enough dense observations, detect the dip region from the
        # raw signal data and narrow the window around it. This runs before the
        # particle step-count guard so early observations can focus the window
        # even before the SMC posterior has converged.
        if self._obs_count >= 5:
            obs_xs_unit = self._obs_x_arr[: self._obs_count].copy()
            obs_ys = self._obs_y_arr[: self._obs_count].copy()
            # Convert unit-space x (relative to current physical window) to physical Hz.
            obs_xs_phys = lo_phys + obs_xs_unit * (hi_phys - lo_phys)

            background = float(np.percentile(obs_ys, 80))
            min_signal = float(np.min(obs_ys))
            depth = background - min_signal

            if depth >= 0.05:
                # Threshold at 50% of dip depth below background.
                threshold = background - 0.5 * depth
                sort_idx = np.argsort(obs_xs_phys)
                xs_sorted = obs_xs_phys[sort_idx]
                ys_sorted = obs_ys[sort_idx]
                below = ys_sorted < threshold
                if np.any(below):
                    below_xs = xs_sorted[below]
                    raw_lo = float(below_xs[0])
                    raw_hi = float(below_xs[-1])
                    span = raw_hi - raw_lo
                    total_width = hi_orig - lo_orig
                    # Padding: wider of 50% of span or 5% of original window.
                    padding = max(0.5 * span, 0.05 * total_width)
                    obs_new_lo = max(raw_lo - padding, lo_orig)
                    obs_new_hi = min(raw_hi + padding, hi_orig)
                    obs_new_width = obs_new_hi - obs_new_lo
                    # Only apply if window actually shrinks by at least 1%.
                    if obs_new_lo < obs_new_hi and obs_new_width < cur_width * 0.99:
                        self.narrow_scan_parameter_physical_bounds(scan_param, obs_new_lo, obs_new_hi)
                        self._cached_cov = None
                        self._cov_step = -1
                        self._generate_epoch_candidates()
                        return

        # --- Narrowing Delay Safeguard -------------------------------------
        # Delay narrowing until we have completed a minimum number of global
        # measurements (default 8 steps) to resolve multi-modal hyperfine peak
        # ambiguity and ensure we never focus on the wrong place.
        min_narrowing_steps = int(os.getenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8"))
        if self._step_count < min_narrowing_steps:
            return

        # Also delay narrowing if we recently expanded the bounds to allow exploration
        last_exp = getattr(self, "_last_expansion_step", -1)
        if last_exp >= 0 and (self._step_count - last_exp) < min_narrowing_steps:
            return

        # --- Unified Active-Range Union Focusing (Zero Fallbacks) -----------
        # Compute believed active frequency range for each particle i:
        # [left_i, right_i] = [f_i - split_i - k * omega_i, f_i + split_i + k * omega_i]
        j_freq = self._param_names.index(scan_param)
        u_freq = self._particles[:, j_freq]
        freq_phys = lo_phys + u_freq * cur_width

        if "split" in self._param_names:
            j_split = self._param_names.index("split")
            u_split = self._particles[:, j_split]
            lo_s, hi_s = self.physical_param_bounds["split"]
            split_phys = lo_s + u_split * (hi_s - lo_s)
        else:
            split_phys = np.zeros_like(freq_phys)

        if "linewidth" in self._param_names:
            j_line = self._param_names.index("linewidth")
            u_line = self._particles[:, j_line]
            lo_l, hi_l = self.physical_param_bounds["linewidth"]
            linewidth_phys = lo_l + u_line * (hi_l - lo_l)
        elif "fwhm_total" in self._param_names:
            j_fwhm = self._param_names.index("fwhm_total")
            u_fwhm = self._particles[:, j_fwhm]
            lo_l, hi_l = self.physical_param_bounds["fwhm_total"]
            linewidth_phys = (lo_l + u_fwhm * (hi_l - lo_l)) / 2.0
        else:
            linewidth_phys = np.full_like(freq_phys, 2.0e6)

        cover_factor = float(os.getenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0"))
        tail_percentile = float(os.getenv("NVISION_SMC_FOCUSING_TAIL_PERCENTILE", "1.0"))

        left_phys = freq_phys - split_phys - cover_factor * linewidth_phys
        right_phys = freq_phys + split_phys + cover_factor * linewidth_phys

        new_lo = float(np.percentile(left_phys, tail_percentile))
        new_hi = float(np.percentile(right_phys, 100.0 - tail_percentile))

        lo_orig, hi_orig = self._original_physical_x_bounds
        new_lo = max(new_lo, lo_orig)
        new_hi = min(new_hi, hi_orig)
        if new_hi <= new_lo:
            return

        # Only apply if the window actually shrinks by at least 5%.
        min_narrowing_fraction = 0.05
        if (cur_width - (new_hi - new_lo)) / cur_width < min_narrowing_fraction:
            return

        self.narrow_scan_parameter_physical_bounds(scan_param, new_lo, new_hi)

        # Clear unit covariance cache so it's recomputed for the new unit particles.
        self._cached_cov = None
        self._cov_step = -1
        # Regenerate the epoch candidates using the updated physical frequency bounds.
        self._generate_epoch_candidates()

    def update(self, obs: Observation) -> None:
        lo_orig, hi_orig = self._original_physical_x_bounds
        lo_curr, hi_curr = self.physical_x_bounds

        if (lo_orig != lo_curr) or (hi_orig != hi_curr):
            x_phys = lo_orig + obs.x * (hi_orig - lo_orig)
            x_curr_unit = (x_phys - lo_curr) / (hi_curr - lo_curr)
            from dataclasses import replace

            obs_eval = replace(obs, x=float(x_curr_unit))
        else:
            obs_eval = obs

        super().update(obs_eval)
        # Restore the original-frame x in the history buffer and last_obs
        # (the rescaled obs_eval.x was recorded by super().update()).
        self._obs_x_arr[self._obs_count - 1] = obs.x
        self.last_obs = obs

    def batch_update(self, observations: list[Observation]) -> None:
        # NOTE: super().batch_update() appends to self._observations; do not
        # extend here as well or every observation is stored twice.
        lo_orig, hi_orig = self._original_physical_x_bounds
        lo_curr, hi_curr = self.physical_x_bounds

        observations_eval = []
        if (lo_orig != lo_curr) or (hi_orig != hi_curr):
            from dataclasses import replace

            for obs in observations:
                x_phys = lo_orig + obs.x * (hi_orig - lo_orig)
                x_curr_unit = (x_phys - lo_curr) / (hi_curr - lo_curr)
                observations_eval.append(replace(obs, x=float(x_curr_unit)))
        else:
            observations_eval = observations

        super().batch_update(observations_eval)
        # Restore the original-frame x values in the history buffer and last_obs
        # (the rescaled observations_eval x values were recorded by super()).
        n_obs = len(observations)
        if n_obs:
            self._obs_x_arr[self._obs_count - n_obs : self._obs_count] = [o.x for o in observations]
            self.last_obs = observations[-1]

    def copy(self) -> UnitCubeSMCMarginalDistribution:
        dist = UnitCubeSMCMarginalDistribution(
            model=self.model,
            parameter_bounds=self.parameter_bounds.copy(),
            num_particles=self.num_particles,
            ess_threshold=self.ess_threshold,
            a_param=self.a_param,
            last_obs=self.last_obs,
            noise_model=self.noise_model,
            auto_resample=self.auto_resample,
            resample_delay=self.resample_delay,
            physical_param_bounds=dict(self.physical_param_bounds),
            physical_x_bounds=self.physical_x_bounds,
            priors=self.priors,
            min_exploration_frac=self.min_exploration_frac,
            tempering_factor=self.tempering_factor,
            noise_discount_factor=self.noise_discount_factor,
            noise_prior_strength=self.noise_prior_strength,
            skip_state_init=True,
        )
        # Grids depend only on bounds, which are identical — share by reference
        # (consumers rebind on narrowing/resample, never mutate in place).
        dist._global_grid = self._global_grid
        dist._current_candidates = self._current_candidates
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy(order="K")  # preserve F-order layout
        dist._weights = self._weights.copy()
        dist._step_count = self._step_count
        dist.resampled = self.resampled
        dist._original_physical_x_bounds = self._original_physical_x_bounds
        dist._obs_x_arr = self._obs_x_arr.copy()
        dist._obs_y_arr = self._obs_y_arr.copy()
        dist._obs_count = self._obs_count
        dist._use_rao_blackwell_noise = getattr(self, "_use_rao_blackwell_noise", False)
        if getattr(self, "_use_rao_blackwell_noise", False):
            dist._noise_alphas = self._noise_alphas.copy()
            dist._noise_betas = self._noise_betas.copy()
        if hasattr(self, "_dip_centers"):
            dist._dip_centers = list(self._dip_centers)
        return dist
