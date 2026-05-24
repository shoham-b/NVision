"""SMC belief with particles on ``[0, 1]`` and physical-scale public summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nvision.belief.abstract_marginal import ParameterValues
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.spectra.unit_cube import UnitCubeSignalModel


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

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeSMCMarginalDistribution requires a UnitCubeSignalModel")

        # Ensure all parameters (including noise) are in unit space [0, 1].
        # We must detect noise parameters here because super().__post_init__
        # expects them to be in parameter_bounds before it iterates over _param_names.
        all_names = list(self.model.parameter_names())
        if self.noise_model is not None:
            all_names.extend(n for n in self.noise_model.spec.names if n not in all_names)

        self.parameter_bounds = {name: (0.0, 1.0) for name in all_names}
        super().__post_init__()

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
        return {k: self._to_physical(k, v) for k, v in raw.items()}

    def _to_physical(self, name: str, u: float) -> float:
        lo, hi = self.physical_param_bounds[name]
        return lo + float(u) * (hi - lo)

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        raw = super()._empirical_uncertainty()
        data = {
            name: u * (self.physical_param_bounds[name][1] - self.physical_param_bounds[name][0])
            for name, u in raw.items()
        }
        return ParameterValues.from_mapping(list(raw.keys()), data)

    def uncertainty(self) -> ParameterValues[float]:
        return self._empirical_uncertainty()

    def converged(self, threshold: float) -> bool:
        # Check convergence uniformly using inner [0, 1] uncertainties
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

        nl = float(max(min(new_lo, new_hi), old_lo))
        nh = float(min(max(new_lo, new_hi), old_hi))
        if nh <= nl:
            return

        sync_x = self.physical_x_bounds == (old_lo, old_hi)
        w_new = nh - nl

        j = self._param_names.index(param_name)
        u_col = self._particles[:, j]
        f = old_lo + u_col * w_old
        u_new = (f - nl) / w_new
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

        dip_frequencies = []
        if hasattr(self, "_observations") and len(self._observations) > 0:
            ys = np.array([o.signal_value for o in self._observations])
            if len(ys) >= 3:
                p30 = float(np.percentile(ys, 30))
                noise_pts = ys[ys >= p30]
                bg_est = float(np.median(noise_pts))
                noise_std_est = float(np.std(noise_pts))
            else:
                bg_est = float(np.max(ys))
                noise_std_est = 0.02

            dip_threshold = bg_est - max(3.0 * noise_std_est, 0.015 * bg_est)
            for o in self._observations:
                if o.signal_value < dip_threshold:
                    dip_frequencies.append(o.x)

        if dip_frequencies:
            # --- Shoulder-based dip span detection --------------------------
            # Instead of a fixed "N × linewidth" buffer, we find the actual
            # physical span of each dip by walking outward from its minimum
            # until the signal recovers to near-baseline (the shoulder).
            # Connected dips (no clear recovery between them) are treated as
            # one span.  The focus window wraps all dip spans + a small guard.

            # Sort all observations by frequency for neighbour-walking.
            obs_sorted = sorted(self._observations, key=lambda o: o.x)
            xs = np.array([o.x for o in obs_sorted])
            ys = np.array([o.signal_value for o in obs_sorted])
            n_obs = len(xs)

            # Shoulder threshold: signal must recover to within 50% of the
            # background level above the dip threshold to count as a shoulder.
            # i.e. the recovery point sits halfway between dip_threshold and bg.
            shoulder_threshold = 0.5 * (dip_threshold + bg_est)

            # Mark which observations are "in dip" territory.
            in_dip = ys < dip_threshold

            if np.any(in_dip):
                dip_indices = np.where(in_dip)[0]

                # Find the left shoulder of the leftmost dip cluster:
                # walk left from the leftmost dip index until signal recovers.
                left_idx = int(dip_indices[0])
                left_shoulder_x = xs[0]  # fallback: leftmost observation
                for k in range(left_idx - 1, -1, -1):
                    if ys[k] >= shoulder_threshold:
                        left_shoulder_x = xs[k]
                        break

                # Find the right shoulder of the rightmost dip cluster:
                # walk right from the rightmost dip index until signal recovers.
                right_idx = int(dip_indices[-1])
                right_shoulder_x = xs[-1]  # fallback: rightmost observation
                for k in range(right_idx + 1, n_obs):
                    if ys[k] >= shoulder_threshold:
                        right_shoulder_x = xs[k]
                        break

                # Add a small guard margin (1 × linewidth) so the locator can
                # still sample just outside the shoulders for EIG computation.
                guard = omega_phys
                new_lo = left_shoulder_x - guard
                new_hi = right_shoulder_x + guard
            else:
                # Dip frequencies list was populated but no consecutive
                # observations are flagged in_dip — fall through to percentile.
                j = self._param_names.index(scan_param)
                u_vals = self._particles[:, j]
                u_lo = float(np.percentile(u_vals, 1.0))
                u_hi = float(np.percentile(u_vals, 99.0))
                new_lo = lo_phys + u_lo * cur_width
                new_hi = lo_phys + u_hi * cur_width
                pad = 3.0 * (new_hi - new_lo)
                new_lo -= pad
                new_hi += pad
        else:
            # --- Fallback: particle-percentile narrowing --------------------
            # No dips seen yet; prune parts of the spectrum with almost no
            # particles using a 1% tail cutoff, then add conservative padding
            # (300% on each side) so we never clip the true peak.
            j = self._param_names.index(scan_param)
            u_vals = self._particles[:, j]
            decay_factor = np.exp(-self._step_count / 25.0)
            frac_rejuv = 0.05 * decay_factor
            num_random = int(len(u_vals) * frac_rejuv)
            u_vals_real = u_vals[:-num_random] if num_random > 0 else u_vals

            u_lo = float(np.percentile(u_vals_real, 1.0))
            u_hi = float(np.percentile(u_vals_real, 99.0))
            new_lo = lo_phys + u_lo * cur_width
            new_hi = lo_phys + u_hi * cur_width

            # Conservative padding so we never clip the true peak while the
            # posterior is still spreading (300% on each side).
            pad = 3.0 * (new_hi - new_lo)
            new_lo -= pad
            new_hi += pad

        new_lo = max(new_lo, lo_phys)
        new_hi = min(new_hi, hi_phys)
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
        )
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy()
        dist._weights = self._weights.copy()
        dist._step_count = self._step_count
        dist.resampled = self.resampled
        if hasattr(self, "_observations"):
            dist._observations = list(self._observations)
        return dist
