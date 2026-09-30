"""SMC belief that *infers* the zero-field centre frequency and narrows its probe window onto it.

:class:`~nvision.belief.smc_marginal.SMCMarginalDistribution` treats ``frequency`` as a known
instrument constant, so its probe window never changes. This subclass is the free-frequency
special case (``with_fixed_frequency=False``): after each resample it

* **expands** the physical probe window when particles pile up on a boundary (the truth lies
  outside the current window), and
* **narrows** it onto the region the posterior still allows, once enough steps have passed to
  resolve multi-modal ambiguity.

Particles and observations are kept on ``[0, 1]`` of the *current* window, so narrowing remaps the
frequency column and observations are re-expressed in the current window before each update.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace

import numpy as np

from nvision.belief.focus_window import FocusWindow, clamp_to_domain
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.spectra.nv_center import effective_hwhm

_LOG = logging.getLogger(__name__)

# Narrowing waits this many steps so multi-modal hyperfine ambiguity resolves before the window
# focuses on a (possibly wrong) place.
NVISION_MIN_STEPS_BEFORE_NARROWING: int = int(os.getenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8"))
# Each particle's active range is [f - split - k*Omega, f + split + k*Omega]; k is this cover factor.
NVISION_SMC_FOCUSING_COVER_FACTOR: float = float(os.getenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0"))
# The 5th/95th percentiles skip stray low-weight tail particles (e.g. from the min_exploration_frac
# floor) while barely eating into the true dense clusters (which span ~100s of kHz, so losing 5% of
# their mass barely moves the boundary).
_FOCUSING_TAIL_PERCENTILE: float = 5.0
# A narrowing is only applied when it shrinks the window by at least this fraction.
_MIN_NARROWING_FRACTION: float = 0.05
# Particles piling up within this unit distance of a window edge signal that the truth lies outside...
_EDGE_BAND_UNIT: float = 0.05
# ...once more than this fraction of them sit in the band.
_EDGE_PILING_FRACTION: float = 0.15


@dataclass
class FreeFrequencySMCMarginalDistribution(SMCMarginalDistribution):
    """SMC belief with ``frequency`` as a particle dimension and an adaptive probe window."""

    # Step of the most recent boundary-escape window expansion (-1: never).
    _last_expansion_step: int = field(init=False, default=-1, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if "frequency" not in self._param_names:
            raise ValueError("FreeFrequencySMCMarginalDistribution requires a model with a free 'frequency'.")

    def batch_update(self, observations: list[Observation]) -> None:
        """Re-express observations (unit coordinates of the *original* window) in the current window."""
        lo_orig, hi_orig = self._original_physical_x_bounds
        lo_curr, hi_curr = self.physical_x_bounds
        if (lo_orig, hi_orig) == (lo_curr, hi_curr):
            super().batch_update(observations)
            return

        observations_eval = [
            replace(obs, x=float((lo_orig + obs.x * (hi_orig - lo_orig) - lo_curr) / (hi_curr - lo_curr)))
            for obs in observations
        ]
        super().batch_update(observations_eval)

        # The history buffer holds the rescaled x values; restore the original-frame ones. A resample
        # inside super() may already have sorted them using the stale (narrowed-frame) values.
        n_obs = len(observations)
        start = self._obs_count - n_obs
        self._obs_x_arr[start : self._obs_count] = [o.x for o in observations]
        for idx in range(start, self._obs_count):
            self._resync_sort_position(idx)
        self.last_obs = observations[-1]

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Move the physical bounds of ``param_name`` to ``[new_lo, new_hi]`` and remap its unit particles."""
        if param_name not in self.physical_param_bounds:
            raise KeyError(param_name)
        old_lo, old_hi = self.physical_param_bounds[param_name]
        w_old = old_hi - old_lo
        if w_old <= 0:
            return

        lo_orig, hi_orig = self._original_physical_x_bounds
        nl, nh = clamp_to_domain(new_lo, new_hi, lo_orig, hi_orig)
        if nh <= nl:
            return

        sync_x = self.physical_x_bounds == (old_lo, old_hi)
        w_new = nh - nl

        j = self._param_names.index(param_name)
        f = old_lo + self._particles[:, j] * w_old
        u_new = (f - nl) / w_new

        out_of_bounds = (u_new < 0.0) | (u_new > 1.0)
        n_out = int(np.sum(out_of_bounds))
        if n_out > 0:
            in_bounds_indices = np.where(~out_of_bounds)[0]
            if len(in_bounds_indices) == 0:
                _LOG.warning(
                    "All particles are out of bounds during narrow_scan_parameter_physical_bounds (%s, %s). "
                    "Rejecting narrowing proposal to protect filter stability.",
                    nl,
                    nh,
                )
                return
            # Replace out-of-window particles by whole copies of in-window ones, preserving joint dependencies.
            replacement_indices = np.random.choice(in_bounds_indices, size=n_out, replace=True)
            self._particles[out_of_bounds] = self._particles[replacement_indices]
            f = old_lo + self._particles[:, j] * w_old
            u_new = (f - nl) / w_new

        tol = 1e-7
        if np.any((u_new < -tol) | (u_new > 1.0 + tol)):
            raise ValueError(
                "Particles out of bounds after resampling in "
                f"narrow_scan_parameter_physical_bounds: min {np.min(u_new)}, max {np.max(u_new)}"
            )
        self._particles[:, j] = np.clip(u_new, 0.0, 1.0)
        self._belief_version += 1

        self.model.narrow_physical_interval_for_param(param_name, nl, nh, update_x_axis=sync_x)
        self.physical_param_bounds[param_name] = (nl, nh)
        if sync_x:
            self.physical_x_bounds = (nl, nh)

    def _apply_window(self, new_lo: float, new_hi: float) -> None:
        self.narrow_scan_parameter_physical_bounds("frequency", new_lo, new_hi)
        self._cached_cov = None
        self._cov_step = -1
        self._generate_epoch_candidates()

    def _resample(self) -> None:
        """Resample, then expand the window on a boundary pile-up or narrow it onto the posterior.

        Only the scan axis (``frequency``) is adapted. All other parameters keep their full
        physical range so the posterior can freely re-explore them once the true frequency is
        located.
        """
        super()._resample()

        lo_phys, hi_phys = self.physical_param_bounds["frequency"]
        cur_width = hi_phys - lo_phys
        if cur_width <= 0:
            return

        estimates = self.estimates()
        omega_phys = max(effective_hwhm(estimates), 1.0e5)  # at least 100 kHz
        zeeman_hat_phys = float(estimates.get("zeeman_split", 0.0))

        # --- Boundary-escape guard: particles piling up on a unit edge mean the true frequency is
        # outside the current window, so expand it in that direction.
        j = self._param_names.index("frequency")
        u_vals = self._particles[:, j]
        left_piling = float(np.mean(u_vals < _EDGE_BAND_UNIT)) > _EDGE_PILING_FRACTION
        right_piling = float(np.mean(u_vals > 1.0 - _EDGE_BAND_UNIT)) > _EDGE_PILING_FRACTION
        lo_orig, hi_orig = self._original_physical_x_bounds
        expansion = max(cur_width, 10.0 * omega_phys + 2.0 * zeeman_hat_phys)

        if left_piling and lo_phys > lo_orig:
            self._last_expansion_step = self._step_count
            self._apply_window(max(lo_phys - expansion, lo_orig), hi_phys)
            return
        if right_piling and hi_phys < hi_orig:
            self._last_expansion_step = self._step_count
            self._apply_window(lo_phys, min(hi_phys + expansion, hi_orig))
            return

        # --- Narrowing delay: resolve multi-modal hyperfine ambiguity first, and give a recent
        # expansion time to be explored.
        if self._step_count < NVISION_MIN_STEPS_BEFORE_NARROWING:
            return
        last_exp = self._last_expansion_step
        if last_exp >= 0 and (self._step_count - last_exp) < NVISION_MIN_STEPS_BEFORE_NARROWING:
            return

        # --- Active-range union. Each particle believes its dips span
        # [f_i - offset_i, f_i + offset_i] with offset_i = zeeman_i + split_i + k * hwhm_i. The offset
        # is exactly symmetric about f_i, so it is pooled across *all* particles into one upper
        # quantile rather than estimating two one-sided quantiles from disjoint tails. The centre
        # term keeps its own percentiles (not the mean) so a wide or not-yet-unimodal frequency
        # posterior is still respected.
        phys = self.particles_phys()
        offset_phys = (
            phys.get("zeeman_split", 0.0)
            + phys.get("split", 0.0)
            + NVISION_SMC_FOCUSING_COVER_FACTOR * effective_hwhm(phys)
        )
        offset_q95 = float(np.percentile(offset_phys, 100.0 - _FOCUSING_TAIL_PERCENTILE))
        freq_lo = float(np.percentile(phys["frequency"], _FOCUSING_TAIL_PERCENTILE))
        freq_hi = float(np.percentile(phys["frequency"], 100.0 - _FOCUSING_TAIL_PERCENTILE))

        # Floor: a single narrowing step must not undershoot the plausible dip span, even if the
        # frequency marginal has already collapsed.
        current_window = FocusWindow(lo=lo_phys, hi=hi_phys, full_lo=lo_orig, full_hi=hi_orig)
        proposed = current_window.propose_narrowing(
            freq_lo - offset_q95,
            freq_hi + offset_q95,
            min_width=2.0 * offset_q95,
            min_narrowing_fraction=_MIN_NARROWING_FRACTION,
        )
        if proposed is None:
            return
        self._apply_window(proposed.lo, proposed.hi)
