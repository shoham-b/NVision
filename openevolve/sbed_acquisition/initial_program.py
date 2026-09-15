"""OpenEvolve target: Bayesian-SBED's acquisition-point-selection logic.

This is a *standalone copy* of three methods from
``nvision.sim.locs.bayesian.sbed_locator.SequentialBayesianExperimentDesignLocator``
(as of the date this file was created). The evaluator
(``openevolve/sbed_acquisition/evaluator.py``) imports this module and
monkeypatches these three functions onto the real locator class before
running simulations — the production file is never modified by the search,
so nothing here reaches ``nvision/`` until a human reviews and manually
ports back a winning candidate.

Only the EVOLVE-BLOCK region may change. Everything outside it (docstring,
imports) is scaffolding OpenEvolve should leave alone. Function signatures
and names must stay identical — the evaluator binds them by name.

Each function receives ``self`` explicitly (the locator instance) exactly
as the original bound methods did, so the bodies are otherwise unchanged
and free to use ``self.belief``, ``self._acquisition_bounds()``, etc.
Returned frequencies are physical Hz and MUST fall strictly within the
domain bounds used elsewhere in the method (see the existing bounds checks
below) — this codebase fails fast on out-of-domain values instead of
clamping them silently, and the evaluator will score a candidate that
raises on any repeat as a hard failure (score 0), not a partial credit.

See ``openevolve/sbed_acquisition/README.md`` for how to run the search and
interpret results, and ``.claude/skills/locator-evaluation`` for why
``splitting_converged_step`` (not the locator's own stop reason) is the
metric that actually gets optimized.
"""

from __future__ import annotations

import numpy as np

from nvision.sim.locs.bayesian.dip_detection import identify_dip_candidates
from nvision.sim.locs.bayesian.sbed_locator import (
    _DUAL_WINDOW_ENABLED,
    _effective_linewidth_and_contrast_estimate,
    _effective_max_linewidth_hz,
    _max_dip_cluster_span_hz,
)

# EVOLVE-BLOCK-START


def _acquire(self) -> float:
    """Select the next measurement point by maximizing EIG over a frequency grid."""
    lo, hi = self._acquisition_bounds()
    if hi <= lo:
        return float(lo)

    # The belief's original (never-narrowed) domain -- used by the exploration branches
    # below so they can still reach a location resampling has already narrowed away from.
    orig_lo, orig_hi = getattr(self.belief, "_original_physical_x_bounds", (lo, hi))

    # Forced background calibration: sample out-of-span until we have
    # enough background points to estimate noise for the CRLB early-stop.
    if self._forced_bg_mode:
        est = self.belief.estimates()
        f_hat = est.get("frequency", (lo + hi) / 2.0)
        lw_hat, _ = _effective_linewidth_and_contrast_estimate(est, self.belief.physical_param_bounds)
        max_split_hz = _max_dip_cluster_span_hz(self.belief.physical_param_bounds, est)
        from nvision.sim.defaults import NVISION_NOISE_BG_SPAN_FACTOR

        span = max(NVISION_NOISE_BG_SPAN_FACTOR * abs(lw_hat), max_split_hz or 0.0)
        left_lo, left_hi = lo, max(lo, f_hat - span)
        right_lo, right_hi = min(hi, f_hat + span), hi
        left_width = max(0.0, left_hi - left_lo)
        right_width = max(0.0, right_hi - right_lo)
        total_width = left_width + right_width
        if total_width > 0:
            if np.random.rand() < left_width / total_width:
                return float(np.random.uniform(left_lo, left_hi))
            else:
                return float(np.random.uniform(right_lo, right_hi))
        self._forced_bg_mode = False

    # Mix EIG with dip-observation-biased exploration. The exploration/dip
    # branches are drawn first so the (much more expensive) EIG grid search
    # in _eig_acquire() is skipped entirely on steps where it would be
    # discarded anyway.
    decay = np.exp(-self.inference_step_count / 25.0)
    rand_val = np.random.rand()
    if rand_val < 0.1 * decay:
        # Explore globally uniformly to find missing peaks (probability decays over time).
        return float(np.random.uniform(orig_lo, orig_hi))
    elif rand_val < 0.2:
        # Dip-observation biased sampling: find the empirically lowest measured signal values
        # and draw near one of them.
        n_obs = getattr(self.belief, "num_observations", None)
        if n_obs is None:
            n_obs = len(getattr(self.belief, "_observations", []))
        if n_obs >= 5:
            dip_centers = getattr(self.belief, "_dip_centers", None)
            if dip_centers is None:
                rescale_maps = self.belief._rescale_maps
                if "frequency" not in rescale_maps:
                    raise RuntimeError(
                        f"{type(self.belief).__name__} is missing _rescale_maps['frequency']. "
                        "Ensure physical_param_bounds includes 'frequency' at construction."
                    )
                freq_rescale = rescale_maps["frequency"]
                if hasattr(self.belief, "observation_arrays"):
                    obs_xs_unit, obs_ys = self.belief.observation_arrays()
                    obs_xs_phys = freq_rescale.to_phys(obs_xs_unit)
                else:
                    obs_list = self.belief._observations
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
                max_linewidth_hz = _effective_max_linewidth_hz(phys_bounds)
                max_split_hz = _max_dip_cluster_span_hz(phys_bounds, self.belief.estimates())

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

            valid_dip_centers = [c for c in dip_centers if orig_lo <= c <= orig_hi]
            if valid_dip_centers:
                center = float(np.random.choice(valid_dip_centers))
                j_min = max(-5e6, orig_lo - center)
                j_max = min(5e6, orig_hi - center)
                if j_max >= j_min:
                    jitter = float(np.random.uniform(j_min, j_max))
                    val = center + jitter
                    if not (orig_lo <= val <= orig_hi):
                        raise ValueError(f"Jittered dip value {val} is outside domain bounds {(orig_lo, orig_hi)}")
                    return val
                else:
                    if not (orig_lo <= center <= orig_hi):
                        raise ValueError(f"Dip center {center} is outside domain bounds {(orig_lo, orig_hi)}")
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

    if _DUAL_WINDOW_ENABLED:
        dual = self._dual_window_acquire()
        if dual is not None:
            return dual
    return self._eig_acquire()


def _dual_window_acquire(self) -> float | None:
    """Explicit dual symmetric-window acquisition for Zeeman-split signals.

    Breaks a sloppy frequency/zeeman_split trade-off ridge by forcing balanced
    measurement between the two Zeeman flanks. See the full rationale in the
    original method's docstring in ``nvision/sim/locs/bayesian/sbed_locator.py``.
    Falls back to None (caller uses plain EIG) when there is no Zeeman-split
    axis, the split estimate is not yet meaningfully nonzero, or a window
    collapses to nothing.
    """
    param_names = getattr(self.belief, "_param_names", None)
    if not param_names or "zeeman_split" not in param_names or "frequency" not in param_names:
        return None
    est = self.belief.estimates()
    center_hat = est.get("frequency")
    split_hat = est.get("zeeman_split")
    if center_hat is None or split_hat is None:
        return None
    lw_hat, _ = _effective_linewidth_and_contrast_estimate(est, self.belief.physical_param_bounds)
    lw_hat = abs(lw_hat)
    if split_hat < 3.0 * lw_hat:
        return None

    orig_lo, orig_hi = getattr(self.belief, "_original_physical_x_bounds", self._acquisition_bounds())
    half_width = 3.0 * lw_hat
    windows = {
        "left": (
            max(orig_lo, center_hat - split_hat - half_width),
            min(orig_hi, center_hat - split_hat + half_width),
        ),
        "right": (
            max(orig_lo, center_hat + split_hat - half_width),
            min(orig_hi, center_hat + split_hat + half_width),
        ),
    }

    target = "left" if self._dual_window_counts["left"] <= self._dual_window_counts["right"] else "right"
    lo, hi = windows[target]
    if hi <= lo:
        other = "right" if target == "left" else "left"
        lo, hi = windows[other]
        if hi <= lo:
            return None
        target = other

    result = float(np.random.uniform(lo, hi))
    self._dual_window_counts[target] += 1
    return result


def _eig_acquire(self) -> float:
    """Maximize EIG over the belief's slope-targeted candidate grid."""
    candidates = self.belief.get_candidates()
    candidates = self._thin_candidates_by_step(candidates)

    lo, hi = self._acquisition_bounds()
    if (
        self._last_eig_physical_x is not None
        and lo <= self._last_eig_physical_x <= hi
        and not np.any(np.isclose(candidates, self._last_eig_physical_x))
    ):
        candidates = np.append(candidates, self._last_eig_physical_x)

    noise_std_for_eig = (
        self._empirical_batch_noise_std if self._empirical_batch_noise_std is not None else self._noise_std
    )
    best = self.belief.select_max_information_gain(candidates, 1, noise_std=noise_std_for_eig)
    result = float(best[0]) if len(best) > 0 else float(candidates[len(candidates) // 2])
    self._last_eig_physical_x = result
    return result


# EVOLVE-BLOCK-END
