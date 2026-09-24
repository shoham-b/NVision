"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math
import os

import numpy as np
from numba import njit

from nvision.belief.dip_detection import effective_max_linewidth_hz
from nvision.models.observation import Observation
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    NVISION_FREQ_CRLB_SAFETY_FACTOR,
    NVISION_SMC_CANDIDATE_STEP_HZ,
)
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

# Minimum number of consecutive converged checks before declaring convergence.
# Prevents false early stops on the first measurement, especially with no noise.
NVISION_CONVERGENCE_PATIENCE: int = int(os.getenv("NVISION_CONVERGENCE_PATIENCE", "8"))

# Adaptive plateau stop: give up when the *estimate itself* stops moving, rather than
# when some derived quantity claims the information limit has been reached.
#
# Motivated by the measured error-vs-steps curve on the NV voigt grid (120 repeats,
# median |error| on `zeeman_split`): 1.483 MHz at step 10 -> 0.237 at 25 -> 0.059 at 50
# -> 0.022 at 100 -> 0.016 at 450. The knee is around step 100; the last 350 steps of a
# 450-step budget buy ~1.3x while costing 4.5x the measurements. So there IS a real
# early-stop win here -- roughly 3-4x fewer measurements for a few percent of accuracy --
# but a FIM-CRLB-driven stop (evaluated and rejected) cannot capture it: it fired at median step 24, i.e. just
# *before* the steepest part of the curve, where the estimate is still 14x worse than it
# will be. Tracking movement of the estimate measures diminishing returns directly, so it
# fires where the curve actually flattens and adapts per-run instead of encoding a budget.
#
# Movement is normalized by each parameter's own current uncertainty, so the test reads
# "the estimate has drifted less than a fraction of its own error bar over the last
# WINDOW steps" -- scale-free across parameters that differ by orders of magnitude.
#
# SIGMA_FRAC was calibrated by replaying the criterion inside full-budget runs (so the
# stop step and the budget estimate come from the SAME run, no cross-arm confound) over
# 120 repeats, then confirmed with a live A/B of the shipped rule (120 configs/arm):
#
#            ordinary configs                  degenerate configs
#   frac  fires  med step  saving          fires  med step  saving
#   0.10   41%      367     1.2x            79%      284     1.6x   <- unreliable
#   0.15   83%      330     1.4x            98%      201     2.2x
#   0.25  100%      197     2.3x           100%      124     3.6x   <- default
#   0.40  100%      132     3.4x           100%       86     5.2x
#
# Live A/B at 0.25 (median steps 450 -> 220 ordinary, 450 -> 124 degenerate):
# ordinary `zeeman_split` error 0.0211 -> 0.0233 MHz, `homogeneous_linewidth` 0.165 ->
# 0.154 (better), `sigma_inhom` 0.136 -> 0.164, `c_total` 0.0025 -> 0.0032. Degenerate:
# `zeeman_split` 0.406 -> 0.604, widths slightly better. Both beat the sqrt(n) rule of
# thumb -- a 2-3.6x cut in measurements costs well under the sqrt(2)-sqrt(3.6) error
# increase it would naively imply.
#
# Cost to be aware of when tuning: on the degenerate configs the catastrophic rate
# (`zeeman_split` error > 1 MHz) went 7.1% -> 12.5% (4 -> 7 of 56, small counts). Drop
# SIGMA_FRAC to 0.15 to keep most of that margin at a 2.2x rather than 3.6x saving.
_PLATEAU_WINDOW: int = int(os.getenv("NVISION_SBED_PLATEAU_WINDOW", "30"))
_PLATEAU_SIGMA_FRAC: float = float(os.getenv("NVISION_SBED_PLATEAU_SIGMA_FRAC", "0.25"))


def _effective_linewidth_and_contrast_estimate(est: dict, phys_bounds: dict) -> tuple[float, float | None]:
    """Return (effective HWHM Hz estimate, realized contrast estimate), lineshape-agnostic.

    Falls back to the bound's max effective linewidth when the current
    estimate is unavailable, mirroring the pre-existing ``lw_bounds[1]``
    fallback. Contrast is ``None`` when the model has no population-style
    amplitude estimate.
    """
    if "linewidth" in est:
        lw_bounds = phys_bounds.get("linewidth", (0.0, 1e6))
        return est.get("linewidth", lw_bounds[1]), est.get("c_total")
    if "saturation" in est and "sigma_inhom" in est:
        from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar

        saturation = est.get("saturation")
        sigma_inhom = est.get("sigma_inhom")
        if saturation is None or sigma_inhom is None:
            return effective_max_linewidth_hz(phys_bounds), None
        fwhm_total, _, c_total = _saturation_voigt_reparam_scalar(saturation, sigma_inhom, NV_SATURATION_C_MAX)
        return fwhm_total / 2.0, c_total
    if "homogeneous_linewidth" in est:
        hl_bounds = phys_bounds.get("homogeneous_linewidth", (0.0, 2e6))
        return est.get("homogeneous_linewidth", hl_bounds[1]), est.get("c_total")
    return effective_max_linewidth_hz(phys_bounds), None


@njit(cache=True)
def _thin_by_step_indices(candidates: np.ndarray, step: float) -> np.ndarray:
    """Indices of a greedy minimum-spacing subset of sorted *candidates*.

    Keeps the first and last candidates so the full range stays represented.
    """
    n = candidates.shape[0]
    kept = np.empty(n, dtype=np.int64)
    kept[0] = 0
    m = 1
    last = candidates[0]
    for i in range(1, n - 1):
        if candidates[i] - last >= step:
            kept[m] = i
            m += 1
            last = candidates[i]
    kept[m] = n - 1
    return kept[: m + 1]


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
        self.belief.auto_resample = False
        self._is_converged = False
        # Own streak, separate from `_convergence_streak` (which gates `_target_params_converged`):
        # `_check_crlb_early_stop`'s `all_crlb_done` is an independent single-snapshot signal and
        # must not be allowed to set `_is_converged` on one lucky reading.
        self._crlb_convergence_streak: int = 0
        # Rolling history of parameter estimates, for the plateau stop (see
        # _check_estimate_plateau). One dict per convergence check, oldest first.
        self._estimate_history: list[dict[str, float]] = []
        self._plateau_streak: int = 0
        self.plateau_stop_step: int | None = None
        # Physical frequency of the most recent EIG selection, re-injected into
        # the candidate grid so a second batch there is a legitimate EIG outcome
        # rather than being dropped by minimum-spacing thinning.
        self._last_eig_physical_x: float | None = None
        # Theoretical step budget: K_theory × n_theory, computed from the conjugate noise
        # estimate and belief estimates. None until the first check has run.
        self._theory_step_budget: int | None = None

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

    def _thin_candidates_by_step(self, candidates: np.ndarray) -> np.ndarray:
        """Return a subset of *candidates* (physical space) with minimum physical spacing.

        Walks the sorted candidate array once and keeps a candidate only when it
        is at least ``candidate_step_hz`` away from the previously kept one.
        This is O(n) and preserves the first and last candidates so the full
        acquisition range is always represented.
        """
        if len(candidates) <= 1:
            return candidates
        kept = _thin_by_step_indices(np.ascontiguousarray(candidates, dtype=np.float64), self.candidate_step_hz)
        return candidates[kept]

    def _acquire(self) -> float:
        """Select the next measurement point by maximizing EIG over a frequency grid.

        A decaying share of steps instead explores: uniformly over the whole probe window
        (to find dips the posterior has narrowed away from), or near a dip the data already
        show (:meth:`SMCMarginalDistribution.dip_candidates`).
        """
        lo, hi = self._acquisition_bounds()
        if hi <= lo:
            return float(lo)

        # The belief's original (never-narrowed) probe window -- used by the exploration branches
        # so they can still reach a location resampling has already narrowed away from.
        # `_to_experiment_normalized` normalizes against this same full domain, not
        # `_acquisition_bounds()`, so returning a value outside `lo, hi` here is valid.
        orig_lo, orig_hi = self.belief._original_physical_x_bounds

        # The exploration branches are drawn first so the (much more expensive) EIG grid
        # search in _eig_acquire() is skipped entirely on steps where it would be discarded.
        # The uniform exploration probability decays exponentially to focus on EIG as the scan
        # progresses.
        decay = np.exp(-self.inference_step_count / 25.0)
        rand_val = np.random.rand()
        if rand_val < 0.1 * decay:
            return float(np.random.uniform(orig_lo, orig_hi))
        if rand_val < 0.2:
            # Dip-biased sampling: draw within +/-5 MHz of a dip the observations show. This
            # corrects a biased posterior that has drifted away from the true dip location.
            dip_centers = [d.centroid_hz for d in self.belief.dip_candidates if orig_lo <= d.centroid_hz <= orig_hi]
            if dip_centers:
                center = float(np.random.choice(dip_centers))
                return center + float(np.random.uniform(max(-5e6, orig_lo - center), min(5e6, orig_hi - center)))

            # No dip found yet: Thompson sampling of the scanned parameter from the posterior.
            if self._scan_param in self.belief._param_names:
                idx = int(np.random.choice(len(self.belief._weights), p=self.belief._weights))
                p_idx = self.belief._param_names.index(self._scan_param)
                return self.belief._to_physical(self._scan_param, float(self.belief._particles[idx, p_idx]))

        return self._eig_acquire()

    def _eig_acquire(self) -> float:
        """Maximize EIG over the belief's slope-targeted candidate grid."""
        # Retrieve candidates directly from the belief (slope-targeted epoch grid)
        candidates = self.belief.get_candidates()

        # Thin candidates to minimum physical step spacing.
        # The epoch grid window is ±3σ_f, so candidate count ≈ 6σ_f / step_hz:
        # many candidates early (large σ_f), few near convergence (σ_f ≈ step_hz).
        candidates = self._thin_candidates_by_step(candidates)

        # Keep the most recently EIG-selected frequency in the candidate set so a
        # second batch there is a legitimate EIG outcome rather than being dropped
        # by minimum-spacing thinning. EIG's diminishing returns decide when
        # re-batching stops paying off (no explicit repeat counter needed).
        lo, hi = self._acquisition_bounds()
        if (
            self._last_eig_physical_x is not None
            and lo <= self._last_eig_physical_x <= hi
            and not np.any(np.isclose(candidates, self._last_eig_physical_x))
        ):
            candidates = np.append(candidates, self._last_eig_physical_x)

        best = self.belief.select_max_information_gain(candidates, 1)
        result = float(best[0]) if len(best) > 0 else float(candidates[len(candidates) // 2])
        self._last_eig_physical_x = result
        return result

    def _observe_acquisition(self, obs: Observation) -> None:
        """Handle acquisition observations and manually trigger resample checks.

        Updates the belief directly instead of via super(), which would run the
        convergence-milestone check a second time — _check_and_resample already
        performs it with a single shared uncertainty pass.
        """
        self.belief.update(obs)
        self.belief.accumulate_fim(obs)
        self._check_and_resample()

    def _check_and_resample(self) -> None:
        self._resample_if_degenerate()

        # One uncertainty pass shared by the milestone/plateau/CRLB checks
        # (each belief.uncertainty() call is a full O(particles x params) pass).
        # This is the raw (non-robust) value deliberately: milestones, the
        # plateau check, and the CRLB comparison should all reflect the
        # belief's actual claimed precision, not a smoothed one -- see
        # robust_uncertainty's docstring on why it must not become the
        # general-purpose uncertainty.
        physical_uncertainties = self.belief.uncertainty()

        # The streak counter below is the one place raw uncertainty causes a
        # real (not just cosmetic) problem: every SMC resample transiently
        # inflates it (a decaying fraction of particles redrawn from the
        # prior), and _target_params_converged failing on that single step
        # resets the whole streak to 0 -- so a run sitting on the verge of
        # convergence can lose all its progress purely from the resample
        # artifact and cost extra measurements waiting to rebuild the streak.
        # robust_uncertainty() (weighted IQR/1.349) is immune to that: the
        # transient particles are a small minority of the mass and don't move
        # the interquartile range.
        streak_uncertainties = self.belief.robust_uncertainty()
        if self._target_params_converged(streak_uncertainties):
            self._convergence_streak += 1
            if self._convergence_streak >= self._convergence_patience_steps:
                self._is_converged = True
        else:
            self._convergence_streak = 0
        self._check_convergence_milestones(physical_uncertainties)

        if not self._is_converged:
            self._check_estimate_plateau(physical_uncertainties)

        # CRLB-based early-stop: evaluated every step for deterministic convergence reporting.
        if not self._is_converged:
            self._check_crlb_early_stop(physical_uncertainties)

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Extent of the most significant dip the observations show, or None if there is none yet."""
        dips = self.belief.dip_candidates
        return (dips[0].f_min, dips[0].f_max) if dips else None

    def per_dip_windows(self) -> list[tuple[float, float]] | None:
        """Extents of every detected dip, when more than one is still competing.

        None once the detector shows a single dip -- at that point ``bayesian_focus_window()``
        is the window to show, matching the sweep locators' convention of using
        ``per_dip_windows`` only for genuinely multiple regions.
        """
        dips = self.belief.dip_candidates
        return [(d.f_min, d.f_max) for d in dips] if len(dips) >= 2 else None

    def focus_window_candidates(self) -> list[tuple[float, float]] | None:
        """Extent of every detected dip (>=1), for per-step UI animation.

        Unlike ``per_dip_windows()``, this is never gated to "more than one" -- it is read every
        step (see ``Observer.watch``) to animate the candidates narrowing down to the single
        settled focus window over the course of a run.
        """
        dips = self.belief.dip_candidates
        return [(d.f_min, d.f_max) for d in dips] if dips else None

    def _acquisition_done(self) -> bool:
        """Extend base stop logic with a permissive theory-step-budget backstop.

        When enough background points exist to estimate σ̂, computes the theoretical
        minimum step count for uniform sampling to reach the convergence threshold
        and stops if ``inference_step_count`` exceeds ``K_theory × n_theory``.
        This only fires when something has gone wrong — EIG should converge much sooner.
        """
        if super()._acquisition_done():
            return True
        return self._theory_step_budget is not None and self.inference_step_count > self._theory_step_budget

    def _check_estimate_plateau(self, physical_uncertainties) -> None:
        """Stop once the estimate has stopped moving relative to its own error bar.

        For each target parameter, compares the current estimate against the one from
        ``_PLATEAU_WINDOW`` checks ago and expresses the drift in units of that
        parameter's current uncertainty. When *every* target parameter has drifted less
        than ``_PLATEAU_SIGMA_FRAC`` sigma for ``_convergence_patience_steps``
        consecutive checks, further measurements are not changing the answer and the run
        stops.

        Two deliberate choices:

        * Movement is scaled by the parameter's own sigma, not by its bounds, so one
          threshold works across parameters spanning orders of magnitude (Hz-scale
          widths and a dimensionless contrast) without a per-parameter table.
        * It watches the *estimate*, not the uncertainty or a CRLB. Derived quantities
          have repeatedly produced premature stops here (four separate bugs) because
          they can look converged while the estimate is still walking -- a near-singular
          FIM inflates the CRLB, and the particle spread can be narrow and wrong. The
          estimate holding still is the thing actually being claimed at the end of a run.

        Requires the uncertainty to be positive and finite before it will fire, so a
        collapsed or degenerate belief cannot trivially satisfy it.
        """
        if _PLATEAU_WINDOW <= 0:
            return

        est = self.belief.estimates()
        target_params = list(self.belief.model.parameter_names())
        self._estimate_history.append({p: float(est[p]) for p in target_params if p in est})
        # Only the window endpoints are ever compared; keep the list bounded.
        if len(self._estimate_history) > _PLATEAU_WINDOW + 1:
            self._estimate_history.pop(0)
        if len(self._estimate_history) <= _PLATEAU_WINDOW:
            return

        past = self._estimate_history[0]
        current = self._estimate_history[-1]
        checked = 0
        plateaued = True
        for name, now in current.items():
            if name not in past:
                continue
            sigma = float(physical_uncertainties.get(name, math.nan))
            if not math.isfinite(sigma) or sigma <= 0:
                # No usable error bar for this parameter -> cannot judge the drift, and
                # must not silently treat "unmeasurable" as "converged".
                return
            checked += 1
            if abs(now - past[name]) > _PLATEAU_SIGMA_FRAC * sigma:
                plateaued = False
                break

        if checked == 0:
            return

        if plateaued:
            self._plateau_streak += 1
            if self._plateau_streak >= self._convergence_patience_steps:
                self._is_converged = True
                if self.plateau_stop_step is None:
                    self.plateau_stop_step = self.step_count
        else:
            self._plateau_streak = 0

    def _check_crlb_early_stop(self, physical_uncertainties) -> None:
        """Closed-form CRLB convergence check plus the theoretical step-budget backstop.

        The noise level is the belief's conjugate (Inverse-Gamma) estimate -- the only noise
        estimate in the locator. If every target parameter's uncertainty is within
        NVISION_FREQ_CRLB_SAFETY_FACTOR x its CRLB for ``_convergence_patience_steps``
        consecutive checks, marks the run as converged.
        """
        est = self.belief.estimates()
        sigma_hat = self.belief.estimated_noise_std()
        lw_hat, c_hat = _effective_linewidth_and_contrast_estimate(est, self.belief.physical_param_bounds)

        # Compute permissive theoretical step budget from sigma_hat + belief estimates.
        # n_theory = 4σ̂²·lw·bandwidth / (π·c²·T²) — steps needed for uniform sampling to reach T.
        # SBED is better than uniform. (4, not 2: matches the verified CRLB constant in
        # UnitCubeSMCMarginalDistribution.crlb_frequency — n_theory is that same CRLB_var
        # solved for n at rho = n/bandwidth, evaluated at the convergence threshold T.)
        if c_hat is not None and c_hat > 0 and lw_hat > 0:
            freq_lo, freq_hi = self.belief.physical_param_bounds.get("frequency", (0.0, 1.0))
            bandwidth = freq_hi - freq_lo
            if bandwidth > 0:
                from nvision.sim.defaults import NVISION_FREQ_CONVERGENCE_THRESHOLD, NVISION_SBED_STEPS_THEORY_FACTOR

                n_theory = (4.0 * sigma_hat**2 * lw_hat * bandwidth) / (
                    math.pi * c_hat**2 * NVISION_FREQ_CONVERGENCE_THRESHOLD**2
                )
                self._theory_step_budget = max(self.max_steps, int(NVISION_SBED_STEPS_THEORY_FACTOR * n_theory) + 1)

        # Closed-form frequency CRLB (computed at the same conjugate noise estimate); the
        # models define no other analytical Fisher information.
        crlb_f = self.belief.crlb_frequency()
        if not math.isfinite(crlb_f) or crlb_f <= 0:
            return
        crlbs_stored = {"frequency": crlb_f}

        bounds = self.belief.physical_param_bounds
        target_params = list(self.belief.model.parameter_names())

        from nvision.sim.defaults import PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS
        from nvision.sim.locs.bayesian.sequential_bayesian_locator import (
            _SATURATION_VOIGT_RAW_PARAMS,
            saturation_voigt_derived_bounds,
            saturation_voigt_derived_sigmas,
        )

        # Saturation-Voigt: gate via derived effective-HWHM / realized-contrast rather
        # than the raw params (see saturation_voigt_derived_sigmas). Uncertainties and
        # CRLBs propagate through the same local Jacobian, so both are mapped.
        derived_unc: dict[str, float] | None = None
        derived_crlbs: dict[str, float] = {}
        if "saturation" in physical_uncertainties and "sigma_inhom" in physical_uncertainties:
            est = self.belief.estimates()
            derived_unc = saturation_voigt_derived_sigmas(est, physical_uncertainties)
            if derived_unc is not None:
                scaled_crlbs = {p: crlbs_stored.get(p, math.inf) for p in _SATURATION_VOIGT_RAW_PARAMS}
                derived_crlbs = saturation_voigt_derived_sigmas(est, scaled_crlbs) or {}
                bounds = {**bounds, **saturation_voigt_derived_bounds(bounds)}

        # (name, uncertainty, crlb) triples to evaluate.
        eval_items: list[tuple[str, float, float]] = [
            (
                name,
                float(physical_uncertainties.get(name, math.inf)),
                crlbs_stored.get(name, math.inf),
            )
            for name in target_params
            if not (derived_unc is not None and name in _SATURATION_VOIGT_RAW_PARAMS)
        ]
        if derived_unc is not None:
            eval_items.extend((dname, dunc, derived_crlbs.get(dname, math.inf)) for dname, dunc in derived_unc.items())

        # Per-param evaluation. For each param with a valid threshold:
        #   done = unc < convergence threshold
        #
        # Stop conditions:
        #   all_converged_step       : ALL checked params are done (abs or crlb)
        #   splitting_converged_step : self._primary_param is done (abs or crlb)
        #   _is_converged             : ALL checked params hit strict CRLB floor
        checked = 0
        all_crlb_done = True
        all_milestone_done = True
        splitting_milestone_done = False

        for name, unc, crlb_scaled in eval_items:
            # 1. CRLB check
            crlb_threshold = NVISION_FREQ_CRLB_SAFETY_FACTOR * crlb_scaled
            crlb_done = unc < crlb_threshold if crlb_threshold > 0 and math.isfinite(crlb_threshold) else False

            # 2. Absolute threshold check
            absolute_threshold = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS.get(name)
            if absolute_threshold is None:
                lo, hi = bounds.get(name, (0.0, 0.0))
                absolute_threshold = (hi - lo) * self.convergence_threshold
            abs_done = unc < absolute_threshold if absolute_threshold > 0 else False

            # We must be able to evaluate at least one threshold to count this param
            if not (crlb_threshold > 0 and math.isfinite(crlb_threshold)) and not (absolute_threshold > 0):
                continue

            checked += 1

            if not crlb_done:
                all_crlb_done = False

            if not (crlb_done or abs_done):
                all_milestone_done = False

            if name == self._primary_param and (crlb_done or abs_done):
                splitting_milestone_done = True

        if checked == 0:
            return

        # `all_crlb_done` is a single-snapshot read of the current (possibly still
        # locally-plausible-but-wrong-mode) belief state -- require it to hold for
        # `_convergence_patience_steps` consecutive checks before trusting it enough
        # to stop, same bar `_target_params_converged` already has to clear via
        # `_convergence_streak`. Without this, one lucky low-uncertainty snapshot
        # (common in the first ~10 steps, before the particle cloud has had a
        # chance to discriminate between candidate modes) locks in a confidently
        # wrong answer -- empirically ~1-2% of Bayesian-SBED/Voigt repeats stopped
        # at exactly step 9-11 with >1 MHz final error before this gate existed.
        if all_crlb_done:
            self._crlb_convergence_streak += 1
            if self._crlb_convergence_streak >= self._convergence_patience_steps:
                self._is_converged = True
        else:
            self._crlb_convergence_streak = 0

        if splitting_milestone_done and self.splitting_converged_step is None:
            self.splitting_converged_step = self.step_count

        if all_milestone_done and self.all_converged_step is None:
            self.all_converged_step = self.step_count
