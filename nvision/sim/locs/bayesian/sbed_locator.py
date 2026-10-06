"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math
import os

import numpy as np
from numba import njit

from nvision.belief.focus_window import NVISION_SMC_FOCUSING_COVER_FACTOR, next_focus_window
from nvision.belief.smc_marginal import NVISION_EXPLORATION_DECAY_STEPS
from nvision.models.observation import Observation
from nvision.sim.defaults import (
    NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR,
    NVISION_CONVERGENCE_THRESHOLD,
    NVISION_SMC_CANDIDATE_STEP_HZ,
)
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.spectra.nv_center import effective_hwhm

# Minimum number of consecutive converged checks before declaring convergence.
# Prevents false early stops on the first measurement, especially with no noise.
NVISION_CONVERGENCE_PATIENCE: int = int(os.getenv("NVISION_CONVERGENCE_PATIENCE", "8"))

# Adaptive plateau stop: give up when the *estimate itself* stops moving, rather than when a derived
# quantity (e.g. a FIM-CRLB) claims the information limit has been reached -- those fired before the
# error-vs-steps curve flattened. Movement is normalized by each parameter's own current uncertainty
# ("drifted less than a fraction of its own error bar over the last WINDOW steps"), so one threshold is
# scale-free across parameters. SIGMA_FRAC was calibrated by replaying the rule inside full-budget runs
# (120 repeats) and confirmed by a live A/B: at 0.25 the median stop falls from 450 to ~220 steps
# (ordinary configs) for a few percent of accuracy; 0.15 keeps more margin on degenerate configs
# (fewer catastrophic fits) at a smaller saving.
_PLATEAU_WINDOW: int = int(os.getenv("NVISION_SBED_PLATEAU_WINDOW", "30"))
_PLATEAU_SIGMA_FRAC: float = float(os.getenv("NVISION_SBED_PLATEAU_SIGMA_FRAC", "0.25"))


@njit(cache=True)
def _thin_by_step_indices(candidate_drive_freq_phys: np.ndarray, step: float) -> np.ndarray:
    """Indices of a greedy minimum-spacing subset of sorted *candidate_drive_freq_phys* (shape: (n,), Hz).

    Keeps the first and last candidate so the full range stays represented.
    """
    n = candidate_drive_freq_phys.shape[0]
    kept = np.empty(n, dtype=np.int64)
    kept[0] = 0
    m = 1
    last = candidate_drive_freq_phys[0]
    for i in range(1, n - 1):
        if candidate_drive_freq_phys[i] - last >= step:
            kept[m] = i
            m += 1
            last = candidate_drive_freq_phys[i]
    kept[m] = n - 1
    return kept[: m + 1]


class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Each step measures the candidate x (inside the focus) with the highest Expected Information
    Gain (prediction variance disagreement across particles), except for a uniform probe over the
    whole drive-frequency axis with probability ``0.1 * exp(-step / NVISION_EXPLORATION_DECAY_STEPS)``.
    """

    REQUIRES_BELIEF = True
    USES_SWEEP_MAX_STEPS = True

    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        center_param: str | None = None,
        candidate_step_hz: float | None = None,
        convergence_patience_steps: int = NVISION_CONVERGENCE_PATIENCE,
    ) -> None:
        super().__init__(
            belief,
            max_steps,
            convergence_threshold,
            center_param,
            convergence_patience_steps=convergence_patience_steps,
        )
        self.candidate_step_hz: float = (
            float(candidate_step_hz) if candidate_step_hz is not None else NVISION_SMC_CANDIDATE_STEP_HZ
        )

        # We handle resampling manually to check convergence at the right moment
        self.belief.auto_resample = False
        self._is_converged = False
        # Own streak, separate from `_convergence_streak` (which gates `_target_params_converged`):
        # `_check_crlb_early_stop`'s primary-parameter test is an independent single-snapshot signal and
        # must not be allowed to set `_is_converged` on one lucky reading.
        self._crlb_convergence_streak: int = 0
        # Rolling history of parameter estimates, for the plateau stop (see
        # _check_estimate_plateau). One dict per convergence check, oldest first.
        self._estimate_history: list[dict[str, float]] = []
        self._plateau_streak: int = 0
        self.plateau_stop_step: int | None = None
        # Step of the most recent focus expansion (-1: never); see _update_focus_window.
        self._last_focus_expansion_step: int = -1

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        center_param: str | None = None,
        parameter_bounds=None,
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
            center_param=center_param,
            candidate_step_hz=candidate_step_hz,
            convergence_patience_steps=convergence_patience_steps,
        )

    def _thin_candidate_drive_freq_by_step(self, candidate_drive_freq_phys: np.ndarray) -> np.ndarray:
        """Return a subset of *candidate_drive_freq_phys* (physical space) with minimum physical spacing.

        Walks the sorted candidate array once and keeps a candidate only when it
        is at least ``candidate_step_hz`` away from the previously kept one.
        This is O(n) and preserves the first and last candidate so the full
        acquisition range is always represented.
        """
        if len(candidate_drive_freq_phys) <= 1:
            return candidate_drive_freq_phys
        kept = _thin_by_step_indices(
            np.ascontiguousarray(candidate_drive_freq_phys, dtype=np.float64), self.candidate_step_hz
        )
        return candidate_drive_freq_phys[kept]

    def _acquire(self) -> float:
        """Next measurement x (physical Hz): a decaying-probability uniform probe, else the EIG maximiser."""
        # Drawn first so the (much more expensive) EIG search is skipped on explore steps. The probe is
        # uniform over the *full* drive-frequency axis so it can still reach a location the focus has narrowed away
        # from; `_to_experiment_normalized` normalizes against that same full axis.
        rng = self.belief._rng
        explore_probability = 0.1 * np.exp(-self.inference_step_count / NVISION_EXPLORATION_DECAY_STEPS)
        if rng.random() < explore_probability:
            drive_freq_min_phys, drive_freq_max_phys = self.belief.drive_freq_bounds_phys
            return float(rng.uniform(drive_freq_min_phys, drive_freq_max_phys))
        return self._eig_acquire()

    def _eig_acquire(self) -> float:
        """The EIG-maximising candidate among the belief's epoch candidate points inside the focus."""
        candidate_drive_freq_phys = self.belief.get_candidate_drive_freq_phys()
        focus_lo, focus_hi = self._acquisition_bounds()
        candidate_drive_freq_phys = candidate_drive_freq_phys[
            (candidate_drive_freq_phys >= focus_lo) & (candidate_drive_freq_phys <= focus_hi)
        ]
        if candidate_drive_freq_phys.size == 0:
            raise ValueError(f"No epoch candidate lies inside the focus [{focus_lo}, {focus_hi}] Hz.")
        candidate_drive_freq_phys = self._thin_candidate_drive_freq_by_step(candidate_drive_freq_phys)
        return float(self.belief.select_max_information_gain(candidate_drive_freq_phys, 1)[0])

    def _observe_acquisition(self, obs: Observation) -> None:
        """Handle acquisition observations and manually trigger resample checks.

        Updates the belief directly instead of via super(), which would run the
        convergence-milestone check a second time — _check_and_resample already
        performs it with a single shared uncertainty pass.
        """
        self.belief.update(obs)
        self.belief.accumulate_fim(obs)
        self._check_and_resample()

    def _update_focus_window(self) -> None:
        """Re-decide the drive-frequency focus from the posterior over ``center_freq`` (right after a resample).

        Only meaningful when ``center_freq`` is inferred (a particle dimension); with the default fixed
        ``center_freq`` the focus stays the full drive-frequency axis. Changes only which candidates may be scanned.
        """
        belief = self.belief
        if "center_freq" not in belief.model.parameter_names():
            return
        particles = belief.particles_phys()
        est = belief.estimates()
        half_span = (
            particles.get("zeeman_split", 0.0)
            + particles.get("split", 0.0)
            + NVISION_SMC_FOCUSING_COVER_FACTOR * effective_hwhm(particles)
        )
        half_span = np.broadcast_to(half_span, particles["center_freq"].shape)
        self._focus, expanded = next_focus_window(
            self._focus,
            center_freq_particles_phys=particles["center_freq"],
            active_half_span_particles_phys=half_span,
            omega_phys=float(effective_hwhm(est)),
            zeeman_split_phys=float(est.get("zeeman_split", 0.0)),
            step=belief._step_count,
            last_expansion_step=self._last_focus_expansion_step,
        )
        if expanded:
            self._last_focus_expansion_step = belief._step_count

    def _check_and_resample(self) -> None:
        if self._resample_if_degenerate():
            self._update_focus_window()

        # One uncertainty pass shared by the milestone/plateau/CRLB checks
        # (each belief.uncertainty() call is a full O(particles x params) pass).
        # This is the raw (non-robust) value deliberately: milestones, the
        # plateau check, and the CRLB comparison should all reflect the
        # belief's actual claimed precision, not a smoothed one -- see
        # robust_uncertainty's docstring on why it must not become the
        # general-purpose uncertainty.
        physical_uncertainties = self.belief.uncertainty()

        # The streak is gated on the robust spread (weighted IQR/1.349), not the raw std: a few outlier
        # particles inflate the std for a step, and _target_params_converged failing on that single step
        # resets the whole streak to 0, while they barely move the interquartile range.
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

    def _primary_crlb_done(self, physical_uncertainties, crlb_f: float) -> bool:
        """Whether the primary parameter has reached its information limit, tightly enough to stop.

        True when both hold for ``self._primary_param`` (physical units, scalars):

        * ``uncertainty < NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR x CRLB`` -- the posterior is no wider
          than the information limit allows, so more measurements cannot shrink it much; and
        * ``CRLB < the parameter's convergence threshold`` -- the limit itself is tight enough.

        The second condition is what keeps this from passing trivially. At a near-degenerate
        point (e.g. ``zeeman_split`` against the widths below the dip-resolution threshold) the
        cumulative FIM is near-singular, so the marginal CRLB is hugely inflated (11.8 MHz where
        the achieved error was 1.3 MHz) and the first condition alone holds from the first steps,
        exactly when the problem is hardest. An inflated CRLB fails the second condition.

        The CRLB is the closed-form center_freq CRLB when the primary parameter is ``center_freq``,
        otherwise its marginal CRLB from the belief's cumulative FIM (``crlb_per_param``). It is
        absent until a FIM exists, in which case the parameter is not done.
        """
        primary = self._primary_param
        if primary is None:
            return False
        crlb = crlb_f if primary == "center_freq" else self.belief.crlb_per_param().get(primary, math.nan)
        unc = float(physical_uncertainties.get(primary, math.nan))
        if not (math.isfinite(crlb) and crlb > 0 and math.isfinite(unc)):
            return False
        return unc < NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR * crlb and crlb < self._effective_primary_threshold()

    def _check_crlb_early_stop(self, physical_uncertainties) -> None:
        """CRLB convergence check on the primary parameter.

        The noise level is the belief's conjugate (Inverse-Gamma) estimate -- the only noise
        estimate in the locator. The run is marked converged once the primary parameter
        (``zeeman_split``/``split``, else ``center_freq``; see ``resolve_primary_param``) passes
        :meth:`_primary_crlb_done` for ``_convergence_patience_steps`` consecutive checks.
        """
        # Closed-form center_freq CRLB (computed at the same conjugate noise estimate); the
        # models define no other analytical Fisher information.
        crlb_f = self.belief.crlb_center_freq()
        if not math.isfinite(crlb_f) or crlb_f <= 0:
            return
        crlbs_stored = {"center_freq": crlb_f}

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
        #   primary_converged_step : self._primary_param is done (abs or crlb)
        #   _is_converged             : primary param passes _primary_crlb_done (streak-gated, below)
        # The milestone `crlb_scaled` is only the closed-form center_freq CRLB (inf for every other
        # parameter), so the milestones for non-center_freq params are decided by the absolute threshold.
        checked = 0
        all_milestone_done = True
        splitting_milestone_done = False

        for name, unc, crlb_scaled in eval_items:
            # 1. CRLB check
            crlb_threshold = NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR * crlb_scaled
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

            if not (crlb_done or abs_done):
                all_milestone_done = False

            if name == self._primary_param and (crlb_done or abs_done):
                splitting_milestone_done = True

        if checked == 0:
            return

        # The primary-parameter CRLB test is a single-snapshot read of the current (possibly still
        # locally-plausible-but-wrong-mode) belief state -- require it to hold for
        # `_convergence_patience_steps` consecutive checks before trusting it enough
        # to stop, same bar `_target_params_converged` already has to clear via
        # `_convergence_streak`. Without this, one lucky low-uncertainty snapshot
        # (common in the first ~10 steps, before the particle cloud has had a
        # chance to discriminate between candidate modes) locks in a confidently
        # wrong answer -- empirically ~1-2% of Bayesian-SBED/Voigt repeats stopped
        # at exactly step 9-11 with >1 MHz final error before this gate existed.
        if self._primary_crlb_done(physical_uncertainties, crlb_f):
            self._crlb_convergence_streak += 1
            if self._crlb_convergence_streak >= self._convergence_patience_steps:
                self._is_converged = True
        else:
            self._crlb_convergence_streak = 0

        if splitting_milestone_done and self.primary_converged_step is None:
            self.primary_converged_step = self.step_count

        if all_milestone_done and self.all_converged_step is None:
            self.all_converged_step = self.step_count
