"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from nvision.belief.focus_window import NVISION_SMC_FOCUSING_COVER_FACTOR, next_focus_region
from nvision.belief.smc_marginal import NVISION_EXPLORATION_DECAY_STEPS
from nvision.models.observation import Observation
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    NVISION_SMC_CANDIDATE_STEP_HZ,
)
from nvision.sim.locs.bayesian.sequential_bayesian_locator import (
    NVISION_CONVERGENCE_PATIENCE,
    SequentialBayesianLocator,
)
from nvision.sim.locs.search_grid import (
    COARSE_SEARCH_POINTS_PER_FWHM,
    FOCUS_SWEEP_POINTS_PER_FWHM,
    FocusSweepGrid,
    RepeatingGrid,
    prior_fwhm_phys,
)
from nvision.spectra.nv_center import effective_hwhm


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
        """Next drive frequency in Hz (``next`` maps it to unit): uniform explore draw, else the EIG maximiser."""
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
        """The EIG-maximising candidate among the belief's epoch candidate points inside a focus section."""
        candidate_drive_freq_phys = self._eig_candidate_drive_freq_phys()
        return float(self.belief.select_max_information_gain(candidate_drive_freq_phys, 1)[0])

    def _eig_candidate_drive_freq_phys(self) -> np.ndarray:
        """The epoch's candidate drive frequencies (Hz) inside a focus section, thinned to ``candidate_step_hz``."""
        candidate_drive_freq_phys = self.belief.get_candidate_drive_freq_phys()
        candidate_drive_freq_phys = candidate_drive_freq_phys[self._focus.contains(candidate_drive_freq_phys)]
        if candidate_drive_freq_phys.size == 0:
            raise ValueError(f"No epoch candidate lies inside the focus sections {self._focus.sections} Hz.")
        return self._thin_candidate_drive_freq_by_step(candidate_drive_freq_phys)

    def _observe_acquisition(self, obs: Observation) -> None:
        """Handle acquisition observations and manually trigger resample checks.

        Updates the belief directly instead of via super(), which would run the
        convergence-milestone check a second time — _check_and_resample already
        performs it with a single shared uncertainty pass.
        """
        self.belief.update(obs)
        self.belief.accumulate_fim(obs)
        self._check_and_resample()

    def _focus_dips(self):
        """The dips the focus is decided from: those found when the current epoch's candidates were generated.

        ``_resample`` regenerates the candidates, which refreshes ``belief.dip_candidates`` from the
        measurements (``find_dips``), so right after a resample they are current.
        """
        return self.belief.dip_candidates

    def _update_focus_window(self) -> None:
        """Re-decide the drive-frequency focus from the dips the observations show.

        The dips are those the measurements showed when the epoch's candidates were generated (see
        :meth:`_focus_dips`). The focus has one
        section per dip -- its extent padded by ``NVISION_SMC_FOCUSING_COVER_FACTOR`` x the posterior-mean
        effective HWHM, overlapping sections merged (see :func:`next_focus_region`); with no significant
        dip the focus is unchanged.
        Dips are measurement facts, so this is active whether or not ``zero_field_freq`` is inferred.
        Changes only which candidates may be scanned.
        """
        belief = self.belief
        # Widest width the drive can produce (drive-controlled), so each section covers every candidate drive.
        width_amps = belief.width_drive_amp_units()
        width_amp = None if width_amps == [None] else max(width_amps)
        self._focus, expanded = next_focus_region(
            self._focus,
            dips=self._focus_dips(),
            pad_phys=NVISION_SMC_FOCUSING_COVER_FACTOR
            * float(effective_hwhm(belief.estimates(), drive_amp_unit=width_amp)),
            step=belief._step_count,
            last_expansion_step=self._last_focus_expansion_step,
        )
        if expanded:
            self._last_focus_expansion_step = belief._step_count

    def _check_and_resample(self) -> None:
        if self._resample_if_degenerate():
            self._update_focus_window()

        self._update_stop_rules()

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Outer extent of the focus sections (end-of-run display), or None while the focus is the full axis.

        The single-interval summary the end-of-run UI field expects (shared with the sweep locators).
        When the focus has several sections, the gaps between them are inside this extent but are NOT
        scanned; the exact sections are :meth:`focus_window_candidates`.
        """
        if self._focus.is_full_domain():
            return None
        return (self._focus.lo, self._focus.hi)

    def per_dip_windows(self) -> list[tuple[float, float]] | None:
        """Extents of every detected dip (evidence from the observations), when more than one is competing.

        Display only: the unpadded extents behind the focus sections.
        None for a single dip, matching the sweep locators' convention of using ``per_dip_windows``
        only for genuinely multiple regions.
        """
        dips = self.belief.dip_candidates
        return [(d.f_min, d.f_max) for d in dips] if len(dips) >= 2 else None

    def focus_window_candidates(self) -> list[tuple[float, float]] | None:
        """The focus sections at this step (one entry per section), for per-step UI animation.

        Read every step (see ``Observer.watch``) so the timeline shows the real sections the
        acquisition is restricted to expanding and narrowing over the run. None while the focus is
        the full axis.
        """
        if self._focus.is_full_domain():
            return None
        return list(self._focus.sections)


class _SearchThenFocusMixin:
    """Shared structure of the two locators that search a coarse grid first, then work inside the focus.

    The focus (the :class:`~nvision.belief.focus_window.FocusRegion` of the dips measured so far) is
    re-decided after every measurement, not only at resamples, so the switch from search to focus happens
    the step the data first show a dip. Both subclasses are Bayesian-SBED locators in everything but
    acquisition -- same belief, same convergence stopping, same focus rule -- so a comparison
    with :class:`SequentialBayesianExperimentDesignLocator` or with each other isolates how the next
    measurement is chosen. Neither makes SBED's random exploration probe: the grid is the exploration.
    """

    _grid: RepeatingGrid
    _last_focus: object

    def _init_search_grid(self, points_per_fwhm: int) -> None:
        """Size the search grid from the prior FWHM: ``points_per_fwhm`` grid points per FWHM."""
        self._fwhm_prior_phys = prior_fwhm_phys(self.belief)
        self._grid = RepeatingGrid(
            self._drive_freq_min_phys, self._drive_freq_max_phys, self._fwhm_prior_phys / points_per_fwhm
        )
        self._last_focus = self._focus

    def _focus_dips(self):
        """Dips from the measurements so far (``belief.current_dips()``), not only as of the last resample.

        Re-scanned at most every ``NVISION_DIP_SCAN_EVERY`` measurements (see ``_scanned_dips``), so the switch from
        search to focus can come up to that many measurements after the dip first becomes visible.
        """
        return self._scanned_dips()

    def _refresh_focus(self) -> bool:
        """Re-decide the focus from the dips measured so far; True if it changed."""
        self._update_focus_window()
        changed = self._focus is not self._last_focus
        self._last_focus = self._focus
        return changed

    def _acquire(self) -> float:
        changed = self._refresh_focus()
        drive_freq_phys = self._pick(focus_changed=changed)
        self._grid.record(drive_freq_phys)
        return drive_freq_phys

    def _pick(self, *, focus_changed: bool) -> float:
        raise NotImplementedError


class SmartSBEDLocator(_SearchThenFocusMixin, SequentialBayesianExperimentDesignLocator):
    """SBED that searches a coarse FWHM-spaced grid first, then runs ordinary SBED inside the focus.

    Before the data show a dip, the candidates are the not-yet-measured points of a uniform grid with one
    point per prior FWHM (swept again if the whole axis is covered without a dip) and the
    expected-information-gain maximiser among them is measured. Once a dip is found the focus narrows to it
    and acquisition is plain SBED: the EIG maximiser over the epoch's slope-targeted candidates inside the
    focus. Intended for a resonance whose width is already known and whose position moved.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_search_grid(COARSE_SEARCH_POINTS_PER_FWHM)

    def _pick(self, *, focus_changed: bool) -> float:
        if self._focus.is_full_domain():
            candidates = self._grid.next_unmeasured(self._focus)
            return float(self.belief.select_max_information_gain(candidates, 1)[0])
        if focus_changed:
            # The epoch candidates predate the dip that moved the focus; rebuild them around it.
            self.belief.refresh_candidates()
        return self._eig_acquire()


# Smart-Sweep stops re-sweeping and measures only at the estimated max-information points once the posterior
# expects those points to carry at least (1 - this) of the information the true optimum would. Same delta as the
# stage-3 ("refine") rule's REFINE_INFO_LOSS in nvision.metrics.stages, deliberately a separate constant so that
# editing the metric never silently changes the acquisition.
SMART_SWEEP_REFINE_INFO_LOSS: float = 0.1
# Spacing of the drive-frequency grid the sensitivity argmax is taken over, as a fraction of the estimated HWHM.
# eta loses only second order in the grid offset (Lorentzian: 4.5 (eps/HWHM)^2), so HWHM/50 costs < 0.1%.
_REFINE_GRID_HWHM_FRACTION: float = 1.0 / 50.0
# Central-difference step on the primary parameter, as a fraction of the estimated HWHM (the primary in Hz).
_REFINE_GRAD_STEP_HWHM_FRACTION: float = 0.01
# Particles (deterministic weight-systematic subsample) the posterior-expected efficiency is averaged over.
_REFINE_EXPECTATION_PARTICLES: int = 256


class SmartSweepLocator(_SearchThenFocusMixin, SequentialBayesianExperimentDesignLocator):
    """A FWHM/2 sweep over the axis, a FWHM/10 sweep inside the focus, then measurements at the max-information points.

    **Search / focus.** Each step measures the lowest-frequency grid point not yet measured. With the focus still
    the whole axis that is a plain FWHM/2 sweep (``COARSE_SEARCH_POINTS_PER_FWHM``). Once the data show a dip, the
    focus narrows to it and a fresh grid of spacing FWHM/10 (``FOCUS_SWEEP_POINTS_PER_FWHM``) is laid inside each
    focus section, starting at its lower edge and ignoring the coarser search measurements: fine enough that a
    grid point lies within the refine efficiency tolerance of each max-information point (see ``search_grid``).
    If the focus grid is exhausted before the refine rule fires, it is swept again from the focus's lower edge at
    the same spacing. The run may also converge (the stop rule) without ever reaching refine mode; that is fine.

    **Refine.** A sweep keeps spending measurements across the whole dip, most of them far from where the
    signal is sensitive to the primary parameter ``theta`` (``zeeman_shift``). Once the dip is located the
    Fisher-optimal points follow from the model itself: the argmax over drive frequency of the sensitivity
    ``g(x)^2 = (d signal / d theta)^2`` at the current posterior-mean estimate (for a Lorentzian the two
    steepest-slope points ``x0 +- HWHM / sqrt(3)``; computed numerically, so Voigt etc. work too). In refine
    mode every measurement goes to one of those maxima -- each local maximum of ``g^2`` within ``1 - delta`` of
    the global one, i.e. both slopes of a single dip -- cycling through them, and the maxima are recomputed from
    the updated estimate at every step. No information-gain (EIG) criterion is involved.

    **Switch rule.** Measuring at the estimated optimum ``x*`` only pays if ``x*`` is known well enough. For a
    posterior particle ``p`` taken as the truth, a measurement at ``x*`` would carry the fraction
    ``eta_p = g_p(x*)^2 / max_x g_p(x)^2`` of the information the true optimum gives. Refine mode is used on a
    step when, inside a focus, every chosen point has posterior-expected efficiency ``E_post[eta] >= 1 - delta``
    with ``delta = SMART_SWEEP_REFINE_INFO_LOSS``: the stage-3 criterion of ``nvision.metrics.stages`` (refine
    CRLB within ``1 / (1 - delta)`` of the oracle's), evaluated under the posterior instead of the unknown truth.
    Because the loss is second order in the position error (Lorentzian ``eta ~ 1 - 4.5 (eps / HWHM)^2``) this is
    "``x*`` known to about 0.15 HWHM", but taken from the model, not a Lorentzian formula, and it accounts for
    the nuisance-parameter (width) uncertainty too. The rule is re-evaluated every step, not latched: if the
    posterior broadens again (a new dip, a resample) the focus sweep resumes. The belief is known to be somewhat
    overconfident, so the switch may come slightly early; measuring near ``x*`` then tightens the estimate.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_search_grid(COARSE_SEARCH_POINTS_PER_FWHM)
        self._focus_grid = FocusSweepGrid(self._fwhm_prior_phys / FOCUS_SWEEP_POINTS_PER_FWHM)
        if self._primary_param is None:
            raise ValueError("SmartSweepLocator needs a primary parameter for its max-information refine mode.")
        if self.belief.model.controls_drive:
            raise ValueError("SmartSweepLocator measures at one fixed drive; got a drive-controlled model.")
        self._refine_count: int = 0
        # 0-based index of each measurement refine mode chose (`next` has already counted it when `_pick` runs).
        self.refine_steps: list[int] = []

    def _pick(self, *, focus_changed: bool) -> float:
        if self._focus.is_full_domain():
            return float(self._grid.next_unmeasured(self._focus)[0])
        points_phys = self._refine_points_phys()
        if points_phys is not None:
            self.refine_steps.append(self.inference_step_count - 1)
            pick_phys = float(points_phys[self._refine_count % points_phys.size])
            self._refine_count += 1
            return pick_phys
        pick_phys = float(self._focus_grid.next_unmeasured(self._focus)[0])
        self._focus_grid.record(pick_phys)
        return pick_phys

    def _primary_sensitivity_sq(self, xs_unit: np.ndarray, part_cols: list[np.ndarray]) -> np.ndarray:
        """``(d signal / d theta_unit)^2`` at ``xs_unit`` for each particle, by central differences.

        ``xs_unit``: drive frequencies on the full-axis unit coordinate, shape: (n_points,).
        ``part_cols``: unit-cube parameter columns in belief order, each shape: (n_particles,).
        Returns shape: (n_points, n_particles). The derivative is w.r.t. the *unit* primary coordinate, which
        differs from the physical one by the constant bound width -- irrelevant to argmaxes and ratios.
        """
        belief = self.belief
        j = belief._param_names.index(self._primary_param)
        lo_phys, hi_phys = belief.physical_param_bounds[self._primary_param]
        hwhm_phys = float(effective_hwhm(belief.estimates()))
        h_unit = _REFINE_GRAD_STEP_HWHM_FRACTION * hwhm_phys / (hi_phys - lo_phys)
        theta_unit = part_cols[j]
        up_unit = np.minimum(theta_unit + h_unit, 1.0).astype(theta_unit.dtype)
        dn_unit = np.maximum(theta_unit - h_unit, 0.0).astype(theta_unit.dtype)
        denom_unit = up_unit.astype(np.float64) - dn_unit.astype(np.float64)
        if not np.all(denom_unit > 0.0):
            raise ValueError(f"Degenerate finite-difference step for {self._primary_param!r} (h_unit={h_unit!r}).")
        drives = [belief.model.fixed_drive] * xs_unit.size
        cols_up = [up_unit if k == j else c for k, c in enumerate(part_cols)]
        cols_dn = [dn_unit if k == j else c for k, c in enumerate(part_cols)]
        y_up = np.asarray(belief._predict_at_drives(xs_unit, cols_up, drives), dtype=np.float64)
        y_dn = np.asarray(belief._predict_at_drives(xs_unit, cols_dn, drives), dtype=np.float64)
        return ((y_up - y_dn) / denom_unit[None, :]) ** 2

    def _refine_points_phys(self) -> np.ndarray | None:
        """The max-information drive frequencies (Hz, ascending, shape: (n_points,)) if the switch rule holds.

        None while the rule fails (see the class docstring). The argmax is taken over the focus sections at a
        spacing of ``_REFINE_GRID_HWHM_FRACTION`` x the estimated HWHM.
        """
        belief = self.belief
        hwhm_phys = float(effective_hwhm(belief.estimates()))
        if not (hwhm_phys > 0.0) or not math.isfinite(hwhm_phys):
            raise ValueError(f"Estimated HWHM is not positive and finite: {hwhm_phys!r} Hz.")
        grid_phys = np.concatenate(
            [
                np.linspace(lo, hi, max(3, math.ceil((hi - lo) / (_REFINE_GRID_HWHM_FRACTION * hwhm_phys)) + 1))
                for lo, hi in self._focus.sections
            ]
        )
        x_lo_phys, x_hi_phys = belief.drive_freq_bounds_phys
        grid_unit = (grid_phys - x_lo_phys) / (x_hi_phys - x_lo_phys)

        # The sensitivity maxima at the posterior-mean estimate.
        est_unit = belief._estimates_unit()
        mean_cols = [np.array([est_unit[name]], dtype=belief._particles.dtype) for name in belief._param_names]
        g2_mean = self._primary_sensitivity_sq(grid_unit, mean_cols)[:, 0]
        g2_best = float(np.max(g2_mean))
        if not g2_best > 0.0:
            return None  # the estimated dip lies outside the focus: no optimum to measure at
        peaks = np.flatnonzero((g2_mean[1:-1] >= g2_mean[:-2]) & (g2_mean[1:-1] > g2_mean[2:])) + 1
        peaks = peaks[g2_mean[peaks] >= (1.0 - SMART_SWEEP_REFINE_INFO_LOSS) * g2_best]
        if peaks.size == 0:
            return None  # the global maximum is on a section edge (the focus cuts the dip): no interior optimum yet

        # Posterior-expected efficiency at those points over a deterministic weight-systematic particle subsample.
        w = belief._weights / belief._weights.sum()
        n_sub = min(_REFINE_EXPECTATION_PARTICLES, w.size)
        sub_idx = np.minimum(np.searchsorted(np.cumsum(w), (np.arange(n_sub) + 0.5) / n_sub), w.size - 1)
        sub_cols = [belief._particles[sub_idx, k] for k in range(len(belief._param_names))]
        g2_sub = self._primary_sensitivity_sq(grid_unit, sub_cols)  # (n_grid, n_sub)
        g2_sub_best = np.max(g2_sub, axis=0)
        # A particle with no sensitivity anywhere in the focus gains nothing at x* either: its eta is 0.
        has_info = g2_sub_best > 0.0
        eta_sub = np.where(has_info, g2_sub[peaks, :] / np.where(has_info, g2_sub_best, 1.0), 0.0)
        if float(np.min(eta_sub.mean(axis=1))) < 1.0 - SMART_SWEEP_REFINE_INFO_LOSS:
            return None
        return grid_phys[peaks]
