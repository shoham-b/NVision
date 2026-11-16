"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import numpy as np
from numba import njit

from nvision.belief.focus_window import NVISION_SMC_FOCUSING_COVER_FACTOR, next_focus_region
from nvision.models.observation import Observation
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    NVISION_SMC_CANDIDATE_STEP_HZ,
)
from nvision.sim.locs.bayesian.sequential_bayesian_locator import (
    NVISION_CONVERGENCE_PATIENCE,
    SequentialBayesianLocator,
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


# Falsification probes: chance of a probe when the axis is unmeasured (largest gap = 1) and when it is densely
# covered (largest gap -> 0). The floor keeps checking away from a suspected dip for the whole run.
EXPLORE_PROBABILITY_MAX: float = 0.10
EXPLORE_PROBABILITY_FLOOR: float = 0.02


def largest_gap_midpoint_unit(obs_drive_freqs_unit: np.ndarray) -> tuple[float, float]:
    """Midpoint of the widest unmeasured stretch of the unit drive-frequency axis, and that gap's width.

    The axis ``[0, 1]`` is cut at the measured positions (the ends count as boundaries, so an unmeasured
    edge is a gap too). Depends on where measurements are, never on the belief.

    Args:
        obs_drive_freqs_unit: Measured drive frequencies in unit space, any order. shape: (n_observations,)

    Returns:
        ``(midpoint_unit, gap_unit)``, both in ``[0, 1]``.
    """
    xs = np.asarray(obs_drive_freqs_unit, dtype=float)
    if xs.size and (xs.min() < 0.0 or xs.max() > 1.0):
        raise ValueError(f"largest_gap_midpoint_unit: positions must lie in [0, 1], got [{xs.min()}, {xs.max()}]")
    edges = np.concatenate(([0.0], np.sort(xs), [1.0]))
    gaps = np.diff(edges)
    k = int(np.argmax(gaps))
    return 0.5 * (edges[k] + edges[k + 1]), float(gaps[k])


def exploration_probability(largest_gap_unit: float) -> float:
    """Chance of a falsification probe: ``FLOOR + (MAX - FLOOR) * largest_gap_unit``.

    Falls as the axis gets covered (no step clock), never below the floor.
    """
    if not (0.0 <= largest_gap_unit <= 1.0):
        raise ValueError(f"exploration_probability: gap must lie in [0, 1], got {largest_gap_unit!r}")
    return EXPLORE_PROBABILITY_FLOOR + (EXPLORE_PROBABILITY_MAX - EXPLORE_PROBABILITY_FLOOR) * largest_gap_unit


class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Each step measures the candidate x (inside the focus) with the highest Expected Information
    Gain (prediction variance disagreement across particles), except for a falsification probe: with
    probability :func:`exploration_probability` the step measures the middle of the widest unmeasured
    stretch of the whole drive-frequency axis instead. Once SBED has piled measurements onto a
    suspected dip the data cannot contradict it, so only a measurement elsewhere can.
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
        """Next drive frequency in Hz (``next`` maps it to unit): falsification probe, else the EIG maximiser."""
        probe = self._exploration_probe()
        return self._eig_acquire() if probe is None else probe

    def _exploration_probe(self) -> float | None:
        """Drive frequency (Hz) of a falsification probe this step, or None for an EIG step.

        Drawn first so the (much more expensive) EIG search is skipped on probe steps. The probe sits in the
        widest unmeasured stretch of the *full* axis, so it can reach a location the focus has narrowed away
        from; `_to_experiment_normalized` normalizes against that same full axis.
        """
        obs_unit, _ = self.belief.observation_arrays()
        midpoint_unit, gap_unit = largest_gap_midpoint_unit(obs_unit)
        if self.belief._rng.random() >= exploration_probability(gap_unit):
            return None
        drive_freq_min_phys, drive_freq_max_phys = self.belief.drive_freq_bounds_phys
        return float(drive_freq_min_phys + midpoint_unit * (drive_freq_max_phys - drive_freq_min_phys))

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
