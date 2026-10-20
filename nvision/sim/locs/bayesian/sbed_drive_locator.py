"""Bayesian SBED locator that chooses the microwave drive amplitude together with the drive frequency.

Registered as ``Bayesian-SBED-Drive``. Each step picks the ``(drive_freq, drive)`` pair with the highest
expected information gain over the joint grid ``drive candidates x drive-frequency candidates``.

Why the drive is a design lever: in CW ODMR a strong drive power-broadens the dip and deepens it, so
while the dip's location is still unknown a probe is far more likely to land on it; once it is found, a
weak drive gives a narrow line and a steeper slope (better precision). In pulsed ODMR the π-pulse line
width scales with the Rabi frequency in the same way. The EIG trades these off by itself -- the
expected pattern is high amplitude early in the search and lower amplitude later.

Drive candidates are rebuilt once per belief epoch (each resample) from the current posterior, the
same way the drive-frequency candidates are, so there are no hand-tuned amplitude grids:

* CW: amplitudes spanning saturation ``s ~ 0.3 .. 30`` for the posterior's plausible ``drive_amp_sat``
  range (``a = drive_amp_sat * sqrt(s)``), plus full amplitude for the search phase.
* Pulsed: amplitudes log-spaced over the whole range; each carries a π-pulse calibrated from the
  posterior's **dominant mode** of ``rabi_per_amp``. The calibration ratio aliases (depth
  ``sin²(π/2·Ω/Ω_est)`` is periodic), so the posterior can be multimodal and its mean would be a
  meaningless Rabi frequency -- the mode is the calibration an experimenter would trust.
"""

from __future__ import annotations

import math

import numpy as np

from nvision.belief.smc_marginal import NVISION_EXPLORATION_DECAY_STEPS
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT, DriveSetting
from nvision.spectra.nv_center_drive import pi_pulse_duration_s

# Number of drive amplitudes per epoch. Discretisation of the design axis (like the drive-frequency
# candidate budget), not a physics knob: the placement itself follows the posterior.
_N_DRIVE_CANDIDATES: int = 8
# CW saturation range the amplitude candidates cover: below ~0.3 the dip is too shallow to matter, and
# above ~30 it is already broad enough for search (full amplitude is always added on top).
_CW_SATURATION_RANGE: tuple[float, float] = (0.3, 30.0)
# Posterior quantiles of drive_amp_sat used to place the CW amplitudes.
_CW_QUANTILES: tuple[float, float] = (0.1, 0.9)
# Histogram bins for the dominant-mode estimate of rabi_per_amp.
_MODE_BINS: int = 40


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """Weighted quantiles of ``values`` (shape (n,)) with ``weights`` (shape (n,)); returns shape (len(qs),)."""
    order = np.argsort(values)
    v = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cdf = np.cumsum(w)
    if not cdf[-1] > 0:
        raise ValueError("weighted_quantiles: weights sum to zero.")
    cdf /= cdf[-1]
    return np.interp(np.asarray(qs, dtype=np.float64), cdf, v)


def dominant_mode(values: np.ndarray, weights: np.ndarray, bins: int = _MODE_BINS) -> float:
    """Location of the highest-weight mode of a weighted sample (shape (n,) each).

    Histogram over the sample's range; the mode is the weighted mean inside the heaviest bin and its two
    neighbours, which keeps it inside one mode when the sample is multimodal (unlike the overall mean).
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi), weights=w)
    k = int(np.argmax(hist))
    window_lo, window_hi = edges[max(k - 1, 0)], edges[min(k + 2, bins)]
    inside = (v >= window_lo) & (v <= window_hi)
    w_in = w[inside]
    if not w_in.sum() > 0:
        return 0.5 * (edges[k] + edges[k + 1])
    return float(np.sum(v[inside] * w_in) / w_in.sum())


class DriveChoosingSBEDLocator(SequentialBayesianExperimentDesignLocator):
    """SBED over the joint (drive frequency, microwave drive) design space (see module docstring)."""

    CHOOSES_DRIVE = True

    def __init__(self, belief, *args, **kwargs) -> None:
        super().__init__(belief, *args, **kwargs)
        if not belief.model.controls_drive:
            raise ValueError(
                f"{type(self).__name__} chooses the microwave drive, but the belief's model "
                f"({type(belief.model.inner).__name__}) is not drive-controlled."
            )
        self._is_pulsed = bool(belief.model.is_pulsed)
        self._next_drive: DriveSetting | None = None
        self._drive_candidates: tuple[DriveSetting, ...] = ()
        self._drive_candidates_epoch: int = -1
        self._rabi_per_amp_mode: float | None = None

    # ------------------------------------------------------------------
    # Drive candidates
    # ------------------------------------------------------------------

    def _make_drive(self, drive_amp_unit: float) -> DriveSetting:
        """A drive at ``drive_amp_unit``; pulsed drives carry the π-pulse calibrated from the posterior mode."""
        drive_amp_unit = float(min(max(drive_amp_unit, DRIVE_AMP_MIN_UNIT), 1.0))
        if not self._is_pulsed:
            return DriveSetting(drive_amp_unit)
        if self._rabi_per_amp_mode is None:
            raise RuntimeError("Pulsed drive requested before the Rabi calibration was computed.")
        return DriveSetting(drive_amp_unit, pi_pulse_duration_s(self._rabi_per_amp_mode * drive_amp_unit))

    def _amp_candidates_unit(self) -> np.ndarray:
        """This epoch's drive amplitudes (fraction of max field), ascending. shape: (<= _N_DRIVE_CANDIDATES,)"""
        if self._is_pulsed:
            amps = np.geomspace(DRIVE_AMP_MIN_UNIT, 1.0, _N_DRIVE_CANDIDATES)
        else:
            sat = self.belief.particles_phys()["drive_amp_sat"]
            q_lo, q_hi = weighted_quantiles(sat, self.belief._weights, _CW_QUANTILES)
            a_lo = max(q_lo * math.sqrt(_CW_SATURATION_RANGE[0]), DRIVE_AMP_MIN_UNIT)
            a_hi = min(q_hi * math.sqrt(_CW_SATURATION_RANGE[1]), 1.0)
            if not a_hi > a_lo:
                a_lo, a_hi = DRIVE_AMP_MIN_UNIT, 1.0
            amps = np.append(np.geomspace(a_lo, a_hi, _N_DRIVE_CANDIDATES - 1), 1.0)
        return np.unique(np.round(amps, 6))

    def _current_drive_candidates(self) -> tuple[DriveSetting, ...]:
        """Drive candidates for the current belief epoch (rebuilt after every resample)."""
        epoch = self.belief.epoch
        if epoch != self._drive_candidates_epoch:
            if self._is_pulsed:
                rabi = self.belief.particles_phys()["rabi_per_amp"]
                self._rabi_per_amp_mode = dominant_mode(rabi, self.belief._weights)
            self._drive_candidates = tuple(self._make_drive(a) for a in self._amp_candidates_unit())
            self._drive_candidates_epoch = epoch
        return self._drive_candidates

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def _acquire(self) -> float:
        """Next drive frequency in Hz; the matching drive is stored for :meth:`next_drive`."""
        drives = self._current_drive_candidates()
        rng = self.belief._rng
        explore_probability = 0.1 * np.exp(-self.inference_step_count / NVISION_EXPLORATION_DECAY_STEPS)
        if rng.random() < explore_probability:
            # Uniform probe over the full axis, at a log-uniform amplitude.
            drive_freq_min_phys, drive_freq_max_phys = self.belief.drive_freq_bounds_phys
            amp = math.exp(rng.uniform(math.log(DRIVE_AMP_MIN_UNIT), 0.0))
            self._next_drive = self._make_drive(amp)
            return float(rng.uniform(drive_freq_min_phys, drive_freq_max_phys))
        candidate_drive_freq_phys = self._eig_candidate_drive_freq_phys()
        drive_freq_phys, drive = self.belief.select_max_information_gain_driven(candidate_drive_freq_phys, drives)
        self._next_drive = drive
        return drive_freq_phys

    def next_drive(self) -> DriveSetting:
        """The drive chosen together with the last :meth:`next` frequency."""
        if self._next_drive is None:
            raise RuntimeError("next_drive() called before next().")
        drive, self._next_drive = self._next_drive, None
        return drive

    @property
    def rabi_per_amp_calibration(self) -> float | None:
        """Pulsed: the Rabi frequency at full amplitude (Hz) the current π-pulses are calibrated to."""
        return self._rabi_per_amp_mode
