"""The plain sweep's measurements, persisted once and reused by everything that needs a sweep of the same repeat.

``SimpleSweep-3perFWHM`` (``SWEEP_POINTS_PER_FWHM`` grid points per prior FWHM) is the reference sweep. Its
measurements and fitted curve are stored on the repeat's scan entry (``SWEEP_DATA_KEY``) and read back by

* the Bayesian locators' scan figures, which overlay the sweep's points and fit (no separate baseline run), and
* Smart-Sweep / Smart-SBED, whose search phase *is* that sweep until the focus: the same grid, so each search
  measurement is the sweep's own measurement at that point (see :class:`SweepReplay`).

The sweep of repeat ``r`` and a locator's repeat ``r`` share the generated signal (same seed, generator and
repeat index), so the stored data belongs to the same experiment. A sweep that is not in the cache is an error,
never silently re-simulated.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from nvision.models.observation import Observation
from nvision.sim.grid_enums import sweep_strategy_name
from nvision.sim.locs.search_grid import SWEEP_POINTS_PER_FWHM

if TYPE_CHECKING:
    from nvision.cache.locator_repository import LocatorResultsRepository
    from nvision.models.experiment import CoreExperiment

SWEEP_DATA_KEY = "sweep_data"
# A search measurement may only stand in for a sweep point this close in unit coordinates (the same grid, so ~0).
_MATCH_TOL_UNIT = 1e-9
# How long a dependent task waits for a sweep repeat still being computed by another worker.
SWEEP_WAIT_S = 3600.0
_POLL_S = 2.0


class MissingSweepDataError(RuntimeError):
    """The reference sweep's repeat is not in the cache."""


def reference_sweep_strategy() -> str:
    """Strategy name of the sweep whose measurements are shared, e.g. ``SimpleSweep-3perFWHM``."""
    return sweep_strategy_name(SWEEP_POINTS_PER_FWHM)


def sweep_data_from_run(run_result: Any) -> dict[str, Any]:
    """The persistable measurements and fit of a finished plain-sweep repeat.

    All lists have one entry per measurement, in measurement order. ``mode_estimates`` is the sweep's
    least-squares fit (None when the fit found no dip: then there is no curve to draw).
    """
    observations = [snapshot.obs for snapshot in run_result.snapshots]
    if not observations:
        raise ValueError("A sweep repeat without measurements has nothing to persist.")
    if any(o.drive is not None for o in observations):
        raise ValueError("Sweep data is not defined for drive-controlled measurements.")
    return {
        "drive_freqs_unit": [float(o.drive_freq_unit) for o in observations],
        "signal_values": [float(o.signal_value) for o in observations],
        "noise_std": [float(o.noise_std) for o in observations],
        "n_shots": [int(o.n_shots) for o in observations],
        "sample_var": [None if o.sample_var is None else float(o.sample_var) for o in observations],
        "mode_estimates": (
            None
            if not run_result.fit_mode_estimates
            else {k: float(v) for k, v in run_result.fit_mode_estimates.items()}
        ),
    }


@dataclass
class SweepReplay:
    """One repeat's persisted sweep, handed out point by point to a locator's search phase.

    Each stored measurement is used once: a search pass that comes back to the same point (the grid was
    covered without a dip) measures afresh.
    """

    data: dict[str, Any]

    def __post_init__(self) -> None:
        self._xs_unit = np.asarray(self.data["drive_freqs_unit"], dtype=np.float64)  # shape: (n_points,)
        self._used = np.zeros(self._xs_unit.size, dtype=bool)

    def take(self, experiment: CoreExperiment, drive_freq_unit: float) -> Observation | None:
        """The sweep's measurement at ``drive_freq_unit``, or None if that point was already handed out.

        Raises ``ValueError`` if no sweep point lies at ``drive_freq_unit``: the search grid and the sweep grid
        are meant to be identical, so a mismatch is a bug, not something to approximate.
        """
        idx = int(np.argmin(np.abs(self._xs_unit - drive_freq_unit)))
        if abs(float(self._xs_unit[idx]) - drive_freq_unit) > _MATCH_TOL_UNIT:
            raise ValueError(
                f"Search point {drive_freq_unit!r} (unit) is not on the persisted sweep grid "
                f"(nearest {float(self._xs_unit[idx])!r})."
            )
        if self._used[idx]:
            return None
        self._used[idx] = True
        return experiment.replay_observation(
            drive_freq_unit=float(self._xs_unit[idx]),
            signal_value=float(self.data["signal_values"][idx]),
            noise_std=float(self.data["noise_std"][idx]),
            n_shots=int(self.data["n_shots"][idx]),
            sample_var=self.data["sample_var"][idx],
        )


def load_sweep_data(
    repo: LocatorResultsRepository, combo_kw: dict[str, Any], repeat_idx: int, *, wait_s: float = 0.0
) -> dict[str, Any]:
    """The reference sweep's stored data for repeat ``repeat_idx`` of the combination ``combo_kw``.

    ``combo_kw`` identifies a combination (generator, noise, strategy, seed, max_steps, timeout_s); the sweep is
    the same combination with the reference sweep as strategy. With ``wait_s`` > 0 a repeat another worker is
    still computing is waited for. Raises :class:`MissingSweepDataError` otherwise.
    """
    sweep_kw = {**combo_kw, "strategy": reference_sweep_strategy()}
    deadline = time.monotonic() + wait_s
    while True:
        data = repo.load_scan_field(**sweep_kw, repeat_idx=repeat_idx, field=SWEEP_DATA_KEY)
        if data is not None:
            return data
        if time.monotonic() >= deadline:
            raise MissingSweepDataError(
                f"No {reference_sweep_strategy()} data for repeat {repeat_idx} of {combo_kw['generator']} / "
                f"{combo_kw['noise']} (seed {combo_kw['seed']}): run that strategy first."
            )
        time.sleep(_POLL_S)


SweepLoader = Callable[[int], dict[str, Any]]
