"""The plain sweep's measurements, persisted once and reused by everything that needs a sweep of the same repeat.

``SimpleSweep-optimal`` (the sweep at its task's optimal step k*) is the reference sweep. Its
measurements and fitted curve are stored on the repeat's scan entry (``SWEEP_DATA_KEY``) and read back by

the Bayesian locators' scan figures, which overlay the sweep's points and fit (no separate baseline run).

The sweep of repeat ``r`` and a locator's repeat ``r`` share the generated signal (same seed, generator and
repeat index), so the stored data belongs to the same experiment. A sweep that is not in the cache is an error,
never silently re-simulated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nvision.sim.grid_enums import OPTIMAL_SWEEP_STRATEGY

if TYPE_CHECKING:
    from nvision.cache.locator_repository import LocatorResultsRepository

SWEEP_DATA_KEY = "sweep_data"
# A search measurement may only stand in for a sweep point this close in unit coordinates (the same grid, so ~0).
_MATCH_TOL_UNIT = 1e-9
# How long a dependent task waits for a sweep repeat still being computed by another worker.


class MissingSweepDataError(RuntimeError):
    """The reference sweep's repeat is not in the cache."""


def reference_sweep_strategy() -> str:
    """Strategy name of the sweep whose measurements are shared: ``SimpleSweep-optimal`` (the sweep at k*)."""
    return OPTIMAL_SWEEP_STRATEGY


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


def load_sweep_data(repo: LocatorResultsRepository, combo_kw: dict[str, Any], repeat_idx: int) -> dict[str, Any]:
    """The reference sweep's stored data for repeat ``repeat_idx`` of the combination ``combo_kw``.

    ``combo_kw`` identifies a combination (generator, noise, strategy, seed, max_steps, timeout_s); the sweep is
    the same combination with the reference sweep as strategy. Raises :class:`MissingSweepDataError` otherwise.
    """
    sweep_kw = {**combo_kw, "strategy": reference_sweep_strategy()}
    data = repo.load_scan_field(**sweep_kw, repeat_idx=repeat_idx, field=SWEEP_DATA_KEY)
    if data is None:
        raise MissingSweepDataError(
            f"No {reference_sweep_strategy()} data for repeat {repeat_idx} of {combo_kw['generator']} / "
            f"{combo_kw['noise']} (seed {combo_kw['seed']}): run that strategy first."
        )
    return data


SweepLoader = Callable[[int], dict[str, Any]]
