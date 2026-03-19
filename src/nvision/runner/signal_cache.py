"""Process-local cache so the same :class:`~nvision.models.experiment.CoreExperiment` (and thus the same
``true_signal`` object) is reused for every locator task that shares the same
``(seed, generator_name, noise_name, repeat_idx)``.

This makes cross-strategy comparisons refer to identical ground-truth signal instances, not only
value-equal draws from the same RNG.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from nvision.models.experiment import CoreExperiment
from nvision.runner.repeat_keys import repeat_seed_int, signal_repeat_key

if TYPE_CHECKING:
    from nvision.models.task import LocatorTask

_LOCK = threading.Lock()
_EXPERIMENT_BY_SIGNAL_KEY: dict[str, CoreExperiment] = {}


def clear_signal_experiment_cache() -> None:
    """Drop all cached experiments (e.g. between tests)."""
    with _LOCK:
        _EXPERIMENT_BY_SIGNAL_KEY.clear()


def get_shared_core_experiment(
    task: LocatorTask,
    repeat_idx: int,
    build: Callable[[random.Random], CoreExperiment],
) -> CoreExperiment:
    """Return a cached :class:`CoreExperiment` if this repeat was seen before.

    On a **cache hit**, the stored instance is returned immediately: ``build`` is not
    called and the true signal is not regenerated.

    On the **first** request for a given key, ``build`` runs once, the result is stored,
    and the same object is returned for all later retrievals.

    The cache key omits ``strategy_name`` so all strategies for the same generator/noise/repeat
    share one experiment and one ``true_signal`` object identity.
    """
    key = signal_repeat_key(task.seed, task.generator_name, task.noise_name, repeat_idx)
    with _LOCK:
        cached = _EXPERIMENT_BY_SIGNAL_KEY.get(key)
        if cached is not None:
            return cached

        # Cache miss only: one-time generation, then reuse for every later lookup.
        rng = random.Random(repeat_seed_int(key))
        exp = build(rng)
        _EXPERIMENT_BY_SIGNAL_KEY[key] = exp
        return exp
