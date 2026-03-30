"""Process-local cache so the same ``true_signal`` (and domain) is reused for every locator task that
shares the same ``(seed, generator_name, repeat_idx)``.

Each task still gets its own :class:`~nvision.models.experiment.CoreExperiment` with the correct
``noise`` model; only the ground-truth draw is shared. That aligns cross-noise and cross-strategy
comparisons on identical underlying signals.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from nvision.models.experiment import CoreExperiment
from nvision.runner.repeat_keys import repeat_seed_int, signal_repeat_key
from nvision.spectra.signal import TrueSignal

if TYPE_CHECKING:
    from nvision.models.task import LocatorTask

_LOCK = threading.Lock()
_SIGNAL_BUNDLE_BY_KEY: dict[str, tuple[TrueSignal, float, float]] = {}


def clear_signal_experiment_cache() -> None:
    """Drop all cached signal bundles (e.g. between tests)."""
    with _LOCK:
        _SIGNAL_BUNDLE_BY_KEY.clear()


def get_shared_core_experiment(
    task: LocatorTask,
    repeat_idx: int,
    build: Callable[[random.Random], CoreExperiment],
) -> CoreExperiment:
    """Return a :class:`CoreExperiment` with shared ``true_signal`` for this repeat.

    On a **cache hit**, ``build`` is not called; a new experiment is returned with the cached
    signal/domain and this task's ``noise``.

    On the **first** request for a given key, ``build`` runs once; ``true_signal`` and domain
    are stored and reused for all later tasks (any strategy or noise name).

    The cache key omits ``strategy_name`` and ``noise_name`` so every combination shares one
    ground-truth draw per repeat.
    """
    key = signal_repeat_key(task.seed, task.generator_name, repeat_idx)
    with _LOCK:
        cached = _SIGNAL_BUNDLE_BY_KEY.get(key)
        if cached is not None:
            true_signal, x_min, x_max = cached
            return CoreExperiment(true_signal=true_signal, noise=task.noise, x_min=x_min, x_max=x_max)

        rng = random.Random(repeat_seed_int(key))
        exp = build(rng)
        _SIGNAL_BUNDLE_BY_KEY[key] = (exp.true_signal, exp.x_min, exp.x_max)
        return CoreExperiment(true_signal=exp.true_signal, noise=task.noise, x_min=exp.x_min, x_max=exp.x_max)
