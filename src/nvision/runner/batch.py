"""Simulation batch runner — executes N repeats of a locator task end-to-end.

Owns per-repeat seed derivation, experiment setup, and the measurement loop.
Result conversion is handled by ``runner.convert``.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from typing import Any

import polars as pl

from nvision.models.experiment import CoreExperiment
from nvision.models.locator import Locator
from nvision.models.observer import Observer, RunResult
from nvision.models.task import LocatorTask
from nvision.runner.convert import run_result_to_finalize_record, run_result_to_history_df
from nvision.runner.loop import run_loop

log = logging.getLogger(__name__)


def run_simulation_batch(
    task: LocatorTask,
) -> tuple[pl.DataFrame, pl.DataFrame, list[CoreExperiment], list[float], list[str]]:
    """Run N repeats of a locator task and return aggregated results.

    Parameters
    ----------
    task : LocatorTask
        Full task configuration. ``task.strategy`` must be a ``Locator``
        subclass or a dict ``{"class": LocatorClass, "config": {...}}``.

    Returns
    -------
    tuple
        ``(history_df, finalize_df, experiments, repeat_start_times, stop_reasons)``

        - ``history_df``: all measurement snapshots across all repeats
          (columns: repeat_id, step, x, signal_values)
        - ``finalize_df``: one row per repeat with final estimates and metrics
        - ``experiments``: per-repeat ``CoreExperiment`` instances
        - ``repeat_start_times``: ``time.perf_counter()`` value at repeat start
        - ``stop_reasons``: ``"locator_stop"`` or ``"repeat_timeout"`` per repeat
    """
    # Resolve locator class and config from task.strategy
    if isinstance(task.strategy, type) and issubclass(task.strategy, Locator):
        locator_class = task.strategy
        locator_config: dict[str, Any] = {}
    elif isinstance(task.strategy, dict):
        locator_class = task.strategy.get("class")
        locator_config = task.strategy.get("config", {})
        if not isinstance(locator_class, type) or not issubclass(locator_class, Locator):
            raise TypeError("Strategy dict must have 'class' as a Locator subclass")
    else:
        raise TypeError(f"run_simulation_batch requires a Locator class or dict, got {type(task.strategy)}")

    gen_obj = task.generator
    noise_obj = task.noise
    n_repeats = task.repeats
    main_seed = task.seed

    # ── Phase 1: set up per-repeat RNGs and experiments ──────────────────────
    repeat_rngs: list[random.Random] = []
    experiments: list[CoreExperiment] = []
    repeat_start_times: list[float] = []
    stop_reasons: list[str] = [""] * n_repeats

    for attempt_idx in range(n_repeats):
        seed_str = f"{main_seed}-{task.generator_name}-{task.strategy_name}-{task.noise_name}-{attempt_idx}"
        attempt_seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (10**8)
        rng = random.Random(attempt_seed)
        repeat_rngs.append(rng)

        true_signal = gen_obj.generate(rng)

        x_min = x_max = None
        for param in true_signal.parameters:
            if param.name == "x_min":
                x_min = param.value
            elif param.name == "x_max":
                x_max = param.value
            elif param.name == "frequency" and x_min is None:
                x_min, x_max = param.bounds

        if x_min is None or x_max is None:
            raise ValueError("TrueSignal must expose x_min/x_max parameters or frequency bounds")

        experiments.append(CoreExperiment(true_signal=true_signal, noise=noise_obj, x_min=x_min, x_max=x_max))
        repeat_start_times.append(time.perf_counter())

    # ── Phase 2: run measurement loops ───────────────────────────────────────
    history_dfs: list[pl.DataFrame] = []
    finalize_records: list[dict[str, Any]] = []

    for rid in range(n_repeats):
        experiment = experiments[rid]
        observer = Observer(experiment.true_signal, experiment.x_min, experiment.x_max)

        try:
            result = observer.watch(run_loop(locator_class, experiment, repeat_rngs[rid], **locator_config))
            stop_reasons[rid] = "locator_stop"
        except TimeoutError:
            result = RunResult(snapshots=observer.snapshots, true_signal=experiment.true_signal)
            stop_reasons[rid] = "repeat_timeout"

        locator_instance = locator_class.create(**locator_config)
        if result.snapshots:
            locator_instance.belief = result.snapshots[-1].belief
        locator_final_result = locator_instance.result()

        hist_df = run_result_to_history_df(result, rid, experiment.x_min, experiment.x_max)
        if not hist_df.is_empty():
            history_dfs.append(hist_df)

        finalize_records.append(
            run_result_to_finalize_record(result, locator_final_result, rid, experiment.x_min, experiment.x_max)
        )

    # ── Phase 3: assemble DataFrames ─────────────────────────────────────────
    _empty_history = pl.DataFrame(
        {
            "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
            "step": pl.Series("step", [], dtype=pl.Int64),
            "x": pl.Series("x", [], dtype=pl.Float64),
            "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
        }
    )
    history_df = pl.concat(history_dfs) if history_dfs else _empty_history
    finalize_df = pl.DataFrame(finalize_records) if finalize_records else pl.DataFrame({"repeat_id": []})

    return history_df, finalize_df, experiments, repeat_start_times, stop_reasons
