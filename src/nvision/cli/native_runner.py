"""Native CLI runner using core architecture end-to-end.

No adapters, no bridges. Direct integration with TrueSignal, SignalModel, and core Runner.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from typing import Any

import polars as pl

from nvision.core import Locator, Observer, RunResult
from nvision.core.experiment import CoreExperiment
from nvision.core.runner import Runner
from nvision.core.structures import LocatorTask

log = logging.getLogger(__name__)


def denormalize_x(x_norm: float, x_min: float, x_max: float) -> float:
    """Convert normalized [0,1] x to physical domain."""
    return x_min + x_norm * (x_max - x_min)


def run_result_to_history_df(
    result: RunResult,
    repeat_id: int,
    x_min: float,
    x_max: float,
) -> pl.DataFrame:
    """Convert RunResult to history DataFrame in physical domain.

    Parameters
    ----------
    result : RunResult
        Result from Observer
    repeat_id : int
        Repeat index
    x_min : float
        Physical domain minimum
    x_max : float
        Physical domain maximum

    Returns
    -------
    pl.DataFrame
        History with columns: repeat_id, step, x, signal_values
    """
    rows = []
    for step, snapshot in enumerate(result.snapshots):
        x_norm = snapshot.obs.x
        x_phys = denormalize_x(x_norm, x_min, x_max)
        rows.append(
            {
                "repeat_id": repeat_id,
                "step": step,
                "x": x_phys,
                "signal_values": snapshot.obs.signal_value,
            }
        )

    if not rows:
        return pl.DataFrame(
            {
                "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                "step": pl.Series("step", [], dtype=pl.Int64),
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


def extract_peak_estimates(
    true_signal,
    belief_estimates: dict[str, float],
    locator_result: dict[str, float],
    x_min: float,
    x_max: float,
) -> dict[str, float]:
    """Extract peak position estimates from belief and locator result.

    Parameters
    ----------
    true_signal : TrueSignal
        Ground truth signal
    belief_estimates : dict[str, float]
        Estimates from belief.estimates()
    locator_result : dict[str, float]
        Result from locator.result()
    x_min : float
        Physical domain minimum
    x_max : float
        Physical domain maximum

    Returns
    -------
    dict[str, float]
        Peak estimates in physical domain with conventional naming
    """
    estimates = {}

    # First, try to extract from locator result (highest priority)
    if locator_result:
        for key, value in locator_result.items():
            if isinstance(value, (int, float)):
                # Check if this is a position estimate (needs denormalization)
                if "x" in key.lower() or "pos" in key.lower() or "freq" in key.lower():
                    if 0 <= value <= 1:
                        # Likely normalized
                        estimates[key] = denormalize_x(value, x_min, x_max)
                    else:
                        # Already physical
                        estimates[key] = value
                else:
                    estimates[key] = value

    # Extract frequency parameter if present (typically the main peak location)
    if "frequency" in belief_estimates:
        freq_phys = belief_estimates["frequency"]
        if "peak_x" not in estimates and "frequency" not in estimates:
            estimates["peak_x"] = freq_phys
        if "x1_hat" not in estimates:
            estimates["x1_hat"] = freq_phys

    # For NV center signals, also track split if present
    if "split" in belief_estimates:
        estimates["split"] = belief_estimates["split"]

    return estimates


def run_result_to_finalize_record(
    result: RunResult,
    locator_result: dict[str, float],
    repeat_id: int,
    x_min: float,
    x_max: float,
) -> dict[str, Any]:
    """Convert RunResult and locator result to finalize record.

    Parameters
    ----------
    result : RunResult
        Result from Observer
    locator_result : dict[str, float]
        Result from locator.result() method
    repeat_id : int
        Repeat index
    x_min : float
        Physical domain minimum
    x_max : float
        Physical domain maximum

    Returns
    -------
    dict[str, Any]
        Finalize record with estimated parameters and metrics
    """
    record: dict[str, Any] = {"repeat_id": repeat_id}

    # Get belief estimates
    belief_estimates = result.final_estimates()

    # Extract peak positions
    peak_estimates = extract_peak_estimates(
        result.true_signal,
        belief_estimates,
        locator_result,
        x_min,
        x_max,
    )
    record.update(peak_estimates)

    # Add all belief estimates (already in physical units)
    for key, value in belief_estimates.items():
        if key not in record:
            record[key] = value

    # Add uncertainties and convergence metrics
    if result.snapshots:
        last_belief = result.snapshots[-1].belief
        uncertainties = last_belief.uncertainty()

        # Add parameter uncertainties (denormalize if needed)
        width = x_max - x_min
        for param_name, uncert in uncertainties.items():
            # Position/frequency uncertainties need scaling
            if "pos" in param_name or "freq" in param_name or "x" in param_name:
                record[f"uncert_{param_name}"] = uncert * width
            else:
                record[f"uncert_{param_name}"] = uncert

        # Total entropy
        record["entropy"] = last_belief.entropy()

        # Convergence flag
        record["converged"] = last_belief.converged(threshold=0.01)

        # Number of steps
        record["measurements"] = len(result.snapshots)

    return record


def run_native_simulation_batch(
    task: LocatorTask,
) -> tuple[pl.DataFrame, pl.DataFrame, list[Any], list[float], list[str]]:
    """Run simulation batch natively with core architecture.

    No adapters or bridges. Expects:
    - task.generator to produce TrueSignal directly
    - task.strategy to be a Locator class (with create() classmethod) or dict config

    Parameters
    ----------
    task : LocatorTask
        Task configuration where:
        - task.strategy is Locator class OR dict with {"class": LocatorClass, "config": {...}}

    Returns
    -------
    tuple
        (final_history_df, finalize_results, experiments, repeat_start_times, repeat_stop_reasons)
    """
    # Extract locator class and config
    if isinstance(task.strategy, type) and issubclass(task.strategy, Locator):
        locator_class = task.strategy
        locator_config = {}
    elif isinstance(task.strategy, dict):
        locator_class = task.strategy.get("class")
        locator_config = task.strategy.get("config", {})
        if not isinstance(locator_class, type) or not issubclass(locator_class, Locator):
            raise TypeError(f"Strategy dict must have 'class' as Locator subclass")
    else:
        raise TypeError(f"Native runner requires Locator class or dict, got {type(task.strategy)}")
    gen_obj = task.generator
    noise_obj = task.noise
    n_repeats = task.repeats
    main_seed = task.seed

    # Setup per-repeat state
    repeat_rngs = []
    experiments = []
    repeat_start_times = []
    repeat_stop_reasons = ["" for _ in range(n_repeats)]

    for attempt_idx in range(n_repeats):
        combo_seed_str = f"{main_seed}-{task.generator_name}-{task.strategy_name}-{task.noise_name}-{attempt_idx}"
        attempt_seed = int(hashlib.sha256(combo_seed_str.encode("utf-8")).hexdigest(), 16) % (10**8)

        repeat_rngs.append(random.Random(attempt_seed))

        # Generate true signal
        true_signal = gen_obj.generate(repeat_rngs[-1])

        # Extract domain bounds from true signal parameters
        # Look for x_min/x_max in parameters, or use frequency bounds
        x_min = None
        x_max = None
        for param in true_signal.parameters:
            if param.name == "x_min":
                x_min = param.value
            elif param.name == "x_max":
                x_max = param.value
            elif param.name == "frequency" and x_min is None:
                # Use frequency bounds as domain
                x_min, x_max = param.bounds

        if x_min is None or x_max is None:
            raise ValueError("True signal must have x_min/x_max parameters or frequency bounds")

        # Create experiment
        experiment = CoreExperiment(
            true_signal=true_signal,
            noise=noise_obj,
            x_min=x_min,
            x_max=x_max,
        )
        experiments.append(experiment)
        repeat_start_times.append(time.perf_counter())

    # Run using core architecture
    runner = Runner()
    all_history_rows: list[dict[str, Any]] = []
    finalize_records: list[dict[str, Any]] = []

    for rid in range(n_repeats):
        experiment = experiments[rid]

        # Create observer
        observer = Observer(
            experiment.true_signal,
            experiment.x_min,
            experiment.x_max,
        )

        # Run with observer watching
        try:
            result = observer.watch(runner.run(locator_class, experiment, repeat_rngs[rid], **locator_config))

            # Get final result from a fresh locator instance
            locator_instance = locator_class.create(**locator_config)
            if result.snapshots:
                locator_instance.belief = result.snapshots[-1].belief
            locator_final_result = locator_instance.result()

            stop_reason = "locator_stop"
        except TimeoutError:
            stop_reason = "repeat_timeout"
            result = RunResult(
                snapshots=observer.snapshots,
                true_signal=experiment.true_signal,
            )
            locator_instance = locator_class.create(**locator_config)
            if result.snapshots:
                locator_instance.belief = result.snapshots[-1].belief
            locator_final_result = locator_instance.result()

        repeat_stop_reasons[rid] = stop_reason

        # Convert result to DataFrames
        hist_df = run_result_to_history_df(result, rid, experiment.x_min, experiment.x_max)
        if not hist_df.is_empty():
            all_history_rows.extend(hist_df.to_dicts())

        # Convert result to finalize record
        finalize_record = run_result_to_finalize_record(
            result, locator_final_result, rid, experiment.x_min, experiment.x_max
        )
        finalize_records.append(finalize_record)

    # Build final DataFrames
    final_history_df = (
        pl.DataFrame(all_history_rows)
        if all_history_rows
        else pl.DataFrame(
            {
                "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                "step": pl.Series("step", [], dtype=pl.Int64),
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )
    )

    finalize_results = pl.DataFrame(finalize_records) if finalize_records else pl.DataFrame({"repeat_id": []})

    return final_history_df, finalize_results, experiments, repeat_start_times, repeat_stop_reasons
