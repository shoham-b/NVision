from __future__ import annotations

import hashlib
import logging
import random
import time
from typing import Any

import polars as pl

from nvision.core.structures import LocatorTask

log = logging.getLogger(__name__)


def run_simulation_batch(  # noqa: C901
    task: LocatorTask,
) -> tuple[pl.DataFrame, pl.DataFrame, list[Any], list[float], list[str]]:
    """Runs a batch of simulations concurrently until completion or timeout."""
    gen_name = task.generator_name
    gen_obj = task.generator
    noise_name = task.noise_name
    noise_obj = task.noise
    strat_name = task.strategy_name
    strat_obj = task.strategy
    n_repeats = task.repeats
    main_seed = task.seed
    loc_max_steps = task.loc_max_steps
    loc_timeout_s = task.loc_timeout_s

    repeat_rngs = []
    initial_scans = []
    repeat_start_times = []
    repeat_stop_reasons = ["" for _ in range(n_repeats)]

    for attempt_idx_in_combo in range(n_repeats):
        combo_seed_str = f"{main_seed}-{gen_name}-{strat_name}-{noise_name}-{attempt_idx_in_combo}"
        attempt_seed = int(hashlib.sha256(combo_seed_str.encode("utf-8")).hexdigest(), 16) % (10**8)

        repeat_rngs.append(random.Random(attempt_seed))
        initial_scans.append(gen_obj.generate(repeat_rngs[-1]))
        repeat_start_times.append(time.perf_counter())

    history_rows = []
    repeats_df = pl.DataFrame(
        {
            "repeat_id": list(range(n_repeats)),
            "active": [True] * n_repeats,
        }
    )

    global_start_time = time.perf_counter()
    for step_num in range(loc_max_steps):
        active_repeats = repeats_df.filter(pl.col("active")).get_column("repeat_id").to_list()
        if not active_repeats:
            break

        if time.perf_counter() - global_start_time > loc_timeout_s:
            log.warning(f"Combination timeout ({loc_timeout_s}s) reached. Finalizing.")
            for rid in active_repeats:
                if not repeat_stop_reasons[rid]:
                    repeat_stop_reasons[rid] = "combination_timeout"
            break

        if history_rows:
            history_df = pl.DataFrame(history_rows)
        else:
            history_df = pl.DataFrame(
                {
                    "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                    "step": pl.Series("step", [], dtype=pl.Int64),
                    "x": pl.Series("x", [], dtype=pl.Float64),
                    "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
                }
            )

        stop_decisions = strat_obj.should_stop(history_df, repeats_df, initial_scans[0])
        if isinstance(stop_decisions, bool):
            if stop_decisions:
                for rid in active_repeats:
                    repeats_df = repeats_df.with_columns(
                        pl.when(pl.col("repeat_id") == rid)
                        .then(pl.lit(False))
                        .otherwise(pl.col("active"))
                        .alias("active")
                    )
                    if not repeat_stop_reasons[rid]:
                        repeat_stop_reasons[rid] = "locator_stop"
        else:
            for row_dict in stop_decisions.to_dicts():
                rid = row_dict["repeat_id"]
                if row_dict["stop"] and rid in active_repeats:
                    repeats_df = repeats_df.with_columns(
                        pl.when(pl.col("repeat_id") == rid)
                        .then(pl.lit(False))
                        .otherwise(pl.col("active"))
                        .alias("active")
                    )
                    if not repeat_stop_reasons[rid]:
                        repeat_stop_reasons[rid] = "locator_stop"

        active_repeats = repeats_df.filter(pl.col("active")).get_column("repeat_id").to_list()
        if not active_repeats:
            break

        proposals = strat_obj.propose_next(history_df, repeats_df, initial_scans[0])
        # Fix: If proposals is a float, wrap it in a DataFrame
        if isinstance(proposals, (float, int)):
            # Only one active repeat is expected in this case
            if len(active_repeats) == 1:
                proposals = pl.DataFrame(
                    {
                        "repeat_id": [active_repeats[0]],
                        "x": [float(proposals)],
                    }
                )
            else:
                raise RuntimeError("Strategy returned a single value but multiple repeats are active.")

        for row_dict in proposals.to_dicts():
            rid = row_dict["repeat_id"]
            if rid not in active_repeats:
                continue

            x_next = row_dict["x"]
            current_scan = initial_scans[rid]
            y_ideal = current_scan.signal(x_next)

            y_measured = (
                noise_obj.over_probe_noise.apply(y_ideal, repeat_rngs[rid], strat_obj)
                if noise_obj and noise_obj.over_probe_noise
                else y_ideal
            )

            row_data = {
                "repeat_id": rid,
                "step": step_num,
                "x": x_next,
                "signal_values": y_measured,
            }

            if hasattr(strat_obj, "_get_locator"):
                try:
                    locator_instance = strat_obj._get_locator(rid)
                    if hasattr(locator_instance, "current_estimates"):
                        est = locator_instance.current_estimates
                        if "entropy" in est:
                            row_data["entropy"] = est["entropy"]
                        if "max_prob" in est:
                            row_data["max_prob"] = est["max_prob"]
                        if "uncertainty" in est:
                            row_data["uncertainty"] = est["uncertainty"]
                except Exception:
                    pass

            history_rows.append(row_data)

    for rid in range(n_repeats):
        if not repeat_stop_reasons[rid]:
            repeat_stop_reasons[rid] = "max_steps_reached"

    if history_rows:
        final_history_df = pl.DataFrame(history_rows)
    else:
        final_history_df = pl.DataFrame(
            {
                "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                "step": pl.Series("step", [], dtype=pl.Int64),
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    finalize_results = strat_obj.finalize(final_history_df, repeats_df, initial_scans[0])
    if isinstance(finalize_results, dict):
        repeat_ids = repeats_df.get_column("repeat_id").to_list() if "repeat_id" in repeats_df.columns else [0]
        finalize_results = pl.DataFrame(
            [{"repeat_id": int(rid), **finalize_results} for rid in repeat_ids],
        )

    return final_history_df, finalize_results, initial_scans, repeat_start_times, repeat_stop_reasons
