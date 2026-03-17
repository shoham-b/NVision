from __future__ import annotations

import hashlib
import logging
import random
import time
from typing import Any

import polars as pl

from nvision.core.structures import LocatorTask
from nvision.sim.locs.v2.experiment import Experiment
from nvision.sim.locs.v2.runner import Runner

log = logging.getLogger(__name__)


def run_simulation_batch(  # noqa: C901
    task: LocatorTask,
) -> tuple[pl.DataFrame, pl.DataFrame, list[Any], list[float], list[str]]:
    """Run a batch of simulations using the v2 stateless locator interface.

    The v2 locator interface operates in normalized x-space ([0, 1]). This runner
    handles scaling between the scan's physical domain and normalized space so that:
    - locators always see normalized `x`
    - plots/metrics always receive physical-domain `x`
    """
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

    runner = Runner()
    all_history_rows: list[dict[str, Any]] = []
    finalize_records: list[dict[str, Any]] = []

    for rid in range(n_repeats):
        scan = initial_scans[rid]
        width = scan.x_max - scan.x_min
        experiment = Experiment(scan=scan, noise=noise_obj)

        # Run one repeat in normalized x-space
        results, histories, stop_reasons = runner.run(
            locator=strat_obj,
            experiment=experiment,
            repeats=1,
            rng=repeat_rngs[rid],
            max_steps=loc_max_steps,
            timeout_s=float(loc_timeout_s),
            return_history=True,
        )

        hist_norm = histories[0] if histories else pl.DataFrame(schema={"x": pl.Float64, "signal_value": pl.Float64})
        stop_reason = stop_reasons[0] if stop_reasons else "locator_stop"
        repeat_stop_reasons[rid] = stop_reason

        # Convert normalized history to physical-domain history for plots/metrics.
        if not hist_norm.is_empty():
            xs_norm = hist_norm.get_column("x").to_list()
            ys = hist_norm.get_column("signal_value").to_list()
            for step_num, (xn, y) in enumerate(zip(xs_norm, ys, strict=True)):
                all_history_rows.append(
                    {
                        "repeat_id": int(rid),
                        "step": int(step_num),
                        "x": float(scan.x_min + float(xn) * width),
                        "signal_values": float(y),
                    }
                )

        # Scale any x-like outputs back to physical domain.
        res = results[0] if results else {}
        res_scaled: dict[str, Any] = {"repeat_id": int(rid)}
        for k, v in res.items():
            if isinstance(v, int | float) and ("x" in k.lower()):
                res_scaled[k] = float(scan.x_min + float(v) * width)
            else:
                res_scaled[k] = v

        # Provide a conventional x1_hat for single-peak cases if only peak_x exists.
        if "peak_x" in res_scaled and "x1_hat" not in res_scaled:
            res_scaled["x1_hat"] = res_scaled["peak_x"]

        finalize_records.append(res_scaled)

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

    return final_history_df, finalize_results, initial_scans, repeat_start_times, repeat_stop_reasons
