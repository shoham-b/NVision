import re

with open("nvision/runner/metrics.py", "r") as f:
    code = f.read()

search = """    if not final_history_df.is_empty():
        current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")
    else:
        current_history_df = pl.DataFrame(
            {
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    if current_history_df.is_empty():
        log.info(
            "No measurements recorded for repeat %s (reason=%s); generating baseline scan plot.",
            attempt_idx_in_combo + 1,
            repeat_stop_reasons[attempt_idx_in_combo],
        )

    finalize_row = finalize_results.filter(pl.col("repeat_id") == attempt_idx_in_combo)"""

replace = """    if not final_history_df.is_empty():
        if "repeat_id" in final_history_df.columns:
            current_history_df = final_history_df.filter(pl.col("repeat_id") == attempt_idx_in_combo).drop("repeat_id")
        else:
            current_history_df = final_history_df
    else:
        current_history_df = pl.DataFrame(
            {
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )

    if current_history_df.is_empty():
        log.info(
            "No measurements recorded for repeat %s (reason=%s); generating baseline scan plot.",
            attempt_idx_in_combo + 1,
            repeat_stop_reasons[attempt_idx_in_combo],
        )

    if "repeat_id" in finalize_results.columns:
        finalize_row = finalize_results.filter(pl.col("repeat_id") == attempt_idx_in_combo)
    else:
        finalize_row = finalize_results"""

code = code.replace(search, replace)
with open("nvision/runner/metrics.py", "w") as f:
    f.write(code)

print("metrics OK")

with open("nvision/runner/executor.py", "r") as f:
    code = f.read()

search2 = """        # Pad list inputs for generate_attempt_metrics to align with attempt_idx (0 to self.repeats - 1)
        full_stop_reasons = [""] * start_idx + list(artifacts.stop_reasons)
        full_start_times = [0.0] * start_idx + list(artifacts.repeat_start_times)
        full_timestamps = [""] * start_idx + list(artifacts.repeat_timestamps)

        for i in range(n_repeats):
            attempt_idx = start_idx + i
            entry_base, main_result_row, current_history_df = generate_attempt_metrics(
                n_repeats=self.task.repeat_total or self.repeats,
                attempt_idx_in_combo=attempt_idx,
                gen_name=self.generator_name,
                noise_name=self.noise_name,
                strat_name=self.strategy_name,
                repeat_stop_reasons=full_stop_reasons,
                repeat_start_times=full_start_times,
                repeat_timestamps=full_timestamps,
                current_scan=artifacts.experiments[i],
                final_history_df=artifacts.final_history_df,
                finalize_results=artifacts.finalize_results,
                strat_obj=self.strategy_name,
                max_steps=effective_max_steps,
                seed=self.task.seed,
                run_result=artifacts.run_results[i] if artifacts.run_results else None,
            )"""

replace2 = """        # Pad list inputs for generate_attempt_metrics to align with attempt_idx (0 to self.repeats - 1)
        full_stop_reasons = [""] * start_idx + list(artifacts.stop_reasons)
        full_start_times = [0.0] * start_idx + list(artifacts.repeat_start_times)
        full_timestamps = [""] * start_idx + list(artifacts.repeat_timestamps)

        hist_parts = artifacts.final_history_df.partition_by("repeat_id", as_dict=True) if not artifacts.final_history_df.is_empty() else {}
        fin_parts = artifacts.finalize_results.partition_by("repeat_id", as_dict=True) if not artifacts.finalize_results.is_empty() else {}

        for i in range(n_repeats):
            attempt_idx = start_idx + i

            # Pass pre-partitioned slices down, avoiding O(N^2) filter performance
            current_history_df_part = hist_parts.get((attempt_idx,), pl.DataFrame())
            finalize_row_part = fin_parts.get((attempt_idx,), pl.DataFrame())

            entry_base, main_result_row, current_history_df = generate_attempt_metrics(
                n_repeats=self.task.repeat_total or self.repeats,
                attempt_idx_in_combo=attempt_idx,
                gen_name=self.generator_name,
                noise_name=self.noise_name,
                strat_name=self.strategy_name,
                repeat_stop_reasons=full_stop_reasons,
                repeat_start_times=full_start_times,
                repeat_timestamps=full_timestamps,
                current_scan=artifacts.experiments[i],
                final_history_df=current_history_df_part,
                finalize_results=finalize_row_part,
                strat_obj=self.strategy_name,
                max_steps=effective_max_steps,
                seed=self.task.seed,
                run_result=artifacts.run_results[i] if artifacts.run_results else None,
            )"""

code = code.replace(search2, replace2)
with open("nvision/runner/executor.py", "w") as f:
    f.write(code)
print("executor OK")
