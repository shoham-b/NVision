import re

with open("nvision/runner/executor.py", "r") as f:
    code = f.read()

search = """        # Pad list inputs for generate_attempt_metrics to align with attempt_idx (0 to self.repeats - 1)
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
                final_history_df=artifacts.history_df,
                finalize_results=artifacts.finalize_df,
                strat_obj=self.task.strategy,
                max_steps=effective_max_steps,
                seed=self.task.seed,
                run_result=artifacts.run_results[i] if i < len(artifacts.run_results) else None,
            )"""

replace = """        # Pad list inputs for generate_attempt_metrics to align with attempt_idx (0 to self.repeats - 1)
        full_stop_reasons = [""] * start_idx + list(artifacts.stop_reasons)
        full_start_times = [0.0] * start_idx + list(artifacts.repeat_start_times)
        full_timestamps = [""] * start_idx + list(artifacts.repeat_timestamps)

        hist_parts = artifacts.history_df.partition_by("repeat_id", as_dict=True) if not artifacts.history_df.is_empty() else {}
        fin_parts = artifacts.finalize_df.partition_by("repeat_id", as_dict=True) if not artifacts.finalize_df.is_empty() else {}

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
                strat_obj=self.task.strategy,
                max_steps=effective_max_steps,
                seed=self.task.seed,
                run_result=artifacts.run_results[i] if i < len(artifacts.run_results) else None,
            )"""

code = code.replace(search, replace)
with open("nvision/runner/executor.py", "w") as f:
    f.write(code)
print("executor OK")
