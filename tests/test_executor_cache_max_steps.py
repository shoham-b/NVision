from pathlib import Path

from nvision.runner.executor import _TaskRunner
from nvision.runner.task_builder import TaskListBuildConfig, build_task_list
from nvision.sim.presets import DEFAULT_LOC_MAX_STEPS


class MockMonitor:
    def register_task(self, *args, **kwargs):
        return 1

    def set_total_weight(self, *args, **kwargs):
        pass


def test_cache_kwargs_max_steps_match_run_kwargs(tmp_path: Path):
    tasks, _ = build_task_list(
        TaskListBuildConfig(
            repeats=1,
            seed=42,
            out_dir=tmp_path / "out",
            scans_dir=tmp_path / "out/scans",
            bayes_dir=tmp_path / "out/bayes",
            cache_dir=tmp_path / "cache",
            filter_category="NVCenter",
            filter_strategy="Bayesian-SBED",
            loc_max_steps=DEFAULT_LOC_MAX_STEPS,  # 1500
            log_queue=None,
            progress_queue=None,
            log_level_value=20,
            sweep_max_steps=None,  # DEFAULT
            loc_timeout_s=30,
            no_cache=False,
            ignore_cache_strategy=None,
            require_cache=False,
            filter_generator=None,
            filter_noise=None,
            filter_signal=None,
        ),
        monitor=MockMonitor(),
    )

    task = tasks[0]
    runner = _TaskRunner(task)

    experiment = runner._build_experiment(runner._rng_for_measurement(0))
    locator_class = task.strategy_spec.locator_class
    uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)

    max_steps_for_run = (
        runner._resolve_sweep_max_steps(experiment) if uses_sweep_max_steps else runner.task.loc_max_steps
    )

    cache_kwargs_max_steps = runner._combination_cache_kwargs()["max_steps"]

    assert cache_kwargs_max_steps == max_steps_for_run
