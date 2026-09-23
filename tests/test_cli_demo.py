"""Regression tests for `nv demo`.

`demo()` used to call `run()` with `loc_max_steps=`/`sweep_max_steps=` kwargs that
`run()`'s Typer signature does not accept (TypeError). Step budgets are now applied
through `nvision.sim.defaults` before `run()` resolves its strategies.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import nvision.sim.defaults as sim_defaults
from nvision.cli import demo as demo_module
from nvision.cli.run import run as real_run
from nvision.sim import run_groups as sim_run_groups


def test_demo_default_run_group_is_registered():
    from nvision.cli import defaults as cli_defaults

    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group(cli_defaults.DEMO_RUN_GROUP)
    assert group.generator_names
    assert group.strategy_names


def test_demo_calls_run_with_only_supported_kwargs_and_applies_step_budget(tmp_path: Path, monkeypatch):
    captured: dict = {}

    def fake_run(**kwargs):
        inspect.signature(real_run).bind(**kwargs)
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(demo_module, "run", fake_run)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_MAX_STEPS", sim_defaults.NVISION_SBED_MAX_STEPS)
    monkeypatch.setattr(sim_defaults, "NVISION_SWEEP_MAX_STEPS", sim_defaults.NVISION_SWEEP_MAX_STEPS)

    result = demo_module.demo(
        repeats=1,
        loc_max_steps=7,
        no_cache=False,
        open_browser=False,
        runners=1,
        out=tmp_path,
        run_group=demo_module.cli_defaults.DEMO_RUN_GROUP,
    )

    assert result == 0
    assert "loc_max_steps" not in captured
    assert "sweep_max_steps" not in captured
    assert sim_defaults.NVISION_SBED_MAX_STEPS == 7
    assert sim_defaults.NVISION_SWEEP_MAX_STEPS == 7
