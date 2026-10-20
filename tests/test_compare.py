"""Stage-by-stage locator comparison table and sweep strategy names."""

from __future__ import annotations

import polars as pl
import pytest

from nvision.metrics.compare import compare_locators, usable_runs
from nvision.sim.combinations import SWEEP_SPACING_DIVISORS, CombinationGrid
from nvision.sim.grid_enums import is_plain_sweep, sweep_strategy_name
from nvision.sim.run_groups import get_run_group


def _results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": ["A", "A", "A", "B", "B"],
            "measurements": [10, 20, 0, 30, 30],
            "failure_reason": [None, "max_steps", "timeout", None, None],
            "final_err_primary": [100.0, 300.0, 1.0, 50.0, 70.0],
            "uncert": [150.0, 100.0, 1.0, 40.0, 60.0],
            "focus_found_step": [5, 7, None, None, 12],
            "max_info_step": [9, None, None, None, None],
            "stage1_steps": [5, 7, None, 30, 12],
            "stage1_unc_ratio": [0.1, 0.3, None, 0.9, 0.2],
        }
    )


def test_usable_runs_drops_timeouts_and_empty_runs_but_keeps_budget_hits():
    assert usable_runs(_results()).height == 4


def test_compare_reports_reach_rates_and_medians_per_strategy():
    out = {r["strategy"]: r for r in compare_locators(_results()).iter_rows(named=True)}
    assert out["A"]["runs"] == 2
    assert out["A"]["focus_found"] == pytest.approx(1.0)
    assert out["A"]["max_info"] == pytest.approx(0.5)
    assert out["A"]["err_median"] == pytest.approx(200.0)
    assert out["A"]["cover_2sigma"] == pytest.approx(0.5)  # 100 <= 300 yes, 300 <= 200 no
    assert out["B"]["focus_found"] == pytest.approx(0.5)
    assert out["B"]["stage1_unc_ratio"] == pytest.approx(0.55)


def test_compare_tolerates_missing_stage_columns():
    df = _results().drop("stage1_steps", "stage1_unc_ratio", "focus_found_step", "max_info_step")
    assert compare_locators(df).height == 2


def test_sweep_strategies_are_named_by_points_per_fwhm():
    assert SWEEP_SPACING_DIVISORS == (1, 2, 3, 4, 5)
    names = [n for n, _ in CombinationGrid().strategies_for("NVCenter-lorentzian")]
    for k in SWEEP_SPACING_DIVISORS:
        assert sweep_strategy_name(k) in names
        assert is_plain_sweep(sweep_strategy_name(k))
    assert not is_plain_sweep("Smart-Sweep")
    assert not is_plain_sweep("Bayesian-SBED")


def test_standard_groups_run_every_sweep_spacing_and_both_smart_locators():
    for group in ("voigt-grid", "lorentzian-drift"):
        strategies = get_run_group(group).strategy_names
        assert {sweep_strategy_name(k) for k in SWEEP_SPACING_DIVISORS} <= set(strategies)
        assert {"Smart-Sweep", "Smart-SBED", "Bayesian-SBED"} <= set(strategies)
