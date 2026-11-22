"""Per-locator summary table and sweep strategy names."""

from __future__ import annotations

import polars as pl
import pytest

from nvision.metrics.compare import compare_locators, usable_runs
from nvision.sim.combinations import PLAIN_SWEEP_STRATEGIES, CombinationGrid
from nvision.sim.grid_enums import (
    OPTIMAL_SWEEP_STRATEGY,
    is_plain_sweep,
    optimal_sweep_multiplier,
    optimal_sweep_strategy_name,
    sweep_strategy_name,
)
from nvision.sim.run_groups import get_run_group


def _results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": ["A", "A", "A", "B", "B"],
            "measurements": [10, 20, 0, 30, 30],
            "failure_reason": [None, "max_steps", "timeout", None, None],
            "final_err_primary": [100.0, 300.0, 1.0, 50.0, 70.0],
            "uncert": [150.0, 100.0, 1.0, 40.0, 60.0],
            "detection_step": [5, 7, None, None, 12],
        }
    )


def test_usable_runs_drops_timeouts_and_empty_runs_but_keeps_budget_hits():
    assert usable_runs(_results()).height == 4


def test_compare_reports_reach_rates_and_medians_per_strategy():
    out = {r["strategy"]: r for r in compare_locators(_results()).iter_rows(named=True)}
    assert out["A"]["runs"] == 2
    assert out["A"]["detected"] == pytest.approx(1.0)
    assert out["A"]["detect_step"] == pytest.approx(6.0)
    assert out["A"]["err_median"] == pytest.approx(200.0)
    assert out["A"]["cover_2sigma"] == pytest.approx(0.5)  # 100 <= 300 yes, 300 <= 200 no
    assert out["B"]["detected"] == pytest.approx(0.5)


def test_compare_tolerates_a_missing_detection_column():
    df = _results().drop("detection_step")
    assert compare_locators(df).height == 2


def test_plain_sweeps_are_the_reference_and_the_optimal_step():
    assert list(PLAIN_SWEEP_STRATEGIES) == ["SimpleSweep-optimal-x0.5", "SimpleSweep-optimal", "SimpleSweep-optimal-x2"]
    assert OPTIMAL_SWEEP_STRATEGY in PLAIN_SWEEP_STRATEGIES
    assert [optimal_sweep_multiplier(s) for s in PLAIN_SWEEP_STRATEGIES] == [0.5, 1.0, 2.0]
    assert optimal_sweep_strategy_name(1) == OPTIMAL_SWEEP_STRATEGY
    assert optimal_sweep_multiplier("Bayesian-SBED") is None
    assert optimal_sweep_multiplier(sweep_strategy_name(3)) is None
    names = [n for n, _ in CombinationGrid().strategies_for("NVCenter-lorentzian")]
    for name in PLAIN_SWEEP_STRATEGIES:
        assert name in names
        assert is_plain_sweep(name)
    assert not is_plain_sweep("Bayesian-SBED")


def test_standard_groups_run_both_plain_sweeps_and_sbed():
    for group in ("voigt-grid",):
        strategies = get_run_group(group).strategy_names
        assert set(PLAIN_SWEEP_STRATEGIES) <= set(strategies)
        assert "Bayesian-SBED" in strategies


def test_sweep_strategy_name_accepts_fractional_points_per_fwhm():
    assert sweep_strategy_name(3) == sweep_strategy_name(3.0) == "SimpleSweep-3perFWHM"
    assert sweep_strategy_name(1.5) == "SimpleSweep-1.5perFWHM"
    assert is_plain_sweep(sweep_strategy_name(1.5))
