"""CombinationGrid exposes Bayesian strategies for multiple true signals."""

from __future__ import annotations

from nvision import CombinationGrid


def test_bayesian_strategies_include_other_true_signals():
    grid = CombinationGrid()
    combos = list(grid.iter(filter_strategy="Bayesian"))
    names = {c.generator_name for c in combos}
    assert "NVCenter-one_peak" in names
    assert "OnePeak-gaussian" in names
    assert "OnePeak-lorentzian" in names
    assert "TwoPeak-gaussian" in names
    assert "TwoPeak-lorentzian" in names
