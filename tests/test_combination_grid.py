from __future__ import annotations
"""CombinationGrid exposes Bayesian strategies for multiple true signals."""


from nvision import CombinationGrid


def test_bayesian_strategies_include_other_true_signals():
    grid = CombinationGrid()
    combos = list(grid.iter(filter_strategy="Bayesian"))
    names = {c.generator_name for c in combos}
    assert "NVCenter-lorentzian" in names
    assert "NVCenter-voigt" in names
