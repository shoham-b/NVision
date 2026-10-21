"""GenericSweepLocator repeats its grid until the Bayesian stopping rules fire (at a pass boundary)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nvision.models.observation import Observation
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator

_N_GRID = 5


class _FakeBelief:
    drive_freq_bounds_phys = (0.0, 1.0)

    def __init__(self):
        self.model = SimpleNamespace(parameter_names=lambda: ["zeeman_shift"])
        self.last_obs = None

    def update(self, obs) -> None:
        pass


def _signal_model():
    return SimpleNamespace(
        spec=SimpleNamespace(fixed_values={}), parameter_names=lambda: ["zeeman_shift"], inner=SimpleNamespace()
    )


def _locator(passes: int, converged_at_step: int | None) -> GenericSweepLocator:
    """A sweep whose stopping rules (`_update_stop_rules`) report convergence from ``converged_at_step`` on."""
    loc = GenericSweepLocator(
        belief=_FakeBelief(), signal_model=_signal_model(), max_steps=_N_GRID, spacing_divisor=None, max_passes=passes
    )

    def fake_rules() -> None:
        if converged_at_step is not None and loc.step_count >= converged_at_step:
            loc._is_converged = True

    loc._update_stop_rules = fake_rules
    return loc


def _drive(loc: GenericSweepLocator) -> list[float]:
    xs = []
    while not loc.done():
        x = loc.next()
        xs.append(x)
        loc.observe(Observation(drive_freq_unit=x, signal_value=1.0))
    return xs


def test_stops_at_the_pass_boundary_after_the_rules_fire():
    # The rules fire mid-way through the second pass; the sweep finishes that pass.
    loc = _locator(passes=20, converged_at_step=_N_GRID + 2)
    xs = _drive(loc)
    assert len(xs) == 2 * _N_GRID
    assert xs[:_N_GRID] == xs[_N_GRID:]
    assert loc.passes_done == 2
    assert loc.history.count == 2 * _N_GRID


def test_stops_at_max_passes_when_the_rules_never_fire():
    assert len(_drive(_locator(passes=3, converged_at_step=None))) == 3 * _N_GRID


def test_rules_firing_in_the_first_pass_give_a_single_pass():
    assert len(_drive(_locator(passes=20, converged_at_step=1))) == _N_GRID


def test_requires_at_least_one_pass():
    with pytest.raises(ValueError, match="max_passes"):
        GenericSweepLocator(
            belief=_FakeBelief(),
            signal_model=_signal_model(),
            max_steps=_N_GRID,
            spacing_divisor=None,
            max_passes=0,
        )
