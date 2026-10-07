"""BisectionFocusLocator acquires in bisection order inside the focus window; no EIG, no exploration."""

from __future__ import annotations

import pytest

from nvision.belief.focus_window import FocusWindow
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.bisect_locator import BisectionFocusLocator
from tests.noise import gaussian_noise


def _locator() -> BisectionFocusLocator:
    belief = nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=200, seed=0)
    return BisectionFocusLocator(belief=belief, max_steps=50)


def test_acquisition_bisects_the_focus_window_and_never_calls_eig():
    loc = _locator()
    loc._eig_acquire = lambda: pytest.fail("bisection must not evaluate EIG")
    lo, hi = loc._acquisition_bounds()
    width = hi - lo
    got = [loc._acquire() for _ in range(7)]
    expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    assert got == pytest.approx([lo + u * width for u in expected])


def test_bisection_restarts_when_the_focus_window_changes():
    loc = _locator()
    full_lo, full_hi = loc._full_domain_lo, loc._full_domain_hi
    for _ in range(3):
        loc._acquire()
    lo, hi = full_lo + 0.4 * (full_hi - full_lo), full_lo + 0.6 * (full_hi - full_lo)
    loc._focus = FocusWindow(lo=lo, hi=hi, full_lo=full_lo, full_hi=full_hi)
    assert loc._acquire() == pytest.approx(0.5 * (lo + hi))
    assert loc._acquire() == pytest.approx(lo + 0.25 * (hi - lo))


def test_next_stays_inside_the_unit_interval():
    loc = _locator()
    assert all(0.0 <= loc.next() <= 1.0 for _ in range(10))
