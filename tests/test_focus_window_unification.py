"""Tests for the shared focus-window module (``nvision.belief.focus_window``).

What's actually unified, and what isn't
----------------------------------------
``clamp_to_domain`` is the one piece of arithmetic genuinely shared by every
narrowing site: the Bayesian locator's drive-frequency focus
(``nvision.belief.focus_window.next_focus_region``, via ``FocusRegion.from_intervals``),
and the sweep locators' ``nvision.sim.locs.refocus.window``.
``TestClampToDomainRouting`` proves
each of those call sites actually calls the shared function rather than a copy
of its arithmetic.

The Bayesian focus is a multi-section ``FocusRegion`` (tested in
``test_focus_policy.py``). The sweep locators' window updates go through
``FocusWindow.from_candidate``, which only clamps and rejects a collapsed window -- it does not know about a
width floor or a minimum-shrink-fraction. ``test_from_candidate_has_no_gating_parameters``
pins that boundary down so a future refactor can't quietly claim more sharing
than actually exists.

``TestIsFullDomainAgainstOracle`` fuzzes ``FocusWindow.is_full_domain`` against a
literal transcription of the pre-refactor inline check, asserting *exact* equality.
"""

from __future__ import annotations

import inspect

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from nvision.belief.focus_window import FocusWindow, clamp_to_domain

# ---------------------------------------------------------------------------
# FocusWindow construction and to_measure_drive_freq_unit
# ---------------------------------------------------------------------------


class TestFocusWindowConstruction:
    def test_valid_construction(self):
        w = FocusWindow(lo=1.0, hi=2.0, full_lo=0.0, full_hi=3.0)
        assert (w.lo, w.hi, w.full_lo, w.full_hi) == (1.0, 2.0, 0.0, 3.0)

    def test_rejects_hi_not_greater_than_lo(self):
        with pytest.raises(ValueError, match="requires lo < hi"):
            FocusWindow(lo=1.0, hi=1.0, full_lo=0.0, full_hi=2.0)
        with pytest.raises(ValueError, match="requires lo < hi"):
            FocusWindow(lo=2.0, hi=1.0, full_lo=0.0, full_hi=3.0)

    def test_rejects_full_hi_not_greater_than_full_lo(self):
        with pytest.raises(ValueError, match="requires full_lo < full_hi"):
            FocusWindow(lo=0.0, hi=1.0, full_lo=1.0, full_hi=1.0)

    def test_to_measure_x_uses_full_bounds_not_narrowed_bounds(self):
        """to_measure_drive_freq_unit must always normalize against full_lo/full_hi, even
        once the window has narrowed away from them -- that's the entire
        point of keeping full_lo/full_hi immutable.
        """
        w = FocusWindow(lo=0.4, hi=0.6, full_lo=0.0, full_hi=1.0)
        assert w.to_measure_drive_freq_unit(0.5) == pytest.approx(0.5)
        assert w.to_measure_drive_freq_unit(0.0) == pytest.approx(0.0)
        assert w.to_measure_drive_freq_unit(1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# clamp_to_domain
# ---------------------------------------------------------------------------


class TestClampToDomain:
    def test_already_inside_domain_is_unchanged(self):
        assert clamp_to_domain(1.0, 2.0, 0.0, 3.0) == (1.0, 2.0)

    def test_clips_both_sides(self):
        assert clamp_to_domain(-5.0, 10.0, 0.0, 3.0) == (0.0, 3.0)

    def test_clips_one_side(self):
        assert clamp_to_domain(-5.0, 2.0, 0.0, 3.0) == (0.0, 2.0)
        assert clamp_to_domain(1.0, 10.0, 0.0, 3.0) == (1.0, 3.0)

    def test_fully_outside_domain_collapses(self):
        lo, hi = clamp_to_domain(10.0, 20.0, 0.0, 3.0)
        assert lo >= hi

    def test_normalizes_swapped_order(self):
        """Defensive behavior added during extraction, not present in either
        pre-refactor call site verbatim -- both original call sites always
        passed already-ordered (lo, hi). Documented here rather than left
        implicit.
        """
        assert clamp_to_domain(2.0, 1.0, 0.0, 3.0) == (1.0, 2.0)


# ---------------------------------------------------------------------------
# FocusWindow.from_candidate / is_full_domain
# ---------------------------------------------------------------------------


class TestFromCandidate:
    def test_basic_narrowing(self):
        w = FocusWindow.from_candidate(1.0, 2.0, full_lo=0.0, full_hi=3.0)
        assert w is not None
        assert (w.lo, w.hi, w.full_lo, w.full_hi) == (1.0, 2.0, 0.0, 3.0)

    def test_collapsed_candidate_returns_none(self):
        assert FocusWindow.from_candidate(10.0, 20.0, full_lo=0.0, full_hi=3.0) is None

    def test_has_no_gating_parameters(self):
        """Pins the boundary of what's shared: from_candidate (used by the
        sweep locators) only clamps and rejects collapse. It has no min_width
        or min_narrowing_fraction knobs -- the Bayesian focus's minimum-shrink gate
        lives in next_focus_region, on a FocusRegion. If this test
        starts failing because someone added those parameters, the module
        docstring's claim about what's NOT unified needs updating too.
        """
        params = set(inspect.signature(FocusWindow.from_candidate).parameters)
        assert "min_width" not in params
        assert "min_narrowing_fraction" not in params


class TestIsFullDomain:
    def test_true_for_the_full_domain_itself(self):
        w = FocusWindow(lo=0.0, hi=1.0, full_lo=0.0, full_hi=1.0)
        assert w.is_full_domain()

    def test_false_once_narrowed(self):
        w = FocusWindow(lo=0.1, hi=0.9, full_lo=0.0, full_hi=1.0)
        assert not w.is_full_domain()

    def test_eps_tolerance_matches_boundary(self):
        # 1e-9 narrower than full domain: still "full" under the default eps.
        w = FocusWindow(lo=0.0, hi=1.0 - 1e-10, full_lo=0.0, full_hi=1.0)
        assert w.is_full_domain()


# ---------------------------------------------------------------------------
# Oracle tests: reference implementations transcribed from the pre-refactor
# inline code in SMCMarginalDistribution._resample.
# ---------------------------------------------------------------------------


def _oracle_is_full_domain(lo: float, hi: float, domain_lo: float, domain_hi: float, eps: float = 1e-9) -> bool:
    """Pre-refactor sobol_locator.py check: ``hi - lo < domain_width * (1 - eps)``."""
    domain_width = domain_hi - domain_lo
    return not ((hi - lo) < domain_width * (1.0 - eps))


# Bounded, finite floats -- the full-domain check doesn't need to handle inf/nan and
# neither did the pre-refactor code.
_finite_floats = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False)


class TestIsFullDomainAgainstOracle:
    @settings(deadline=None, max_examples=200)
    @given(
        full_lo=_finite_floats,
        full_width=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
        frac_lo=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        frac_hi=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    def test_matches_pre_refactor_check_exactly(self, full_lo, full_width, frac_lo, frac_hi):
        full_hi = full_lo + full_width
        lo = full_lo + frac_lo * full_width
        hi = full_lo + frac_hi * full_width
        assume(hi > lo)

        window = FocusWindow(lo=lo, hi=hi, full_lo=full_lo, full_hi=full_hi)
        assert window.is_full_domain() == _oracle_is_full_domain(lo, hi, full_lo, full_hi)


# ---------------------------------------------------------------------------
# Routing: every documented call site actually calls the shared function,
# not a local copy of its arithmetic.
# ---------------------------------------------------------------------------


class TestClampToDomainRouting:
    def test_next_focus_region_routes_through_clamp(self, monkeypatch):
        import nvision.belief.focus_window as fw_mod
        from nvision.belief.dip_detection import DipCandidate

        calls: list[tuple] = []
        real_clamp = fw_mod.clamp_to_domain

        def spy(lo, hi, domain_lo, domain_hi):
            calls.append((lo, hi, domain_lo, domain_hi))
            return real_clamp(lo, hi, domain_lo, domain_hi)

        monkeypatch.setattr(fw_mod, "clamp_to_domain", spy)

        focus = fw_mod.FocusRegion.full(0.0, 100.0)
        fw_mod.next_focus_region(
            focus,
            dips=[DipCandidate(50.0, 6.0, 6, 48.0, 52.0, 0.999, 1.0)],
            pad_phys=5.0,
            step=100,
            last_expansion_step=-1,
        )
        assert calls[0] == (43.0, 57.0, 0.0, 100.0)

    def test_refocus_window_infer_focus_window_routes_through_clamp(self, monkeypatch):
        import nvision.sim.locs.refocus.window as refocus_window_mod
        from nvision.models.observation import Observation, ObservationHistory

        calls: list[tuple] = []
        real_clamp = refocus_window_mod.clamp_to_domain

        def spy(lo, hi, domain_lo, domain_hi):
            calls.append((lo, hi, domain_lo, domain_hi))
            return real_clamp(lo, hi, domain_lo, domain_hi)

        monkeypatch.setattr(refocus_window_mod, "clamp_to_domain", spy)

        x = [i / 299.0 for i in range(300)]
        hist = ObservationHistory(500)
        for drive_freq_unit in x:
            y = 1.0 - 0.9 * pow(2.718281828, -0.5 * ((drive_freq_unit - 0.5) / 0.025) ** 2)
            hist.append(Observation(drive_freq_unit=drive_freq_unit, signal_value=y))

        refocus_window_mod.infer_focus_window(hist, 0.0, 1.0, expected_dips=1, noise_threshold=0.5)
        assert len(calls) >= 1
        assert all(c[2:] == (0.0, 1.0) for c in calls)
