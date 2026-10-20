"""Tests for the shared focus-window module (``nvision.belief.focus_window``).

What's actually unified, and what isn't
----------------------------------------
``clamp_to_domain`` is the one piece of arithmetic genuinely shared by every
narrowing site: the Bayesian locator's drive-frequency focus
(``nvision.belief.focus_window.next_focus_window``), the sweep locators'
``nvision.sim.locs.refocus.window``, and ``StagedSobolSweepLocator.per_dip_windows``.
``TestClampToDomainRouting`` proves each of those call sites actually calls the
shared function rather than a copy of its arithmetic.

``FocusWindow.propose_narrowing`` (the floor + clamp + minimum-shrink-fraction
decision) has exactly **one** caller: ``next_focus_window``.
The sweep locators' ``StagedSobolSweepLocator``/``Stage3SobolLocator`` window
updates go through ``FocusWindow.from_candidate`` instead, which only clamps
and rejects a collapsed window -- it does not know about a width floor or a
minimum-shrink-fraction. ``test_from_candidate_has_no_gating_parameters``
pins that boundary down so a future refactor can't quietly claim more sharing
than actually exists.

``TestProposeNarrowingAgainstOracle`` and ``TestIsFullDomainAgainstOracle``
fuzz the shared functions against reference implementations that are literal
transcriptions of the pre-refactor inline code in
``SMCMarginalDistribution._resample`` (see git history), asserting
*exact* equality -- the refactor was meant to move code, not change any
formula.
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
        or min_narrowing_fraction knobs -- those live only on
        propose_narrowing, whose only caller is next_focus_window. If this test
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


def _oracle_propose_narrowing(
    lo_phys: float,
    hi_phys: float,
    lo_orig: float,
    hi_orig: float,
    candidate_lo: float,
    candidate_hi: float,
    min_width: float | None,
    min_narrowing_fraction: float,
) -> tuple[float, float] | None:
    """Literal transcription of the old inlined gate in ``_resample``:

    floor to min_width -> clamp to [lo_orig, hi_orig] -> reject on collapse
    -> reject if shrink fraction is under the minimum.
    """
    new_lo, new_hi = candidate_lo, candidate_hi
    if min_width is not None and (new_hi - new_lo) < min_width:
        center = 0.5 * (new_hi + new_lo)
        new_lo = center - min_width / 2.0
        new_hi = center + min_width / 2.0

    new_lo = max(new_lo, lo_orig)
    new_hi = min(new_hi, hi_orig)
    if new_hi <= new_lo:
        return None

    cur_width = hi_phys - lo_phys
    shrink_frac = (cur_width - (new_hi - new_lo)) / cur_width if cur_width > 0.0 else None
    if min_narrowing_fraction > 0.0 and shrink_frac is not None and shrink_frac < min_narrowing_fraction:
        return None

    return (new_lo, new_hi)


# Bounded, finite floats -- narrowing math doesn't need to handle inf/nan and
# neither did the pre-refactor code.
_finite_floats = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False)


@st.composite
def _narrowing_scenarios(draw):
    full_lo = draw(_finite_floats)
    full_width = draw(st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False))
    full_hi = full_lo + full_width

    # Current window: a sub-interval of the domain (as FocusWindow requires).
    lo = draw(st.floats(min_value=full_lo, max_value=full_hi, allow_nan=False, allow_infinity=False))
    hi = draw(st.floats(min_value=lo, max_value=full_hi, allow_nan=False, allow_infinity=False))
    assume(hi > lo)

    # Candidate: allowed to straddle or fall entirely outside the domain.
    span = draw(st.floats(min_value=-2e6, max_value=2e6, allow_nan=False, allow_infinity=False))
    candidate_lo = draw(st.floats(min_value=full_lo - full_width, max_value=full_hi + full_width))
    candidate_hi = candidate_lo + abs(span)

    min_width = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=full_width * 3, allow_nan=False)))
    min_narrowing_fraction = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))

    return full_lo, full_hi, lo, hi, candidate_lo, candidate_hi, min_width, min_narrowing_fraction


class TestProposeNarrowingAgainstOracle:
    @settings(deadline=None, max_examples=300)
    @given(_narrowing_scenarios())
    def test_matches_pre_refactor_gate_exactly(self, scenario):
        full_lo, full_hi, lo, hi, candidate_lo, candidate_hi, min_width, min_narrowing_fraction = scenario

        window = FocusWindow(lo=lo, hi=hi, full_lo=full_lo, full_hi=full_hi)
        actual = window.propose_narrowing(
            candidate_lo, candidate_hi, min_width=min_width, min_narrowing_fraction=min_narrowing_fraction
        )
        expected = _oracle_propose_narrowing(
            lo, hi, full_lo, full_hi, candidate_lo, candidate_hi, min_width, min_narrowing_fraction
        )

        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert (actual.lo, actual.hi) == expected

    def test_rejects_insufficient_shrink(self):
        window = FocusWindow(lo=0.0, hi=100.0, full_lo=-1e6, full_hi=1e6)
        # Shrinks by exactly 4% when 5% is required.
        result = window.propose_narrowing(2.0, 98.0, min_narrowing_fraction=0.05)
        assert result is None

    def test_accepts_exactly_at_the_shrink_boundary(self):
        window = FocusWindow(lo=0.0, hi=100.0, full_lo=-1e6, full_hi=1e6)
        # Shrinks by exactly 5% when 5% is required -- boundary is inclusive
        # (oracle's reject condition is strict '<').
        result = window.propose_narrowing(2.5, 97.5, min_narrowing_fraction=0.05)
        assert result is not None
        assert (result.lo, result.hi) == (2.5, 97.5)

    def test_min_width_floor_recenters_narrow_candidate(self):
        window = FocusWindow(lo=0.0, hi=100.0, full_lo=-1e6, full_hi=1e6)
        # Candidate [49, 51] (width 2) floored to width 10, centered on 50.
        result = window.propose_narrowing(49.0, 51.0, min_width=10.0, min_narrowing_fraction=0.0)
        assert result is not None
        assert (result.lo, result.hi) == (45.0, 55.0)

    def test_min_width_larger_than_domain_clamps_to_domain(self):
        window = FocusWindow(lo=40.0, hi=60.0, full_lo=0.0, full_hi=100.0)
        result = window.propose_narrowing(49.0, 51.0, min_width=1000.0, min_narrowing_fraction=0.0)
        assert result is not None
        assert (result.lo, result.hi) == (0.0, 100.0)

    def test_candidate_entirely_outside_domain_is_rejected(self):
        window = FocusWindow(lo=0.0, hi=10.0, full_lo=0.0, full_hi=10.0)
        assert window.propose_narrowing(1000.0, 2000.0) is None

    def test_candidate_wider_than_current_window_rejected_by_fraction_gate(self):
        """A candidate that would *widen* the window has a negative shrink
        fraction, so any positive min_narrowing_fraction rejects it --
        propose_narrowing is a narrowing-only decision, unlike the sweep
        path's from_candidate which happily accepts a wider window.
        """
        window = FocusWindow(lo=40.0, hi=60.0, full_lo=0.0, full_hi=100.0)
        assert window.propose_narrowing(0.0, 100.0, min_narrowing_fraction=0.05) is None
        # With no fraction gate it's accepted, matching from_candidate's semantics.
        result = window.propose_narrowing(0.0, 100.0, min_narrowing_fraction=0.0)
        assert result is not None
        assert (result.lo, result.hi) == (0.0, 100.0)


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
    def test_next_focus_window_routes_through_clamp(self, monkeypatch):
        import numpy as np

        import nvision.belief.focus_window as fw_mod

        calls: list[tuple] = []
        real_clamp = fw_mod.clamp_to_domain

        def spy(lo, hi, domain_lo, domain_hi):
            calls.append((lo, hi, domain_lo, domain_hi))
            return real_clamp(lo, hi, domain_lo, domain_hi)

        monkeypatch.setattr(fw_mod, "clamp_to_domain", spy)

        focus = FocusWindow(lo=0.0, hi=100.0, full_lo=0.0, full_hi=100.0)
        fw_mod.next_focus_window(
            focus,
            center_freq_particles_phys=np.full(50, 50.0),
            active_half_span_particles_phys=np.full(50, 5.0),
            omega_phys=1.0,
            zeeman_split_phys=0.0,
            step=100,
            last_expansion_step=-1,
        )
        assert calls == [(45.0, 55.0, 0.0, 100.0)]

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

    def test_staged_sobol_per_dip_windows_routes_through_clamp(self, monkeypatch):
        import nvision.sim.locs.coarse.sobol_locator as sobol_mod

        calls: list[tuple] = []
        real_clamp = sobol_mod.clamp_to_domain

        def spy(lo, hi, domain_lo, domain_hi):
            calls.append((lo, hi, domain_lo, domain_hi))
            return real_clamp(lo, hi, domain_lo, domain_hi)

        monkeypatch.setattr(sobol_mod, "clamp_to_domain", spy)

        locator = sobol_mod.StagedSobolSweepLocator.__new__(sobol_mod.StagedSobolSweepLocator)
        locator.domain_lo = 0.0
        locator.domain_hi = 1.0
        locator.noise_std = 0.01
        from nvision.models.observation import ObservationHistory

        hist = ObservationHistory(500, (0.0, 1.0))
        import math

        for i in range(200):
            drive_freq_unit = i / 199.0
            y = 1.0
            y -= 0.8 * math.exp(-0.5 * ((drive_freq_unit - 0.3) / 0.02) ** 2)
            y -= 0.8 * math.exp(-0.5 * ((drive_freq_unit - 0.7) / 0.02) ** 2)
            hist.append(sobol_mod.Observation(drive_freq_unit=drive_freq_unit, signal_value=y))
        locator.history = hist

        windows = locator.per_dip_windows()
        assert windows is not None
        assert len(windows) >= 2
        assert len(calls) == len(windows)


# ---------------------------------------------------------------------------
# Fail-fast: a collapsed candidate from _infer_tight_focus_window must raise,
# not silently keep the previous window (AGENTS.md "No Algorithmic Fallbacks").
# ---------------------------------------------------------------------------


class TestSobolFailsFastOnCollapsedCandidate:
    def _make_stage3(self, monkeypatch, collapsed_tuple):
        import nvision.sim.locs.coarse.sobol_locator as sobol_mod

        monkeypatch.setattr(sobol_mod, "_infer_tight_focus_window", lambda *a, **k: collapsed_tuple)

        def vdc():
            while True:
                yield 0.5

        from nvision.models.observation import ObservationHistory

        return sobol_mod.Stage3SobolLocator(
            vdc(),
            0.0,
            1.0,
            ObservationHistory(50),
            expected_dips=1,
            noise_std=0.01,
        )

    def test_infer_bounds_raises_on_collapsed_candidate(self, monkeypatch):
        with pytest.raises(ValueError, match="collapsed window"):
            self._make_stage3(monkeypatch, (0.5, 0.5))

    def test_check_for_remaining_dips_raises_on_collapsed_candidate(self, monkeypatch):
        import nvision.sim.locs.coarse.sobol_locator as sobol_mod

        stage3 = self._make_stage3(monkeypatch, (0.0, 1.0))
        # Shrink the tracked window first so a later "wider but out-of-domain"
        # candidate from re-inference is what triggers the expand branch.
        stage3.window = sobol_mod.FocusWindow(lo=0.4, hi=0.6, full_lo=0.0, full_hi=1.0)
        # Width 1.0 > current window's 0.2, so the expand branch fires; but the
        # candidate lies entirely outside [0, 1], so clamping collapses it.
        monkeypatch.setattr(sobol_mod, "_infer_tight_focus_window", lambda *a, **k: (2.0, 3.0))

        for i in range(10):
            stage3.history.append(sobol_mod.Observation(drive_freq_unit=i / 9.0, signal_value=0.5))

        with pytest.raises(ValueError, match="collapsed window"):
            stage3._check_for_remaining_dips()
