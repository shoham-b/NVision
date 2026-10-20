"""The drive-frequency focus: locator-owned sections limiting which candidate x positions may be scanned.

Invariants checked here:

* ``FocusRegion`` keeps its sections sorted, disjoint, merged and inside the full axis.
* ``next_focus_region`` (pure policy) is a function of the significant dips only: it narrows to one
  padded section per dip (overlaps merged), expands when a dip section lies outside the focus, honours
  the narrowing delays and leaves the focus alone when there are no dips.
* The focus never reaches into the belief: its parameter bounds, drive-frequency axis, and the model's x-range
  are identical before and after the focus changes.
* Every EIG pick lies inside the focus; an empty focus fails loudly instead of falling back.
* The locator derives the focus from ``belief.dip_candidates`` -- also with the default fixed ``zero_field_freq``.
"""

from __future__ import annotations

import numpy as np
import pytest

import nvision.belief.focus_window as focus_mod
from nvision.belief.dip_detection import DipCandidate
from nvision.belief.focus_window import NVISION_SMC_FOCUSING_COVER_FACTOR, FocusRegion, next_focus_region
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import effective_hwhm
from tests.noise import gaussian_noise

FULL = (2.6e9, 3.1e9)
NARROW_STEP = 20  # comfortably past NVISION_MIN_STEPS_BEFORE_NARROWING
PAD = 10e6


def _focus(*sections: tuple[float, float]) -> FocusRegion:
    return FocusRegion.from_intervals(list(sections) or [FULL], full_lo=FULL[0], full_hi=FULL[1])


def _dip(f_min: float, f_max: float) -> DipCandidate:
    return DipCandidate(
        centroid_hz=0.5 * (f_min + f_max),
        significance=6.0,
        n_points=6,
        f_min=f_min,
        f_max=f_max,
        confidence=0.999,
        width_hz=(f_max - f_min),
    )


def _next(focus: FocusRegion, dips, *, step: int = NARROW_STEP, last_expansion_step: int = -1, pad: float = PAD):
    return next_focus_region(focus, dips=dips, pad_phys=pad, step=step, last_expansion_step=last_expansion_step)


def _approx_sections(*sections: tuple[float, float]):
    return pytest.approx([v for s in sections for v in s], abs=1.0)


def _flat(region: FocusRegion) -> list[float]:
    return [v for s in region.sections for v in s]


# ---------------------------------------------------------------------------
# FocusRegion: the value object
# ---------------------------------------------------------------------------


def test_from_intervals_sorts_and_merges_overlapping_and_touching_sections():
    region = FocusRegion.from_intervals([(5.0, 6.0), (1.0, 3.0), (2.0, 4.0), (4.0, 4.5)], full_lo=0.0, full_hi=10.0)
    assert region.sections == ((1.0, 4.5), (5.0, 6.0))
    assert region.width == pytest.approx(4.5)
    assert (region.lo, region.hi) == (1.0, 6.0)


def test_from_intervals_clamps_to_the_axis():
    region = FocusRegion.from_intervals([(-5.0, 2.0), (9.0, 15.0)], full_lo=0.0, full_hi=10.0)
    assert region.sections == ((0.0, 2.0), (9.0, 10.0))


def test_from_intervals_fails_fast_off_the_axis():
    with pytest.raises(ValueError, match="outside the full drive-frequency axis"):
        FocusRegion.from_intervals([(11.0, 12.0)], full_lo=0.0, full_hi=10.0)


@pytest.mark.parametrize(
    ("sections", "match"),
    [
        ((), "at least one section"),
        (((2.0, 1.0),), "lo < hi"),
        (((1.0, 3.0), (2.0, 4.0)), "sorted and disjoint"),
        (((4.0, 5.0), (1.0, 2.0)), "sorted and disjoint"),
        (((-1.0, 2.0),), "outside the full axis"),
    ],
)
def test_constructor_rejects_invalid_sections(sections, match):
    with pytest.raises(ValueError, match=match):
        FocusRegion(sections=sections, full_lo=0.0, full_hi=10.0)


def test_contains_excludes_the_gap_between_sections():
    region = FocusRegion.from_intervals([(1.0, 2.0), (5.0, 6.0)], full_lo=0.0, full_hi=10.0)
    xs = np.array([0.5, 1.0, 1.5, 2.0, 3.5, 5.0, 6.0, 7.0])
    assert region.contains(xs).tolist() == [False, True, True, True, False, True, True, False]


def test_covers_requires_each_section_inside_one_section():
    region = FocusRegion.from_intervals([(1.0, 2.0), (5.0, 6.0)], full_lo=0.0, full_hi=10.0)
    assert region.covers(FocusRegion.from_intervals([(1.2, 1.8), (5.5, 6.0)], full_lo=0.0, full_hi=10.0))
    # Inside the outer extent, but spans the gap.
    assert not region.covers(FocusRegion.from_intervals([(1.5, 5.5)], full_lo=0.0, full_hi=10.0))


def test_full_region_is_full_domain_and_a_narrowed_one_is_not():
    assert FocusRegion.full(0.0, 10.0).is_full_domain()
    assert not FocusRegion.from_intervals([(0.0, 4.0), (6.0, 10.0)], full_lo=0.0, full_hi=10.0).is_full_domain()


# ---------------------------------------------------------------------------
# next_focus_region: the pure policy
# ---------------------------------------------------------------------------


def test_narrows_to_padded_dip_extent():
    new, expanded = _next(_focus(), [_dip(2.85e9, 2.86e9)])
    assert not expanded
    assert _flat(new) == _approx_sections((2.85e9 - PAD, 2.86e9 + PAD))
    assert (new.full_lo, new.full_hi) == FULL


def test_disjoint_dips_give_one_section_each():
    new, _ = _next(_focus(), [_dip(2.80e9, 2.81e9), _dip(2.90e9, 2.91e9)])
    assert _flat(new) == _approx_sections((2.80e9 - PAD, 2.81e9 + PAD), (2.90e9 - PAD, 2.91e9 + PAD))
    # The empty axis between the dips is not scanned.
    assert not new.contains(np.array([2.85e9]))[0]


def test_dips_whose_padded_sections_overlap_merge_into_one_section():
    new, _ = _next(_focus(), [_dip(2.850e9, 2.851e9), _dip(2.860e9, 2.861e9)])  # gap 9 MHz < 2 * PAD
    assert _flat(new) == _approx_sections((2.850e9 - PAD, 2.861e9 + PAD))


def test_a_dip_in_the_gap_between_sections_expands_the_focus():
    focus = _focus((2.79e9, 2.82e9), (2.89e9, 2.92e9))
    new, expanded = _next(focus, [_dip(2.85e9, 2.851e9)], step=0)
    assert expanded
    assert _flat(new) == _approx_sections((2.79e9, 2.82e9), (2.85e9 - PAD, 2.851e9 + PAD), (2.89e9, 2.92e9))


def test_a_spurious_section_dropping_out_narrows_the_focus_to_the_rest():
    focus = _focus((2.79e9, 2.82e9), (2.89e9, 2.92e9))
    new, expanded = _next(focus, [_dip(2.80e9, 2.81e9)])
    assert not expanded
    assert _flat(new) == _approx_sections((2.80e9 - PAD, 2.81e9 + PAD))


def test_no_dips_leaves_the_focus_unchanged():
    focus = _focus()
    new, expanded = _next(focus, [])
    assert new is focus
    assert not expanded


def test_narrowing_waits_for_min_steps():
    focus = _focus()
    new, expanded = _next(focus, [_dip(2.85e9, 2.86e9)], step=0)
    assert new is focus
    assert not expanded


def test_narrowing_is_delayed_after_a_recent_expansion():
    focus = _focus()
    new, expanded = _next(focus, [_dip(2.85e9, 2.86e9)], last_expansion_step=NARROW_STEP - 1)
    assert new is focus
    assert not expanded


def test_dip_outside_the_focus_expands_it_even_before_the_narrowing_delay():
    focus = _focus((2.80e9, 2.90e9))
    new, expanded = _next(focus, [_dip(2.84e9, 2.85e9), _dip(2.95e9, 2.96e9)], step=0)
    assert expanded
    # Union: the old section is kept, the new dip adds its own section.
    assert _flat(new) == _approx_sections((2.80e9, 2.90e9), (2.95e9 - PAD, 2.96e9 + PAD))


def test_expansion_never_leaves_the_full_probe_axis():
    new, expanded = _next(_focus((2.7e9, 3.0e9)), [_dip(FULL[0] + 1e6, FULL[0] + 2e6)], step=0)
    assert expanded
    assert new.lo == FULL[0]


def test_dip_inside_a_focus_that_already_covers_it_does_not_expand():
    focus = _focus((2.80e9, 2.90e9))
    new, expanded = _next(focus, [_dip(2.84e9, 2.85e9)], step=0)
    assert new is focus
    assert not expanded


def test_negligible_shrink_is_rejected():
    focus = _focus((2.85e9 - PAD, 2.86e9 + PAD + 1e6))  # target is <5% narrower
    new, expanded = _next(focus, [_dip(2.85e9, 2.86e9)])
    assert new is focus
    assert not expanded


def test_dips_outside_the_axis_fail_fast():
    with pytest.raises(ValueError, match="outside the full drive-frequency axis"):
        _next(_focus(), [_dip(3.5e9, 3.6e9)])


@pytest.mark.parametrize("pad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_pad_fails_fast(pad):
    with pytest.raises(ValueError, match="pad_phys"):
        _next(_focus(), [_dip(2.85e9, 2.86e9)], pad=pad)


def test_narrowing_routes_through_the_shared_clamp(monkeypatch):
    calls: list[tuple] = []
    real = focus_mod.clamp_to_domain

    def spy(lo, hi, domain_lo, domain_hi):
        calls.append((domain_lo, domain_hi))
        return real(lo, hi, domain_lo, domain_hi)

    monkeypatch.setattr(focus_mod, "clamp_to_domain", spy)
    _next(_focus(), [_dip(2.85e9, 2.86e9)])
    assert calls
    assert all(c == FULL for c in calls)


# ---------------------------------------------------------------------------
# Locator-owned focus never touches the belief
# ---------------------------------------------------------------------------


def _locator(**belief_kwargs):
    belief = nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=500, seed=0, **belief_kwargs)
    return SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=100)


def _belief_geometry(belief):
    return (
        dict(belief.physical_param_bounds),
        belief.drive_freq_bounds_phys,
        dict(belief.model.param_bounds_phys),
        belief.model.x_bounds_phys,
    )


def _set_dips(loc, *extents: tuple[float, float]) -> None:
    loc.belief._dip_candidates = [_dip(*e) for e in extents]
    loc.belief._step_count = NARROW_STEP


def _mid_axis_dip(loc, half_width: float = 2e6) -> tuple[float, float]:
    lo, hi = loc.belief.drive_freq_bounds_phys
    mid = 0.5 * (lo + hi)
    return mid - half_width, mid + half_width


def test_default_fixed_center_locator_narrows_to_the_dips():
    loc = _locator()
    assert "zero_field_freq" not in loc.belief.model.parameter_names()
    lo, hi = loc.belief.drive_freq_bounds_phys
    assert loc.bayesian_focus_window() is None
    dip = _mid_axis_dip(loc)
    _set_dips(loc, dip)

    loc._update_focus_window()

    pad = NVISION_SMC_FOCUSING_COVER_FACTOR * float(effective_hwhm(loc.belief.estimates()))
    assert _flat(loc._focus) == _approx_sections((dip[0] - pad, dip[1] + pad))
    assert loc._focus.width < hi - lo
    # The UI reports exactly the sections that restrict the candidates.
    assert loc.focus_window_candidates() == list(loc._focus.sections)
    assert loc.bayesian_focus_window() == (loc._focus.lo, loc._focus.hi)


def test_no_dips_keeps_the_full_probe_axis_focus():
    loc = _locator()
    full = loc._focus
    loc.belief._step_count = NARROW_STEP
    loc._update_focus_window()
    assert loc._focus is full
    assert loc._focus.is_full_domain()


def test_focus_change_leaves_belief_geometry_untouched():
    loc = _locator(with_zeeman_splitting=True, hyperfine="unresolved")
    belief = loc.belief
    before = _belief_geometry(belief)
    particles_before = {k: np.array(v) for k, v in belief.particles_phys().items()}
    _set_dips(loc, _mid_axis_dip(loc))

    loc._update_focus_window()

    full_lo, full_hi = belief.drive_freq_bounds_phys
    assert loc._focus.width < full_hi - full_lo
    assert _belief_geometry(belief) == before
    for name, values in belief.particles_phys().items():
        assert np.array_equal(values, particles_before[name])


def test_eig_pick_lies_inside_the_focus():
    loc = _locator(with_zeeman_splitting=True, hyperfine="unresolved")
    belief = loc.belief
    # A dip sitting on actual epoch candidates, so some candidate lies inside the narrowed focus.
    cands = np.sort(belief.get_candidate_drive_freq_phys())
    centre = float(cands[len(cands) // 2])
    _set_dips(loc, (centre - 1e6, centre + 1e6))
    loc._update_focus_window()
    full_lo, full_hi = belief.drive_freq_bounds_phys
    assert loc._focus.width < full_hi - full_lo

    for _ in range(10):
        assert loc._focus.contains(np.array([loc._eig_acquire()]))[0]


def test_eig_never_picks_in_the_gap_between_two_sections():
    loc = _locator(with_zeeman_splitting=True, hyperfine="unresolved")
    cands = np.sort(loc.belief.get_candidate_drive_freq_phys())
    left, right = float(cands[len(cands) // 4]), float(cands[3 * len(cands) // 4])
    full = loc._focus
    loc._focus = FocusRegion.from_intervals(
        [(left - 1e5, left + 1e5), (right - 1e5, right + 1e5)], full_lo=full.full_lo, full_hi=full.full_hi
    )
    assert len(loc._focus.sections) == 2

    for _ in range(10):
        x = loc._eig_acquire()
        assert loc._focus.contains(np.array([x]))[0]
        assert not (left + 1e5 < x < right - 1e5)


def test_empty_focus_fails_loudly():
    loc = _locator(with_zeeman_splitting=True, hyperfine="unresolved")
    cands = np.sort(loc.belief.get_candidate_drive_freq_phys())
    i = int(np.argmax(np.diff(cands)))
    mid = 0.5 * (cands[i] + cands[i + 1])
    full = loc._focus
    loc._focus = FocusRegion.from_intervals([(mid - 0.1, mid + 0.1)], full_lo=full.full_lo, full_hi=full.full_hi)
    with pytest.raises(ValueError, match="No epoch candidate lies inside the focus"):
        loc._eig_acquire()
