"""The drive-frequency focus: a locator-owned window limiting which candidate x positions may be scanned.

Invariants checked here:

* ``next_focus_window`` (pure policy) is a function of the significant dips only: it narrows to their
  padded bounding interval, expands when a dip lies outside the focus, honours the narrowing delays and
  leaves the focus alone when there are no dips.
* The focus never reaches into the belief: its parameter bounds, drive-frequency axis, and the model's x-range
  are identical before and after the focus changes.
* Every EIG pick lies inside the focus; an empty focus fails loudly instead of falling back.
* The locator derives the focus from ``belief.dip_candidates`` -- also with the default fixed ``center_freq``.
"""

from __future__ import annotations

import numpy as np
import pytest

import nvision.belief.focus_window as focus_mod
from nvision.belief.dip_detection import DipCandidate
from nvision.belief.focus_window import NVISION_SMC_FOCUSING_COVER_FACTOR, FocusWindow, next_focus_window
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import effective_hwhm
from tests.noise import gaussian_noise

FULL = (2.6e9, 3.1e9)
NARROW_STEP = 20  # comfortably past NVISION_MIN_STEPS_BEFORE_NARROWING
PAD = 10e6


def _focus(lo: float = FULL[0], hi: float = FULL[1]) -> FocusWindow:
    return FocusWindow(lo=lo, hi=hi, full_lo=FULL[0], full_hi=FULL[1])


def _dip(f_min: float, f_max: float) -> DipCandidate:
    return DipCandidate(
        centroid_hz=0.5 * (f_min + f_max),
        significance=6.0,
        n_points=6,
        f_min=f_min,
        f_max=f_max,
        confidence=0.999,
        background=1.0,
    )


def _next(focus: FocusWindow, dips, *, step: int = NARROW_STEP, last_expansion_step: int = -1, pad: float = PAD):
    return next_focus_window(focus, dips=dips, pad_phys=pad, step=step, last_expansion_step=last_expansion_step)


# ---------------------------------------------------------------------------
# next_focus_window: the pure policy
# ---------------------------------------------------------------------------


def test_narrows_to_padded_dip_extent():
    new, expanded = _next(_focus(), [_dip(2.85e9, 2.86e9)])
    assert not expanded
    assert new.lo == pytest.approx(2.85e9 - PAD, abs=1.0)
    assert new.hi == pytest.approx(2.86e9 + PAD, abs=1.0)
    assert (new.full_lo, new.full_hi) == FULL


def test_disjoint_dips_give_one_bounding_window():
    new, _ = _next(_focus(), [_dip(2.80e9, 2.81e9), _dip(2.90e9, 2.91e9)])
    assert new.lo == pytest.approx(2.80e9 - PAD, abs=1.0)
    assert new.hi == pytest.approx(2.91e9 + PAD, abs=1.0)


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
    focus = _focus(2.80e9, 2.90e9)
    new, expanded = _next(focus, [_dip(2.84e9, 2.85e9), _dip(2.95e9, 2.96e9)], step=0)
    assert expanded
    assert new.lo == focus.lo  # union: the old edge is kept
    assert new.hi == pytest.approx(2.96e9 + PAD, abs=1.0)


def test_expansion_never_leaves_the_full_probe_axis():
    new, expanded = _next(_focus(2.7e9, 3.0e9), [_dip(FULL[0] + 1e6, FULL[0] + 2e6)], step=0)
    assert expanded
    assert new.lo == FULL[0]


def test_dip_inside_a_focus_that_already_covers_it_does_not_expand():
    focus = _focus(2.80e9, 2.90e9)
    new, expanded = _next(focus, [_dip(2.84e9, 2.85e9)], step=0)
    assert new is focus
    assert not expanded


def test_negligible_shrink_is_rejected():
    focus = _focus(2.85e9 - PAD, 2.86e9 + PAD + 1e6)  # target is <5% narrower
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
    assert "center_freq" not in loc.belief.model.parameter_names()
    lo, hi = loc.belief.drive_freq_bounds_phys
    assert loc.bayesian_focus_window() is None
    dip = _mid_axis_dip(loc)
    _set_dips(loc, dip)

    loc._update_focus_window()

    pad = NVISION_SMC_FOCUSING_COVER_FACTOR * float(effective_hwhm(loc.belief.estimates()))
    focus_lo, focus_hi = loc._acquisition_bounds()
    assert (focus_lo, focus_hi) == pytest.approx((dip[0] - pad, dip[1] + pad), abs=1.0)
    assert focus_hi - focus_lo < hi - lo
    # The UI reports exactly the window that restricts the candidates.
    assert loc.bayesian_focus_window() == (focus_lo, focus_hi)


def test_no_dips_keeps_the_full_probe_axis_focus():
    loc = _locator()
    full = loc._acquisition_bounds()
    loc.belief._step_count = NARROW_STEP
    loc._update_focus_window()
    assert loc._acquisition_bounds() == full


def test_focus_change_leaves_belief_geometry_untouched():
    loc = _locator(with_zeeman_splitting=True, hyperfine="unresolved")
    belief = loc.belief
    before = _belief_geometry(belief)
    particles_before = {k: np.array(v) for k, v in belief.particles_phys().items()}
    _set_dips(loc, _mid_axis_dip(loc))

    loc._update_focus_window()

    full_lo, full_hi = belief.drive_freq_bounds_phys
    assert loc._focus.hi - loc._focus.lo < full_hi - full_lo
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
    lo, hi = loc._acquisition_bounds()
    full_lo, full_hi = belief.drive_freq_bounds_phys
    assert hi - lo < full_hi - full_lo

    for _ in range(10):
        assert lo <= loc._eig_acquire() <= hi


def test_empty_focus_fails_loudly():
    loc = _locator(with_zeeman_splitting=True, hyperfine="unresolved")
    cands = np.sort(loc.belief.get_candidate_drive_freq_phys())
    i = int(np.argmax(np.diff(cands)))
    mid = 0.5 * (cands[i] + cands[i + 1])
    full = loc._focus
    loc._focus = FocusWindow(lo=mid - 0.1, hi=mid + 0.1, full_lo=full.full_lo, full_hi=full.full_hi)
    with pytest.raises(ValueError, match="No epoch candidate lies inside the focus"):
        loc._eig_acquire()
