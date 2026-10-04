"""The probe-axis focus: a locator-owned window limiting which candidate x positions may be scanned.

Invariants checked here:

* ``next_focus_window`` (pure policy) narrows around the posterior, expands on boundary pile-up, and
  honours the narrowing delays.
* The focus never reaches into the belief: its parameter bounds, probe axis, and the model's x-range
  are identical before and after the focus changes.
* Every EIG pick lies inside the focus; an empty focus fails loudly instead of falling back.
* With the default fixed ``center_freq`` the focus stays the full probe axis.
"""

from __future__ import annotations

import numpy as np
import pytest

import nvision.belief.focus_window as focus_mod
from nvision.belief.focus_window import FocusWindow, next_focus_window
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from tests.noise import gaussian_noise

FULL = (2.6e9, 3.1e9)
NARROW_STEP = 20  # comfortably past NVISION_MIN_STEPS_BEFORE_NARROWING


def _focus(lo: float = FULL[0], hi: float = FULL[1]) -> FocusWindow:
    return FocusWindow(lo=lo, hi=hi, full_lo=FULL[0], full_hi=FULL[1])


def _concentrated(center: float, span: float, n: int = 200) -> dict:
    return dict(
        center_freq_particles_phys=np.full(n, center),
        active_half_span_particles_phys=np.full(n, span),
        omega_phys=5e6,
        zeeman_split_phys=0.0,
    )


# ---------------------------------------------------------------------------
# next_focus_window: the pure policy
# ---------------------------------------------------------------------------


def test_narrows_to_exact_active_range_union():
    """Identical particles -> the focus is exactly center +/- half-span."""
    center, span = 2.85e9, 12e6
    new, expanded = next_focus_window(_focus(), **_concentrated(center, span), step=NARROW_STEP, last_expansion_step=-1)
    assert not expanded
    assert new.lo == pytest.approx(center - span, abs=1.0)
    assert new.hi == pytest.approx(center + span, abs=1.0)
    assert (new.full_lo, new.full_hi) == FULL


def test_narrowing_waits_for_min_steps():
    focus = _focus()
    new, expanded = next_focus_window(focus, **_concentrated(2.85e9, 12e6), step=0, last_expansion_step=-1)
    assert new is focus
    assert not expanded


def test_left_boundary_pile_up_expands_left_even_before_narrowing_delay():
    focus = _focus(2.8e9, 3.0e9)
    new, expanded = next_focus_window(focus, **_concentrated(2.79e9, 5e6), step=0, last_expansion_step=-1)
    assert expanded
    assert new.lo < focus.lo
    assert new.hi == focus.hi


def test_right_boundary_pile_up_expands_right():
    focus = _focus(2.7e9, 2.9e9)
    new, expanded = next_focus_window(focus, **_concentrated(2.91e9, 5e6), step=0, last_expansion_step=-1)
    assert expanded
    assert new.hi > focus.hi
    assert new.lo == focus.lo


def test_expansion_never_leaves_the_full_probe_axis():
    focus = _focus(2.7e9, 3.0e9)
    new, _ = next_focus_window(focus, **_concentrated(2.0e9, 5e6), step=0, last_expansion_step=-1)
    assert new.lo >= FULL[0]


def test_no_expansion_at_the_full_axis_edge():
    focus = _focus()
    new, expanded = next_focus_window(focus, **_concentrated(FULL[0], 5e6), step=0, last_expansion_step=-1)
    assert not expanded
    assert new is focus


def test_narrowing_is_delayed_after_a_recent_expansion():
    focus = _focus()
    new, expanded = next_focus_window(
        focus, **_concentrated(2.85e9, 12e6), step=NARROW_STEP, last_expansion_step=NARROW_STEP - 1
    )
    assert new is focus
    assert not expanded


def test_mismatched_particle_arrays_fail_fast():
    with pytest.raises(ValueError, match="must match"):
        next_focus_window(
            _focus(),
            center_freq_particles_phys=np.zeros(10),
            active_half_span_particles_phys=np.zeros(9),
            omega_phys=1e6,
            zeeman_split_phys=0.0,
            step=NARROW_STEP,
            last_expansion_step=-1,
        )


def test_narrowing_routes_through_the_shared_clamp(monkeypatch):
    calls: list[tuple] = []
    real = focus_mod.clamp_to_domain

    def spy(lo, hi, domain_lo, domain_hi):
        calls.append((domain_lo, domain_hi))
        return real(lo, hi, domain_lo, domain_hi)

    monkeypatch.setattr(focus_mod, "clamp_to_domain", spy)
    next_focus_window(_focus(), **_concentrated(2.85e9, 12e6), step=NARROW_STEP, last_expansion_step=-1)
    assert calls == [FULL]


# ---------------------------------------------------------------------------
# Locator-owned focus never touches the belief
# ---------------------------------------------------------------------------


def _free_center_locator(**belief_kwargs):
    belief = nv_center_smc_belief(
        noise_model=gaussian_noise(), num_particles=500, with_fixed_center_freq=False, seed=0, **belief_kwargs
    )
    return SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=100)


def _belief_geometry(belief):
    return (
        dict(belief.physical_param_bounds),
        belief.physical_x_bounds,
        dict(belief.model.param_bounds_phys),
        belief.model.x_bounds_phys,
    )


def _concentrate(belief, center: float, zeeman: float | None, n: int) -> None:
    rng = np.random.default_rng(0)
    f_lo, f_hi = belief.physical_param_bounds["center_freq"]
    j = belief._param_names.index("center_freq")
    belief._particles[:, j] = np.clip(rng.normal((center - f_lo) / (f_hi - f_lo), 1e-4, n), 0.0, 1.0)
    if zeeman is not None:
        z_lo, z_hi = belief.physical_param_bounds["zeeman_split"]
        jz = belief._param_names.index("zeeman_split")
        belief._particles[:, jz] = np.clip(rng.normal((zeeman - z_lo) / (z_hi - z_lo), 1e-4, n), 0.0, 1.0)


def test_focus_change_leaves_belief_geometry_untouched():
    loc = _free_center_locator(with_zeeman_splitting=True, hyperfine="unresolved")
    belief = loc.belief
    before = _belief_geometry(belief)
    f_lo, f_hi = belief.physical_param_bounds["center_freq"]
    _concentrate(belief, 0.5 * (f_lo + f_hi), 40e6, belief.num_particles)
    freq_before = belief.particles_phys()["center_freq"].copy()
    belief._step_count = NARROW_STEP
    belief._resample()
    freq_after_resample = belief.particles_phys()["center_freq"].copy()

    loc._update_focus_window()

    assert loc._focus.hi - loc._focus.lo < f_hi - f_lo, "focus should have narrowed"
    assert _belief_geometry(belief) == before
    # The focus update itself moves no particle (only the resample's own nudge did).
    assert np.array_equal(belief.particles_phys()["center_freq"], freq_after_resample)
    assert not np.array_equal(freq_before, freq_after_resample)


@pytest.mark.parametrize("lineshape", ["lorentzian", "saturation_voigt"])
def test_focus_contains_both_zeeman_dips(lineshape):
    loc = _free_center_locator(with_zeeman_splitting=True, hyperfine="unresolved", lineshape=lineshape)
    belief = loc.belief
    f_lo, f_hi = belief.physical_param_bounds["center_freq"]
    f0, delta0 = 0.5 * (f_lo + f_hi), 35e6
    _concentrate(belief, f0, delta0, belief.num_particles)
    belief._step_count = NARROW_STEP
    belief._resample()
    loc._update_focus_window()

    lo, hi = loc._acquisition_bounds()
    assert lo <= f0 - delta0
    assert hi >= f0 + delta0
    assert hi - lo < f_hi - f_lo


def test_single_dip_focus_narrows_well_below_half_the_axis():
    loc = _free_center_locator(with_zeeman_splitting=False, hyperfine="unresolved")
    belief = loc.belief
    assert "zeeman_split" not in belief._param_names
    f_lo, f_hi = belief.physical_param_bounds["center_freq"]
    f0 = 0.5 * (f_lo + f_hi)
    _concentrate(belief, f0, None, belief.num_particles)
    belief._step_count = NARROW_STEP
    belief._resample()
    loc._update_focus_window()

    lo, hi = loc._acquisition_bounds()
    assert hi - lo < 0.5 * (f_hi - f_lo)
    assert lo <= f0 <= hi


def test_eig_pick_lies_inside_the_focus():
    loc = _free_center_locator(with_zeeman_splitting=True, hyperfine="unresolved")
    belief = loc.belief
    f_lo, f_hi = belief.physical_param_bounds["center_freq"]
    f0 = 0.5 * (f_lo + f_hi)
    _concentrate(belief, f0, 40e6, belief.num_particles)
    belief._step_count = NARROW_STEP
    belief._resample()
    loc._update_focus_window()
    lo, hi = loc._acquisition_bounds()
    assert hi - lo < f_hi - f_lo

    for _ in range(10):
        assert lo <= loc._eig_acquire() <= hi


def test_empty_focus_fails_loudly():
    loc = _free_center_locator(with_zeeman_splitting=True, hyperfine="unresolved")
    cands = np.sort(loc.belief.get_candidate_x_phys())
    i = int(np.argmax(np.diff(cands)))
    mid = 0.5 * (cands[i] + cands[i + 1])
    loc._focus = FocusWindow(lo=mid - 0.1, hi=mid + 0.1, full_lo=FULL[0], full_hi=FULL[1])
    with pytest.raises(ValueError, match="No epoch candidate lies inside the focus"):
        loc._eig_acquire()


def test_fixed_center_freq_keeps_the_full_probe_axis_focus():
    belief = nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=200, seed=0)
    assert "center_freq" not in belief.model.parameter_names()
    loc = SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=50)
    full = loc._acquisition_bounds()
    belief._step_count = NARROW_STEP
    belief._resample()
    loc._update_focus_window()
    assert loc._acquisition_bounds() == full
