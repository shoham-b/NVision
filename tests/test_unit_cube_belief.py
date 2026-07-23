"""Unit-cube NV belief: normalized parameter grids, physical signal values."""

from __future__ import annotations

import random

import numpy as np
import pytest

from nvision import (
    CoreExperiment,
    NVCenterCoreGenerator,
    Observer,
    UnitCubeSignalModel,
    UnitCubeSMCMarginalDistribution,
    nv_center_smc_belief,
    run_loop,
)
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_bayesian_sbed_nv_updates_with_normalized_probe_and_physical_signal():
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    x_min, x_max = true_signal.get_param_bounds("frequency")
    assert x_min is not None
    exp = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)
    pb = {name: true_signal.get_param_bounds(name) for name in true_signal.parameter_names}
    cfg = {
        "builder": nv_center_smc_belief,
        "max_steps": 80,
        "convergence_threshold": 0.15,
        "parameter_bounds": pb,
        "noise_std": 0.05,
        "n_grid_freq": 48,
        "n_grid_linewidth": 24,
        "n_grid_split": 24,
        "n_grid_k_np": 16,
        "n_grid_amplitude": 16,
    }
    final = Observer(true_signal, exp.x_min, exp.x_max).watch(
        run_loop(SequentialBayesianExperimentDesignLocator, exp, rng, **cfg)
    )
    assert final.snapshots
    # frequency is fixed (a known, calibrated zero-field reference), not inferred
    # by the SBED belief -- it has no particle dimension or estimate entry.
    assert "frequency" not in final.snapshots[-1].belief.estimates()


def test_narrow_scan_parameter_physical_bounds_smc():
    b = nv_center_smc_belief(num_particles=200, with_fixed_frequency=False)
    assert isinstance(b, UnitCubeSMCMarginalDistribution)
    old_lo, old_hi = b.physical_param_bounds["frequency"]
    mid = 0.5 * (old_lo + old_hi)
    quarter = 0.25 * (old_hi - old_lo)
    nl, nh = mid - quarter, mid + quarter
    b.narrow_scan_parameter_physical_bounds("frequency", nl, nh)
    assert b.physical_param_bounds["frequency"] == (nl, nh)
    assert b.physical_x_bounds == (nl, nh)
    j = b._param_names.index("frequency")
    assert np.all((b._particles[:, j] >= 0.0) & (b._particles[:, j] <= 1.0))


def test_smc_narrowing_delay_and_boundary_escape(monkeypatch):
    # Test 1: Narrowing delay safeguard
    # Set the environment variable to 8 steps
    monkeypatch.setenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8")

    b = nv_center_smc_belief(num_particles=100, with_fixed_frequency=False)
    assert isinstance(b, UnitCubeSMCMarginalDistribution)

    # Capture original bounds
    orig_lo, orig_hi = b._original_physical_x_bounds

    # Check step count is initially 0
    assert b._step_count == 0

    # Trigger _resample directly. Since _step_count < 8, it should return early and not narrow
    # Wait, super()._resample() inside _resample() calls systematic resampling.
    # To run _resample safely, we can just call it directly since particles are already initialized.
    old_lo, old_hi = b.physical_param_bounds["frequency"]
    b._resample()
    new_lo, new_hi = b.physical_param_bounds["frequency"]
    # Bounds should NOT have narrowed
    assert (old_lo, old_hi) == (new_lo, new_hi)

    # Test 2: Left boundary piling triggers left expansion
    # First, narrow the bounds manually so there is space to expand to the left
    mid = 0.5 * (orig_lo + orig_hi)
    narrow_lo = mid
    narrow_hi = orig_hi
    b.narrow_scan_parameter_physical_bounds("frequency", narrow_lo, narrow_hi)

    # Verify narrowing worked
    lo_phys, hi_phys = b.physical_param_bounds["frequency"]
    assert lo_phys == narrow_lo

    # Force particles to pile up near 0.0 in unit space (u_vals < 0.05)
    j = b._param_names.index("frequency")
    b._particles[:, j] = 0.02

    # Call _resample. Even though _step_count < 8, the active boundary-escape guard is checked
    # BEFORE the narrowing delay safeguard, so it should trigger expansion!
    b._resample()

    # Verify that the bounds expanded to the left (lo boundary decreased)
    lo_expanded, hi_expanded = b.physical_param_bounds["frequency"]
    assert lo_expanded < lo_phys
    assert hi_expanded == hi_phys

    # Test 3: Right boundary piling triggers right expansion
    # Reset bounds, narrow to the left, so there is space to expand to the right.
    # Test 2's left-expansion remapped the piled-at-0.02 particles into unit space
    # under the *new* (wider) bounds, landing them just above `mid` (not exactly at
    # it) plus a little resampling jitter -- so narrow a bit past `mid` here to
    # reliably keep them in-bounds rather than triggering the boundary-escape reject.
    mid_hi = mid + 0.1 * (orig_hi - orig_lo)
    b.narrow_scan_parameter_physical_bounds("frequency", orig_lo, mid_hi)
    lo_phys, hi_phys = b.physical_param_bounds["frequency"]
    assert hi_phys == mid_hi

    # Force particles to pile up near 1.0 in unit space (u_vals > 0.95)
    b._particles[:, j] = 0.98

    b._resample()

    # Verify that the bounds expanded to the right (hi boundary increased)
    lo_expanded, hi_expanded = b.physical_param_bounds["frequency"]
    assert lo_expanded == lo_phys
    assert hi_expanded > hi_phys

    # Test 4: Narrowing is delayed after an expansion even if step_count > min_narrowing_steps
    # Reset step count to 10 (> 8) and set _last_expansion_step to 9 (only 1 step since expansion)
    b._step_count = 10
    b._last_expansion_step = 9

    # We will manually set the bounds to be narrow, so if narrowing runs, it would narrow even further
    # We will trigger _resample. Since step_count - last_expansion_step < 8, it should return early
    # without running standard percentile narrowing.
    b.narrow_scan_parameter_physical_bounds("frequency", lo_expanded, hi_expanded)
    old_lo, old_hi = b.physical_param_bounds["frequency"]

    # Reset particles so they don't pile up and trigger another expansion
    b._particles[:, j] = 0.5

    b._resample()

    new_lo, new_hi = b.physical_param_bounds["frequency"]
    assert (old_lo, old_hi) == (new_lo, new_hi)


def test_smc_exact_active_range_union_narrowing(monkeypatch):
    # Set step count > min_narrowing_steps so narrowing runs
    monkeypatch.setenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "5")
    monkeypatch.setenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0")
    monkeypatch.setenv("NVISION_SMC_FOCUSING_TAIL_PERCENTILE", "1.0")

    # Mock the base class _resample to be a no-op so it doesn't resample, shrink,
    # or nudge our manually controlled particles.
    from nvision.belief.smc_marginal import SMCMarginalDistribution

    monkeypatch.setattr(SMCMarginalDistribution, "_resample", lambda self: None)

    b = nv_center_smc_belief(
        num_particles=100, with_hyperfine_splitting=True, with_zeeman_splitting=False, with_fixed_frequency=False
    )
    assert isinstance(b, UnitCubeSMCMarginalDistribution)
    b._step_count = 10  # > 5, narrowing runs

    # Let's inspect parameter names and map indices
    j_freq = b._param_names.index("frequency")
    j_split = b._param_names.index("split")
    j_line = b._param_names.index("linewidth")

    # Set up controlled unit particles
    # frequency center = 0.5 (unit space)
    b._particles[:, j_freq] = 0.5
    # split = 0.2 (unit space)
    b._particles[:, j_split] = 0.2
    # linewidth = 0.1 (unit space)
    b._particles[:, j_line] = 0.1

    # Get physical ranges and values
    lo_f, hi_f = b.physical_param_bounds["frequency"]
    lo_s, hi_s = b.physical_param_bounds["split"]
    lo_l, hi_l = b.physical_param_bounds["linewidth"]

    freq_phys = lo_f + 0.5 * (hi_f - lo_f)
    split_phys = lo_s + 0.2 * (hi_s - lo_s)
    line_phys = lo_l + 0.1 * (hi_l - lo_l)

    # Hand-calculate expected active range bounds
    k = 3.0
    expected_left = freq_phys - split_phys - k * line_phys
    expected_right = freq_phys + split_phys + k * line_phys

    # Trigger resampling which triggers the unified active-range union narrowing
    b._resample()

    new_lo, new_hi = b.physical_param_bounds["frequency"]

    # Since all particles are identical, percentiles will yield exactly the same values
    assert abs(new_lo - expected_left) < 1000.0
    assert abs(new_hi - expected_right) < 1000.0


# ---------------------------------------------------------------------------
# compute_vectorized_many_fast dispatch
# ---------------------------------------------------------------------------


def _make_unit_cube_nv_model():
    from nvision.spectra.nv_center import NVCenterLorentzianModel

    model = NVCenterLorentzianModel(with_fixed_frequency=False)
    phys_bounds = {
        "frequency": (2.86e9, 2.88e9),
        "linewidth": (5e6, 15e6),
        "split": (1e6, 5e6),
        "k_np": (0.5, 1.5),
        "c_total": (0.05, 0.2),
    }
    wrapped = UnitCubeSignalModel(model, phys_bounds, phys_bounds["frequency"])
    return wrapped, model


def test_unit_cube_compute_vectorized_many_fast_dispatches_to_inner_fast():
    """UnitCubeSignalModel.compute_vectorized_many_fast must call the inner model's
    fast kernel, not the regular _many kernel.

    Before the fix, the base-class fallback silently routed to compute_vectorized_many
    (non-fastmath), so the fastmath Numba kernels were never reached through the
    unit-cube wrapper.
    """
    wrapped, inner = _make_unit_cube_nv_model()
    rng = np.random.default_rng(0)
    param_arrays = [rng.random(100).astype(np.float32) for _ in range(5)]
    xs = rng.random(50).astype(np.float32)

    fast_calls: list[int] = []
    many_calls: list[int] = []

    orig_fast = inner.compute_vectorized_many_fast
    orig_many = inner.compute_vectorized_many

    def _track_fast(*a, **kw):
        fast_calls.append(1)
        return orig_fast(*a, **kw)

    def _track_many(*a, **kw):
        many_calls.append(1)
        return orig_many(*a, **kw)

    inner.compute_vectorized_many_fast = _track_fast
    inner.compute_vectorized_many = _track_many

    try:
        wrapped.compute_vectorized_many_fast(xs, param_arrays)
        assert len(fast_calls) == 1, "inner.compute_vectorized_many_fast was not called"
        assert len(many_calls) == 0, "compute_vectorized_many was called instead of fast variant"
    finally:
        inner.compute_vectorized_many_fast = orig_fast
        inner.compute_vectorized_many = orig_many


def test_unit_cube_compute_vectorized_many_dispatches_to_inner_exact():
    """compute_vectorized_many must NOT call the fast kernel."""
    wrapped, inner = _make_unit_cube_nv_model()
    rng = np.random.default_rng(1)
    param_arrays = [rng.random(100).astype(np.float32) for _ in range(5)]
    xs = rng.random(50).astype(np.float32)

    fast_calls: list[int] = []
    many_calls: list[int] = []

    orig_fast = inner.compute_vectorized_many_fast
    orig_many = inner.compute_vectorized_many

    def _track_fast(*a, **kw):
        fast_calls.append(1)
        return orig_fast(*a, **kw)

    def _track_many(*a, **kw):
        many_calls.append(1)
        return orig_many(*a, **kw)

    inner.compute_vectorized_many_fast = _track_fast
    inner.compute_vectorized_many = _track_many

    try:
        wrapped.compute_vectorized_many(xs, param_arrays)
        assert len(many_calls) == 1, "inner.compute_vectorized_many was not called"
        assert len(fast_calls) == 0, "fast kernel was called from exact path"
    finally:
        inner.compute_vectorized_many_fast = orig_fast
        inner.compute_vectorized_many = orig_many


def test_unit_cube_fast_and_exact_outputs_are_close():
    """fast and exact variants should agree closely (fastmath rounding is small)."""
    wrapped, _ = _make_unit_cube_nv_model()
    rng = np.random.default_rng(2)
    param_arrays = [rng.random(200).astype(np.float32) for _ in range(5)]
    xs = rng.random(100).astype(np.float32)

    out_exact = wrapped.compute_vectorized_many(xs, param_arrays)
    out_fast = wrapped.compute_vectorized_many_fast(xs, param_arrays)

    assert out_exact.shape == out_fast.shape
    np.testing.assert_allclose(out_fast, out_exact, rtol=1e-4, atol=1e-6)
