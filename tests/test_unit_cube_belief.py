"""Unit-cube NV belief: normalized parameter grids, physical signal values."""

from __future__ import annotations

import random

import numpy as np
import pytest

from nvision import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    CoreExperiment,
    NVCenterCoreGenerator,
    Observer,
    UnitCubeGridMarginalDistribution,
    UnitCubeSignalModel,
    UnitCubeSMCMarginalDistribution,
    nv_center_belief,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_smc_belief,
    run_loop,
)
from nvision.belief.grid_marginal import GridMarginalDistribution
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator


def test_nv_center_default_bounds_align_with_generation_formulas():
    gen = nv_center_lorentzian_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)
    b = nv_center_belief()
    for key in ("frequency", "linewidth", "split", "k_np"):
        assert b.physical_param_bounds[key] == gen[key]
    glo, ghi = gen["c_total"]
    blo, bhi = b.physical_param_bounds["c_total"]
    assert bhi == ghi
    assert glo <= blo < bhi


def test_nv_center_belief_is_unit_cube_with_wrapped_model():
    b = nv_center_belief()
    assert isinstance(b, UnitCubeGridMarginalDistribution)
    assert isinstance(b.model, UnitCubeSignalModel)
    for p in b.parameters:
        assert p.bounds == (0.0, 1.0)
        assert float(p.grid[0]) == 0.0
        assert float(p.grid[-1]) == 1.0


def test_unit_cube_estimates_are_physical_hz():
    rng = random.Random(2025)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    phys_bounds = true_signal.all_param_bounds()
    b = nv_center_belief(phys_bounds)
    # Posterior means start at box centers in *unit* space → physical midpoints
    est = b.estimates()
    assert 2.6e9 < est["frequency"] < 3.1e9
    assert est["k_np"] > 2.0


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_bayesian_sbed_nv_updates_with_normalized_probe_and_physical_signal(monkeypatch):
    # The SBED locator's acquisition (exploration/dip-jitter/Thompson-sampling
    # branches) draws from the global numpy RNG, and UnitCubeSMCMarginalDistribution
    # seeds its own internal `_rng` from OS entropy on every construction/copy. Pin
    # both so this test is deterministic instead of flaking around the threshold.
    np.random.seed(11)
    shared_rng = np.random.default_rng(11)
    monkeypatch.setattr(np.random, "default_rng", lambda *args, **kwargs: shared_rng)

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
    freq_est = final.snapshots[-1].belief.estimates()["frequency"]
    freq_true = true_signal.get_param_value("frequency")
    assert abs(freq_est - freq_true) < 0.2e9


def test_physical_param_grid_for_plots():
    b = nv_center_belief()
    g = b.physical_param_grid("frequency")
    assert np.isfinite(g).all()
    assert float(g[0]) < float(g[-1])


def test_narrow_scan_parameter_physical_bounds_grid():
    b = nv_center_belief(n_grid_freq=40)
    old_lo, old_hi = b.physical_param_bounds["frequency"]
    mid = 0.5 * (old_lo + old_hi)
    quarter = 0.25 * (old_hi - old_lo)
    nl, nh = mid - quarter, mid + quarter
    b.narrow_scan_parameter_physical_bounds("frequency", nl, nh)
    flo, fhi = b.physical_param_bounds["frequency"]
    assert abs(flo - nl) < 1e-6 * (old_hi - old_lo)
    assert abs(fhi - nh) < 1e-6 * (old_hi - old_lo)
    assert b.physical_x_bounds == (flo, fhi)
    assert b.model.param_bounds_phys["frequency"] == (flo, fhi)
    assert b.model.x_bounds_phys == (flo, fhi)
    g = GridMarginalDistribution.get_grid_param(b, "frequency")
    assert abs(float(np.sum(g.posterior)) - 1.0) < 1e-9


def test_narrow_scan_parameter_physical_bounds_smc():
    b = nv_center_smc_belief(num_particles=200)
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

    b = nv_center_smc_belief(num_particles=100)
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
    # Reset bounds, narrow to the left, so there is space to expand to the right
    b.narrow_scan_parameter_physical_bounds("frequency", orig_lo, mid)
    lo_phys, hi_phys = b.physical_param_bounds["frequency"]
    assert hi_phys == mid

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

    b = nv_center_smc_belief(num_particles=100)
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

    model = NVCenterLorentzianModel()
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
