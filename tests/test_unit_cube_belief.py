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
    nv_center_smc_belief,
    run_loop,
)
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from tests.noise import gaussian_noise


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
        "noise_model": gaussian_noise(),
        "max_steps": 80,
        "convergence_threshold": 0.15,
        "parameter_bounds": pb,
    }
    final = Observer(true_signal, exp.x_min, exp.x_max).watch(
        run_loop(SequentialBayesianExperimentDesignLocator, exp, rng, **cfg)
    )
    assert final.snapshots
    # frequency is fixed (a known instrument constant, not inferred -- see
    # NVCenterCoreGenerator's docstring) under the default with_fixed_frequency=True
    # used by both the generator and nv_center_smc_belief, so it's not a particle
    # dimension / belief.estimates() key here. zeeman_split is the actual free,
    # randomized "location" parameter in this default config, so check convergence
    # on that instead (this test's purpose is verifying the normalized-probe /
    # physical-signal pipeline updates the belief correctly, not frequency specifically).
    zeeman_est = final.snapshots[-1].belief.estimates()["zeeman_split"]
    zeeman_true = true_signal.get_param_value("zeeman_split")
    assert abs(zeeman_est - zeeman_true) < 24e6


# ---------------------------------------------------------------------------
# compute_vectorized_many_fast dispatch
# ---------------------------------------------------------------------------


def _make_unit_cube_nv_model():
    from nvision.spectra.nv_center import NVCenterLorentzianModel

    # phys_bounds below carries split/k_np, so the model must actually have them
    # as free parameters -- i.e. a resolved N-14 triplet, which used to be this
    # model's default before hyperfine="unresolved" became it.
    model = NVCenterLorentzianModel(hyperfine="n14", infer_hyperfine=True, with_fixed_frequency=False)
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
