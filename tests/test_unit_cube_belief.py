"""Unit-cube NV belief: normalized parameter grids, physical signal values."""

from __future__ import annotations

import random

import numpy as np

from nvision.belief.grid_belief import GridBeliefDistribution
from nvision.belief.unit_cube_grid_belief import UnitCubeGridBeliefDistribution
from nvision.belief.unit_cube_smc_belief import UnitCubeSMCBeliefDistribution
from nvision.models.experiment import CoreExperiment
from nvision.models.observer import Observer
from nvision.runner import run_loop
from nvision.signal.unit_cube import UnitCubeSignalModel
from nvision.sim.gen.core_generators import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    NVCenterCoreGenerator,
    nv_center_lorentzian_bounds_for_domain,
)
from nvision.sim.locs.bayesian.belief_builders import nv_center_belief, nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator


def test_nv_center_default_bounds_align_with_generation_formulas():
    gen = nv_center_lorentzian_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)
    b = nv_center_belief()
    for key in ("frequency", "linewidth", "split", "k_np", "background"):
        assert b.physical_param_bounds[key] == gen[key]
    glo, ghi = gen["amplitude"]
    blo, bhi = b.physical_param_bounds["amplitude"]
    assert bhi == ghi
    assert glo <= blo < bhi


def test_nv_center_belief_is_unit_cube_with_wrapped_model():
    b = nv_center_belief()
    assert isinstance(b, UnitCubeGridBeliefDistribution)
    assert isinstance(b.model, UnitCubeSignalModel)
    for p in b.parameters:
        assert p.bounds == (0.0, 1.0)
        assert float(p.grid[0]) == 0.0
        assert float(p.grid[-1]) == 1.0


def test_unit_cube_estimates_are_physical_hz():
    rng = random.Random(2025)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    phys_bounds = {p.name: p.bounds for p in true_signal.parameters}
    b = nv_center_belief(phys_bounds)
    # Posterior means start at box centers in *unit* space → physical midpoints
    est = b.estimates()
    assert 2.6e9 < est["frequency"] < 3.1e9
    assert est["k_np"] > 2.0


def test_bayesian_sbed_nv_updates_with_normalized_probe_and_physical_signal():
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    x_min = x_max = None
    for p in true_signal.parameters:
        if p.name == "frequency":
            x_min, x_max = p.bounds
            break
    assert x_min is not None
    exp = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)
    pb = {p.name: p.bounds for p in true_signal.parameters}
    cfg = {
        "builder": nv_center_belief,
        "max_steps": 80,
        "convergence_threshold": 0.15,
        "parameter_bounds": pb,
        "n_grid_freq": 48,
        "n_grid_linewidth": 24,
        "n_grid_split": 24,
        "n_grid_k_np": 16,
        "n_grid_amplitude": 16,
        "n_grid_background": 16,
    }
    final = Observer(true_signal, exp.x_min, exp.x_max).watch(
        run_loop(SequentialBayesianExperimentDesignLocator, exp, rng, **cfg)
    )
    assert final.snapshots
    freq_est = final.snapshots[-1].belief.estimates()["frequency"]
    freq_true = true_signal.get_param("frequency").value
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
    g = GridBeliefDistribution.get_param(b, "frequency")
    assert abs(float(np.sum(g.posterior)) - 1.0) < 1e-9


def test_narrow_scan_parameter_physical_bounds_smc():
    b = nv_center_smc_belief(num_particles=200)
    assert isinstance(b, UnitCubeSMCBeliefDistribution)
    old_lo, old_hi = b.physical_param_bounds["frequency"]
    mid = 0.5 * (old_lo + old_hi)
    quarter = 0.25 * (old_hi - old_lo)
    nl, nh = mid - quarter, mid + quarter
    b.narrow_scan_parameter_physical_bounds("frequency", nl, nh)
    assert b.physical_param_bounds["frequency"] == (nl, nh)
    assert b.physical_x_bounds == (nl, nh)
    j = b._param_names.index("frequency")
    assert np.all((b._particles[:, j] >= 0.0) & (b._particles[:, j] <= 1.0))
