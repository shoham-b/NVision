"""Unit-cube NV belief: normalized parameter grids, physical signal values."""

from __future__ import annotations

import random

import numpy as np

from nvision.models.experiment import CoreExperiment
from nvision.models.observer import Observer
from nvision.runner import run_loop
from nvision.signal.unit_cube_grid_belief import UnitCubeGridBeliefDistribution
from nvision.signal.unit_cube_model import UnitCubeSignalModel
from nvision.sim.gen.core_generators import NVCenterCoreGenerator
from nvision.sim.locs.bayesian.belief_builders import nv_center_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator


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
