"""generators_basic() fixed/swept width & contrast behavior."""

from __future__ import annotations

import nvision.sim.defaults as sim_defaults
from nvision.sim import presets
from nvision.sim.combinations import CombinationGrid


def test_generators_basic_default_unchanged(monkeypatch):
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_LINEWIDTH", None)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_CONTRAST", None)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_PARAM", None)
    gens = presets.generators_basic()
    names = [name for name, _ in gens]
    assert names == [
        "NVCenter-lorentzian",
        "NVCenter-voigt",
        "NVCenter-inhom-0",
        "NVCenter-inhom-low",
        "NVCenter-inhom-high",
    ]


def test_generators_basic_fixed_values_single_generator(monkeypatch):
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_LINEWIDTH", 1e6)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_CONTRAST", 0.1)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_PARAM", None)
    gens = presets.generators_basic()
    assert len(gens) == 1
    name, gen = gens[0]
    assert name == "NVCenter-lorentzian-w1.00MHz-c0.10"
    assert gen.linewidth == 1e6
    assert gen.c_total == 0.1


def test_generators_basic_sweep_width(monkeypatch):
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_CONTRAST", 0.1)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_PARAM", "width")
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_MIN", 2e5)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_MAX", 5e6)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_STEPS", 5)
    gens = presets.generators_basic()
    assert len(gens) == 5
    names = [name for name, _ in gens]
    assert len(set(names)) == 5
    for _name, gen in gens:
        assert gen.c_total == 0.1
        assert gen.linewidth is not None


def test_generators_basic_sweep_contrast(monkeypatch):
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_LINEWIDTH", 1e6)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_PARAM", "contrast")
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_MIN", 0.1)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_MAX", 0.4)
    monkeypatch.setattr(sim_defaults, "NVISION_SIGNAL_SWEEP_STEPS", 4)
    gens = presets.generators_basic()
    assert len(gens) == 4
    for _name, gen in gens:
        assert gen.linewidth == 1e6
        assert gen.c_total is not None


def test_combination_grid_expands_swept_generators_across_noise_and_strategy():
    """Study-1 shape: fixed width/contrast, 5 noise levels, filtered to 2 strategies -> 10 combos."""
    orig_linewidth = sim_defaults.NVISION_SIGNAL_LINEWIDTH
    orig_contrast = sim_defaults.NVISION_SIGNAL_CONTRAST
    orig_sweep_param = sim_defaults.NVISION_SIGNAL_SWEEP_PARAM
    orig_noise_steps = sim_defaults.NVISION_NOISE_GAUSS_STEPS
    orig_noise_max = sim_defaults.NVISION_NOISE_MAX_GAUSS
    try:
        sim_defaults.NVISION_SIGNAL_LINEWIDTH = 1e6
        sim_defaults.NVISION_SIGNAL_CONTRAST = 0.1
        sim_defaults.NVISION_SIGNAL_SWEEP_PARAM = None
        sim_defaults.NVISION_NOISE_GAUSS_STEPS = 5
        sim_defaults.NVISION_NOISE_MAX_GAUSS = 0.2

        grid = CombinationGrid()
        combos = list(grid.iter(filter_signal="lorentzian", filter_strategy="SimpleSweep,Bayesian-SBED"))
        assert len(combos) == 10
        assert all(c.generator_name == "NVCenter-lorentzian-w1.00MHz-c0.10" for c in combos)
        assert {c.noise_name for c in combos} == set(dict(presets.noises_single_each()).keys())
    finally:
        sim_defaults.NVISION_SIGNAL_LINEWIDTH = orig_linewidth
        sim_defaults.NVISION_SIGNAL_CONTRAST = orig_contrast
        sim_defaults.NVISION_SIGNAL_SWEEP_PARAM = orig_sweep_param
        sim_defaults.NVISION_NOISE_GAUSS_STEPS = orig_noise_steps
        sim_defaults.NVISION_NOISE_MAX_GAUSS = orig_noise_max
