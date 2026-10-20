"""Smart-Sweep and Smart-SBED: a coarse width-sized search first, then work inside the focus.

Both are SBED locators in everything but acquisition. Checked on a real resonance-search signal (tight
nuisance priors, unknown ``zeeman_split``): the search grid is sized from the prior FWHM, the switch to the
focus happens as soon as the data show a dip, and from then on every measurement lies inside the focus.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from nvision import CoreExperiment, NVCenterCoreGenerator
from nvision.belief.focus_window import FocusRegion
from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.noises import OverFrequencyGaussianNoise
from nvision.sim.gen.nv_center_generator import RESONANCE_SEARCH_NUISANCE_STD_SCALE
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import (
    SequentialBayesianExperimentDesignLocator,
    SmartSBEDLocator,
    SmartSweepLocator,
)
from nvision.sim.locs.search_grid import SBED_COARSE_POINTS_PER_FWHM, SWEEP_POINTS_PER_FWHM
from tests.noise import gaussian_noise

NOISE_SIGMA = 0.01


def _experiment(seed: int = 3, linewidth: float = 2.75e6, contrast: float = 0.4):
    gen = NVCenterCoreGenerator(variant="lorentzian", linewidth=linewidth, c_total=contrast, resonance_search=True)
    true_signal = gen.generate(random.Random(seed))
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(NOISE_SIGMA)]))
    exp = CoreExperiment(
        true_signal=true_signal,
        noise=noise,
        drive_freq_min_phys=gen.drive_freq_min_phys,
        drive_freq_max_phys=gen.drive_freq_max_phys,
    )
    return exp, true_signal.bounds


def _locator(cls, bounds, max_steps: int = 120):
    return cls.create(
        builder=nv_center_smc_belief,
        max_steps=max_steps,
        parameter_bounds=bounds,
        noise_model=gaussian_noise(),
        num_particles=400,
        seed=7,
    )


def _drive(locator, exp, rng, n_steps: int):
    """Run ``n_steps`` measurements; return [(focus the pick was made under, drive_freq_phys measured)]."""
    lo, hi = exp.drive_freq_min_phys, exp.drive_freq_max_phys
    trace = []
    for _ in range(n_steps):
        if locator.done():
            break
        x_unit = locator.next_drive_freq_unit()
        trace.append((locator._focus, lo + x_unit * (hi - lo)))  # the focus the pick was made under
        locator.observe(exp.measure(x_unit, rng))
    return trace


def test_resonance_search_priors_are_tight_on_nuisance_and_absent_on_the_resonance():
    gen = NVCenterCoreGenerator(variant="lorentzian", linewidth=2.0e6, c_total=0.3)
    default = gen.generate(random.Random(1)).bounds["_priors"]
    searched = (
        NVCenterCoreGenerator(variant="lorentzian", linewidth=2.0e6, c_total=0.3, resonance_search=True)
        .generate(random.Random(1))
        .bounds["_priors"]
    )
    assert "zeeman_split" in default
    assert "zeeman_split" not in searched
    assert searched["linewidth"][1] == pytest.approx(default["linewidth"][1] * RESONANCE_SEARCH_NUISANCE_STD_SCALE)
    assert searched["c_total"][1] == pytest.approx(default["c_total"][1] * RESONANCE_SEARCH_NUISANCE_STD_SCALE)


def test_smart_sweep_sweeps_at_fwhm_over_three_then_stays_in_the_focus():
    exp, bounds = _experiment()
    loc = _locator(SmartSweepLocator, bounds)
    spacing = loc._fwhm_prior_phys / SWEEP_POINTS_PER_FWHM
    trace = _drive(loc, exp, random.Random(11), 100)

    first = np.array([x for _, x in trace[:6]])
    assert np.all(np.diff(first) > 0), "the search is an ascending sweep"
    assert np.max(np.diff(first)) <= spacing * (1 + 1e-9)

    focus_steps = [i for i, (focus, _) in enumerate(trace) if not focus.is_full_domain()]
    assert focus_steps, "a 0.4-contrast dip must be found within 100 measurements"
    for i in focus_steps:
        focus, x = trace[i]
        assert focus.contains(np.array([x]))[0], f"step {i + 1} measured {x:g} outside the focus {focus.sections}"
    assert focus_steps[0] > 0
    assert all(trace[i][0].is_full_domain() for i in range(focus_steps[0]))


def test_smart_sbed_searches_a_fwhm_grid_by_information_gain_then_runs_sbed_in_the_focus():
    exp, bounds = _experiment()
    loc = _locator(SmartSBEDLocator, bounds)
    coarse_spacing = loc._fwhm_prior_phys / SBED_COARSE_POINTS_PER_FWHM
    trace = _drive(loc, exp, random.Random(11), 60)

    search = [x for focus, x in trace if focus.is_full_domain()]
    assert len(search) >= 3
    # The search grid is refined by halving once a pass finds no dip, so every point lies on some level.
    grid = np.linspace(
        exp.drive_freq_min_phys, exp.drive_freq_max_phys, loc._grid._base_intervals * 2**loc._grid.level + 1
    )
    on_grid = np.min(np.abs(np.asarray(search)[:, None] - grid[None, :]), axis=1) < 1.0
    assert on_grid.all(), "before the focus exists every measurement is a point of the (nested) search grid"
    assert loc._grid.spacing_phys <= coarse_spacing / 2**loc._grid.level * (1 + 1e-9)
    assert len(set(np.round(search, 0))) == len(search), "no grid point is measured twice"

    for focus, x in trace:
        if not focus.is_full_domain():
            assert focus.contains(np.array([x]))[0]


def test_smart_variants_share_sbed_inference_and_stop_rules():
    _, bounds = _experiment()
    plain = _locator(SequentialBayesianExperimentDesignLocator, bounds)
    for cls in (SmartSBEDLocator, SmartSweepLocator):
        loc = _locator(cls, bounds)
        assert isinstance(loc, SequentialBayesianExperimentDesignLocator)
        assert loc._convergence_patience_steps == plain._convergence_patience_steps
        assert loc.convergence_threshold == plain.convergence_threshold
        assert isinstance(loc._focus, FocusRegion)
        assert loc._focus.is_full_domain()
