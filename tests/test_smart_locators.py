"""Smart-Sweep and Smart-SBED: a coarse width-sized search first, then work inside the focus.

Both are SBED locators in everything but acquisition. Checked on a standard generator's signal: the search grid
is sized from the prior FWHM, the switch to the focus happens as soon as the
data show a dip, and from then on every measurement lies inside the focus.
"""

from __future__ import annotations

import itertools
import random

import numpy as np

from nvision import CoreExperiment, NVCenterCoreGenerator
from nvision.belief.focus_window import FocusRegion
from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.noises import OverFrequencyGaussianNoise
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import (
    SequentialBayesianExperimentDesignLocator,
    SmartSBEDLocator,
    SmartSweepLocator,
)
from nvision.sim.locs.search_grid import COARSE_SEARCH_POINTS_PER_FWHM, FOCUS_SWEEP_POINTS_PER_FWHM
from tests.noise import gaussian_noise

NOISE_SIGMA = 0.01


def _experiment(seed: int = 3, hwhm: float = 2.75e6, contrast: float = 0.4):
    gen = NVCenterCoreGenerator(variant="lorentzian", hwhm=hwhm, contrast_total=contrast)
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


def test_smart_sweep_sweeps_its_coarse_grid_then_stays_in_the_focus():
    exp, bounds = _experiment()
    loc = _locator(SmartSweepLocator, bounds)
    spacing = loc._fwhm_prior_phys / COARSE_SEARCH_POINTS_PER_FWHM
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


def test_smart_sweep_sweeps_a_fwhm_over_7_grid_inside_the_focus():
    exp, bounds = _experiment()
    loc = _locator(SmartSweepLocator, bounds, max_steps=200)
    fine_spacing = loc._fwhm_prior_phys / FOCUS_SWEEP_POINTS_PER_FWHM
    assert fine_spacing < loc._fwhm_prior_phys / COARSE_SEARCH_POINTS_PER_FWHM

    trace = _drive(loc, exp, random.Random(11), 200)
    refine = set(loc.refine_steps)
    sweep = [(focus, x) for i, (focus, x) in enumerate(trace) if not focus.is_full_domain() and i not in refine]
    assert len(sweep) >= 5, "a focus sweep must have run before refine mode"
    first_focus, first_x = sweep[0]
    assert abs(first_x - first_focus.sections[0][0]) < 1.0, "the fresh focus grid starts at the focus's lower edge"
    sweep_x = np.array([x for _, x in sweep])
    gaps = np.abs(sweep_x[:, None] - sweep_x[None, :])[np.triu_indices(len(sweep_x), k=1)]
    assert gaps.min() > 0.5 * fine_spacing * (1 - 1e-6), "no point is measured within half a spacing of another"


def test_smart_sbed_searches_its_coarse_grid_by_information_gain_then_runs_sbed_in_the_focus():
    exp, bounds = _experiment()
    loc = _locator(SmartSBEDLocator, bounds)
    coarse_spacing = loc._fwhm_prior_phys / COARSE_SEARCH_POINTS_PER_FWHM
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


def _true_dip_center_phys(true_signal, lo_phys: float, hi_phys: float) -> float:
    """Drive frequency (Hz) of the true signal's minimum, located on a dense grid then a fine one around it."""
    coarse = np.linspace(lo_phys, hi_phys, 200_001)
    x0 = coarse[int(np.argmin([true_signal(float(x)) for x in coarse]))]
    fine = np.linspace(x0 - 3e3, x0 + 3e3, 601)
    return float(fine[int(np.argmin([true_signal(float(x)) for x in fine]))])


def test_smart_sweep_refines_at_both_steepest_slopes_once_the_dip_is_known():
    hwhm = 2.75e6
    exp, bounds = _experiment(hwhm=hwhm)
    loc = _locator(SmartSweepLocator, bounds, max_steps=200)
    trace = _drive(loc, exp, random.Random(11), 200)

    assert loc.refine_steps, "a 0.4-contrast dip must be pinned down well enough to switch to refine mode"
    first_refine = loc.refine_steps[0]
    assert not trace[first_refine][0].is_full_domain(), "refine mode only starts once a focus exists"
    refine_x = np.array([trace[i][1] for i in loc.refine_steps])

    # Lorentzian: the d signal / d zeeman_shift maxima are x0 +- HWHM / sqrt(3). The locator finds them
    # numerically from its own estimate. The standard priors leave hwhm itself uncertain, so the picks only
    # land within about half an HWHM of the true optima (a tight-nuisance prior would give 0.15 HWHM); with the
    # FWHM/2 coarse grid the switch to refine comes a few measurements earlier and the first picks reach 0.50 HWHM.
    x0 = _true_dip_center_phys(exp.true_signal, exp.drive_freq_min_phys, exp.drive_freq_max_phys)
    optima = np.array([x0 - hwhm / np.sqrt(3.0), x0 + hwhm / np.sqrt(3.0)])
    offset = np.min(np.abs(refine_x[:, None] - optima[None, :]), axis=1)
    assert np.all(offset < 0.55 * hwhm)
    sides = refine_x > x0
    assert sides.any(), "the upper slope is measured"
    assert (~sides).any(), "the lower slope is measured"
    consecutive = [b - a == 1 for a, b in itertools.pairwise(loc.refine_steps)]
    alternating = [s != t for s, t in itertools.pairwise(sides)]
    assert all(a for a, c in zip(alternating, consecutive, strict=True) if c), "back-to-back refine picks alternate"


def test_smart_sweep_refine_points_come_from_the_model_not_a_lorentzian_formula():
    gen = NVCenterCoreGenerator(variant="voigt", hwhm=1.5e6, contrast_total=0.4, sigma_inhom=1.0e6)
    true_signal = gen.generate(random.Random(5))
    noise = CompositeNoise(over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(NOISE_SIGMA)]))
    exp = CoreExperiment(
        true_signal=true_signal,
        noise=noise,
        drive_freq_min_phys=gen.drive_freq_min_phys,
        drive_freq_max_phys=gen.drive_freq_max_phys,
    )
    loc = _locator(SmartSweepLocator, true_signal.bounds, max_steps=200)
    trace = _drive(loc, exp, random.Random(11), 200)
    assert loc.refine_steps, "the Voigt dip must also reach refine mode"
    refine_x = np.array([trace[i][1] for i in loc.refine_steps])
    x0 = _true_dip_center_phys(true_signal, exp.drive_freq_min_phys, exp.drive_freq_max_phys)
    assert (refine_x > x0).any()
    assert (refine_x < x0).any()
    # On the slopes, not at the (zero-sensitivity) dip centre.
    assert np.min(np.abs(refine_x - x0)) > 0.2 * loc._fwhm_prior_phys / 2
