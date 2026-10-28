"""GenericSweepLocator(fit_only=True): dip scan + least-squares fits instead of an SMC belief."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.random import default_rng

from nvision.belief.fit_belief import FitBelief
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel, NVCenterLorentzianSpectrum
from nvision.spectra.unit_cube import UnitCubeSignalModel
from tests.noise import gaussian_noise

_LO, _HI = 2.82, 2.92
_NOISE = 0.005
_BOUNDS = {
    "zero_field_freq": (_LO, _HI),
    "hwhm": (1e-4, 0.02),
    "hyperfine_shift": (0.001, 0.015),
    "k_np": (1.0, 5.0),
    "contrast_total": (0.05, 0.5),
}
_MODEL = NVCenterLorentzianModel(hyperfine="n14", infer_hyperfine=True, with_fixed_zero_field_freq=False)


def _locator(passes: int = 3, n_grid: int = 100) -> GenericSweepLocator:
    unit_model = UnitCubeSignalModel(_MODEL, param_bounds_phys=dict(_BOUNDS), x_bounds_phys=(_LO, _HI))
    template = SMCMarginalDistribution(
        unit_model,
        num_particles=32,
        physical_param_bounds=unit_model.param_bounds_phys,
        drive_freq_bounds_phys=(_LO, _HI),
        noise_model=gaussian_noise(),
    )
    return GenericSweepLocator(
        belief=template,
        spacing_divisor=None,
        signal_model=unit_model,
        max_steps=n_grid,
        max_measurements=passes * n_grid,
        noise_std=_NOISE,
        domain_lo=_LO,
        domain_hi=_HI,
        fit_only=True,
    )


def _signal(center: float, contrast: float = 0.3) -> NVCenterLorentzianSpectrum:
    return NVCenterLorentzianSpectrum(
        zero_field_freq=center, hwhm=0.003, hyperfine_shift=0.003, k_np=2.0, contrast_total=contrast
    )


def _run(loc: GenericSweepLocator, truth: NVCenterLorentzianSpectrum | None, seed: int = 3) -> list[FitBelief]:
    """Drive ``loc`` to completion on ``truth`` (None: a flat, noisy line); return its per-step belief snapshots."""
    rng = default_rng(seed)
    snapshots = []
    while not loc.done():
        x = loc.next()
        clean = 1.0 if truth is None else float(_MODEL.compute(_LO + x * (_HI - _LO), truth))
        loc.observe(Observation(drive_freq_unit=x, signal_value=clean + rng.normal(0, _NOISE), noise_std=_NOISE))
        snapshots.append(loc.belief.copy())
    loc.finalize()
    return snapshots


def test_fit_only_replaces_the_smc_belief_and_finds_the_dip():
    loc = _locator()
    assert isinstance(loc.belief, FitBelief)
    truth_center = 2.861
    _run(loc, _signal(truth_center))

    res = loc.result()
    assert not res["fit_failed"]
    assert abs(res["zero_field_freq"] - truth_center) < 0.002  # 2 MHz of a 100 MHz axis
    est = loc.belief.estimates()
    assert est["zero_field_freq"] == pytest.approx(res["zero_field_freq"])
    assert all(math.isfinite(u) and u > 0 for u in loc.belief.uncertainty().values())
    assert loc.belief.current_dips()


def test_nothing_is_fitted_until_a_dip_is_seen_and_later_fits_are_warm():
    loc = _locator()
    calls: list[tuple[int, bool]] = []
    real_fit = loc._fit_model

    def spy(x, ys, *, warm_start=None):
        calls.append((len(x), warm_start is not None))
        return real_fit(x, ys, warm_start=warm_start)

    loc._fit_model = spy
    snapshots = _run(loc, _signal(2.861))

    assert calls, "a dip was present, so a fit must have run"
    assert calls[0][1] is False, "the first fit is the full multi-start fit"
    assert all(warm for _, warm in calls[1:]), "every later fit starts from the previous one"
    first_fit_step = next(i for i, s in enumerate(snapshots) if s.estimates() != snapshots[0].estimates())
    assert snapshots[0].estimates() == snapshots[first_fit_step - 1].estimates(), "estimates hold until the first fit"


def test_no_dip_means_no_fit_and_a_recorded_failure():
    loc = _locator(passes=1)
    loc._fit_model = lambda *a, **k: pytest.fail("a sweep that shows no dip must not be fitted")
    _run(loc, None)

    res = loc.result()
    assert res["fit_failed"]
    assert math.isnan(res["fit_chi2_gain"])
    assert "zero_field_freq" not in res
    assert loc.fit_mode_estimates() is None


def test_snapshots_are_independent_of_later_fits():
    loc = _locator()
    snapshots = _run(loc, _signal(2.861))
    assert snapshots[0].current_dips() == []
    assert snapshots[-1].estimates() != snapshots[0].estimates()
    assert [s._count for s in snapshots] == list(range(1, len(snapshots) + 1))


def test_fit_belief_matches_the_smc_belief_api_the_metrics_use():
    loc = _locator()
    belief = loc.belief
    assert set(belief.uncertainty()) == set(belief.model.parameter_names())
    assert belief.drive_freq_bounds_phys == (_LO, _HI)
    assert set(belief.physical_param_bounds) >= set(belief.model.parameter_names())
    assert np.isfinite(belief.entropy())
    with pytest.raises(NotImplementedError):
        belief.update(Observation(drive_freq_unit=0.5, signal_value=1.0))
