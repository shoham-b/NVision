"""Chunked belief updates in GenericSweepLocator must equal the one-by-one SMC update (up to resample timing)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import default_rng

from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.sim.locs.coarse import generic_sweep_locator as gsl
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
_TRUTH = NVCenterLorentzianSpectrum(
    zero_field_freq=2.861, hwhm=0.003, hyperfine_shift=0.003, k_np=2.0, contrast_total=0.3
)


def _belief(*, auto_resample: bool) -> SMCMarginalDistribution:
    unit_model = UnitCubeSignalModel(_MODEL, param_bounds_phys=dict(_BOUNDS), x_bounds_phys=(_LO, _HI))
    return SMCMarginalDistribution(
        unit_model,
        num_particles=64,
        physical_param_bounds=unit_model.param_bounds_phys,
        drive_freq_bounds_phys=(_LO, _HI),
        noise_model=gaussian_noise(),
        auto_resample=auto_resample,
        seed=11,
    )


def _locator(belief, n_grid: int = 40, passes: int = 2) -> GenericSweepLocator:
    return GenericSweepLocator(
        belief=belief,
        spacing_divisor=None,
        signal_model=belief.model,
        max_steps=n_grid,
        max_measurements=passes * n_grid,
        noise_std=_NOISE,
        domain_lo=_LO,
        domain_hi=_HI,
    )


def _drive(loc: GenericSweepLocator, seed: int = 5) -> None:
    rng = default_rng(seed)
    while not loc.done():
        x = loc.next()
        clean = float(_MODEL.compute(_LO + x * (_HI - _LO), _TRUTH))
        loc.observe(Observation(drive_freq_unit=x, signal_value=clean + rng.normal(0, _NOISE), noise_std=_NOISE))


def test_chunked_update_matches_one_by_one_without_resampling(monkeypatch):
    """With resampling off, only the order-independent weight / noise recursion remains: it must agree."""
    monkeypatch.setattr(gsl, "_BELIEF_CHUNK", 1)
    ref = _locator(_belief(auto_resample=False))
    _drive(ref)
    monkeypatch.setattr(gsl, "_BELIEF_CHUNK", 8)
    chunked = _locator(_belief(auto_resample=False))
    _drive(chunked)

    assert chunked.step_count == ref.step_count
    np.testing.assert_allclose(chunked.belief._weights, ref.belief._weights, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(chunked.belief._noise_betas, ref.belief._noise_betas, rtol=1e-4)
    for name, value in ref.belief.estimates().items():
        assert chunked.belief.estimates()[name] == pytest.approx(value, rel=1e-4)


def test_belief_is_complete_at_every_pass_end(monkeypatch):
    """A chunk is flushed at the pass boundary, where done() may stop the sweep (chunk size not dividing the pass)."""
    monkeypatch.setattr(gsl, "_BELIEF_CHUNK", 7)  # 40-point pass: 5 full chunks + a flush of 5
    belief = _belief(auto_resample=False)
    loc = _locator(belief)
    seen = []
    while not loc.done():
        x = loc.next()
        loc.observe(Observation(drive_freq_unit=x, signal_value=1.0, noise_std=_NOISE))
        if loc.step_count % loc.max_steps == 0:
            seen.append((loc.step_count, belief._step_count, len(loc._pending_obs)))
    assert seen == [(40, 40, 0), (80, 80, 0)]


def test_chunked_update_keeps_calibrated_uncertainty_with_resampling(monkeypatch):
    """With resampling on the random stream differs, but the estimate must stay near the truth and finite."""
    monkeypatch.setattr(gsl, "_BELIEF_CHUNK", 8)
    loc = _locator(_belief(auto_resample=True))
    _drive(loc)
    est = loc.belief.estimates()["zero_field_freq"]
    unc = loc.belief.uncertainty()["zero_field_freq"]
    assert np.isfinite(unc)
    assert abs(est - _TRUTH.zero_field_freq) < max(5 * unc, 0.003)
