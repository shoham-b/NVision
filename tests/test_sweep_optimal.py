"""The optimal plain-sweep step k*: float spacing, the Fisher ripple of a grid, and the k* scan."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.noises import OverFrequencyGaussianNoise
from nvision.sim.locs import sweep_optimal as so
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel

_MAX_MEASUREMENTS = 2500


def _belief(sigma: float = 0.01):
    noise = CompositeNoise(
        over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma)])
    ).to_noise_signal_model()
    return nv_center_smc_belief(None, noise_model=noise, num_particles=50)


def _optimal(sigma: float) -> so.OptimalSweepSpacing:
    belief = _belief(sigma)
    lo, hi = belief.drive_freq_bounds_phys
    return so.optimal_sweep_spacing(
        belief,
        noise_sigma_phys=sigma,
        lo_phys=lo,
        hi_phys=hi,
        max_steps=_MAX_MEASUREMENTS,
        max_measurements=_MAX_MEASUREMENTS,
    )


def _locator(**kwargs) -> GenericSweepLocator:
    belief = _belief()
    lo, hi = belief.drive_freq_bounds_phys
    return GenericSweepLocator(
        belief=belief,
        signal_model=belief.model,
        max_steps=_MAX_MEASUREMENTS,
        domain_lo=lo,
        domain_hi=hi,
        **kwargs,
    )


def test_float_spacing_divisor_is_accepted():
    loc = _locator(spacing_divisor=2.5)
    assert loc.spacing_divisor == 2.5
    # A fractional k gives a grid between the neighbouring integers' sizes.
    assert _locator(spacing_divisor=2).max_steps < loc.max_steps < _locator(spacing_divisor=3).max_steps


@pytest.mark.parametrize("bad", [0, -1.5, math.inf, math.nan, "bogus"])
def test_non_positive_or_non_finite_spacing_divisor_is_rejected(bad):
    with pytest.raises(ValueError, match="spacing_divisor"):
        _locator(spacing_divisor=bad)


def test_optimal_spacing_needs_the_true_noise_sigma():
    with pytest.raises(ValueError, match="noise_std"):
        _locator(spacing_divisor=so.OPTIMAL_SPACING)


def test_optimal_locator_uses_and_reports_k_star():
    loc = _locator(spacing_divisor=so.OPTIMAL_SPACING, noise_std=0.01)
    assert loc.optimal_spacing is not None
    assert loc.spacing_divisor == loc.optimal_spacing.k_opt
    assert loc.max_steps == loc.optimal_spacing.n_points
    assert so.K_MIN <= loc.spacing_divisor <= so.K_MAX


def test_fisher_ripple_of_a_grid_matches_the_lorentzian_table():
    """Position information of a uniform grid vs a dense grid, worst/best dip offset (docs/locator_comparison.md)."""
    model = NVCenterLorentzianModel(with_zeeman_splitting=False, with_fixed_zero_field_freq=False)
    fwhm, f0 = 8e6, 2.87e9
    params = {"zero_field_freq": f0, "hwhm": fwhm / 2, "contrast_total": 0.25}
    bounds = {"zero_field_freq": (f0 - 1e8, f0 + 1e8), "hwhm": (1e6, 1e7), "contrast_total": (0.05, 0.5)}
    lo, hi = f0 - 10 * fwhm, f0 + 10 * fwhm
    xs = so.uniform_grid_phys(lo - 3 * fwhm, hi + 3 * fwhm, fwhm / 100)
    sens = so.tabulate_sensitivity(model, params, bounds, xs)
    i = sens.names.index("zero_field_freq")
    continuum = np.trapezoid(sens.grads[:, i] ** 2, xs)

    def worst_best(k: float) -> tuple[float, float]:
        grid = so.grid_points_phys(lo, hi, fwhm, k)
        step = grid[1] - grid[0]
        ratios = [
            so.pass_fisher_phys(sens, grid, float(o), 1.0)[i, i] * step / continuum for o in np.arange(16) * step / 16
        ]
        return min(ratios), max(ratios)

    assert worst_best(2.0) == pytest.approx((0.71, 1.28), abs=0.03)
    assert worst_best(3.0) == pytest.approx((0.96, 1.04), abs=0.03)
    assert worst_best(4.0) == pytest.approx((1.00, 1.00), abs=0.02)
    # Stepping a whole FWHM (k = 1) can lose most of the information when the dip falls between grid points.
    assert worst_best(1.0)[0] < 0.2


def test_optimal_k_is_finite_and_inside_the_scan_for_the_default_task():
    res = _optimal(0.01)
    assert math.isfinite(res.k_opt)
    assert so.K_MIN <= res.k_opt <= so.K_MAX
    assert res.n_points >= 2
    assert res.passes_pred >= 1.0
    assert res.cost_pred == pytest.approx(res.passes_pred * res.n_points)
    assert res.within_budget


def test_optimal_is_deterministic():
    assert _optimal(0.01) == _optimal(0.01)


def test_higher_snr_needs_fewer_measurements():
    """Quieter noise never needs more measurements, and at high SNR one pass already suffices."""
    quiet, medium = _optimal(0.001), _optimal(0.01)
    assert quiet.cost_pred < medium.cost_pred
    assert quiet.passes_pred == 1.0
    # At high SNR the coarsest grid that still detects the dip wins: fewer points than the reference 3 per FWHM.
    reference = _locator(spacing_divisor=3).max_steps
    assert quiet.n_points < reference


def test_optimal_never_costs_more_than_the_reference_grid(monkeypatch):
    """k* is the argmin over a scan that contains the reference's k = 3 (3.0 is on the scan grid)."""
    sigma = 0.01
    belief = _belief(sigma)
    lo, hi = belief.drive_freq_bounds_phys
    kwargs = {
        "noise_sigma_phys": sigma,
        "lo_phys": lo,
        "hi_phys": hi,
        "max_steps": _MAX_MEASUREMENTS,
        "max_measurements": _MAX_MEASUREMENTS,
    }
    res = so.optimal_sweep_spacing(belief, **kwargs)
    # Cost of the reference k = 3 by the same objective, with the scan narrowed to that single value.
    monkeypatch.setattr(so, "K_MIN", 3.0)
    monkeypatch.setattr(so, "K_MAX", 3.0)
    ref = so.optimal_sweep_spacing(belief, **kwargs)
    assert ref.k_opt == pytest.approx(3.0)
    assert res.cost_pred <= ref.cost_pred


def test_nonpositive_sigma_is_rejected():
    belief = _belief()
    lo, hi = belief.drive_freq_bounds_phys
    for sigma in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="sigma"):
            so.optimal_sweep_spacing(
                belief, noise_sigma_phys=sigma, lo_phys=lo, hi_phys=hi, max_measurements=_MAX_MEASUREMENTS
            )


def test_unreachable_budget_minimises_the_error_at_the_cap():
    """When no k reaches the stop rule, k* spends (nearly) the whole budget and is not just the finest grid."""
    res = _optimal(0.05)
    assert not res.within_budget
    assert math.isfinite(res.error_ratio_at_cap)
    assert res.error_ratio_at_cap > 1.0
    assert res.passes_pred * res.n_points >= 0.9 * _MAX_MEASUREMENTS
    assert res.passes_pred * res.n_points <= _MAX_MEASUREMENTS
    assert res.k_opt < so.K_MAX - 5 * so.K_STEP
