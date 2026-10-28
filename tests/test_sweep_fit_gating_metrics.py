"""Gating warm refits on new in-window data must not move the sweep's metrics (stop step, estimate, uncertainty)."""

from __future__ import annotations

import numpy as np
import pytest

from nvision.sim.locs.coarse import generic_sweep_locator as gsl
from tests.test_sweep_fit_only import _locator, _run, _signal

_PARAM = "zero_field_freq"
_CASES = [(2.861, 0.3, 3), (2.845, 0.3, 4), (2.90, 0.15, 5), (2.83, 0.5, 6)]  # (center, contrast, seed)


def _run_with_gate(monkeypatch, min_window_obs: int, center: float, contrast: float, seed: int):
    """(locator, per-step uncertainty of the primary parameter, number of fits) for one gate setting."""
    monkeypatch.setattr(gsl, "_FIT_MIN_WINDOW_OBS", min_window_obs)
    loc = _locator()
    fits = []
    real = loc._fit_and_publish
    loc._fit_and_publish = lambda *, warm: (fits.append(warm), real(warm=warm))[1]
    snaps = _run(loc, _signal(center, contrast), seed=seed)
    unc = np.array([s.uncertainty()[_PARAM] for s in snaps])
    return loc, unc, len(fits)


@pytest.mark.parametrize(("center", "contrast", "seed"), _CASES)
def test_gating_keeps_stop_step_estimate_and_uncertainty(monkeypatch, center, contrast, seed):
    ref, _, n_ref = _run_with_gate(monkeypatch, 0, center, contrast, seed)
    gated, _, n_gated = _run_with_gate(monkeypatch, 1, center, contrast, seed)

    assert n_gated <= n_ref
    assert gated.step_count == ref.step_count  # same stop step
    r_ref, r_gated = ref.result(), gated.result()
    assert not r_ref["fit_failed"]
    assert not r_gated["fit_failed"]
    # Same data, same final fit: the estimate agrees to a small fraction of its own uncertainty.
    assert abs(r_gated[_PARAM] - r_ref[_PARAM]) < 0.1 * ref.belief.uncertainty()[_PARAM]
    assert gated.belief.uncertainty()[_PARAM] == pytest.approx(ref.belief.uncertainty()[_PARAM], rel=0.05)


def test_gating_reduces_fits_and_barely_moves_the_uncertainty_trajectory(monkeypatch):
    n_ref = n_gated = 0
    ratios = []
    for center, contrast, seed in _CASES:
        _, unc_ref, nr = _run_with_gate(monkeypatch, 0, center, contrast, seed)
        _, unc_gated, ng = _run_with_gate(monkeypatch, 1, center, contrast, seed)
        n_ref, n_gated = n_ref + nr, n_gated + ng
        ratios.append(unc_gated / unc_ref)
    assert n_gated < n_ref
    ratios = np.concatenate(ratios)
    # Between fits the uncertainty is a step function; gating only lengthens some steps, so the per-snapshot
    # uncertainty stays within a factor of the ungated one.
    assert np.median(np.abs(np.log(ratios))) < 0.05
    assert np.max(ratios) < 3.0
    assert np.min(ratios) > 1 / 3.0
