"""Candidate racing in GenericSweepLocator._run_curve_fit_candidates: dedupe + noise-floor early stop."""

import numpy as np
import scipy.optimize as so

from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator

XS = np.linspace(0.0, 1.0, 50)  # shape (n_pts,)
TRUE_SLOPE = 2.0
NOISE_STD = 0.01
YS = TRUE_SLOPE * XS + np.random.default_rng(0).normal(0.0, NOISE_STD, XS.shape)


def _curve_fn(xs: np.ndarray, a: float) -> np.ndarray:
    return a * xs


def _make_p0(freq, _half, _hf) -> list[float]:
    return [float(freq)]  # the "frequency" candidate component is the slope start


def _run(candidates, monkeypatch, **kwargs):
    calls = {"n": 0}
    real = so.curve_fit

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(so, "curve_fit", counting)
    popt, _ = GenericSweepLocator._run_curve_fit_candidates(
        _curve_fn, XS, YS, candidates, _make_p0, [-10.0], [10.0], 1e-10, **kwargs
    )
    return popt, calls["n"]


def test_duplicate_starts_are_fit_once(monkeypatch):
    """Candidates that resolve to the same start vector must not be fit again."""
    cands = [(1.0, None, None)] * 4 + [(3.0, None, None)] * 3
    _, n_fits = _run(cands, monkeypatch)
    assert n_fits == 2


def test_early_stop_ends_race_at_noise_floor(monkeypatch):
    """A start that reaches reduced chi2 <= threshold ends the race; without it every distinct start runs."""
    cands = [(1.0, None, None), (3.0, None, None), (-2.0, None, None)]

    popt_all, n_all = _run(cands, monkeypatch)
    assert n_all == 3

    popt_es, n_es = _run(cands, monkeypatch, early_stop_redchi2=1.5, data_noise_std=NOISE_STD)
    assert n_es == 1
    np.testing.assert_allclose(popt_es, popt_all, rtol=1e-6)


def test_physical_priors_come_from_parameter_bounds_priors():
    """The sweep's own belief has no priors; the generator's `_priors` arrive via parameter_bounds."""
    from types import SimpleNamespace

    loc = GenericSweepLocator.__new__(GenericSweepLocator)
    loc.belief = SimpleNamespace()  # flat grid belief: no `priors` attribute
    loc._parameter_bounds = {
        "homogeneous_linewidth": (1e5, 3e6),
        "_priors": {
            "homogeneous_linewidth": (5e5, 2e5),
            "sigma_inhom": (4e5, 1e5),
            "frequency": ("sin^2", 1.0),  # coarse non-Gaussian shape: never a Gaussian prior
            "k_np": (2.0, 0.5),
        },
    }
    bounds = {"homogeneous_linewidth": (1e5, 3e6), "sigma_inhom": (0.0, 1.2e6), "k_np": (1.0, 5.0)}
    names = ["frequency", "homogeneous_linewidth", "sigma_inhom", "k_np"]

    map_priors = loc._resolve_physical_priors(names, "frequency", bounds)
    assert map_priors == {"homogeneous_linewidth": (5e5, 2e5), "sigma_inhom": (4e5, 1e5)}

    all_priors = loc._resolve_physical_priors(names, "frequency", bounds, names=frozenset(names))
    assert all_priors["k_np"] == (2.0, 0.5)
    assert "frequency" not in all_priors


def test_early_stop_does_not_trigger_above_floor(monkeypatch):
    """If no start reaches the floor, all distinct starts are still raced."""
    cands = [(1.0, None, None), (3.0, None, None)]
    # data_noise_std far below the real scatter -> reduced chi2 huge -> never at the floor
    _, n_fits = _run(cands, monkeypatch, early_stop_redchi2=1.5, data_noise_std=NOISE_STD / 100.0)
    assert n_fits == 2
