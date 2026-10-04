"""Tests for the CRLB feasibility gate, Fisher information and the SBED step-budget backstop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from nvision.models.fisher_information import marginal_crlbs_at_budget
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import (
    NVCenterLorentzianModel,
    NVCenterVoigtModel,
    NVCenterVoigtSpectrum,
)
from tests.noise import gaussian_noise

# ---------------------------------------------------------------------------
# Minimal synthetic model with analytical gradient (for FIM tests)
# ---------------------------------------------------------------------------
# Signal: S(x; a, mu) = a * exp(-0.5 * ((x - mu) / 0.1)^2)
# Gradient: dS/da = exp(...),  dS/dmu = a * (x - mu) / 0.01 * exp(...)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GaussParams:
    amplitude: float
    center: float


class _GaussSpec:
    names: ClassVar[list[str]] = ["amplitude", "center"]

    def pack_params(self, p: _GaussParams):
        return (p.amplitude, p.center)

    def unpack_params(self, vals):
        return _GaussParams(amplitude=float(vals[0]), center=float(vals[1]))

    def unpack_samples(self, args):
        return _GaussParams(amplitude=args[0], center=args[1])


class _SimpleGaussModel:
    """Gaussian peak with analytical gradient — for FIM unit tests."""

    spec = _GaussSpec()
    _sigma = 0.1

    def parameter_names(self):
        return ["amplitude", "center"]

    def compute_from_params(self, x: float, p: _GaussParams) -> float:
        return float(p.amplitude * np.exp(-0.5 * ((x - p.center) / self._sigma) ** 2))

    def gradient(self, x: float, p: _GaussParams) -> dict[str, float]:
        z = (x - p.center) / self._sigma
        g = np.exp(-0.5 * z**2)
        return {
            "amplitude": float(g),
            "center": float(p.amplitude * z / self._sigma * g),
        }


# ---------------------------------------------------------------------------
# marginal_crlbs_at_budget
# ---------------------------------------------------------------------------


def test_marginal_crlbs_feasible_clean_signal() -> None:
    """Low noise + many steps → CRLB well below convergence threshold."""
    model = _SimpleGaussModel()
    params = _GaussParams(amplitude=0.5, center=0.5)

    crlbs = marginal_crlbs_at_budget(
        model=model,
        true_typed_params=params,
        x_lo=0.0,
        x_hi=1.0,
        noise_std=0.01,
        n_steps=500,
        n_grid=256,
    )

    assert "center" in crlbs
    assert crlbs["center"] > 0
    assert math.isfinite(crlbs["center"])
    assert crlbs["center"] < 0.01, f"CRLB too large: {crlbs['center']:.4f}"


def test_marginal_crlbs_infeasible_high_noise() -> None:
    """Very high noise + few steps → CRLB above threshold."""
    model = _SimpleGaussModel()
    params = _GaussParams(amplitude=0.5, center=0.5)

    crlbs_few = marginal_crlbs_at_budget(
        model=model,
        true_typed_params=params,
        x_lo=0.0,
        x_hi=1.0,
        noise_std=1.0,
        n_steps=5,
        n_grid=256,
    )
    crlbs_many = marginal_crlbs_at_budget(
        model=model,
        true_typed_params=params,
        x_lo=0.0,
        x_hi=1.0,
        noise_std=0.001,
        n_steps=500,
        n_grid=256,
    )

    assert crlbs_few["center"] > crlbs_many["center"], "High-noise/low-step CRLB should exceed low-noise/many-step CRLB"


def test_marginal_crlbs_scales_with_noise() -> None:
    """CRLB ∝ σ: doubling noise should roughly double the CRLB."""
    model = _SimpleGaussModel()
    params = _GaussParams(amplitude=0.5, center=0.5)
    n_steps = 200

    crlbs_low = marginal_crlbs_at_budget(
        model=model, true_typed_params=params, x_lo=0.0, x_hi=1.0, noise_std=0.01, n_steps=n_steps
    )
    crlbs_high = marginal_crlbs_at_budget(
        model=model, true_typed_params=params, x_lo=0.0, x_hi=1.0, noise_std=0.02, n_steps=n_steps
    )

    ratio = crlbs_high["center"] / crlbs_low["center"]
    assert 1.8 < ratio < 2.2, f"CRLB ratio for 2× noise = {ratio:.3f}, expected ~2.0"


def test_marginal_crlbs_empty_for_no_gradient() -> None:
    """Returns empty dict for models with no gradient method.

    NVCenterLorentzianModel now has an analytical gradient (see
    NVCenterLorentzianModel.gradient), so this uses NVCenterVoigtModel --
    still numerical-gradient-only -- to keep testing the actual no-gradient
    fallback path rather than a premise the Lorentzian gradient rollout
    invalidated.
    """
    model = NVCenterVoigtModel()  # has no .gradient
    params = NVCenterVoigtSpectrum(
        center_freq=2.87e9, homogeneous_linewidth=1e6, sigma_inhom=1e6, split=4e6, k_np=1.5, c_total=0.15
    )

    result = marginal_crlbs_at_budget(
        model=model,
        true_typed_params=params,
        x_lo=2.6e9,
        x_hi=3.1e9,
        noise_std=0.01,
        n_steps=100,
    )
    assert result == {}


# ---------------------------------------------------------------------------
# fisher_history (nvision.models.fisher_information) -- the per-step Fisher history
# feeding the UI's CRLB overlay, as opposed to marginal_crlbs_at_budget's
# upfront feasibility estimate above.
# ---------------------------------------------------------------------------


def testfisher_history_bounds_are_dicts_not_ndarrays() -> None:
    """Regression test.

    write_fisher_data's fisher_bounds_hist parameter is documented as
    list[dict[str, float]] and calls .items() on each element -- but
    single_shot_marginal_stds_from_fim returns a raw np.ndarray, and the
    plots.py call site used to append that ndarray directly. write_fisher_data
    then crashed with AttributeError: 'numpy.ndarray' object has no attribute
    'items', silently caught by _bayesian_auxiliary_entries's caller and wiping
    out every Bayesian auxiliary entry for that repeat (posterior, convergence,
    covariance ellipses, jitter -- not just Fisher, since they all share one
    try/except). This asserts the fixed shape and that write_fisher_data
    actually succeeds end to end.
    """
    from types import SimpleNamespace

    from nvision.models.fisher_information import fisher_history
    from nvision.models.observation import Observation
    from nvision.runner.plots_data import write_fisher_data

    model = _SimpleGaussModel()
    param_names = model.parameter_names()
    xs = np.linspace(0.2, 0.8, 10)
    true_params = _GaussParams(amplitude=0.5, center=0.5)
    snapshots = [
        SimpleNamespace(
            obs=Observation(x=float(x), signal_value=model.compute_from_params(float(x), true_params), noise_std=0.01),
            belief=SimpleNamespace(model=model),
        )
        for x in xs
    ]
    # estimates_hist is belief.estimates()'s contract: dict[str, float], not the
    # model's typed params object -- fisher_history converts internally.
    estimates_hist = [dict(zip(param_names, model.spec.pack_params(true_params), strict=True)) for _ in xs]
    physical_bounds = {"amplitude": (0.0, 1.0), "center": (0.0, 1.0)}

    fisher_hist, fisher_bounds_hist, fim_is_degenerate = fisher_history(
        snapshots, estimates_hist, param_names, physical_bounds
    )

    assert not fim_is_degenerate
    assert len(fisher_bounds_hist) == len(fisher_hist) == len(xs)
    for bounds in fisher_bounds_hist:
        assert isinstance(bounds, dict), f"expected dict, got {type(bounds)}"
        assert set(bounds) == set(param_names)
        assert all(math.isfinite(v) and v > 0 for v in bounds.values())

    actual_uncertainty_hist = [dict.fromkeys(param_names, 0.05) for _ in xs]
    data = write_fisher_data(fisher_bounds_hist, actual_uncertainty_hist, fisher_hist, param_names)
    assert data is not None


def testfisher_history_normalizes_across_wildly_different_scales() -> None:
    """Without per-parameter range normalization, single_shot_marginal_stds_from_fim's
    ridge dominates any Hz-scale direction and its CRLB saturates at a constant
    sqrt(1/ridge) regardless of the data (see that function's docstring). Uses
    NVCenterLorentzianModel, whose params span ~1e9 Hz (center_freq) to ~0.1-1
    (c_total) -- if normalization regresses, the center_freq CRLB collapses to the
    same fixed constant independent of how much data/noise is fed in.
    """
    from types import SimpleNamespace

    from nvision.models.fisher_information import fisher_history
    from nvision.models.observation import Observation

    model = NVCenterLorentzianModel()
    param_names = model.parameter_names()
    from nvision.spectra.nv_center import NVCenterLorentzianSpectrum

    true_params = NVCenterLorentzianSpectrum(center_freq=2.87e9, linewidth=2e6, split=4e6, k_np=1.0, c_total=0.2)
    physical_bounds = {
        "center_freq": (2.6e9, 3.1e9),
        "linewidth": (0.5e6, 5e6),
        "split": (0.0, 2e7),
        "k_np": (0.1, 10.0),
        "c_total": (0.0, 1.0),
    }
    xs = np.linspace(2.8e9, 2.94e9, 40)

    def _fisher_bounds_for_noise(noise_std: float) -> dict[str, float]:
        snapshots = [
            SimpleNamespace(
                obs=Observation(x=float(x), signal_value=model.compute(float(x), true_params), noise_std=noise_std),
                belief=SimpleNamespace(model=model),
            )
            for x in xs
        ]
        # estimates_hist is belief.estimates()'s contract: dict[str, float], not the
        # model's typed params object -- fisher_history converts internally.
        estimates_hist = [dict(zip(param_names, model.spec.pack_params(true_params), strict=True)) for _ in xs]
        _, fisher_bounds_hist, fim_is_degenerate = fisher_history(
            snapshots, estimates_hist, param_names, physical_bounds
        )
        assert not fim_is_degenerate
        return fisher_bounds_hist[-1]

    low_noise = _fisher_bounds_for_noise(0.001)
    high_noise = _fisher_bounds_for_noise(0.1)

    # A saturated (unnormalized) CRLB would be identical regardless of noise level.
    assert low_noise["linewidth"] != high_noise["linewidth"]
    assert low_noise["linewidth"] < high_noise["linewidth"]
    assert math.isfinite(low_noise["linewidth"])
    # Physically meaningful: well below the center_freq search span, not ~1000 Hz-in-
    # wrong-units or ~1e9 (a fully degenerate/uninformative bound).
    assert 0 < low_noise["linewidth"] < 5e7


# ---------------------------------------------------------------------------
# oracle_crlb_history (nvision.models.fisher_information) -- the "best any ideal
# acquisition could do" reference curve, as distinct from fisher_history's
# data-driven "how well did THIS run's actual measurements do" curve above.
# ---------------------------------------------------------------------------


def test_oracle_crlb_history_decreases_as_one_over_sqrt_n() -> None:
    """CRLB ~ 1/sqrt(N): step k's oracle bound should equal step 0's divided by sqrt(k+1)."""
    from nvision.models.fisher_information import oracle_crlb_history

    model = _SimpleGaussModel()
    param_names = model.parameter_names()
    true_params = _GaussParams(amplitude=0.5, center=0.5)
    physical_bounds = {"amplitude": (0.0, 1.0), "center": (0.0, 1.0)}

    history = oracle_crlb_history(
        n_steps=9,
        model=model,
        true_typed_params=true_params,
        x_lo=0.0,
        x_hi=1.0,
        noise_std=0.01,
        bounds=physical_bounds,
    )

    assert len(history) == 9
    for name in param_names:
        step0 = history[0][name]
        assert math.isfinite(step0)
        assert step0 > 0
        for k in (1, 3, 8):
            expected = step0 / math.sqrt(k + 1)
            assert math.isclose(history[k][name], expected, rel_tol=1e-6)


def test_oracle_crlb_history_scales_with_noise() -> None:
    """Doubling the noise std should double every oracle CRLB value (linear in sigma)."""
    from nvision.models.fisher_information import oracle_crlb_history

    model = _SimpleGaussModel()
    param_names = model.parameter_names()
    true_params = _GaussParams(amplitude=0.5, center=0.5)
    physical_bounds = {"amplitude": (0.0, 1.0), "center": (0.0, 1.0)}

    low = oracle_crlb_history(
        n_steps=3,
        model=model,
        true_typed_params=true_params,
        x_lo=0.0,
        x_hi=1.0,
        noise_std=0.01,
        bounds=physical_bounds,
    )
    high = oracle_crlb_history(
        n_steps=3,
        model=model,
        true_typed_params=true_params,
        x_lo=0.0,
        x_hi=1.0,
        noise_std=0.02,
        bounds=physical_bounds,
    )
    for name in param_names:
        assert math.isclose(high[0][name], 2.0 * low[0][name], rel_tol=1e-6)


def test_oracle_crlb_history_no_gradient_returns_empty_dicts() -> None:
    """A model with no analytical gradient and a degenerate numerical fallback

    (pack_params raising, here) should degrade to empty per-step dicts rather
    than crashing -- mirrors fisher_history's fim_i is None handling.
    """
    from nvision.models.fisher_information import oracle_crlb_history

    class _NoGradSpec:
        def pack_params(self, p):
            raise RuntimeError("no analytical or numerical gradient available")

    class _NoGradModel:
        spec = _NoGradSpec()

        def parameter_names(self):
            return ["amplitude", "center"]

    history = oracle_crlb_history(
        n_steps=2,
        model=_NoGradModel(),
        true_typed_params=object(),
        x_lo=0.0,
        x_hi=1.0,
        noise_std=0.01,
        bounds={"amplitude": (0.0, 1.0), "center": (0.0, 1.0)},
    )
    assert history == [{}, {}]


# ---------------------------------------------------------------------------
# Theory step budget
# ---------------------------------------------------------------------------


def _make_sbed_locator(max_steps: int = 500):
    """Return a minimal SBED locator over a standard NV-center belief."""
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.spectra.unit_cube import UnitCubeSignalModel

    model = NVCenterLorentzianModel()
    phys_bounds = {
        "center_freq": (2.6e9, 3.1e9),
        "linewidth": (1e6, 5e6),
        "split": (3e6, 8.5e6),
        "k_np": (1.0, 5.0),
        "c_total": (0.05, 0.3),
    }
    x_bounds = phys_bounds["center_freq"]
    wrapped_model = UnitCubeSignalModel(model, phys_bounds, x_bounds)
    param_bounds = {name: (0.0, 1.0) for name in phys_bounds}
    belief = SMCMarginalDistribution(
        model=wrapped_model,
        parameter_bounds=param_bounds,
        num_particles=50,
        physical_param_bounds=phys_bounds,
        physical_x_bounds=x_bounds,
        noise_model=gaussian_noise(),
    )
    return SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=max_steps)


def _locator_with_primary_crlb(crlb: float | None):
    """SBED locator whose primary parameter's marginal CRLB is pinned to ``crlb`` (None: no FIM yet)."""
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief

    belief = nv_center_smc_belief(noise_model=gaussian_noise(), num_particles=50, seed=0)
    locator = SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=500)
    primary = locator._primary_param
    assert primary == "zeeman_split"
    locator.belief.crlb_per_param = lambda: {} if crlb is None else {primary: crlb}
    return locator, primary


def test_primary_crlb_done_requires_tight_crlb_and_matching_uncertainty() -> None:
    """Stop only when unc < K x CRLB AND the CRLB itself is below the parameter's threshold."""
    from nvision.sim.defaults import NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR

    locator, primary = _locator_with_primary_crlb(5e4)
    threshold = locator._effective_primary_threshold()
    assert threshold > 5e4
    assert locator._primary_crlb_done({primary: 5e4}, 1.0)
    # Uncertainty still wider than the information limit allows.
    assert not locator._primary_crlb_done({primary: NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR * 5e4 * 1.01}, 1.0)

    # Near-singular FIM: the CRLB is inflated past the threshold, so unc < K x CRLB must NOT pass.
    inflated, primary = _locator_with_primary_crlb(threshold * 10)
    assert not inflated._primary_crlb_done({primary: threshold}, 1.0)


def test_primary_crlb_done_false_without_fim_or_uncertainty() -> None:
    locator, primary = _locator_with_primary_crlb(None)
    assert not locator._primary_crlb_done({primary: 1.0}, 1.0)
    locator, primary = _locator_with_primary_crlb(5e4)
    assert not locator._primary_crlb_done({}, 1.0)


def test_crlb_stop_needs_consecutive_primary_passes() -> None:
    """_is_converged is set only after `patience` consecutive passes; a failure resets the streak."""
    from nvision.models.observation import Observation

    locator = _make_sbed_locator()
    for x in np.linspace(0.05, 0.95, 12):
        locator.belief.update(Observation(x=float(x), signal_value=0.97, noise_std=0.02))
    patience = locator._convergence_patience_steps

    verdicts = iter([True] * (patience - 1) + [False] + [True] * patience)
    locator._primary_crlb_done = lambda *_: next(verdicts)
    for _ in range(patience - 1):
        locator._check_crlb_early_stop(locator.belief.uncertainty())
    assert not locator._is_converged
    locator._check_crlb_early_stop(locator.belief.uncertainty())  # the failing check
    assert locator._crlb_convergence_streak == 0
    for _ in range(patience - 1):
        locator._check_crlb_early_stop(locator.belief.uncertainty())
        assert not locator._is_converged
    locator._check_crlb_early_stop(locator.belief.uncertainty())
    assert locator._is_converged
