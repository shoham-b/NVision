"""Tests for SMCMarginalDistribution.robust_uncertainty() and its use in the
SBED locator's convergence-streak gate (nvision/sim/locs/bayesian/sbed_locator.py).

Covers:
- robust_uncertainty() (weighted IQR/1.349) staying tight when a small minority
  of particles sit far from the bulk of the cloud, unlike uncertainty() (raw
  weighted std), which is quadratic in distance and gets inflated by them.
- UnitCubeSMCMarginalDistribution's unit -> physical rescaling of the robust value.
- The default AbstractMarginalDistribution.robust_uncertainty() falling back to
  uncertainty() for belief types with no override.
- SequentialBayesianExperimentDesignLocator._check_and_resample not resetting
  the convergence streak on a step whose only problem is a raw-uncertainty
  spike from a handful of SMC resample rejuvenation particles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.spectra.unit_cube import UnitCubeSignalModel
from tests.noise import gaussian_noise

# ---------------------------------------------------------------------------
# Minimal synthetic model (no gradient needed -- robust_uncertainty only
# touches particles/weights, not the signal model).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Params:
    amplitude: float
    center: float


class _Spec:
    names: ClassVar[list[str]] = ["amplitude", "center"]

    def pack_params(self, p: _Params):
        return (p.amplitude, p.center)

    def unpack_params(self, vals):
        return _Params(amplitude=float(vals[0]), center=float(vals[1]))


class _SimpleModel:
    spec = _Spec()

    def parameter_names(self):
        return ["amplitude", "center"]

    def compute(self, x: float, p: _Params) -> float:
        return float(p.amplitude * math.exp(-0.5 * ((x - p.center) / 0.1) ** 2))


def _fill_particles(belief: SMCMarginalDistribution, num_particles: int) -> None:
    """Give a skip_state_init=True belief real particle/weight arrays.

    Mirrors SMCMarginalDistribution.copy()'s own pattern for the same flag:
    skip_state_init bypasses __post_init__'s NV-center-specific epoch-candidate-
    grid construction (which needs a "frequency" bound/fixed value this test's
    generic model has no use for), on the understanding that the caller fills
    in real particle state right after construction.
    """
    d_dim = len(belief._param_names)
    belief._particles = np.zeros((num_particles, d_dim), dtype=belief._particles.dtype, order="F")
    belief._weights = (np.ones(num_particles, dtype=belief._weights.dtype) / num_particles).astype(
        belief._weights.dtype
    )


def _make_belief(num_particles: int = 200) -> SMCMarginalDistribution:
    belief = SMCMarginalDistribution(
        model=_SimpleModel(),
        parameter_bounds={"amplitude": (0.0, 1.0), "center": (0.0, 1.0)},
        num_particles=num_particles,
        skip_state_init=True,
        noise_model=gaussian_noise(),
    )
    _fill_particles(belief, num_particles)
    return belief


def _set_particles_with_outliers(belief: SMCMarginalDistribution, n_outliers: int, outlier_value: float = 0.95) -> None:
    """Put a tight cluster around 0.5 plus a small minority of far-off particles.

    Mirrors what a handful of SMC resample rejuvenation particles (freshly
    redrawn from the prior) look like relative to an otherwise-converged cloud.
    """
    n = belief.num_particles
    center_idx = belief._param_names.index("center")
    rng = np.random.default_rng(0)
    values = 0.5 + rng.normal(0.0, 0.01, size=n)
    values[:n_outliers] = outlier_value
    belief._particles[:, center_idx] = values
    belief._weights[:] = 1.0 / n


class TestRobustUncertaintyIgnoresMinorityOutliers:
    def test_raw_uncertainty_is_inflated_by_a_few_outliers(self) -> None:
        belief = _make_belief(num_particles=200)
        _set_particles_with_outliers(belief, n_outliers=3)
        raw = belief.uncertainty()["center"]
        # 3/200 particles at 0.95 vs a cluster at ~0.5 (std ~0.01) inflates the
        # weighted std well past the cluster's own spread.
        assert raw > 0.03

    def test_robust_uncertainty_stays_tight(self) -> None:
        belief = _make_belief(num_particles=200)
        _set_particles_with_outliers(belief, n_outliers=3)
        robust = belief.robust_uncertainty()["center"]
        # The middle 50% of the mass never touches the 3 outlier particles.
        assert robust < 0.03

    def test_robust_uncertainty_much_smaller_than_raw_for_same_cloud(self) -> None:
        belief = _make_belief(num_particles=200)
        _set_particles_with_outliers(belief, n_outliers=3)
        raw = belief.uncertainty()["center"]
        robust = belief.robust_uncertainty()["center"]
        assert robust < raw / 3

    def test_robust_uncertainty_reflects_genuine_balanced_bimodality(self) -> None:
        """IQR shouldn't mask a real ~50/50 split -- only a small-minority outlier."""
        belief = _make_belief(num_particles=200)
        n = belief.num_particles
        center_idx = belief._param_names.index("center")
        values = np.where(np.arange(n) < n // 2, 0.2, 0.8)
        belief._particles[:, center_idx] = values.astype(float)
        belief._weights[:] = 1.0 / n
        robust = belief.robust_uncertainty()["center"]
        raw = belief.uncertainty()["center"]
        # Both should report substantial spread -- genuine ambiguity isn't hidden.
        assert robust > 0.1
        assert raw > 0.1

    def test_other_parameters_unaffected(self) -> None:
        """robust_uncertainty is per-parameter; a spike in one axis shouldn't
        touch another particle dimension's own (tight, unperturbed) estimate."""
        belief = _make_belief(num_particles=200)
        amp_idx = belief._param_names.index("amplitude")
        rng = np.random.default_rng(1)
        belief._particles[:, amp_idx] = 0.5 + rng.normal(0.0, 0.01, size=belief.num_particles)
        _set_particles_with_outliers(belief, n_outliers=3)
        robust = belief.robust_uncertainty()
        assert robust["amplitude"] < 0.03
        assert robust["center"] < 0.03


class TestRobustUncertaintyDegenerateCases:
    def test_too_few_particles_returns_nan_not_crash(self) -> None:
        belief = _make_belief(num_particles=3)
        result = belief.robust_uncertainty()
        assert math.isnan(result["center"])
        assert math.isnan(result["amplitude"])

    def test_zero_weight_returns_nan_not_crash(self) -> None:
        belief = _make_belief(num_particles=50)
        belief._weights[:] = 0.0
        result = belief.robust_uncertainty()
        assert math.isnan(result["center"])


def _make_unit_cube_belief(phys_bounds: dict, num_particles: int = 200) -> UnitCubeSMCMarginalDistribution:
    wrapped = UnitCubeSignalModel(_SimpleModel(), phys_bounds, phys_bounds["center"])
    belief = UnitCubeSMCMarginalDistribution(
        model=wrapped,
        parameter_bounds={"amplitude": (0.0, 1.0), "center": (0.0, 1.0)},
        num_particles=num_particles,
        physical_param_bounds=phys_bounds,
        physical_x_bounds=phys_bounds["center"],
        skip_state_init=True,
        noise_model=gaussian_noise(),
    )
    _fill_particles(belief, num_particles)
    return belief


class TestUnitCubeRobustUncertaintyRescaling:
    def test_scales_by_physical_bound_width(self) -> None:
        phys_bounds = {"amplitude": (0.0, 2.0), "center": (100.0, 200.0)}
        belief = _make_unit_cube_belief(phys_bounds)
        _set_particles_with_outliers(belief, n_outliers=3)
        unit_robust = belief._robust_uncertainty_unit()["center"]
        phys_robust = belief.robust_uncertainty()["center"]
        width = phys_bounds["center"][1] - phys_bounds["center"][0]
        assert math.isclose(phys_robust, unit_robust * width, rel_tol=1e-9)

    def test_physical_robust_smaller_than_physical_raw(self) -> None:
        phys_bounds = {"amplitude": (0.0, 2.0), "center": (100.0, 200.0)}
        belief = _make_unit_cube_belief(phys_bounds)
        _set_particles_with_outliers(belief, n_outliers=3)
        assert belief.robust_uncertainty()["center"] < belief.uncertainty()["center"] / 3


class _StubBeliefWithNoRobustOverride(AbstractMarginalDistribution):
    """Minimal concrete belief that overrides nothing but the required
    abstractmethods -- exercises AbstractMarginalDistribution's own default
    _empirical_robust_uncertainty (return _empirical_uncertainty()) with
    nothing else in the class hierarchy able to shadow it."""

    def update(self, obs):
        raise NotImplementedError

    def estimates(self):
        raise NotImplementedError

    def mode_estimates(self):
        raise NotImplementedError

    def _empirical_uncertainty(self):
        return ParameterValues.from_mapping(["amplitude", "center"], {"amplitude": 0.07, "center": 0.13})

    def entropy(self):
        raise NotImplementedError

    def converged(self, threshold):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def sample(self, n):
        raise NotImplementedError

    def marginal_pdf(self, param_name, x):
        raise NotImplementedError

    def marginal_cdf(self, param_name, x):
        raise NotImplementedError

    @property
    def physical_param_bounds(self):
        return {"amplitude": (0.0, 1.0), "center": (0.0, 1.0)}


class TestDefaultRobustUncertaintyFallsBackToRaw:
    def test_belief_with_no_override_robust_equals_raw(self) -> None:
        belief = _StubBeliefWithNoRobustOverride(model=_SimpleModel())
        raw = belief.uncertainty()
        robust = belief.robust_uncertainty()
        for name in raw:
            assert robust[name] == raw[name]


# ---------------------------------------------------------------------------
# Locator-level: the convergence streak must survive a step whose only
# problem is a raw-uncertainty spike (the actual resample-rejuvenation
# scenario, not just the belief-layer math in isolation above).
# ---------------------------------------------------------------------------


class TestSbedStreakSurvivesRawUncertaintySpike:
    def _make_locator(self):
        from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
        from nvision.spectra.nv_center import NVCenterLorentzianModel

        model = NVCenterLorentzianModel()
        phys_bounds = {
            "frequency": (2.6e9, 3.1e9),
            "linewidth": (1e6, 5e6),
            "split": (3e6, 8.5e6),
            "k_np": (1.0, 5.0),
            "c_total": (0.05, 0.3),
        }
        wrapped = UnitCubeSignalModel(model, phys_bounds, phys_bounds["frequency"])
        belief = UnitCubeSMCMarginalDistribution(
            model=wrapped,
            parameter_bounds={name: (0.0, 1.0) for name in phys_bounds},
            num_particles=200,
            physical_param_bounds=phys_bounds,
            physical_x_bounds=phys_bounds["frequency"],
            noise_model=gaussian_noise(),
        )
        loc = SequentialBayesianExperimentDesignLocator(
            belief=belief,
            max_steps=200,
            noise_std=0.01,
            convergence_threshold=0.5,  # loose, so a tight (non-outlier) cloud passes
            convergence_patience_steps=5,
        )
        return loc

    def test_streak_not_reset_by_a_transient_raw_spike(self) -> None:
        loc = self._make_locator()
        belief = loc.belief
        # Plain NVCenterLorentzianModel() (no Zeeman/hyperfine, fixed frequency
        # by default) only infers linewidth and c_total.
        assert list(belief._param_names) == ["linewidth", "c_total"]
        target_idx = belief._param_names.index("linewidth")
        n = belief.num_particles

        # Tight, well-converged cloud on every free parameter.
        rng = np.random.default_rng(2)
        for name in belief._param_names:
            idx = belief._param_names.index(name)
            belief._particles[:, idx] = 0.5 + rng.normal(0.0, 0.002, size=n)
        belief._weights[:] = 1.0 / n

        # Build up a genuine streak first (tight cloud, no spike).
        for _ in range(3):
            loc._check_and_resample()
        assert loc._convergence_streak == 3
        assert not loc._is_converged

        # Now simulate exactly what a resample's rejuvenation particles look
        # like: a small minority (3/200) of the cloud redrawn far from the
        # bulk, for linewidth specifically -- everything else still tight.
        spiked = belief._particles[:, target_idx].copy()
        spiked[:3] = 0.95
        belief._particles[:, target_idx] = spiked

        # Confirm this really would have been a raw-uncertainty spike (the
        # bug this fix targets), before checking the streak survives it.
        raw_unc = belief.uncertainty()["linewidth"]
        robust_unc = belief.robust_uncertainty()["linewidth"]
        assert raw_unc > robust_unc * 3

        loc._check_and_resample()
        assert loc._convergence_streak == 4, (
            "convergence streak was reset by a transient raw-uncertainty spike "
            "that robust_uncertainty() should have filtered out"
        )
