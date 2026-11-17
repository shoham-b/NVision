"""Truth-free stuck-belief check (nvision/metrics/fit_gap.py) and the SMC accessors it reads."""

from typing import ClassVar

import numpy as np
import pytest

from nvision.metrics.fit_gap import FIT_GAP_METRIC_KEYS, STUCK_FIT_GAP, compute_fit_gap, fit_starts_unit
from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise

_PHYS_SPAN = 1.0e7  # physical width (Hz) of every unit-cube parameter in the fake belief


class _LorentzianBelief:
    """Minimal particle belief over (center, hwhm, depth) on the unit cube, with a Lorentzian dip model.

    ``mean``: the belief's weighted-mean estimate, shape (3,); ``particles``: shape (n_particles, 3);
    ``xs``/``ys``: observations, shape (n_observations,).
    """

    parameter_names: ClassVar[list[str]] = ["center", "hwhm", "depth"]

    def __init__(self, mean: np.ndarray, particles: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
        self._mean, self._particles, self._xs, self._ys = mean, particles, xs, ys
        self.physical_param_bounds = {name: (0.0, _PHYS_SPAN) for name in self.parameter_names}

    @staticmethod
    def model(xs: np.ndarray, params: np.ndarray) -> np.ndarray:
        """shape: (n_observations, n_sets)"""
        center, hwhm, depth = (params[:, j][None, :] for j in range(3))
        hwhm = 0.01 + 0.2 * hwhm  # keep the width positive on the whole unit cube
        return 1.0 - depth * hwhm**2 / ((xs[:, None] - center) ** 2 + hwhm**2)

    def predict_at_observations(self, params_unit: np.ndarray) -> np.ndarray:
        return self.model(self._xs, np.asarray(params_unit, dtype=np.float64))

    def observation_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self._xs, self._ys

    def observation_shots(self) -> np.ndarray:
        return np.ones_like(self._xs)

    def particles_unit(self) -> np.ndarray:
        return self._particles

    def estimates_unit(self) -> dict[str, float]:
        return dict(zip(self.parameter_names, self._mean.tolist(), strict=True))


def _data(truth: np.ndarray, n: int = 400, sigma: float = 0.01, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 1.0, n)
    ys = _LorentzianBelief.model(xs, truth[None, :])[:, 0] + rng.normal(0.0, sigma, n)
    return xs, ys


def _belief_at(mean: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> _LorentzianBelief:
    rng = np.random.default_rng(1)
    particles = np.clip(mean[None, :] + rng.normal(0.0, 0.002, (200, 3)), 0.0, 1.0)
    return _LorentzianBelief(mean, particles, xs, ys)


TRUTH = np.array([0.5, 0.3, 0.4])


def test_converged_belief_scores_low():
    xs, ys = _data(TRUTH)
    out = compute_fit_gap(_belief_at(TRUTH, xs, ys), "center")
    assert set(out) == set(FIT_GAP_METRIC_KEYS)
    assert 0.0 <= out["belief_fit_gap"] < STUCK_FIT_GAP
    assert out["belief_fit_gap_shift_phys"] < 0.01 * _PHYS_SPAN


def test_wrong_width_belief_is_flagged_and_shift_reported():
    xs, ys = _data(TRUTH)
    stuck = np.array([0.52, 0.05, 0.4])  # wrong width and a shifted center
    out = compute_fit_gap(_belief_at(stuck, xs, ys), "center")
    assert out["belief_fit_gap"] > STUCK_FIT_GAP
    assert out["belief_fit_gap_shift_phys"] == pytest.approx(0.02 * _PHYS_SPAN, rel=0.25)


def test_not_applicable_returns_none():
    xs, ys = _data(TRUTH)
    assert compute_fit_gap(object(), "center") == dict.fromkeys(FIT_GAP_METRIC_KEYS)
    assert compute_fit_gap(_belief_at(TRUTH, xs, ys), None) == dict.fromkeys(FIT_GAP_METRIC_KEYS)
    assert compute_fit_gap(_belief_at(TRUTH, xs[:3], ys[:3]), "center") == dict.fromkeys(FIT_GAP_METRIC_KEYS)


def test_unknown_primary_raises():
    xs, ys = _data(TRUTH)
    with pytest.raises(ValueError, match="not a belief parameter"):
        compute_fit_gap(_belief_at(TRUTH, xs, ys), "zeeman_shift")


def test_starts_ladder_leaves_primary_at_mean():
    mean = np.array([0.5, 0.3, 0.4])
    starts = fit_starts_unit(mean, mean.copy(), primary_index=0)
    assert starts.shape == (2 + 2 * 4, 3)
    assert np.all(starts[:, 0] == 0.5)
    assert {0.0, 0.05, 0.15, 0.4} <= set(starts[:, 1].tolist())


def test_smc_accessors_shapes_and_validation():
    belief = nv_center_smc_belief(num_particles=100, noise_model=gaussian_noise(), seed=2)
    for i, x in enumerate((0.2, 0.5, 0.8)):
        belief.update(Observation(drive_freq_unit=x, signal_value=0.95 + 0.01 * i, noise_std=0.02))
    k = len(belief.parameter_names)
    assert belief.particles_unit().shape == (100, k)
    assert belief.observation_shots().shape == (3,)
    mean = np.array([belief.estimates_unit()[n] for n in belief.parameter_names])
    assert belief.predict_at_observations(np.vstack([mean, mean])).shape == (3, 2)
    with pytest.raises(ValueError, match="params_unit must have shape"):
        belief.predict_at_observations(mean)
    out = compute_fit_gap(belief, "zeeman_shift" if "zeeman_shift" in belief.parameter_names else None)
    assert set(out) == set(FIT_GAP_METRIC_KEYS)
