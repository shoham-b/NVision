"""Tests for multi-shot (batch) measurements per frequency.

Covers the sufficient-statistic aggregation (``aggregate_shots``), the batched
``CoreExperiment.measure`` path, the Rao-Blackwell noise-posterior update that
consumes the within-batch variance, and the SBED locator using the batch-mean
noise for EIG scoring.
"""

from __future__ import annotations

import math
import random

import numpy as np

from nvision.models.experiment import CoreExperiment
from nvision.models.measurement_noise import DEFAULT_MEASUREMENT_NOISE_STD
from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.models.observation import Observation, aggregate_shots
from nvision.noises.over_frequency.gaussian_noise import OverFrequencyGaussianNoise


# --------------------------------------------------------------------------- #
# aggregate_shots
# --------------------------------------------------------------------------- #
def test_aggregate_shots_identical_shots():
    """Degenerate batch (all shots equal): mean is exact, empirical std is zero."""
    obs = aggregate_shots(2.75e9, np.full(5, 0.7), prior_noise_std=0.05)
    assert obs.signal_value == 0.7
    assert obs.n_shots == 5
    # s == 0 -> noise_std falls back to prior/sqrt(k); sample_var is still 0.0 (k >= 2)
    assert obs.noise_std == 0.05 / math.sqrt(5)
    assert obs.sample_var == 0.0


def test_aggregate_shots_known_variance():
    """Batch mean has precision s/sqrt(k) and sample_var matches np.var(ddof=1)."""
    ys = np.array([0.40, 0.50, 0.60, 0.55, 0.45])
    obs = aggregate_shots(2.75e9, ys, prior_noise_std=0.05)
    s = float(np.std(ys, ddof=1))
    assert obs.signal_value == float(np.mean(ys))
    assert obs.n_shots == 5
    assert obs.sample_var == np.float64(s * s)
    assert obs.noise_std == np.float64(s / math.sqrt(5))


def test_aggregate_shots_single_shot():
    """k == 1 reproduces a single-measurement Observation (no variance estimable)."""
    obs = aggregate_shots(2.75e9, np.array([0.3]), prior_noise_std=0.05)
    assert obs.signal_value == 0.3
    assert obs.n_shots == 1
    assert obs.sample_var is None
    assert obs.noise_std == 0.05  # prior / sqrt(1)


# --------------------------------------------------------------------------- #
# CoreExperiment.measure
# --------------------------------------------------------------------------- #
def test_measure_n_shots_1_identity():
    """n_shots=1 with no noise is value-identical to a single clean measurement."""
    exp = CoreExperiment(true_signal=lambda x: 0.7, noise=None, x_min=2.7e9, x_max=2.8e9)
    obs = exp.measure(0.5, random.Random(0), n_shots=1)
    assert obs.x == 0.5
    assert obs.signal_value == 0.7
    assert obs.noise_std == DEFAULT_MEASUREMENT_NOISE_STD
    assert obs.n_shots == 1
    assert obs.sample_var is None


def test_measure_batch_produces_variance():
    """A stochastic batch yields n_shots=k, a positive sample_var, and noise_std = s/sqrt(k)."""
    ofn = CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(std=0.1)])
    noise = CompositeNoise(over_frequency_noise=ofn)
    exp = CoreExperiment(true_signal=lambda x: 1.0, noise=noise, x_min=2.7e9, x_max=2.8e9)

    obs = exp.measure(0.5, random.Random(0), n_shots=16)
    assert obs.n_shots == 16
    assert obs.sample_var is not None and obs.sample_var > 0.0
    # noise_std must equal the batch-mean precision s/sqrt(k)
    assert obs.noise_std == np.float64(math.sqrt(obs.sample_var) / math.sqrt(16))
    # The empirical spread should be in the right ballpark of the injected sigma
    assert 0.03 < math.sqrt(obs.sample_var) < 0.3


# --------------------------------------------------------------------------- #
# Rao-Blackwell noise-posterior update
# --------------------------------------------------------------------------- #
def _make_rb_belief(seed: int = 0):
    """Minimal real SMC belief with the Rao-Blackwell noise path enabled."""
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.spectra.gaussian import GaussianModel
    from nvision.spectra.noise_model import GaussianNoiseSignalModel

    class _NoCandidateSMC(SMCMarginalDistribution):
        # The base epoch-grid needs a 'linewidth' estimate the GaussianModel
        # doesn't expose; a no-op keeps construction cheap for the update test.
        def _generate_epoch_candidates(self) -> None:
            self._current_candidates = np.linspace(0.0, 1.0, 10).astype(np.float32)

    model = GaussianModel()
    model.signal_min_span = lambda w: 1e5
    noise_model = GaussianNoiseSignalModel({"noise_sigma": (0.01, 0.2)})
    bounds = {
        "frequency": (2.7e9, 2.8e9),
        "sigma": (1e6, 10e6),
        "dip_depth": (0.0, 1.0),
        "background": (0.0, 1.0),
    }
    np.random.seed(seed)
    belief = _NoCandidateSMC(
        model=model,
        parameter_bounds=bounds,
        num_particles=500,
        noise_model=noise_model,
        auto_resample=False,
    )
    assert belief._use_rao_blackwell_noise
    return belief


def test_rb_batch_tightens_noise_posterior():
    """A k-shot batch updates the Inverse-Gamma posterior more than a single shot."""
    ys = np.array([0.40, 0.50, 0.60, 0.55, 0.45])
    k = len(ys)
    batch = aggregate_shots(2.75e9, ys, prior_noise_std=0.05)
    single = Observation(x=2.75e9, signal_value=float(np.mean(ys)), noise_std=0.05)

    b_batch = _make_rb_belief()
    b_single = _make_rb_belief()
    a0 = float(b_batch._noise_alphas.mean())
    disc = b_batch.noise_discount_factor

    b_batch.update(batch)
    b_single.update(single)

    # alpha gains 0.5 (between-batch) + 0.5*(k-1) (within-batch) on top of discount.
    expected_batch_alpha = a0 * disc + 0.5 + 0.5 * (k - 1)
    expected_single_alpha = a0 * disc + 0.5
    assert math.isclose(float(b_batch._noise_alphas.mean()), expected_batch_alpha, rel_tol=1e-5)
    assert math.isclose(float(b_single._noise_alphas.mean()), expected_single_alpha, rel_tol=1e-5)

    # The batch carries strictly more evidence (larger beta increment via variance).
    assert float(b_batch._noise_betas.mean()) > float(b_single._noise_betas.mean())


def test_rb_single_shot_update_unchanged():
    """n_shots=1 (or absent) leaves the RB update numerically identical to before."""
    plain = Observation(x=2.75e9, signal_value=0.42, noise_std=0.05)
    with_default = Observation(x=2.75e9, signal_value=0.42, noise_std=0.05, n_shots=1)

    b_plain = _make_rb_belief()
    b_default = _make_rb_belief()
    b_plain.update(plain)
    b_default.update(with_default)

    assert np.allclose(b_plain._noise_alphas, b_default._noise_alphas)
    assert np.allclose(b_plain._noise_betas, b_default._noise_betas)
    assert np.allclose(b_plain._weights, b_default._weights)


# --------------------------------------------------------------------------- #
# SBED locator: EIG uses the batch-mean noise
# --------------------------------------------------------------------------- #
class _FakeModel:
    def parameter_names(self):
        return ["frequency"]


class _FakeBelief:
    """Just enough surface for SBED construction and _eig_acquire."""

    def __init__(self):
        self.model = _FakeModel()
        self.physical_param_bounds = {"frequency": (2.7e9, 2.8e9)}
        self.parameter_bounds = self.physical_param_bounds
        self.auto_resample = True
        self.last_noise_std = None

    def get_candidates(self):
        return np.linspace(2.7e9, 2.8e9, 64)

    def select_max_information_gain(self, candidates, n, noise_std=None):
        self.last_noise_std = noise_std
        return np.array([candidates[len(candidates) // 2]])


def _make_sbed(fake_belief, prior_noise=0.02):
    from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator

    return SequentialBayesianExperimentDesignLocator(
        fake_belief, max_steps=10, noise_std=prior_noise
    )


def test_eig_uses_constructor_prior_before_any_batch():
    fake = _FakeBelief()
    loc = _make_sbed(fake, prior_noise=0.02)
    assert loc._empirical_batch_noise_std is None
    loc._eig_acquire()
    assert fake.last_noise_std == 0.02


def test_eig_uses_empirical_batch_noise_once_available():
    fake = _FakeBelief()
    loc = _make_sbed(fake, prior_noise=0.02)
    loc._empirical_batch_noise_std = 0.005
    loc._eig_acquire()
    assert fake.last_noise_std == 0.005


# --------------------------------------------------------------------------- #
# run_loop wiring: n_shots reaches experiment.measure
# --------------------------------------------------------------------------- #
class _FakeLocator:
    """Minimal Locator stand-in that records every Observation it receives."""

    REQUIRES_BELIEF = False

    def __init__(self, max_calls: int = 3):
        self.max_calls = max_calls
        self.calls = 0
        self.received: list[Observation] = []

    @classmethod
    def create(cls, **config):
        return cls(**{k: v for k, v in config.items() if k == "max_calls"})

    def next(self) -> float:
        self.calls += 1
        return 0.5

    def observe(self, obs: Observation) -> None:
        self.received.append(obs)

    def done(self) -> bool:
        return self.calls >= self.max_calls

    def result(self) -> dict:
        return {}


def test_run_loop_passes_n_shots_to_measure():
    from nvision.runner.executor import run_loop

    ofn = CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(std=0.1)])
    noise = CompositeNoise(over_frequency_noise=ofn)
    exp = CoreExperiment(true_signal=lambda x: 1.0, noise=noise, x_min=2.7e9, x_max=2.8e9)

    locators = list(run_loop(_FakeLocator, exp, random.Random(0), n_shots=8))
    obs = locators[-1].received
    assert len(obs) == 3
    assert all(o.n_shots == 8 for o in obs)
    assert all(o.sample_var is not None for o in obs)


def test_run_loop_default_n_shots_is_single_shot():
    from nvision.runner.executor import run_loop

    exp = CoreExperiment(true_signal=lambda x: 1.0, noise=None, x_min=2.7e9, x_max=2.8e9)
    locators = list(run_loop(_FakeLocator, exp, random.Random(0)))
    obs = locators[-1].received
    assert all(o.n_shots == 1 for o in obs)
