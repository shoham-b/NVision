"""Benchmarks for locator single-step and overall run performance.

Run with::

    uv run --no-sync pytest tests/test_locator_benchmarks.py -v

Or with extra repeats for stable averages::

    uv run --no-sync pytest tests/test_locator_benchmarks.py -v --benchmark-repeats=10
"""

from __future__ import annotations

import random
import statistics
import time
from typing import Any

import numpy as np
import pytest

from nvision import (
    CoreExperiment,
    MultiPeakCoreGenerator,
    NVCenterCoreGenerator,
    run_loop,
)
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.models.locator import Locator
from nvision.sim.locs.bayesian.belief_builders import (
    nv_center_belief,
    two_peak_gaussian_belief,
)
from nvision.sim.locs.bayesian.maximum_likelihood_locator import MaximumLikelihoodLocator
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator as SbedLocator
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.sim.locs.bayesian.utility_sampling_locator import UtilitySamplingLocator
from nvision.sim.locs.coarse import SimpleSweepLocator
from nvision.sim.locs.coarse.sobol_locator import SobolSweepLocator, StagedSobolSweepLocator

_BENCHMARK_REPEATS = 3
_WARMUP_STEPS = 2


def _make_experiment(generator, rng: random.Random, noise=None) -> CoreExperiment:
    true_signal = generator.generate(rng)
    x_min, x_max = None, None
    for name in true_signal.parameter_names:
        if "frequency" in name:
            x_min, x_max = true_signal.get_param_bounds(name)
            break
    assert x_min is not None
    return CoreExperiment(true_signal=true_signal, noise=noise, x_min=x_min, x_max=x_max)


def _one_peak_experiment() -> CoreExperiment:
    rng = random.Random(42)
    gen = MultiPeakCoreGenerator(x_min=0.0, x_max=1.0, count=1)
    return _make_experiment(gen, rng)


def _two_peak_experiment() -> CoreExperiment:
    rng = random.Random(43)
    gen = MultiPeakCoreGenerator(x_min=0.0, x_max=1.0, count=2)
    return _make_experiment(gen, rng)


def _nv_center_experiment() -> CoreExperiment:
    rng = random.Random(44)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    return _make_experiment(gen, rng)


def _dummy_belief(model) -> GridMarginalDistribution:
    grid = np.linspace(0.0, 1.0, 64)
    posterior = np.ones(64) / 64
    parameters = [
        GridParameter(name=name, bounds=(0.0, 1.0), grid=grid, posterior=posterior) for name in model.parameter_names()
    ]
    return GridMarginalDistribution(model=model, parameters=parameters)


def _measure(fn: Any, repeats: int = _BENCHMARK_REPEATS) -> float:
    """Return mean elapsed time in milliseconds over *repeats* calls."""
    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    return statistics.mean(times_ms)


# ---------------------------------------------------------------------------
# Single-step benchmarks
# ---------------------------------------------------------------------------


class _StepTimer:
    """Helper to warm up a locator and time one step."""

    def __init__(self, locator: Locator, experiment: CoreExperiment, rng: random.Random):
        self.locator = locator
        self.experiment = experiment
        self.rng = rng
        self._warmup()

    def _warmup(self) -> None:
        for _ in range(_WARMUP_STEPS):
            x = self.locator.next()
            obs = self.experiment.measure(x, self.rng)
            self.locator.observe(obs)

    def step(self) -> None:
        x = self.locator.next()
        obs = self.experiment.measure(x, self.rng)
        self.locator.observe(obs)


def _sweep_step_ms(locator_class: type[Locator], experiment: CoreExperiment, **config: Any) -> float:
    rng = random.Random(1)
    needs_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
    if needs_belief:
        config.setdefault("belief", _dummy_belief(experiment.true_signal.model))
        config.setdefault("signal_model", experiment.true_signal.model)
    loc = locator_class.create(**config)
    timer = _StepTimer(loc, experiment, rng)
    return _measure(timer.step)


def _bayesian_step_ms(
    locator_class: type[SequentialBayesianLocator],
    experiment: CoreExperiment,
    builder: Any,
    **extra: Any,
) -> float:
    rng = random.Random(1)
    loc = locator_class.create(
        builder=builder,
        parameter_bounds=None,
        max_steps=12,
        initial_sweep_steps=4,
        noise_std=0.02,
        n_grid_freq=16,
        n_grid_width=8,
        n_grid_depth=8,
        n_grid_background=4,
        **extra,
    )
    timer = _StepTimer(loc, experiment, rng)
    return _measure(timer.step)


@pytest.mark.benchmark
class TestSingleStepOnePeak:
    """Single-step latency on a simple one-peak Gaussian."""

    def test_simple_sweep(self):
        exp = _one_peak_experiment()
        ms = _sweep_step_ms(SimpleSweepLocator, exp, max_steps=30, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"SimpleSweepLocator one-step: {ms:.3f} ms")

    def test_sobol_sweep(self):
        exp = _one_peak_experiment()
        ms = _sweep_step_ms(SobolSweepLocator, exp, max_steps=30, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"SobolSweepLocator one-step: {ms:.3f} ms")

    def test_staged_sobol(self):
        exp = _one_peak_experiment()
        ms = _sweep_step_ms(StagedSobolSweepLocator, exp, max_steps=30, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"StagedSobolSweepLocator one-step: {ms:.3f} ms")

    def test_maximum_likelihood(self):
        exp = _one_peak_experiment()
        ms = _bayesian_step_ms(
            MaximumLikelihoodLocator,
            exp,
            builder=two_peak_gaussian_belief,
            exploration_rate=8.0,
        )
        print(f"MaximumLikelihoodLocator one-step: {ms:.3f} ms")

    def test_utility_sampling(self):
        exp = _one_peak_experiment()
        ms = _bayesian_step_ms(
            UtilitySamplingLocator,
            exp,
            builder=two_peak_gaussian_belief,
            pickiness=4.0,
            n_mc_samples=16,
            n_candidates=16,
        )
        print(f"UtilitySamplingLocator one-step: {ms:.3f} ms")

    @pytest.mark.skip(reason="SBEDLocator requires SMC belief, not available in benchmarks")
    def test_sbed(self):
        exp = _one_peak_experiment()
        ms = _bayesian_step_ms(
            SbedLocator,
            exp,
            builder=two_peak_gaussian_belief,
            n_mc_samples=8,
            n_candidates=8,
        )
        print(f"SbedLocator one-step: {ms:.3f} ms")


# ---------------------------------------------------------------------------
# Overall run benchmarks
# ---------------------------------------------------------------------------


def _overall_run_ms(
    locator_class: type[Locator], experiment: CoreExperiment, max_steps: int = 20, **config: Any
) -> float:
    rng = random.Random(2)
    needs_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
    if needs_belief:
        config.setdefault("belief", _dummy_belief(experiment.true_signal.model))
        config.setdefault("signal_model", experiment.true_signal.model)
    return _measure(lambda: list(run_loop(locator_class, experiment, rng, max_steps=max_steps, **config)))


def _overall_bayesian_ms(
    locator_class: type[SequentialBayesianLocator],
    experiment: CoreExperiment,
    builder: Any,
    max_steps: int = 20,
    **extra: Any,
) -> float:
    rng = random.Random(2)
    return _measure(
        lambda: list(
            run_loop(
                locator_class,
                experiment,
                rng,
                max_steps=max_steps,
                builder=builder,
                parameter_bounds=None,
                initial_sweep_steps=5,
                noise_std=0.02,
                **extra,
            )
        )
    )


@pytest.mark.benchmark
class TestOverallOnePeak:
    """Full-run latency on a simple one-peak Gaussian (max_steps=20)."""

    def test_simple_sweep(self):
        exp = _one_peak_experiment()
        ms = _overall_run_ms(SimpleSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"SimpleSweepLocator overall: {ms:.1f} ms")

    def test_sobol_sweep(self):
        exp = _one_peak_experiment()
        ms = _overall_run_ms(SobolSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"SobolSweepLocator overall: {ms:.1f} ms")

    def test_staged_sobol(self):
        exp = _one_peak_experiment()
        ms = _overall_run_ms(StagedSobolSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"StagedSobolSweepLocator overall: {ms:.1f} ms")

    def test_maximum_likelihood(self):
        exp = _one_peak_experiment()
        ms = _overall_bayesian_ms(
            MaximumLikelihoodLocator,
            exp,
            builder=two_peak_gaussian_belief,
            max_steps=20,
            exploration_rate=8.0,
        )
        print(f"MaximumLikelihoodLocator overall: {ms:.1f} ms")

    def test_utility_sampling(self):
        exp = _one_peak_experiment()
        ms = _overall_bayesian_ms(
            UtilitySamplingLocator,
            exp,
            builder=two_peak_gaussian_belief,
            max_steps=20,
            pickiness=4.0,
            n_mc_samples=16,
            n_candidates=16,
        )
        print(f"UtilitySamplingLocator overall: {ms:.1f} ms")

    @pytest.mark.skip(reason="SBEDLocator requires SMC belief, not available in benchmarks")
    def test_sbed(self):
        exp = _one_peak_experiment()
        ms = _overall_bayesian_ms(
            SbedLocator,
            exp,
            builder=two_peak_gaussian_belief,
            max_steps=12,
        )
        print(f"SbedLocator overall: {ms:.1f} ms")


@pytest.mark.benchmark
class TestOverallTwoPeak:
    """Full-run latency on a two-peak Gaussian (max_steps=20)."""

    def test_simple_sweep(self):
        exp = _two_peak_experiment()
        ms = _overall_run_ms(SimpleSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"SimpleSweepLocator two-peak overall: {ms:.1f} ms")

    def test_staged_sobol(self):
        exp = _two_peak_experiment()
        ms = _overall_run_ms(StagedSobolSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"StagedSobolSweepLocator two-peak overall: {ms:.1f} ms")

    def test_maximum_likelihood(self):
        exp = _two_peak_experiment()
        ms = _overall_bayesian_ms(
            MaximumLikelihoodLocator,
            exp,
            builder=two_peak_gaussian_belief,
            max_steps=20,
            exploration_rate=8.0,
        )
        print(f"MaximumLikelihoodLocator two-peak overall: {ms:.1f} ms")


@pytest.mark.benchmark
class TestOverallNVCenter:
    """Full-run latency on an NV-center Lorentzian (max_steps=20)."""

    def test_simple_sweep(self):
        exp = _nv_center_experiment()
        ms = _overall_run_ms(SimpleSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"SimpleSweepLocator NV-center overall: {ms:.1f} ms")

    def test_staged_sobol(self):
        exp = _nv_center_experiment()
        ms = _overall_run_ms(StagedSobolSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)
        print(f"StagedSobolSweepLocator NV-center overall: {ms:.1f} ms")

    def test_maximum_likelihood(self):
        exp = _nv_center_experiment()
        ms = _overall_bayesian_ms(
            MaximumLikelihoodLocator,
            exp,
            builder=nv_center_belief,
            max_steps=20,
            exploration_rate=8.0,
            n_grid_freq=64,
            n_grid_linewidth=16,
            n_grid_split=16,
            n_grid_k_np=8,
            n_grid_depth=16,
            n_grid_background=8,
        )
        print(f"MaximumLikelihoodLocator NV-center overall: {ms:.1f} ms")
