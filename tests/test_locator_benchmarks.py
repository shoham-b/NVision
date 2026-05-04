"""Benchmarks for locator single-step and overall run performance.

Run with::

    uv run --no-sync pytest tests/test_locator_benchmarks.py -v

Or with extra repeats for stable averages::

    uv run --no-sync pytest tests/test_locator_benchmarks.py -v --benchmark-repeats=10
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from nvision import (
    CoreExperiment,
    NVCenterCoreGenerator,
    run_loop,
)
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.sim.locs.bayesian.belief_builders import (
    nv_center_belief,
    nv_center_smc_belief,
)
from nvision.sim.locs.bayesian.maximum_likelihood_locator import MaximumLikelihoodLocator
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator as SbedLocator
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.sim.locs.bayesian.utility_sampling_locator import UtilitySamplingLocator
from nvision.sim.locs.coarse import SimpleSweepLocator
from nvision.sim.locs.coarse.sobol_locator import StagedSobolSweepLocator


def _make_experiment(generator, rng: random.Random, noise=None) -> CoreExperiment:
    true_signal = generator.generate(rng)
    x_min, x_max = None, None
    for name in true_signal.parameter_names:
        if "frequency" in name:
            x_min, x_max = true_signal.get_param_bounds(name)
            break
    assert x_min is not None
    return CoreExperiment(true_signal=true_signal, noise=noise, x_min=x_min, x_max=x_max)


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


# ---------------------------------------------------------------------------
# Overall run benchmarks
# ---------------------------------------------------------------------------


def _overall_run_ms(
    benchmark: Any, locator_class: Any, experiment: CoreExperiment, max_steps: int = 20, **config: Any
) -> None:
    rng = random.Random(2)
    needs_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
    if needs_belief:
        config.setdefault("belief", _dummy_belief(experiment.true_signal.model))
        config.setdefault("signal_model", experiment.true_signal.model)
    benchmark(lambda: list(run_loop(locator_class, experiment, rng, max_steps=max_steps, **config)))


def _overall_bayesian_ms(
    benchmark: Any,
    locator_class: type[SequentialBayesianLocator],
    experiment: CoreExperiment,
    builder: Any,
    max_steps: int = 20,
    **extra: Any,
) -> None:
    rng = random.Random(2)
    benchmark(
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
class TestOverallNVCenter:
    """Full-run latency on an NV-center Lorentzian (max_steps=20)."""

    def test_simple_sweep(self, benchmark):
        exp = _nv_center_experiment()
        _overall_run_ms(benchmark, SimpleSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)

    def test_staged_sobol(self, benchmark):
        exp = _nv_center_experiment()
        _overall_run_ms(benchmark, StagedSobolSweepLocator, exp, max_steps=20, domain_lo=exp.x_min, domain_hi=exp.x_max)

    def test_maximum_likelihood(self, benchmark):
        exp = _nv_center_experiment()
        _overall_bayesian_ms(
            benchmark,
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

    def test_utility_sampling(self, benchmark):
        exp = _nv_center_experiment()
        _overall_bayesian_ms(
            benchmark,
            UtilitySamplingLocator,
            exp,
            builder=nv_center_belief,
            max_steps=20,
            pickiness=4.0,
            n_mc_samples=16,
            n_candidates=16,
            n_grid_freq=64,
            n_grid_linewidth=16,
            n_grid_split=16,
            n_grid_k_np=8,
            n_grid_depth=16,
            n_grid_background=8,
        )

    @pytest.mark.skip(reason="SBEDLocator is slow")
    def test_sbed(self, benchmark):
        exp = _nv_center_experiment()
        _overall_bayesian_ms(
            benchmark,
            SbedLocator,
            exp,
            builder=nv_center_smc_belief,
            max_steps=12,
            n_mc_samples=8,
            n_candidates=8,
            num_particles=1024,
        )


@pytest.mark.benchmark
class TestSBEDAcquireBottleneck:
    """Isolated timing test for SbedLocator._acquire()."""

    def test_acquire_bottleneck(self, benchmark):
        exp = _nv_center_experiment()
        rng = random.Random(42)
        loc = SbedLocator.create(
            builder=nv_center_smc_belief,
            max_steps=12,
            n_mc_samples=8,
            n_candidates=8,
            num_particles=1024,
            parameter_bounds=None,
            initial_sweep_steps=4,
            noise_std=0.02,
        )

        # Warmup
        for _ in range(4):
            x = loc.next()
            obs = exp.measure(x, rng)
            loc.observe(obs)

        benchmark.pedantic(loc._acquire, rounds=2, iterations=1)
