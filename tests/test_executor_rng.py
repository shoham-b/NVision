"""Repeat RNG semantics for shared true signal vs strategy-specific measurement noise."""

import random

from nvision import (
    clear_signal_experiment_cache,
    get_shared_core_experiment,
    measurement_repeat_key,
    repeat_seed_int,
    signal_repeat_key,
)


def test_signal_repeat_key_ignores_strategy_and_noise():
    """Same (seed, generator, repeat) must map to one signal key for all strategies and noises."""
    s = signal_repeat_key(7, "NVCenter", 3)
    assert "Bayesian" not in s
    assert "NoNoise" not in s
    assert s == "7-NVCenter-3"


def test_measurement_repeat_key_includes_strategy():
    a = measurement_repeat_key(7, "NVCenter", "Bayesian-SBED", "NoNoise", 3)
    b = measurement_repeat_key(7, "NVCenter", "Bayesian-UCB", "NoNoise", 3)
    assert a != b
    assert "Bayesian-SBED" in a
    assert "Bayesian-UCB" in b


def test_signal_seed_matches_across_strategies():
    """Signal RNG seed does not depend on strategy; measurement seeds do."""
    k_sig = signal_repeat_key(42, "G", 0)
    k_a = measurement_repeat_key(42, "G", "StratA", "N", 0)
    k_b = measurement_repeat_key(42, "G", "StratB", "N", 0)
    assert repeat_seed_int(k_sig) == repeat_seed_int(signal_repeat_key(42, "G", 0))
    assert repeat_seed_int(k_a) != repeat_seed_int(k_b)


def test_random_streams_differ_for_measurement_keys():
    """Measurement RNGs for two strategies must not be identical."""
    k_a = measurement_repeat_key(1, "Gen", "S1", "Noise", 0)
    k_b = measurement_repeat_key(1, "Gen", "S2", "Noise", 0)
    ra = random.Random(repeat_seed_int(k_a))
    rb = random.Random(repeat_seed_int(k_b))
    assert ra.random() != rb.random()


def test_shared_core_experiment_same_true_signal_object():
    """Second lookup with the same repeat key returns the same true_signal; noise is wired per task."""
    from unittest.mock import MagicMock

    from nvision import CoreExperiment

    clear_signal_experiment_cache()
    task = MagicMock()
    task.seed = 1
    task.generator_name = "G"
    task.noise_name = "N"
    task.noise = None

    built: list[CoreExperiment] = []

    def build(rng: random.Random) -> CoreExperiment:
        exp = CoreExperiment(true_signal=object(), noise=None, x_min=0.0, x_max=1.0)
        built.append(exp)
        return exp

    a = get_shared_core_experiment(task, 0, build)
    b = get_shared_core_experiment(task, 0, build)
    assert a is not b
    assert a.true_signal is b.true_signal
    assert len(built) == 1

    clear_signal_experiment_cache()


def test_shared_core_experiment_same_true_signal_across_noise_names():
    """Different noise tasks for the same repeat must reuse one true_signal; noise differs per task."""
    from unittest.mock import MagicMock

    from nvision import CoreExperiment

    clear_signal_experiment_cache()
    task_a = MagicMock()
    task_a.seed = 1
    task_a.generator_name = "G"
    task_a.noise_name = "NoNoise"
    task_a.noise = object()
    task_b = MagicMock()
    task_b.seed = 1
    task_b.generator_name = "G"
    task_b.noise_name = "HeavyNoise"
    task_b.noise = object()
    assert task_a.noise is not task_b.noise

    built: list[CoreExperiment] = []

    def build(rng: random.Random) -> CoreExperiment:
        exp = CoreExperiment(true_signal=object(), noise=None, x_min=0.0, x_max=1.0)
        built.append(exp)
        return exp

    a = get_shared_core_experiment(task_a, 0, build)
    b = get_shared_core_experiment(task_b, 0, build)
    assert a.true_signal is b.true_signal
    assert a.noise is task_a.noise
    assert b.noise is task_b.noise
    assert len(built) == 1

    clear_signal_experiment_cache()


def test_shared_signal_does_not_leak_noise_model_across_noise_levels():
    """Each noise level must get its own noise_sigma prior window, whichever task builds the signal first."""
    from types import SimpleNamespace

    from nvision.runner.executor import _TaskRunner
    from nvision.sim.combinations import CombinationGrid

    gen_name = "NVCenter-voigt-w0.50MHz-c0.10-si0.60MHz-hfn14"
    grid = CombinationGrid()

    def runner_for(noise_name: str) -> _TaskRunner:
        combo = grid.resolve(gen_name, noise_name, "Bayesian-SBED")
        assert combo is not None
        runner = _TaskRunner.__new__(_TaskRunner)
        runner.task = SimpleNamespace(
            seed=42,
            generator_name=gen_name,
            generator=combo.generator,
            noise=combo.noise,
            noise_name=noise_name,
        )
        return runner

    noise_names = ["Gauss(0.01)", "Gauss(0.002)", "Gauss(0.008)"]
    expected = {n: runner_for(n)._build_experiment(random.Random(0)).true_signal.noise_bounds for n in noise_names}
    assert len({tuple(sorted(b.items())) for b in expected.values()}) == len(noise_names)

    # Try every task as the first builder: the result must not depend on who populates the cache.
    for first in noise_names:
        clear_signal_experiment_cache()
        order = [first, *[n for n in noise_names if n != first]]
        got = {n: runner_for(n)._shared_experiment_for_repeat(0) for n in order}
        for n in noise_names:
            assert got[n].true_signal.noise_bounds == expected[n], f"{n} (first builder: {first})"
            assert got[n].noise is not None
        # The noise-independent part of the signal is still shared.
        assert got[noise_names[0]].true_signal.model is got[noise_names[1]].true_signal.model

    clear_signal_experiment_cache()
