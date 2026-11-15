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
        exp = CoreExperiment(true_signal=object(), noise=None, drive_freq_min_phys=0.0, drive_freq_max_phys=1.0)
        built.append(exp)
        return exp

    a = get_shared_core_experiment(task, 0, build, n_repeats=1)
    b = get_shared_core_experiment(task, 0, build, n_repeats=1)
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
        exp = CoreExperiment(true_signal=object(), noise=None, drive_freq_min_phys=0.0, drive_freq_max_phys=1.0)
        built.append(exp)
        return exp

    a = get_shared_core_experiment(task_a, 0, build, n_repeats=1)
    b = get_shared_core_experiment(task_b, 0, build, n_repeats=1)
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
        runner.repeats = 5
        runner.task = SimpleNamespace(
            seed=42,
            generator_name=gen_name,
            generator=combo.generator,
            noise=combo.noise,
            noise_name=noise_name,
            repeat_total=5,
            repeat_offset=0,
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


# ---- Stratified zeeman_shift draws -------------------------------------------------------------------------------

_STRAT_GEN = "NVCenter-voigt-w0.50MHz-c0.10-si0.60MHz-hfn14"


def _strat_runner(noise_name: str, strategy: str, n_total: int):
    from types import SimpleNamespace

    from nvision.runner.executor import _TaskRunner
    from nvision.sim.combinations import CombinationGrid

    combo = CombinationGrid().resolve(_STRAT_GEN, noise_name, strategy)
    assert combo is not None
    runner = _TaskRunner.__new__(_TaskRunner)
    runner.repeats = n_total
    runner.task = SimpleNamespace(
        seed=42,
        generator_name=_STRAT_GEN,
        generator=combo.generator,
        noise=combo.noise,
        noise_name=noise_name,
        repeat_total=n_total,
        repeat_offset=0,
    )
    return runner, combo.generator


def _zeeman(exp) -> float:
    return float(exp.true_signal.get_param_value("zeeman_shift"))


def test_stratified_zeeman_paired_across_locators_and_noise():
    """Repeat i sees the same theta for every strategy and noise level."""
    n = 6
    thetas: dict[tuple[str, str], list[float]] = {}
    for noise, strat in [
        ("Gauss(0.01)", "Bayesian-SBED"),
        ("Gauss(0.002)", "Bayesian-SBED"),
        ("Gauss(0.008)", "Bayesian-SBED"),
    ]:
        clear_signal_experiment_cache()
        runner, _ = _strat_runner(noise, strat, n)
        thetas[(noise, strat)] = [_zeeman(runner._shared_experiment_for_repeat(i)) for i in range(n)]
    ref = next(iter(thetas.values()))
    assert all(v == ref for v in thetas.values())
    clear_signal_experiment_cache()


def test_stratified_zeeman_one_draw_per_slice():
    n = 20
    runner, gen = _strat_runner("Gauss(0.01)", "Bayesian-SBED", n)
    lo, hi = gen.zeeman_min_phys, gen.zeeman_max_phys
    width = (hi - lo) / n
    clear_signal_experiment_cache()
    thetas = [_zeeman(runner._shared_experiment_for_repeat(i)) for i in range(n)]
    for i, t in enumerate(thetas):
        assert lo + i * width <= t < lo + (i + 1) * width
    clear_signal_experiment_cache()


def test_stratified_zeeman_deterministic_and_other_params_unchanged():
    n = 5
    runner, gen = _strat_runner("Gauss(0.01)", "Bayesian-SBED", n)
    clear_signal_experiment_cache()
    a = [_zeeman(runner._shared_experiment_for_repeat(i)) for i in range(n)]
    clear_signal_experiment_cache()
    b = [_zeeman(runner._shared_experiment_for_repeat(i)) for i in range(n)]
    assert a == b

    # Same RNG consumption as the former rng.uniform draw: only zeeman_shift differs from the unstratified path.
    sig = gen.generate(random.Random(7), repeat_idx=2, n_repeats=n)
    ref = gen.generate(random.Random(7), repeat_idx=0, n_repeats=1)
    for name, value in ref.parameter_values().items():
        if name != "zeeman_shift":
            assert sig.get_param_value(name) == value, name
    clear_signal_experiment_cache()


def test_stratified_zeeman_fails_fast():
    import pytest

    _, gen = _strat_runner("Gauss(0.01)", "Bayesian-SBED", 3)
    with pytest.raises(ValueError, match="needs repeat_idx"):
        gen.generate(random.Random(0))
    with pytest.raises(ValueError, match="repeat_idx=3"):
        gen.generate(random.Random(0), repeat_idx=3, n_repeats=3)


def test_shared_signal_rejects_changed_total_repeats():
    import pytest

    clear_signal_experiment_cache()
    runner5, _ = _strat_runner("Gauss(0.01)", "Bayesian-SBED", 5)
    runner7, _ = _strat_runner("Gauss(0.01)", "Bayesian-SBED", 7)
    runner5._shared_experiment_for_repeat(0)
    with pytest.raises(ValueError, match="n_repeats=5"):
        runner7._shared_experiment_for_repeat(0)
    clear_signal_experiment_cache()
