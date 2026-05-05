"""Standalone benchmark for SBED _acquire bottleneck."""
from __future__ import annotations

import os
import random
import time

os.environ["DEBUG_SBED_TIMING"] = "1"

from nvision import CoreExperiment, NVCenterCoreGenerator, run_loop
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator as SbedLocator


def _make_experiment() -> CoreExperiment:
    rng = random.Random(44)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    x_min, x_max = None, None
    for name in true_signal.parameter_names:
        if "frequency" in name:
            x_min, x_max = true_signal.get_param_bounds(name)
            break
    assert x_min is not None
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)


def main() -> None:
    exp = _make_experiment()
    rng = random.Random(42)

    loc = SbedLocator.create(
        builder=nv_center_smc_belief,
        max_steps=12,
        n_candidates=200,
        n_draws=100,
        num_particles=10000,
        parameter_bounds=None,
        initial_sweep_steps=4,
        noise_std=0.02,
    )

    # Warmup: run initial sweep + a few Bayesian steps
    print("Warming up...")
    for _ in range(8):
        x = loc.next()
        obs = exp.measure(x, rng)
        loc.observe(obs)

    print("\n--- Benchmarking _acquire ---")
    n = 5
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        loc._acquire()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    print(f"\nAverage over {n} runs: {avg:.3f}s")

    # Also run with cProfile if requested via env var
    if os.environ.get("SBED_PROFILE") == "1":
        import cProfile
        import pstats
        from io import StringIO

        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(3):
            loc._acquire()
        profiler.disable()
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(30)
        print("\n--- cProfile (cumtime) ---")
        print(stream.getvalue())


if __name__ == "__main__":
    main()
