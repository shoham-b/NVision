"""Benchmark and profiling script for Sequential Bayesian Experiment Design.

This script runs a short simulation repeat and measures performance across imports,
JIT warm-up, candidate grid generation, EIG acquisition calculation, and particle updates.
"""

from __future__ import annotations

import cProfile
import pstats
import random
import time
from typing import Any

import numpy as np
from rich.console import Console

# Trace imports
t0 = time.perf_counter()
from nvision import (  # noqa: E402
    CoreExperiment,
    NVCenterCoreGenerator,
    nv_center_smc_belief,
)
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator  # noqa: E402

t_import = time.perf_counter() - t0


def run_simulation(
    steps: int = 20,
    seed: int = 42,
    num_particles: int = 1000,
    n_candidates: int | None = None,
    points_per_feature: int = 5,
) -> dict[str, Any]:
    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # 1. Setup Signal and Experiment
    x_min_gen, x_max_gen = 2.63e9, 2.65e9
    gen = NVCenterCoreGenerator(x_min=x_min_gen, x_max=x_max_gen, variant="lorentzian")
    true_signal = gen.generate(random.Random(seed))
    x_min, x_max = true_signal.get_param_bounds("frequency")

    noise_std = 0.05
    exp = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)

    pb = {name: true_signal.get_param_bounds(name) for name in true_signal.parameter_names}

    # Configure SMC
    import os

    os.environ["NVISION_SMC_NUM_PARTICLES"] = str(num_particles)
    os.environ["NVISION_SMC_POINTS_PER_MIN_FEATURE"] = str(points_per_feature)

    cfg = {
        "max_steps": steps,
        "convergence_threshold": 0.001,
        "parameter_bounds": pb,
        "noise_std": noise_std,
        "builder": nv_center_smc_belief,
        "num_particles": num_particles,
        "n_candidates": n_candidates,
    }

    # 2. Setup locator
    t_start = time.perf_counter()
    locator = SequentialBayesianExperimentDesignLocator.create(**cfg)
    t_init = time.perf_counter() - t_start

    step_times_next = []
    step_times_observe = []
    candidate_counts = []

    t_step_loop_start = time.perf_counter()
    for step in range(1, steps + 1):
        if locator.done():
            break

        # Time proposal / acquisition
        t_next0 = time.perf_counter()
        x_phys_norm = locator.next()
        t_next = time.perf_counter() - t_next0
        step_times_next.append(t_next)

        # Keep track of active candidate count
        candidates = locator.belief.get_candidates()
        candidate_counts.append(len(candidates))

        # Time observation / particle update
        obs = exp.measure(x_phys_norm, random.Random(seed + step))

        t_obs0 = time.perf_counter()
        locator.observe(obs)
        t_obs = time.perf_counter() - t_obs0
        step_times_observe.append(t_obs)

    t_loop = time.perf_counter() - t_step_loop_start

    return {
        "t_import": t_import,
        "t_init": t_init,
        "t_loop": t_loop,
        "step_times_next": step_times_next,
        "step_times_observe": step_times_observe,
        "candidate_counts": candidate_counts,
    }


def main(
    steps: int = 15,
    particles: int = 1000,
    points_per_feature: int = 5,
    profile: bool = False,
):
    console = Console()
    console.print(
        f"[bold blue]SBED Benchmark[/bold blue]: steps={steps}, particles={particles}, "
        f"points_per_feature={points_per_feature}"
    )

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    metrics = run_simulation(steps=steps, num_particles=particles, points_per_feature=points_per_feature)

    if profile:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
        console.print("[bold yellow]Top 15 Hotspot Functions (Cumulative Time):[/bold yellow]")
        ps.print_stats(15)

    console.print("\n[bold green]Timing Results:[/bold green]")
    console.print(f"  Import overhead:        {metrics['t_import']:.3f} s")
    console.print(f"  Locator initialization:  {metrics['t_init']:.3f} s")
    console.print(f"  Step loop total:        {metrics['t_loop']:.3f} s")

    next_times = metrics["step_times_next"]
    obs_times = metrics["step_times_observe"]
    cands = metrics["candidate_counts"]

    console.print(f"\n[bold green]Per-Step Stats (over {len(next_times)} steps):[/bold green]")
    # First step vs subsequent steps
    if len(next_times) > 0:
        console.print("  Step 1 (JIT warm-up):")
        console.print(f"    next() / acquisition:  {next_times[0]:.3f} s")
        console.print(f"    observe() / update:   {obs_times[0]:.3f} s")
        console.print(f"    candidates count:     {cands[0]}")

    if len(next_times) > 1:
        avg_next_sub = np.mean(next_times[1:])
        avg_obs_sub = np.mean(obs_times[1:])
        avg_cands_sub = np.mean(cands[1:])
        console.print("  Subsequent Steps (Average):")
        console.print(f"    next() / acquisition:  {avg_next_sub:.3f} s")
        console.print(f"    observe() / update:   {avg_obs_sub:.3f} s")
        console.print(f"    candidates count:     {avg_cands_sub:.1f}")


if __name__ == "__main__":
    import typer

    typer.run(main)
