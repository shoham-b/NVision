"""Debug script for EKFParticleFrequencyLocator convergence issues.

This script runs a single EKF-PF experiment repeat and reports detailed per-step
metrics to understand why particles might fail to contract on the true value.
"""

from __future__ import annotations

import random
import numpy as np
from rich.console import Console
from rich.table import Table

from nvision import (
    CoreExperiment,
    NVCenterCoreGenerator,
)
from nvision.sim.locs.ekf.ekf_locator import EKFParticleFrequencyLocator


def debug_run(seed: int = 42, initial_sweep_steps: int = 0):
    console = Console()
    rng = random.Random(seed)
    np.random.seed(seed)

    # 1. Setup Signal and Experiment
    x_min_gen, x_max_gen = 2.63e9, 2.65e9
    gen = NVCenterCoreGenerator(x_min=x_min_gen, x_max=x_max_gen, variant="lorentzian")
    true_signal = gen.generate(rng)
    x_min, x_max = true_signal.get_param_bounds("frequency")

    # 0.05 is a typical noise level in tests
    noise_std = 0.05
    exp = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)

    pb = {name: true_signal.get_param_bounds(name) for name in true_signal.parameter_names}
    true_params = true_signal.parameter_values()

    # 2. Setup Locator Config
    cfg = {
        "max_steps": 100,
        "convergence_threshold": 0.01,
        "parameter_bounds": pb,
        "noise_std": noise_std,
        "initial_sweep_steps": initial_sweep_steps,
        "num_particles": 200,
    }

    console.print(
        f"[bold blue]Starting Debug Run for EKF-PF[/bold blue] (seed={seed}, sweep={initial_sweep_steps})"
    )
    console.print(f"True Frequency: [green]{true_params['frequency']:.4e}[/green]")

    table = Table(title="EKF Particle Frequency Step-by-Step Report")
    table.add_column("Step", justify="right")
    table.add_column("x (Hz)", justify="right")
    table.add_column("y", justify="right")
    table.add_column("ESS", justify="right")
    table.add_column("Freq Mean", justify="right")
    table.add_column("Freq Std", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Resample", justify="center")

    # 3. Iterative Run
    step = 0
    locator = EKFParticleFrequencyLocator.create(**cfg)

    while not locator.done() and step < 100:
        step += 1

        # 1. Propose
        x_phys_norm = locator.next()
        # next() returns normalized [0, 1] for the experiment
        x_phys = x_phys_norm * (exp.x_max - exp.x_min) + exp.x_min

        # 2. Measure
        obs = exp.measure(x_phys_norm, rng)
        y = obs.signal_value

        # 3. Observe
        locator.observe(obs)

        # 4. Report
        belief = locator.belief

        # Get estimates
        est = belief.estimates()
        unc = belief.uncertainty()

        f_mean = est.get("frequency", 0.0)
        f_std = unc.get("frequency", 0.0)
        d_mean = est.get("dip_depth", 0.0)

        # Get ESS
        w_sq = np.sum(belief._frequency_weights**2)
        ess = 1.0 / w_sq if w_sq > 0 else 0.0

        # Detect if resample happened (SMC only)
        resampled = "Yes" if getattr(belief, "resampled", False) else ""

        table.add_row(
            str(step),
            f"{x_phys:.4e}",
            f"{y:.4f}",
            f"{ess:.2f}",
            f"{f_mean:.4e}",
            f"{f_std:.2e}",
            f"{d_mean:.3f}",
            resampled,
        )

    console.print(table)

    # Final check
    est = locator.belief.estimates()
    error = abs(est["frequency"] - true_params["frequency"])
    console.print(f"\nFinal Frequency Error: [bold red]{error:.4e}[/bold red]")
    if error > 1e7:  # 10MHz error is large for a converged run
        console.print("[bold reverse red]FAILED TO CONVERGE TO TRUE VALUE[/bold reverse red]")
    else:
        console.print("[bold green]SUCCESSFULLY CONVERGED[/bold green]")


if __name__ == "__main__":
    debug_run()
