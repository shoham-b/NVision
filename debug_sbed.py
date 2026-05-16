"""Debug script for SBED convergence issues.

This script runs a single SBED experiment repeat and reports detailed per-step
metrics to understand why particles might concentrate in the wrong place.
"""

from __future__ import annotations

import random
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from nvision import (
    CoreExperiment,
    NVCenterCoreGenerator,
    nv_center_smc_belief,
    run_loop,
)
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.sim.locs.bayesian.students_t_locator import StudentsTLocator

def debug_run(seed: int = 42, initial_sweep_steps: int = 0, locator_type: str = "sbed"):
    console = Console()
    rng = random.Random(seed)
    np.random.seed(seed)

    # 1. Setup Signal and Experiment
    # Using Lorentzian variant as it's common for these tests
    # Narrower domain for "narrow scan" scenario (20MHz)
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
    }

    if locator_type == "sbed":
        cfg["builder"] = nv_center_smc_belief
        cfg["num_particles"] = 200
        locator_cls = SequentialBayesianExperimentDesignLocator
    elif locator_type == "students_t":
        cfg["signal_model"] = true_signal.model
        cfg["n_components"] = 3
        locator_cls = StudentsTLocator
    else:
        raise ValueError(f"Unknown locator type: {locator_type}")

    console.print(f"[bold blue]Starting Debug Run[/bold blue] (seed={seed}, sweep={initial_sweep_steps}, type={locator_type})")
    console.print(f"True Frequency: [green]{true_params['frequency']:.4e}[/green]")

    table = Table(title=f"{locator_type.upper()} Step-by-Step Report")
    table.add_column("Step", justify="right")
    table.add_column("Phase", style="cyan")
    table.add_column("x (Hz)", justify="right")
    table.add_column("y", justify="right")
    table.add_column("SNR", justify="right")
    table.add_column("ESS / Mix", justify="right")
    table.add_column("Freq Mean", justify="right")
    table.add_column("Freq Std", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Resample", justify="center")

    # 3. Iterative Run
    step = 0
    last_ess = 0.0

    # Create the locator
    locator = locator_cls.create(**cfg)

    while not locator.done() and step < 150:
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

        # Determine Phase
        if not locator._staged_sobol.done():
            phase = "Sweep"
        elif hasattr(locator, "_warmup_obs_buffer") and len(locator._warmup_obs_buffer) > 0:
            phase = "Warmup"
        else:
            phase = "Bayesian"

        # Get estimates
        est = belief.estimates()
        unc = belief.uncertainty()

        f_mean = est.get("frequency", 0.0)
        f_std = unc.get("frequency", 0.0)
        d_mean = est.get("dip_depth", 0.0)
        b_mean = est.get("background", 1.0)

        # Get ESS or Mixing Weights
        if hasattr(belief, "_weights"):
            w_sq = np.sum(belief._weights**2)
            metric_val = 1.0 / w_sq if w_sq > 0 else 0.0
        elif hasattr(belief, "mixing_weights"):
            # For StudentsT, show the highest weight
            metric_val = float(np.max(belief.mixing_weights))
        else:
            metric_val = 0.0

        # SNR: (background - y) / noise_std
        snr = (b_mean - y) / noise_std if noise_std > 0 else 0.0

        # Detect if resample happened (SMC only)
        resampled = "Yes" if (metric_val > last_ess * 2.0 and step > 1 and phase != "Warmup" and locator_type == "sbed") else ""
        last_ess = metric_val

        table.add_row(
            str(step),
            phase,
            f"{x_phys:.4e}",
            f"{y:.4f}",
            f"{snr:+.1f}",
            f"{metric_val:.2f}",
            f"{f_mean:.4e}",
            f"{f_std:.2e}",
            f"{d_mean:.3f}",
            resampled
        )

    console.print(table)

    # Final check
    est = locator.belief.estimates()
    error = abs(est["frequency"] - true_params["frequency"])
    console.print(f"\nFinal Frequency Error: [bold red]{error:.4e}[/bold red]")
    if error > 1e7: # 10MHz error is large for a converged run
        console.print("[bold reverse red]FAILED TO CONVERGE TO TRUE VALUE[/bold reverse red]")
    else:
        console.print("[bold green]SUCCESSFULLY CONVERGED[/bold green]")

if __name__ == "__main__":
    import typer
    typer.run(debug_run)
