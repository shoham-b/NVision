from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from nvision.sim.scan_batch import ScanBatch
from rich.console import Console
from rich.table import Table

from nvision.sim import NVCenterSequentialBayesianLocator

log = logging.getLogger(__name__)


@dataclass
class EvalScenario:
    name: str
    true_params: dict[str, float]
    noise_level: float = 0.05  # Relative noise level


def run_evaluation(
    scenarios: list[EvalScenario],
    repeats: int = 5,
    max_steps: int = 50,
    output_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Run evaluation of the Bayesian locator against specified scenarios.
    """
    results = []

    # Initialize locator with default settings
    # We use a fresh locator for each run to ensure clean state

    for scenario in scenarios:
        log.info(f"Evaluating scenario: {scenario.name}")

        for r in range(repeats):
            # 1. Setup Ground Truth
            # We need to construct a generator that produces this specific scenario
            # For now, we'll manually construct the "truth" scan object
            # effectively mocking the generator's output for a specific parameter set

            # Create a dummy scan object with the true parameters
            # The locator expects a ScanBatch to query signals

            # We will use a custom ScanBatch that returns values based on the scenario's true params
            # and adds noise.

            true_params = scenario.true_params

            # Create a temporary generator just to get the model function
            # or we can use the locator's model function directly if it's static
            # The locator has `odmr_model`.

            locator = NVCenterSequentialBayesianLocator(max_evals=max_steps)

            # We need a way to simulate the experiment.
            # The locator's `propose_next` takes history and returns a frequency.
            # Then we need to generate the signal for that frequency.

            history: list[dict[str, float]] = []

            # Initial warmup is handled by the locator internally if we use the high-level loop
            # But here we want to control the loop to measure performance step-by-step

            # Let's use a simple loop similar to cli.py but simplified for evaluation

            start_time = time.perf_counter()

            # Mock scan object for bounds
            scan = ScanBatch(
                x_min=2.6e9,
                x_max=3.1e9,
                truth_positions=[true_params.get("frequency", 2.87e9)],  # Simplified truth
                # We might need more complex truth for multi-peak, but let's start simple
                signal_func=lambda x: 0.0,  # Placeholder, we won't use this directly
            )

            # We need a signal generator function
            def get_signal(freq, locator=locator, scenario=scenario, true_params=true_params):
                # Use the locator's model to generate "true" signal
                # We need to adapt the params to what odmr_model expects
                model_params = {
                    "frequency": true_params.get("frequency", 2.87e9),
                    "linewidth": true_params.get("linewidth", 10e6),
                    "amplitude": true_params.get("amplitude", 0.05),
                    "background": true_params.get("background", 1.0),
                    "gaussian_width": true_params.get("gaussian_width", 2e6),
                    "split": true_params.get("split", 0.0),
                    "k_np": true_params.get("k_np", 3.0),
                }

                # We need to temporarily set the distribution in locator to match scenario if needed
                # For now assuming Lorentzian or Voigt based on params

                signal = locator.odmr_model(freq, model_params)

                # Add noise
                noise = np.random.normal(0, scenario.noise_level * np.abs(signal))
                return signal + noise

            # Run the loop
            converged_step = -1
            final_error = math.inf

            for step in range(max_steps):
                # Propose
                # The locator expects history as a DataFrame or list of dicts
                # and a scan object for bounds

                # We need to handle the initial warmup that the locator might expect
                # The locator usually does warmup inside `propose_next` if history is empty/short

                next_freq = locator.propose_next(history, scan)

                # Measure
                if isinstance(next_freq, (float, int)):
                    # Single point
                    signal = get_signal(next_freq)
                    history.append(
                        {
                            "x": next_freq,
                            "signal_values": float(signal),
                            "uncertainty": 0.05,  # Dummy uncertainty
                        }
                    )
                else:
                    # If it returns a DataFrame (batched), handle it
                    # But for this eval we assume single mode or we adapt
                    # The locator code shows it returns float if not batched
                    pass

                # Check convergence / Error tracking
                # We can check the current estimate in the locator
                est_freq = locator.current_estimates["frequency"]
                true_freq = true_params["frequency"]
                error = abs(est_freq - true_freq)

                if error < 1e5 and converged_step == -1:  # Converged to within 100 kHz
                    converged_step = step + 1

                final_error = error

                if locator.should_stop(history, scan):
                    break

            duration = time.perf_counter() - start_time

            results.append(
                {
                    "scenario": scenario.name,
                    "repeat": r,
                    "steps": len(history),
                    "converged_step": converged_step if converged_step != -1 else None,
                    "final_error_hz": final_error,
                    "duration_s": duration,
                    "final_uncertainty": locator.current_estimates["uncertainty"],
                }
            )

    df = pl.DataFrame(results)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_dir / "bayesian_eval_results.csv")

    return df


def print_results(df: pl.DataFrame):
    console = Console()

    # Aggregate results
    agg = df.group_by("scenario").agg(
        [
            pl.col("steps").mean().alias("avg_steps"),
            pl.col("final_error_hz").mean().alias("avg_error_hz"),
            pl.col("duration_s").mean().alias("avg_duration_s"),
            pl.col("converged_step").mean().alias("avg_convergence_step"),
        ]
    )

    table = Table(title="Bayesian Locator Evaluation Results")
    table.add_column("Scenario", style="cyan")
    table.add_column("Avg Steps", justify="right")
    table.add_column("Avg Error (kHz)", justify="right")
    table.add_column("Avg Duration (s)", justify="right")

    for row in agg.iter_rows(named=True):
        table.add_row(
            row["scenario"],
            f"{row['avg_steps']:.1f}",
            f"{row['avg_error_hz'] / 1e3:.1f}",
            f"{row['avg_duration_s']:.3f}",
        )

    console.print(table)


if __name__ == "__main__":
    # Example usage
    scenarios = [
        EvalScenario(
            "Simple Lorentzian",
            {"frequency": 2.87e9, "linewidth": 10e6, "amplitude": 0.05, "background": 1.0},
        ),
        EvalScenario(
            "Shifted Lorentzian",
            {"frequency": 2.90e9, "linewidth": 15e6, "amplitude": 0.04, "background": 1.0},
        ),
    ]
    results = run_evaluation(scenarios, repeats=3, max_steps=30)
    print_results(results)
