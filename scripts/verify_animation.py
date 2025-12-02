"""Verification script for locator animation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from nvision.sim.gen.distributions.nv_center_manufacturer import NVCenterManufacturer
from nvision.sim.locs import run_locator, ScanBatch, NVCenterSequentialBayesianLocator
from nvision.sim.vis.animator import animate_locator_progress


def verify_animation():
    print("Setting up simulation...")

    # 1. Generate a 3-peak signal
    # True parameters
    true_f0 = 2.87e9
    true_split = 0.07e9  # 70 MHz
    true_linewidth = 10e6

    manufacturer = NVCenterManufacturer(delta_f_hf=true_split, omega=true_linewidth, k_np=3.0)

    # Create a scan batch
    x_min, x_max = 2.7e9, 3.0e9
    rng = random.Random(42)

    # We need to manually create the signal function
    # NVCenterManufacturer.build_peak returns (func, params)
    signal_func, params = manufacturer.build_peak(
        center=true_f0, base=1.0, x_min=x_min, x_max=x_max, rng=rng
    )

    # Create a wrapper for the signal function that handles array input
    def signal_wrapper(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.array([signal_func(xi) for xi in x])
        return signal_func(x)

    scan = ScanBatch(
        x_min=x_min,
        x_max=x_max,
        signal=signal_wrapper,
        truth_positions=[true_f0 - true_split, true_f0, true_f0 + true_split],
        meta={},
    )

    print("Running locator...")
    # 2. Run locator
    locator = NVCenterSequentialBayesianLocator(
        max_evals=30,  # Reduced steps for speed
        prior_bounds=(x_min, x_max),
        distribution="voigt-zeeman",
        split_prior=(50e6, 100e6),  # Hint it towards the right split
        n_warmup=10,
    )

    history_df = run_locator(locator=locator, scan=scan, seed=42, max_steps=50)

    print(f"Simulation complete. Steps: {history_df.height}")
    print("Columns:", history_df.columns)

    # 3. Generate animation
    output_path = Path("locator_progress.gif")
    print(f"Generating animation to {output_path}...")

    true_params = {
        "frequency": true_f0,
        "split": true_split,
        "linewidth": true_linewidth,
        "k_np": 3.0,
        "amplitude": manufacturer.a,
        "background": 1.0,
        "gaussian_width": 2e6,  # Approximate
    }

    animate_locator_progress(
        history_df=history_df,
        output_path=output_path,
        true_params=true_params,
        distribution="voigt-zeeman",
        fps=5,
    )

    if output_path.exists():
        print("✓ Animation created successfully!")
    else:
        print("✗ Animation file not found.")


if __name__ == "__main__":
    verify_animation()
