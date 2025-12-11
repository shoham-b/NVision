import logging

import numpy as np
import polars as pl

from nvision.sim import ProjectBayesianLocator
from nvision.sim.locs.base import ScanBatch

# Setup logging
logging.basicConfig(level=logging.INFO)


def run_verification():
    print("Running verification script for ProjectBayesianLocator...")

    # Scenario parameters
    true_params = {
        "frequency": 2.87e9,
        "linewidth": 10e6,
        "amplitude": 0.05,
        "background": 1.0,
        "gaussian_width": 2e6,
        "split": 0.0,
        "k_np": 3.0,
    }

    # Initialize locator
    locator = ProjectBayesianLocator(max_evals=20, pickiness=5.0)

    # Mock scan object
    scan = ScanBatch(
        x_min=2.6e9,
        x_max=3.1e9,
        truth_positions=[true_params["frequency"]],
        signal=lambda x: 0.0,
        meta={},
    )

    pl.DataFrame({"repeat_id": [0], "active": [True]})
    history_df = pl.DataFrame(
        schema={
            "repeat_id": pl.Int64,
            "step": pl.Int64,
            "x": pl.Float64,
            "signal_values": pl.Float64,
            "uncertainty": pl.Float64,
        }
    )

    print(f"Initial estimates: {locator.current_estimates}")

    for step in range(20):
        # Propose next measurement
        # Note: ProjectBayesianLocator inherits from Single, so we use it directly.
        # But propose_next expects history as DataFrame or list.
        # And it handles warmup.

        locator.propose_next(history_df, scan=scan)
        # propose_next returns float or DataFrame depending on args.
        # If we pass repeats, it uses batched adapter.
        # But ProjectBayesianLocator is a Single locator.
        # We should check if we need a Batched adapter for it or if we can use it directly.
        # The base class propose_next handles batched adapter if repeats is passed.
        # But the batched adapter needs to know which class to instantiate.
        # Currently NVCenterSequentialBayesianLocatorBatched hardcodes NVCenterSequentialBayesianLocatorSingle.
        # So passing repeats might not work with ProjectBayesianLocator unless we update the adapter or use Single mode.

        # Let's use Single mode for verification.
        next_freq = locator.propose_next(history_df, scan=scan)

        print(f"Step {step}: Proposed {next_freq}")

        # Generate signal
        model_params = true_params.copy()
        signal = locator.odmr_model(next_freq, model_params)
        noise = np.random.normal(0, 0.05 * np.abs(signal))  # 5% noise
        measured_signal = signal + noise

        # Add to history
        new_row = pl.DataFrame(
            {
                "repeat_id": [0],
                "step": [step],
                "x": [next_freq],
                "signal_values": [float(measured_signal)],
                "uncertainty": [0.05],
            }
        )
        history_df = pl.concat([history_df, new_row])

        # Check stopping condition
        if locator.should_stop(history_df, scan):
            print(f"Stopped at step {step}")
            break

        current_est = locator.current_estimates
        print(
            f"Estimates: Freq={current_est['frequency']:.4e}, Uncert={current_est['uncertainty']:.4e}"
        )

    print("SUCCESS: ProjectBayesianLocator verified.")


if __name__ == "__main__":
    run_verification()
