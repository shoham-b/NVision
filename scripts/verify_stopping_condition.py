import numpy as np
import polars as pl
from nvision.sim.locs.nv_center.sequential_bayesian_locator import NVCenterSequentialBayesianLocator
from nvision.sim.locs.base import ScanBatch


def verify_stopping():
    print("Starting verification of Bayesian stopping condition...")

    # Setup locator
    locator = NVCenterSequentialBayesianLocator(
        max_evals=100,
        n_warmup=10,
        distribution="lorentzian",
        noise_model="gaussian",
        convergence_threshold=1e-3,  # Very small so it doesn't trigger early
    )

    # Ground truth parameters
    true_params = {"frequency": 2.87e9, "linewidth": 10e6, "amplitude": 0.05, "background": 1.0}

    scan = ScanBatch(
        x_min=2.8e9,
        x_max=2.94e9,
        signal=lambda x: 0.0,  # Dummy
        meta={},
        truth_positions=[2.87e9],
    )

    history = []

    stopped_by_crb = False

    for i in range(100):
        # 1. Propose
        next_freq = locator.propose_next(pl.DataFrame(history), scan=scan)

        # 2. Simulate Measurement
        # Simple Lorentzian model
        f0 = true_params["frequency"]
        gamma = true_params["linewidth"]
        amp = true_params["amplitude"]
        bg = true_params["background"]

        hwhm = gamma / 2.0
        diff = next_freq - f0
        denom = diff * diff + hwhm * hwhm
        signal = bg - (amp * hwhm * hwhm) / denom

        # Add noise
        noisy_signal = np.random.normal(signal, 0.05)  # Sigma=0.05 match locator default

        meas = {"x": next_freq, "signal_values": noisy_signal, "uncertainty": 0.05}
        history.append(meas)

        # 3. Check stopping manually to inspect internals before step update?
        # Actually simplest is just to call should_stop after update?
        # NO, should_stop is called with history. locator state is updated by propose_next -> update_posterior?
        # Wait, propose_next calls ingest_history which updates posterior.
        # But we need to call propose_next again to update posterior with new measurement?
        # In the real loop:
        #   propose -> (returns x)
        #   measure -> (get y)
        #   [loop repeats] -> propose(history including y) which updates posterior.

        # So we need to call propose or manually update for the check to see the new state.
        # Let's manually update to be explicit.
        locator.update_posterior(meas)

        should_stop = locator.should_stop(history=pl.DataFrame(history), scan=scan)

        # Check CRB state
        uncert = locator.current_estimates.get("uncertainty", float("inf"))
        # Calculate CRB manually to see
        from nvision.sim.locs.nv_center.fisher_information import calculate_crb

        x_vals = [m["x"] for m in locator.measurement_history]
        crb_hist = calculate_crb(x_vals, locator.current_estimates)
        crb = crb_hist[-1] if crb_hist else float("inf")

        ratio = uncert / crb if crb > 0 else float("inf")

        print(
            f"Step {i + 1}: Freq={next_freq:.6e}, Est={locator.current_estimates['frequency']:.6e}, Uncert={uncert:.2e}, CRB={crb:.2e}, Ratio={ratio:.2f}"
        )

        if should_stop:
            print(f"Locator requested stop at step {i + 1}")
            if ratio <= 2.1:  # Allow slight numerical slop or if it triggered exactly
                print("SUCCESS: Stopping likely due to CRB condition (Ratio ~<= 2.0).")
                stopped_by_crb = True
            else:
                print(f"WARNING: Stopped but ratio is {ratio:.2f}. Might be another condition?")
            break

    if not stopped_by_crb:
        print("FAILED: Did not stop by CRB condition within max_evals.")
    else:
        print("Verification PASSED.")


if __name__ == "__main__":
    verify_stopping()
