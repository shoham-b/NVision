import numpy as np
import polars as pl
from nvision.sim.locs.nv_center.fisher_information import (
    calculate_fisher_information,
    calculate_crb,
    _lorentzian_deriv_f0,
)
from nvision.sim.locs.nv_center.evaluation import BayesianMetrics
from nvision.core.paths import slugify


def test_fisher_calculation():
    print("Testing Fisher Information calculation...")
    # Setup simple case
    true_params = {"frequency": 2.87e9, "linewidth": 10e6, "amplitude": 0.1, "background": 1.0}

    # Measurements exactly on peak and far away
    freqs = np.array([2.87e9, 2.87e9 + 100e6])

    # 1. Check derivative at peak
    # Derivative of Lorentzian at peak (f=f0) should be 0 because it's a critical point
    deriv_peak = _lorentzian_deriv_f0(
        np.array([2.87e9]), true_params["frequency"], true_params["linewidth"], true_params["amplitude"]
    )
    print(f"Derivative at peak: {deriv_peak} (Should be close to 0)")
    assert np.isclose(deriv_peak, 0.0)

    # 2. Check derivative at max slope (approx gamma/ (2*sqrt(3)) or similar)
    # Actually for Lorentzian derivative max is around +/- gamma/(2*sqrt(3))

    # 3. CRB Calculation
    # If derivative at peak is 0, FI at peak is 0.
    # So measurements at peak give NO information about frequency position locally?
    # Yes, for symmetric peak, first order derivative is zero.

    # Let's take a point at half width
    hwhm = 5e6
    f_half = 2.87e9 + hwhm

    fi_cumulative = calculate_fisher_information(
        [f_half], true_params, noise_model="gaussian", noise_params={"sigma": 0.05}
    )
    print(f"FI at HWHM: {fi_cumulative}")

    crb = calculate_crb([f_half], true_params, noise_model="gaussian", noise_params={"sigma": 0.05})
    print(f"CRB at HWHM: {crb}")
    assert len(crb) == 1
    assert crb[0] > 0


def test_metrics_integration():
    print("\nTesting Metrics Integration...")

    true_params = {"frequency": 2.87e9, "linewidth": 10e6, "amplitude": 0.1}

    # Fake history
    param_history = [
        {"frequency": 2.86e9, "uncertainty": 10e6, "entropy": 5.0},
        {"frequency": 2.865e9, "uncertainty": 5e6, "entropy": 4.0},
        {"frequency": 2.87e9, "uncertainty": 1e6, "entropy": 3.0},  # Converged
    ]

    meas_history = [
        {"x": 2.87e9 - 5e6, "signal_values": 0.95},
        {"x": 2.87e9 + 5e6, "signal_values": 0.95},
        {"x": 2.87e9 + 2e6, "signal_values": 0.92},
    ]

    metrics = BayesianMetrics.from_history(
        param_history, ground_truth=true_params, measurement_history=meas_history, convergence_threshold=2e6
    )

    summary = metrics.summary()
    print("Metrics Summary:", summary)

    assert summary["convergence_step"] == 2.0
    assert "mean_error" in summary
    assert "final_crb_std" in summary

    # CRB should decrease (or stay same) as we add measurements
    # Here measurement_history has 3 items
    assert len(metrics.crb_std_history) == 3
    print("CRB History:", metrics.crb_std_history)


if __name__ == "__main__":
    test_fisher_calculation()
    test_metrics_integration()
    print("\nAll verification tests passed!")
