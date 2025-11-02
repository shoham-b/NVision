#!/usr/bin/env python3
"""
Sequential Bayesian Experiment Design Example for NV Center ODMR

This example demonstrates the use of Sequential Bayesian Experiment Design
for ODMR measurements of NV centers, showing the significant speedup compared
to conventional frequency-swept measurements.

Based on:
"Sequential Bayesian Experiment Design for Optically Detected Magnetic Resonance
of Nitrogen-Vacancy Centers" by Dushenko et al., Phys. Rev. Applied 14, 054036 (2020)
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from nvision.sim import GridScanLocator, NVCenterSequentialBayesianLocator
from nvision.sim.locs.models.obs import Obs


def simulate_nv_odmr_signal(
    frequency: float, true_params: dict[str, float], noise_level: float = 0.05
) -> float:
    """
    Simulate NV center ODMR signal with realistic noise.

    Args:
        frequency: Measurement frequency (Hz)
        true_params: True NV center parameters
        noise_level: Relative noise level

    Returns:
        Simulated intensity measurement
    """
    f0 = true_params["frequency"]
    gamma = true_params["linewidth"]
    amplitude = true_params["amplitude"]
    bg = true_params["background"]

    # Lorentzian ODMR signal (dip in fluorescence)
    lorentzian = amplitude * (gamma / 2) ** 2 / ((frequency - f0) ** 2 + (gamma / 2) ** 2)
    clean_signal = bg - lorentzian

    # Add realistic noise (combination of shot noise and technical noise)
    noise_std = noise_level * abs(clean_signal) + 0.01
    noisy_signal = np.random.normal(clean_signal, noise_std)

    return noisy_signal, noise_std


def run_measurement_sequence(
    locator, domain: tuple, true_params: dict[str, float], max_measurements: int = 50
) -> list[Obs]:
    """
    Run a complete measurement sequence using the given locator strategy.

    Args:
        locator: Locator strategy to use
        domain: Frequency domain (Hz)
        true_params: True parameters for simulation
        max_measurements: Maximum number of measurements

    Returns:
        List of measurement observations
    """
    history = []

    for i in range(max_measurements):
        # Check stopping condition
        if locator.should_stop(history):
            print(f"Stopped after {len(history)} measurements (convergence criteria met)")
            break

        # Get next measurement frequency
        next_freq = locator.propose_next(history, domain)

        # Simulate measurement
        intensity, uncertainty = simulate_nv_odmr_signal(next_freq, true_params)

        # Add to history
        obs = Obs(x=next_freq, intensity=intensity, uncertainty=uncertainty)
        history.append(obs)

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} measurements")

    print(f"Total measurements: {len(history)}")
    return history


def compare_strategies():
    """
    Compare Sequential Bayesian Design with conventional grid scan.
    """
    print("=" * 60)
    print("Sequential Bayesian Experiment Design vs Grid Scan Comparison")
    print("=" * 60)

    # Define true NV center parameters (single NV center)
    true_params = {
        "frequency": 2.870e9,  # 2.87 GHz (typical NV center)
        "linewidth": 15e6,  # 15 MHz linewidth
        "amplitude": 0.15,  # 15% contrast
        "background": 1.0,  # Normalized background
    }

    domain = (2.84e9, 2.90e9)  # 60 MHz range

    print(f"True NV center frequency: {true_params['frequency']/1e9:.3f} GHz")
    print(f"True linewidth: {true_params['linewidth']/1e6:.1f} MHz")
    print(f"Search domain: {domain[0]/1e9:.3f} - {domain[1]/1e9:.3f} GHz")
    print()

    # Initialize strategies
    sequential_locator = NVCenterSequentialBayesianLocator(
        max_evals=50,
        prior_bounds=domain,
        convergence_threshold=1e6,  # 1 MHz precision target
        acquisition_function="expected_information_gain",
    )

    grid_locator = GridScanLocator(
        n_points=50,  # Same number of measurements for fair comparison
    )

    print("Running Sequential Bayesian Experiment Design...")
    start_time = time.time()
    sequential_history = run_measurement_sequence(sequential_locator, domain, true_params)
    sequential_time = time.time() - start_time
    sequential_result = sequential_locator.finalize(sequential_history)

    print("\nRunning conventional grid scan...")
    start_time = time.time()
    grid_history = run_measurement_sequence(grid_locator, domain, true_params)
    grid_time = time.time() - start_time
    grid_result = grid_locator.finalize(grid_history)

    # Analyze results
    print("\n" + "=" * 40)
    print("RESULTS COMPARISON")
    print("=" * 40)

    sequential_error = abs(sequential_result["x1_hat"] - true_params["frequency"])
    grid_error = abs(grid_result["x1"] - true_params["frequency"])

    print("Sequential Bayesian Design:")
    print(f"  Measurements: {len(sequential_history)}")
    print(f"  Estimated frequency: {sequential_result['x1_hat']/1e9:.6f} GHz")
    print(f"  Error: {sequential_error/1e6:.3f} MHz")
    print(f"  Uncertainty: {sequential_result['uncert']/1e6:.3f} MHz")
    print(f"  Computation time: {sequential_time:.3f} s")

    print("\nConventional Grid Scan:")
    print(f"  Measurements: {len(grid_history)}")
    print(f"  Estimated frequency: {grid_result['x1']/1e9:.6f} GHz")
    print(f"  Error: {grid_error/1e6:.3f} MHz")
    print(f"  Uncertainty: {grid_result['uncert']/1e6:.3f} MHz")
    print(f"  Computation time: {grid_time:.3f} s")

    # Calculate speedup
    measurement_speedup = len(grid_history) / len(sequential_history)
    print("\nSpeedup Analysis:")
    print(f"  Measurement reduction: {measurement_speedup:.1f}x")
    print(f"  Time speedup: {grid_time/sequential_time:.1f}x")

    if sequential_error < grid_error:
        accuracy_improvement = grid_error / sequential_error
        print(f"  Accuracy improvement: {accuracy_improvement:.1f}x better")

    return sequential_history, grid_history, sequential_result, grid_result


def visualize_adaptive_strategy(history: list[Obs], result: dict[str, Any], title: str):
    """
    Visualize the adaptive measurement strategy.

    Args:
        history: Measurement history
        result: Final results
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Extract data
    frequencies = [obs.x for obs in history]
    intensities = [obs.intensity for obs in history]
    uncertainties = [obs.uncertainty for obs in history]

    # Plot 1: ODMR spectrum with measurement points
    freq_range = np.linspace(min(frequencies) - 5e6, max(frequencies) + 5e6, 1000)

    # True signal for reference (if available)
    true_params = {"frequency": 2.870e9, "linewidth": 15e6, "amplitude": 0.15, "background": 1.0}

    true_signal = []
    for f in freq_range:
        lorentzian = (
            true_params["amplitude"]
            * (true_params["linewidth"] / 2) ** 2
            / ((f - true_params["frequency"]) ** 2 + (true_params["linewidth"] / 2) ** 2)
        )
        true_signal.append(true_params["background"] - lorentzian)

    ax1.plot(freq_range / 1e9, true_signal, "k-", label="True ODMR signal", alpha=0.7)
    ax1.errorbar(
        [f / 1e9 for f in frequencies],
        intensities,
        yerr=uncertainties,
        fmt="ro",
        label="Measurements",
        alpha=0.8,
    )
    ax1.axvline(
        result.get("x1_hat", result.get("x1", float("nan"))) / 1e9,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f'Estimated peak: {result.get("x1_hat", result.get("x1", float("nan")))/1e9:.6f} GHz',
    )
    ax1.axvline(
        true_params["frequency"] / 1e9,
        color="black",
        linestyle=":",
        alpha=0.8,
        label=f'True peak: {true_params["frequency"]/1e9:.6f} GHz',
    )

    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Fluorescence Intensity")
    ax1.set_title(f"{title} - ODMR Spectrum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Measurement sequence (adaptive sampling)
    measurement_order = range(1, len(frequencies) + 1)
    scatter = ax2.scatter(
        [f / 1e9 for f in frequencies], measurement_order, c=measurement_order, cmap="viridis", s=50
    )

    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Measurement Order")
    ax2.set_title(f"{title} - Adaptive Sampling Strategy")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Measurement Order")

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def demonstrate_information_gain():
    """
    Demonstrate the information gain calculation.
    """
    print("\n" + "=" * 40)
    print("INFORMATION GAIN DEMONSTRATION")
    print("=" * 40)

    locator = NVCenterSequentialBayesianLocator(prior_bounds=(2.84e9, 2.90e9), grid_resolution=200)

    # Add a few measurements
    measurements = [
        {"frequency": 2.85e9, "intensity": 0.98, "uncertainty": 0.05},
        {"frequency": 2.87e9, "intensity": 0.92, "uncertainty": 0.05},
        {"frequency": 2.89e9, "intensity": 0.97, "uncertainty": 0.05},
    ]

    for meas in measurements:
        locator.update_posterior(meas)

    # Calculate information gain across frequency range
    test_frequencies = np.linspace(2.84e9, 2.90e9, 50)
    info_gains = []

    print("Calculating information gain across frequency range...")
    for freq in test_frequencies:
        gain = locator.expected_information_gain(freq)
        info_gains.append(gain)

    # Find optimal next measurement
    optimal_idx = np.argmax(info_gains)
    optimal_freq = test_frequencies[optimal_idx]
    max_gain = info_gains[optimal_idx]

    print(f"Optimal next measurement: {optimal_freq/1e9:.6f} GHz")
    print(f"Expected information gain: {max_gain:.4f}")

    # Plot information gain
    plt.figure(figsize=(10, 6))
    plt.plot(test_frequencies / 1e9, info_gains, "b-", linewidth=2)
    plt.axvline(
        optimal_freq / 1e9,
        color="red",
        linestyle="--",
        label=f"Optimal: {optimal_freq/1e9:.6f} GHz",
    )
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Expected Information Gain")
    plt.title("Information Gain vs Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return test_frequencies, info_gains, optimal_freq


def main():
    """
    Main demonstration function.
    """
    print("Sequential Bayesian Experiment Design for NV Center ODMR")
    print("Based on Dushenko et al., Phys. Rev. Applied 14, 054036 (2020)")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run comparison
    sequential_history, grid_history, sequential_result, grid_result = compare_strategies()

    # Visualize results
    print("\nGenerating visualizations...")

    # Plot Sequential Bayesian results
    visualize_adaptive_strategy(
        sequential_history, sequential_result, "Sequential Bayesian Experiment Design"
    )

    # Plot Grid Scan results
    visualize_adaptive_strategy(grid_history, grid_result, "Conventional Grid Scan")

    # Demonstrate information gain
    test_freqs, info_gains, optimal_freq = demonstrate_information_gain()

    # Show all plots
    plt.show()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Sequential Bayesian Experiment Design provides:")
    print("• Significant reduction in required measurements")
    print("• Improved accuracy through optimal measurement selection")
    print("• Real-time adaptive strategy based on accumulated data")
    print("• Information-theoretic optimization of experimental design")
    print("\nThis approach is particularly valuable for:")
    print("• Expensive or time-consuming measurements")
    print("• High-precision parameter estimation")
    print("• Automated experimental systems")
    print("• Quantum sensing applications")


if __name__ == "__main__":
    main()
