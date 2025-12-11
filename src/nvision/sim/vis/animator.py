"""Visualization tools for locator progress."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.animation import FuncAnimation
from scipy.special import voigt_profile
from scipy.stats import cauchy


def odmr_model(
    frequency: np.ndarray,
    params: dict[str, Any],
    distribution: str = "voigt-zeeman",
) -> np.ndarray:
    """Reconstruct the ODMR signal based on parameters and distribution type."""
    f0 = params.get("est_frequency", np.mean(frequency))
    gamma = params.get("est_linewidth", 10e6)
    amplitude = params.get("est_amplitude", 0.05)
    bg = params.get("est_background", 1.0)

    if distribution == "lorentzian":
        scale = gamma / 2
        lorentzian = amplitude * math.pi * scale * cauchy.pdf(frequency, loc=f0, scale=scale)
        return bg - lorentzian

    if distribution == "voigt":
        sigma = params.get("est_gaussian_width", gamma)
        peak_val = voigt_profile(0, sigma, gamma)
        if peak_val < 1e-9:
            return np.full_like(frequency, bg, dtype=float)
        profile = voigt_profile(frequency - f0, sigma, gamma)
        return bg - amplitude * profile / peak_val

    if distribution == "voigt-zeeman":
        split = params.get("est_split", 0.0)
        k_np = params.get("est_k_np", 3.0)

        # If sigma is not estimated, derive it or use default
        sigma = params.get("est_gaussian_width", max(split / 10.0, 1e-9))

        peak_val = voigt_profile(0, sigma, gamma)
        if peak_val < 1e-12:
            return np.full_like(frequency, bg, dtype=float)

        f_left = f0 - split
        f_center = f0
        f_right = f0 + split

        w_left = 1.0 / max(k_np, 1e-9)
        w_center = 1.0
        w_right = max(k_np, 1e-9)

        v_left = voigt_profile(frequency - f_left, sigma, gamma) / peak_val
        v_center = voigt_profile(frequency - f_center, sigma, gamma) / peak_val
        v_right = voigt_profile(frequency - f_right, sigma, gamma) / peak_val

        composite = w_left * v_left + w_center * v_center + w_right * v_right
        return bg - amplitude * composite / max(k_np, 1e-9)

    return np.full_like(frequency, bg, dtype=float)


def animate_locator_progress(  # noqa: C901
    history_df: pl.DataFrame,
    output_path: str | Path,
    true_params: dict[str, float] | None = None,
    distribution: str = "voigt-zeeman",
    fps: int = 10,
):
    """
    Create an animation showing the locator's progress.

    Args:
        history_df: DataFrame containing measurement history and estimates.
        output_path: Path to save the animation (e.g., 'animation.mp4' or 'animation.gif').
        true_params: Dictionary of true parameters for ground truth comparison.
                     Must contain 'frequency', 'split' (for 3-peak), etc.
        distribution: The distribution model assumed by the locator.
        fps: Frames per second for the animation.
    """
    output_path = Path(output_path)

    # Filter out steps where we don't have estimates yet (e.g. very first step if not captured)
    # But our run_locator modification captures estimates at each step.

    history_df["step"].to_list()
    measurements_x = history_df["x"].to_list()
    measurements_y = history_df["signal_values"].to_list()

    # Determine frequency range for plotting
    x_min = min(measurements_x)
    x_max = max(measurements_x)
    margin = (x_max - x_min) * 0.1 if x_max != x_min else 1e7
    freq_grid = np.linspace(x_min - margin, x_max + margin, 1000)

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Plot 1: Distribution
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Signal")
    ax1.set_title("Assumed Distribution vs. Measurements")
    ax1.grid(True, alpha=0.3)

    # Plot true distribution if available
    if true_params:
        # Map true params to expected format (est_ prefix not needed for truth, but helper uses it)
        # We create a dict with 'est_' keys for the helper function
        truth_for_helper = {f"est_{k}": v for k, v in true_params.items()}
        true_signal = odmr_model(freq_grid, truth_for_helper, distribution=distribution)
        ax1.plot(freq_grid, true_signal, "k--", alpha=0.5, label="True Distribution")

    # Elements to update
    (line_est,) = ax1.plot([], [], "b-", linewidth=2, label="Estimated Distribution")
    scat = ax1.scatter([], [], c="r", marker="x", alpha=0.6, label="Measurements")
    (point_current,) = ax1.plot([], [], "ro", markersize=8, label="Current Point")

    ax1.legend(loc="upper right")

    # Plot 2: Metric (Outer Peak Distance Error)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Outer Peak Dist. Error (Hz)")
    ax2.set_title("Convergence Metric")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    metric_history = []
    step_history = []
    (line_metric,) = ax2.plot([], [], "g.-")

    # Pre-calculate metrics if truth is available
    true_outer_dist = 0.0
    if true_params and "split" in true_params:
        true_outer_dist = 2 * true_params["split"]
    elif true_params and "x1" in true_params and "x3" in true_params:
        true_outer_dist = abs(true_params["x3"] - true_params["x1"])

    def init():
        line_est.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        point_current.set_data([], [])
        line_metric.set_data([], [])
        return line_est, scat, point_current, line_metric

    def update(frame_idx):
        row = history_df.row(frame_idx, named=True)
        step = row["step"]

        # Reconstruct signal
        # row contains 'est_frequency', 'est_split', etc.
        est_signal = odmr_model(freq_grid, row, distribution=distribution)
        line_est.set_data(freq_grid, est_signal)

        # Update measurements
        current_x = measurements_x[: frame_idx + 1]
        current_y = measurements_y[: frame_idx + 1]
        scat.set_offsets(np.c_[current_x, current_y])

        # Highlight current point
        point_current.set_data([row["x"]], [row["signal_values"]])

        # Update metric
        if true_outer_dist > 0:
            est_split = row.get("est_split")
            if est_split is not None:
                est_outer_dist = 2 * est_split
                error = abs(est_outer_dist - true_outer_dist)
                metric_history.append(error)
                step_history.append(step)
                line_metric.set_data(step_history, metric_history)

                # Adjust limits
                ax2.set_xlim(0, max(step, 10))
                if metric_history:
                    min_err = min(metric_history)
                    max_err = max(metric_history)
                    if min_err > 0:
                        ax2.set_ylim(min_err * 0.5, max_err * 2)

        return line_est, scat, point_current, line_metric

    ani = FuncAnimation(fig, update, frames=len(history_df), init_func=init, blit=True, interval=1000 / fps)

    # Save
    writer = "pillow" if output_path.suffix == ".gif" else "ffmpeg"
    try:
        ani.save(output_path, writer=writer, dpi=100)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Fallback to saving just the final frame as an image if animation fails (e.g. no ffmpeg)
        fig.savefig(output_path.with_suffix(".png"))
        print(f"Saved final frame to {output_path.with_suffix('.png')} instead.")

    plt.close(fig)
