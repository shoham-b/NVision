"""Plotting utilities for Bayesian NV-center locators."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def plot_bo(
    output_path: Path,
    fig_size: tuple[int, int] = (12, 8),
    dpi: int = 100,
) -> Path | None:
    """Emit a deprecation warning — BO plotting is no longer supported."""
    warnings.warn(
        "Bayesian Optimization plotting is disabled as the dependency was removed.",
        stacklevel=2,
    )
    return None


def plot_posterior_history(
    freq_grid: np.ndarray,
    posterior_history: list[np.ndarray],
    output_path: Path,
) -> Path | None:
    """Plot the evolution of the posterior distribution.

    Args:
        freq_grid: 1-D frequency grid.
        posterior_history: List of posterior snapshots.
        output_path: Destination file path for the saved figure.

    Returns:
        *output_path* on success, ``None`` if *posterior_history* is empty.
    """
    if not posterior_history:
        return None

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    num_posteriors = len(posterior_history)
    indices_to_plot = np.linspace(0, num_posteriors - 1, 5, dtype=int)

    for i in indices_to_plot:
        ax.plot(freq_grid, posterior_history[i], label=f"Step {i}")

    ax.set_title("Posterior Distribution Evolution")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_convergence_stats(
    freq_grid: np.ndarray,
    posterior_history: list[np.ndarray],
    utility_history: list[float],
    output_path: Path,
) -> Path | None:
    """Plot uncertainty convergence and expected information gain.

    Args:
        freq_grid: 1-D frequency grid.
        posterior_history: List of posterior snapshots.
        utility_history: List of per-step utility values.
        output_path: Destination file path for the saved figure.

    Returns:
        *output_path* on success, ``None`` if *utility_history* is empty.
    """
    if not utility_history:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100, sharex=True)

    uncertainty_history = [np.sqrt(np.sum((freq_grid - np.sum(freq_grid * p)) ** 2 * p)) for p in posterior_history]

    steps = range(len(uncertainty_history))
    ax1.plot(steps, uncertainty_history, marker="o")
    ax1.set_title("Uncertainty Convergence")
    ax1.set_ylabel("Uncertainty (Hz)")
    ax1.grid(True)

    utility_steps = range(len(utility_history))
    ax2.plot(utility_steps, utility_history, marker="o", color="orange")
    ax2.set_title("Expected Information Gain")
    ax2.set_xlabel("Measurement Step")
    ax2.set_ylabel("Utility")
    ax2.grid(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), bbox_inches="tight")
    plt.close(fig)
    return output_path
