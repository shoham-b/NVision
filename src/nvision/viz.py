from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import polars as pl

from nvision.pathutils import ensure_out_dir


def plot_experiment_summary(df: pl.DataFrame, out_dir: Path) -> Sequence[Path]:
    """Plot RMSE by (noise, strategy) for each generator in experiment results.

    Returns list of saved image paths.
    """
    ensure_out_dir(out_dir)
    paths: list[Path] = []
    if plt is None or df.height == 0:
        return paths

    # Not yet implemented, return empty list to satisfy types and callers.
    return paths


def plot_scan_measurements(scan, history: pl.DataFrame, out_path: Path) -> Path:
    """Plot the true scan signal distribution and overlay sampled measurements.

    - True signal: computed densely across [x_min, x_max].
    - Measurements: points from `history` colored by step order (gradient).
    """
    ensure_out_dir(out_path.parent)
    if plt is None:
        return out_path

    try:
        import numpy as np  # type: ignore
    except Exception:
        # Fallback: coarse python loop if numpy unavailable
        xs = [scan.x_min + i * (scan.x_max - scan.x_min) / 500 for i in range(501)]
        ys = [float(scan.signal(x)) for x in xs]
    else:
        xs = np.linspace(scan.x_min, scan.x_max, 1000)
        ys = [float(scan.signal(x)) for x in xs]

    # Build figure
    import matplotlib.pyplot as plt2  # local alias

    fig, ax = plt2.subplots(figsize=(10, 5))
    ax.plot(xs, ys, label="true signal", color="tab:blue", alpha=0.8)

    # Overlay samples with gradient by step
    if history.height > 0:
        # Ensure correct order by acquisition if available (index order)
        steps = list(range(history.height))
        xs_s = history.get_column("x").to_list() if "x" in history.columns else []
        ys_s = (
            history.get_column("signal_values").to_list()
            if "signal_values" in history.columns
            else []
        )
        # Use get_cmap to avoid static-typing lookup issues
        cmap = plt2.get_cmap("viridis")
        from matplotlib.cm import ScalarMappable  # type: ignore
        from matplotlib.colors import Normalize  # type: ignore

        norm = Normalize(vmin=0, vmax=max(1, len(steps) - 1))
        ax.scatter(
            xs_s,
            ys_s,
            c=steps,
            cmap=cmap,
            norm=norm,
            s=20,
            edgecolor="k",
            linewidths=0.3,
            zorder=3,
        )
        # Colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("step")

    ax.set_xlabel("x")
    ax.set_ylabel("signal")
    ax.set_title("Scan with sampled measurements")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=150)
    plt2.close(fig)
    return out_path


def _plot_pivot_from_polars(
    pivot_pl: pl.DataFrame, title: str, ylabel: str, out_path: Path
) -> Path:
    """Plot a bar chart from a polars pivoted dataframe.

    Expected format: first column is the index (noise), remaining columns are strategies.
    """
    import matplotlib.pyplot as plt2
    import numpy as np

    # Extract index and strategy columns
    cols = pivot_pl.columns
    if not cols:
        return out_path
    index_col = cols[0]
    strategies = [c for c in cols if c != index_col]
    noises = pivot_pl.get_column(index_col).to_list()
    if len(strategies) == 0:
        # Nothing to plot
        return out_path

    matrix = [pivot_pl.get_column(s).to_list() for s in strategies]
    # Convert to numpy array shape (n_strategies, n_noises)
    arr = np.array(matrix, dtype=float)

    x = np.arange(len(noises))
    width = 0.8 / max(1, arr.shape[0])

    fig, ax = plt2.subplots(figsize=(11, 6))
    for i, row in enumerate(arr):
        ax.bar(x + i * width, row, width=width, label=strategies[i])

    ax.set_xticks(x + width * (arr.shape[0] - 1) / 2)
    ax.set_xticklabels(noises, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=150)
    plt2.close(fig)
    return out_path


def _plot_single_peak(sub: pl.DataFrame, gen: str, out_dir: Path, paths: list[Path]) -> None:
    """Handle plotting for a single-peak generator subset."""
    # Try to use pandas for convenience; fall back to polars-based pivot
    pdf: Any = None
    use_pandas = False
    try:
        pdf = sub.select(["noise", "strategy", "abs_err_x", "uncert"]).to_pandas()
        use_pandas = True
    except Exception:
        use_pandas = False

    if use_pandas:
        # Two subplots: error and uncertainty
        import matplotlib.pyplot as plt2
        import pandas as pd

        fig, axes = plt2.subplots(2, 1, figsize=(11, 8), sharex=True)
        for ax, col, title in zip(
            axes,
            ["abs_err_x", "uncert"],
            ["Abs Error", "Uncertainty"],
            strict=False,
        ):
            # pdf is a pandas DataFrame here; ensure type
            assert pdf is not None
            pdf_pd = pd.DataFrame(pdf)
            piv = pdf_pd.pivot(index="noise", columns="strategy", values=col).sort_index()
            piv.plot(kind="bar", ax=ax)
            ax.set_title(f"{title} — {gen}")
            ax.set_ylabel(col)
            ax.grid(True, axis="y", alpha=0.3)
        axes[-1].set_xlabel("Noise")
        fig.tight_layout()
        path = out_dir / f"locator_summary_single_{gen}.png"
        fig.savefig(path.as_posix(), dpi=150)
        plt2.close(fig)
        paths.append(path)
    else:
        # Build polars pivot for both columns and plot using _plot_pivot_from_polars
        for col, title in zip(["abs_err_x", "uncert"], ["Abs Error", "Uncertainty"], strict=False):
            piv_pl = (
                sub.select(["noise", "strategy", col])
                .group_by(["noise", "strategy"])  # type: ignore[attr-defined]
                .agg(pl.col(col).mean())
                .pivot(values=col, index="noise", columns="strategy")
            )  # type: ignore[attr-defined]
            path = out_dir / f"locator_summary_single_{gen}_{col}.png"
            _plot_pivot_from_polars(piv_pl, f"{title} — {gen}", col, path)
            paths.append(path)


def _plot_double_peak(sub: pl.DataFrame, gen: str, out_dir: Path, paths: list[Path]) -> None:
    """Handle plotting for a two-peak generator subset."""
    pdf: Any = None
    use_pandas = False
    try:
        pdf = sub.select(["noise", "strategy", "pair_rmse", "uncert_sep"]).to_pandas()
        use_pandas = True
    except Exception:
        use_pandas = False

    if use_pandas:
        import matplotlib.pyplot as plt2
        import pandas as pd

        fig, axes = plt2.subplots(2, 1, figsize=(11, 8), sharex=True)
        for ax, col, title in zip(
            axes,
            ["pair_rmse", "uncert_sep"],
            ["Pair RMSE", "Uncertainty (sep)"],
            strict=False,
        ):
            assert pdf is not None
            pdf_pd = pd.DataFrame(pdf)
            piv = pdf_pd.pivot(index="noise", columns="strategy", values=col).sort_index()
            piv.plot(kind="bar", ax=ax)
            ax.set_title(f"{title} — {gen}")
            ax.set_ylabel(col)
            ax.grid(True, axis="y", alpha=0.3)
        axes[-1].set_xlabel("Noise")
        fig.tight_layout()
        path = out_dir / f"locator_summary_double_{gen}.png"
        fig.savefig(path.as_posix(), dpi=150)
        plt2.close(fig)
        paths.append(path)
    else:
        # Build polars pivot for both columns and plot using _plot_pivot_from_polars
        metric_pairs = zip(
            ["pair_rmse", "uncert_sep"],
            ["Pair RMSE", "Uncertainty (sep)"],
            strict=False,
        )
        for col, title in metric_pairs:
            piv_pl = (
                sub.select(["noise", "strategy", col])
                .group_by(["noise", "strategy"])  # type: ignore[attr-defined]
                .agg(pl.col(col).mean())
                .pivot(values=col, index="noise", columns="strategy")
            )  # type: ignore[attr-defined]
            path = out_dir / f"locator_summary_double_{gen}_{col}.png"
            _plot_pivot_from_polars(piv_pl, f"{title} — {gen}", col, path)
            paths.append(path)


def plot_locator_summary(df: pl.DataFrame, out_dir: Path) -> Sequence[Path]:
    """Create comparison plots for locator sweeps.

    - For single-peak generators (name contains 'OnePeak'): plot abs_err_x
      and uncert by (noise, strategy).
    - For two-peak generators (name contains 'TwoPeak'): plot pair_rmse and
      uncert_sep by (noise, strategy).
    """
    ensure_out_dir(out_dir)
    paths: list[Path] = []
    if plt is None or df.height == 0:
        return paths

    # Single-peak
    singles = df.filter(pl.col("generator").str.contains("OnePeak"))
    if singles.height > 0 and all(col in singles.columns for col in ["abs_err_x", "uncert"]):
        for gen in sorted(set(singles.get_column("generator").to_list())):
            sub = singles.filter(pl.col("generator") == gen)
            if sub.height == 0:
                continue
            _plot_single_peak(sub, gen, out_dir, paths)

    # Two-peak
    doubles = df.filter(pl.col("generator").str.contains("TwoPeak"))
    if doubles.height > 0 and all(col in doubles.columns for col in ["pair_rmse", "uncert_sep"]):
        for gen in sorted(set(doubles.get_column("generator").to_list())):
            sub = doubles.filter(pl.col("generator") == gen)
            if sub.height == 0:
                continue
            _plot_double_peak(sub, gen, out_dir, paths)

    return paths
