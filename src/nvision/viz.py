from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

# Use a soft import guard to avoid import errors in environments without matplotlib at build time
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as _e:  # pragma: no cover - plotting backend issues are environment-specific
    plt = None  # type: ignore


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_experiment_summary(df: pl.DataFrame, out_dir: Path) -> Sequence[Path]:
    """Plot RMSE by (noise, strategy) for each generator in experiment results.

    Returns list of saved image paths.
    """
    _ensure_out_dir(out_dir)
    paths: list[Path] = []
    if plt is None or df.height == 0:
        return paths


def plot_scan_measurements(scan, history: pl.DataFrame, out_path: Path) -> Path:
    """Plot the true scan signal distribution and overlay sampled measurements.

    - True signal: computed densely across [x_min, x_max].
    - Measurements: points from `history` colored by step order (gradient).
    """
    _ensure_out_dir(out_path.parent)
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
        cmap = plt2.cm.viridis
        colors = [cmap(i / max(1, len(steps) - 1)) for i in steps]
        ax.scatter(xs_s, ys_s, c=colors, s=20, edgecolor="k", linewidths=0.3, zorder=3)
        # Colorbar
        from matplotlib.cm import ScalarMappable  # type: ignore
        from matplotlib.colors import Normalize  # type: ignore

        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=max(1, len(steps) - 1)), cmap=cmap)
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


def plot_locator_summary(df: pl.DataFrame, out_dir: Path) -> Sequence[Path]:
    """Create comparison plots for locator sweeps.

    - For single-peak generators (name contains 'OnePeak'): plot abs_err_x
      and uncert by (noise, strategy).
    - For two-peak generators (name contains 'TwoPeak'): plot pair_rmse and
      uncert_sep by (noise, strategy).
    """
    _ensure_out_dir(out_dir)
    paths: list[Path] = []
    if plt is None or df.height == 0:
        return paths

    # Single-peak
    singles = df.filter(pl.col("generator").str.contains("OnePeak"))
    if singles.height > 0 and all(col in singles.columns for col in ["abs_err_x", "uncert"]):
        # One figure per generator mode
        for gen in sorted(set(singles.get_column("generator").to_list())):
            sub = singles.filter(pl.col("generator") == gen)
            if sub.height == 0:
                continue
            pdf = sub.select(["noise", "strategy", "abs_err_x", "uncert"]).to_pandas()
            # Two subplots: error and uncertainty
            import matplotlib.pyplot as plt2  # local alias to satisfy type checkers

            fig, axes = plt2.subplots(2, 1, figsize=(11, 8), sharex=True)
            for ax, col, title in zip(
                axes,
                ["abs_err_x", "uncert"],
                ["Abs Error", "Uncertainty"],
                strict=False,
            ):
                piv = pdf.pivot(index="noise", columns="strategy", values=col).sort_index()
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

    # Two-peak
    doubles = df.filter(pl.col("generator").str.contains("TwoPeak"))
    if doubles.height > 0 and all(col in doubles.columns for col in ["pair_rmse", "uncert_sep"]):
        for gen in sorted(set(doubles.get_column("generator").to_list())):
            sub = doubles.filter(pl.col("generator") == gen)
            if sub.height == 0:
                continue
            pdf = sub.select(["noise", "strategy", "pair_rmse", "uncert_sep"]).to_pandas()
            import matplotlib.pyplot as plt2

            fig, axes = plt2.subplots(2, 1, figsize=(11, 8), sharex=True)
            for ax, col, title in zip(
                axes,
                ["pair_rmse", "uncert_sep"],
                ["Pair RMSE", "Uncertainty (sep)"],
                strict=False,
            ):
                piv = pdf.pivot(index="noise", columns="strategy", values=col).sort_index()
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

    return paths
