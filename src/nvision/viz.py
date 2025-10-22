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

    for gen in sorted(set(df.get_column("generator").to_list())):
        sub = df.filter(pl.col("generator") == gen)
        if sub.height == 0 or "rmse" not in sub.columns:
            continue
        pivot = (
            sub.select(["noise", "strategy", "rmse"])  # type: ignore[list-item]
            .to_pandas()
            .pivot(index="noise", columns="strategy", values="rmse")
            .sort_index()
        )
        ax = pivot.plot(
            kind="bar",
            figsize=(10, 6),
            title=f"Experiment RMSE — {gen}",
        )
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Noise")
        ax.grid(True, axis="y", alpha=0.3)
        fig = ax.get_figure()
        path = out_dir / f"experiment_summary_{gen}.png"
        fig.tight_layout()
        fig.savefig(path.as_posix(), dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


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
                ax.grid(True, axis="signal_values", alpha=0.3)
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
                ax.grid(True, axis="signal_values", alpha=0.3)
            axes[-1].set_xlabel("Noise")
            fig.tight_layout()
            path = out_dir / f"locator_summary_double_{gen}.png"
            fig.savefig(path.as_posix(), dpi=150)
            plt2.close(fig)
            paths.append(path)

    return paths
