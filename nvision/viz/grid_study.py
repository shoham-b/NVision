"""Parameter-grid study summaries — heatmaps and vs-noise trends over fixed signal params.

Activated when ``locator_results`` rows carry grid-coordinate columns
(``grid_saturation``/``grid_sigma_inhom`` for the saturation-Voigt study, or
``grid_linewidth``/``grid_c_total`` for the Lorentzian study). Each generator is a
fixed grid point, so aggregation is per grid cell across repeats.

Censoring convention: step metrics are reported as median (+ IQR) **over converged
repeats only**, always paired with ``convergence_rate`` (fraction of repeats that
reached frequency convergence). Cells that never converge are reported with the
failure breakdown (``infeasible_crlb`` vs other) instead of a step value.
"""

from __future__ import annotations

from typing import Any

import polars as pl

# (x_col, y_col, x_label, y_label) — checked in order; first pair with data wins.
_GRID_AXIS_CANDIDATES: tuple[tuple[str, str, str, str], ...] = (
    ("grid_saturation", "grid_sigma_inhom", "Saturation s", "sigma_inhom (Hz)"),
    ("grid_linewidth", "grid_c_total", "Linewidth (Hz)", "Contrast c_total"),
)

_METRIC = "splitting_converged_step"


def _detect_grid_axes(df: pl.DataFrame) -> tuple[str, str, str, str] | None:
    """Return the (x_col, y_col, x_label, y_label) of the active study grid, if any."""
    for x_col, y_col, x_label, y_label in _GRID_AXIS_CANDIDATES:
        if (
            x_col in df.columns
            and y_col in df.columns
            and df[x_col].drop_nulls().len() > 0
            and df[y_col].drop_nulls().len() > 0
        ):
            return x_col, y_col, x_label, y_label
    return None


def _num_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _cell_stats(sub: pl.DataFrame, x_col: str, y_col: str) -> pl.DataFrame:
    """Per-grid-cell censored aggregation (medians over converged repeats + rates)."""
    failure_col = "failure_reason" if "failure_reason" in sub.columns else "stop_reason"
    return sub.group_by([x_col, y_col]).agg(
        pl.len().alias("n_total"),
        pl.col(_METRIC).is_not_null().sum().alias("n_converged"),
        pl.col(_METRIC).median().alias("median_steps"),
        pl.col(_METRIC).quantile(0.25).alias("q25_steps"),
        pl.col(_METRIC).quantile(0.75).alias("q75_steps"),
        (pl.col(failure_col) == "infeasible_crlb").sum().alias("n_infeasible"),
    )


class GridStudyMixin:
    """Summary plots for parameter-grid study runs."""

    def plot_grid_study(self, df: pl.DataFrame) -> list[dict]:
        """Emit grid heatmaps + vs-noise trend data when the run is a parameter-grid study."""
        if df.is_empty() or _METRIC not in df.columns or "strategy" not in df.columns:
            return []
        axes = _detect_grid_axes(df)
        if axes is None:
            return []
        x_col, y_col, x_label, y_label = axes

        grid_df = df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
        if grid_df.is_empty():
            return []

        has_noise_sigma = "noise_sigma" in grid_df.columns and grid_df["noise_sigma"].drop_nulls().len() > 0

        entries: list[dict[str, Any]] = []
        for (strat,), strat_df in grid_df.partition_by("strategy", as_dict=True).items():
            noise_groups = (
                strat_df.partition_by("noise_sigma", as_dict=True)
                if has_noise_sigma
                else strat_df.partition_by("noise", as_dict=True)
            )
            for (noise_key,), noise_df in noise_groups.items():
                entry = self._plot_grid_heatmap(noise_df, strat, noise_key, x_col, y_col, x_label, y_label)
                if entry:
                    entries.append(entry)

            if has_noise_sigma:
                entry = self._plot_grid_vs_noise(strat_df, strat, x_col, y_col)
                if entry:
                    entries.append(entry)

        return entries

    def _plot_grid_heatmap(
        self,
        sub: pl.DataFrame,
        strat: str,
        noise_key: Any,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
    ) -> dict | None:
        """One heatmap: X x Y grid of censored median steps + convergence rates."""
        stats = _cell_stats(sub, x_col, y_col)
        if stats.is_empty():
            return None

        xs = sorted(stats[x_col].unique().to_list())
        ys = sorted(stats[y_col].unique().to_list())
        cell = {(r[x_col], r[y_col]): r for r in stats.iter_rows(named=True)}

        def _grid(key: str) -> list[list[Any]]:
            return [[(cell.get((x, y)) or {}).get(key) for x in xs] for y in ys]

        rates = [
            [(c["n_converged"] / c["n_total"]) if (c := cell.get((x, y))) and c["n_total"] else None for x in xs]
            for y in ys
        ]

        noise_label = f"sigma={noise_key:g}" if isinstance(noise_key, int | float) else str(noise_key)
        payload = {
            "_graph_type": "heatmap",
            "title": f"Steps to freq. convergence — {strat} ({noise_label})",
            "xaxis_title": x_label,
            "yaxis_title": y_label,
            "x": xs,
            "y": ys,
            "z": _grid("median_steps"),
            "q25": _grid("q25_steps"),
            "q75": _grid("q75_steps"),
            "convergence_rate": rates,
            "n_total": _grid("n_total"),
            "n_converged": _grid("n_converged"),
            "n_infeasible": _grid("n_infeasible"),
        }
        noise_slug = _num_slug(noise_key) if isinstance(noise_key, int | float) else str(noise_key)
        out_path = self._emit(payload, f"grid_heatmap_{strat}_n{noise_slug}.json.gz")
        return {
            "type": "grid_summary",
            "kind": "heatmap",
            "path": out_path.as_posix(),
            "strategy": strat,
            "noise": str(noise_key),
            "x_param": x_col,
            "y_param": y_col,
            "metric": _METRIC,
        }

    def _plot_grid_vs_noise(self, sub: pl.DataFrame, strat: str, x_col: str, y_col: str) -> dict | None:
        """Metric-vs-noise trend lines, one per grid point (censored medians + rates)."""
        stats = (
            sub.filter(pl.col("noise_sigma").is_not_null())
            .group_by([x_col, y_col, "noise_sigma"])
            .agg(
                pl.len().alias("n_total"),
                pl.col(_METRIC).is_not_null().sum().alias("n_converged"),
                pl.col(_METRIC).median().alias("median_steps"),
                pl.col(_METRIC).quantile(0.25).alias("q25_steps"),
                pl.col(_METRIC).quantile(0.75).alias("q75_steps"),
            )
            .sort([x_col, y_col, "noise_sigma"])
        )
        if stats.is_empty():
            return None

        series: list[dict[str, Any]] = []
        for (x_val, y_val), pt in stats.partition_by([x_col, y_col], as_dict=True).items():
            rates = [
                (n_conv / n_tot) if n_tot else None
                for n_conv, n_tot in zip(pt["n_converged"].to_list(), pt["n_total"].to_list(), strict=True)
            ]
            series.append(
                {
                    "name": f"{x_col.removeprefix('grid_')}={x_val:g}, {y_col.removeprefix('grid_')}={y_val:g}",
                    "x": pt["noise_sigma"].to_list(),
                    "y": pt["median_steps"].to_list(),
                    "q25": pt["q25_steps"].to_list(),
                    "q75": pt["q75_steps"].to_list(),
                    "convergence_rate": rates,
                }
            )

        payload = {
            "_graph_type": "chart",
            "title": f"Steps to freq. convergence vs noise — {strat}",
            "xaxis_title": "Noise sigma",
            "yaxis_title": "Median steps to freq. convergence (converged repeats)",
            "mode": "lines+markers",
            "series": series,
        }
        out_path = self._emit(payload, f"grid_vs_noise_{strat}.json.gz")
        return {
            "type": "grid_summary",
            "kind": "vs_noise",
            "path": out_path.as_posix(),
            "strategy": strat,
            "x_param": x_col,
            "y_param": y_col,
            "metric": _METRIC,
        }
