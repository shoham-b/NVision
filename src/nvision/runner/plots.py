"""Per-repeat plot generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from nvision.models.experiment import CoreExperiment
from nvision.viz import Viz

log = logging.getLogger(__name__)


def generate_attempt_plots(
    viz: Viz,
    entry_base: dict[str, Any],
    attempt_idx_in_combo: int,
    current_scan: CoreExperiment,
    current_history_df: pl.DataFrame,
    noise_obj: Any,
    strat_obj: Any,
    slug_base: str,
    out_dir: Path,
    scans_dir: Path,
    bayes_dir: Path,
) -> list[dict[str, Any]]:
    """Generate visualizations and graph manifest entries for a single repeat."""
    attempt_slug = f"{slug_base}_r{attempt_idx_in_combo + 1}"
    out_path = scans_dir / f"{attempt_slug}.html"

    history_with_phase = current_history_df

    # Annotate coarse vs fine phase if this is a two-phase sweep strategy.
    try:
        from nvision.sim.locs.two_phase_sweep_locator import TwoPhaseSweepLocator  # noqa: PLC0415

        if isinstance(strat_obj, dict) and strat_obj.get("class") is TwoPhaseSweepLocator:
            config = strat_obj.get("config", {}) or {}
            phase1_max_steps = int(config.get("phase1_max_steps", 0))
            if "step" in current_history_df.columns and phase1_max_steps > 0:
                history_with_phase = current_history_df.with_columns(
                    pl.when(pl.col("step") < phase1_max_steps)
                    .then(pl.lit("coarse"))
                    .otherwise(pl.lit("fine"))
                    .alias("phase")
                )
    except Exception:
        history_with_phase = current_history_df

    viz.plot_scan_measurements(
        current_scan,
        history_with_phase,
        out_path,
        over_frequency_noise=noise_obj.over_frequency_noise if noise_obj else None,
    )

    scan_entry = entry_base.copy()
    scan_entry["type"] = "scan"
    scan_entry["path"] = out_path.relative_to(out_dir).as_posix()
    return [scan_entry]
