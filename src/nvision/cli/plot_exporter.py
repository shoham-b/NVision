from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from nvision.core.experiment import CoreExperiment
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
    """Generate the visualizations and graph entries for a single repeat."""
    entries: list[dict[str, Any]] = []
    attempt_slug = f"{slug_base}_r{attempt_idx_in_combo + 1}"
    out_path = scans_dir / f"{attempt_slug}.html"

    viz.plot_scan_measurements(
        current_scan,
        current_history_df,
        out_path,
        over_frequency_noise=noise_obj.over_frequency_noise if noise_obj else None,
    )

    scan_entry = entry_base.copy()
    scan_entry["type"] = "scan"
    scan_entry["path"] = out_path.relative_to(out_dir).as_posix()
    entries.append(scan_entry)

    __try_bayesian_plotting(viz, strat_obj, attempt_idx_in_combo, attempt_slug, out_dir, bayes_dir, entries)

    return entries


def __try_bayesian_plotting(
    viz: Viz,
    strat_obj: Any,
    attempt_idx_in_combo: int,
    attempt_slug: str,
    out_dir: Path,
    bayes_dir: Path,
    entries: list[dict[str, Any]],
):
    # v2-only: Bayesian (legacy) locator plots were removed during migration.
    _ = (viz, strat_obj, attempt_idx_in_combo, attempt_slug, out_dir, bayes_dir, entries)
    return
