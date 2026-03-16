from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from nvision.sim.locs.base import ScanBatch
from nvision.viz import Viz

log = logging.getLogger(__name__)


def generate_attempt_plots(
    viz: Viz,
    entry_base: dict[str, Any],
    attempt_idx_in_combo: int,
    current_scan: ScanBatch,
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
    try:
        from nvision.sim.locs.nv_center.sequential_bayesian_locator import NVCenterSequentialBayesianLocator

        locator_instance = None
        if hasattr(strat_obj, "_get_locator"):
            locator_instance = strat_obj._get_locator(attempt_idx_in_combo)
        elif isinstance(strat_obj, NVCenterSequentialBayesianLocator):
            locator_instance = strat_obj

        if (
            locator_instance
            and hasattr(locator_instance, "posterior_history")
            and hasattr(locator_instance, "freq_grid")
        ):
            model_history = []
            if hasattr(locator_instance, "parameter_history"):
                for params in locator_instance.parameter_history:
                    model_history.append(locator_instance.odmr_model(locator_instance.freq_grid, params))

            bayes_anim_path = bayes_dir / f"{attempt_slug}_posterior_anim.html"
            viz.plot_posterior_animation(
                locator_instance.posterior_history,
                locator_instance.freq_grid,
                bayes_anim_path,
                model_history=model_history,
            )
            entries.append(
                {
                    "type": "bayesian_interactive",
                    "generator": entries[0]["generator"],
                    "noise": entries[0]["noise"],
                    "strategy": entries[0]["strategy"],
                    "repeat": attempt_idx_in_combo + 1,
                    "path": bayes_anim_path.relative_to(out_dir).as_posix(),
                }
            )

            if hasattr(locator_instance, "parameter_history"):
                param_conv_path = bayes_dir / f"{attempt_slug}_param_convergence.html"
                viz.plot_parameter_convergence(
                    locator_instance.parameter_history,
                    param_conv_path,
                )
                entries.append(
                    {
                        "type": "bayesian_parameter_convergence",
                        "generator": entries[0]["generator"],
                        "noise": entries[0]["noise"],
                        "strategy": entries[0]["strategy"],
                        "repeat": attempt_idx_in_combo + 1,
                        "path": param_conv_path.relative_to(out_dir).as_posix(),
                    }
                )

    except Exception as e:
        log.warning(f"Failed to generate Bayesian animation: {e}")
