from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nvision.tools.paths import ensure_out_dir

LOCATOR_RESULTS_CSV = "locator_results.csv"
PLOTS_MANIFEST_JSON = "plots_manifest.json"
RUN_STATUS_JSON = "run_status.json"


def locator_results_path(out_dir: Path) -> Path:
    return out_dir / LOCATOR_RESULTS_CSV


def plots_manifest_path(out_dir: Path) -> Path:
    return out_dir / PLOTS_MANIFEST_JSON


def run_status_path(out_dir: Path) -> Path:
    return out_dir / RUN_STATUS_JSON


def write_run_status(
    out_dir: Path,
    status: str,
    total_tasks: int | None = None,
    completed_tasks: int | None = None,
    started_at: str | None = None,
    pid: int | None = None,
    message: str | None = None,
) -> Path:
    """Write a lightweight JSON status file for the UI to poll."""
    path = run_status_path(out_dir)
    payload: dict[str, object] = {"status": status}
    if total_tasks is not None:
        payload["total_tasks"] = total_tasks
    if completed_tasks is not None:
        payload["completed_tasks"] = completed_tasks
    if started_at is not None:
        payload["started_at"] = started_at
    if pid is not None:
        payload["pid"] = pid
    if message is not None:
        payload["message"] = message
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_run_status(out_dir: Path) -> dict[str, object] | None:
    """Read the run status file if it exists, else None."""
    path = run_status_path(out_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


@dataclass(frozen=True, slots=True)
class ArtifactTree:
    out_dir: Path
    cache_dir: Path
    graphs_dir: Path
    scans_dir: Path
    bayes_dir: Path


def prepare_artifact_tree(out_dir: Path, *, clear_cache: bool = False) -> ArtifactTree:
    """Create standard cache/graphs subdirectories under ``out_dir``.

    When ``clear_cache`` is True, removes an existing cache directory before recreating it
    (used by ``nvision run --no-cache``).
    """
    ensure_out_dir(out_dir)
    cache_dir = out_dir / "cache"
    if clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    ensure_out_dir(cache_dir)

    graphs_dir = out_dir / "graphs"
    ensure_out_dir(graphs_dir)
    scans_dir = graphs_dir / "scans"
    ensure_out_dir(scans_dir)
    bayes_dir = graphs_dir / "bayes"
    ensure_out_dir(bayes_dir)

    return ArtifactTree(
        out_dir=out_dir,
        cache_dir=cache_dir,
        graphs_dir=graphs_dir,
        scans_dir=scans_dir,
        bayes_dir=bayes_dir,
    )


def merge_locator_results_with_existing(df_loc: pl.DataFrame, out_dir: Path, log: logging.Logger) -> pl.DataFrame:
    """Concatenate with on-disk CSV when present, keeping the newest row per scenario key."""
    out_path = locator_results_path(out_dir)
    if not out_path.exists():
        return df_loc
    if len(df_loc) == 0:
        try:
            return pl.read_csv(out_path)
        except Exception as e:
            log.warning("Could not load existing locator_results.csv: %s", e)
            return df_loc
    try:
        old_df = pl.read_csv(out_path)
        key = ("generator", "noise", "strategy", "repeat")
        if set(key).issubset(old_df.columns):
            return pl.concat([old_df, df_loc], how="diagonal").unique(
                subset=list(key),
                keep="last",
                maintain_order=True,
            )
    except Exception as e:
        log.warning("Could not merge with existing CSV (perhaps schema changed!): %s", e)
    return df_loc


def write_locator_results_csv(df_loc: pl.DataFrame, out_dir: Path) -> Path:
    out_path = locator_results_path(out_dir)
    df_loc.write_csv(out_path.as_posix())
    return out_path


def relativize_summary_plot_paths(summary_plots_meta: list[dict[str, object]], out_dir: Path) -> None:
    for meta in summary_plots_meta:
        meta["path"] = Path(meta["path"]).relative_to(out_dir).as_posix()


def _strip_heavy_fields(entry: dict[str, object]) -> dict[str, object]:
    """Return entry without heavy fields (content, plot_data) that bloat the manifest."""
    return {k: v for k, v in entry.items() if k not in ("content", "plot_data")}


def merge_run_plot_manifest_with_existing_on_disk(
    plot_manifest: list[dict[str, object]],
    out_dir: Path,
    log: logging.Logger,
) -> None:
    """Replace prior scan rows for combos in this run and drop stale summary rows; keep other scans."""
    manifest_path = plots_manifest_path(out_dir)
    if not manifest_path.exists():
        return
    try:
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Strip heavy fields from old entries to prevent manifest bloat
        old_manifest = [_strip_heavy_fields(e) for e in old_manifest]
        new_combos: set[tuple[object, object, object]] = set()
        for entry in plot_manifest:
            if entry.get("type") == "scan":
                new_combos.add((entry.get("generator"), entry.get("noise"), entry.get("strategy")))

        filtered_old: list[dict[str, object]] = []
        for entry in old_manifest:
            if entry.get("type") == "summary":
                continue
            if entry.get("type") == "scan":
                combo = (entry.get("generator"), entry.get("noise"), entry.get("strategy"))
                if combo in new_combos:
                    continue
            filtered_old.append(entry)

        plot_manifest[:] = filtered_old + plot_manifest
    except Exception as e:
        log.warning("Could not merge with existing plots_manifest.json: %s", e)


def dummy_scan_plot_manifest_entry() -> dict[str, object]:
    return {
        "type": "scan",
        "generator": "Dummy-Generator",
        "noise": "None",
        "strategy": "Dummy-Strategy",
        "repeat": 1,
        "repeat_total": 1,
        "stop_reason": "no_data",
        "abs_err_x": None,
        "uncert": None,
        "measurements": 0,
        "duration_ms": 0,
        "metrics": {},
        "path": "",
    }


def ensure_plot_manifest_non_empty(plot_manifest: list[dict[str, object]], log: logging.Logger) -> None:
    if plot_manifest:
        return
    log.warning("No plots were generated. Adding a dummy entry to manifest.")
    plot_manifest.append(dummy_scan_plot_manifest_entry())


def write_plots_manifest(plot_manifest: list[dict[str, object]], out_dir: Path) -> Path:
    path = plots_manifest_path(out_dir)
    # Stream JSON to disk to avoid MemoryError with large manifests
    with path.open("w", encoding="utf-8") as f:
        json.dump(plot_manifest, f, indent=2)
    return path
