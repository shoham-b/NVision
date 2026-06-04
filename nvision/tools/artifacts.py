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


def merge_locator_results_with_existing(df_loc: pl.DataFrame, out_dir: Path, log: logging.Logger, *, no_cache: bool = False) -> pl.DataFrame:  # noqa: C901
    """Concatenate with on-disk CSV when present, keeping the newest row per scenario key.

    When ``no_cache`` is True the old CSV rows for the combinations being
    updated are discarded entirely — the fresh results replace them wholesale.
    """
    out_path = locator_results_path(out_dir)
    if not out_path.exists():
        return df_loc
    if len(df_loc) == 0:
        try:
            return pl.read_csv(out_path)
        except Exception as e:
            log.warning("Could not load existing locator_results.csv: %s", e)
            return df_loc
    # Always merge to ensure we discard all old repeats/attempts for the updated
    # combinations while fully preserving other combinations on disk.
    try:
        old_df = pl.read_csv(out_path)

        # Purge any old rows for the exact combinations being updated to prevent persisting old/different repeats
        if "generator" in df_loc.columns and "noise" in df_loc.columns and "strategy" in df_loc.columns:
            df_loc = df_loc.with_columns([
                pl.col("generator").cast(pl.String, strict=False),
                pl.col("noise").cast(pl.String, strict=False),
                pl.col("strategy").cast(pl.String, strict=False),
            ])
            for col in ("max_steps", "seed"):
                if col in df_loc.columns:
                    df_loc = df_loc.with_columns(pl.col(col).cast(pl.Int64, strict=False))
                if col in old_df.columns:
                    old_df = old_df.with_columns(pl.col(col).cast(pl.Int64, strict=False))

            old_df = old_df.with_columns([
                pl.col("generator").cast(pl.String, strict=False) if "generator" in old_df.columns else pl.lit(None).alias("generator"),
                pl.col("noise").cast(pl.String, strict=False) if "noise" in old_df.columns else pl.lit(None).alias("noise"),
                pl.col("strategy").cast(pl.String, strict=False) if "strategy" in old_df.columns else pl.lit(None).alias("strategy"),
            ])

            join_cols = ["generator", "noise", "strategy"]
            if "max_steps" in df_loc.columns and "max_steps" in old_df.columns:
                join_cols.append("max_steps")
            if "seed" in df_loc.columns and "seed" in old_df.columns:
                join_cols.append("seed")

            updated_combos = df_loc.select(join_cols).unique()
            old_df = old_df.join(updated_combos, on=join_cols, how="anti")

        # Backward compatibility: map 'repeat' column to 'attempt' if 'attempt' is missing
        if "repeat" in old_df.columns and "attempt" not in old_df.columns:
            old_df = old_df.rename({"repeat": "attempt"})

        key = ("generator", "noise", "strategy", "max_steps", "seed", "attempt")
        # Ensure old_df has all columns in key to prevent set(key).issubset failure
        for col in key:
            if col not in old_df.columns:
                old_df = old_df.with_columns(pl.lit(None).alias(col))

        # Also ensure df_loc has all columns in key (just in case)
        for col in key:
            if col not in df_loc.columns:
                df_loc = df_loc.with_columns(pl.lit(None).alias(col))

        # Make sure types are aligned for key columns
        for col in ("max_steps", "seed", "attempt"):
            if col in old_df.columns:
                old_df = old_df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
            if col in df_loc.columns:
                df_loc = df_loc.with_columns(pl.col(col).cast(pl.Int64, strict=False))

        # Align all column datatypes that exist in both dataframes to prevent concat/vstack errors
        for col in set(old_df.columns).intersection(df_loc.columns):
            type_old = old_df.schema[col]
            type_new = df_loc.schema[col]
            if type_old != type_new:
                if type_old == pl.Null:
                    old_df = old_df.with_columns(pl.col(col).cast(type_new, strict=False))
                elif type_new == pl.Null:
                    df_loc = df_loc.with_columns(pl.col(col).cast(type_old, strict=False))
                elif type_old.is_numeric() and type_new.is_numeric():
                    old_df = old_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                    df_loc = df_loc.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                elif type_old.is_numeric() and not type_new.is_numeric():
                    # e.g. old CSV was Float64 but new data was read/inferred as String
                    df_loc = df_loc.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                elif type_new.is_numeric() and not type_old.is_numeric():
                    # e.g. old CSV was corrupted to String, new data is Float64 — recover
                    old_df = old_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                else:
                    old_df = old_df.with_columns(pl.col(col).cast(pl.String, strict=False))
                    df_loc = df_loc.with_columns(pl.col(col).cast(pl.String, strict=False))

        merged = pl.concat([old_df, df_loc], how="diagonal").unique(
            subset=list(key),
            keep="last",
            maintain_order=True,
        )
        # Normalize 'repeats' column to the maximum for each scenario group to ensure consistency
        group_cols = ["generator", "noise", "strategy", "max_steps", "seed"]
        if "repeats" in merged.columns and set(group_cols).issubset(merged.columns):
            merged = merged.with_columns(pl.col("repeats").max().over(group_cols))
        return merged
    except Exception as e:
        log.warning("Could not merge with existing CSV (perhaps schema changed!): %s", e, exc_info=True)
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
    *,
    no_cache: bool = False,
) -> None:
    """Merge new plots into the existing manifest while pruning stale/missing entries.

    1. Identifies combinations (gen, noise, strat, max_steps, seed) in the new manifest and removes
       ALL old entries matching those combinations, regardless of the repeat/attempt number.
    2. Prunes any old entry whose referenced file ('path') no longer exists on disk.
    3. Keeps summary plots fresh by excluding old summary rows.
    4. When a new run changes the Gauss noise grid for a generator (e.g. max_noise went from
       0.2 to 0.1), drops old Gauss entries for that generator whose sigma is no longer in the
       current run's noise set.  This prevents stale ``Gauss(0.2)`` entries from appearing in
       the UI slider when newer runs only go up to ``Gauss(0.1)``.
    """

    manifest_path = plots_manifest_path(out_dir)
    if not manifest_path.exists():
        return

    try:
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Strip heavy fields from old entries to prevent manifest bloat
        old_manifest = [_strip_heavy_fields(e) for e in old_manifest]

        # Identify combinations being updated: (generator, noise, strategy, max_steps, seed)
        updated_combinations = {
            (
                str(row.get("generator")),
                str(row.get("noise")),
                str(row.get("strategy")),
                row.get("max_steps"),
                row.get("seed"),
            )
            for row in plot_manifest
            if all(row.get(k) is not None for k in ["generator", "noise", "strategy"])
        }

        # Build the set of (generator, noise) Gauss pairs present in the NEW run.
        # If the new run includes ANY Gauss entries for a generator, we treat that generator's
        # Gauss noise grid as authoritative and prune old levels that are no longer present.
        new_gauss_noises_by_generator: dict[str, set[str]] = {}
        for row in plot_manifest:
            g = row.get("generator")
            n = row.get("noise")
            if g and n and isinstance(n, str) and "Gauss" in n:
                new_gauss_noises_by_generator.setdefault(str(g), set()).add(str(n))

        # Keep track of active file paths in the new manifest to avoid unlinking them
        new_paths = {str(row.get("path")) for row in plot_manifest if row.get("path")}

        import contextlib
        filtered_old: list[dict[str, object]] = []
        for entry in old_manifest:
            # Drop old summaries - they are rebuilt per run
            if entry.get("type") == "summary":
                continue

            g = entry.get("generator")
            n = entry.get("noise")
            s = entry.get("strategy")
            ms = entry.get("max_steps")
            sd = entry.get("seed")

            # Discard any old entry matching the updated combinations, even for different repeats.
            if g and n and s and (str(g), str(n), str(s), ms, sd) in updated_combinations:
                # Delete the old plot file from disk only if it is a stray file (not in the new manifest)
                path_str = entry.get("path")
                if path_str and str(path_str) not in new_paths:
                    file_path = out_dir / str(path_str)
                    with contextlib.suppress(Exception):
                        file_path.unlink(missing_ok=True)
                continue

            # Prune stale Gauss entries: if the new run changed the Gauss noise grid for this
            # generator, drop any old Gauss level that is no longer in the new run.
            if g and n and isinstance(n, str) and "Gauss" in n:
                new_gauss_set = new_gauss_noises_by_generator.get(str(g))
                if new_gauss_set is not None and str(n) not in new_gauss_set:
                    # Old Gauss level no longer in current run — prune it and its file
                    path_str = entry.get("path")
                    if path_str and str(path_str) not in new_paths:
                        file_path = out_dir / str(path_str)
                        with contextlib.suppress(Exception):
                            file_path.unlink(missing_ok=True)
                    log.debug(
                        "Pruning stale Gauss entry noise=%s for generator=%s (not in current noise grid).",
                        n,
                        g,
                    )
                    continue

            # Aggressive pruning: check if file exists; auto-migrate .json -> .json.gz
            path_str = entry.get("path")
            if path_str:
                file_path = out_dir / str(path_str)
                if not file_path.exists():
                    # Try the .json.gz counterpart before dropping
                    if str(path_str).endswith(".json") and not str(path_str).endswith(".json.gz"):
                        gz_path = out_dir / (str(path_str) + ".gz")
                        if gz_path.exists():
                            entry = dict(entry)
                            entry["path"] = str(path_str) + ".gz"
                        else:
                            continue
                    else:
                        continue
            elif entry.get("type") == "scan" and entry.get("generator") != "Dummy-Generator":
                # Scan entries without a path (and not the dummy) are invalid
                continue

            filtered_old.append(entry)

        # Prepend old entries so new ones are at the end (maintaining order)
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
    # Strip heavy in-memory fields before writing — plot_data can be 600 KB per entry
    _MANIFEST_HEAVY = frozenset({"content", "content_bin", "plot_data", "_bytes"})
    stripped = [{k: v for k, v in e.items() if k not in _MANIFEST_HEAVY} for e in plot_manifest]
    with path.open("w", encoding="utf-8") as f:
        json.dump(stripped, f)
    return path


def purge_artifacts_for_combination(
    out_dir: Path,
    generator: str,
    noise: str,
    strategy: str,
    max_steps: int | None,
    seed: int | None,
    log: logging.Logger,
) -> None:
    """Delete repeat graphs and plots manifest entries for a purged combination."""
    # 1. Generate task slug corresponding to this combination
    from nvision.tools.paths import slugify
    slug = "_".join(slugify(p) for p in (generator, noise, strategy))

    # 2. Delete individual repeat graphs from graphs/scans and graphs/bayes directories
    prefix = f"{slug}_r"
    scans_dir = out_dir / "graphs" / "scans"
    bayes_dir = out_dir / "graphs" / "bayes"
    for directory in [scans_dir, bayes_dir]:
        if directory.exists():
            for p in directory.iterdir():
                if p.is_file() and p.name.startswith(prefix):
                    try:
                        p.unlink()
                        log.debug("Purged old artifact graph: %s", p.name)
                    except Exception as exc:
                        log.warning("Failed to delete old artifact graph %s: %s", p.name, exc)

    # 3. Update plots_manifest.json to remove entries matching this combination
    manifest_path = plots_manifest_path(out_dir)
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            new_manifest = []
            removed_count = 0
            for entry in manifest:
                g = entry.get("generator")
                n = entry.get("noise")
                s = entry.get("strategy")
                ms = entry.get("max_steps")
                sd = entry.get("seed")

                # Check for match (cast to string for safe comparison)
                if (str(g) == str(generator) and
                    str(n) == str(noise) and
                    str(s) == str(strategy) and
                    (ms is None or max_steps is None or int(ms) == int(max_steps)) and
                    (sd is None or seed is None or int(sd) == int(seed))):
                    removed_count += 1
                    continue
                new_manifest.append(entry)

            if removed_count > 0:
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(new_manifest, f, indent=2)
                log.debug("Removed %s entries from plots_manifest.json for combination %s/%s/%s",
                          removed_count, generator, noise, strategy)
        except Exception as exc:
            log.warning("Failed to update plots_manifest.json during cache purge: %s", exc)

