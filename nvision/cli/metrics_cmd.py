"""``nvision metrics recalc`` — refresh UI metrics from cached run data."""

from __future__ import annotations

import fnmatch
import logging
import math
import random
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from nvision.cache import CacheBridge, stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.cli import defaults as cli_defaults
from nvision.cli.app_instance import app
from nvision.runner.cache import strip_heavy_fields
from nvision.sim.combinations import CombinationGrid
from nvision.sim.gen.nv_center_generator import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
)
from nvision.tools.artifacts import (
    prepare_artifact_tree,
    write_locator_results_csv,
    write_plots_manifest,
)
from nvision.tools.paths import ARTIFACTS_ROOT
from nvision.tools.utils import NVISION_RNG_SEED

console = Console()
log = logging.getLogger("nvision")

metrics_app = typer.Typer(help="Metric utilities.", pretty_exceptions_show_locals=False)
app.add_typer(metrics_app, name="metrics")

# NV center sweep bandwidth used in CRLB formula (matches executor x_min/x_max).
_CRLB_BANDWIDTH: float = DEFAULT_NV_CENTER_FREQ_X_MAX - DEFAULT_NV_CENTER_FREQ_X_MIN

# All scalar metric keys carried from main_result_row → scan entry.
_METRIC_KEYS: tuple[str, ...] = (
    "abs_err_x",
    "abs_err_x1",
    "abs_err_x2",
    "pair_rmse",
    "uncert",
    "uncert_pos",
    "uncert_sep",
    "final_entropy",
    "final_max_prob",
    "measurements",
    "duration_ms",
    "sweep_steps",
    "locator_steps",
    "sobol_baseline_steps",
    "sobol_freq_steps",
    "sobol_conv_diff",
    "sobol_freq_uncert_at_conv",
    "sobol_freq_err_at_conv",
    "sobol_difference",
    "steps_to_fb",
    "err_fb_at_milestone",
    "err_fc_at_milestone",
    "uncert_fb_at_milestone",
    "final_err_fb",
    "final_err_fc",
    "err_fb_diff",
    "err_fc_diff",
    "splitting_converged_step",
    "all_converged_step",
    "dips_detected",
    "total_dip_width",
    "min_dip_width",
    "sweep_efficiency",
)


def _crlb_min_obs(noise_std: float, linewidth: float, c_total: float, threshold_hz: float) -> int:
    """Return the minimum observation count N where 2×CRLB_phys < threshold_hz.

    Derived from 2×sqrt(2σ²Ω·bandwidth/(π·a²·N)) < threshold_hz:
        N > 8σ²Ω·bandwidth / (π·a²·threshold_hz²)
    """
    if c_total <= 0 or linewidth <= 0 or noise_std <= 0 or threshold_hz <= 0:
        return 0
    return math.ceil(8.0 * noise_std**2 * linewidth * _CRLB_BANDWIDTH / (math.pi * c_total**2 * threshold_hz**2))


def _apply_crlb_convergence_steps(row: dict[str, Any], noise_std: float, threshold_hz: float) -> dict[str, Any]:
    """Push splitting_converged_step / all_converged_step forward to honour the CRLB check.

    NOTE: this closed-form CRLB formula (``_crlb_min_obs``) is derived specifically
    for frequency/linewidth/contrast physics and is not re-derived for zeeman_split.
    It already no-ops for non-Lorentzian rows (see the linewidth/c_total guard
    below), which happens to also make it a safe no-op for splitting-based
    convergence today -- left as a known follow-up rather than guessed physics.

    The locator originally set these steps when raw uncertainty() first dropped
    below threshold_hz, without a CRLB check.  After the CRLB-aware change the
    effective convergence step is max(old_step, sweep_steps + N_crlb), where
    N_crlb is the minimum number of locator observations needed so that
    2×CRLB_phys < threshold_hz.  If that new step exceeds total measurements
    the run never satisfied both conditions; the field is set to None.
    """
    linewidth = row.get("final_est_linewidth")
    c_total = row.get("final_est_c_total")
    if not all(isinstance(v, int | float) for v in (linewidth, c_total)):
        return row  # non-Lorentzian or missing estimates — leave unchanged

    n_crlb = _crlb_min_obs(noise_std, float(linewidth), float(c_total), threshold_hz)  # type: ignore[arg-type]
    if n_crlb == 0:
        return row

    sweep = int(row.get("sweep_steps") or 0)
    total = int(row.get("measurements") or 0)
    crlb_step = sweep + n_crlb  # locator step at which CRLB condition first satisfied

    updated = dict(row)
    changed = False

    for field in ("splitting_converged_step", "all_converged_step"):
        old = row.get(field)
        if old is None:
            continue  # run never converged empirically — leave as None
        old = int(old)
        new = max(old, crlb_step)
        new_val = None if new > total else new  # None: CRLB requires more measurements than the run had
        if new_val != old:
            updated[field] = new_val
            changed = True

    return updated if changed else row


def _apply_crlb_floor(row: dict[str, Any], noise_std: float) -> dict[str, Any]:
    """Floor the frequency uncertainty at 2× the Lorentzian CRLB.

    Mirrors UnitCubeSMCMarginalDistribution.crlb_frequency() and reported_uncertainty().
    Returns *row* unchanged when any required field is absent, non-positive, or the
    floor has no effect (i.e. the empirical uncertainty already exceeds 2×CRLB).

    Formula: CRLB_f = sqrt(2σ²Ω / (π a² ρ))
        σ   = noise std
        Ω   = linewidth (FWHM, Hz) — from final posterior mean
        a   = c_total (contrast) — from final posterior mean
        ρ   = n_obs / bandwidth

    Note: main_result_row never stores uncert_frequency directly — _promote_uncert
    collapses it into `uncert`.  For NV-center Lorentzian models `uncert` equals
    uncert_frequency because frequency is first in the promotion priority list, so
    we treat `uncert` as the pre-floor frequency uncertainty.
    """
    linewidth = row.get("final_est_linewidth")
    c_total = row.get("final_est_c_total")
    n_obs = row.get("measurements")
    uncert = row.get("uncert")

    if not all(isinstance(v, int | float) for v in (linewidth, c_total, n_obs, uncert)):
        return row

    linewidth = float(linewidth)  # type: ignore[arg-type]
    c_total = float(c_total)  # type: ignore[arg-type]
    n_obs = float(n_obs)  # type: ignore[arg-type]
    uncert = float(uncert)  # type: ignore[arg-type]

    if c_total <= 0 or linewidth <= 0 or noise_std <= 0 or n_obs <= 0 or not math.isfinite(uncert):
        return row

    rho = n_obs / _CRLB_BANDWIDTH
    variance = (2.0 * noise_std**2 * linewidth) / (math.pi * c_total**2 * rho)
    crlb = math.sqrt(max(variance, 0.0))
    floored = max(uncert, 2.0 * crlb)

    if floored == uncert:
        return row  # floor has no effect; avoid spurious cache writes

    updated = dict(row)
    updated["uncert"] = floored
    return updated


def _configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=True,
                log_time_format="%Y-%m-%d %H:%M:%S",
                tracebacks_show_locals=False,
            )
        ],
        force=True,
    )
    logging.getLogger("nvision").setLevel(level)


def _build_metrics_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Extract all known metric fields from a main_result_row dict."""
    metrics: dict[str, Any] = {}
    for key in _METRIC_KEYS:
        if key in row:
            metrics[key] = row[key]
    # Also forward any final_est_* fields
    for key, val in row.items():
        if key.startswith("final_est_"):
            metrics[key] = val
    return metrics


def _apply_metrics_to_entry(entry: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *entry* with metric fields merged in at top level and in entry['metrics']."""
    updated = dict(entry)
    updated.update(metrics)
    existing_metrics = dict(entry.get("metrics") or {})
    existing_metrics.update(metrics)
    updated["metrics"] = existing_metrics
    return updated


def _entries_differ(
    old_entries: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> bool:
    """Return True if any scan entry top-level metric value changed."""
    old_scan = next((e for e in old_entries if e.get("type") == "scan"), None)
    if old_scan is None:
        return False
    return any(old_scan.get(key) != val for key, val in metrics.items())


@metrics_app.command(name="recalc")
def recalc_metrics(
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory (same as used for nvision run/render)"),
    ] = ARTIFACTS_ROOT,
    filter_generator: Annotated[
        str | None,
        typer.Option("--filter-generator", help="Glob filter on generator name (e.g. 'NVCenter*')"),
    ] = None,
    filter_strategy: Annotated[
        str | None,
        typer.Option("--filter-strategy", help="Glob or comma-separated filter on strategy name"),
    ] = None,
    filter_noise: Annotated[
        str | None,
        typer.Option("--filter-noise", help="Prefix filter on noise name"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report what would change without writing anything"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Update entries even when metric values appear unchanged"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging verbosity", case_sensitive=False),
    ] = "INFO",
) -> int:
    """Recalculate UI metrics from cached run data and refresh plots_manifest.json.

    Re-derives abs_err_x / uncert from the stored estimates and ground truth,
    then propagates all remaining metric fields (sweep_steps, sobol_*, milestone
    metrics, convergence steps, …) from the cached main_result_row into the
    manifest scan entries.  Rewrites both the cache entries and
    plots_manifest.json so the UI reflects updated values immediately — no
    need to run ``nvision render`` afterwards.
    """
    _configure_logging(log_level)

    out_dir = out
    cache_dir = out_dir / "cache"

    if not cache_dir.exists():
        console.print(f"[yellow]Cache directory not found: {cache_dir}[/yellow]")
        raise typer.Exit(1)

    from nvision.models.experiment import CoreExperiment
    from nvision.runner.metrics import _scan_attempt_metrics, _truth_positions

    bridge = CacheBridge(cache_dir)
    grid = CombinationGrid()
    prepare_artifact_tree(out_dir)

    # Collect all unique cached combination configs
    stores = [("NVCenter", bridge.nv_center), ("Complementary", bridge.complementary)]

    discovered: list[tuple[str, dict[str, Any]]] = []
    for store_category, store in stores:
        for key in store.backend:
            payload = store.backend.get(key)
            if not isinstance(payload, dict):
                continue
            cfg = payload.get("config")
            if not isinstance(cfg, dict):
                continue
            kind = cfg.get("kind")
            if kind not in ("locator_combination", "locator_combination_pointer"):
                continue
            generator = cfg.get("generator")
            noise = cfg.get("noise")
            strategy = cfg.get("strategy")
            if not (isinstance(generator, str) and isinstance(noise, str) and isinstance(strategy, str)):
                continue

            # Apply filters
            if filter_generator and not fnmatch.fnmatch(generator, filter_generator):
                continue
            if filter_noise and not str(noise).startswith(filter_noise):
                continue
            if filter_strategy:
                patterns = [p.strip() for p in filter_strategy.split(",")]
                if not any(fnmatch.fnmatch(strategy, p) for p in patterns):
                    continue

            # Resolve achieved_repeats for pointer entries
            cfg_copy = dict(cfg)
            if kind == "locator_combination_pointer" and (
                "data" in payload and payload["data"] and isinstance(payload["data"], list)
            ):
                cfg_copy["repeats"] = payload["data"][0].get("achieved_repeats", cfg.get("repeats", 0))

            inferred_category = CombinationGrid.generator_category(generator)
            category = inferred_category if inferred_category != "Unknown" else store_category
            discovered.append((category, cfg_copy))

    if not discovered:
        console.print("[yellow]No cached combinations found matching the given filters.[/yellow]")
        raise typer.Exit(0)

    # Deduplicate: keep the version with the most repeats per (gen, noise, strat, max_steps, seed)
    best: dict[tuple, tuple[str, dict[str, Any]]] = {}
    for category, cfg in discovered:
        gen = str(cfg.get("generator"))
        noise = str(cfg.get("noise"))
        strat = str(cfg.get("strategy"))
        max_steps = int(cfg.get("max_steps", 0))
        seed = int(cfg.get("seed", 0))
        dedup_key = (gen, noise, strat, max_steps, seed)
        curr_repeats = int(cfg.get("repeats", 0))
        if dedup_key not in best or curr_repeats > int(best[dedup_key][1].get("repeats", 0)):
            best[dedup_key] = (category, cfg)

    final_configs = list(best.values())
    log.info("Found %d unique cached scenario(s) to process.", len(final_configs))

    updated_combos = 0
    updated_repeats = 0
    skipped_repeats = 0

    all_manifest_entries: list[dict[str, Any]] = []
    all_df_rows: list[dict[str, Any]] = []

    summary_table = Table(title="Metrics Recalc", box=None, header_style="bold cyan")
    summary_table.add_column("Generator")
    summary_table.add_column("Noise")
    summary_table.add_column("Strategy")
    summary_table.add_column("Repeats", justify="right")
    summary_table.add_column("Updated", justify="right")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task("Processing...", total=None)

        for category, cfg in final_configs:
            gen_name = str(cfg.get("generator"))
            noise_name = str(cfg.get("noise"))
            strat_name = str(cfg.get("strategy"))
            seed = int(cfg.get("seed", NVISION_RNG_SEED))
            max_steps = int(cfg.get("max_steps", 0))
            timeout_s = int(cfg.get("timeout_s", cli_defaults.DEFAULT_LOC_TIMEOUT_S))

            progress.update(task_id, description=f"{gen_name}/{noise_name}/{strat_name}")

            # Resolve pointer key
            ptr_config = combination_base_cache_config(
                generator=gen_name,
                noise=noise_name,
                strategy=strat_name,
                seed=seed,
                max_steps=max_steps,
                timeout_s=timeout_s,
            )
            ptr_key = stable_config_hash(ptr_config)

            # Verify pointer exists (may also be v8 schema)
            cache = bridge.get_cache_for_category(category)
            ptr_df = cache._store.load_df(ptr_key)
            if ptr_df is None or ptr_df.is_empty():
                ptr_config_v8 = dict(ptr_config)
                ptr_config_v8["schema_version"] = 8
                ptr_key_v8 = stable_config_hash(ptr_config_v8)
                ptr_df_v8 = cache._store.load_df(ptr_key_v8)
                if ptr_df_v8 is not None and not ptr_df_v8.is_empty():
                    ptr_key = ptr_key_v8
                    ptr_df = ptr_df_v8

            if ptr_df is None or ptr_df.is_empty():
                log.debug("No pointer found for %s/%s/%s — skipping.", gen_name, noise_name, strat_name)
                continue

            actual_repeats = int(ptr_df.get_column("achieved_repeats")[0])
            if actual_repeats == 0:
                continue

            # Reconstruct experiment for truth positions
            combo = grid.resolve(gen_name, noise_name, strat_name)
            if combo is None:
                log.warning("Could not resolve combo %s/%s/%s — skipping.", gen_name, noise_name, strat_name)
                continue

            rng = random.Random(seed)
            true_signal = combo.generator.generate(rng)
            experiment = CoreExperiment(
                true_signal=true_signal,
                noise=combo.noise,
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
            )
            truth_positions = _truth_positions(experiment)

            combo_updated = 0
            combo_entries: list[dict[str, Any]] = []
            combo_rows: list[dict[str, Any]] = []

            for repeat_idx in range(actual_repeats):
                result = cache._repeats.load_repeat(ptr_key, repeat_idx)
                if result is None:
                    log.debug("Missing repeat %d for %s/%s/%s.", repeat_idx, gen_name, noise_name, strat_name)
                    continue

                old_entries, main_result_row = result

                # Re-derive error/uncertainty metrics from stored estimates + truth
                fresh_metrics = _scan_attempt_metrics(truth_positions, main_result_row)

                # Merge fresh re-derived values into main_result_row
                updated_row = dict(main_result_row)
                updated_row.update(fresh_metrics)

                # Build full metrics dict (re-derived + everything stored in main_result_row)
                full_metrics = _build_metrics_dict(updated_row)

                # Update entries
                new_entries = [
                    _apply_metrics_to_entry(e, full_metrics) if e.get("type") == "scan" else dict(e)
                    for e in old_entries
                ]

                changed = force or _entries_differ(old_entries, new_entries, full_metrics)

                if changed:
                    if not dry_run:
                        cache._repeats.save_repeat(ptr_key, repeat_idx, new_entries, updated_row)
                    combo_updated += 1
                    updated_repeats += 1
                else:
                    skipped_repeats += 1

                # Collect for manifest (always use the freshest version)
                for entry in new_entries:
                    combo_entries.append(strip_heavy_fields(entry))
                combo_rows.append(updated_row)

            all_manifest_entries.extend(combo_entries)
            all_df_rows.extend(combo_rows)

            if combo_updated:
                updated_combos += 1
                summary_table.add_row(gen_name, noise_name, strat_name, str(actual_repeats), str(combo_updated))

    bridge.close()

    if not dry_run:
        # Rebuild manifest and CSV from updated entries
        manifest_path = write_plots_manifest(all_manifest_entries, out_dir)
        log.info("Wrote manifest with %d entries to %s", len(all_manifest_entries), manifest_path)

        df_loc = pl.from_dicts(all_df_rows, infer_schema_length=None) if all_df_rows else pl.DataFrame()
        csv_path = write_locator_results_csv(df_loc, out_dir)
        log.info("Wrote locator results to %s", csv_path)

        if updated_combos:
            console.print(summary_table)
        console.print(
            f"[green]Updated {updated_repeats} repeat(s) across {updated_combos} combo(s).[/green] "
            f"[dim]{skipped_repeats} already up-to-date.[/dim]"
        )
    else:
        if updated_combos:
            console.print(summary_table)
        console.print(
            f"[dim]Dry run: would update {updated_repeats} repeat(s) across {updated_combos} combo(s). "
            f"{skipped_repeats} already up-to-date.[/dim]"
        )

    return 0
