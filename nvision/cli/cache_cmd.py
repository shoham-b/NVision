from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from nvision.cache import CacheBridge
from nvision.cache.data_store import CategoryDataStore
from nvision.cli.app_instance import app
from nvision.sim.combinations import CombinationGrid
from nvision.sim.gen.nv_center_generator import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
)
from nvision.sim.grid_enums import GeneratorName, NoiseName, StrategyFilter
from nvision.tools.utils import NVISION_RNG_SEED

console = Console()
log = logging.getLogger("nvision.cache_cmd")

# Create a Typer app for the cache command group
cache_app = typer.Typer(help="Manage simulation cache.", pretty_exceptions_show_locals=False)
app.add_typer(cache_app, name="cache")

gen_app = typer.Typer(
    help="Manage cache 'generations' -- named snapshots of the whole cache directory.",
    pretty_exceptions_show_locals=False,
)
cache_app.add_typer(gen_app, name="gen")


def _get_caches(root: Path) -> list[tuple[str, CategoryDataStore]]:
    bridge = CacheBridge(root)
    return [("NVCenter", bridge.nv_center), ("Complementary", bridge.complementary)]


@cache_app.command(name="list")
def list_cache(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
) -> None:
    """List cached simulations grouped for readability."""
    cache_root = out / "cache"

    found_any = False
    grouped: dict[tuple[str, str, str, str, str, str, str], set[str]] = defaultdict(set)
    row_counts: dict[tuple[str, str, str, str, str, str, str], int] = defaultdict(int)
    updated_dates: dict[tuple[str, str, str, str, str, str, str], str] = {}
    for cat_name, cat_cache in _get_caches(cache_root):
        backend = cat_cache.backend
        for key in backend:
            payload = backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                config = payload["config"]
                kind = config.get("kind")
                if kind in ("locator_combination", "locator_combination_pointer"):
                    found_any = True
                    # Extract achieved repeats for pointer kind
                    if kind == "locator_combination_pointer":
                        repeats_val = "-"
                        if "data" in payload and isinstance(payload["data"], list) and len(payload["data"]) > 0:
                            repeats_val = str(payload["data"][0].get("achieved_repeats", "-"))
                    else:
                        repeats_val = str(config.get("repeats", "-"))

                    group_key = (
                        cat_name,
                        str(config.get("generator", "-")),
                        str(config.get("strategy", "-")),
                        repeats_val,
                        str(config.get("max_steps", "-")),
                        str(config.get("timeout_s", "-")),
                        str(config.get("schema_version", "-")),
                    )
                    grouped[group_key].add(str(config.get("noise", "-")))
                    row_counts[group_key] += 1

                    # Track the latest updated date
                    updated_at_val = payload.get("updated_at", "-")
                    if group_key not in updated_dates or (
                        updated_at_val != "-"
                        and (updated_dates[group_key] == "-" or updated_at_val > updated_dates[group_key])
                    ):
                        updated_dates[group_key] = updated_at_val

    if found_any:
        grouped_by_category: dict[str, list[tuple[str, str, str, str, str, str]]] = defaultdict(list)
        for cat_name, generator, strategy, repeats, max_steps, timeout_s, schema in sorted(grouped):
            grouped_by_category[cat_name].append((generator, strategy, repeats, max_steps, timeout_s, schema))

        for cat_name in sorted(grouped_by_category):
            table = Table(title=f"{cat_name} Cache (Grouped)")
            table.add_column("Generator", style="green")
            table.add_column("Strategy", style="blue")
            table.add_column("Repeats", justify="right")
            table.add_column("Max", justify="right")
            table.add_column("Timeout", justify="right")
            table.add_column("Schema", justify="right")
            table.add_column("Noises", justify="right")
            table.add_column("Rows", justify="right")
            table.add_column("Updated", justify="right")

            for generator, strategy, repeats, max_steps, timeout_s, schema in grouped_by_category[cat_name]:
                group_key = (cat_name, generator, strategy, repeats, max_steps, timeout_s, schema)
                table.add_row(
                    generator,
                    strategy,
                    repeats,
                    max_steps,
                    timeout_s,
                    schema,
                    str(len(grouped[group_key])),
                    str(row_counts[group_key]),
                    updated_dates.get(group_key, "-"),
                )
            console.print(table)
    else:
        console.print("[yellow]No cached combinations found (or no metadata available).[/yellow]")


def _matches_filter(
    config: dict[str, Any],
    category: str | None,
    strategy: StrategyFilter | None,
    generator: GeneratorName | None,
    noise: NoiseName | None,
    max_steps: int | None = None,
    repeats: int | None = None,
) -> bool:
    """Check if a config matches all the given filters."""
    if strategy and config.get("strategy") != strategy:
        return False
    if generator and config.get("generator") != generator:
        return False
    if max_steps is not None and config.get("max_steps") != max_steps:
        return False
    if repeats is not None and config.get("repeats") != repeats:
        return False
    return not (noise and not str(config.get("noise", "")).startswith(noise))


@cache_app.command(name="progress")
def cache_progress(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    category: Annotated[
        str | None,
        typer.Option("--category", help="Category filter (e.g. 'NVCenter')"),
    ] = None,
    strategy: Annotated[
        StrategyFilter | None,
        typer.Option("--strategy", help="Strategy filter (see StrategyFilter)."),
    ] = None,
    generator: Annotated[
        GeneratorName | None,
        typer.Option("--generator", help="Generator filter (see GeneratorName)."),
    ] = None,
    noise: Annotated[
        NoiseName | None,
        typer.Option("--noise", help="Noise preset filter (see NoiseName)."),
    ] = None,
    target_repeats: Annotated[
        int | None,
        typer.Option("--repeats", help="Expected repeats per combination, to compute percent-complete."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show one row per (generator, noise, strategy) combination."),
    ] = False,
) -> None:
    """Show live repeat-completion progress for cached combinations.

    Reads ``achieved_repeats`` directly off each combination's cache pointer, so it's
    safe to run while ``nv run`` / ``nv groups`` is still writing to the same cache
    (read-only, no lock contention with the writer).
    """
    cache_root = out / "cache"
    rows: list[tuple[str, str, str, str, int, str]] = []

    for cat_name, cat_cache in _get_caches(cache_root):
        if category and category.lower() not in cat_name.lower():
            continue
        backend = cat_cache.backend
        for key in backend:
            payload = backend.get(key)
            if not (isinstance(payload, dict) and "config" in payload):
                continue
            cfg = payload["config"]
            if cfg.get("kind") != "locator_combination_pointer":
                continue
            if not _matches_filter(cfg, None, strategy, generator, noise):
                continue
            achieved = 0
            data = payload.get("data")
            if isinstance(data, list) and data:
                achieved = int(data[0].get("achieved_repeats", 0) or 0)
            rows.append((
                cat_name,
                str(cfg.get("generator", "-")),
                str(cfg.get("noise", "-")),
                str(cfg.get("strategy", "-")),
                achieved,
                str(payload.get("updated_at", "-")),
            ))

    if not rows:
        console.print("[yellow]No matching cached combinations found.[/yellow]")
        return

    rows.sort(key=lambda r: (r[1], r[3], r[2]))
    n_combos = len(rows)
    total_achieved = sum(r[4] for r in rows)

    console.print(f"[bold]{n_combos}[/bold] combination(s) found, [bold]{total_achieved}[/bold] repeats completed so far.")

    if target_repeats:
        target_total = target_repeats * n_combos
        pct = 100.0 * total_achieved / target_total if target_total else 0.0
        done_combos = sum(1 for r in rows if r[4] >= target_repeats)
        console.print(
            f"Target: {target_repeats} repeats/combo -> {target_total} total repeats. "
            f"[bold]{pct:.1f}%[/bold] complete ({done_combos}/{n_combos} combinations fully done)."
        )

    def _print_table(title: str, subset: list[tuple[str, str, str, str, int, str]]) -> None:
        table = Table(title=title)
        table.add_column("Generator", style="green")
        table.add_column("Noise")
        table.add_column("Strategy", style="blue")
        table.add_column("Repeats done", justify="right")
        if target_repeats:
            table.add_column("Target", justify="right")
        table.add_column("Updated", justify="right")
        for _cat, gen, noise_v, strat, achieved, updated_at in subset:
            row = [gen, noise_v, strat, str(achieved)]
            if target_repeats:
                row.append(str(target_repeats))
            row.append(updated_at)
            table.add_row(*row)
        console.print(table)

    if verbose:
        _print_table("Per-combination progress", rows)
    elif target_repeats:
        behind = [r for r in rows if r[4] < target_repeats]
        if behind:
            shown = behind[:30]
            _print_table(f"Combinations still below target — {len(behind)} remaining", shown)
            if len(behind) > 30:
                console.print(f"[dim]... and {len(behind) - 30} more. Use --verbose to see all.[/dim]")
        else:
            console.print("[green]All matching combinations have reached the target repeat count.[/green]")


@cache_app.command(name="convergence")
def cache_convergence(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    csv_path: Annotated[
        Path | None,
        typer.Option("--csv", help="Path to locator_results.csv (default: <out>/locator_results.csv)."),
    ] = None,
    strategy: Annotated[
        StrategyFilter | None,
        typer.Option("--strategy", help="Strategy filter (see StrategyFilter)."),
    ] = None,
    generator: Annotated[
        str | None,
        typer.Option("--generator", help="Generator name substring filter (matches grid-parameterized names too)."),
    ] = None,
    noise: Annotated[
        NoiseName | None,
        typer.Option("--noise", help="Noise preset filter (see NoiseName)."),
    ] = None,
    bad_ratio: Annotated[
        float,
        typer.Option(
            "--bad-ratio",
            help=(
                "abs_err_x / uncert threshold at/above which a repeat is flagged as miscalibrated: "
                "the locator's own reported final uncertainty badly underestimates its true error, "
                "the signature of premature convergence onto the wrong posterior mode rather than "
                "a locator that is merely slow or noisy."
            ),
        ),
    ] = 10.0,
    low_measurements: Annotated[
        int,
        typer.Option(
            "--low-measurements",
            help="Measurement count at/below which a repeat is flagged as a suspiciously fast stop.",
        ),
    ] = 5,
    breakdown: Annotated[
        str,
        typer.Option(
            "--breakdown",
            help=(
                "Comma-separated locator_results.csv columns to group rows by. Default 'strategy' "
                "compares across every locator in the run. Use e.g. 'strategy,noise' or "
                "'grid_linewidth,grid_c_total' to check whether a problem is isolated to part of the "
                "grid or systemic across it."
            ),
        ),
    ] = "strategy",
    worst: Annotated[
        int,
        typer.Option("--worst", help="Also list the N individual repeats with the highest error/uncertainty ratio."),
    ] = 0,
) -> None:
    """Check whether completed repeats are actually converging well, not just finishing.

    Reads the aggregate metrics CSV that ``nv render`` (or the results UI's "Recalculate
    results" button) writes to ``<out>/locator_results.csv``. For every repeat with a
    reported uncertainty, computes ``abs_err_x / uncert`` -- how many multiples of its own
    claimed uncertainty the locator's final estimate actually missed by. A locator that is
    well-calibrated keeps this ratio near or below 1 most of the time; a ratio that is
    consistently >>1 across many repeats means the locator is confidently wrong, not just
    imprecise -- typically a sign of premature stopping or an SMC/particle posterior that
    collapsed onto a plausible-looking but incorrect mode (e.g. the wrong Zeeman dip).

    This is a purely quantitative, read-only check over already-computed metrics: it does
    not touch the cache, does not re-run anything, and is safe to run while ``nv run``/
    ``nv groups`` is still writing to the cache. Run ``nv render`` first (also safe against
    a live cache) if you want the CSV to reflect the most recent repeats.
    """
    resolved_csv = csv_path or (out / "locator_results.csv")
    if not resolved_csv.exists():
        console.print(f"[yellow]{resolved_csv} not found. Run `nv render` first to generate it.[/yellow]")
        raise typer.Exit(1)

    import polars as pl

    df = pl.read_csv(resolved_csv, infer_schema_length=None)

    if strategy:
        df = df.filter(pl.col("strategy") == str(strategy))
    if generator:
        df = df.filter(pl.col("generator").str.contains(generator, literal=True))
    if noise:
        df = df.filter(pl.col("noise").str.starts_with(str(noise)))

    if df.is_empty():
        console.print("[yellow]No matching rows in locator_results.csv.[/yellow]")
        return

    group_cols = [c.strip() for c in breakdown.split(",") if c.strip()]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        console.print(f"[red]Unknown breakdown column(s): {missing}. Available columns: {sorted(df.columns)}[/red]")
        raise typer.Exit(1)

    required = {
        "abs_err_x",
        "uncert",
        "measurements",
        "stop_reason",
        "failure_reason",
        "generator",
        "noise",
        "attempt",
        *group_cols,
    }
    missing_required = required - set(df.columns)
    if missing_required:
        console.print(f"[red]locator_results.csv is missing expected column(s): {sorted(missing_required)}[/red]")
        raise typer.Exit(1)

    rows = df.select(*sorted(required)).to_dicts()

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(c)) for c in group_cols)
        groups[key].append(row)

    table = Table(title=f"Convergence check (bad-ratio >= {bad_ratio}, low-measurements <= {low_measurements})")
    for c in group_cols:
        table.add_column(c, style="blue")
    table.add_column("n", justify="right")
    table.add_column("n w/ uncert", justify="right")
    table.add_column("median ratio", justify="right")
    table.add_column("% ratio>=bad", justify="right", style="red")
    table.add_column("% low meas.", justify="right", style="yellow")
    table.add_column("top failure_reason", justify="left")

    worst_rows: list[tuple[float, tuple[str, ...], dict[str, Any]]] = []

    for key in sorted(groups):
        group_rows = groups[key]
        n_total = len(group_rows)
        ratios = []
        for r in group_rows:
            err = r.get("abs_err_x")
            unc = r.get("uncert")
            if err is None or unc is None or unc <= 0:
                continue
            ratio = err / unc
            ratios.append(ratio)
            if worst:
                worst_rows.append((ratio, key, r))

        n_meas = [r["measurements"] for r in group_rows if r.get("measurements") is not None]
        low_meas_pct = 100.0 * sum(1 for m in n_meas if m <= low_measurements) / len(n_meas) if n_meas else 0.0

        failure_counts: dict[str, int] = defaultdict(int)
        for r in group_rows:
            fr = r.get("failure_reason")
            if fr:
                failure_counts[str(fr)] += 1
        top_failure = max(failure_counts.items(), key=lambda kv: kv[1])[0] if failure_counts else "-"
        if failure_counts:
            top_failure += f" ({100.0 * failure_counts[top_failure] / n_total:.0f}%)"

        if ratios:
            ratios.sort()
            median_ratio = ratios[len(ratios) // 2]
            bad_pct = 100.0 * sum(1 for x in ratios if x >= bad_ratio) / len(ratios)
            median_str = f"{median_ratio:.2f}"
            bad_str = f"{bad_pct:.1f}%"
        else:
            median_str = "-"
            bad_str = "-"

        table.add_row(
            *key,
            str(n_total),
            str(len(ratios)),
            median_str,
            bad_str,
            f"{low_meas_pct:.1f}%",
            top_failure,
        )

    console.print(table)

    if worst:
        worst_rows.sort(key=lambda t: t[0], reverse=True)
        wtable = Table(title=f"{worst} worst-calibrated individual repeats")
        for c in group_cols:
            wtable.add_column(c, style="blue")
        wtable.add_column("generator")
        wtable.add_column("noise")
        wtable.add_column("attempt", justify="right")
        wtable.add_column("measurements", justify="right")
        wtable.add_column("abs_err_x", justify="right")
        wtable.add_column("uncert", justify="right")
        wtable.add_column("ratio", justify="right", style="red")
        for ratio, key, r in worst_rows[:worst]:
            wtable.add_row(
                *key,
                str(r.get("generator", "-")),
                str(r.get("noise", "-")),
                str(r.get("attempt", "-")),
                str(r.get("measurements", "-")),
                f"{r.get('abs_err_x', 0):.1f}",
                f"{r.get('uncert', 0):.1f}",
                f"{ratio:.1f}",
            )
        console.print(wtable)


def _iter_typed_array_payloads(obj: Any) -> Any:
    """Yield every base64 typed-array payload nested in a decoded Plotly figure dict.

    Mirrors the two encodings ``static/plotly-utils.js`` decodes client-side: this
    project's own ``{"__f32__": b64}`` format, and Plotly Python 5.x's numpy
    serialization ``{"bdata": b64, "dtype": ...}``.
    """
    if isinstance(obj, dict):
        if "__f32__" in obj and isinstance(obj["__f32__"], str):
            yield obj["__f32__"]
        elif "bdata" in obj and "dtype" in obj and isinstance(obj["bdata"], str):
            yield obj["bdata"]
        else:
            for v in obj.values():
                yield from _iter_typed_array_payloads(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_typed_array_payloads(v)


@cache_app.command(name="check-plots")
def check_plots(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    plot_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="plots_manifest.json entry 'type' to check (default 'scan', the per-repeat measurement plot). "
            "Pass 'all' to check every type.",
        ),
    ] = "scan",
    strategy: Annotated[
        str | None,
        typer.Option("--strategy", help="Strategy filter (exact match against the manifest entry, where present)."),
    ] = None,
    generator: Annotated[
        str | None,
        typer.Option("--generator", help="Generator name substring filter."),
    ] = None,
    noise: Annotated[
        str | None,
        typer.Option("--noise", help="Noise name substring filter."),
    ] = None,
    sample: Annotated[
        int | None,
        typer.Option("--sample", help="Randomly sample N matching entries instead of checking every match."),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", help="Sampling seed, for reproducible spot-checks.")
    ] = NVISION_RNG_SEED,
    show_ok: Annotated[bool, typer.Option("--show-ok", help="Also list entries that passed.")] = False,
    restore: Annotated[
        bool,
        typer.Option(
            "--restore/--no-restore",
            help=(
                "Materialize cache-only graph files to disk before checking -- the same one-time "
                "walk `nv serve` does at startup. Graph bytes live in the cache DB during a run and "
                "are only written to <out>/graphs/... on demand; without this, repeats completed "
                "after the last restore pass (e.g. after `nv serve` has been up for a while, since "
                "its walk only ever runs once per process) report as 'file not found' even though "
                "the data is fine. Can take a while against a large cache, so it defaults off."
            ),
        ),
    ] = False,
) -> None:
    """Verify every referenced plot file on disk actually decodes -- no browser needed.

    For each matching entry in ``plots_manifest.json``, opens its ``path`` (a gzip'd JSON
    Plotly figure under ``<out>/graphs/...``) and replays the same decode
    ``static/plotly-utils.js`` does in the browser: gzip -> ``json.loads`` (after the same
    Infinity/NaN -> null substitution the frontend applies) -> base64-decode every
    ``{bdata,dtype}``/``{__f32__}`` typed-array payload. A payload that fails to
    base64-decode here is the exact server-side reproduction of the UI's
    "Failed to load plot: ... atob ... not correctly encoded" error -- so this catches
    that failure mode across the whole run in seconds, instead of clicking through
    combinations in a browser one at a time.

    A missing file is reported separately from a decode failure: it just as often means the
    graph was never materialized to disk (see ``--restore``) as it does a real problem, so
    don't treat "file not found" and "base64 decode failed" as the same severity of finding.

    Otherwise read-only and safe to run against a cache a live ``nv run``/``nv groups`` job is
    still writing to; it only touches files already flushed to disk (plus whatever ``--restore``
    writes, which mirrors what ``nv serve`` itself would eventually write). Run ``nv render``
    first if you want the manifest to reflect the most recent repeats.
    """
    import base64
    import binascii
    import gzip
    import re

    manifest_path = out / "plots_manifest.json"
    if not manifest_path.exists():
        console.print(f"[yellow]{manifest_path} not found. Run `nv render` first to generate it.[/yellow]")
        raise typer.Exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if restore:
        cache_dir = out / "cache"
        if cache_dir.is_dir():
            from nvision.runner.cache import restore_graphs

            bridge = CacheBridge(cache_dir)
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
            ) as progress:
                combos = bridge.list_combinations()
                task_id = progress.add_task(
                    f"Restoring graphs for {len(combos)} cached combination(s)...", total=len(combos)
                )
                for combo in combos:
                    progress.advance(task_id)
                    try:
                        results = bridge.get_cached_combination(**combo)
                        if results:
                            restore_graphs(results, out)
                    except Exception as e:
                        log.debug("Failed to restore graphs for combo %s: %s", combo, e)
            bridge.close()

    entries = manifest if plot_type == "all" else [e for e in manifest if e.get("type") == plot_type]
    if strategy:
        entries = [e for e in entries if e.get("strategy") == strategy]
    if generator:
        entries = [e for e in entries if generator in str(e.get("generator", ""))]
    if noise:
        entries = [e for e in entries if noise in str(e.get("noise", ""))]
    entries = [e for e in entries if e.get("path")]

    if not entries:
        console.print("[yellow]No matching manifest entries with a plot path.[/yellow]")
        return

    if sample and sample < len(entries):
        entries = random.Random(seed).sample(entries, sample)

    # Must mirror static/plotly-utils.js's decoder exactly, including the bug being guarded
    # against here: a \b-bounded (no structural-char requirement) version of this regex also
    # matches "NaN"/"Infinity" occurring as a substring inside a quoted base64 typed-array
    # payload (whenever it sits next to a non-word base64 char like +, /, =), corrupting the
    # payload. Requiring adjacency to a JSON structural char on both sides avoids that; our own
    # writer (_f32_json.py) always emits compact JSON with no whitespace, so this is exact.
    infinity_re = re.compile(r"(?<=[:,\[])(-Infinity|Infinity|NaN)(?=[,\]}])")

    missing: list[tuple[dict[str, Any], str]] = []
    corrupt: list[tuple[dict[str, Any], str]] = []
    ok_entries: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        task_id = progress.add_task(f"Checking {len(entries)} plot file(s)...", total=len(entries))
        for entry in entries:
            progress.advance(task_id)
            rel_path = entry["path"]
            file_path = out / rel_path
            if not file_path.exists():
                missing.append((entry, f"file not found: {file_path}"))
                continue
            try:
                raw = file_path.read_bytes()
                text = gzip.decompress(raw).decode("utf-8") if rel_path.endswith(".gz") else raw.decode("utf-8")
                text = infinity_re.sub("null", text)
                parsed = json.loads(text)
            except Exception as e:
                corrupt.append((entry, f"{type(e).__name__}: {e}"))
                continue

            bad_payload = False
            for payload in _iter_typed_array_payloads(parsed):
                try:
                    base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError) as e:
                    corrupt.append((entry, f"base64 decode failed ({e}) on a typed-array payload"))
                    bad_payload = True
                    break
            if bad_payload:
                continue
            ok_entries.append(entry)

    console.print(
        f"Checked [bold]{len(entries)}[/bold] plot file(s): [green]{len(ok_entries)} OK[/green], "
        f"[yellow]{len(missing)} missing[/yellow] (rerun with --restore if unexpected), "
        f"[red]{len(corrupt)} corrupt[/red] (real bug -- data on disk that fails to decode)."
    )

    def _print_failure_table(title: str, rows: list[tuple[dict[str, Any], str]]) -> None:
        table = Table(title=title)
        table.add_column("generator")
        table.add_column("noise")
        table.add_column("strategy")
        table.add_column("repeat", justify="right")
        table.add_column("path")
        table.add_column("error")
        for entry, err in rows[:50]:
            table.add_row(
                str(entry.get("generator", "-")),
                str(entry.get("noise", "-")),
                str(entry.get("strategy", "-")),
                str(entry.get("repeat", "-")),
                str(entry.get("path", "-")),
                err,
            )
        console.print(table)
        if len(rows) > 50:
            console.print(f"[dim]... and {len(rows) - 50} more.[/dim]")

    if corrupt:
        _print_failure_table(f"{len(corrupt)} corrupt plot file(s) -- real bug", corrupt)
    if missing:
        _print_failure_table(f"{len(missing)} missing plot file(s) -- likely just un-materialized", missing)

    if show_ok and ok_entries:
        table = Table(title=f"{len(ok_entries)} passing plot file(s)")
        table.add_column("generator")
        table.add_column("noise")
        table.add_column("strategy")
        table.add_column("repeat", justify="right")
        table.add_column("path")
        for entry in ok_entries[:50]:
            table.add_row(
                str(entry.get("generator", "-")),
                str(entry.get("noise", "-")),
                str(entry.get("strategy", "-")),
                str(entry.get("repeat", "-")),
                str(entry.get("path", "-")),
            )
        console.print(table)
        if len(ok_entries) > 50:
            console.print(f"[dim]... and {len(ok_entries) - 50} more.[/dim]")


@gen_app.command(name="save")
def gen_save(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    label: Annotated[
        str | None,
        typer.Option("--label", help="Label for this generation (default: current git commit short hash)."),
    ] = None,
) -> None:
    """Snapshot the current cache into a new generation.

    Copies ``<out>/cache`` into ``<out>/cache_generations/<timestamp>_<label>/`` -- never
    moves or clears the live cache. Run this before starting a run that will change
    numerics (not just performance), so the old batch of results stays reachable
    instead of being silently overwritten in place.
    """
    from nvision.cache.generations import save_generation

    dest = save_generation(out, label=label)
    console.print(f"[green]Saved generation[/green] {dest.name} -> {dest}")


@gen_app.command(name="list")
def gen_list(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
) -> None:
    """List saved cache generations, most recent first."""
    from nvision.cache.generations import list_generations

    infos = list_generations(out)
    if not infos:
        console.print(f"[yellow]No generations found under {out / 'cache_generations'}.[/yellow]")
        return

    table = Table(title="Cache generations")
    table.add_column("Name", style="blue")
    table.add_column("Label")
    table.add_column("Commit")
    table.add_column("Created")
    table.add_column("Size", justify="right")
    for info in infos:
        table.add_row(info.name, info.label, info.commit or "-", info.created_at, f"{info.size_bytes / 1e6:.1f} MB")
    console.print(table)


@gen_app.command(name="restore")
def gen_restore(
    name: Annotated[str, typer.Argument(help="Generation name or label to restore.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    archive_current: Annotated[
        bool,
        typer.Option(
            "--archive-current/--no-archive-current",
            help="Save the current live cache as its own generation before overwriting it.",
        ),
    ] = True,
    force: Annotated[bool, typer.Option("--force", help="Skip confirmation")] = False,
) -> None:
    """Replace the live cache with a previously saved generation."""
    from nvision.cache.generations import restore_generation

    if not force and not Confirm.ask(f"Replace the live cache at {out / 'cache'} with generation {name!r}?"):
        return
    restore_generation(out, name, archive_current=archive_current)
    console.print(f"[green]Restored generation[/green] {name} into {out / 'cache'}")


@gen_app.command(name="prune")
def gen_prune(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    keep: Annotated[int, typer.Option("--keep", help="Number of most recent generations to keep.")] = 2,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be removed without deleting")] = False,
) -> None:
    """Delete all but the N most recent generations."""
    from nvision.cache.generations import prune_generations

    removed = prune_generations(out, keep=keep, dry_run=dry_run)
    if not removed:
        console.print(f"[green]Nothing to prune (kept {keep} most recent).[/green]")
        return
    verb = "Would remove" if dry_run else "Removed"
    console.print(f"[yellow]{verb}[/yellow] {len(removed)} generation(s): {', '.join(removed)}")


@cache_app.command(name="clean")
def cache_clean(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    category: Annotated[
        str | None,
        typer.Option("--category", help="Category filter (e.g. 'NVCenter')"),
    ] = None,
    strategy: Annotated[
        StrategyFilter | None,
        typer.Option("--strategy", help="Strategy filter (see StrategyFilter)."),
    ] = None,
    generator: Annotated[
        GeneratorName | None,
        typer.Option("--generator", help="Generator filter (see GeneratorName)."),
    ] = None,
    noise: Annotated[
        NoiseName | None,
        typer.Option("--noise", help="Noise preset filter (see NoiseName)."),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", help="Max steps filter"),
    ] = None,
    repeats: Annotated[
        int | None,
        typer.Option("--repeats", help="Repeats filter"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show matches without deleting")] = False,
    force: Annotated[bool, typer.Option("--force", help="Skip confirmation")] = False,
) -> None:
    """Delete cached simulation artifacts matching optional filters."""
    cache_root = out / "cache"

    keys_to_delete: list[tuple[str, Any, str]] = []  # (CategoryName, CacheInstance, Key)

    for cat_name, cat_cache in _get_caches(cache_root):
        if category and category.lower() not in cat_name.lower():
            continue

        backend = cat_cache.backend
        for key in backend:
            payload = backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                cfg = payload["config"]
                kind = cfg.get("kind")
                if kind in ("locator_combination", "locator_combination_pointer"):
                    cfg_copy = dict(cfg)
                    if kind == "locator_combination_pointer" and "data" in payload and len(payload["data"]) > 0:
                        cfg_copy["repeats"] = payload["data"][0].get("achieved_repeats")
                    if _matches_filter(cfg_copy, None, strategy, generator, noise, max_steps, repeats):
                        keys_to_delete.append((cat_name, cat_cache, key))

    if not keys_to_delete:
        console.print("[yellow]No matching cache entries found.[/yellow]")
        return

    console.print(f"Found {len(keys_to_delete)} entries to delete.")

    if not dry_run and not force and not Confirm.ask("Are you sure you want to delete these?"):
        return

    if dry_run:
        console.print("[dim]Dry run: no files deleted.[/dim]")
    else:
        deleted_count = 0
        from nvision.cache.locator_repository import LocatorResultsRepository

        for _, cat_cache, key in keys_to_delete:
            payload = cat_cache.backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                cfg = payload["config"]
                # Resolve repeats if pointer config
                resolved_repeats = cfg.get("repeats")
                if cfg.get("kind") == "locator_combination_pointer" and "data" in payload and len(payload["data"]) > 0:
                    resolved_repeats = payload["data"][0].get("achieved_repeats")
                if resolved_repeats is None:
                    resolved_repeats = 0
                repo = LocatorResultsRepository(cat_cache)
                repo.purge_cached_combination(
                    generator=cfg.get("generator"),
                    noise=cfg.get("noise"),
                    strategy=cfg.get("strategy"),
                    repeats=resolved_repeats,
                    seed=cfg.get("seed", NVISION_RNG_SEED),
                    max_steps=cfg.get("max_steps"),
                    timeout_s=cfg.get("timeout_s"),
                )

                # Also clean up artifact graphs and plots_manifest.json entries
                import logging

                from nvision.tools.artifacts import purge_artifacts_for_combination

                purge_artifacts_for_combination(
                    out_dir=out,
                    generator=cfg.get("generator"),
                    noise=cfg.get("noise"),
                    strategy=cfg.get("strategy"),
                    max_steps=cfg.get("max_steps"),
                    seed=cfg.get("seed", NVISION_RNG_SEED),
                    log=logging.getLogger("nvision.cache.clean"),
                )
            else:
                cat_cache.backend.delete(key)
            deleted_count += 1

        console.print(f"[green]Deleted {deleted_count} entries.[/green]")


@cache_app.command(name="clean-manifest")
def clean_manifest(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show matches without deleting")] = False,
    stale_physics: Annotated[
        bool,
        typer.Option(
            "--stale-physics",
            help=(
                "Also remove scan entries whose true_params.config_fingerprint doesn't match "
                "the current physics config (nvision.spectra.nv_center.PHYSICS_CONFIG_FINGERPRINT). "
                "Catches artifacts generated under old domain/Zeeman/hyperfine/linewidth bounds that "
                "a later code or .env change silently left behind -- e.g. a generator whose run group "
                "was never re-run since, so it keeps serving pre-change data with nothing flagging it."
            ),
        ),
    ] = False,
) -> None:
    """Remove old/invalid entries from plots_manifest.json and cache.

    Removes entries for generators that no longer exist (TwoPeak-*) and
    entries with outdated generator_type categorization. Also cleans
    corresponding cache entries.
    """
    import json

    from nvision.spectra.nv_center import PHYSICS_CONFIG_FINGERPRINT

    manifest_path = out / "plots_manifest.json"
    if not manifest_path.exists():
        console.print("[yellow]No plots_manifest.json found.[/yellow]")
        return

    with open(manifest_path) as f:
        plots = json.load(f)

    original_count = len(plots)
    valid_generators = {g.value for g in GeneratorName}
    invalid_generators = set()
    stale_physics_generators = set()

    # Identify invalid entries and collect invalid generator names. Dynamically-named grid
    # generators (e.g. "NVCenter-voigt-w0.50MHz-c0.10-si0.00MHz") are absent from the static
    # GeneratorName enum by construction, so they're matched by prefix against the enum's base
    # names rather than exact membership -- otherwise every grid-sweep entry would be flagged
    # invalid regardless of the (separate) stale-physics check below.
    valid_plots = []
    for p in plots:
        gen = p.get("generator", "")
        is_known_generator = any(gen == base or gen.startswith(base + "-") for base in valid_generators)
        is_invalid_generator = not is_known_generator or p.get("generator_type") == "Supplemental"
        is_stale_physics = False
        if stale_physics and p.get("type") == "scan":
            fp = (p.get("true_params") or {}).get("config_fingerprint")
            is_stale_physics = fp != PHYSICS_CONFIG_FINGERPRINT

        if is_invalid_generator:
            invalid_generators.add(gen)
        if is_stale_physics:
            stale_physics_generators.add(gen)
        if not is_invalid_generator and not is_stale_physics:
            valid_plots.append(p)

    removed = original_count - len(valid_plots)

    if removed == 0:
        console.print("[green]No invalid entries found.[/green]")
        return

    if stale_physics_generators:
        console.print(
            f"[yellow]Stale-physics generators (regenerate with the owning "
            f"``nvision groups run <name> --no-cache``): {sorted(stale_physics_generators)}[/yellow]"
        )
    invalid_generators |= stale_physics_generators

    # Also clean cache entries for invalid generators
    cache_removed = 0
    if invalid_generators:
        cache_root = out / "cache"
        for _cat_name, cat_cache in _get_caches(cache_root):
            backend = cat_cache.backend
            keys_to_delete = []
            for key in backend:
                payload = backend.get(key)
                if isinstance(payload, dict) and "config" in payload:
                    cfg = payload["config"]
                    if cfg.get("generator") in invalid_generators:
                        keys_to_delete.append(key)

            if dry_run:
                cache_removed += len(keys_to_delete)
            else:
                for key in keys_to_delete:
                    backend.delete(key)
                    cache_removed += 1

    if dry_run:
        console.print(f"[dim]Would remove {removed} manifest entries and {cache_removed} cache entries.[/dim]")
    else:
        from nvision.tools.artifacts import _atomic_write_bytes

        _atomic_write_bytes(manifest_path, json.dumps(valid_plots, indent=2).encode("utf-8"))
        console.print(f"[green]Removed {removed} manifest entries and {cache_removed} cache entries.[/green]")


@cache_app.command(name="recalc")
def recalculate_metrics(  # noqa: C901
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    category: Annotated[str | None, typer.Option("--category", help="Category filter")] = None,
    strategy: Annotated[StrategyFilter | None, typer.Option("--strategy", help="Strategy filter")] = None,
    generator: Annotated[GeneratorName | None, typer.Option("--generator", help="Generator filter")] = None,
    noise: Annotated[NoiseName | None, typer.Option("--noise", help="Noise filter")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Max steps filter")] = None,
    repeats: Annotated[int | None, typer.Option("--repeats", help="Repeats filter")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show matches without updating")] = False,
    force: Annotated[bool, typer.Option("--force", help="Update even if metrics already exist")] = False,
) -> None:
    """Recalculate metrics for cached simulation runs."""

    from nvision.models.experiment import CoreExperiment

    cache_root = out / "cache"
    grid = CombinationGrid()
    updated_count = 0
    total_repeats = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Recalculating metrics...", total=None)

        for cat_name, cat_cache in _get_caches(cache_root):
            if category and category.lower() not in cat_name.lower():
                continue

            backend = cat_cache.backend
            for key in backend:
                payload = backend.get(key)
                if not isinstance(payload, dict) or "config" not in payload:
                    continue

                cfg = payload["config"]
                cfg_copy = dict(cfg)
                if cfg.get("kind") == "locator_combination_pointer" and "data" in payload and len(payload["data"]) > 0:
                    cfg_copy["repeats"] = payload["data"][0].get("achieved_repeats")
                if not _matches_filter(cfg_copy, None, strategy, generator, noise, max_steps, repeats):
                    continue

                gen_name = cfg.get("generator")
                noise_name = cfg.get("noise")
                strat_name = cfg.get("strategy")
                seed = cfg.get("seed", NVISION_RNG_SEED)

                combo = grid.resolve(gen_name, noise_name, strat_name)
                if not combo:
                    continue

                # Reconstruct experiment
                rng = random.Random(seed)
                true_signal = combo.generator.generate(rng)
                x_min, x_max = DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX  # Matches _TaskRunner
                experiment = CoreExperiment(true_signal=true_signal, noise=combo.noise, x_min=x_min, x_max=x_max)

                # Load results
                if payload.get("__nvision_cache__") != "dataframe":
                    continue
                data = payload.get("data", [])
                if not data:
                    continue

                results_json = data[0].get("results")
                if not results_json:
                    continue

                try:
                    results = json.loads(results_json)
                except Exception:
                    continue

                if not results:
                    continue

                progress.update(task_id, description=f"Processing {gen_name}/{noise_name}/{strat_name}")

                new_results = []
                combo_updated = False

                for _rid, record in enumerate(results):
                    # Results are stored as list of dicts: [{"entries": ..., "main_result_row": ...}, ...]
                    if not isinstance(record, dict):
                        continue

                    entries = record.get("entries", [])
                    main_result_row = record.get("main_result_row", {})

                    # We'll use the main_result_row as the 'estimate' source.
                    # It contains the flattened results from run_result_to_finalize_record.
                    from nvision.runner.metrics import _scan_attempt_metrics, _truth_positions

                    truth_positions = _truth_positions(experiment)
                    new_metrics = _scan_attempt_metrics(truth_positions, main_result_row)

                    if not force and all(main_result_row.get(k) == v for k, v in new_metrics.items()):
                        new_results.append(record)
                        continue

                    # Update main_result_row
                    updated_row = dict(main_result_row)
                    updated_row.update(new_metrics)

                    # Update entries
                    new_entries = []
                    for entry in entries:
                        if "metrics" in entry:
                            # entry["metrics"] might be a dict or a list of dicts?
                            # Usually it's a dict.
                            entry["metrics"].update(new_metrics)
                        # Also update top-level metrics in entry
                        entry.update(new_metrics)
                        new_entries.append(entry)

                    new_results.append({"entries": new_entries, "main_result_row": updated_row})
                    combo_updated = True
                    total_repeats += 1

                if combo_updated and not dry_run:
                    data[0]["results"] = json.dumps(new_results)
                    backend.set(key, payload)
                    updated_count += 1
                elif combo_updated:
                    updated_count += 1

    if dry_run:
        console.print(f"[dim]Dry run: would update {updated_count} combinations ({total_repeats} repeats).[/dim]")
    else:
        console.print(f"[green]Updated {updated_count} combinations ({total_repeats} repeats).[/green]")
