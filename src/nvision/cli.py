from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from nvision.cache import DataFrameCache
from nvision.index_html import compile_html_index
from nvision.pathutils import ensure_out_dir, slugify
from nvision.sim import (
    CompositeNoise,
    GoldenSectionSearchLocator,
    GridScanLocator,
    TwoPeakGreedyLocator,
)
from nvision.sim import cases as sim_cases
from nvision.sim.loc_runner import LocatorRunner
from nvision.viz import Viz


def _noise_presets() -> list[tuple[str, CompositeNoise | None]]:
    """Return the predefined noise combinations for scenarios."""
    return sim_cases.noises_none() + sim_cases.noises_single_each() + sim_cases.noises_complex()


def _locator_strategies() -> list[tuple[str, object]]:
    """Construct the set of locator strategies used in sweeps."""
    return [
        ("Grid21", GridScanLocator(n_points=21)),
        ("Golden20", GoldenSectionSearchLocator(max_evals=20)),
        ("TwoGreedy", TwoPeakGreedyLocator(coarse_points=21, refine_points=5)),
    ]


def _maybe_finite(value: object) -> float | None:
    if isinstance(value, int | float):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


def run_locator_workflow(
    out_dir: Path,
    repeats: int,
    rng_seed: int | None,
    max_steps: int,
    use_cache: bool = True,
) -> pl.DataFrame:
    """Execute locator sweeps and persist the resulting CSV and plots."""
    runner = LocatorRunner(rng_seed=rng_seed)

    generators: list[tuple[str, object]] = sim_cases.generators_basic()
    strategies: list[tuple[str, object]] = _locator_strategies()
    noises = _noise_presets()

    cache_dir = out_dir / "cache"
    cfg = {
        "kind": "locator",
        "generators": [name for name, _ in generators],
        "noises": [name for name, _ in noises],
        "strategies": [name for name, _ in strategies],
        "repeats": int(repeats),
        "seed": int(rng_seed) if rng_seed is not None else None,
        "max_steps": int(max_steps),
    }
    cache = DataFrameCache(cache_dir)
    key = DataFrameCache.make_key(cfg)
    cached = cache.load_df(key) if use_cache else None
    required_object_columns = {"scan", "noise_obj"}

    if cached is not None and required_object_columns.issubset({*cached.columns}):
        df = cached
    else:
        df = runner.sweep(
            generators,
            strategies,
            noises,
            repeats=repeats,
            max_steps=max_steps,
        )
        if use_cache:
            cache.save_df(df, key)

    object_cols = [col for col, dtype in df.schema.items() if dtype == pl.Object]
    df_serializable = df.drop(object_cols)

    out_path = out_dir / "locator_results.csv"
    df_serializable.write_csv(out_path.as_posix())
    return df


def cli(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    repeats: Annotated[int, typer.Option("--repeats", help="Number of repeats per scenario")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed (int)")] = 123,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps for locator measurement loop"),
    ] = 150,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable caching for this run"),
    ] = False,
) -> int:
    """Typer-driven command-line interface entry point."""
    out_dir: Path = out
    ensure_out_dir(out_dir)

    cache_dir = out_dir / "cache"
    if no_cache and cache_dir.exists():
        print("Clearing cache.")
        shutil.rmtree(cache_dir)

    graphs_dir = out_dir / "graphs"
    ensure_out_dir(graphs_dir)
    scans_dir = graphs_dir / "scans"
    ensure_out_dir(scans_dir)

    viz = Viz(graphs_dir)
    plot_manifest: list[dict[str, object]] = []

    df_loc = run_locator_workflow(
        out_dir,
        repeats=repeats,
        rng_seed=seed,
        max_steps=loc_max_steps,
        use_cache=not no_cache,
    )

    generator_map = dict(sim_cases.generators_basic())
    noise_map = dict(_noise_presets())
    strategy_map = dict(_locator_strategies())

    for idx, row in enumerate(df_loc.iter_rows(named=True)):
        gen_name = row["generator"]
        noise_name = row["noise"]
        strategy_name = row["strategy"]

        generator = generator_map[gen_name]
        noise_obj = noise_map[noise_name]
        strategy = strategy_map[strategy_name]

        combo_seed = (seed or 0) + idx
        scan_rng = random.Random(combo_seed)
        scan = generator.generate(scan_rng)

        plot_runner = LocatorRunner(rng_seed=combo_seed)
        history_df, _ = plot_runner.run_once(scan, strategy, noise_obj, loc_max_steps)

        slug = "_".join(
            slugify(part)
            for part in (
                gen_name,
                noise_name,
                strategy_name,
            )
        )
        out_path = scans_dir / f"{slug}.html"
        viz.plot_scan_measurements(scan, history_df, out_path, noise=noise_obj)
        plot_manifest.append(
            {
                "type": "scan",
                "generator": gen_name,
                "noise": noise_name,
                "strategy": strategy_name,
                "repeat": int(row["repeats"]),
                "abs_err_x": _maybe_finite(row.get("abs_err_x")),
                "uncert": _maybe_finite(row.get("uncert")),
                "path": out_path.relative_to(out_dir).as_posix(),
            }
        )

    try:
        summary_plots_meta = viz.plot_locator_summary(df_loc)
        for meta in summary_plots_meta:
            meta["path"] = Path(meta["path"]).relative_to(out_dir).as_posix()
        plot_manifest.extend(summary_plots_meta)
        print(f"Saved {len(summary_plots_meta)} summary plots")
        print(f"Saved scan plots to: {scans_dir}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[warn] plotting failed: {exc}")

    manifest_path = out_dir / "plots_manifest.json"
    manifest_path.write_text(json.dumps(plot_manifest, indent=2), encoding="utf-8")

    try:
        idx = compile_html_index(out_dir)
        print(f"Generated HTML index at: {idx.absolute().as_uri()}")
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[warn] failed to build HTML index: {exc}")

    print(f"Wrote locator results to: {out_dir}")
    return 0


__all__ = ["cli", "run_locator_workflow"]
