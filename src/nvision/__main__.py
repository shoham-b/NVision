from __future__ import annotations

import html
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from nvision.cache import DataFrameCache
from nvision.sim import (
    CompositeNoise,
    GoldenSectionSearch,
    # Locator layer
    GridScan,
    TwoPeakGreedy,
)
from nvision.sim import cases as sim_cases
from nvision.sim.loc_runner import LocatorRunner
from nvision.viz import plot_locator_summary


def _slugify_for_path(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value)
    slug = slug.strip("_")
    return slug or "item"


# -----------------------------
# Scenario presets using existing components
# -----------------------------


def _noise_presets() -> list[tuple[str, CompositeNoise | None]]:
    # Start simple and evolve: no noise -> single noises -> complex combos
    return sim_cases.noises_none() + sim_cases.noises_single_each() + sim_cases.noises_complex()


def _locator_strategies() -> list[tuple[str, object]]:
    return [
        ("Grid21", GridScan(n_points=21)),
        ("Golden20", GoldenSectionSearch(max_evals=20)),
        ("TwoGreedy", TwoPeakGreedy(coarse_points=21, refine_points=5)),
    ]


# -----------------------------
# Orchestration
# -----------------------------


def compile_html_index(out_dir: Path) -> Path:
    """Create a simple index.html in out_dir that links to outputs and embeds generated graphs.

    - Links to locator_results.csv (if present)
    - Embeds images from out_dir/graphs
    - If project-level htmlcov exists, copy it into out_dir/htmlcov and link to it
    Returns the path to the generated index file.
    """
    index_path = out_dir / "index.html"

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append('<head><meta charset="utf-8"><title>NVision results</title></head>')
    parts.append("<body>")
    parts.append(f"<h1>NVision results ({html.escape(out_dir.as_posix())})</h1>")

    # Link to CSV result
    csv_path = out_dir / "locator_results.csv"
    if csv_path.exists():
        parts.append(
            f'<p><a href="{html.escape(csv_path.name)}">Download locator_results.csv</a></p>'
        )

    # Embed graphs if any
    graphs_dir = out_dir / "graphs"
    if graphs_dir.exists() and graphs_dir.is_dir():
        parts.append("<h2>Generated graphs</h2>")
        parts.append('<div style="display:flex;flex-wrap:wrap;gap:12px;">')
        for img in sorted(graphs_dir.iterdir()):
            if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".gif"):
                rel = Path("graphs") / img.name
                parts.append(
                    f'<figure style="width:300px;">'
                    f'<img src="{html.escape(str(rel.as_posix()))}" '
                    f'alt="{html.escape(img.name)}" '
                    f'style="max-width:100%;height:auto;"/>'
                    f"<figcaption>{html.escape(img.name)}</figcaption></figure>"
                )
            parts.append("</div>")

        # Optionally include coverage report if present at project root/htmlcov
        project_htmlcov = Path("htmlcov")
        target_htmlcov = out_dir / "htmlcov"
        if project_htmlcov.exists() and project_htmlcov.is_dir():
            # Copy into out_dir/htmlcov (replace if exists)
            try:
                if target_htmlcov.exists():
                    shutil.rmtree(target_htmlcov)
                shutil.copytree(project_htmlcov, target_htmlcov)
                parts.extend(
                    [
                        "<h2>Coverage report</h2>",
                        '<p><a href="htmlcov/index.html">View coverage report</a></p>',
                    ]
                )
            except Exception as e:  # pragma: no cover - best-effort copy
                parts.extend(
                    [
                        '<p style="color:#b91c1c">',
                        "Couldn't copy coverage report: ",
                        html.escape(str(e)),
                        "</p>",
                    ]
                )

    parts.append("</body></html>")

    index_path.write_text("\n".join(parts), encoding="utf-8")
    return index_path


def run_locator_workflow(
    out_dir: Path,
    repeats: int,
    rng_seed: int | None,
    max_steps: int,
    history_callback: Callable[[str, str, str, int, object, pl.DataFrame], None] | None = None,
) -> pl.DataFrame:
    runner = LocatorRunner(rng_seed=rng_seed)

    # Generators: start with simple, then expand (managed in sim/cases.py)
    generators: list[tuple[str, object]] = sim_cases.generators_basic()
    strategies: list[tuple[str, object]] = _locator_strategies()
    noises = _noise_presets()

    # Polars-defined scenario grid (for visibility and extension)
    df_g = pl.DataFrame({"generator": [name for name, _ in generators]})
    df_n = pl.DataFrame({"noise": [name for name, _ in noises]})
    df_s = pl.DataFrame({"strategy": [name for name, _ in strategies]})
    _ = df_g.join(df_n, how="cross").join(df_s, how="cross")

    # Cache key based on scenario configuration
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
    cached = cache.load_df(key)
    if cached is not None:
        df = cached
    else:
        df = runner.sweep(
            generators,
            strategies,
            noises,
            repeats=repeats,
            max_steps=max_steps,
            history_callback=history_callback,
        )
        cache.save_df(df, key)

    out_path = out_dir / "locator_results.csv"
    df.write_csv(out_path.as_posix())
    return df


def cli(
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("artifacts"),
    repeats: Annotated[int, typer.Option("--repeats", help="Number of repeats per scenario")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed (int)")] = 123,
    loc_max_steps: Annotated[
        int,
        typer.Option("--loc-max-steps", help="Max steps for locator measurement loop"),
    ] = 150,
) -> int:
    out_dir: Path = out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run locator workflow only
    df_loc = run_locator_workflow(out_dir, repeats=repeats, rng_seed=seed, max_steps=loc_max_steps)

    # Visualizations
    try:
        graphs_dir = out_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        loc_imgs = plot_locator_summary(df_loc, graphs_dir)
        if loc_imgs:
            print("Saved plots:")
            for p in loc_imgs:
                print(f" - {p}")
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"[warn] plotting failed: {e}")

    # Save locator results
    # write_csv already saved to disk in run_locator_workflow; retrying for older
    # polars version compatibility
    from contextlib import suppress

    with suppress(Exception):
        # df_loc.write_csv() may return None in newer versions; file already written
        (out_dir / "locator_results.csv").write_text(df_loc.write_csv())

    # Build an HTML index to browse outputs locally
    try:
        idx = compile_html_index(out_dir)
        print(f"Generated HTML index at: {idx.absolute().as_uri()}")
    except Exception as e:  # pragma: no cover - best-effort
        print(f"[warn] failed to build HTML index: {e}")

    print(f"Wrote locator results to: {out_dir}")
    return 0


def main() -> None:
    """Entry point that runs the Typer-powered CLI."""
    typer.run(cli)


if __name__ == "__main__":
    main()
