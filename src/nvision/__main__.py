from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from nvision.cache import DataFrameCache
from nvision.sim import (
    CompositeNoise,
    DriftNoise,
    GaussianNoise,
    GoldenSectionSearch,
    # Locator layer
    GridScan,
    OutlierSpikes,
    PoissonNoise,
    TwoPeakGreedy,
)
from nvision.sim.gen import (
    GaussianManufacturer,
    OnePeakGenerator,
    RabiManufacturer,
    T1DecayManufacturer,
    TwoPeakGenerator,
)
from nvision.sim.loc_runner import LocatorRunner
from nvision.viz import plot_locator_summary
from nvision.sim import cases as sim_cases

# -----------------------------
# Scenario presets using existing components
# -----------------------------


def _noise_presets() -> list[tuple[str, CompositeNoise | None]]:
    # Start simple and evolve: no noise -> single noises -> complex combos
    return (
        sim_cases.noises_none()
        + sim_cases.noises_single_each()
        + sim_cases.noises_complex()
    )


def _locator_strategies() -> list[tuple[str, object]]:
    return [
        ("Grid21", GridScan(n_points=21)),
        ("Golden20", GoldenSectionSearch(max_evals=20)),
        ("TwoGreedy", TwoPeakGreedy(coarse_points=21, refine_points=5)),
    ]


# -----------------------------
# Orchestration
# -----------------------------


def run_locator_workflow(
    out_dir: Path,
    repeats: int,
    rng_seed: int | None,
    max_steps: int,
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
        df = runner.sweep(generators, strategies, noises, repeats=repeats, max_steps=max_steps)
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
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    # Save locator results only
    (out_dir / "locator_results.csv").write_text(df_loc.write_csv())

    print(f"Wrote locator results to: {out_dir}")
    return 0


def main() -> None:
    """Entry point that runs the Typer-powered CLI."""
    typer.run(cli)


if __name__ == "__main__":
    main()
