from __future__ import annotations

import typer
from pathlib import Path
from typing import List, Tuple

import polars as pl

from nvcenter.sim import (
    CompositeNoise,
    DriftNoise,
    ExperimentRunner,
    FluorescenceCount,
    GaussianNoise,
    OutlierSpikes,
    PoissonNoise,
    RabiEstimate,
    RabiGenerator,
    T1Estimate,
    T1Generator,
    # Locator layer
    GridScan,
    GoldenSectionSearch,
    TwoPeakGreedy,
    OnePeakGenerator,
    TwoPeakGenerator,
)
from nvcenter.sim.loc_runner import LocatorRunner
from nvcenter.cache import DataFrameCache
from nvcenter.viz import plot_experiment_summary, plot_locator_summary


# -----------------------------
# Scenario presets using existing components
# -----------------------------

def _noise_presets() -> List[Tuple[str, CompositeNoise | None]]:
    return [
        ("NoNoise", None),
        ("Gauss(0.05)", CompositeNoise([GaussianNoise(0.05)])),
        (
            "Heavy",
            CompositeNoise([GaussianNoise(0.1), DriftNoise(0.05), OutlierSpikes(0.02, 0.5)]),
        ),
        ("Poisson(50)", CompositeNoise([PoissonNoise(scale=50.0)])),
    ]


def _exp_strategies() -> List:
    # Include a small suite; RMSE will be computed on overlapping keys per generator
    return [FluorescenceCount(), RabiEstimate(), T1Estimate()]


def _locator_strategies() -> List[Tuple[str, object]]:
    return [
        ("Grid21", GridScan(n_points=21)),
        ("Golden20", GoldenSectionSearch(max_evals=20)),
        ("TwoGreedy", TwoPeakGreedy(coarse_points=21, refine_points=5)),
    ]


# -----------------------------
# Orchestration
# -----------------------------

def run_experiment_workflow(out_dir: Path, repeats: int, rng_seed: int | None) -> pl.DataFrame:
    runner = ExperimentRunner(rng_seed=rng_seed)
    noises = [n for _, n in _noise_presets()]
    strategies = _exp_strategies()

    # Cache key based on scenario configuration
    cache_dir = out_dir / "cache"
    cfg = {
        "kind": "experiment",
        "generators": ["Rabi", "T1"],
        "noises": [name for name, _ in _noise_presets()],
        "strategies": [s.__class__.__name__ for s in strategies],
        "repeats": int(repeats),
        "seed": int(rng_seed) if rng_seed is not None else None,
    }
    cache = DataFrameCache(cache_dir)
    key = DataFrameCache.make_key(cfg)
    cached = cache.load_df(key)
    if cached is not None:
        df_all = cached
    else:
        # Execute sweeps per generator and annotate with Polars
        df_rabi = runner.sweep(RabiGenerator(), noises, strategies, repeats=repeats).with_columns(
            pl.lit("Rabi").alias("generator")
        )
        df_t1 = runner.sweep(T1Generator(), noises, strategies, repeats=repeats).with_columns(
            pl.lit("T1").alias("generator")
        )
        df_all = pl.concat([df_rabi, df_t1], how="diagonal_relaxed")
        cache.save_df(df_all, key)

    out_path = out_dir / "experiment_results.csv"
    df_all.write_csv(out_path.as_posix())
    return df_all


def run_locator_workflow(
    out_dir: Path, repeats: int, rng_seed: int | None, max_steps: int
) -> pl.DataFrame:
    runner = LocatorRunner(rng_seed=rng_seed)

    # Generators: OnePeak in different modes + TwoPeak
    generators: List[Tuple[str, callable]] = [
        ("OnePeak-gaussian", lambda rng: OnePeakGenerator(mode="gaussian").generate(rng)),
        ("OnePeak-rabi", lambda rng: OnePeakGenerator(mode="rabi").generate(rng)),
        (
            "OnePeak-t1_decay",
            lambda rng: OnePeakGenerator(mode="t1_decay").generate(rng),
        ),
        ("TwoPeak", lambda rng: TwoPeakGenerator().generate(rng)),
    ]
    strategies: List[Tuple[str, object]] = _locator_strategies()
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
    out: Path = typer.Option(Path("artifacts"), "--out", help="Output directory"),
    repeats: int = typer.Option(5, "--repeats", help="Number of repeats per scenario"),
    seed: int = typer.Option(123, "--seed", help="RNG seed (int)"),
    loc_max_steps: int = typer.Option(150, "--loc-max-steps", help="Max steps for locator measurement loop"),
) -> int:
    out_dir: Path = out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run both workflows
    df_exp = run_experiment_workflow(out_dir, repeats=repeats, rng_seed=seed)
    df_loc = run_locator_workflow(out_dir, repeats=repeats, rng_seed=seed, max_steps=loc_max_steps)

    # Visualizations
    try:
        exp_imgs = plot_experiment_summary(df_exp, out_dir)
        loc_imgs = plot_locator_summary(df_loc, out_dir)
        if exp_imgs or loc_imgs:
            print("Saved plots:")
            for p in [*exp_imgs, *loc_imgs]:
                print(f" - {p}")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    # Combine into a single table for convenience using Polars' relaxed concat
    df_combined = pl.concat(
        [
            df_exp.with_columns(pl.lit("experiment").alias("kind")),
            df_loc.with_columns(pl.lit("locator").alias("kind")),
        ],
        how="diagonal_relaxed",
    )
    (out_dir / "combined_results.csv").write_text(df_combined.write_csv())

    # Also print a tiny summary
    try:
        by_noise = (
            df_exp.group_by(["generator", "noise", "strategy"]).agg(pl.col("rmse").mean())
        )
        print("Experiment summary (mean rmse):")
        print(by_noise.sort(["generator", "noise", "strategy"]))
    except Exception:
        pass

    print(f"Wrote results to: {out_dir}")
    return 0


def main() -> None:
    """Entry point that runs the Typer-powered CLI."""
    typer.run(cli)


if __name__ == "__main__":
    main()
