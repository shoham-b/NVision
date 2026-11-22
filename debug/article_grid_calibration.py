"""Calibration and sweep placement on the cached article grid (read-only over locator_results.parquet).

Item 4: SBED (and sweep) coverage of the final error by the claimed sigma, per window span and SNR, on the pure
Lorentzian single-dip generators (gs0, hf unresolved). Item 5: the sweep's placement efficiency after detection
(stage sensitivities) against 8 pi Gamma / (27 W_drive).

Usage: uv run python debug/article_grid_calibration.py [path to locator_results.parquet]
"""

import sys

import numpy as np
import polars as pl

path = sys.argv[1] if len(sys.argv) > 1 else "../../../artifacts/locator_results.parquet"
df = pl.read_parquet(path).filter(
    pl.col("generator").str.contains("-gs0-") & pl.col("generator").str.contains("hfunresolved")
)
df = df.with_columns((pl.col("final_err_shift").abs() / pl.col("uncert")).alias("z"))
print("strategies:", sorted(df["strategy"].unique().to_list()))
rows = []
for (gen, _noise, strat), g in df.group_by(["generator", "noise", "strategy"]):
    z = g["z"].drop_nulls().drop_nans().to_numpy()
    if z.size == 0:
        continue
    rows.append(
        dict(
            span=float(g["stage_window_fwhm"].median())
            if g["stage_window_fwhm"].null_count() < len(g)
            else float("nan"),
            gen=gen,
            snr=float(g["snr"].median()),
            strat=strat,
            n=z.size,
            cov1=float(np.mean(z < 1)),
            cov2=float(np.mean(z < 2)),
            cov3=float(np.mean(z < 3)),
            med_z=float(np.median(z)),
            gross=float(np.mean(z > 10)),
        )
    )
out = pl.DataFrame(rows).sort(["strat", "span", "snr"])
with pl.Config(tbl_rows=200, tbl_width_chars=200, tbl_cols=20, ascii_tables=True):
    print(out.select("strat", "gen", "span", "snr", "n", "cov1", "cov2", "cov3", "med_z", "gross"))
    print("\nPooled per strategy x span (nominal Gaussian: cov1 0.683, cov2 0.954, med_z 0.674):")
    print(
        out.group_by(["strat", "gen"])
        .agg(
            pl.col("n").sum(),
            (pl.col("cov1") * pl.col("n")).sum() / pl.col("n").sum(),
            (pl.col("cov2") * pl.col("n")).sum() / pl.col("n").sum(),
            pl.col("med_z").median(),
            pl.col("gross").max(),
        )
        .sort(["strat", "gen"])
    )
