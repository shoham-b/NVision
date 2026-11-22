"""Item 5: why the sweep's placement efficiency sits below 8 pi Gamma / (27 W). Read-only over the parquet.

Measured: shot-weighted mean of stage{1,2,3}_sens_eff (= mean eta over the whole run; a sweep does not change
behaviour at detection) for SimpleSweep-optimal on the pure-Lorentzian article grid.
Predicted, three ways, per span:
  closed_drive  8 pi Gamma / (27 W_drive), W_drive = span*Gamma + 2 pads (what /api/refine-rates calls theory)
  closed_prior  8 pi Gamma / (27 span*Gamma)
  exact_edge    the continuous average of g^2/max g^2 over the drive window, averaged over the dip drawn uniformly in
                [pad, W_drive - pad] (edge truncation of the slope tails, no grid)
  grid_k        the same on the sweep's own grid, linspace(0, W_drive, n) with both ends, n = ceil(W/(FWHM_prior/k))+1,
                for k = 2, 3, 4 points per true FWHM (the realised k* is not in the parquet). The sum over the grid
                is about integral / spacing with spacing W/(n-1), so the grid mean is about (n-1)/n of the
                continuous one: a bias that vanishes as n grows.
Usage: uv run python debug/sweep_q_finite_window.py <parquet>
"""

import math
import sys

import numpy as np
import polars as pl


def g2(u):  # squared slope of a unit Lorentzian in units of gamma: (2u/(1+u^2)^2)^2
    return (2 * u / (1 + u * u) ** 2) ** 2


G2MAX = g2(1 / math.sqrt(3))
df = pl.read_parquet(sys.argv[1]).filter(
    pl.col("generator").str.contains("-gs0-") & pl.col("generator").str.contains("hfunresolved")
)
print(
    "span W_drive/G  measured  closed_drive  closed_prior  exact_edge  grid_k2  grid_k3  grid_k4"
    "  meas/closed_drive  meas/grid_k3"
)
for gen in sorted(df["generator"].unique().to_list(), key=lambda s: float(s.split("span")[1].split("-")[0])):
    g = df.filter((pl.col("generator") == gen) & (pl.col("strategy") == "SimpleSweep-optimal"))
    num = sum((g[f"stage{k}_sens_eff"].fill_null(0) * g[f"stage{k}_steps"].fill_null(0)).sum() for k in (1, 2, 3))
    den = sum(g[f"stage{k}_steps"].fill_null(0).sum() for k in (1, 2, 3))
    measured = num / den
    span = float(gen.split("span")[1].split("-")[0])
    wd = float(g["stage_window_fwhm"].median())  # drive window in FWHM
    fwhm = float(g["stage_fwhm_true_phys"].median())
    gamma_fwhm = 0.5  # HWHM in FWHM units
    pad = (wd - span) / 2
    # exact continuous, dip uniform in [pad, wd - pad]
    x = np.linspace(0, wd, 20001)
    centres = np.linspace(pad, wd - pad, 401)
    edge = np.mean([np.mean(g2((x - c) / gamma_fwhm)) / G2MAX for c in centres])
    # k counted per TRUE FWHM here: the stored prior FWHM is ~3.85x the true one, so k* per prior FWHM is ~4x larger.
    prior_fwhm = 1.0
    grids = {}
    for k in (2, 3, 4):
        n_pts = math.ceil(wd / (prior_fwhm / k)) + 1
        grid = np.linspace(0, wd, n_pts)
        grids[k] = np.mean([np.mean(g2((grid - c) / gamma_fwhm)) / G2MAX for c in centres])
    cd = 8 * math.pi / (27 * wd)
    cp = 8 * math.pi / (27 * span)
    print(
        f"{span:5.1f} {wd:6.1f} {measured:9.4f} {cd:11.4f} {cp:12.4f} {edge:10.4f} {grids[2]:8.4f} {grids[3]:8.4f}"
        f" {grids[4]:8.4f} {measured / cd:10.3f} {measured / grids[3]:12.3f}"
    )
