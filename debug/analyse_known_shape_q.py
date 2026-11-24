"""Summarise the dry-run studies written by ``debug/repro_known_shape_q.py`` (scratch/ks/*.json).

usage: uv run python debug/analyse_known_shape_q.py <json> [<json> ...]

Per file: placement q (mean oracle eta) over fixed step ranges -- fixed ranges, not "k steps after detection", so arms
that detect at different times are compared at the same run time --, where the measurements sit (share of the steps
in the range at |offset| < 0.3 HWHM = the dip centre, 0.3-1 = the steep sides, 1-3 = the wings, > 3 = far), and the
claimed sigma's coverage of the true error (cov1/cov2 of err/sigma, median err/sigma) at a few steps. With
``--track SNR STD_FWHM FWHM_HZ`` also the van Trees ratio R(n) = sigma_vT(n)^2 / mean(err(n)^2),
sigma_vT^-2 = J0 + n I_max with J0 = 1/sigma_0^2 and I_max = (SNR * 3 sqrt3 / 8 / gamma)^2 for a Lorentzian of
depth C and noise C / SNR (R <= 1 is required; R near 1 means the bound is followed).
"""

import argparse
import json
import math

import numpy as np

RANGES = ((20, 100), (100, 300), (300, 800))
STEPS = (50, 100, 300, 800)
p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--track", nargs=3, type=float, metavar=("SNR", "STD_FWHM", "FWHM_HZ"))
a = p.parse_args()
for path in a.files:
    with open(path) as f:
        d = json.load(f)
    reps = d["repeats"]
    if not reps:
        print(path, "no repeats")
        continue
    n = min(len(r["eta"]) for r in reps)
    print(f"== {d['generator']} {d['noise']} {d['strategy']}  ({len(reps)} repeats, {n} steps)")
    for lo, hi in RANGES:
        if hi > n:
            continue
        q = [float(np.mean(r["eta"][lo:hi])) for r in reps]
        off = np.abs(np.concatenate([np.asarray(r["offset_hwhm"][lo:hi]) for r in reps]))
        shares = [
            np.mean(off < 0.3),
            np.mean((off >= 0.3) & (off < 1)),
            np.mean((off >= 1) & (off < 3)),
            np.mean(off >= 3),
        ]
        print(
            f"  steps {lo:4d}-{hi:4d}: q median {np.median(q):.3f} [q25 {np.quantile(q, 0.25):.3f}, "
            f"q75 {np.quantile(q, 0.75):.3f}]  offsets centre/sides/wings/far " + "/".join(f"{x:.2f}" for x in shares)
        )
    for k in STEPS:
        if k > n:
            continue
        z = np.array([r["err"][k - 1] / r["unc"][k - 1] for r in reps])
        print(
            f"  n={k:4d}: cov1 {np.mean(z < 1):.2f} cov2 {np.mean(z < 2):.2f} median err/sigma {np.median(z):.2f} "
            f"(calibrated: 0.68 / 0.95 / 0.67)"
        )
    if a.track:
        snr, std_fwhm, fwhm = a.track
        gamma = fwhm / 2.0
        i_max = (snr * 3.0 * math.sqrt(3.0) / 8.0 / gamma) ** 2
        if "i_max" in reps[0]:  # the run's own, from the true signal and noise
            i_max = float(np.median([r["i_max"] for r in reps]))
        j0 = 1.0 / (std_fwhm * fwhm) ** 2
        for k in (1, 3, 10, 30, 100, 300):
            if k > n:
                continue
            mse = float(np.mean([r["err"][k - 1] ** 2 for r in reps]))
            print(
                f"  van Trees R(n={k:3d}) = {1.0 / (j0 + k * i_max) / mse:.3f}  (n_c = J0/I_max = {j0 / i_max:.2f}, "
                f"{len(reps)} runs: R's relative standard error about {math.sqrt(2.0 / len(reps)):.2f})"
            )
