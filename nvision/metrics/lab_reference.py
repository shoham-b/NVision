"""Reference answer for a real ESR recording, from the sweeps the locator never saw.

Real data has no ground truth, so the honesty of a locator's claimed sigma is checked against an
independent fit: :meth:`~nvision.tools.matlab_loader.MatlabDataFile.split_sweeps` gives the locator
the even sweeps, and :func:`fit_reference` fits the odd ones with the *same* forward model the
locator's belief uses (Zeeman Lorentzian, hyperfine unresolved, background 1). Because the two
halves share no shots, ``z = (estimate - ref) / sqrt(sigma^2 + sigma_ref^2)`` is a fair coverage
check.

One recording gives one independent z per parameter: repeats with other seeds draw from the same
locator half and meet the same reference, so coverage is a share *across files*, never across
seeds on one file.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from nvision.spectra.numba_kernels import nv_center_zeeman_lorentzian_eval
from nvision.spectra.nv_center import _check_hyperfine

log = logging.getLogger(__name__)

_MHZ = 1e6
# Parameters whose z the holdout check reports: the primary (zeeman_shift) and the centre.
HOLDOUT_PARAMS: tuple[str, ...] = ("zeeman_shift", "zero_field_freq")
# chi^2/dof above this means the model does not describe the held-out data; sigma_ref is then too small.
CHI2_DOF_WARN = 2.0


@dataclass(frozen=True)
class ReferenceFit:
    """Weighted least-squares fit of the held-out sweeps; physical units (Hz for frequencies)."""

    values_phys: dict[str, float]
    sigmas_phys: dict[str, float]
    chi2_dof: float
    n_bins: int
    n_shots: int
    noise_std: float  # pooled per-shot std of the held-out shots (ddof per bin)
    # Free parameters the fit left on a bound of the locator's prior (e.g. zeeman_shift = 0 when the doublet is not
    # resolved): their sigma is meaningless (the Gaussian covariance does not apply at an edge), so no z is formed.
    at_bound: tuple[str, ...] = ()


def _model_signal(drive_freq_phys: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    """The locator belief's forward model, evaluated per bin; shape (n_bins,)."""
    hf_offset, w_center, _ = _check_hyperfine("unresolved", False)
    return np.array(
        [
            nv_center_zeeman_lorentzian_eval(
                float(x),
                p["zero_field_freq"],
                p["hwhm"],
                p["zeeman_shift"],
                hf_offset,
                1.0,
                w_center,
                p["contrast_total"],
                1.0,
            )
            for x in drive_freq_phys
        ]
    )


def fit_reference(
    drive_freq_phys: np.ndarray,
    shot_ratios: np.ndarray,
    bounds_phys: Mapping[str, tuple[float, float]],
    fixed_zero_field_freq_phys: float | None = None,
) -> ReferenceFit:
    """Fit the held-out shots' per-bin means with the locator's model.

    *drive_freq_phys*: shape (n_bins,), Hz. *shot_ratios*: shape (n_bins, n_shots), the held-out
    half (other columns NaN). *bounds_phys*: the locator's bounds for ``zero_field_freq``, ``hwhm``,
    ``zeeman_shift`` and ``contrast_total``. Each bin mean is weighted by ``noise_std / sqrt(n)``,
    the same homoscedastic per-shot noise the locator assumes, rather than by its own few-shot std.
    ``noise_std`` is pooled over bins with one degree of freedom spent per bin mean: with only a
    few sweeps per half, the median of per-bin ``ddof=0`` stds (what the loader uses) is biased
    low by up to sqrt(2) and would inflate chi^2 and shrink sigma_ref.
    With *fixed_zero_field_freq_phys* the centre is held there, as ``--no-infer-center-freq`` does.
    """
    drive_freq_phys = np.asarray(drive_freq_phys, dtype=float)
    n_per_bin = np.sum(np.isfinite(shot_ratios), axis=1)
    keep = n_per_bin > 0
    free = ["zero_field_freq", "hwhm", "zeeman_shift", "contrast_total"]
    if fixed_zero_field_freq_phys is not None:
        free.remove("zero_field_freq")
    if keep.sum() < len(free) + 5:
        raise ValueError(f"Only {int(keep.sum())} bins hold held-out shots; too few for a reference fit")
    x_phys = drive_freq_phys[keep]
    with np.errstate(all="ignore"):
        y = np.nanmean(shot_ratios[keep], axis=1)
    resid_dof = int(np.sum(np.maximum(n_per_bin - 1, 0)))
    if resid_dof < 10:
        raise ValueError(f"Only {resid_dof} residual degrees of freedom to estimate the shot noise from")
    with np.errstate(all="ignore"):
        bin_means = np.nanmean(shot_ratios, axis=1, keepdims=True)
    noise_std = float(np.sqrt(np.nansum((shot_ratios - bin_means) ** 2) / resid_dof))
    sigma = noise_std / np.sqrt(n_per_bin[keep])

    # Fit in MHz offsets from the window centre: raw Hz around 2.87e9 conditions the Jacobian badly.
    centre_phys = 0.5 * (drive_freq_phys.min() + drive_freq_phys.max())
    scale = {"zero_field_freq": _MHZ, "hwhm": _MHZ, "zeeman_shift": _MHZ, "contrast_total": 1.0}
    offset = {"zero_field_freq": centre_phys, "hwhm": 0.0, "zeeman_shift": 0.0, "contrast_total": 0.0}

    def to_phys(vec: np.ndarray) -> dict[str, float]:
        p = {name: float(v) * scale[name] + offset[name] for name, v in zip(free, vec, strict=True)}
        if fixed_zero_field_freq_phys is not None:
            p["zero_field_freq"] = float(fixed_zero_field_freq_phys)
        return p

    def to_fit(name: str, value_phys: float) -> float:
        return (value_phys - offset[name]) / scale[name]

    def f(_x: np.ndarray, *vec: float) -> np.ndarray:
        return _model_signal(x_phys, to_phys(np.asarray(vec)))

    lo = np.array([to_fit(n, bounds_phys[n][0]) for n in free])
    hi = np.array([to_fit(n, bounds_phys[n][1]) for n in free])

    # Coarse grid over (centre, shift, width) with the contrast solved in closed form: the model is
    # 1 - contrast * g, linear in contrast. A single start can land in the mode where one dip is
    # parked outside the window and the other "explains" everything.
    w = 1.0 / sigma**2
    seeds = []
    centres = (
        [fixed_zero_field_freq_phys]
        if fixed_zero_field_freq_phys is not None
        else np.linspace(*bounds_phys["zero_field_freq"], 25)
    )
    for c in centres:
        for s in np.linspace(*bounds_phys["zeeman_shift"], 16)[1:]:
            for h in np.geomspace(*bounds_phys["hwhm"], 4):
                g = 1.0 - _model_signal(
                    x_phys, {"zero_field_freq": c, "hwhm": h, "zeeman_shift": s, "contrast_total": 1.0}
                )
                denom = float(np.sum(w * g * g))
                if denom <= 0:
                    continue
                ct = float(np.clip(np.sum(w * g * (1.0 - y)) / denom, *bounds_phys["contrast_total"]))
                chi2 = float(np.sum(w * (y - (1.0 - ct * g)) ** 2))
                seeds.append((chi2, {"zero_field_freq": c, "hwhm": h, "zeeman_shift": s, "contrast_total": ct}))
    seeds.sort(key=lambda t: t[0])

    best = None
    for _, seed in seeds[:5]:
        p0 = np.clip([to_fit(n, seed[n]) for n in free], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        popt, pcov = curve_fit(f, x_phys, y, p0=p0, sigma=sigma, absolute_sigma=True, bounds=(lo, hi), method="trf")
        chi2 = float(np.sum(((y - f(x_phys, *popt)) / sigma) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, popt, pcov)
    if best is None:
        raise ValueError("Reference fit found no usable starting point")
    chi2, popt, pcov = best
    diag = np.diag(pcov)
    if not np.all(np.isfinite(diag)) or np.any(diag < 0):
        raise ValueError("Reference fit covariance is not finite (a parameter is probably pinned on a bound)")

    values = to_phys(popt)
    sigmas = {name: float(math.sqrt(d)) * scale[name] for name, d in zip(free, diag, strict=True)}
    edge = 1e-6 * (hi - lo)
    at_bound = tuple(n for n, v, a, b, e in zip(free, popt, lo, hi, edge, strict=True) if v - a <= e or b - v <= e)
    if fixed_zero_field_freq_phys is not None:
        sigmas["zero_field_freq"] = 0.0
    dof = int(keep.sum()) - len(free)
    chi2_dof = chi2 / dof
    if chi2_dof > CHI2_DOF_WARN:
        log.warning(
            "Reference fit chi2/dof = %.2f: the model does not describe the held-out data, so sigma_ref is too small",
            chi2_dof,
        )
    return ReferenceFit(
        values_phys=values,
        sigmas_phys=sigmas,
        chi2_dof=chi2_dof,
        n_bins=int(keep.sum()),
        n_shots=int(n_per_bin.sum()),
        noise_std=noise_std,
        at_bound=at_bound,
    )


def holdout_metrics(
    estimates_phys: Mapping[str, float], uncertainties_phys: Mapping[str, float], ref: ReferenceFit
) -> dict[str, float | int]:
    """Flat ``ref_*`` / ``holdout_z_*`` columns for the result row (physical units).

    ``holdout_z_<p>`` is signed: ``(estimate - ref) / sqrt(sigma^2 + sigma_ref^2)``. A parameter the reference
    left on a prior bound gets ``ref_<p>_at_bound = 1`` and no z: there is nothing to check it against.
    """
    out: dict[str, float | int] = {
        "ref_chi2_dof": ref.chi2_dof,
        "ref_n_bins": ref.n_bins,
        "ref_n_shots": ref.n_shots,
        "ref_noise_std": ref.noise_std,
    }
    for name, value in ref.values_phys.items():
        out[f"ref_{name}"] = value
        out[f"ref_{name}_sigma"] = ref.sigmas_phys[name]
    for name in HOLDOUT_PARAMS:
        sigma_ref = ref.sigmas_phys[name]
        if sigma_ref == 0.0:  # held fixed: nothing to check
            continue
        if name in ref.at_bound:
            out[f"ref_{name}_at_bound"] = 1
            continue
        combined = math.hypot(float(uncertainties_phys[name]), sigma_ref)
        out[f"holdout_z_{name}"] = (float(estimates_phys[name]) - ref.values_phys[name]) / combined
    return out


# The lab-study name under which the UI lists every processed raw sweep (see lab_sweeps_payload).
SWEEPS_LAB_NAME = "raw sweeps (SBED)"


def _mhz(value: float | None) -> float | None:
    return None if value is None else value / _MHZ


def lab_sweeps_payload(records: list[dict]) -> dict:
    """The UI's raw-sweeps lab page: one group per (source, folder), coverage within each group.

    *records*: :func:`nvision.tools.lab_sources.load_sweep_records`. Frequencies in the rows are MHz. A folder is
    one experiment (same sample, averaging and window), so |z| coverage is never pooled across folders; ``n`` is
    the number of files it is over. Failed files are listed under ``failed``.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    failed = []
    for rec in records:
        if "error" in rec:
            failed.append({"label": rec["label"], "file": rec["file"], "error": rec["error"]})
            continue
        full, held = rec["full"], rec["holdout"]
        ref = held["holdout"]
        groups.setdefault((rec["label"], rec["folder"]), []).append(
            {
                "file": rec["file"].rsplit("/", 1)[-1],
                "n_sweeps": full["n_valid_shots"],
                "window_mhz": full["drive_freq_max_mhz"] - full["drive_freq_min_mhz"],
                "steps": full["total_steps"],
                "converged_step": full["primary_converged_step"],
                "zeeman_mhz": _mhz(full["estimates"]["zeeman_shift"]),
                "zeeman_sigma_mhz": _mhz(full["uncertainties"]["zeeman_shift"]),
                "holdout_zeeman_mhz": _mhz(held["estimates"]["zeeman_shift"]),
                "holdout_zeeman_sigma_mhz": _mhz(held["uncertainties"]["zeeman_shift"]),
                "ref_zeeman_mhz": _mhz(ref["ref_zeeman_shift"]),
                "ref_zeeman_sigma_mhz": _mhz(ref["ref_zeeman_shift_sigma"]),
                "z_zeeman": ref.get("holdout_z_zeeman_shift"),
                "z_center": ref.get("holdout_z_zero_field_freq"),
                "ref_zeeman_at_bound": bool(ref.get("ref_zeeman_shift_at_bound")),
                "ref_center_at_bound": bool(ref.get("ref_zero_field_freq_at_bound")),
                "ref_chi2_dof": ref["ref_chi2_dof"],
                "shot_reuse": held["shot_reuse"],
            }
        )
    out = []
    for (label, folder), rows in sorted(groups.items()):
        coverage = {}
        for key in ("z_zeeman", "z_center"):
            zs = np.array([abs(r[key]) for r in rows if r[key] is not None])
            coverage[key] = (
                {
                    "n": int(zs.size),
                    "within_1": float(np.mean(zs <= 1)),
                    "within_2": float(np.mean(zs <= 2)),
                    "median_abs": float(np.median(zs)),
                }
                if zs.size
                else {"n": 0}
            )
        out.append({"label": label, "folder": folder or ".", "rows": rows, "coverage": coverage})
    return {"groups": out, "failed": failed, "chi2_dof_warn": CHI2_DOF_WARN}
