"""The lab version of :mod:`nvision.metrics.detection_aligned`: live Bayesian ESR runs aligned at their own detection.

A lab run has no true resonance and stores no per-step particles, so three things differ from the simulation:

* **Reference.** The error of a run is its posterior mean minus a reference centre (MHz) given by the caller, here
  the lab study's reference (the median final mean of its converged runs, :func:`nvision.metrics.lab_study.
  lab_study_payload`). The line drifts 0.4-0.6 MHz between recordings, so the squared error has a floor of about
  0.2-0.4 MHz^2 that no number of measurements removes: late lags measure the drift, not the strategy.
* **Detection and restart.** Detection is :func:`nvision.metrics.lab_search_fit.detection_step`, the step right after
  the run's first steep, lasting drop of its claimed sigma (the run's own history, no truth). The restart point is
  that step, ``n_d``: lag ``k = n - n_d``, so lag 0 is the first posterior after the drop (as in the lab van Trees fit).
* **Bound.** There is no posterior Fisher information ``J_tau`` (no per-step particles), so the restarted curve uses
  ``J_d = 12 / Delta d^2`` of the elimination model only (Eq. after-detection, a comparison rather than a strict
  bound), with ``Delta d`` from eq:sm-reach for the lab line. ``I_max = (b C 3 sqrt3 / (8 gamma))^2 / sigma^2`` is the
  Fisher information of one measurement at the steepest slope of the Lorentzian dip the run's model assumes
  (baseline ``b``, contrast ``C``, half width ``gamma``; the lab holds them fixed), with ``sigma`` the noise of the
  run's own measurements around that model at its final estimate.
* **Pinned restart.** At the lab SNR (about 4-5) eq:sm-reach often gives no single-measurement reach, so ``J_d``
  does not exist. The curve is then also drawn from ``J_pinned = 1 / <sigma_claimed^2>`` at lag 0, as the lab van
  Trees fit pins it: a description of the data from detection on, not a bound. Both are reported, by name.
* **Claimed rate.** ``q_hat``, the slope of the inverse mean claimed variance per measurement in units of ``I_max``
  over the second half of the lags (Sec. protocol): the true-error rate is not readable past the drift floor.
* **Placement.** Where the runs measure after detection: the mean ``I(d)/I_max`` about each run's own final mean,
  and the share of measurements more than ``FAR_HWHM`` half widths off the line.
* **Gaussian onset.** Not available: the lab saves only the final particles, so the per-step shape of the posterior
  is unknown.

Runs that stop at a claimed-sigma target (``stop_reason == "precision"``) end at different steps, so only lags every
detected run reaches are used (as in the simulation); for such a set the lag range is short.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from nvision.metrics import search_bounds as sb
from nvision.metrics.claims import MIN_RUNS
from nvision.metrics.detection_aligned import REPORT_LAGS, lag_grid, localize_ratio, localize_ratio_ci
from nvision.metrics.lab_search_fit import detection_step

FAR_HWHM = 3.0  # a measurement farther than this from the line teaches almost nothing about its centre
MIN_LAGS = 3  # fewer common lags leave no curve to read
STEEPEST_SLOPE = 3.0 * math.sqrt(3.0) / 8.0  # max over u of 2|u| / (1 + u^2)^2, at u = 1/sqrt3


def lorentzian_signal(drive_freq_mhz: np.ndarray, centre_mhz: float, baseline: float, contrast: float, fwhm_mhz: float):
    """The lab's dip model ``b (1 - C / (1 + ((d - centre) / gamma)^2))``; ``drive_freq_mhz``: shape (n,) -> (n,)."""
    u = (np.asarray(drive_freq_mhz, dtype=np.float64) - centre_mhz) / (0.5 * fwhm_mhz)
    return baseline * (1.0 - contrast / (1.0 + u * u))


def noise_sigma(trace: dict) -> float:
    """Std of the run's measurements around its model at the final posterior mean (shape parameters fixed)."""
    m = trace["model"]
    if m.get("shape_final_only"):
        raise ValueError(f"{trace['file']}: the shape was inferred, so the per-step model is unknown")
    y = np.asarray(trace["signal"], dtype=np.float64)
    mu = lorentzian_signal(
        np.asarray(trace["drive_freq_mhz"]), trace["mean_mhz"][-1], m["baseline"], m["contrast"], m["fwhm_mhz"]
    )
    if y.shape != mu.shape or y.size < 3:
        raise ValueError(f"{trace['file']}: {y.size} measurements, need at least 3")
    return float(np.std(y - mu, ddof=1))


def relative_information(drive_freq_mhz: np.ndarray, centre_mhz: float, fwhm_mhz: float) -> np.ndarray:
    """``I(d) / I_max`` of a measurement at ``drive_freq_mhz`` (shape (n,)) on a Lorentzian dip centred at
    ``centre_mhz``: ``(2u / (1 + u^2)^2)^2 / STEEPEST_SLOPE^2`` with ``u = (d - centre) / gamma``."""
    u = (np.asarray(drive_freq_mhz, dtype=np.float64) - centre_mhz) / (0.5 * fwhm_mhz)
    return (2.0 * u / (1.0 + u * u) ** 2) ** 2 / STEEPEST_SLOPE**2


def placement(detected: Sequence[tuple[int, dict]], fwhm_mhz: float) -> dict:
    """Where the runs measure after their own detection, about each run's own final mean (removes the drift between
    runs): the placement efficiency (mean ``I(d)/I_max``, median over runs) and the share of measurements farther
    than ``FAR_HWHM`` half widths from the line (median over runs)."""
    eff, far = [], []
    for n_d, t in detected:
        d = np.asarray(t["drive_freq_mhz"], dtype=np.float64)[n_d:]  # measurements after step n_d
        if d.size == 0:
            continue
        centre = float(t["mean_mhz"][-1])
        eff.append(float(np.mean(relative_information(d, centre, fwhm_mhz))))
        far.append(float(np.mean(np.abs(d - centre) > FAR_HWHM * 0.5 * fwhm_mhz)))
    if not eff:
        return {"unavailable": "no run measures after its detection"}
    return {
        "placement_efficiency_median": float(np.median(eff)),
        "far_share_median": float(np.median(far)),
        "n_runs": len(eff),
    }


def i_max_per_mhz2(baseline: float, contrast: float, fwhm_mhz: float, sigma: float) -> float:
    """Fisher information (MHz^-2) of one measurement at the steepest slope of the dip."""
    if not (baseline > 0 and contrast > 0 and fwhm_mhz > 0 and sigma > 0):
        raise ValueError(f"need positive baseline, contrast, fwhm, sigma; got {baseline, contrast, fwhm_mhz, sigma}")
    return (baseline * contrast * STEEPEST_SLOPE / (0.5 * fwhm_mhz)) ** 2 / sigma**2


def _same(traces: Sequence[dict], key: str) -> float:
    vals = {float(t["model"][key]) for t in traces}
    if len(vals) != 1:
        raise ValueError(f"runs of one window disagree on the model {key}: {sorted(vals)}")
    return vals.pop()


def analyse_window(traces: Sequence[dict], reference_mhz: float) -> dict:
    """Detection-aligned error, claimed sigma and the ``J_d`` curve of one window's runs (``lab_process.run_trace``).

    Returns lags, the mean squared error and mean claimed variance per lag (MHz^2), ``rho_j_d`` (``J_d`` curve over
    the MSE) and ``rho_claimed`` (the same curve over the mean claimed variance; > 1 means a claim tighter than the
    model allows), and the line numbers used.
    """
    if not traces:
        raise ValueError("no runs")
    widths = {round(t["scan_range_mhz"][1] - t["scan_range_mhz"][0], 6) for t in traces}
    if len(widths) != 1:
        raise ValueError(f"runs of one window have different scan ranges: {sorted(widths)}")
    window = widths.pop()
    b, c, fwhm = _same(traces, "baseline"), _same(traces, "contrast"), _same(traces, "fwhm_mhz")
    sigma = float(np.median([noise_sigma(t) for t in traces]))
    imax = i_max_per_mhz2(b, c, fwhm, sigma)
    snr = c / sigma
    reach = sb.probe_spacing(window, fwhm, snr)
    detected = []
    for t in traces:
        n_d = detection_step(np.asarray(t["std_mhz"], dtype=np.float64))
        if n_d is not None:
            detected.append((n_d, t))
    out: dict = {
        "window_mhz": window,
        "n_runs": len(traces),
        "n_detected": len(detected),
        "stop_reasons": sorted({str(t["stop_reason"]) for t in traces}),
        "model_fwhm_mhz": fwhm,
        "model_contrast": c,
        "noise_sigma": sigma,
        "snr": snr,
        "i_max_per_mhz2": imax,
        "reference_mhz": reference_mhz,
        "gaussian": {"unavailable": "the lab saves only the final particles: no per-step posterior shape"},
    }
    out["placement"] = placement(detected, fwhm)
    if len(detected) < MIN_RUNS:
        out["unavailable"] = f"{len(detected)} runs detected the dip (need {MIN_RUNS})"
        return out
    max_lag = min(len(t["std_mhz"]) - n_d for n_d, t in detected)
    if max_lag < MIN_LAGS:
        out["unavailable"] = (
            f"a run ends {max_lag} measurements after its own detection (stop reasons {out['stop_reasons']}): "
            f"no lag range common to all runs (need {MIN_LAGS}); use a full-budget set"
        )
        return out
    lags = lag_grid(max_lag)
    rows_err, rows_unc = [], []
    for n_d, t in detected:
        i = (n_d - 1 + lags).astype(int)  # 0-based index of step n_d + k
        rows_err.append(np.asarray(t["mean_mhz"], dtype=np.float64)[i] - reference_mhz)
        rows_unc.append(np.asarray(t["std_mhz"], dtype=np.float64)[i])
    err, unc = np.array(rows_err), np.array(rows_unc)  # shape (n_runs, n_lag), MHz
    mse, claimed = np.mean(err**2, axis=0), np.mean(unc**2, axis=0)
    n_ds = [n_d for n_d, _ in detected]
    at = {str(k): int(np.argmin(np.abs(lags - k))) for k in REPORT_LAGS if k <= lags[-1]}
    out.update(
        {
            "detection_step_median": float(np.median(n_ds)),
            "detection_step_q25": float(np.percentile(n_ds, 25)),
            "detection_step_q75": float(np.percentile(n_ds, 75)),
            "max_lag": int(max_lag),
            "lags": lags.astype(int).tolist(),
            "mse": mse.tolist(),
            "mean_claimed_var": claimed.tolist(),
            "rmse_at": {k: float(np.sqrt(mse[i])) for k, i in at.items()},
            "rms_claimed_at": {k: float(np.sqrt(claimed[i])) for k, i in at.items()},
            "claimed_rate": claimed_rate(lags, claimed, imax),
        }
    )
    # Restart pinned on the data, as the lab van Trees fit does: J = 1 / (mean claimed variance at detection).
    j_pin = 1.0 / float(claimed[0])
    pinned = np.tile(1.0 / (j_pin + lags * imax), (len(detected), 1))  # shape (n_runs, n_lag), MHz^2
    rho_pin = localize_ratio(err, pinned)
    rho_pin_c = localize_ratio(unc, pinned)
    out.update(
        {
            "j_pinned_per_mhz2": j_pin,
            "bound_pinned": pinned[0].tolist(),
            "rho_pinned_at": {k: float(rho_pin[i]) for k, i in at.items()},
            "rho_pinned_claimed_at": {k: float(rho_pin_c[i]) for k, i in at.items()},
        }
    )
    if reach is None:
        out["j_d_unavailable"] = f"no single-measurement reach at SNR {snr:.2f}: no elimination-model J_d"
        return out
    j_d = 12.0 / reach.dd_phys**2
    bound = np.tile(1.0 / (j_d + lags * imax), (len(detected), 1))  # shape (n_runs, n_lag), MHz^2
    rho = localize_ratio(err, bound)
    lo, hi = localize_ratio_ci(err, bound)
    rho_c = localize_ratio(unc, bound)
    out.update(
        {
            "dd_mhz": reach.dd_phys,
            "j_d_per_mhz2": j_d,
            "elimination_bound": bound[0].tolist(),
            "rho_elimination": rho.tolist(),
            "rho_elimination_at": {k: float(rho[i]) for k, i in at.items()},
            "rho_elimination_ci_at": {k: [float(lo[i]), float(hi[i])] for k, i in at.items()},
            "rho_claimed_at": {k: float(rho_c[i]) for k, i in at.items()},
        }
    )
    return out


def claimed_rate(lags: np.ndarray, claimed_var: np.ndarray, imax: float) -> dict:
    """Claimed information rate ``q_hat``: slope of ``1 / mean claimed variance`` against the lag, in units of
    ``I_max``, fitted over the second half of the lag range (refine, after localize). ``lags``, ``claimed_var``: shape
    (n_lag,), MHz^2. ``q_hat > 1`` means the claimed sigma fell faster than the data allow."""
    half = lags >= 0.5 * lags[-1]
    if np.count_nonzero(half) < MIN_LAGS:
        return {"unavailable": f"{np.count_nonzero(half)} lags in the second half (need {MIN_LAGS})"}
    slope, _ = np.polyfit(lags[half], 1.0 / claimed_var[half], 1)
    return {"claimed_rate": float(slope / imax), "window": [int(lags[half][0]), int(lags[-1])]}
