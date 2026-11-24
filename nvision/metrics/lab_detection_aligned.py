"""The lab source of the shared run record: live Bayesian ESR runs, read by the same metrics as the simulation.

A lab run becomes a :class:`~nvision.metrics.run_record.RunRecord` (:func:`record_from_trace`) and goes through the
functions of :mod:`nvision.metrics.detection_aligned`. Only the explicit differences of the record are lab-specific:

* **Truth** (``LAB_REFERENCE``). The error of a run is its posterior mean minus the lab study's reference (the median
  final mean of its converged runs, :func:`nvision.metrics.lab_study.lab_study_payload`). The line drifts 0.4-0.6 MHz
  between recordings, so quantities averaged over the prior ensemble are not valid: only claimed quantities are
  reported (the claimed rate q-hat, the placement, the claimed variance against the restarted curves, the detection
  step). The scatter of repeated runs is the lab study's calibration (:mod:`nvision.metrics.lab_study`).
* **Detection** (``SIGMA_DROP``). The step right after the run's first steep, lasting drop of its claimed sigma
  (:func:`nvision.metrics.lab_search_fit.detection_step`); lag 0 is the first posterior after the drop.
* **Known nuisances.** ``I_max = (b C 3 sqrt3 / (8 gamma))^2 / sigma^2`` is the information of one measurement at the
  steepest slope of the Lorentzian dip the run's model assumes (baseline ``b``, contrast ``C``, half width ``gamma``,
  held fixed by the lab), with ``sigma`` the noise of the run's own measurements around that model.
* **Stop.** Runs that stop at a claimed-sigma target end at different steps, so only lags every detected run reaches
  are used (as in the simulation); for such a set the lag range is short.

Restarted curves: there is no posterior Fisher information ``J_tau`` (no per-step particles), so the claimed variance
is compared with ``J_d = 12 / Delta d^2`` of the elimination model (Delta d from eq:sm-reach for the lab line; at the
lab SNR of about 4-5 it often does not exist) and with ``J_pinned = 1 / <sigma_claimed^2>`` at lag 0, a description
of the data from detection on, not a bound. The Gaussian onset is not available: the lab saves only the final
particles.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from nvision.metrics import search_bounds as sb
from nvision.metrics.claims import MIN_RUNS
from nvision.metrics.detection_aligned import (
    MIN_LAGS,
    REPORT_LAGS,
    AlignedRun,
    align,
    aligned,
    claimed_rate,
    lag_grid,
    localize_ratio,
)
from nvision.metrics.run_record import LAB_REFERENCE, SIGMA_DROP, RunRecord

FAR_HWHM = 3.0  # a measurement farther than this from the line teaches almost nothing about its centre
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


def placement(detected: Sequence[tuple[int, RunRecord]], fwhm_mhz: float) -> dict:
    """Where the runs measure after their own detection, about each run's own final estimate (removes the drift
    between runs): the placement efficiency (mean ``I(d)/I_max``, median over runs) and the share of measurements
    farther than ``FAR_HWHM`` half widths from the line (median over runs). A claimed quantity: no truth needed.

    ``detected``: ``(detection step, record)`` pairs; every record needs ``drive_freq`` (n,) and ``estimate``.
    """
    eff, far = [], []
    for n_d, r in detected:
        if r.drive_freq is None or r.estimate is None:
            raise ValueError("placement needs every measurement's drive_freq and the run's estimate")
        d = r.drive_freq[n_d:]  # measurements after step n_d
        if d.size == 0:
            continue
        centre = float(r.estimate[-1])
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


def record_from_trace(trace: dict, reference_mhz: float, *, i_max: float, window: float, snr: float) -> RunRecord:
    """One lab run (``lab_process.run_trace``) as the shared :class:`~nvision.metrics.run_record.RunRecord`.

    Truth: the lab reference (``LAB_REFERENCE``). Every measurement is kept, so ``steps`` is ``1..n``, in MHz; the
    line model (baseline, contrast, FWHM) is the run's fixed one (known nuisances).
    """
    mean = np.asarray(trace["mean_mhz"], dtype=np.float64)
    return RunRecord(
        source="lab",
        truth=LAB_REFERENCE,
        steps=np.arange(1, mean.size + 1, dtype=np.float64),
        estimate=mean,
        claimed_sigma=np.asarray(trace["std_mhz"], dtype=np.float64),
        error=np.abs(mean - reference_mhz),
        i_max=i_max,
        window=window,
        fwhm=float(trace["model"]["fwhm_mhz"]),
        snr=snr,
        budget=int(mean.size),
        stop_rule="claimed-sigma target" if str(trace["stop_reason"]) == "precision" else "full-budget",
        known_nuisances=True,
        drive_freq=np.asarray(trace["drive_freq_mhz"], dtype=np.float64),
        outcomes=np.asarray(trace["signal"], dtype=np.float64),
        line_params=dict(trace["model"]),
    )


def analyse_window(traces: Sequence[dict], reference_mhz: float) -> dict:
    """The claimed quantities of one window's lab runs (``lab_process.run_trace``), aligned at their own detection.

    Returns the lags, the mean claimed variance per lag (MHz^2), the claimed rate q-hat, the claimed variance against
    the restarted curves (``rho_pinned_claimed_at``; ``rho_claimed_at`` for ``J_d``; > 1 means a claim tighter than the
    model allows), the placement and the line numbers used. No error-based quantity: ``LAB_REFERENCE`` is not
    ensemble-valid.
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
    records = [record_from_trace(t, reference_mhz, i_max=imax, window=window, snr=snr) for t in traces]
    pairs: list[tuple[AlignedRun, RunRecord]] = [(a, r) for r in records if (a := align(r, SIGMA_DROP)) is not None]
    runs = [a for a, _ in pairs]
    out: dict = {
        "window_mhz": window,
        "n_runs": len(traces),
        "n_detected": len(runs),
        "detector": SIGMA_DROP.name,
        "truth": LAB_REFERENCE.name,
        "stop_reasons": sorted({str(t["stop_reason"]) for t in traces}),
        "model_fwhm_mhz": fwhm,
        "model_contrast": c,
        "noise_sigma": sigma,
        "snr": snr,
        "i_max_per_mhz2": imax,
        "reference_mhz": reference_mhz,
        "gaussian": {"unavailable": "the lab saves only the final particles: no per-step posterior shape"},
    }
    out["placement"] = placement([(a.detection_step, r) for a, r in pairs], fwhm)
    if len(runs) < MIN_RUNS:
        out["unavailable"] = f"{len(runs)} runs detected the dip (need {MIN_RUNS})"
        return out
    max_lag = int(min(r.steps[-1] - a.restart for a, r in pairs))
    if max_lag < MIN_LAGS:
        out["unavailable"] = (
            f"a run ends {max_lag} measurements after its own detection (stop reasons {out['stop_reasons']}): "
            f"no lag range common to all runs (need {MIN_LAGS}); use a full-budget set"
        )
        return out
    lags = lag_grid(max_lag)
    unc = aligned(runs, lags, "claimed")  # shape (n_runs, n_lag), MHz
    claimed = np.mean(unc**2, axis=0)
    n_ds = [a.detection_step for a in runs]
    at = {str(k): int(np.argmin(np.abs(lags - k))) for k in REPORT_LAGS if k <= lags[-1]}
    out.update(
        {
            "detection_step_median": float(np.median(n_ds)),
            "detection_step_q25": float(np.percentile(n_ds, 25)),
            "detection_step_q75": float(np.percentile(n_ds, 75)),
            "max_lag": max_lag,
            "lags": lags.astype(int).tolist(),
            "mean_claimed_var": claimed.tolist(),
            "rms_claimed_at": {k: float(np.sqrt(claimed[i])) for k, i in at.items()},
            "claimed_rate": claimed_rate(lags, claimed, imax),
        }
    )
    # Restart pinned on the data, as the lab van Trees fit does: J = 1 / (mean claimed variance at detection).
    j_pin = 1.0 / float(claimed[0])
    pinned = np.tile(1.0 / (j_pin + lags * imax), (len(runs), 1))  # shape (n_runs, n_lag), MHz^2
    rho_pin_c = localize_ratio(unc, pinned)
    out.update(
        {
            "j_pinned_per_mhz2": j_pin,
            "bound_pinned": pinned[0].tolist(),
            "rho_pinned_claimed_at": {k: float(rho_pin_c[i]) for k, i in at.items()},
        }
    )
    if reach is None:
        out["j_d_unavailable"] = f"no single-measurement reach at SNR {snr:.2f}: no elimination-model J_d"
        return out
    j_d = 12.0 / reach.dd_phys**2
    bound = np.tile(1.0 / (j_d + lags * imax), (len(runs), 1))  # shape (n_runs, n_lag), MHz^2
    rho_c = localize_ratio(unc, bound)
    out.update(
        {
            "dd_mhz": reach.dd_phys,
            "j_d_per_mhz2": j_d,
            "elimination_bound": bound[0].tolist(),
            "rho_claimed_at": {k: float(rho_c[i]) for k, i in at.items()},
        }
    )
    return out
