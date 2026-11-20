"""Runs aligned at their own dip detection: the localize ratio against the restarted van Trees bound, and the onset of
a Gaussian posterior.

The article reads localize and refine from each run's own detection, never at a fixed lag after the start: the bound
that holds there is van Trees restarted at detection, with the posterior at detection as the prior
(``articles/sections/methods.tex``, "Localize and refine"). This module puts every run on the lag axis
``k = n - n_d`` (measurements since its own detection) and averages over runs on that axis.

* **Detection step.** ``dip_detected_step`` = tau is the 1-based measurement at which the dip test first fires, and
  ``posterior_J_tau_phys`` is the Fisher information of the posterior snapshot ``tau - 1`` (the belief after
  ``tau - 1`` measurements; :mod:`nvision.metrics.posterior`). So the restart point is ``n_d = tau - 1`` and lag ``k``
  is the error after ``n_d + k`` measurements: the bound gets the ``k`` measurements taken after that posterior.
* **Localize ratio.** ``rho(k) = <B_i(k)>_runs / <e_i(n_d,i + k)^2>_runs``, with ``B_i(k) = 1 / (J_i + k I_max)``.
  With ``J_i = J_tau`` of each run, ``rho <= 1`` is the bound (it bounds the mean over runs, not one run). With
  ``J_i = J_d = 12 / Delta d^2`` (the detection cell of the elimination model) it is the model of Eq. after-detection,
  a comparison, not a test: the posterior's ``J_tau`` overstates what a detection pins down until the side of the dip
  is known.
* **Common lag range.** Only lags every included run reaches inside the budget are used
  (``k <= budget - max_i n_d,i``), so the same runs enter every lag; dropping late detectors at long lags would bias
  the mean toward the easy runs.
* **Gaussian onset.** The stored negentropy ``g`` of the posterior (nats; 0 for a Gaussian, about 0.18 for a uniform,
  more for two peaks) has a floor of about 0.01 from the density estimate itself. The onset is the first stored step
  from which ``g <= GAUSSIAN_NEGENTROPY_MAX`` at every later stored step.

Conventions: ``steps`` are a run's stored measurement counts (ascending, 1-based), ``err`` its absolute true error of
the primary parameter in Hz at those steps, ``negentropy`` the stored ``g`` at the same steps (``nan`` where the
posterior was undefined). Values between stored steps are interpolated linearly; the last stored value is held to the
budget (every run measures its whole budget).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.metrics.claims import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, MIN_RUNS

# Negentropy (nats) below which the posterior counts as a single Gaussian peak: well above the ~0.01 floor of the
# density estimate on a Gaussian cloud, well below the 0.18 of a uniform posterior and the 0.3-0.5 of two peaks.
GAUSSIAN_NEGENTROPY_MAX = 0.05
REPORT_LAGS = (0, 1, 3, 10, 30, 100, 300, 1000)
DENSE_LAGS = 10  # lags 0..DENSE_LAGS-1 are kept verbatim, then log-spaced
LAG_POINTS = 70


@dataclass(frozen=True)
class AlignedRun:
    """One run's stored series and its detection. ``steps``, ``err``, ``negentropy``: shape (m,)."""

    steps: np.ndarray  # measurement counts, ascending, 1-based
    err: np.ndarray  # |estimate - truth| of the primary parameter, Hz
    negentropy: np.ndarray  # posterior negentropy, nats; nan where undefined
    detected_step: int  # tau, 1-based measurement at which the dip test first fired
    j_tau: float | None  # Fisher information of the posterior at detection, Hz^-2

    def __post_init__(self) -> None:
        s, e, g = self.steps, self.err, self.negentropy
        if s.ndim != 1 or s.size == 0 or e.shape != s.shape or g.shape != s.shape:
            raise ValueError(
                f"steps, err, negentropy must be equal-length 1-D arrays, got {s.shape}, {e.shape}, {g.shape}"
            )
        if not np.all(np.diff(s) > 0):
            raise ValueError("steps must be strictly ascending")
        if not np.all(np.isfinite(e)) or np.any(e < 0):
            raise ValueError("err must be finite and >= 0 (absolute errors)")
        if self.detected_step < 1:
            raise ValueError(f"detected_step is 1-based, got {self.detected_step}")
        if self.j_tau is not None and not (math.isfinite(self.j_tau) and self.j_tau >= 0):
            raise ValueError(f"j_tau must be finite and >= 0, got {self.j_tau}")

    @property
    def restart(self) -> int:
        """``n_d = tau - 1``: the measurements behind the posterior whose information is ``j_tau``."""
        return self.detected_step - 1


def lag_grid(max_lag: int) -> np.ndarray:
    """Lags to evaluate at: ``0..DENSE_LAGS-1`` verbatim, then log-spaced to ``max_lag``; shape (n_lag,)."""
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    dense = np.arange(0, min(DENSE_LAGS, max_lag + 1), dtype=np.float64)
    if max_lag < DENSE_LAGS:
        return dense
    tail = np.unique(np.round(np.geomspace(DENSE_LAGS, max_lag, LAG_POINTS)))
    return np.unique(np.concatenate([dense, tail]))


def _at(steps: np.ndarray, vals: np.ndarray, n: np.ndarray) -> np.ndarray:
    """``vals`` at measurement counts ``n`` (linear interpolation, last value held; ``nan`` before the first valid)."""
    ok = np.isfinite(vals)
    if not np.any(ok):
        return np.full(n.shape, np.nan)
    s, v = steps[ok], vals[ok]
    out = np.interp(n, s, v)
    out[n < s[0]] = np.nan
    return out


def aligned(runs: Sequence[AlignedRun], lags: np.ndarray, field: str) -> np.ndarray:
    """``field`` (``"err"`` or ``"negentropy"``) of every run at ``n_d + lag``; shape (n_runs, n_lag)."""
    if field not in ("err", "negentropy"):
        raise ValueError(f"field must be 'err' or 'negentropy', got {field!r}")
    return np.array([_at(r.steps, getattr(r, field), r.restart + lags) for r in runs])


def common_max_lag(runs: Sequence[AlignedRun], budget: int) -> int:
    """Largest lag every run reaches inside the budget: ``budget - max n_d``."""
    if not runs:
        raise ValueError("no runs")
    worst = max(r.restart for r in runs)
    if worst >= budget:
        raise ValueError(f"a run detected at measurement {worst + 1}, past the budget {budget}")
    return budget - worst


def restarted_bound(j: np.ndarray, lags: np.ndarray, i_max: float) -> np.ndarray:
    """``1 / (J_i + k I_max)`` for every run and lag. ``j``: shape (n_runs,), Hz^-2 -> shape (n_runs, n_lag), Hz^2."""
    if not i_max > 0:
        raise ValueError(f"i_max must be positive, got {i_max}")
    if np.any(~np.isfinite(j)) or np.any(j < 0):
        raise ValueError("J must be finite and >= 0 for every run")
    denom = j[:, None] + lags[None, :] * i_max
    if np.any(denom <= 0):
        raise ValueError("J + k I_max must be positive (J = 0 at lag 0 has no bound)")
    return 1.0 / denom


def localize_ratio(err: np.ndarray, bound: np.ndarray) -> np.ndarray:
    """``rho(k) = mean bound / mean err^2`` over runs; ``err``, ``bound``: shape (n_runs, n_lag) -> (n_lag,)."""
    if err.shape != bound.shape:
        raise ValueError(f"err {err.shape} and bound {bound.shape} differ")
    return bound.mean(axis=0) / np.mean(err**2, axis=0)


def localize_ratio_ci(
    err: np.ndarray, bound: np.ndarray, *, bootstrap: int = BOOTSTRAP_SAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap (2.5 %, 97.5 %) of :func:`localize_ratio` over the runs, each shape (n_lag,)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = err.shape[0]
    draws = []
    for _ in range(bootstrap):
        pick = rng.integers(0, n, n)
        draws.append(localize_ratio(err[pick], bound[pick]))
    d = np.array(draws)
    return np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


def gaussian_onset(steps: np.ndarray, negentropy: np.ndarray, threshold: float = GAUSSIAN_NEGENTROPY_MAX) -> int | None:
    """First stored step from which the negentropy stays ``<= threshold`` at every later stored step.

    ``None`` when the last defined value is above the threshold (never Gaussian within the budget). Undefined values
    (``nan``) are skipped, so an undefined stretch neither starts nor breaks the Gaussian tail.
    """
    ok = np.isfinite(negentropy)
    if not np.any(ok):
        return None
    s, g = steps[ok], negentropy[ok]
    above = np.flatnonzero(g > threshold)
    if above.size == 0:
        return int(s[0])
    last = int(above[-1])
    return None if last == g.size - 1 else int(s[last + 1])


def _quantiles(mat: np.ndarray) -> dict[str, list[float | None]]:
    """Median and quartiles over runs (columns with no finite value give ``None``); ``mat``: (n_runs, n_lag)."""
    out: dict[str, list[float | None]] = {}
    for name, q in (("q25", 25), ("median", 50), ("q75", 75)):
        col = []
        for j in range(mat.shape[1]):
            v = mat[:, j][np.isfinite(mat[:, j])]
            col.append(float(np.percentile(v, q)) if v.size else None)
        out[name] = col
    return out


def _at_lags(lags: np.ndarray, values: np.ndarray, wanted: Sequence[int]) -> dict[str, float]:
    return {str(k): float(values[int(np.argmin(np.abs(lags - k)))]) for k in wanted if k <= lags[-1]}


def _med(v: Sequence[float]) -> float | None:
    return float(np.median(v)) if len(v) else None


def analyse(runs: Sequence[AlignedRun], *, budget: int, i_max: float, dd_phys: float | None, n_undetected: int) -> dict:
    """Localize ratio and Gaussian onset of one locator in one use case.

    ``runs``: the runs that detected the dip. ``dd_phys``: the detection cell ``Delta d`` (Hz) of the elimination model
    for ``J_d = 12 / Delta d^2``, or ``None`` when the SNR gives no single-measurement reach (then no ``J_d`` curve).
    """
    if len(runs) < MIN_RUNS:
        return {"unavailable": f"{len(runs)} runs detected the dip (need {MIN_RUNS})", "n_runs": len(runs)}
    max_lag = common_max_lag(runs, budget)
    lags = lag_grid(max_lag)
    err = aligned(runs, lags, "err")
    if not np.all(np.isfinite(err)):
        raise ValueError("a run's error series does not cover its own detection: the stored series start too late")
    mse = np.mean(err**2, axis=0)
    out: dict = {
        "n_runs": len(runs),
        "n_undetected": n_undetected,
        "detected_step_median": _med([r.detected_step for r in runs]),
        "detected_step_max": max(r.detected_step for r in runs),
        "max_lag": max_lag,
        "lags": lags.astype(int).tolist(),
        "mse": mse.tolist(),
    }
    with_j = [r for r in runs if r.j_tau is not None and r.j_tau > 0]
    out["n_runs_j_tau"] = len(with_j)
    if len(with_j) >= MIN_RUNS:
        keep = [i for i, r in enumerate(runs) if r.j_tau is not None and r.j_tau > 0]
        e_j = err[keep]
        b_tau = restarted_bound(np.array([r.j_tau for r in with_j]), lags, i_max)
        rho = localize_ratio(e_j, b_tau)
        lo, hi = localize_ratio_ci(e_j, b_tau)
        out["j_tau_median"] = _med([r.j_tau for r in with_j])
        out["bound_j_tau"] = b_tau.mean(axis=0).tolist()
        out["rho_j_tau"] = rho.tolist()
        out["rho_j_tau_at"] = _at_lags(lags, rho, REPORT_LAGS)
        out["rho_j_tau_ci_at"] = {
            k: [_at_lags(lags, lo, [int(k)])[k], _at_lags(lags, hi, [int(k)])[k]] for k in out["rho_j_tau_at"]
        }
        out["rho_j_tau_max"] = float(np.max(rho))
        out["bound_violated"] = bool(np.any(lo > 1.0 + 0.01))
    if dd_phys is not None:
        j_d = 12.0 / dd_phys**2
        b_d = restarted_bound(np.full(len(runs), j_d), lags, i_max)
        rho_d = localize_ratio(err, b_d)
        out["j_d"] = j_d
        out["dd_phys"] = dd_phys
        out["bound_j_d"] = b_d[0].tolist()
        out["rho_j_d"] = rho_d.tolist()
        out["rho_j_d_at"] = _at_lags(lags, rho_d, REPORT_LAGS)
    g = aligned(runs, lags, "negentropy")
    onsets = [gaussian_onset(r.steps, r.negentropy) for r in runs]
    reached = [(o, r) for o, r in zip(onsets, runs, strict=True) if o is not None]
    onset_lags = [max(0, o - r.restart) for o, r in reached]
    out["gaussian"] = {
        "threshold": GAUSSIAN_NEGENTROPY_MAX,
        "share_reached": len(reached) / len(runs),
        "onset_step_median": _med([o for o, _ in reached]),
        "onset_lag_median": _med(onset_lags),
        "onset_lag_q25": float(np.percentile(onset_lags, 25)) if onset_lags else None,
        "onset_lag_q75": float(np.percentile(onset_lags, 75)) if onset_lags else None,
        "negentropy_by_lag": _quantiles(g),
    }
    return out
