"""Tests of the article's quantitative claims, from per-step true-error / claimed-sigma curves.

Everything here is a *measurement that can fail the claim*: nothing is tuned toward the prediction, a fit that has too
little data says why instead of widening its window, and every comparison carries a bootstrap interval over repeats.
The claims are those of ``articles/sections/theory.tex`` (van Trees bound, elimination model, ``(n0, q)``, speedup,
sensitivity) and ``methods.tex`` (coverage). ``docs/evaluation_metrics.md`` (section "Testing the article's claims")
says which claim each function tests and how to read it; ``scratch/claims_to_metrics.md`` is the claim -> metric table.

Conventions
-----------
* A *use case* is one signal at one noise level; every function takes the curves of **one locator in one use case**.
* ``n``: shape ``(n_grid,)``, measurement counts (grid of :func:`step_grid`).
* ``err``, ``unc``: shape ``(n_repeats, n_grid)``, Hz, the true error ``|estimate - truth|`` and the claimed
  uncertainty of the primary parameter after ``n`` measurements. ``NaN`` after a run has stopped: a stopped run is
  **not** held at its final value (that would flatten ``1/MSE`` late in the run and bias ``q`` low), it leaves the
  average. A grid step is used only where at least ``ALIVE_MIN`` of the repeats are still running.
* ``imax`` (``stage_fisher_star``, Hz^-2 per measurement), ``j0`` (``stage_prior_info``, Hz^-2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

# A grid step enters an average / fit only where this share of the repeats is still running (survivor bias <= 1 - this).
ALIVE_MIN = 0.9
# Fewer points than this make a straight-line fit or a slope meaningless: the result says so instead.
MIN_FIT_POINTS = 4
# The fit window of (n0, q) starts this many detection steps in (the -1/2 tail needs n >> n_half).
FIT_START_FACTOR = 2.0
BOOTSTRAP_SAMPLES = 400
BOOTSTRAP_SEED = 20261108
GAUSS_WITHIN = {1: 0.6826894921370859, 2: 0.9544997361036416}  # share of a Gaussian within k sigma
SQRT3_OVER_PI = math.sqrt(3.0) / math.pi


# --------------------------------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------------------------------
def step_grid(max_n: int, dense: int = 10, points: int = 70) -> np.ndarray:
    """Measurement counts to evaluate at: the first ``dense`` verbatim, then log-spaced up to ``max_n``.

    Returns shape ``(n_grid,)``, integers >= 1, strictly increasing, last value ``max_n``.
    """
    if max_n < 1:
        raise ValueError(f"max_n must be >= 1, got {max_n}")
    if max_n <= dense:
        return np.arange(1, max_n + 1)
    tail = np.unique(np.round(np.geomspace(dense, max_n, points)).astype(int))
    return np.unique(np.concatenate([np.arange(1, dense), tail]))


def series_matrix(steps: list[list[float]], values: list[list[float | None]], grid: np.ndarray) -> np.ndarray:
    """Per-run stored series (downsampled: dense head, log-spaced tail) as a matrix on ``grid``.

    ``steps[r]``, ``values[r]``: the run's stored step indices (1-indexed, ascending) and values (Hz; ``None`` allowed).
    Between stored points the value is interpolated linearly in log-log (errors fall as power laws); outside
    ``[steps[0], steps[-1]]`` the entry is ``NaN`` (the run had stopped): no extrapolation, no hold.
    Returns shape ``(n_repeats, n_grid)``.
    """
    if len(steps) != len(values):
        raise ValueError("steps and values must list the same runs")
    out = np.full((len(steps), grid.size), np.nan)
    for r, (s, v) in enumerate(zip(steps, values, strict=True)):
        s_arr = np.asarray(s, dtype=float)
        v_arr = np.array([math.nan if x is None else float(x) for x in v])
        ok = np.isfinite(v_arr) & (v_arr > 0) & (s_arr > 0)
        if ok.sum() < 2:
            continue
        s_ok, v_ok = s_arr[ok], v_arr[ok]
        inside = (grid >= s_ok[0]) & (grid <= s_ok[-1])
        out[r, inside] = np.exp(np.interp(np.log(grid[inside]), np.log(s_ok), np.log(v_ok)))
    return out


def alive_fraction(a: np.ndarray) -> np.ndarray:
    """Share of repeats still running at each grid step. ``a``: ``(n_repeats, n_grid)``; returns ``(n_grid,)``."""
    return np.isfinite(a).mean(axis=0)


def rms(a: np.ndarray) -> np.ndarray:
    """Root mean square over the repeats still running, per grid step. ``a``: ``(n_repeats, n_grid)`` -> ``(n_grid,)``.

    ``NaN`` where no repeat is running. Stopped runs leave the average (see the module docstring).
    """
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean(a * a, axis=0))


def van_trees_sigma(n: np.ndarray, j0: float, imax: float) -> np.ndarray:
    """``sigma_vT(n) = (J0 + n Imax)^-1/2`` (Hz), article Eq. vantrees. ``n``: ``(n_grid,)``."""
    if not (j0 > 0 and imax > 0):
        raise ValueError(f"J0 and Imax must be positive, got J0={j0}, Imax={imax}")
    return (j0 + np.asarray(n, dtype=float) * imax) ** -0.5


def bound_ratio(n: np.ndarray, err: np.ndarray, j0: float, imax: float) -> np.ndarray:
    """``R(n) = sigma_vT(n)^2 / mean(e^2)``, shape ``(n_grid,)``. 1 = on the bound; the bound says ``R <= 1`` (to ~1 %).

    A value clearly above 1 means the run-averaged true error is *below* the bound: a defect in ``J0`` / ``Imax``
    (or a prior that does not match where the truth is drawn, see :func:`truth_prior_spread_ratio`), not a success.
    """
    return van_trees_sigma(n, j0, imax) ** 2 / rms(err) ** 2


def prior_gap(j0: float, j0_var: float) -> float:
    """Before any measurement: ``sigma_0 / sigma_vT(0) = sqrt(J0 / J0_var)``, 1.81 = pi/sqrt3 for a uniform prior.

    ``j0`` = ``stage_prior_info`` (bound), ``j0_var`` = ``stage_prior_info_var`` = ``1/Var(prior)``. By construction it
    equals ``pi/sqrt3`` for a uniform prior, so it only checks the stored columns; whether the *runs* have that gap
    depends on how the truth is drawn, which :func:`truth_prior_spread_ratio` tests.
    """
    if not (j0 > 0 and j0_var > 0):
        raise ValueError(f"J0 and Var^-1 must be positive, got {j0}, {j0_var}")
    return math.sqrt(j0 / j0_var)


def truth_prior_spread_ratio(truth_phys: np.ndarray, lo_phys: float, hi_phys: float) -> float:
    """``rms(truth - window centre) / (W / sqrt12)``: 1 when the truths are uniform over the prior window.

    ``truth_phys``: shape ``(n_repeats,)``, the true primary parameter of each repeat (Hz); ``[lo, hi]`` the prior
    window the locator uses. The claim that a uniform window's average error is ``W/sqrt12`` (so that the gap to the
    bound is ``pi/sqrt3``) assumes this is 1; a truth kept away from the edges gives < 1.
    """
    t = np.asarray(truth_phys, dtype=float)
    if t.ndim != 1 or t.size < 2 or not hi_phys > lo_phys:
        raise ValueError("need >= 2 repeats and hi > lo")
    return float(np.sqrt(np.mean((t - 0.5 * (lo_phys + hi_phys)) ** 2)) / ((hi_phys - lo_phys) / math.sqrt(12.0)))


# --------------------------------------------------------------------------------------------------------------
# Regimes (plateau / detection / -1/2 tail) and the elimination model
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Regimes:
    """Shape of the run-averaged true error; a field is ``None`` when the data cannot support it (see ``why``)."""

    plateau_ratio: float | None  # rms e at n <= n_half/2, over sigma_0 (~1 = a plateau at the prior)
    n_half: int | None  # first n where rms e <= sigma_0 / 2
    drop_factor: float | None  # rms e just before n_half over rms e at the first grid step after it
    # log-log slope of rms e vs n on the late window. -0.5 is the Cramer-Rao tail, but an offset n0 > 0 makes the local
    # slope steeper, -0.5 n/(n - n0), until n >> n0: read q and n0 (fit_strategy_numbers) for the clean statement.
    tail_slope: float | None
    tail_points: int
    why: list[str] = field(default_factory=list)


def _loglog_slope(n: np.ndarray, y: np.ndarray) -> float:
    design = np.vstack([np.log(n), np.ones(n.size)]).T
    return float(np.linalg.lstsq(design, np.log(y), rcond=None)[0][0])


def regimes(n: np.ndarray, err: np.ndarray, sigma0: float) -> Regimes:
    """Plateau level, detection step and tail slope of the run-averaged error (Fig. budget, claim ii).

    ``sigma0``: Hz, the prior's standard deviation ``J0_var^-1/2`` (``W/sqrt12`` for a uniform window).
    The tail is read where at least ``ALIVE_MIN`` of the repeats run and ``n >= FIT_START_FACTOR * n_half``; with fewer
    than ``MIN_FIT_POINTS`` such steps there is no slope (``why`` says so).
    """
    if not sigma0 > 0:
        raise ValueError(f"sigma0 must be positive, got {sigma0}")
    n = np.asarray(n, dtype=float)
    r = rms(err)
    alive = alive_fraction(err) >= ALIVE_MIN
    why: list[str] = []
    below = np.where(alive & (r <= 0.5 * sigma0))[0]
    if below.size == 0:
        why.append("run-averaged error never fell to sigma_0/2 while >= 90% of the repeats ran: no detection step")
        return Regimes(None, None, None, None, 0, why)
    i_half = int(below[0])
    n_half = int(n[i_half])
    early = alive & (n <= 0.5 * n_half)
    plateau = float(np.mean(r[early]) / sigma0) if early.sum() >= 2 else None
    if plateau is None:
        why.append("fewer than 2 grid steps before n_half/2: no plateau level")
    drop = float(r[i_half - 1] / r[i_half]) if i_half >= 1 and alive[i_half - 1] else None
    late = alive & (n >= FIT_START_FACTOR * n_half) & np.isfinite(r) & (r > 0)
    slope = None
    if late.sum() >= MIN_FIT_POINTS:
        slope = _loglog_slope(n[late], r[late])
    else:
        why.append(
            f"only {int(late.sum())} grid steps with >= 90% of repeats running after 2 n_half: no tail slope "
            "(runs stop before the tail)"
        )
    return Regimes(plateau, n_half, drop, slope, int(late.sum()), why)


def elimination_curve(n: np.ndarray, window: float, dd: float, n_d: float, imax: float) -> np.ndarray:
    """The elimination model for one detection step, Eqs. elimination / after-detection: ``sigma_theta(n)`` in Hz.

    ``n``: ``(n_grid,)``; ``window`` and ``dd`` in Hz (``W`` and the grid step), ``n_d`` the detection step.
    """
    if not (window > 0 and dd > 0 and imax > 0 and 0 < n_d <= window / dd + 1e-9):
        raise ValueError("need window, dd, imax > 0 and 0 < n_d <= W/dd")
    n = np.asarray(n, dtype=float)
    before = (window - n * dd) / math.sqrt(12.0)
    after = (12.0 / dd**2 + np.maximum(n - n_d, 0.0) * imax) ** -0.5
    return np.where(n < n_d, before, after)


def elimination_mean_curve(n: np.ndarray, window: float, dd: float, imax: float) -> np.ndarray:
    """Root of the elimination model's MSE averaged over a uniformly distributed detection step ``n_d in {1..W/dd}``.

    This is the run-averaged error the model predicts for a strategy that stops searching when it first lands in a
    uniformly placed dip: the curve to put next to a run average (a single ``n_d`` fitted to the data cannot disagree
    with it). Returns shape ``(n_grid,)``, Hz.
    """
    n_total = int(window // dd)
    if n_total < 1:
        raise ValueError("window must hold at least one grid step")
    mse = np.zeros(np.size(n), dtype=float)
    for n_d in range(1, n_total + 1):
        mse += elimination_curve(n, window, dd, float(n_d), imax) ** 2
    return np.sqrt(mse / n_total)


# --------------------------------------------------------------------------------------------------------------
# (n0, q), information rate, sensitivity, speedup, cost
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyFit:
    """Late-run fit ``1/MSE = q Imax (n - n0)`` (Eq. strategy-numbers). ``q`` and ``n0`` are ``None`` with a ``why``."""

    q: float | None
    n0: float | None
    q_ci: tuple[float, float] | None  # bootstrap 2.5 / 97.5 % over repeats
    n0_ci: tuple[float, float] | None
    n_lo: int | None  # fit window, measurements
    n_hi: int | None
    points: int
    why: str | None = None


def _wls_inverse_mse(n: np.ndarray, mse: np.ndarray) -> tuple[float, float]:
    """Weighted line ``1/mse = a + b n``, weights ``mse^2`` (the variance of ``1/mse`` goes as ``1/mse^2``)."""
    y = 1.0 / mse
    w = mse**2
    total = w.sum()
    mx, my = (w * n).sum() / total, (w * y).sum() / total
    sxx = (w * (n - mx) ** 2).sum()
    if not sxx > 0:
        raise ValueError("no spread in n")
    b = (w * (n - mx) * (y - my)).sum() / sxx
    return my - b * mx, b


def _window(n: np.ndarray, err: np.ndarray, n_min: float) -> np.ndarray:
    return (alive_fraction(err) >= ALIVE_MIN) & (np.asarray(n) >= n_min)


def fit_strategy_numbers(
    n: np.ndarray,
    err: np.ndarray,
    imax: float,
    *,
    n_min: float,
    bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> StrategyFit:
    """``(n0, q)`` of ``err`` (true error, or the claimed uncertainty for the claimed twin): claim iii.

    ``err``: ``(n_repeats, n_grid)`` Hz. The fit line is ``1/mean(err^2)`` against ``n`` over the grid steps with
    ``n >= n_min`` and >= ``ALIVE_MIN`` of the repeats running (stopped runs leave, never held). ``q = slope / imax``
    is **not** clipped to 1: ``q > 1`` is overconfidence (when ``err`` is the claimed sigma) or a defect in ``imax``.
    ``n0 = -intercept / slope``. The bootstrap resamples repeats with a fixed seed, on the same window.
    """
    if not imax > 0:
        raise ValueError(f"imax must be positive, got {imax}")
    n = np.asarray(n, dtype=float)
    sel = _window(n, err, n_min)
    if sel.sum() < MIN_FIT_POINTS:
        return StrategyFit(
            None, None, None, None, None, None, int(sel.sum()),
            f"{int(sel.sum())} grid steps from n >= {n_min:g} with >= 90% of repeats running (need {MIN_FIT_POINTS}): "
            "runs stop before the late regime",
        )  # fmt: skip
    mse = np.nanmean(err[:, sel] ** 2, axis=0)
    if not np.all(mse > 0):
        raise ValueError("zero mean-square error in the fit window")
    a, b = _wls_inverse_mse(n[sel], mse)
    if not b > 0:
        return StrategyFit(None, None, None, None, int(n[sel][0]), int(n[sel][-1]), int(sel.sum()),
                           "1/MSE does not grow with n in the window (no information gain)")  # fmt: skip
    q, n0 = b / imax, -a / b
    q_ci = n0_ci = None
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        qs, n0s = [], []
        for _ in range(bootstrap):
            pick = rng.integers(0, err.shape[0], err.shape[0])
            sub = err[pick][:, sel]
            if not np.all(np.isfinite(sub).any(axis=0)):
                continue
            m = np.nanmean(sub**2, axis=0)
            if not np.all(m > 0):
                continue
            ab, bb = _wls_inverse_mse(n[sel], m)
            if bb > 0:
                qs.append(bb / imax)
                n0s.append(-ab / bb)
        if len(qs) >= 20:
            q_ci = (float(np.percentile(qs, 2.5)), float(np.percentile(qs, 97.5)))
            n0_ci = (float(np.percentile(n0s, 2.5)), float(np.percentile(n0s, 97.5)))
    return StrategyFit(q, n0, q_ci, n0_ci, int(n[sel][0]), int(n[sel][-1]), int(sel.sum()))


def information_rate_curve(n: np.ndarray, err: np.ndarray, imax: float) -> np.ndarray:
    """Run-averaged information rate in units of ``imax``: ``(J(n) - J(n')) / ((n - n') imax)``, ``J = 1/mean(err^2)``.

    Between consecutive grid steps both of which have >= ``ALIVE_MIN`` of the repeats running; ``NaN`` elsewhere.
    Shape ``(n_grid,)`` (entry 0 is ``NaN``). With ``err`` the claimed uncertainty, a value above 1 means the claimed
    sigma fell faster than any measurements could justify (overconfidence); with the true error it is the same
    quantity for what was actually achieved, and exceeds 1 only by noise or a defect in ``imax``. Local differences
    are noisy: read the window fit (:func:`fit_strategy_numbers`) for a number.
    """
    n = np.asarray(n, dtype=float)
    ok = alive_fraction(err) >= ALIVE_MIN
    with np.errstate(all="ignore"):
        j = 1.0 / np.nanmean(err**2, axis=0)
    out = np.full(n.size, np.nan)
    both = ok[1:] & ok[:-1]
    out[1:][both] = (j[1:][both] - j[:-1][both]) / ((n[1:][both] - n[:-1][both]) * imax)
    return out


def sensitivity_ratio(n: np.ndarray, err: np.ndarray, imax: float, *, n_min: float) -> float | None:
    """``eta_measured / eta_article = sqrt(imax * MSE(n) * n)`` averaged (median) over the late window; claim ix.

    ``eta = sigma_B sqrt(T)`` of Eq. sensitivity is ``sqrt(t_m / Imax) / gamma_e``; ``t_m``, ``gamma_e`` and ``SNR``
    cancel against the bound's own ``Imax``, so the dimensionless ratio is what the data can say (multiply the article's
    ``eta`` by it for the measured one; ``t_s`` is not in the data). It tends to ``1/sqrt(q)`` times
    ``sqrt(n/(n - n0))``, so it is a consistency check of :func:`fit_strategy_numbers` with the offset ``n0`` left in,
    **not** an independent confirmation. ``None`` when fewer than ``MIN_FIT_POINTS`` late steps exist.
    """
    n = np.asarray(n, dtype=float)
    sel = _window(n, err, n_min)
    if sel.sum() < MIN_FIT_POINTS:
        return None
    mse = np.nanmean(err[:, sel] ** 2, axis=0)
    return float(np.median(np.sqrt(imax * mse * n[sel])))


def speedup_prediction(window_phys: float, fwhm_phys: float) -> float:
    """``R_inf = 27 W / (8 pi Gamma)`` (Eq. speedup). Approximate for ``W`` not much larger than ``Gamma``."""
    if not (window_phys > 0 and fwhm_phys > 0):
        raise ValueError("window and fwhm must be positive")
    return 27.0 * window_phys / (8.0 * math.pi * fwhm_phys)


def predicted_cost(n0: float, q: float, imax: float, sigma_target_phys: float) -> float:
    """Measurements to a precise target ``n0 + sigma_tgt^-2 / (q imax)`` (Eq. strategy-numbers)."""
    if not (q > 0 and imax > 0 and sigma_target_phys > 0):
        raise ValueError("q, imax and the target must be positive")
    return n0 + sigma_target_phys**-2 / (q * imax)


def passage_steps(n: np.ndarray, err: np.ndarray, sigma_target_phys: float) -> np.ndarray:
    """First grid step from which each repeat's true error stays <= target to its last step; ``NaN`` if it ends above.

    ``err``: ``(n_repeats, n_grid)`` with NaN after the run stops (the run's last finite entry is its end).
    Returns shape ``(n_repeats,)``.
    """
    out = np.full(err.shape[0], np.nan)
    for r in range(err.shape[0]):
        fin = np.where(np.isfinite(err[r]))[0]
        if fin.size == 0:
            continue
        e = err[r, fin]
        above = np.where(e > sigma_target_phys)[0]
        if above.size == 0:
            out[r] = n[fin[0]]
        elif above[-1] < e.size - 1:
            out[r] = n[fin[above[-1] + 1]]
    return out


# --------------------------------------------------------------------------------------------------------------
# Coverage and calibration
# --------------------------------------------------------------------------------------------------------------
def coverage_curve(err: np.ndarray, unc: np.ndarray, gate: np.ndarray, n: np.ndarray) -> pl.DataFrame:
    """Share of runs with ``e <= k u`` (k = 1, 2) at each grid step; claim vi-a.

    ``err``, ``unc``: ``(n_repeats, n_grid)``; ``gate``: ``(n_repeats,)`` the step from which a run's posterior is
    taken to be Gaussian (``refine_start_step``; ``NaN`` = never reached, excluded). A run counts at step ``n`` only
    when ``n >= gate`` and it is still running. Columns: ``n``, ``runs`` (counted), ``within_1``, ``within_2``
    (``NaN`` where fewer than 5 runs), ``gauss_1``, ``gauss_2`` (0.683, 0.954), plus ``ungated_within_1``/``_2`` over
    every running run so the effect of the gate is visible. Below the Gaussian share = overconfident.
    """
    n = np.asarray(n, dtype=float)
    gate = np.asarray(gate, dtype=float)
    if err.shape != unc.shape or gate.shape != (err.shape[0],):
        raise ValueError("err, unc and gate shapes disagree")
    live = np.isfinite(err) & np.isfinite(unc) & (unc > 0)
    gated = live & (n[None, :] >= gate[:, None])  # NaN gate: comparison False
    rows: dict[str, list[float]] = {
        k: [] for k in ("runs", "within_1", "within_2", "ungated_within_1", "ungated_within_2")
    }
    for j in range(n.size):
        for tag, mask in (("", gated[:, j]), ("ungated_", live[:, j])):
            m = int(mask.sum())
            if tag == "":
                rows["runs"].append(m)
            for k in (1, 2):
                share = float((err[mask, j] <= k * unc[mask, j]).mean()) if m >= 5 else math.nan
                rows[f"{tag}within_{k}"].append(share)
    return pl.DataFrame({"n": n, **rows, "gauss_1": GAUSS_WITHIN[1], "gauss_2": GAUSS_WITHIN[2]})


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a share ``hits / total`` (95 % for the default ``z``)."""
    if total < 1 or not 0 <= hits <= total:
        raise ValueError(f"need 0 <= hits <= total and total >= 1, got {hits}/{total}")
    p = hits / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return centre - half, centre + half


def pit_coverage(pit: np.ndarray) -> dict[str, float | tuple[float, float]]:
    """Calibration of the particle posterior from the PIT value (posterior CDF at the truth); claim vi-b.

    ``pit``: shape ``(n_runs,)`` in ``[0, 1]``. A calibrated posterior gives uniform PIT: the central 68 % interval
    holds the truth in 68 % of the runs, the central 95 % interval in 95 %. Returns the shares with Wilson intervals,
    and ``below`` / ``above`` the shares with the truth in the lower / upper 2.5 % tail (bias vs. spread).
    """
    p = np.asarray(pit, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("pit must be a non-empty 1-D array in [0, 1]")
    n = p.size
    c68 = int(np.sum((p >= 0.1587) & (p <= 0.8413)))
    c95 = int(np.sum((p >= 0.025) & (p <= 0.975)))
    return {
        "runs": float(n),
        "central_68": c68 / n,
        "central_68_ci": wilson_interval(c68, n),
        "central_95": c95 / n,
        "central_95_ci": wilson_interval(c95, n),
        "below": float(np.mean(p < 0.025)),
        "above": float(np.mean(p > 0.975)),
    }


def far_placement(
    share: np.ndarray, mass: np.ndarray, seed: int = BOOTSTRAP_SEED
) -> dict[str, float | tuple[float, float]]:
    """Claim vii: share of localize measurements far from the peak against the residual posterior mass there.

    ``share``, ``mass``: shape ``(n_runs,)``, per run ``stage2_far_share`` and ``stage2_far_mass`` (runs with a localize
    stage). The claim is that the variance-EIG spends measurements on residual mass far from the dip, i.e. that
    ``share`` exceeds ``mass``; the paired mean difference with a bootstrap interval over runs says whether it does
    (interval above 0), whether it matches (spans 0) or whether measurements avoid the mass (below 0). The falsification
    probe (a deliberate far measurement) cannot be told apart in the snapshots, so ``share`` is an upper bound for EIG.
    """
    s, m = np.asarray(share, dtype=float), np.asarray(mass, dtype=float)
    if s.shape != m.shape or s.ndim != 1 or s.size < 2:
        raise ValueError("share and mass must be 1-D and the same length >= 2")
    d = s - m
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(d[rng.integers(0, d.size, d.size)])) for _ in range(BOOTSTRAP_SAMPLES)]
    return {
        "runs": float(s.size),
        "far_share": float(np.mean(s)),
        "far_mass": float(np.mean(m)),
        "diff": float(np.mean(d)),
        "diff_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
    }
