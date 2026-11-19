"""``/api/search-bounds``: the search-phase bounds against the measured search, per use case and locator.

Reads only what is already in the cache (per-repeat ``series``, ``true_params`` and stage columns), so it works with
``nv serve`` on an existing cache and needs no re-run. The formulas live in :mod:`nvision.metrics.search_bounds`; this
module rebuilds the true line of a use case, checks the stored columns against it and assembles the payload.

A *use case* is one generator at one noise level (rule 7 of ``docs/evaluation_metrics.md``). It is supported when
it is a global search of ``zeeman_shift`` whose line shape is the same in every repeat (the SNR x window study:
fixed width, contrast, ``sigma_inhom``); otherwise its entry carries an ``error`` string instead of numbers (fail
fast, no fallback).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from nvision.metrics import search_bounds as sb
from nvision.sim.combinations import PLAIN_SWEEP_STRATEGIES
from nvision.sim.presets import line_contrast

# The locators the search claim is about (never the Smart-* variants).
SEARCH_BOUND_LOCATORS = ("Bayesian-SBED", *PLAIN_SWEEP_STRATEGIES)
PRIMARY_PARAM = "zeeman_shift"
# Points of the measurement-count grid the curves are drawn on (log spaced, 1 .. budget).
N_GRID_POINTS = 160
# theta values (quantile midpoints of the drawn range) over which the line integral is averaged.
THETA_SAMPLES = 33
_MIN_X_POINTS = 4001
_X_POINTS_PER_FWHM = 200.0
# Relative tolerances of the cross-checks between stored columns and the rebuilt line.
_WIDTH_REL_TOL = 1e-6
_SNR_REL_TOL = 1e-3
_SHAPE_REL_TOL = 1e-9
_FINAL_ERR_REL_TOL = 2e-3

LoadMeta = Callable[[dict], list[tuple[list[dict] | None, dict]]]


class UseCaseError(ValueError):
    """The use case cannot be judged against the search bounds (reason in the message)."""


def _num(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _num_list(values: Iterable[float]) -> list[float | None]:
    return [_num(float(v)) for v in values]


def n_grid(budget: int) -> np.ndarray:
    """Log-spaced integer measurement counts from 1 to ``budget``; shape (<= N_GRID_POINTS,)."""
    return np.unique(np.round(np.geomspace(1, budget, N_GRID_POINTS)).astype(int)).astype(float)


class _Run:
    """One repeat: the fields the search metrics read, from the scan entry and the results row."""

    __slots__ = (
        "detected",
        "errs",
        "fisher_star",
        "fwhm",
        "noise_sigma",
        "params",
        "prior_info",
        "snr",
        "steps",
        "theta",
        "width",
    )

    def __init__(self, entries: list[dict] | None, row: dict) -> None:
        scan = next((e for e in (entries or []) if e.get("type") == "scan"), None)
        if scan is None:
            raise UseCaseError("a repeat has no scan entry")
        metrics = scan.get("metrics") or {}

        def field(name: str) -> Any:
            return row[name] if row.get(name) is not None else metrics.get(name)

        if field("stage_primary_param") != PRIMARY_PARAM:
            raise UseCaseError(f"primary parameter is {field('stage_primary_param')!r}, not {PRIMARY_PARAM!r}")
        series = scan.get("series") or {}
        if not series.get("s"):
            raise UseCaseError("a repeat has no per-step series")
        keep = [i for i, e in enumerate(series["e"]) if e is not None]
        if not keep:
            raise UseCaseError("a repeat has no true-error series")
        self.steps = np.array([series["s"][i] for i in keep], dtype=float)
        self.errs = np.array([series["e"][i] for i in keep], dtype=float)
        # The series parameter is chosen by the series exporter, not by stage_primary_param: confirm it is the
        # zeeman_shift error through its last point (the series is rounded to 4 significant digits). That point is
        # the belief's error after the last measurement (the stage metrics' error of the last reached stage) for a
        # belief locator, and the fit's error ``final_err_primary`` for a sweep, whose series ends on its fit
        # estimate (nvision.metrics.series.extract_step_series; series cached before that change end on the belief).
        # Both are errors of the primary parameter, so either identifies the series.
        last_err = next((field(f"stage{k}_err_end") for k in (3, 2, 1) if field(f"stage{k}_err_end") is not None), None)
        candidates = [float(v) for v in (last_err, field("final_err_primary")) if v is not None]
        if not any(math.isclose(float(self.errs[-1]), v, rel_tol=_FINAL_ERR_REL_TOL) for v in candidates):
            raise UseCaseError(
                f"the last point of the true-error series ({self.errs[-1]:g}) is neither the stage metrics' error "
                f"after the last measurement ({last_err}) nor the final error ({field('final_err_primary')}): the "
                "series is not the primary parameter's error"
            )
        true = scan.get("true_params") or {}
        self.params = dict(true.get("params") or {})
        lo_phys, hi_phys = (true.get("bounds") or {}).get(PRIMARY_PARAM, (None, None))
        if lo_phys is None:
            raise UseCaseError("true_params carry no zeeman_shift bounds")
        self.width = float(hi_phys) - float(lo_phys)
        if PRIMARY_PARAM not in self.params:
            raise UseCaseError("true_params carry no zeeman_shift value")
        self.theta = float(self.params[PRIMARY_PARAM])
        for name in ("stage_fisher_star", "stage_fwhm_true_phys", "stage_prior_info", "snr", "noise_sigma"):
            if field(name) is None:
                raise UseCaseError(f"stored column {name!r} is missing (rows from before it was added need a re-run)")
        self.fisher_star = float(field("stage_fisher_star"))
        self.fwhm = float(field("stage_fwhm_true_phys"))
        self.prior_info = float(field("stage_prior_info"))
        self.snr = float(field("snr"))
        self.noise_sigma = float(field("noise_sigma"))
        det = field("dip_detected_step")
        self.detected = None if det is None else int(det)


def _check_same(runs: list[_Run]) -> None:
    """Every repeat of a use case must share width, line shape, noise and fwhm (the integral is per use case)."""
    ref = runs[0]
    shape_ref = {k: v for k, v in ref.params.items() if k != PRIMARY_PARAM}
    for r in runs[1:]:
        shape = {k: v for k, v in r.params.items() if k != PRIMARY_PARAM}
        if shape.keys() != shape_ref.keys() or any(
            not math.isclose(shape[k], shape_ref[k], rel_tol=_SHAPE_REL_TOL, abs_tol=1e-12) for k in shape
        ):
            raise UseCaseError("the line shape differs between repeats (only fixed-shape use cases are supported)")
        if not math.isclose(r.width, ref.width, rel_tol=_WIDTH_REL_TOL):
            raise UseCaseError("the zeeman_shift range differs between repeats")
        if not math.isclose(r.noise_sigma, ref.noise_sigma, rel_tol=_SNR_REL_TOL):
            raise UseCaseError("the noise sigma differs between repeats")


def _line_statistics(
    generator_name: str, width_phys: float, fwhm_phys: float, run_thetas: list[float]
) -> tuple[float, dict[float, float]]:
    """``(mean over theta of int dip^2 dx, {theta: max g^2 at the run's own theta})`` for one generator.

    Rebuilds the true signal from the generator (fixed shape in the supported use cases) and evaluates it on the drive
    window for ``THETA_SAMPLES`` values of ``zeeman_shift`` spread over the range the generator draws, i.e. exactly
    the average over ``theta`` the bounds are on. ``dip = baseline - signal`` with the baseline the signal far from
    every line. The drawn range is read from the generator and must equal the width stored with the runs.
    """
    from nvision.metrics.stages import _grad_sq_fn
    from nvision.sim.combinations import CombinationGrid

    gen = CombinationGrid()._resolve_generators.get(generator_name)  # the grid's name registry
    if gen is None:
        raise UseCaseError(f"generator {generator_name!r} is not registered; cannot rebuild its line")
    drawn_lo, drawn_hi = float(gen.zeeman_min_phys), float(gen.zeeman_max_phys)
    if not math.isclose(drawn_hi - drawn_lo, width_phys, rel_tol=_WIDTH_REL_TOL):
        raise UseCaseError(
            f"stored zeeman_shift range ({width_phys:g} Hz) differs from the generator's drawn range "
            f"({drawn_hi - drawn_lo:g} Hz)"
        )
    # Any repeat gives the same fixed line (zeeman_shift is overwritten below); the generator stratifies
    # zeeman_shift across repeats, so it needs a repeat index.
    signal = gen.generate(random.Random(0), repeat_idx=0, n_repeats=1)
    if signal.model.controls_drive:
        raise UseCaseError("drive-controlled signals have no fixed line to integrate")
    x_lo, x_hi = float(gen.drive_freq_min_phys), float(gen.drive_freq_max_phys)
    n_x = max(_MIN_X_POINTS, int(_X_POINTS_PER_FWHM * (x_hi - x_lo) / fwhm_phys) + 1)
    xs = np.linspace(x_lo, x_hi, n_x)
    spec = signal.model.spec
    names = list(signal.parameter_names)
    idx = names.index(PRIMARY_PARAM)
    base = [float(v) for v in spec.pack_params(signal.typed_parameters)]
    far = np.array([x_hi + 1.0e4 * (x_hi - x_lo)])
    dip_sq: list[float] = []
    best_g2: dict[float, float] = {}
    quantiles = drawn_lo + (np.arange(THETA_SAMPLES) + 0.5) / THETA_SAMPLES * (drawn_hi - drawn_lo)
    for theta in [*quantiles, *sorted(set(run_thetas))]:
        values = list(base)
        values[idx] = float(theta)
        params = spec.unpack_params(values)
        y = signal.model.compute_many_float64(xs, params)
        baseline = float(signal.model.compute_many_float64(far, params)[0])
        ints = sb.line_integrals(xs, baseline - y, _grad_sq_fn(signal, PRIMARY_PARAM, values)(None, xs))
        if len(dip_sq) < THETA_SAMPLES:
            dip_sq.append(ints.dip_sq_integral)
        best_g2[float(theta)] = ints.max_grad_sq
    return float(np.mean(dip_sq)), best_g2


def _locator_block(
    runs: list[_Run], grid: np.ndarray, tolerance_phys: float, kappa: float, width: float, dd: float | None, budget: int
) -> dict:
    errors, running = sb.align_errors([r.steps for r in runs], [r.errs for r in runs], grid)
    usable = sb.usable_columns(running)
    mse = sb.mean_squared_error(errors)
    surv = sb.survival(errors, tolerance_phys)
    kappa_eff = sb.fit_kappa_eff(grid[usable], surv[usable])
    det = sb.detection_summary([r.detected for r in runs], budget)
    never_norm = 1.0 / kappa
    out: dict[str, Any] = {
        "n_runs": len(runs),
        # Share of runs whose stored J0 (stage_prior_info) is 4 pi^2 / W^2 of the drawn range: a check on the stored
        # columns only (W itself comes from the signal's bounds, confirmed against the generator). Not 1 means the rows
        # were recorded with a different prior and need a re-run.
        "prior_info_match_share": float(
            np.mean([math.isclose(4.0 * math.pi**2 / width**2, r.prior_info, rel_tol=_WIDTH_REL_TOL) for r in runs])
        ),
        "mse": _num_list(np.where(usable, mse, np.nan)),
        "survival": _num_list(np.where(usable, surv, np.nan)),
        "running_share": _num_list(running.mean(axis=0)),
        "kappa_eff": _num(kappa_eff),
        "kappa_ratio": None if kappa_eff is None else _num(kappa_eff / kappa),
        "detection": {
            "median": det.median,
            "p75": det.p75,
            "capped_share": det.capped_share,
            "median_over_inv_kappa": det.median / never_norm,
            "p75_over_inv_kappa": det.p75 / never_norm,
            "median_over_cells": None if dd is None else det.median / (width / dd),
            "p75_over_cells": None if dd is None else det.p75 / (width / dd),
        },
    }
    return out


def use_case_payload(generator: str, noise: str, rows: dict[str, list[_Run]], budget: int) -> dict:
    """The payload of one use case from its loaded runs (``rows``: locator name -> runs)."""
    all_runs = [r for runs in rows.values() for r in runs]
    if not all_runs:
        raise UseCaseError("no runs for the selected locators")
    _check_same(all_runs)
    ref = all_runs[0]
    width = ref.width
    contrast = ref.params.get("contrast_total")
    if contrast is None:
        raise UseCaseError("true_params carry no contrast_total: the article's SNR = C / sigma cannot be formed")
    # The SNR of the article (per-line contrast over the per-shot sigma) from the true signal and the stored sigma.
    # The run's own SNR label (``snr_label``) can differ in rows recorded under an older convention: reported, not used.
    snr = line_contrast(contrast) / ref.noise_sigma
    dip_sq, best_g2 = _line_statistics(generator, width, ref.fwhm, [r.theta for r in all_runs])
    sigma = ref.noise_sigma
    k = sb.SIM_SHOTS_PER_MEASUREMENT
    fisher = float(np.median([r.fisher_star for r in all_runs]))
    k_implied = float(np.median([sb.implied_shots(r.fisher_star, sigma, best_g2[r.theta]) for r in all_runs]))
    if not math.isclose(k_implied, k, rel_tol=sb.SHOTS_CROSS_CHECK_REL_TOL):
        raise UseCaseError(
            f"stored Fisher information implies {k_implied:.3f} shots per measurement, expected {k:g}: "
            "the rebuilt line and the cache disagree"
        )
    kappa = sb.kappa_bar(dip_sq, width, sigma, k)
    reach = sb.probe_spacing(width, ref.fwhm, snr, k)
    dd = None if reach is None else reach.dd_phys
    hwhm = ref.fwhm / 2.0
    tolerance = sb.SURVIVAL_TOLERANCE_HWHM * hwhm
    if not 2.0 * tolerance < width:
        raise UseCaseError(
            f"the drawn zeeman_shift range ({width:g} Hz) is not wider than 2 t = {2 * tolerance:g} Hz "
            f"(t = {sb.SURVIVAL_TOLERANCE_HWHM:g} HWHM): the event |error| > t cannot be bounded"
        )
    grid = n_grid(budget)
    curves: dict[str, Any] = {
        "change_of_measure": _num_list(sb.change_of_measure_curve(grid, width, kappa)),
        "survival_floor": _num_list(sb.survival_floor_curve(grid, width, kappa, tolerance)),
    }
    if dd is not None:
        curves["finite_reach"] = _num_list(sb.finite_reach_curve(grid, width, dd))
        curves["mean_model"] = _num_list(sb.mean_curve(grid, width, dd, fisher))
    cells = None if dd is None else width / dd
    predicted = None if dd is None or not width > dd else sb.predicted_sweep_ratio(width, dd)
    return {
        "generator": generator,
        "noise": noise,
        "width_phys": width,
        "fwhm_phys": ref.fwhm,
        "snr": snr,
        "snr_label": ref.snr,
        "noise_sigma": sigma,
        "shots": k,
        "fisher_max": fisher,
        "kappa_bar": kappa,
        "inv_kappa_bar": 1.0 / kappa,
        "kappa_lorentzian": sb.kappa_bar_lorentzian(width, hwhm, snr, k),
        "tolerance_phys": tolerance,
        "tolerance_hwhm": sb.SURVIVAL_TOLERANCE_HWHM,
        "kappa_ratio_floor": 1.0 / (1.0 - 2.0 * tolerance / width),
        "reach": None if reach is None else {"dd_phys": reach.dd_phys, "z": reach.z, "kind": reach.kind},
        "cells": cells,
        "regime": None if cells is None else ("finite-reach" if 1.0 / kappa < cells else "noise-limited"),
        "predicted_sweep_ratio": predicted,
        "budget": budget,
        "n": grid.astype(int).tolist(),
        "curves": curves,
        "locators": {
            name: _locator_block(runs, grid, tolerance, kappa, width, dd, budget) for name, runs in rows.items() if runs
        },
    }


def search_bounds_payload(
    combos: list[dict], load_meta: LoadMeta, generators: set[str], noises: set[str]
) -> list[dict]:
    """One entry per (generator, noise) among ``combos`` limited to the search locators and the given selection."""
    wanted = [
        c
        for c in combos
        if c["strategy"] in SEARCH_BOUND_LOCATORS
        and c["generator"] in generators
        and c["noise"] in noises
        and not str(c["generator"]).startswith("MATLAB:")
    ]
    by_case: dict[tuple[str, str], list[dict]] = {}
    for combo in wanted:
        by_case.setdefault((combo["generator"], combo["noise"]), []).append(combo)
    out: list[dict] = []
    for (generator, noise), case_combos in sorted(by_case.items()):
        try:
            rows: dict[str, list[_Run]] = {}
            for combo in case_combos:
                if combo["strategy"] in rows:
                    raise UseCaseError(f"several cached combos for {combo['strategy']} (older fingerprints?)")
                rows[combo["strategy"]] = [_Run(entries, row) for entries, row in load_meta(combo)]
            budgets = {int(c["max_steps"]) for c in case_combos if c.get("max_steps")}
            if len(budgets) != 1:
                raise UseCaseError(f"locators of this use case ran with different budgets: {sorted(budgets)}")
            out.append(use_case_payload(generator, noise, rows, budgets.pop()))
        except UseCaseError as exc:
            out.append({"generator": generator, "noise": noise, "error": str(exc)})
    return out


def add_search_bounds_route(
    app: Any, get_combos: Callable[[], list[dict]], make_bridge: Callable[[], Any], combo_key: Callable[[dict], str]
) -> None:
    """Register ``GET /api/search-bounds?generator=a,b&noise=x,y`` on ``app`` (one entry per use case, see module doc).

    ``get_combos``: the server's memoised combo listing; ``make_bridge``: opens a cache bridge (closed here);
    ``combo_key``: the combo's cache key (the server rebuilds it from the stored schema version and fingerprint).
    """
    from nvision.sim.combinations import CombinationGrid

    @app.get("/api/search-bounds")
    def search_bounds(generator: str, noise: str) -> list[dict]:
        bridge = make_bridge()
        try:

            def load_meta(combo: dict) -> list[tuple[list[dict] | None, dict]]:
                repo = bridge.get_cache_for_category(CombinationGrid.generator_category(str(combo["generator"])))
                key = combo_key(combo)
                achieved = int(combo.get("repeats", 0))
                meta = repo._repeats.load_repeats_meta(key, achieved)
                return meta if meta is not None else repo._repeats.load_repeats(key, achieved)

            return search_bounds_payload(get_combos(), load_meta, set(generator.split(",")), set(noise.split(",")))
        finally:
            bridge.close()
