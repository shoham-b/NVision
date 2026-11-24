"""``/api/detection-aligned``: localize ratio and Gaussian onset per use case and locator, every run aligned at its own
dip detection (:mod:`nvision.metrics.detection_aligned`).

Read-only: built from the stored per-step series that ``/api/scan-fields`` serves (``s``, ``e``, ``u``, ``g``, ``x``)
and the stored scalars ``detection_step``, ``j_tau_phys`` and ``i_max``, so it works on an
existing cache.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import FastAPI

from nvision.metrics import detection_aligned as da
from nvision.metrics import search_bounds as sb
from nvision.metrics.run_record import DIP_TEST, SIGMA_DROP, SIM_TRUTH, RunRecord, compare_detectors

PRIMARY_PARAM = "zeeman_shift"


def _float(v: Any) -> float | None:
    return None if v is None else float(v)


def record_from_row(row: dict[str, Any]) -> RunRecord:
    """One ``/api/scan-fields`` row as the shared :class:`~nvision.metrics.run_record.RunRecord` (truth: simulated).

    Raises when the row lacks the series or scalars the analysis needs (a run recorded before they existed: re-run).
    """
    cmp = row.get("compare") or {}
    series = row.get("series") or {}
    s, e, u = series.get("step"), series.get("error"), series.get("claimed_sigma")
    g, x = series.get("negentropy"), series.get("far_placed")
    if not s or not e or g is None or not u:
        raise ValueError(
            "a run has no stored step/error/negentropy/claimed_sigma series (recorded before them: re-run)"
        )
    for name in ("i_max", "window_phys", "fwhm_phys"):
        if cmp.get(name) is None:
            raise ValueError(f"a run has no stored {name!r} (recorded before it: re-run)")
    keep = [i for i, v in enumerate(e) if v is not None and u[i] is not None]
    det = cmp.get("detection_step")
    params = (row.get("true_params") or {}).get("params") or {}
    return RunRecord(
        source="sim",
        truth=SIM_TRUTH,
        steps=np.array([s[i] for i in keep], dtype=np.float64),
        estimate=None,
        claimed_sigma=np.array([u[i] for i in keep], dtype=np.float64),
        error=np.abs(np.array([e[i] for i in keep], dtype=np.float64)),
        i_max=float(cmp["i_max"]),
        window=float(cmp["window_phys"]),
        fwhm=float(cmp["fwhm_phys"]),
        snr=None,  # the article SNR needs the per-shot sigma, which /api/search-bounds rebuilds per use case
        budget=int(s[-1]),
        stop_rule="full-budget",
        known_nuisances=False,
        dip_test_step=None if det is None else int(det),
        negentropy=np.array([np.nan if g[i] is None else g[i] for i in keep], dtype=np.float64),
        j_tau=_float(cmp.get("j_tau_phys")),
        far_placed=None if x is None else np.array([np.nan if x[i] is None else x[i] for i in keep], dtype=np.float64),
        far_mass_tau=_float(cmp.get("far_mass_tau")),
        line_params=dict(params),
    )


def _detection_cell(rows: list[dict[str, Any]]) -> float | None:
    """``Delta d`` (Hz) of the elimination model for the use case: eq:sm-reach on the drawn ``zeeman_shift`` range.

    Uses the stored SNR label (C / sigma) of the rows; ``None`` when a single measurement has no reach at this SNR.
    """
    ref = rows[0]
    bounds = ((ref.get("true_params") or {}).get("bounds") or {}).get(PRIMARY_PARAM)
    cmp = ref.get("compare") or {}
    fwhm, snr = cmp.get("fwhm_phys"), cmp.get("snr")
    if not bounds or fwhm is None or snr is None:
        raise ValueError("rows carry no zeeman_shift bounds / fwhm_phys / snr")
    width = float(bounds[1]) - float(bounds[0])
    reach = sb.probe_spacing(width, float(fwhm), float(snr))
    return None if reach is None else reach.dd_phys


def analyse_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The detection-aligned numbers of one locator's repeats in one use case (rows of ``/api/scan-fields``).

    Runs are aligned at the dip test (the simulation's detector); ``detector_comparison`` also runs the lab's
    sigma-drop detector on the same runs and reports how far the two definitions are apart.
    """
    rows = [r for r in rows if r.get("series")]
    if not rows:
        return {"unavailable": "no runs with a per-step series"}
    records = [record_from_row(r) for r in rows]
    imax = float(np.median([r.i_max for r in records]))
    budget = max(r.budget for r in records)
    runs = [a for a in (da.align(r, DIP_TEST) for r in records) if a is not None]
    out = da.analyse(runs, budget=budget, i_max=imax, dd_phys=_detection_cell(rows), n_undetected=len(rows) - len(runs))
    out["detector"] = DIP_TEST.name
    out["detector_comparison"] = compare_detectors(records, DIP_TEST, SIGMA_DROP)
    return out


def add_detection_aligned_route(app: FastAPI, scan_fields: Callable[[str, str], list[dict[str, Any]]]) -> None:
    """Register ``GET /api/detection-aligned?generator=a,b&noise=x,y`` (comma-separated, as ``/api/scan-fields``)."""

    @app.get("/api/detection-aligned")
    def detection_aligned_endpoint(generator: str, noise: str) -> list[dict]:
        grouped: dict[tuple[str, str, str], list[dict]] = {}
        for row in scan_fields(generator, noise):
            grouped.setdefault((row["generator"], row["noise"], row["strategy"]), []).append(row)
        return [
            {"generator": g, "noise": n, "strategy": s, **analyse_rows(rows)}
            for (g, n, s), rows in sorted(grouped.items())
        ]
