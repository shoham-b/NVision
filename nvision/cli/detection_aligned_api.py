"""``/api/detection-aligned``: localize ratio and Gaussian onset per use case and locator, every run aligned at its own
dip detection (:mod:`nvision.metrics.detection_aligned`).

Read-only: built from the stored per-step series that ``/api/scan-fields`` serves (``s``, ``e``, ``g``) and the stored
scalars ``dip_detected_step``, ``posterior_J_tau_phys`` and ``stage_fisher_star``, so it works on an existing cache.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import FastAPI

from nvision.metrics import detection_aligned as da
from nvision.metrics import search_bounds as sb

PRIMARY_PARAM = "zeeman_shift"


def _float(v: Any) -> float | None:
    return None if v is None else float(v)


def aligned_run(row: dict[str, Any]) -> da.AlignedRun | None:
    """One scan-fields row as an :class:`~nvision.metrics.detection_aligned.AlignedRun`; ``None`` if it never detected.

    Raises when the row lacks the series or scalars the analysis needs (a run recorded before they existed).
    """
    cmp = row.get("compare") or {}
    det = cmp.get("dip_detected_step")
    if det is None:
        return None
    series = row.get("series") or {}
    s, e, g = series.get("s"), series.get("e"), series.get("g")
    if not s or not e or g is None:
        raise ValueError("a run has no stored s/e/g series (recorded before the posterior metrics: re-run)")
    keep = [i for i, v in enumerate(e) if v is not None]
    steps = np.array([s[i] for i in keep], dtype=np.float64)
    err = np.abs(np.array([e[i] for i in keep], dtype=np.float64))
    neg = np.array([np.nan if g[i] is None else g[i] for i in keep], dtype=np.float64)
    return da.AlignedRun(steps, err, neg, int(det), _float(cmp.get("posterior_J_tau_phys")))


def _detection_cell(rows: list[dict[str, Any]]) -> float | None:
    """``Delta d`` (Hz) of the elimination model for the use case: eq:sm-reach on the drawn ``zeeman_shift`` range.

    Uses the stored SNR label (C / sigma) of the rows; ``None`` when a single measurement has no reach at this SNR.
    """
    ref = rows[0]
    bounds = ((ref.get("true_params") or {}).get("bounds") or {}).get(PRIMARY_PARAM)
    cmp = ref.get("compare") or {}
    fwhm, snr = cmp.get("stage_fwhm_true_phys"), cmp.get("snr")
    if not bounds or fwhm is None or snr is None:
        raise ValueError("rows carry no zeeman_shift bounds / stage_fwhm_true_phys / snr")
    width = float(bounds[1]) - float(bounds[0])
    reach = sb.probe_spacing(width, float(fwhm), float(snr))
    return None if reach is None else reach.dd_phys


def analyse_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The detection-aligned numbers of one locator's repeats in one use case (rows of ``/api/scan-fields``)."""
    rows = [r for r in rows if r.get("series")]
    if not rows:
        return {"unavailable": "no runs with a per-step series"}
    imax = float(np.median([float(r["compare"]["stage_fisher_star"]) for r in rows]))
    budget = int(max(r["series"]["s"][-1] for r in rows))
    runs = [a for a in (aligned_run(r) for r in rows) if a is not None]
    return da.analyse(
        runs, budget=budget, i_max=imax, dd_phys=_detection_cell(rows), n_undetected=len(rows) - len(runs)
    )


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
