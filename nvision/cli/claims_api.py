"""``/api/claims``: the article-claim numbers (``nvision/metrics/claims_report.py``) per use case and locator.

Read-only: computed from the stored per-step series that the ``/api/scan-fields`` handler serves, so it works on an
existing cache with no re-run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from nvision.metrics.claims_report import analyse_use_case


def add_claims_route(app: FastAPI, scan_fields: Callable[[str, str], list[dict[str, Any]]]) -> None:
    """Register ``GET /api/claims?generator=a,b&noise=x,y`` (comma-separated, as ``/api/scan-fields``)."""

    @app.get("/api/claims")
    def claims_endpoint(generator: str, noise: str) -> list[dict]:
        grouped: dict[tuple[str, str, str], list[dict]] = {}
        for row in scan_fields(generator, noise):
            grouped.setdefault((row["generator"], row["noise"], row["strategy"]), []).append(row)
        return [
            {"generator": g, "noise": n, "strategy": s, **analyse_use_case(rows)}
            for (g, n, s), rows in sorted(grouped.items())
        ]
