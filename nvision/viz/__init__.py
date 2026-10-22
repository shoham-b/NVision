from __future__ import annotations

from nvision.viz.base import VizBase
from nvision.viz.bayesian import BayesianMixin
from nvision.viz.measurements import MeasurementsMixin


class Viz(VizBase, MeasurementsMixin, BayesianMixin):
    """Visualization facade: per-repeat scan and Bayesian plots."""


__all__ = ["Viz"]
