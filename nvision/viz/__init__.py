from __future__ import annotations

from nvision.viz.base import VizBase
from nvision.viz.bayesian import BayesianMixin
from nvision.viz.comparisons import ComparisonsMixin
from nvision.viz.experiments import ExperimentsMixin
from nvision.viz.measurements import MeasurementsMixin
from nvision.viz.metrics import MetricsVizMixin


class Viz(VizBase, ExperimentsMixin, MeasurementsMixin, BayesianMixin, ComparisonsMixin, MetricsVizMixin):
    """Visualization facade combining all plotting capabilities."""

    pass


__all__ = ["Viz"]
