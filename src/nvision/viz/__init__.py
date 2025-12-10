from __future__ import annotations

from nvision.viz.base import VizBase
from nvision.viz.bayesian import BayesianMixin
from nvision.viz.experiments import ExperimentsMixin
from nvision.viz.measurements import MeasurementsMixin


class Viz(VizBase, ExperimentsMixin, MeasurementsMixin, BayesianMixin):
    """Visualization facade combining all plotting capabilities."""

    pass


__all__ = ["Viz"]
