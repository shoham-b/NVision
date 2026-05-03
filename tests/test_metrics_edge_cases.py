import importlib.util
import math
import sys
import unittest
from unittest.mock import MagicMock

# Extensive mocking to bypass missing dependencies in the restricted environment
sys.modules["polars"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["numba"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["scipy.optimize"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["nvision.models.experiment"] = MagicMock()

# Since we can't easily import the module without triggering a cascade of imports
# we'll use the faithful reproduction of the math tools for the test environment
# but the CORE logic being tested is the function from metrics.py.


def _maybe_finite(value: object) -> float | None:
    if isinstance(value, int | float):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


def _first_finite(estimate: dict, keys) -> float | None:
    for key in keys:
        value = _maybe_finite(estimate.get(key))
        if value is not None:
            return value
    return None


def _promote_uncert(estimate, metrics):
    if "uncert" in metrics:
        return
    preferred_uncert = _first_finite(
        estimate,
        ("uncert_frequency", "uncert_position", "uncert_x1", "uncert_peak_x"),
    )
    if preferred_uncert is not None:
        metrics["uncert"] = preferred_uncert
        return
    for key, raw in estimate.items():
        if key.startswith("uncert_"):
            value = _maybe_finite(raw)
            if value is not None:
                metrics["uncert"] = value
                return


mock_math = MagicMock()
mock_math._maybe_finite = _maybe_finite
mock_math._first_finite = _first_finite
mock_math._promote_uncert = _promote_uncert
sys.modules["nvision.tools.math"] = mock_math


spec = importlib.util.spec_from_file_location("nvision.runner.metrics", "nvision/runner/metrics.py")
metrics_module = importlib.util.module_from_spec(spec)
sys.modules["nvision.runner.metrics"] = metrics_module
spec.loader.exec_module(metrics_module)
_scan_attempt_metrics = metrics_module._scan_attempt_metrics


class TestMetricsEdgeCases(unittest.TestCase):
    def test_empty_truth(self):
        """Verify that empty truth_positions do not cause IndexError even with multiple estimates."""
        truth = []
        estimate = {"x1_hat": 1.0, "x2_hat": 2.0, "uncert": 0.1}
        result = _scan_attempt_metrics(truth, estimate)
        assert result.get("uncert") == 0.1
        assert "abs_err_x1" not in result
        assert "abs_err_x2" not in result
        assert "pair_rmse" not in result

    def test_single_truth(self):
        """Verify metric calculation for a single peak."""
        truth = [1.0]
        estimate = {"x_hat": 1.1, "uncert": 0.1}
        result = _scan_attempt_metrics(truth, estimate)
        assert math.isclose(result.get("abs_err_x"), 0.1)
        assert result.get("uncert") == 0.1

    def test_two_truth_happy_path(self):
        """Verify pair metrics calculation for two peaks."""
        truth = [1.0, 2.0]
        estimate = {"x1_hat": 1.1, "x2_hat": 2.1, "uncert": 0.1}
        result = _scan_attempt_metrics(truth, estimate)
        assert math.isclose(result.get("abs_err_x1"), 0.1)
        assert math.isclose(result.get("abs_err_x2"), 0.1)
        assert math.isclose(result.get("pair_rmse"), 0.1)
        assert result.get("uncert") == 0.1

    def test_three_truth_positions(self):
        """Verify that pair metrics are NOT calculated when there are more than 2 truth positions."""
        truth = [1.0, 2.0, 3.0]
        estimate = {"x1_hat": 1.1, "x2_hat": 2.1, "uncert": 0.1}
        result = _scan_attempt_metrics(truth, estimate)
        assert result == {"uncert": 0.1}

    def test_uncert_promotion_integration(self):
        """Verify that uncertainty promotion works."""
        truth = [1.0]
        estimate = {"x_hat": 1.1, "uncert_frequency": 0.05}
        result = _scan_attempt_metrics(truth, estimate)
        assert result["uncert"] == 0.05

    def test_entropy_fallback(self):
        """Verify entropy fallback logic."""
        truth = [1.0]
        estimate = {"x_hat": 1.1, "entropy": -2.5}
        result = _scan_attempt_metrics(truth, estimate)
        assert result["final_entropy"] == -2.5


if __name__ == "__main__":
    unittest.main()
