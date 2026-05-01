from unittest.mock import MagicMock

from nvision.models.experiment import CoreExperiment
from nvision.runner.metrics import _truth_positions


def test_truth_positions_filters_correctly():
    mock_true_signal = MagicMock()
    mock_true_signal.parameter_values.return_value = {
        "center_frequency": 2.87e9,
        "peak_position": 1.5,
        "amplitude": 0.5,
        "width": 0.01,
        "frequency_offset": 100.0,
        "something_else": 42.0
    }

    experiment = MagicMock(spec=CoreExperiment)
    experiment.true_signal = mock_true_signal

    positions = _truth_positions(experiment)

    assert len(positions) == 3
    assert 2.87e9 in positions
    assert 1.5 in positions
    assert 100.0 in positions
    assert 0.5 not in positions

def test_truth_positions_empty():
    mock_true_signal = MagicMock()
    mock_true_signal.parameter_values.return_value = {
        "amplitude": 0.5,
        "width": 0.01,
    }
    experiment = MagicMock(spec=CoreExperiment)
    experiment.true_signal = mock_true_signal

    assert _truth_positions(experiment) == []

def test_truth_positions_no_params():
    mock_true_signal = MagicMock()
    mock_true_signal.parameter_values.return_value = {}
    experiment = MagicMock(spec=CoreExperiment)
    experiment.true_signal = mock_true_signal

    assert _truth_positions(experiment) == []
