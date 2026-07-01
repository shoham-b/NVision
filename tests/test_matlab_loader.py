"""Tests for MATLAB ESR file loading and measurement utilities."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from nvision.models.observation import Observation
from nvision.sim.defaults import NVISION_NOISE_GAUSS
from nvision.tools.matlab_loader import (
    MatlabDataFile,
    _fill_nan_nearest,
    _resolve_mat_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mat(freq_mhz, signal_3d, curr_iter=None):
    """Build a mock mat dict that mimics scipy.io.loadmat output."""
    if curr_iter is None:
        curr_iter = signal_3d.shape[2]
    esr = SimpleNamespace()
    esr.frequency = np.asarray(freq_mhz, dtype=np.float64)
    esr.signal = signal_3d.astype(np.float64)
    esr.currIter = curr_iter
    top = SimpleNamespace()
    top.ESR = esr
    return {"myStruct": top}


def _flat_signal(n_freqs, n_shots, value=1.0, dip_idx=None, dip_value=0.95):
    """Return a (2, n_freqs, n_shots) signal array with flat baseline and optional dip."""
    baseline = np.full((n_freqs, n_shots), 130.0)
    with_freq = np.full((n_freqs, n_shots), 130.0 * value)
    if dip_idx is not None:
        with_freq[dip_idx, :] = 130.0 * dip_value
    return np.stack([baseline, with_freq], axis=0)


# ---------------------------------------------------------------------------
# _fill_nan_nearest
# ---------------------------------------------------------------------------


def test_fill_nan_nearest_no_nans():
    arr = np.array([1.0, 2.0, 3.0])
    result = _fill_nan_nearest(arr)
    np.testing.assert_array_equal(result, arr)


def test_fill_nan_nearest_single_interior():
    arr = np.array([1.0, float("nan"), 3.0])
    result = _fill_nan_nearest(arr)
    assert math.isfinite(result[1])
    assert result[1] in (1.0, 3.0)


def test_fill_nan_nearest_leading():
    arr = np.array([float("nan"), float("nan"), 5.0])
    result = _fill_nan_nearest(arr)
    assert result[0] == 5.0
    assert result[1] == 5.0
    assert result[2] == 5.0


def test_fill_nan_nearest_trailing():
    arr = np.array([7.0, float("nan"), float("nan")])
    result = _fill_nan_nearest(arr)
    assert result[1] == 7.0
    assert result[2] == 7.0


def test_fill_nan_nearest_all_nan_raises():
    arr = np.array([float("nan"), float("nan")])
    with pytest.raises(ValueError, match="All signal values are NaN"):
        _fill_nan_nearest(arr)


def test_fill_nan_nearest_does_not_mutate_original():
    arr = np.array([1.0, float("nan"), 3.0])
    original = arr.copy()
    _fill_nan_nearest(arr)
    np.testing.assert_array_equal(arr, original)


# ---------------------------------------------------------------------------
# _resolve_mat_path
# ---------------------------------------------------------------------------


def test_resolve_mat_path_full_path_returned_as_is(tmp_path):
    f = tmp_path / "myfile.mat"
    f.touch()
    result = _resolve_mat_path(f)
    assert result == f


def test_resolve_mat_path_bare_name_resolved(tmp_path, monkeypatch):
    f = tmp_path / "data.mat"
    f.touch()
    import nvision.tools.matlab_loader as ml
    monkeypatch.setattr(ml, "_MATLAB_DATA_DIR", tmp_path)
    result = _resolve_mat_path(Path("data.mat"))
    assert result == f


def test_resolve_mat_path_bare_name_missing_returns_as_is(tmp_path, monkeypatch):
    import nvision.tools.matlab_loader as ml
    monkeypatch.setattr(ml, "_MATLAB_DATA_DIR", tmp_path)
    result = _resolve_mat_path(Path("nonexistent.mat"))
    assert result == Path("nonexistent.mat")


# ---------------------------------------------------------------------------
# MatlabDataFile.load — valid cases
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_mat():
    """5-frequency, 10-shot flat signal with a 4% dip at index 2."""
    freq_mhz = np.linspace(2770.0, 2970.0, 5)
    signal_3d = _flat_signal(5, 10, value=1.0, dip_idx=2, dip_value=0.96)
    return _make_mat(freq_mhz, signal_3d, curr_iter=10)


def test_load_freq_hz(simple_mat, tmp_path):
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=simple_mat):
        data = MatlabDataFile.load(mat_file)
    assert data.freq_hz[0] == pytest.approx(2770e6)
    assert data.freq_hz[-1] == pytest.approx(2970e6)
    assert len(data.freq_hz) == 5


def test_load_signal_ratio(simple_mat, tmp_path):
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=simple_mat):
        data = MatlabDataFile.load(mat_file)
    # Non-dip positions should be ~1.0; dip at index 2 should be ~0.96
    np.testing.assert_allclose(data.signal[[0, 1, 3, 4]], 1.0, atol=1e-9)
    assert data.signal[2] == pytest.approx(0.96, abs=1e-6)


def test_load_n_valid_shots_from_curr_iter(tmp_path):
    freq_mhz = np.linspace(2800.0, 2900.0, 4)
    signal_3d = _flat_signal(4, 20, value=1.0)
    mat = _make_mat(freq_mhz, signal_3d, curr_iter=8)
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        data = MatlabDataFile.load(mat_file)
    assert data.n_valid_shots == 8


def test_load_valid_shots_override(tmp_path):
    freq_mhz = np.linspace(2800.0, 2900.0, 4)
    signal_3d = _flat_signal(4, 20, value=1.0)
    mat = _make_mat(freq_mhz, signal_3d, curr_iter=20)
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        data = MatlabDataFile.load(mat_file, valid_shots=5)
    assert data.n_valid_shots == 5


def test_load_noise_std_override(simple_mat, tmp_path):
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=simple_mat):
        data = MatlabDataFile.load(mat_file, noise_std_override=0.042)
    assert data.noise_std == pytest.approx(0.042)


def test_load_signal_clipped_to_range(tmp_path):
    """Signal values outside [1e-6, 2.0] should be clipped."""
    freq_mhz = np.array([2800.0, 2850.0, 2900.0])
    baseline = np.full((3, 5), 100.0)
    with_freq = np.array([[50.0] * 5, [100.0] * 5, [250.0] * 5])  # 0.5, 1.0, 2.5
    signal_3d = np.stack([baseline, with_freq], axis=0)
    mat = _make_mat(freq_mhz, signal_3d, curr_iter=5)
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        data = MatlabDataFile.load(mat_file)
    assert data.signal[0] >= 1e-6
    assert data.signal[2] <= 2.0


def test_load_nan_bins_filled(tmp_path):
    """Frequency bins where all shots are NaN should be filled with nearest valid."""
    freq_mhz = np.array([2800.0, 2850.0, 2900.0])
    baseline = np.full((3, 5), 100.0)
    with_freq = np.full((3, 5), 100.0)
    # Make the middle bin's baseline zero so ratio becomes NaN
    baseline[1, :] = 0.0
    signal_3d = np.stack([baseline, with_freq], axis=0)
    mat = _make_mat(freq_mhz, signal_3d, curr_iter=5)
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        data = MatlabDataFile.load(mat_file)
    assert math.isfinite(data.signal[1])


# ---------------------------------------------------------------------------
# MatlabDataFile.load — error cases
# ---------------------------------------------------------------------------


def test_load_missing_my_struct_raises(tmp_path):
    mat_file = tmp_path / "bad.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value={"other_key": None}):
        with pytest.raises(KeyError, match="myStruct"):
            MatlabDataFile.load(mat_file)


def test_load_missing_esr_field_raises(tmp_path):
    top = SimpleNamespace()  # no .ESR attribute
    mat = {"myStruct": top}
    mat_file = tmp_path / "bad.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        with pytest.raises(KeyError, match="ESR"):
            MatlabDataFile.load(mat_file)


def test_load_wrong_signal_shape_raises(tmp_path):
    esr = SimpleNamespace()
    esr.frequency = np.array([2800.0, 2850.0])
    esr.signal = np.ones((3, 2, 10))  # shape[0] != 2
    esr.currIter = 10
    top = SimpleNamespace()
    top.ESR = esr
    mat = {"myStruct": top}
    mat_file = tmp_path / "bad.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        with pytest.raises(ValueError, match="shape"):
            MatlabDataFile.load(mat_file)


def test_load_freq_length_mismatch_raises(tmp_path):
    esr = SimpleNamespace()
    esr.frequency = np.array([2800.0, 2850.0, 2900.0])  # 3 freqs
    esr.signal = np.ones((2, 5, 10))  # 5 freqs in signal
    esr.currIter = 10
    top = SimpleNamespace()
    top.ESR = esr
    mat = {"myStruct": top}
    mat_file = tmp_path / "bad.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        with pytest.raises(ValueError, match="Frequency array length"):
            MatlabDataFile.load(mat_file)


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        MatlabDataFile.load(Path("/nonexistent/path/file.mat"))


def test_load_zero_valid_shots_raises(tmp_path):
    freq_mhz = np.array([2800.0, 2850.0])
    signal_3d = _flat_signal(2, 10, value=1.0)
    mat = _make_mat(freq_mhz, signal_3d, curr_iter=0)
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        with pytest.raises(ValueError, match="No valid shots"):
            MatlabDataFile.load(mat_file)


def test_load_implausible_noise_falls_back_to_default(tmp_path):
    """All-identical shots give zero shot-to-shot std — should fall back to default."""
    freq_mhz = np.linspace(2800.0, 2900.0, 5)
    baseline = np.full((5, 10), 100.0)
    with_freq = np.full((5, 10), 100.0)  # identical shots → std = 0
    signal_3d = np.stack([baseline, with_freq], axis=0)
    mat = _make_mat(freq_mhz, signal_3d, curr_iter=10)
    mat_file = tmp_path / "esr.mat"
    mat_file.touch()
    with patch("nvision.tools.matlab_loader._load_mat_v5", return_value=mat):
        data = MatlabDataFile.load(mat_file)
    assert data.noise_std == pytest.approx(NVISION_NOISE_GAUSS)


# ---------------------------------------------------------------------------
# MatlabDataFile.measure
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_data():
    """A 5-frequency MatlabDataFile with a dip at index 2."""
    freq_hz = np.array([2770e6, 2820e6, 2870e6, 2920e6, 2970e6])
    signal = np.array([1.0, 1.0, 0.96, 1.0, 1.0])
    return MatlabDataFile(freq_hz=freq_hz, signal=signal, noise_std=0.01, n_valid_shots=8)


def test_measure_returns_observation(simple_data):
    freq_lo, freq_hi = 2770e6, 2970e6
    obs = simple_data.measure(0.5, freq_lo, freq_hi)
    assert isinstance(obs, Observation)


def test_measure_x_unit_preserved(simple_data):
    freq_lo, freq_hi = 2770e6, 2970e6
    x_unit = 0.3
    obs = simple_data.measure(x_unit, freq_lo, freq_hi)
    assert obs.x == pytest.approx(x_unit)


def test_measure_snaps_to_nearest_grid_point(simple_data):
    """x_unit=0.5 maps to 2870 MHz (index 2), which has signal=0.96."""
    freq_lo, freq_hi = 2770e6, 2970e6
    obs = simple_data.measure(0.5, freq_lo, freq_hi)
    assert obs.signal_value == pytest.approx(0.96)


def test_measure_noise_std_from_data(simple_data):
    obs = simple_data.measure(0.0, 2770e6, 2970e6)
    assert obs.noise_std == pytest.approx(0.01)


def test_measure_boundary_low(simple_data):
    """x_unit=0.0 should snap to the first grid point."""
    freq_lo, freq_hi = 2770e6, 2970e6
    obs = simple_data.measure(0.0, freq_lo, freq_hi)
    assert obs.signal_value == pytest.approx(1.0)


def test_measure_boundary_high(simple_data):
    """x_unit=1.0 should snap to the last grid point."""
    freq_lo, freq_hi = 2770e6, 2970e6
    obs = simple_data.measure(1.0, freq_lo, freq_hi)
    assert obs.signal_value == pytest.approx(1.0)


def test_measure_round_trip_unit_conversion(simple_data):
    """Converting any x_unit back to Hz and snapping must land within one grid step."""
    freq_lo, freq_hi = 2770e6, 2970e6
    grid_step = (freq_hi - freq_lo) / (len(simple_data.freq_hz) - 1)
    for x_unit in np.linspace(0.0, 1.0, 11):
        obs = simple_data.measure(x_unit, freq_lo, freq_hi)
        phys_hz = freq_lo + x_unit * (freq_hi - freq_lo)
        nearest = simple_data.freq_hz[np.argmin(np.abs(simple_data.freq_hz - phys_hz))]
        assert abs(phys_hz - nearest) <= grid_step / 2 + 1e-3


# ---------------------------------------------------------------------------
# _MatlabSignalProxy and _MatlabExperiment from matlab_cmd
# ---------------------------------------------------------------------------


def test_matlab_signal_proxy_parameter_values():
    from nvision.cli.matlab_cmd import _MatlabSignalProxy

    proxy = _MatlabSignalProxy(2770e6, 2970e6)
    vals = proxy.parameter_values()
    assert "frequency" in vals
    assert math.isnan(vals["frequency"])


def test_matlab_signal_proxy_get_param_value():
    from nvision.cli.matlab_cmd import _MatlabSignalProxy

    proxy = _MatlabSignalProxy(2770e6, 2970e6)
    assert math.isnan(proxy.get_param_value("frequency"))
    assert math.isnan(proxy.get_param_value("linewidth"))


def test_matlab_signal_proxy_all_bounds():
    from nvision.cli.matlab_cmd import _MatlabSignalProxy

    proxy = _MatlabSignalProxy(2770e6, 2970e6)
    bounds = proxy.all_bounds()
    assert bounds["frequency"] == (2770e6, 2970e6)


def test_matlab_signal_proxy_callable_returns_nan():
    from nvision.cli.matlab_cmd import _MatlabSignalProxy

    proxy = _MatlabSignalProxy(2770e6, 2970e6)
    assert math.isnan(proxy(0.5))


def test_matlab_experiment_x_bounds():
    from nvision.cli.matlab_cmd import _MatlabExperiment, _MatlabSignalProxy

    freq_hz = np.array([2770e6, 2870e6, 2970e6])
    signal = np.array([1.0, 0.96, 1.0])
    data = MatlabDataFile(freq_hz=freq_hz, signal=signal, noise_std=0.01, n_valid_shots=8)
    proxy = _MatlabSignalProxy(2770e6, 2970e6)
    exp = _MatlabExperiment(data, proxy, 2770e6, 2970e6)

    assert exp.x_min == 2770e6
    assert exp.x_max == 2970e6


def test_matlab_experiment_measure_delegates():
    from nvision.cli.matlab_cmd import _MatlabExperiment, _MatlabSignalProxy

    freq_hz = np.array([2770e6, 2870e6, 2970e6])
    signal = np.array([1.0, 0.96, 1.0])
    data = MatlabDataFile(freq_hz=freq_hz, signal=signal, noise_std=0.01, n_valid_shots=8)
    proxy = _MatlabSignalProxy(2770e6, 2970e6)
    exp = _MatlabExperiment(data, proxy, 2770e6, 2970e6)

    obs = exp.measure(0.5)
    assert isinstance(obs, Observation)
    assert obs.signal_value == pytest.approx(0.96, abs=1e-6)
