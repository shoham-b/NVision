"""Tests for the per-step series export (nvision.metrics.series)."""

from itertools import pairwise

import numpy as np

from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.metrics.series import _downsample_indices, extract_step_series
from nvision.models.observation import Observation
from nvision.models.observer import RunResult, StepSnapshot
from nvision.spectra.signal import SignalModel, TrueSignal
from nvision.spectra.spec import BasicParamSpec


class MockSignalModel(SignalModel):
    @property
    def spec(self):
        return BasicParamSpec(names=["frequency"], bounds={"frequency": (0.0, 1.0)})

    def parameter_names(self):
        return ["frequency"]

    def compute(self, x, params):
        return 1.0

    def compute_from_params(self, x, params):
        return 1.0

    def compute_vectorized_samples(self, x, samples):
        return np.ones(len(samples))


def _make_mock_belief(model, est, uncert):
    class MockBelief(GridMarginalDistribution):
        def estimates(self):
            return {"frequency": est}

        def uncertainty(self):
            return {"frequency": uncert}

        def copy(self):
            return self

        @property
        def physical_param_bounds(self):
            return {"frequency": (0.0, 1.0)}

    param = GridParameter(
        name="frequency",
        bounds=(0.0, 1.0),
        grid=np.linspace(0.0, 1.0, 10),
        posterior=np.ones(10) / 10,
    )
    return MockBelief(model=model, parameters=[param])


def _make_run(n_steps=10, true_value=0.5):
    model = MockSignalModel()
    true_signal = TrueSignal(model=model, typed_parameters=(true_value,), bounds={"frequency": (0.0, 1.0)})
    snapshots = []
    for i in range(n_steps):
        est = true_value + 0.1 / (i + 1)
        uncert = 0.2 / (i + 1)
        belief = _make_mock_belief(model, est, uncert)
        obs = Observation(x=0.5, signal_value=1.0, noise_std=0.01)
        snapshots.append(StepSnapshot(obs=obs, belief=belief, true_signal=true_signal))
    return RunResult(snapshots=snapshots, true_signal=true_signal)


def test_extract_step_series_basic():
    run = _make_run(n_steps=10)
    series = extract_step_series(run, param="frequency")

    assert series is not None
    assert series["s"] == list(range(1, 11))
    assert len(series["e"]) == 10
    assert len(series["u"]) == 10
    # Error at step 1 is |0.1| and decreases monotonically
    assert abs(series["e"][0] - 0.1) < 1e-9
    assert series["e"][-1] < series["e"][0]
    assert series["u"][-1] < series["u"][0]
    # tau is present (absolute frequency threshold or relative fallback)
    assert series.get("tau") is not None
    assert series["tau"] > 0


def test_extract_step_series_downsamples_long_runs():
    run = _make_run(n_steps=500)
    series = extract_step_series(run, param="frequency", max_points=80)

    assert len(series["s"]) <= 80
    # First and last steps always retained
    assert series["s"][0] == 1
    assert series["s"][-1] == 500
    # Steps strictly increasing
    assert all(a < b for a, b in zip(series["s"], series["s"][1:], strict=False))


def test_extract_step_series_handles_empty_run():
    model = MockSignalModel()
    true_signal = TrueSignal(model=model, typed_parameters=(0.5,), bounds={"frequency": (0.0, 1.0)})
    run = RunResult(snapshots=[], true_signal=true_signal)
    assert extract_step_series(run) is None
    assert extract_step_series(None) is None


class MockZeemanSignalModel(SignalModel):
    """A signal model where 'frequency' is fixed and 'zeeman_split' is the free parameter."""

    @property
    def spec(self):
        return BasicParamSpec(names=["zeeman_split"], bounds={"zeeman_split": (0.0, 1.0)})

    def parameter_names(self):
        return ["zeeman_split"]

    def compute(self, x, params):
        return 1.0

    def compute_from_params(self, x, params):
        return 1.0

    def compute_vectorized_samples(self, x, samples):
        return np.ones(len(samples))


def _make_mock_zeeman_belief(model, est, uncert):
    class MockBelief(GridMarginalDistribution):
        def estimates(self):
            return {"zeeman_split": est}

        def uncertainty(self):
            return {"zeeman_split": uncert}

        def copy(self):
            return self

        @property
        def physical_param_bounds(self):
            return {"zeeman_split": (0.0, 1.0)}

    param = GridParameter(
        name="zeeman_split",
        bounds=(0.0, 1.0),
        grid=np.linspace(0.0, 1.0, 10),
        posterior=np.ones(10) / 10,
    )
    return MockBelief(model=model, parameters=[param])


def _make_fixed_frequency_run(n_steps=10, true_value=0.5):
    """A run where the belief only tracks 'zeeman_split' -- 'frequency' is fixed."""
    model = MockZeemanSignalModel()
    true_signal = TrueSignal(
        model=model,
        typed_parameters=(true_value,),
        bounds={"zeeman_split": (0.0, 1.0)},
    )
    snapshots = []
    for i in range(n_steps):
        est = true_value + 0.1 / (i + 1)
        uncert = 0.2 / (i + 1)
        belief = _make_mock_zeeman_belief(model, est, uncert)
        obs = Observation(x=0.5, signal_value=1.0, noise_std=0.01)
        snapshots.append(StepSnapshot(obs=obs, belief=belief, true_signal=true_signal))
    return RunResult(snapshots=snapshots, true_signal=true_signal)


def test_extract_step_series_falls_back_when_frequency_is_fixed():
    """When 'frequency' isn't a free parameter (belief never estimates it), the
    series should fall back to the model's splitting parameter instead of
    silently returning None for every repeat."""
    run = _make_fixed_frequency_run(n_steps=10)
    series = extract_step_series(run, param="frequency")

    assert series is not None
    assert series["s"] == list(range(1, 11))
    assert len(series["e"]) == 10
    assert abs(series["e"][0] - 0.1) < 1e-9


def test_downsample_indices_short_input_kept_verbatim():
    assert _downsample_indices(5, 80) == [0, 1, 2, 3, 4]


def test_downsample_indices_bounds_and_count():
    idx = _downsample_indices(10_000, 80)
    assert len(idx) <= 80
    assert idx[0] == 0
    assert idx[-1] == 9_999
    assert all(a < b for a, b in pairwise(idx))
