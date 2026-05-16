"""Tests for the metrics package."""

import numpy as np

from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.metrics.calculator import calculate_strategy_metrics
from nvision.metrics.convergence import analyze_run_convergence
from nvision.models.observer import RunResult, StepSnapshot
from nvision.spectra.signal import SignalModel, TrueSignal
from nvision.spectra.spec import BasicParamSpec


class MockSignalModel(SignalModel):
    @property
    def spec(self):
        return BasicParamSpec(names=["x1"], bounds={"x1": (0, 1)})

    def parameter_names(self):
        return ["x1"]

    def compute(self, x, params):
        return 1.0

    def compute_from_params(self, x, params):
        return 1.0

    def compute_vectorized_samples(self, x, samples):
        return np.ones(len(samples))


def create_mock_run_result(final_error=0.01, final_uncertainty=0.005, converged_at=None):
    model = MockSignalModel()
    true_signal = TrueSignal(model=model, typed_parameters=(0.5,), bounds={"x1": (0, 1)})

    snapshots = []
    for i in range(10):
        # Create a belief that shows decreasing uncertainty
        uncert = 0.1 / (i + 1)
        est = 0.5 + (0.1 if i < 5 else 0.01)  # Simulated error

        param = GridParameter(
            name="x1",
            bounds=(0, 1),
            grid=np.linspace(0, 1, 10),
            posterior=np.ones(10) / 10,  # dummy
        )

        # Mock belief to return specific estimates/uncertainty
        class MockBelief(GridMarginalDistribution):
            def estimates(self):
                return {"x1": est}

            def uncertainty(self):
                return {"x1": uncert}

            def copy(self):
                return self

            @property
            def physical_param_bounds(self):
                return {"x1": (0, 1)}

        belief = MockBelief(model=model, parameters=[param])

        from nvision.models.observation import Observation

        obs = Observation(x=0.5, signal_value=1.0, noise_std=0.01)

        snapshots.append(StepSnapshot(obs=obs, belief=belief, true_signal=true_signal))

    # Overwrite final state if requested
    if final_error is not None or final_uncertainty is not None:
        # We'd need to mock the last snapshot's belief more carefully if we want to test exact values
        pass

    return RunResult(snapshots=snapshots, true_signal=true_signal)


def test_analyze_run_convergence():
    run = create_mock_run_result()
    conv_map = analyze_run_convergence(run, uncertainty_threshold=0.02)

    assert "x1" in conv_map
    assert conv_map["x1"].converged is True
    assert conv_map["x1"].convergence_step is not None
    assert conv_map["x1"].convergence_step < 10


def test_calculate_strategy_metrics():
    runs = [create_mock_run_result() for _ in range(5)]
    metrics = calculate_strategy_metrics("test_strat", runs, uncertainty_threshold=0.02)

    assert metrics.strategy_name == "test_strat"
    assert metrics.n_repeats == 5
    assert metrics.total_convergence_rate == 1.0
    assert "x1" in metrics.parameter_convergence_rates
    assert metrics.parameter_convergence_rates["x1"] == 1.0
