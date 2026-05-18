import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nvision.sim.locs.ekf.acquisition import AOptimalAcquisition, DOptimalAcquisition
from nvision.sim.locs.ekf.executor import ODMRActiveLearningExecutor
from nvision.sim.locs.ekf.models import jacobian_mu


# Helper to generate valid positive semi-definite covariance matrices
@st.composite
def psd_matrices(draw):
    # Draw a 5x5 matrix
    A = draw(arrays(np.float64, (5, 5), elements=st.floats(-10.0, 10.0)))  # noqa: N806
    # A @ A.T is symmetric positive semi-definite
    return A @ A.T + np.eye(5) * 1e-3  # Add small ridge for strict positive definiteness


# Strategy for theta = [f_B, delta_f_HF, a, Omega, k_NP]
# Typical values: f_B ~ 2870, delta_f_HF ~ 2.1, a ~ 0.05, Omega ~ 5, k_NP ~ 1
theta_strategy = arrays(
    np.float64,
    (5,),
    elements=st.floats(0.01, 10.0),
).map(lambda t: t + np.array([2870.0, 2.1, 0.05, 5.0, 1.0]))


@given(
    theta=theta_strategy,
    P=psd_matrices(),
    f_candidates=arrays(np.float64, (10,), elements=st.floats(2800.0, 2950.0)),
    R=st.floats(1e-4, 1e-1),
)
@settings(max_examples=10, deadline=None)
def test_acquisition_strategies(theta, P, f_candidates, R):  # noqa: N803
    # D-Optimal
    d_acq = DOptimalAcquisition()
    f_next_d = d_acq.select_next_frequency(theta, P, R, f_candidates)

    assert isinstance(f_next_d, float)
    assert f_next_d in f_candidates

    # Check metric (det)
    H_d = jacobian_mu(f_next_d, theta)  # noqa: N806
    S_d = (H_d @ P @ H_d.T)[0, 0] + R  # noqa: N806
    K_d = (P @ H_d.T) / S_d  # noqa: N806
    P_next_d = P - K_d @ H_d @ P  # noqa: N806
    assert np.linalg.det(P_next_d) <= np.linalg.det(P) + 1e-9  # Allow small num. error

    # A-Optimal
    a_acq = AOptimalAcquisition()
    f_next_a = a_acq.select_next_frequency(theta, P, R, f_candidates)

    assert isinstance(f_next_a, float)
    assert f_next_a in f_candidates

    # Check metric (trace)
    H_a = jacobian_mu(f_next_a, theta)  # noqa: N806
    S_a = (H_a @ P @ H_a.T)[0, 0] + R  # noqa: N806
    K_a = (P @ H_a.T) / S_a  # noqa: N806
    P_next_a = P - K_a @ H_a @ P  # noqa: N806
    assert np.trace(P_next_a) <= np.trace(P) + 1e-9


@given(
    theta=theta_strategy,
    P=psd_matrices(),
    f_candidates=arrays(np.float64, (10,), elements=st.floats(2800.0, 2950.0)),
    R=st.floats(1e-4, 1e-1),
    y=st.floats(0.8, 1.2),
)
@settings(max_examples=10, deadline=None)
def test_executor_execution(theta, P, f_candidates, R, y):  # noqa: N803
    # Test with D-Optimal
    d_acq = DOptimalAcquisition()
    executor_d = ODMRActiveLearningExecutor(d_acq, theta, P, R)
    f_next_d = executor_d.run_step(f_candidates, y)

    assert isinstance(f_next_d, float)
    assert f_next_d in f_candidates
    # Check state was updated
    assert executor_d.theta_hat is not theta  # ensure copy or new object

    # Test with A-Optimal
    a_acq = AOptimalAcquisition()
    executor_a = ODMRActiveLearningExecutor(a_acq, theta, P, R)
    f_next_a = executor_a.run_step(f_candidates, y)

    assert isinstance(f_next_a, float)
    assert f_next_a in f_candidates
    assert executor_a.theta_hat is not theta
