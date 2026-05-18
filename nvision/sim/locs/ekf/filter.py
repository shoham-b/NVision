import numpy as np
from numba import njit

from nvision.sim.locs.ekf.models import jacobian_mu, mu


@njit
def ekf_update(theta_hat: np.ndarray, P: np.ndarray, y: float, f: float, R: float) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
    """Extended Kalman Filter update step.

    Args:
        theta_hat: Current state estimate (shape: 5,).
        P: Current covariance matrix (shape: 5x5).  # noqa: N803
        y: Measured PL value.
        f: Microwave frequency in MHz.
        R: Measurement noise variance.  # noqa: N803

    Returns:
        theta_next: Updated state estimate (shape: 5,).
        P_next: Updated covariance matrix (shape: 5x5).  # noqa: N806
    """
    # Prediction
    y_pred = mu(f, theta_hat)
    H = jacobian_mu(f, theta_hat)  # Shape (1, 5)  # noqa: N806

    # Innovation
    innovation = y - y_pred

    # Innovation covariance
    # S = H @ P @ H.T + R  where H is 1x5, P is 5x5. H.T is 5x1
    # Result is a 1x1 matrix, extract scalar
    S = (H @ P @ H.T)[0, 0] + R  # noqa: N806

    # Kalman gain
    # K = P @ H.T @ inv(S)
    # Shape: (5, 5) @ (5, 1) = (5, 1) -> flatten to (5,)
    K = (P @ H.T) / S  # noqa: N806
    K_flat = K.flatten()  # noqa: N806

    # State update
    theta_next = theta_hat + K_flat * innovation

    # Covariance update
    # P_next = P - K @ H @ P (Joseph form is better for numerical stability if needed, but basic form works here)
    P_next = P - K @ H @ P  # noqa: N806

    return theta_next, P_next
