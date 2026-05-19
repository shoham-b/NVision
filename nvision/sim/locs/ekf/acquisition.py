from abc import ABC, abstractmethod

import numpy as np
from numba import njit

from nvision.sim.locs.ekf.models import jacobian_mu


@njit
def _compute_p_next_det(
    f_candidates: np.ndarray,
    theta_hat: np.ndarray,
    P: np.ndarray,  # noqa: N803
    R: float,  # noqa: N803
) -> int:
    """Computes det(P_next) for each candidate and returns the index of the minimum.

    Args:
        f_candidates: 1D array of frequency candidates.
        theta_hat: Current state estimate (shape: 5,).
        P: Current covariance matrix (shape: 5x5).  # noqa: N803
        R: Measurement noise variance.  # noqa: N803

    Returns:
        Index of the frequency candidate that minimizes the determinant.
    """
    best_idx = 0
    min_det = np.inf

    for i in range(len(f_candidates)):
        f = f_candidates[i]
        H = jacobian_mu(f, theta_hat)  # noqa: N806

        S = (H @ P @ H.T)[0, 0] + R  # noqa: N806
        K = (P @ H.T) / S  # noqa: N806
        P_next = P - K @ H @ P  # noqa: N806

        det = np.linalg.det(P_next)

        if det < min_det:
            min_det = det
            best_idx = i

    return best_idx


@njit
def _compute_p_next_trace(
    f_candidates: np.ndarray,
    theta_hat: np.ndarray,
    P: np.ndarray,  # noqa: N803
    R: float,  # noqa: N803
) -> int:
    """Computes trace(P_next) for each candidate and returns the index of the minimum.

    Args:
        f_candidates: 1D array of frequency candidates.
        theta_hat: Current state estimate (shape: 5,).
        P: Current covariance matrix (shape: 5x5).  # noqa: N803
        R: Measurement noise variance.  # noqa: N803

    Returns:
        Index of the frequency candidate that minimizes the trace.
    """
    best_idx = 0
    min_trace = np.inf

    for i in range(len(f_candidates)):
        f = f_candidates[i]
        H = jacobian_mu(f, theta_hat)  # noqa: N806

        S = (H @ P @ H.T)[0, 0] + R  # noqa: N806
        K = (P @ H.T) / S  # noqa: N806
        P_next = P - K @ H @ P  # noqa: N806

        trace = np.trace(P_next)

        if trace < min_trace:
            min_trace = trace
            best_idx = i

    return best_idx


class BaseAcquisition(ABC):
    """Abstract base class for EKF optimal experiment design acquisition strategies."""

    @abstractmethod
    def select_next_frequency(
        self,
        theta_hat: np.ndarray,
        P: np.ndarray,  # noqa: N803
        R: float,  # noqa: N803
        f_candidates: np.ndarray,
    ) -> float:
        """Selects the next frequency to measure.

        Args:
            theta_hat: Current state estimate (shape: 5,).
            P: Current covariance matrix (shape: 5x5).
            R: Measurement noise variance.
            f_candidates: 1D array of frequency candidates.

        Returns:
            The chosen frequency.
        """
        raise NotImplementedError


class DOptimalAcquisition(BaseAcquisition):
    """D-Optimal acquisition strategy (minimizes determinant of covariance)."""

    def select_next_frequency(
        self,
        theta_hat: np.ndarray,
        P: np.ndarray,  # noqa: N803
        R: float,  # noqa: N803
        f_candidates: np.ndarray,
    ) -> float:
        idx = _compute_p_next_det(f_candidates, theta_hat, P, R)
        return float(f_candidates[idx])


class AOptimalAcquisition(BaseAcquisition):
    """A-Optimal acquisition strategy (minimizes trace of covariance)."""

    def select_next_frequency(
        self,
        theta_hat: np.ndarray,
        P: np.ndarray,  # noqa: N803
        R: float,  # noqa: N803
        f_candidates: np.ndarray,
    ) -> float:
        idx = _compute_p_next_trace(f_candidates, theta_hat, P, R)
        return float(f_candidates[idx])
