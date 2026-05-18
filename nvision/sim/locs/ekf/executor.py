import numpy as np

from nvision.sim.locs.ekf.acquisition import BaseAcquisition
from nvision.sim.locs.ekf.filter import ekf_update


class ODMRActiveLearningExecutor:
    """Executor for EKF-based ODMR Active Learning."""

    def __init__(
        self,
        acquisition_strategy: BaseAcquisition,
        theta_initial: np.ndarray,
        P_initial: np.ndarray,  # noqa: N803
        R: float,  # noqa: N803
    ):
        """Initializes the executor.

        Args:
            acquisition_strategy: Strategy to select the next frequency.
            theta_initial: Initial state estimate (shape: 5,).
            P_initial: Initial covariance matrix (shape: 5x5).
            R: Measurement noise variance.
        """
        self.acquisition_strategy = acquisition_strategy
        self.theta_hat = theta_initial.copy()
        self.P = P_initial.copy()
        self.R = R

    def run_step(self, f_candidates: np.ndarray, current_y_measurement: float) -> float:
        """Runs a single step of the active learning loop.

        Routes the current state and covariance to the acquisition strategy to get the
        next frequency, and then updates the state using the EKF update function.

        Args:
            f_candidates: 1D array of frequency candidates to choose from.
            current_y_measurement: The measured photoluminescence at the LAST chosen frequency.
                                   (Note: In a real closed-loop, you select f, measure y, then update.
                                   Based on the prompt, we route P to get next f, then update state.
                                   This implies the measurement y is already obtained for the 'next f'.
                                   We'll assume the standard flow:
                                   1. Get f_next from acquisition.
                                   2. (Implicitly measure y at f_next, which is passed in here).
                                   3. Update EKF.)

        Returns:
            The selected frequency `f_next`.
        """
        # 1. Select next frequency based on current P and theta_hat
        f_next = self.acquisition_strategy.select_next_frequency(
            self.theta_hat, self.P, self.R, f_candidates
        )

        # 2. Update EKF state and covariance using the measurement at the selected frequency
        self.theta_hat, self.P = ekf_update(
            self.theta_hat, self.P, current_y_measurement, f_next, self.R
        )

        return f_next
