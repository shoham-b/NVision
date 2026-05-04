"""Expected Information Gain (EIG) Bayesian acquisition locator."""

from __future__ import annotations

import abc
import math
import os
import time

import numpy as np
import polars as pl
from numba import njit, prange

from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator





class SequentialBayesianExperimentDesignLocator(SequentialBayesianLocator):
    """Sequential Bayesian Experiment Design acquisition.

    Uses exact Expected Information Gain (posterior Shannon entropy reduction).
    """

    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        noise_std: float = 0.02,
        n_candidates: int = 200,
        n_draws: int = 100,
    ) -> None:
        super().__init__(
            belief,
            max_steps,
            convergence_threshold,
            scan_param,
            initial_sweep_steps=initial_sweep_steps,
            noise_std=noise_std,
        )
        self.n_candidates = int(n_candidates)
        self.n_draws = int(n_draws)

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds=None,
        initial_sweep_steps: int | None = None,
        noise_std: float | None = None,
        n_candidates: int = 200,
        n_draws: int = 100,
        **grid_config,
    ):
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            noise_std=noise_std,
            n_candidates=n_candidates,
            n_draws=n_draws,
        )

    def _acquire(self) -> float:
        debug_timing = os.environ.get("DEBUG_SBED_TIMING") == "1"

        if debug_timing:
            t_start = time.perf_counter()

        candidates = self._generate_candidates(self.n_candidates)

        if debug_timing:
            t_cand = time.perf_counter()

        candidates_arr = np.asarray(candidates)
        noise_std = self.belief.last_obs.noise_std if self.belief.last_obs is not None else self.noise_std

        if debug_timing:
            t0 = time.perf_counter()

        utilities = self.belief.expected_information_gain(candidates_arr, noise_std)

        if debug_timing:
            time_eig = time.perf_counter() - t0

        best_idx = int(pl.Series(utilities).arg_max())

        if debug_timing:
            total = time.perf_counter() - t_start
            print("\nSBED Timing Breakdown:")
            print(f"  Generate Candidates: {t_cand - t_start:.4f}s")
            print(f"  EIG Calc:            {time_eig:.4f}s")
            print(f"  Total _acquire:      {total:.4f}s")

        return float(candidates[best_idx])

