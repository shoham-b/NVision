"""
Analytical Bayesian Locator using Compressed Sensing for initialization.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.nv_center.sequential_bayesian_locator import (
    NVCenterSequentialBayesianLocator,
)

log = logging.getLogger(__name__)


@dataclass
class AnalyticalBayesianLocator(NVCenterSequentialBayesianLocator):
    """
    Analytical Bayesian Locator that uses Compressed Sensing (OMP) for initialization.

    Phase 1 (Warmup): Randomly samples frequencies (Compressed Sensing measurements).
    Phase 2 (Transition): Uses Orthogonal Matching Pursuit (OMP) to estimate the
                          signal from warmup measurements and initializes the
                          Bayesian posterior centered on this estimate.
    Phase 3 (Adaptive): Continues with Sequential Bayesian Experiment Design (EIG).
    """

    def _omp_estimate(self, history: pl.DataFrame) -> float:
        """
        Estimate frequency using Orthogonal Matching Pursuit (OMP) concept.

        Since we expect a single peak (sparse in frequency domain), this is
        simplified to finding the dictionary atom (Lorentzian on grid) that
        best matches the observed random measurements.
        """
        measurements_x = history["x"].to_numpy()
        measurements_y = history["signal_values"].to_numpy()

        # Estimate background as the maximum observed value (dip signal)
        bg_est = np.max(measurements_y)

        # Invert signal to get positive peaks: signal = bg - measurement
        # We assume Lorentzian dip.
        y_prime = bg_est - measurements_y

        # Normalize to avoid amplitude scaling issues during correlation
        if np.max(y_prime) > 1e-9:
            y_prime = y_prime / np.max(y_prime)

        # Create Dictionary
        # Atoms are Lorentzians centered at each grid point
        # Prepare data for JIT
        gamma = float(np.mean(self.linewidth_prior))

        # Use JIT OMP Correlation
        # self.freq_grid is already initialized in reset_posterior() which is called in __post_init__
        # but check if available
        if not hasattr(self, "freq_grid") or self.freq_grid is None:
            self.freq_grid = np.linspace(self.prior_bounds[0], self.prior_bounds[1], self.grid_resolution)

        from nvision.sim.locs.nv_center._jit_kernels import _omp_correlation_jit

        best_freq = _omp_correlation_jit(y_prime, measurements_x, self.freq_grid, gamma)

        return best_freq

    def propose_next(
        self,
        history: Sequence | pl.DataFrame,
        scan: ScanBatch | None = None,
        repeats: pl.DataFrame | None = None,
    ) -> float | pl.DataFrame:
        if not isinstance(history, pl.DataFrame):
            history = pl.DataFrame(history) if history else pl.DataFrame()

        hist_len = history.height
        self._rescale_priors_if_needed(scan)

        # Phase 1: Warmup (Compressed Sensing / Random Sampling)
        if hist_len < self.n_warmup:
            # Ingest history to keep track (though we don't update posterior yet)
            self._ingest_history(history)

            # Return a random frequency within bounds
            return np.random.uniform(self.prior_bounds[0], self.prior_bounds[1])

        # Phase 2: Transition (OMP Initialization)
        # We detect transition by checking if posterior is still uniform (max_prob is small)
        # or if we haven't "really" started the Bayesian part.
        # But `_ingest_history` updates posterior every step in the base class.
        # So we need to prevent `_ingest_history` from updating posterior during warmup?

        # Actually, the base class `propose_next` calls `_ingest_history`.
        # We are overriding `propose_next`.

        # If we are exactly at n_warmup, we perform the OMP initialization.
        if hist_len == self.n_warmup:
            # We haven't ingested the latest history fully into the posterior
            # in a "Bayesian" way if we were just doing random sampling.
            # But `_ingest_history` in base class DOES update posterior.

            # Strategy:
            # 1. Reset posterior to clear any updates from random samples
            #    (if we want to start fresh with OMP result).
            #    Or, we can just use the OMP result to *set* the posterior.

            self.reset_posterior()

            # Use OMP detection to initialize posterior
            est_freq = self._omp_estimate(history)

            # Center posterior around OMP estimate (Gaussian initialization)
            # Use a width related to expected linewidth or search range
            init_sigma = float(np.mean(self.linewidth_prior)) * 2.0

            diff = self.freq_grid - est_freq
            self.freq_posterior = np.exp(-0.5 * (diff / init_sigma) ** 2)

            # Avoid zero probability elsewhere (add small uniform background)
            self.freq_posterior += 1e-6
            self.freq_posterior /= np.sum(self.freq_posterior)

            # Note: We do NOT ingest the history again for the posterior update
            # because we used it for OMP (double counting).
            # Or we assume OMP gives the prior and we *should* ingest?
            # If we treat OMP as "finding the prior", then utilizing the data again for likelihood
            # update is double counting.
            # However, standard practice in this specific locator (from context of project/other tests)
            # However, standard practice in this specific locator (from context of project/other tests)
            # often implies using the data to initialize.
            # Let's trust the "Initialize the Bayesian posterior centered on this estimate" description.

            # Clear measurement history so that _ingest_history (called by super)
            # sees the warmup points as 'new' and updates the posterior with their likelihoods.
            # This combines the OMP-based prior with the actual data constraints.
            self.measurement_history.clear()

            # Update current estimates based on this new posterior
            self.current_estimates["frequency"] = est_freq

            return super().propose_next(history, scan, repeats)

        # Phase 3: Adaptive (Standard Bayesian)
        # Just call super, but we need to make sure we don't double-ingest or mess up state.
        # super().propose_next() calls _ingest_history.

        # If we call super().propose_next(), it will:
        # 1. _rescale_priors (ok)
        # 2. _ingest_history (ok, it filters new measurements)
        # 3. Check warmup (we are past it)
        # 4. Optimize acquisition (EIG)

        return super().propose_next(history, scan, repeats)

    def _ingest_history(self, history: Sequence | pl.DataFrame) -> None:
        """
        Override to control when posterior updates happen.
        During warmup (random sampling), we might NOT want to update posterior
        step-by-step if we plan to reset it at step n_warmup.
        However, keeping it updated doesn't hurt, as we reset it anyway.
        """
        super()._ingest_history(history)
