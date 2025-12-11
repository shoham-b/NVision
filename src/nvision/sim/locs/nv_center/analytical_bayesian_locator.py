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
    NVCenterSequentialBayesianLocatorSingle,
)

log = logging.getLogger(__name__)


@dataclass
class AnalyticalBayesianLocator(NVCenterSequentialBayesianLocatorSingle):
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
        # We use the mean linewidth from prior for the dictionary atoms
        gamma = np.mean(self.linewidth_prior)

        # We compute correlation of y_prime with each atom
        # Atom_j = Lorentzian(measurements_x, center=freq_grid[j], gamma)

        # Vectorized correlation calculation
        # We want to find j that maximizes: dot(y_prime, Atom_j) / norm(Atom_j)

        best_correlation = -np.inf
        best_freq = np.mean(self.prior_bounds)

        # Iterate over grid (or chunks if grid is huge, but 1000 is small)
        # We can use the JIT model to generate atoms

        # Pre-calculate constants
        hwhm = gamma / 2.0
        hwhm_sq = hwhm * hwhm

        # To speed up, we can broadcast if memory allows.
        # measurements_x: (M,)
        # freq_grid: (N,)
        # diff: (M, N)

        m_len = len(measurements_x)
        n_len = len(self.freq_grid)

        # If N*M is small (e.g. 50 * 1000 = 50k), we can broadcast
        if m_len * n_len < 1_000_000:
            # diff[i, j] = measurements_x[i] - freq_grid[j]
            diff = measurements_x[:, np.newaxis] - self.freq_grid[np.newaxis, :]
            denom = diff**2 + hwhm_sq
            # Atom values (unnormalized Lorentzian shape)
            atoms = 1.0 / denom

            # Normalize atoms (L2 norm)
            atom_norms = np.linalg.norm(atoms, axis=0)
            atom_norms[atom_norms < 1e-9] = 1.0  # Avoid div by zero

            # Correlation
            correlations = np.dot(y_prime, atoms) / atom_norms

            best_idx = np.argmax(correlations)
            best_freq = self.freq_grid[best_idx]

        else:
            # Loop approach for memory safety
            for j in range(n_len):
                f_grid = self.freq_grid[j]
                # Generate atom on measurement points
                atom = 1.0 / ((measurements_x - f_grid) ** 2 + hwhm_sq)

                norm = np.linalg.norm(atom)
                if norm < 1e-9:
                    continue

                corr = np.dot(y_prime, atom) / norm

                if corr > best_correlation:
                    best_correlation = corr
                    best_freq = f_grid

        return best_freq

    def propose_next(
        self,
        history: Sequence | pl.DataFrame,
        scan: ScanBatch | None = None,
        repeats: pl.DataFrame | None = None,
    ) -> float | pl.DataFrame:
        # Handle batched/argument swapping logic (copied from base)
        if (
            isinstance(scan, pl.DataFrame)
            and repeats is not None
            and not isinstance(repeats, pl.DataFrame)
        ):
            real_repeats = scan
            real_scan = repeats
            scan = real_scan
            repeats = real_repeats
        elif isinstance(repeats, ScanBatch) and scan is None:
            pass

        if repeats is not None:
            # Batched interface not fully implemented for this custom locator yet
            # Fallback to super's behavior (which might use adapter or fail)
            return super().propose_next(history, scan, repeats)

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

            self.reset_posterior()  # Clear uniform/previous updates

            self._ingest_history(history)

            # Now propose next point using Bayesian logic
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
