"""Experiment setup for core architecture.

Replaces the legacy ScanBatch + Experiment pattern with a native TrueSignal approach.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nvision.models.noise import CompositeNoise
from nvision.models.observation import DEFAULT_MEASUREMENT_NOISE_STD, Observation, aggregate_shots
from nvision.spectra.signal import TrueSignal

if TYPE_CHECKING:
    from nvision.noises.drift import DriftTrajectory


@dataclass
class CoreExperiment:
    """Experimental setup with TrueSignal and noise.

    This replaces the legacy Experiment+ScanBatch pattern with native core types.

    Attributes
    ----------
    true_signal : TrueSignal
        Ground truth signal to measure
    noise : CompositeNoise | None
        Noise model to apply to measurements
    x_min : float
        Physical domain minimum
    x_max : float
        Physical domain maximum
    drift : DriftTrajectory | None
        This repeat's realization of ``noise.drift`` (see
        :func:`nvision.noises.drift.attach_drift_for_repeat`). Required whenever
        ``noise.drift`` is set.
    """

    true_signal: TrueSignal
    noise: CompositeNoise | None
    x_min: float
    x_max: float
    drift: DriftTrajectory | None = None

    def frequency_noise_model(self) -> tuple[dict[str, object], ...] | None:
        """Structured description of over-frequency noise for this experiment.

        Locators can use this to choose likelihood families (Gaussian vs Poisson
        vs mixtures) without needing to introspect concrete noise classes.
        """
        if self.noise is None or self.noise.over_frequency_noise is None:
            return None
        spec_getter = getattr(self.noise.over_frequency_noise, "likelihood_spec", None)
        if callable(spec_getter):
            return spec_getter()
        return None

    def measure(
        self,
        x_normalized: float,
        rng: random.Random,
        n_shots: int = 1,
        shot_index: int | None = None,
    ) -> Observation:
        """Take a measurement at normalized position.

        Parameters
        ----------
        x_normalized : float
            Position in [0, 1] normalized space
        rng : random.Random
            Random number generator for noise
        n_shots : int
            Number of repeated shots to take at this position and average into a
            single sufficient-statistic Observation (batch mean ȳ with precision
            σ/√n_shots plus the within-batch variance). Defaults to 1, which is
            value-identical to a single measurement.
        shot_index : int | None
            Shots already taken in this run (0 for the first). Required when the
            experiment drifts, since the true signal then depends on time; shot ``i``
            of the batch is taken at ``shot_index + i``. Ignored otherwise.

        Returns
        -------
        Observation
            Measurement with noise applied (batch-aggregated when n_shots > 1)
        """
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")

        # Denormalize to physical domain
        width = self.x_max - self.x_min
        x_physical = self.x_min + x_normalized * width

        if self.drift is not None or (self.noise is not None and self.noise.drift is not None):
            return self._measure_drifting(x_normalized, x_physical, rng, n_shots, shot_index)

        # Get true (clean) signal value
        signal_value = self.true_signal(x_physical)

        # Determine noise metadata
        noise_std = DEFAULT_MEASUREMENT_NOISE_STD  # default for no-noise case (aligned with Observation)
        frequency_noise_model = None
        if self.noise is not None:
            noise_std = self.noise.estimated_noise_std()
            if self.noise.over_frequency_noise is not None:
                frequency_noise_model = self.frequency_noise_model()

        # Draw n_shots noisy realizations at x. With no over-frequency noise the
        # signal is deterministic, so all shots coincide (zero empirical variance);
        # aggregate_shots then falls back to the prior noise_std.
        if frequency_noise_model is not None:
            shots = np.empty(n_shots, dtype=np.float64)
            for i in range(n_shots):
                shots[i] = self.noise.over_frequency_noise.apply_scalar(x_physical, signal_value, rng)
        else:
            shots = np.full(n_shots, float(signal_value), dtype=np.float64)

        # Aggregate into a single sufficient-statistic Observation (normalized space)
        return aggregate_shots(
            x=x_normalized,
            ys=shots,
            prior_noise_std=noise_std,
            frequency_noise_model=frequency_noise_model,
        )

    def _measure_drifting(
        self,
        x_normalized: float,
        x_physical: float,
        rng: random.Random,
        n_shots: int,
        shot_index: int | None,
    ) -> Observation:
        if self.drift is None:
            raise ValueError(
                "Experiment noise has drift but no drift realization is attached; "
                "call nvision.noises.drift.attach_drift_for_repeat first."
            )
        if shot_index is None:
            raise ValueError("A drifting experiment needs shot_index: the true signal depends on when it is measured.")

        noise_std = DEFAULT_MEASUREMENT_NOISE_STD
        frequency_noise_model = None
        if self.noise is not None:
            noise_std = self.noise.estimated_noise_std()
            if self.noise.over_frequency_noise is not None:
                frequency_noise_model = self.frequency_noise_model()

        model = self.true_signal.model
        base = self.true_signal.typed_parameters
        shots = np.empty(n_shots, dtype=np.float64)
        for i in range(n_shots):
            clean = float(model.compute(float(x_physical), self.drift.apply(base, shot_index + i)))
            if frequency_noise_model is not None:
                shots[i] = self.noise.over_frequency_noise.apply_scalar(x_physical, clean, rng)
            else:
                shots[i] = clean

        return aggregate_shots(
            x=x_normalized,
            ys=shots,
            prior_noise_std=noise_std,
            frequency_noise_model=frequency_noise_model,
        )

    @property
    def signal(self):
        """Physical-domain signal callable — for viz compatibility."""
        return self.true_signal

    @property
    def truth_positions(self) -> list[float]:
        """Ground truth peak positions extracted from TrueSignal parameters."""
        values = self.true_signal.parameter_values()
        return [value for name, value in values.items() if "frequency" in name or "position" in name]

    def denormalize_x(self, x_normalized: float) -> float:
        """Convert normalized x to physical domain."""
        return self.x_min + x_normalized * (self.x_max - self.x_min)
