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
from nvision.spectra.drive import DriveSetting
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
    drive_freq_min_phys : float
        Physical domain minimum
    drive_freq_max_phys : float
        Physical domain maximum
    drift : DriftTrajectory | None
        This repeat's realization of ``noise.drift`` (see
        :func:`nvision.noises.drift.attach_drift_for_repeat`). Required whenever
        ``noise.drift`` is set.
    """

    true_signal: TrueSignal
    noise: CompositeNoise | None
    drive_freq_min_phys: float
    drive_freq_max_phys: float
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
        drive_freq_unit: float,
        rng: random.Random,
        n_shots: int = 1,
        shot_index: int | None = None,
        drive: DriveSetting | None = None,
    ) -> Observation:
        """Take a measurement at normalized position.

        Parameters
        ----------
        drive_freq_unit : float
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
        drive : DriveSetting | None
            Microwave drive to apply, for drive-controlled signals. ``None`` applies the true
            signal's ``baseline_drive``; giving a drive to a signal without drive control raises.
            The drive actually applied is recorded on the returned Observation.

        Returns
        -------
        Observation
            Measurement with noise applied (batch-aggregated when n_shots > 1)
        """
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")

        drive = self.resolve_drive(drive)

        # Denormalize to physical domain
        width = self.drive_freq_max_phys - self.drive_freq_min_phys
        x_physical = self.drive_freq_min_phys + drive_freq_unit * width

        if self.drift is not None or (self.noise is not None and self.noise.drift is not None):
            return self._measure_drifting(drive_freq_unit, x_physical, rng, n_shots, shot_index, drive)

        # Get true (clean) signal value
        signal_value = self.true_signal(x_physical) if drive is None else self.true_signal(x_physical, drive)

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
            drive_freq_unit=drive_freq_unit,
            ys=shots,
            prior_noise_std=noise_std,
            frequency_noise_model=frequency_noise_model,
            drive=drive,
        )

    def resolve_drive(self, drive: DriveSetting | None) -> DriveSetting | None:
        """The drive a measurement requested with ``drive`` is actually taken at.

        Drive-controlled signals: ``drive`` itself, or the baseline drive when the locator chose none.
        Other signals: always ``None`` (raises if a drive was requested).
        """
        # Duck-typed true signals (plain callables, MATLAB proxies) have no drive control.
        model = getattr(self.true_signal, "model", None)
        if getattr(model, "controls_drive", False):
            return drive if drive is not None else self.true_signal.baseline_drive
        if drive is not None:
            raise ValueError(f"This experiment's signal is not drive-controlled; got drive {drive!r}.")
        return None

    def _measure_drifting(
        self,
        drive_freq_unit: float,
        x_physical: float,
        rng: random.Random,
        n_shots: int,
        shot_index: int | None,
        drive: DriveSetting | None,
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

        model = self.true_signal.model if drive is None else self.true_signal.model_at(drive)
        base = self.true_signal.typed_parameters
        shots = np.empty(n_shots, dtype=np.float64)
        for i in range(n_shots):
            clean = float(model.compute(float(x_physical), self.drift.apply(base, shot_index + i)))
            if frequency_noise_model is not None:
                shots[i] = self.noise.over_frequency_noise.apply_scalar(x_physical, clean, rng)
            else:
                shots[i] = clean

        return aggregate_shots(
            drive_freq_unit=drive_freq_unit,
            ys=shots,
            prior_noise_std=noise_std,
            frequency_noise_model=frequency_noise_model,
            drive=drive,
        )

    @property
    def signal(self):
        """Physical-domain signal callable — for viz compatibility."""
        return self.true_signal

    @property
    def truth_positions(self) -> list[float]:
        """Ground truth peak positions extracted from TrueSignal parameters."""
        values = self.true_signal.parameter_values()
        return [value for name, value in values.items() if "center_freq" in name or "position" in name]

    def denormalize_drive_freq(self, drive_freq_unit: float) -> float:
        """Convert normalized x to physical domain."""
        return self.drive_freq_min_phys + drive_freq_unit * (self.drive_freq_max_phys - self.drive_freq_min_phys)
