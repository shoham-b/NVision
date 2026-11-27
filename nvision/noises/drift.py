"""Time-dependent drift of the true NV resonance during a run.

Two physical drivers, each a :class:`DriftProcess` that starts at 0 when the run starts:

* sample temperature (kelvin) shifts the zero-field ``zero_field_freq`` by
  :data:`NV_D_TEMPERATURE_COEFF_HZ_PER_K`, and a permanent magnet's field (so
  ``zeeman_shift``) by ``magnet_tempco_per_k`` of its starting value;
* the field along the NV axis (milligauss) shifts ``zeeman_shift`` by
  :data:`NV_GYROMAGNETIC_HZ_PER_MG` (``zeeman_shift`` is the center-to-dip distance, gamma*B).

A signal may also carry its own line jitter (``TrueSignal.line_walk_step_phys``): a Gaussian random walk of
``zeeman_shift`` with that standard deviation (Hz) per shot, on top of any drift scenario.

Time advances per shot. A realization (:class:`DriftTrajectory`) is a pure function of
the shot index, so every strategy measuring the same repeat sees the identical path no
matter how many shots it takes or in what order it asks.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nvision.models.experiment import CoreExperiment

NV_D_TEMPERATURE_COEFF_HZ_PER_K: float = -74.2e3
NV_GYROMAGNETIC_HZ_PER_MG: float = 2802.5

# Stochastic parts are generated in fixed-size chunks, in order, so the value at a given
# shot never depends on how far ahead some earlier caller happened to look.
_CHUNK = 4096


@dataclass(frozen=True, slots=True)
class DriftProcess:
    """A scalar process in its own unit (K or mG), equal to 0 at the first shot.

    Sum of: an exponential warm-up to ``warmup_amplitude``; mean-reverting random wander
    (Ornstein-Uhlenbeck, stationary std ``ou_sigma``); a sinusoid; and Poisson-timed
    zero-mean Gaussian steps.
    """

    warmup_amplitude: float = 0.0
    warmup_tau_s: float = 1.0
    ou_sigma: float = 0.0
    ou_tau_s: float = 1.0
    periodic_amplitude: float = 0.0
    periodic_period_s: float = 1.0
    step_rate_per_s: float = 0.0
    step_sigma: float = 0.0

    def __post_init__(self) -> None:
        for name in ("warmup_tau_s", "ou_tau_s", "periodic_period_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"DriftProcess.{name} must be > 0, got {getattr(self, name)}")
        for name in ("ou_sigma", "step_rate_per_s", "step_sigma"):
            if getattr(self, name) < 0:
                raise ValueError(f"DriftProcess.{name} must be >= 0, got {getattr(self, name)}")

    def is_zero(self) -> bool:
        return (
            self.warmup_amplitude == 0.0
            and self.ou_sigma == 0.0
            and self.periodic_amplitude == 0.0
            and (self.step_rate_per_s == 0.0 or self.step_sigma == 0.0)
        )


@dataclass(frozen=True, slots=True)
class DriftSpec:
    """A named drift scenario. ``label`` is part of the noise name, so it keys the result cache."""

    label: str
    shot_duration_s: float
    temperature_k: DriftProcess = DriftProcess()
    field_mg: DriftProcess = DriftProcess()
    magnet_tempco_per_k: float = 0.0
    # Std (Hz) of the Gaussian random-walk step ``zeeman_shift`` takes between consecutive shots.
    zeeman_walk_step_hz: float = 0.0

    def __post_init__(self) -> None:
        if self.shot_duration_s <= 0:
            raise ValueError(f"DriftSpec.shot_duration_s must be > 0, got {self.shot_duration_s}")
        if not (self.zeeman_walk_step_hz >= 0 and math.isfinite(self.zeeman_walk_step_hz)):
            raise ValueError(f"DriftSpec.zeeman_walk_step_hz must be finite and >= 0, got {self.zeeman_walk_step_hz}")

    @property
    def moves_center(self) -> bool:
        return not self.temperature_k.is_zero()

    @property
    def moves_zeeman_shift(self) -> bool:
        return (
            self.zeeman_walk_step_hz > 0.0
            or not self.field_mg.is_zero()
            or (self.magnet_tempco_per_k != 0.0 and self.moves_center)
        )

    def realize(self, seed: int, base_zeeman_shift: float | None) -> DriftTrajectory:
        return DriftTrajectory(self, seed, base_zeeman_shift)


class _ProcessPath:
    def __init__(self, process: DriftProcess, dt: float, rng: np.random.Generator) -> None:
        self._p = process
        self._dt = dt
        self._rng = rng
        self._phase = float(rng.uniform(0.0, 2.0 * math.pi)) if process.periodic_amplitude != 0.0 else 0.0
        self._ou_a = math.exp(-dt / process.ou_tau_s)
        self._ou_innov = process.ou_sigma * math.sqrt(1.0 - self._ou_a**2)
        self._stoch = np.zeros(1, dtype=np.float64)  # stochastic part at shot 0 is 0
        self._ou_last = 0.0
        self._steps_last = 0.0

    def _extend_to(self, n: int) -> None:
        p = self._p
        while self._stoch.size < n:
            ou_last, steps_last = self._ou_last, self._steps_last
            ou = np.empty(_CHUNK)
            steps = np.empty(_CHUNK)
            xi = self._rng.standard_normal(_CHUNK)
            counts = self._rng.poisson(p.step_rate_per_s * self._dt, _CHUNK)
            jumps = self._rng.standard_normal(_CHUNK) * p.step_sigma * np.sqrt(counts)
            for i in range(_CHUNK):
                ou_last = self._ou_a * ou_last + self._ou_innov * xi[i]
                steps_last += jumps[i]
                ou[i] = ou_last
                steps[i] = steps_last
            self._ou_last, self._steps_last = ou_last, steps_last
            self._stoch = np.concatenate([self._stoch, ou + steps])

    def value(self, shot_index: int) -> float:
        if shot_index < 0:
            raise ValueError(f"shot_index must be >= 0, got {shot_index}")
        p = self._p
        t = shot_index * self._dt
        v = 0.0
        if p.warmup_amplitude != 0.0:
            v += p.warmup_amplitude * (1.0 - math.exp(-t / p.warmup_tau_s))
        if p.periodic_amplitude != 0.0:
            angle = 2.0 * math.pi * t / p.periodic_period_s + self._phase
            v += p.periodic_amplitude * (math.sin(angle) - math.sin(self._phase))
        if p.ou_sigma != 0.0 or (p.step_rate_per_s != 0.0 and p.step_sigma != 0.0):
            self._extend_to(shot_index + 1)
            v += float(self._stoch[shot_index])
        return v


class _RandomWalkPath:
    """Gaussian random walk equal to 0 at the first shot, generated in order in fixed-size chunks."""

    def __init__(self, step_std: float, rng: np.random.Generator) -> None:
        self._step_std = step_std
        self._rng = rng
        self._path = np.zeros(1, dtype=np.float64)  # shape: (n_shots_generated,), Hz

    def value(self, shot_index: int) -> float:
        if shot_index < 0:
            raise ValueError(f"shot_index must be >= 0, got {shot_index}")
        while self._path.size <= shot_index:
            steps = self._rng.standard_normal(_CHUNK) * self._step_std
            self._path = np.concatenate([self._path, self._path[-1] + np.cumsum(steps)])
        return float(self._path[shot_index])


class DriftTrajectory:
    """One seeded realization of a :class:`DriftSpec`: offsets of the true parameters per shot."""

    def __init__(self, spec: DriftSpec, seed: int, base_zeeman_shift: float | None) -> None:
        if spec.magnet_tempco_per_k != 0.0 and spec.moves_center and base_zeeman_shift is None:
            raise ValueError(
                f"Drift {spec.label!r} couples the magnet to temperature but the signal has no zeeman_shift"
            )
        self.spec = spec
        self._base_zeeman_shift = base_zeeman_shift
        temp_seq, field_seq, walk_seq = np.random.SeedSequence(int(seed)).spawn(3)
        dt = spec.shot_duration_s
        self._temperature = _ProcessPath(spec.temperature_k, dt, np.random.default_rng(temp_seq))
        self._field = _ProcessPath(spec.field_mg, dt, np.random.default_rng(field_seq))
        self._walk = (
            _RandomWalkPath(spec.zeeman_walk_step_hz, np.random.default_rng(walk_seq))
            if spec.zeeman_walk_step_hz > 0.0
            else None
        )

    def center_offset_hz(self, shot_index: int) -> float:
        return NV_D_TEMPERATURE_COEFF_HZ_PER_K * self._temperature.value(shot_index)

    def zeeman_shift_offset_hz(self, shot_index: int) -> float:
        offset = NV_GYROMAGNETIC_HZ_PER_MG * self._field.value(shot_index)
        if self._walk is not None:
            offset += self._walk.value(shot_index)
        if self.spec.magnet_tempco_per_k != 0.0 and self._base_zeeman_shift is not None:
            offset += self.spec.magnet_tempco_per_k * self._temperature.value(shot_index) * self._base_zeeman_shift
        return offset

    def apply(self, typed_parameters: Any, shot_index: int) -> Any:
        """Return ``typed_parameters`` as they truly are at ``shot_index``."""
        changes: dict[str, float] = {}
        if self.spec.moves_center:
            if not hasattr(typed_parameters, "zero_field_freq"):
                raise ValueError(f"Drift {self.spec.label!r} moves the center but the signal has no zero_field_freq")
            changes["zero_field_freq"] = float(typed_parameters.zero_field_freq) + self.center_offset_hz(shot_index)
        if self.spec.moves_zeeman_shift:
            if not hasattr(typed_parameters, "zeeman_shift"):
                raise ValueError(f"Drift {self.spec.label!r} moves the splitting but the signal has no zeeman_shift")
            # The dip separation is |gamma*B|: a field swinging through zero reopens the line splitting.
            changes["zeeman_shift"] = abs(
                float(typed_parameters.zeeman_shift) + self.zeeman_shift_offset_hz(shot_index)
            )
        return dataclasses.replace(typed_parameters, **changes) if changes else typed_parameters

    def truth_traces(self, typed_parameters: Any, n_steps: int, n_shots: int = 1) -> dict[str, np.ndarray]:
        """True value of every parameter that moves, at each measurement of a run.

        Each array has shape (n_steps,), Hz: entry ``i`` is the truth at the last shot of measurement ``i``
        (``n_shots`` shots per measurement), i.e. where the line is when the estimate after ``i + 1`` measurements
        is made. Parameters that do not move are left out.
        """
        applied = [self.apply(typed_parameters, (i + 1) * n_shots - 1) for i in range(n_steps)]
        moved = [
            name
            for name, moves in (
                ("zero_field_freq", self.spec.moves_center),
                ("zeeman_shift", self.spec.moves_zeeman_shift),
            )
            if moves
        ]
        return {name: np.array([float(getattr(a, name)) for a in applied], dtype=np.float64) for name in moved}

    def truth_summary(self, typed_parameters: Any, n_shots: int, prefix: str = "drift_") -> dict[str, float]:
        """True center/splitting at the last shot and averaged over shots ``0..n_shots-1``."""
        if n_shots <= 0:
            return {}
        out: dict[str, float] = {f"{prefix}shots": float(n_shots)}
        for name, moves, offset in (
            ("zero_field_freq", self.spec.moves_center, self.center_offset_hz),
            ("zeeman_shift", self.spec.moves_zeeman_shift, self.zeeman_shift_offset_hz),
        ):
            if not moves or not hasattr(typed_parameters, name):
                continue
            base = float(getattr(typed_parameters, name))
            path = np.array([base + offset(i) for i in range(n_shots)])
            if name == "zeeman_shift":
                path = np.abs(path)
            out[f"{prefix}true_{name}_start"] = base
            out[f"{prefix}true_{name}_end"] = float(path[-1])
            out[f"{prefix}true_{name}_mean"] = float(path.mean())
        return out


def attach_drift_for_repeat(
    experiment: CoreExperiment, seed: int, generator_name: str, repeat_idx: int
) -> CoreExperiment:
    """Give ``experiment`` its drift realization for this repeat.

    Covers the noise's drift scenario and the signal's own line jitter (``line_walk_step_phys``); a no-op when it
    has neither, or if already attached.

    Seeded from (seed, generator, drift label, repeat) — not strategy or Gaussian sigma — so
    every strategy and noise level in a repeat faces the same drift path.
    """
    from nvision.runner.repeat_keys import measurement_repeat_key, repeat_seed_int

    scenario = None if experiment.noise is None else experiment.noise.drift
    walk_step_hz = float(getattr(experiment.true_signal, "line_walk_step_phys", 0.0))
    if experiment.drift is not None or (scenario is None and walk_step_hz == 0.0):
        return experiment
    spec = scenario if scenario is not None else DriftSpec(label="line-walk", shot_duration_s=1.0)
    key = measurement_repeat_key(seed, generator_name, "drift", spec.label, repeat_idx)
    base_zeeman_shift = getattr(experiment.true_signal.typed_parameters, "zeeman_shift", None)
    if walk_step_hz > 0.0:
        if base_zeeman_shift is None:
            raise ValueError(f"{generator_name!r} has line jitter but its signal has no zeeman_shift to move")
        spec = dataclasses.replace(spec, zeeman_walk_step_hz=walk_step_hz)
    trajectory = spec.realize(repeat_seed_int(key), None if base_zeeman_shift is None else float(base_zeeman_shift))
    return dataclasses.replace(experiment, drift=trajectory)
