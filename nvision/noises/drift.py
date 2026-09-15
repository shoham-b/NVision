"""Time-dependent drift of the true NV resonance during a run.

Two physical drivers, each a :class:`DriftProcess` that starts at 0 when the run starts:

* sample temperature (kelvin) shifts the zero-field center ``frequency`` by
  :data:`NV_D_TEMPERATURE_COEFF_HZ_PER_K`, and a permanent magnet's field (so
  ``zeeman_split``) by ``magnet_tempco_per_k`` of its starting value;
* the field along the NV axis (milligauss) shifts ``zeeman_split`` by
  :data:`NV_GYROMAGNETIC_HZ_PER_MG` (``zeeman_split`` is the center-to-dip distance, gamma*B).

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

    def __post_init__(self) -> None:
        if self.shot_duration_s <= 0:
            raise ValueError(f"DriftSpec.shot_duration_s must be > 0, got {self.shot_duration_s}")

    @property
    def moves_center(self) -> bool:
        return not self.temperature_k.is_zero()

    @property
    def moves_split(self) -> bool:
        return not self.field_mg.is_zero() or (self.magnet_tempco_per_k != 0.0 and self.moves_center)

    def realize(self, seed: int, base_zeeman_split: float | None) -> DriftTrajectory:
        return DriftTrajectory(self, seed, base_zeeman_split)


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


class DriftTrajectory:
    """One seeded realization of a :class:`DriftSpec`: offsets of the true parameters per shot."""

    def __init__(self, spec: DriftSpec, seed: int, base_zeeman_split: float | None) -> None:
        if spec.magnet_tempco_per_k != 0.0 and spec.moves_center and base_zeeman_split is None:
            raise ValueError(
                f"Drift {spec.label!r} couples the magnet to temperature but the signal has no zeeman_split"
            )
        self.spec = spec
        self._base_split = base_zeeman_split
        temp_seq, field_seq = np.random.SeedSequence(int(seed)).spawn(2)
        dt = spec.shot_duration_s
        self._temperature = _ProcessPath(spec.temperature_k, dt, np.random.default_rng(temp_seq))
        self._field = _ProcessPath(spec.field_mg, dt, np.random.default_rng(field_seq))

    def center_offset_hz(self, shot_index: int) -> float:
        return NV_D_TEMPERATURE_COEFF_HZ_PER_K * self._temperature.value(shot_index)

    def split_offset_hz(self, shot_index: int) -> float:
        offset = NV_GYROMAGNETIC_HZ_PER_MG * self._field.value(shot_index)
        if self.spec.magnet_tempco_per_k != 0.0 and self._base_split is not None:
            offset += self.spec.magnet_tempco_per_k * self._temperature.value(shot_index) * self._base_split
        return offset

    def apply(self, typed_parameters: Any, shot_index: int) -> Any:
        """Return ``typed_parameters`` as they truly are at ``shot_index``."""
        changes: dict[str, float] = {}
        if self.spec.moves_center:
            if not hasattr(typed_parameters, "frequency"):
                raise ValueError(f"Drift {self.spec.label!r} moves the center but the signal has no frequency")
            changes["frequency"] = float(typed_parameters.frequency) + self.center_offset_hz(shot_index)
        if self.spec.moves_split:
            if not hasattr(typed_parameters, "zeeman_split"):
                raise ValueError(f"Drift {self.spec.label!r} moves the splitting but the signal has no zeeman_split")
            # The dip separation is |gamma*B|: a field swinging through zero reopens the split.
            changes["zeeman_split"] = abs(float(typed_parameters.zeeman_split) + self.split_offset_hz(shot_index))
        return dataclasses.replace(typed_parameters, **changes) if changes else typed_parameters

    def truth_summary(self, typed_parameters: Any, n_shots: int, prefix: str = "drift_") -> dict[str, float]:
        """True center/splitting at the last shot and averaged over shots ``0..n_shots-1``."""
        if n_shots <= 0:
            return {}
        out: dict[str, float] = {f"{prefix}shots": float(n_shots)}
        for name, moves, offset in (
            ("frequency", self.spec.moves_center, self.center_offset_hz),
            ("zeeman_split", self.spec.moves_split, self.split_offset_hz),
        ):
            if not moves or not hasattr(typed_parameters, name):
                continue
            base = float(getattr(typed_parameters, name))
            path = np.array([base + offset(i) for i in range(n_shots)])
            if name == "zeeman_split":
                path = np.abs(path)
            out[f"{prefix}true_{name}_start"] = base
            out[f"{prefix}true_{name}_end"] = float(path[-1])
            out[f"{prefix}true_{name}_mean"] = float(path.mean())
        return out


def attach_drift_for_repeat(
    experiment: CoreExperiment, seed: int, generator_name: str, repeat_idx: int
) -> CoreExperiment:
    """Give ``experiment`` its drift realization for this repeat (no-op without drift, or if already attached).

    Seeded from (seed, generator, drift label, repeat) — not strategy or Gaussian sigma — so
    every strategy and noise level in a repeat faces the same drift path.
    """
    from nvision.runner.repeat_keys import measurement_repeat_key, repeat_seed_int

    noise = experiment.noise
    if noise is None or noise.drift is None or experiment.drift is not None:
        return experiment
    key = measurement_repeat_key(seed, generator_name, "drift", noise.drift.label, repeat_idx)
    base_split = getattr(experiment.true_signal.typed_parameters, "zeeman_split", None)
    trajectory = noise.drift.realize(repeat_seed_int(key), None if base_split is None else float(base_split))
    return dataclasses.replace(experiment, drift=trajectory)
