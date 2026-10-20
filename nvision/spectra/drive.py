"""Microwave drive settings for drive-controlled signal models.

Most NV signal models describe a spectrum at one fixed, implicit microwave drive. A
*drive-controlled* model (``SignalModel.controls_drive``) instead predicts the signal as a
function of the drive the experimenter applies at each measurement:

* CW ODMR — the drive amplitude sets the saturation parameter, which power-broadens the dip and
  deepens its contrast together.
* Pulsed ODMR — the drive amplitude sets the Rabi frequency, and the pulse duration is the
  experimenter's π-pulse calibration for that amplitude.

A drive-controlled model can only be evaluated at an explicit :class:`DriveSetting`. Code that
needs an ordinary (frequency-only) model asks for a *fixed-drive view* with
``model.at_drive(setting)``; code that evaluates many measurements taken at different drives
uses :func:`compute_vectorized_many_driven`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nvision.spectra.dtypes import FLOAT_DTYPE

if TYPE_CHECKING:
    from nvision.spectra.signal import SignalModel

# Smallest drive amplitude any locator may request, as a fraction of the source's maximum field
# amplitude. Physical floor rather than a tuning knob: at zero amplitude there is no ODMR signal at
# all (s = 0 / Omega = 0), so the design space must stop short of it.
DRIVE_AMP_MIN_UNIT: float = 0.05


@dataclass(frozen=True)
class DriveSetting:
    """One microwave drive setting, as applied to a single measurement.

    Attributes
    ----------
    drive_amp_unit : float
        Microwave field amplitude as a fraction of the source's maximum amplitude, in
        ``[DRIVE_AMP_MIN_UNIT, 1]``. Linear in field (so proportional to the Rabi frequency),
        not in power.
    pulse_duration_s : float | None
        Pulse length in seconds for pulsed ODMR (the locator's π-pulse calibration for this
        amplitude); ``None`` for CW ODMR.
    """

    drive_amp_unit: float
    pulse_duration_s: float | None = None

    def __post_init__(self) -> None:
        if not (DRIVE_AMP_MIN_UNIT - 1e-12 <= self.drive_amp_unit <= 1.0 + 1e-12):
            raise ValueError(
                f"DriveSetting.drive_amp_unit must lie in [{DRIVE_AMP_MIN_UNIT}, 1]; got {self.drive_amp_unit!r}"
            )
        if self.pulse_duration_s is not None and not (
            math.isfinite(self.pulse_duration_s) and self.pulse_duration_s > 0.0
        ):
            raise ValueError(f"DriveSetting.pulse_duration_s must be a positive time; got {self.pulse_duration_s!r}")


def compute_vectorized_many_driven(
    model: SignalModel,
    xs: Sequence[float] | np.ndarray,
    samples: object,
    drives: Sequence[DriveSetting],
    *,
    fast: bool = False,
) -> np.ndarray:
    """Predict many measurements, each taken at its own drive, over shared parameter samples.

    Groups the points by drive setting and evaluates each group through the model's fixed-drive
    view, so the per-group cost is the ordinary vectorized kernel.

    Parameters
    ----------
    model : SignalModel
        A drive-controlled model (``model.controls_drive`` is True).
    xs : array-like
        Drive frequencies in the model's own x coordinate. shape: (n_points,)
    samples : object
        Parameter samples in any form ``model.compute_vectorized_many`` accepts (n_samples each).
    drives : sequence of DriveSetting
        The drive of each point, aligned with ``xs``. shape: (n_points,)
    fast : bool
        Use the fastmath acquisition kernel (EIG scoring only, never likelihood updates).

    Returns
    -------
    np.ndarray
        Predictions. shape: (n_points, n_samples)
    """
    if not model.controls_drive:
        raise ValueError(f"{type(model).__name__} is not drive-controlled; evaluate it without drive settings.")
    xs_arr = np.asarray(xs, dtype=FLOAT_DTYPE)
    if xs_arr.ndim != 1:
        raise ValueError("xs must be one-dimensional")
    if len(drives) != xs_arr.shape[0]:
        raise ValueError(f"drives ({len(drives)}) must be aligned with xs ({xs_arr.shape[0]})")

    groups: dict[DriveSetting, list[int]] = {}
    for i, drive in enumerate(drives):
        if drive is None:
            raise ValueError(f"Point {i} has no drive setting; a drive-controlled model needs one for every point.")
        groups.setdefault(drive, []).append(i)

    out: np.ndarray | None = None
    for drive, idx in groups.items():
        view = model.at_drive(drive)
        idx_arr = np.asarray(idx, dtype=np.int64)
        evaluate = view.compute_vectorized_many_fast if fast else view.compute_vectorized_many
        block = evaluate(xs_arr[idx_arr], samples)
        if out is None:
            out = np.empty((xs_arr.shape[0], block.shape[1]), dtype=block.dtype)
        out[idx_arr] = block
    if out is None:
        return np.empty((0, 0), dtype=FLOAT_DTYPE)
    return out
