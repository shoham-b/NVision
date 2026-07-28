"""MATLAB ESR data file loader for real-measurement SBED runs."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nvision.models.observation import Observation
from nvision.sim.defaults import NVISION_NOISE_GAUSS

log = logging.getLogger(__name__)

# Canonical repo-relative location for .mat files.
_MATLAB_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "matlab"


def _resolve_mat_path(path: str | Path) -> Path:
    """Resolve a .mat file path, checking data/matlab/ for bare filenames."""
    p = Path(path)
    if p.parent == Path(".") and not p.exists():
        candidate = _MATLAB_DATA_DIR / p.name
        if candidate.exists():
            log.debug("Resolved %s → %s", path, candidate)
            return candidate
    return p


def _load_mat_v5(path: Path):
    """Load a v4/v5 .mat file via scipy.io. Returns the raw mat dict."""
    import scipy.io

    return scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _load_mat_v73(path: Path):
    """Load a v7.3 (HDF5) .mat file via h5py. Returns a lightweight wrapper."""
    if importlib.util.find_spec("h5py") is None:
        raise ImportError(
            "This .mat file uses the HDF5/v7.3 format which requires h5py. Install it with: pip install h5py"
        )

    raise NotImplementedError(
        "HDF5/v7.3 .mat files are not yet supported. Save the file in MATLAB v5 format (-v7.3 flag off) and retry."
    )


def _extract_esr(mat: dict):
    """Extract the ESR sub-struct from the loaded mat dict."""
    if "myStruct" not in mat:
        available = [k for k in mat if not k.startswith("_")]
        raise KeyError(
            f"Expected top-level key 'myStruct' in .mat file, but found: {available}. Is this an NVision ESR file?"
        )
    top = mat["myStruct"]
    if not hasattr(top, "ESR"):
        raise KeyError("'myStruct' exists but has no 'ESR' field. Check that this is an ESR measurement file.")
    return top.ESR


@dataclass
class MatlabDataFile:
    """Preloaded ESR measurement data from a MATLAB file.

    Attributes
    ----------
    freq_hz : np.ndarray
        Frequency grid in Hz, shape (N_freqs,).
    signal : np.ndarray
        Normalised signal ratio (with_freq / baseline), shape (N_freqs,).
        Values ≈ 1.0 in background, dipping below at resonance.
    noise_std : float
        Initial noise estimate passed to the SBED locator. The locator refines
        this adaptively from background measurements.
    n_valid_shots : int
        Number of valid shot slots used for the signal computation.
    """

    freq_hz: np.ndarray
    signal: np.ndarray
    noise_std: float
    n_valid_shots: int

    @classmethod
    def load(
        cls,
        path: str | Path,
        valid_shots: int | None = None,
        noise_std_override: float | None = None,
    ) -> MatlabDataFile:
        """Load an ESR .mat file and compute signal + noise estimate.

        Parameters
        ----------
        path :
            Path to the .mat file, or a bare filename resolved via data/matlab/.
        valid_shots :
            Override the number of valid shot columns to use. Defaults to
            ``esr.currIter`` (the number of averages completed in the file).
        noise_std_override :
            If given, skip noise estimation and use this value directly.
        """
        resolved = _resolve_mat_path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"MATLAB file not found: {path!r}\nAlso checked: {_MATLAB_DATA_DIR / Path(path).name}"
            )

        log.debug("Loading MATLAB file: %s", resolved)
        try:
            mat = _load_mat_v5(resolved)
        except NotImplementedError:
            mat = _load_mat_v73(resolved)

        esr = _extract_esr(mat)

        # --- Frequency axis (MHz → Hz) ---
        freq_mhz = np.asarray(esr.frequency, dtype=np.float64).ravel()
        freq_hz = freq_mhz * 1e6

        # --- Signal array ---
        raw = np.asarray(esr.signal, dtype=np.float64)  # (2, N_freqs, N_shots_max)
        if raw.ndim != 3 or raw.shape[0] != 2:
            raise ValueError(f"Expected esr.signal shape (2, N_freqs, N_shots), got {raw.shape}")
        n_freqs, n_shots_max = raw.shape[1], raw.shape[2]

        if len(freq_hz) != n_freqs:
            raise ValueError(
                f"Frequency array length ({len(freq_hz)}) does not match signal N_freqs dimension ({n_freqs})"
            )

        # Determine valid shot count
        curr_iter = int(getattr(esr, "currIter", n_shots_max))
        n_valid = min(valid_shots, n_shots_max) if valid_shots is not None else min(curr_iter, n_shots_max)
        if n_valid <= 0:
            raise ValueError(
                f"No valid shots available (currIter={curr_iter}, valid_shots={valid_shots}). "
                "The measurement may not have started yet."
            )

        baseline = raw[0, :, :n_valid].copy()  # (N_freqs, N_valid)
        with_freq = raw[1, :, :n_valid].copy()  # (N_freqs, N_valid)

        # Mask zero/NaN slots per frequency
        good = (baseline > 0) & np.isfinite(baseline) & np.isfinite(with_freq)
        baseline = np.where(good, baseline, np.nan)
        with_freq = np.where(good, with_freq, np.nan)

        b_mean = np.nanmean(baseline, axis=1)  # (N_freqs,)
        w_mean = np.nanmean(with_freq, axis=1)

        # Guard divide-near-zero
        safe_b = np.where(b_mean > 0, b_mean, np.nan)
        signal = np.clip(w_mean / safe_b, 1e-6, 2.0)

        if np.any(np.isnan(signal)):
            n_nan = int(np.sum(np.isnan(signal)))
            log.warning(
                "%d frequency bins have NaN signal (all shots masked). They will return the nearest valid neighbour.",
                n_nan,
            )
            # Fill NaN bins with nearest valid neighbour
            signal = _fill_nan_nearest(signal)

        # --- Noise estimate ---
        if noise_std_override is not None:
            noise_std = float(noise_std_override)
            log.info("Using user-supplied noise_std=%.4g", noise_std)
        else:
            shot_ratios = np.where(good, with_freq / np.where(baseline > 0, baseline, np.nan), np.nan)
            per_freq_std = np.nanstd(shot_ratios, axis=1)
            noise_std = float(np.nanmedian(per_freq_std))
            if not (1e-6 < noise_std < 1.0):
                log.warning(
                    "Auto-estimated noise_std=%.4g is implausible; "
                    "falling back to default %.4g. Use --noise-std to override.",
                    noise_std,
                    NVISION_NOISE_GAUSS,
                )
                noise_std = NVISION_NOISE_GAUSS
            log.info("Auto-estimated noise_std=%.4g from %d shots", noise_std, n_valid)

        return cls(
            freq_hz=freq_hz,
            signal=signal,
            noise_std=noise_std,
            n_valid_shots=n_valid,
        )

    def measure(self, x_unit: float, freq_lo: float, freq_hi: float) -> Observation:
        """Return an Observation for the MATLAB grid point nearest to x_unit.

        Parameters
        ----------
        x_unit :
            Normalised position in [0, 1] as returned by ``locator.next()``.
        freq_lo, freq_hi :
            Physical Hz bounds used by the locator's belief — must match
            ``self.freq_hz.min()`` / ``self.freq_hz.max()``.
        """
        phys_hz = freq_lo + x_unit * (freq_hi - freq_lo)
        idx = int(np.argmin(np.abs(self.freq_hz - phys_hz)))
        return Observation(
            x=x_unit,
            signal_value=float(self.signal[idx]),
            noise_std=self.noise_std,
        )


def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with the nearest non-NaN neighbour (1-D)."""
    result = arr.copy()
    nan_mask = np.isnan(result)
    if not nan_mask.any():
        return result
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("All signal values are NaN — no valid measurements in file.")
    for i in np.where(nan_mask)[0]:
        nearest = valid_idx[int(np.argmin(np.abs(valid_idx - i)))]
        result[i] = result[nearest]
    return result
