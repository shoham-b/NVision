"""MATLAB ESR data file loader for real-measurement SBED runs."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
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
        Normalised signal ratio (baseline / with_freq), shape (N_freqs,).
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
    shot_ratios: np.ndarray | None = None
    # Per-frequency mean/std/min/max of shot_ratios (i.e. mean-of-ratios, not
    # `signal`'s ratio-of-means) — the actual empirical average, spread, and
    # extremes of the shots recorded at each bin, kept for the "actual averages
    # per frequency" plot.
    signal_mean: np.ndarray | None = None
    signal_std: np.ndarray | None = None
    signal_min: np.ndarray | None = None
    signal_max: np.ndarray | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._visit_counts = np.zeros(len(np.atleast_1d(self.freq_hz)), dtype=np.int64)

    @property
    def visited_mask(self) -> np.ndarray:
        """Boolean mask over frequency bins: True where ``measure()`` has drawn a shot.

        Distinct from ``shot_ratios`` having real data — a bin can hold plenty of
        recorded shots in the file yet never be sampled by a given locator run (it
        converged early, or stopped short of visiting every bin). Callers that want
        to represent only what *this run* actually measured use this, not the
        presence of raw data.
        """
        return self._visit_counts > 0

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

        # Mask zero/NaN slots per frequency. with_freq is now the ratio's denominator (see
        # shot_ratios below), so it needs the same positivity guard baseline always had.
        good = (baseline > 0) & (with_freq > 0) & np.isfinite(baseline) & np.isfinite(with_freq)
        baseline = np.where(good, baseline, np.nan)
        with_freq = np.where(good, with_freq, np.nan)

        # Per-shot ratios, kept so measure() can hand the locator one shot at a time.
        # Feeding it the bin mean instead makes every revisit of a bin return the identical
        # number, which the locator has no way to recognise as a repeat -- it folds the same
        # evidence in again on each visit and the posterior collapses, confidently, onto
        # whichever mode it happened to reach first.
        #
        # Computed as baseline / with_freq, not with_freq / baseline: every recorded file
        # shows the driven-shot channel rising above baseline at resonance, the reverse of
        # the textbook NV dip. Inverting here, once and the same way for every file, turns
        # every resonance back into a dip so the regular positive-only c_total fit
        # (nv_center_lorentzian_bounds_for_domain's default (0.1, 0.4)) applies uniformly —
        # see matlab_cmd.py's locator_bounds, which no longer needs to special-case this.
        shot_ratios = np.where(good, baseline / np.where(with_freq > 0, with_freq, np.nan), np.nan)

        # Per-frequency mean/std of the actual recorded shots (mean-of-ratios), computed
        # unconditionally (unlike noise_std below) since it's needed for the per-frequency
        # averages+spread view regardless of whether noise_std was overridden.
        per_freq_mean = np.nanmean(shot_ratios, axis=1)
        per_freq_std = np.nanstd(shot_ratios, axis=1)
        per_freq_min = np.nanmin(shot_ratios, axis=1)
        per_freq_max = np.nanmax(shot_ratios, axis=1)

        b_mean = np.nanmean(baseline, axis=1)  # (N_freqs,)
        w_mean = np.nanmean(with_freq, axis=1)

        # Guard divide-near-zero
        safe_w = np.where(w_mean > 0, w_mean, np.nan)
        signal = np.clip(b_mean / safe_w, 1e-6, 2.0)

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
            # Per-*shot* spread, matching what measure() hands back (one shot at a time).
            # It must stay consistent with that: quoting the standard error of the bin mean
            # here instead would tell the locator each observation is sqrt(n) more precise
            # than it is, and the particle filter collapses onto the first mode it finds.
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
            shot_ratios=np.clip(shot_ratios, 1e-6, 2.0),
            signal_mean=per_freq_mean,
            signal_std=per_freq_std,
            signal_min=per_freq_min,
            signal_max=per_freq_max,
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
        span = freq_hi - freq_lo
        phys_hz = freq_lo + x_unit * span
        idx = int(np.argmin(np.abs(self.freq_hz - phys_hz)))
        # Report the bin the value actually came from, not the frequency that was asked
        # for. The two differ by up to half a grid step from snapping — and an observation
        # labelled with the wrong frequency is worse than no observation at all: it tells
        # the likelihood the signal has a given value at a point where it does not.
        x_used = (float(self.freq_hz[idx]) - freq_lo) / span if span > 0 else x_unit
        value, sweep_index = self._draw_shot(idx)
        self._visit_counts[idx] += 1
        return Observation(
            x=float(np.clip(x_used, 0.0, 1.0)),
            signal_value=value,
            noise_std=self.noise_std,
            sweep_index=sweep_index,
        )

    def _draw_shot(self, idx: int) -> tuple[float, int | None]:
        """A random recorded shot for bin ``idx``, and the raw shot/sweep column it came from.

        Every step is free to pick any frequency and any shot recorded at it, so the run
        is not forced to consume the file's data: a bin the locator keeps coming back to
        just draws from its shots again (with replacement), rather than being diverted to
        a neighbouring frequency once some quota runs out. The draw is random rather than
        in recorded order, so a heavily-revisited bin doesn't systematically get the late
        sweeps (and whatever drift the instrument had by then).

        Falls back to the bin mean (and no sweep index) when no per-shot data is available
        (a bin whose slots were all masked, or an instance built without ``shot_ratios``).

        The instrument scans every frequency once per sweep, then scans them all again,
        so shot column *j* is the same sweep *j* for every bin.
        """
        if self.shot_ratios is None:
            return float(self.signal[idx]), None
        row = self.shot_ratios[idx]
        valid_cols = np.flatnonzero(np.isfinite(row))
        if valid_cols.size == 0:
            return float(self.signal[idx]), None
        col = int(self.rng.choice(valid_cols))
        return float(row[col]), col


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
