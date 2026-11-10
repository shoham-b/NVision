"""CLI command for running SBED on real MATLAB ESR measurements."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import time
import zlib
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import typer

from nvision.cli.app_instance import app
from nvision.metrics.lab_search_fit import fit_search_curve
from nvision.models.locator import Locator
from nvision.tools.paths import ARTIFACTS_ROOT
from nvision.tools.utils import NVISION_RNG_SEED

log = logging.getLogger(__name__)

# Subdirectory of --dir holding a MATLAB-only *copy* of the results cache (<dir>/matlab/cache, served
# with `nv serve --dir <dir>/matlab` for a fast, dedicated view) and the per-run standalone bundles
# (<dir>/matlab/runs/). Every run is also written into <dir>/cache (see _write_artifacts), so it shows
# up in the main `nv serve --dir <dir>` too.
MATLAB_UI_DIRNAME = "matlab"


@dataclass(frozen=True)
class _MatlabRunSummary:
    """Per-file convergence outcome, used by ``--all`` to report a per-criterion breakdown."""

    file_name: str
    primary_converged_step: int | None
    all_converged_step: int | None
    total_steps: int
    # Signed holdout z per parameter (see nvision.metrics.lab_reference); None for a run on all the data.
    holdout_z: dict[str, float] | None = None
    # The run's result record (what the lab page and --out show).
    record: dict[str, Any] | None = None


def _echo_convergence_breakdown(summaries: list[_MatlabRunSummary]) -> None:
    """Print how many files converged under each criterion, and at what step.

    "Split converged" (``primary_converged_step``) is the primary milestone -- the
    locator has resolved the Zeeman doublet -- while "All converged" (``all_converged_step``)
    additionally requires every other parameter (zero_field_freq, hwhm, contrast_total) to have
    narrowed too, so it's strictly harder to hit and frequently stays None even on files
    that shift-converged fine. Reporting both separately (rather than only "did it finish")
    mirrors the results UI's stopping-criteria toggle so a batch run's summary means the
    same thing as what you'd see there.
    """
    n = len(summaries)
    typer.echo("\nConvergence breakdown:")
    for label, field in (("Split converged", "primary_converged_step"), ("All converged", "all_converged_step")):
        steps = [getattr(s, field) for s in summaries]
        n_converged = sum(1 for step in steps if step is not None)
        typer.echo(f"  {label}: {n_converged}/{n}")
        for s, step in zip(summaries, steps, strict=True):
            status = f"step {step}" if step is not None else "not converged"
            typer.echo(f"    {s.file_name:40s} {status}")
    _echo_holdout_coverage([s.holdout_z for s in summaries if s.holdout_z is not None])


def _echo_holdout_coverage(zs: list[dict[str, float]]) -> None:
    """Share of files whose held-out z is within 1 and 2 sigma, against the Gaussian 68% / 95%.

    One file is one independent z per parameter, so this is coverage across files.
    """
    if not zs:
        return
    typer.echo(f"\nHoldout coverage over {len(zs)} file(s) (nominal 68% / 95%):")
    for name in sorted({k for z in zs for k in z}):
        vals = np.array([abs(z[name]) for z in zs if name in z])
        typer.echo(
            f"  {name:16s}: |z|<=1 {np.mean(vals <= 1):.0%}  |z|<=2 {np.mean(vals <= 2):.0%}  "
            f"median |z| {np.median(vals):.2f}  (n={vals.size})"
        )


# ---------------------------------------------------------------------------
# Duck-typed helpers — satisfy TrueSignal / CoreExperiment interfaces without
# importing those concrete types (which carry heavy simulation dependencies).
# ---------------------------------------------------------------------------


class _MatlabSignalProxy:
    """Minimal TrueSignal duck-type for real measurements (no *parametric* ground truth).

    There's no fitted model to compare against, so ``parameter_values``/``get_param_value``
    stay NaN (unchanged — those feed error metrics that genuinely have nothing to measure
    against). But when ``data`` is given, ``__call__`` interpolates the recorded per-bin
    signal so the scan plot's dense curve can show the real measured spectrum instead of
    drawing nothing.
    """

    model = None  # needed by some code paths that check hasattr(true_signal, 'model')

    def __init__(self, drive_freq_min_phys: float, drive_freq_max_phys: float, data: Any = None) -> None:
        self._drive_freq_min_phys = drive_freq_min_phys
        self._drive_freq_max_phys = drive_freq_max_phys
        self._interp_drive_freq_phys: np.ndarray | None = None
        self._interp_signal: np.ndarray | None = None
        if data is not None:
            order = np.argsort(data.drive_freq_phys)
            self._interp_drive_freq_phys = np.asarray(data.drive_freq_phys, dtype=float)[order]
            signal = np.asarray(data.signal, dtype=float)[order]
            # NaN out any bin this run never actually measured. The file can hold a real
            # recorded mean for every bin, but a given run may have converged early or
            # stopped short of visiting all of them — np.interp propagates a NaN endpoint
            # across the whole segment touching it, so this breaks the curve over any
            # unvisited stretch instead of bridging it with an unmeasured value.
            visited = np.asarray(data.visited_mask, dtype=bool)[order]
            self._interp_signal = np.where(visited, signal, np.nan)

    def __call__(self, x: float) -> float:
        if self._interp_drive_freq_phys is None or self._interp_signal is None:
            return float("nan")
        return float(np.interp(x, self._interp_drive_freq_phys, self._interp_signal))

    def parameter_values(self) -> dict[str, float]:
        return {"zero_field_freq": float("nan")}

    def get_param_value(self, name: str) -> float:
        return float("nan")

    def all_bounds(self) -> dict[str, tuple[float, float]]:
        return {"zero_field_freq": (self._drive_freq_min_phys, self._drive_freq_max_phys)}


class _MatlabExperiment:
    """Minimal CoreExperiment duck-type backed by a MatlabDataFile."""

    def __init__(
        self, data: Any, true_signal: _MatlabSignalProxy, drive_freq_min_phys: float, drive_freq_max_phys: float
    ) -> None:
        self.true_signal = true_signal
        self.noise = None  # real measurements: no CompositeNoise model, mirrors CoreExperiment.noise
        self.drive_freq_min_phys = drive_freq_min_phys
        self.drive_freq_max_phys = drive_freq_max_phys
        self._data = data
        self._drive_freq_min_phys = drive_freq_min_phys
        self._drive_freq_max_phys = drive_freq_max_phys

    def measure(self, drive_freq_unit: float, rng: Any = None):
        return self._data.measure(drive_freq_unit, self._drive_freq_min_phys, self._drive_freq_max_phys)

    @property
    def signal(self):
        """Physical-domain signal callable — for viz compatibility (mirrors CoreExperiment.signal)."""
        return self.true_signal


# ---------------------------------------------------------------------------
# Run generator
# ---------------------------------------------------------------------------


def _matlab_loop(
    locator: Locator,
    data: Any,
    drive_freq_min_phys: float,
    drive_freq_max_phys: float,
    no_progress: bool,
) -> Generator[Locator]:
    """Adaptive measurement loop — yields locator state after each observation.

    Each step measures whichever frequency the locator picks, drawing one of that
    frequency's recorded shots (see ``MatlabDataFile.measure``), and stops when the
    locator does — the run is not obliged to consume every point in the file.
    """
    while not locator.done():
        drive_freq_unit = locator.next_drive_freq_unit()
        if locator.next_drive() is not None:
            raise ValueError(
                f"{type(locator).__name__} chooses the microwave drive, but recorded MATLAB data has no drive axis."
            )
        obs = data.measure(drive_freq_unit, drive_freq_min_phys, drive_freq_max_phys)
        locator.observe(obs)

        if not no_progress:
            phys_mhz = (drive_freq_min_phys + drive_freq_unit * (drive_freq_max_phys - drive_freq_min_phys)) / 1e6
            est = locator.belief.estimates()
            unc = locator.belief.uncertainty()
            freq_est_mhz = est.get("zero_field_freq", float("nan")) / 1e6
            freq_unc_mhz = unc.get("zero_field_freq", float("nan")) / 1e6
            typer.echo(
                f"Step {locator.step_count:3d}: {phys_mhz:7.2f} MHz -> "
                f"signal={obs.signal_value:.4f}  "
                f"freq={freq_est_mhz:.2f} +/- {freq_unc_mhz:.2f} MHz"
            )

        yield locator


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


# Sweep settings of `nv matlab`; the same options apply with or without a SOURCE.
_DEFAULT_MAX_STEPS = 300
_DEFAULT_PARTICLES = 10000


@app.command("matlab")
def matlab(
    source: Annotated[
        str | None,
        typer.Argument(
            help=(
                "A Google Drive folder link (or id), a local folder, or a single .mat file. "
                "Omit to re-sync every source added before (same options apply)."
            )
        ),
    ] = None,
    directory: Annotated[
        Path,
        typer.Option("--dir", help="Artifacts directory results are stored under (the one given to `nv serve`)."),
    ] = ARTIFACTS_ROOT,
    sweeps: Annotated[
        bool,
        typer.Option("--sweeps/--no-sweeps", help="Run SBED on the raw ESR sweep files (a few seconds per file)."),
    ] = True,
    noise_std: Annotated[
        float | None,
        typer.Option("--noise-std", help="Sweeps: override the auto-estimated measurement noise std."),
    ] = None,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Sweeps: maximum SBED measurement steps."),
    ] = _DEFAULT_MAX_STEPS,
    valid_shots: Annotated[
        int | None,
        typer.Option("--valid-shots", help="Sweeps: number of shot columns to use (default: esr.currIter)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Sweeps: also write each file's result record (JSON) into this folder."),
    ] = None,
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Sweeps: suppress per-step output."),
    ] = False,
    no_ui: Annotated[
        bool,
        typer.Option("--no-ui", help="Sweeps: write only the result records, not the per-run plots and cache entries."),
    ] = False,
    particles: Annotated[
        int,
        typer.Option("--particles", help="Sweeps: SMC particle count (default 10x the simulation default)."),
    ] = _DEFAULT_PARTICLES,
    infer_zero_field_freq: Annotated[
        bool,
        typer.Option(
            "--infer-center-freq/--no-infer-center-freq",
            help=(
                "Sweeps: fit the NV zero-field-splitting center instead of fixing it to the "
                "textbook 2.87 GHz. Real samples run 1-2 MHz off that value from strain/"
                "temperature, which is comparable to the Zeeman splitting itself — a fixed "
                "wrong center visibly distorts the fit. The simulated grid always uses the "
                "exact value it was generated with, so it doesn't need this."
            ),
        ),
    ] = True,
) -> None:
    """Add the lab's MATLAB data (a Drive link or a local path) and show it in ``nv serve``.

    Run it once per SOURCE: it syncs now (a Drive link is fetched read-only with rclone, downloading only the
    files that are missing or changed, and only new or changed files are processed). The source is remembered
    under ``<dir>/labs/sources.json``; run ``nv matlab`` with no SOURCE (or press the lab page's Sync button)
    to pick up later changes. ``nv serve`` never syncs by itself. Every
    ``.mat`` file below SOURCE is handled by what it holds:

    - the lab's Bayesian ESR runs (``myStruct.BayesianESR``): one lab study per folder (calibration,
      steps to convergence), listed under "Lab: ..." in the UI's Study dropdown;
    - raw ESR sweeps (``myStruct.ESR``): run through SBED twice -- on all the data (noise ``real``) and on the even
      sweeps only, checked against a fit of the odd ones (noise ``real-holdout``, see
      :mod:`nvision.metrics.lab_reference`) -- listed under "Lab: raw sweeps (SBED)" and "MATLAB (real data)".
    """
    from nvision.tools.lab_sources import add_source, load_sources

    if source is None:
        entries = load_sources(directory)
        if not entries:
            typer.echo(f"No lab sources in {directory}; add one with `nv matlab <Drive link or folder>`.")
            return
    else:
        try:
            entry = add_source(directory, source)
        except ValueError as exc:
            typer.echo(f"Error: {exc}")
            raise typer.Exit(code=1) from exc
        typer.echo(f"Source '{entry['label']}' saved in {directory}; sync it again later with `nv matlab`.\n")
        entries = [entry]
    options = dict(
        noise_std=noise_std,
        max_steps=max_steps,
        valid_shots=valid_shots,
        no_progress=no_progress,
        no_ui=no_ui,
        particles=particles,
        infer_zero_field_freq=infer_zero_field_freq,
    )
    ok = _sync_entries(entries, directory, sweeps=sweeps, out=out, **options)
    if source is not None:
        typer.echo(f'\nView: uv run nv serve --dir "{directory}"')
    if not ok:
        raise typer.Exit(code=1)


def _sync_entries(entries: list[dict[str, Any]], directory: Path, **options: Any) -> bool:
    """Sync *entries* (one just-added source, or every remembered one) the same way; True if all succeeded.

    Only new or changed files are processed. Progress and per-source failures go to
    ``<dir>/labs/sync_status.json``, which the lab page shows.
    """
    from nvision.tools.lab_sources import SyncLock, write_sync_status

    with SyncLock(directory) as lock:
        if not lock.acquired:
            typer.echo(
                "Another lab sync is already running for this directory (e.g. the lab page's Sync button); "
                "the source list is saved and its files are processed by the next sync."
            )
            return True
        status: dict[str, Any] = {"state": "running", "started": datetime.now(UTC).isoformat(), "sources": []}
        write_sync_status(directory, status)
        try:
            for entry in entries:
                item: dict[str, Any] = {"label": entry["label"], "failures": [], "error": None}
                status["sources"].append(item)
                try:
                    item["failures"] = sync_source(entry, directory, **options)
                except Exception as exc:  # reported on the lab page; the other sources still sync
                    log.exception("lab sync failed for %s", entry["label"])
                    typer.echo(f"Error: {exc}")
                    item["error"] = f"{type(exc).__name__}: {exc}"
                write_sync_status(directory, status)
        finally:
            failed = any(s["error"] or s["failures"] for s in status["sources"])
            status["state"] = "failed" if failed or len(status["sources"]) < len(entries) else "done"
            status["finished"] = datetime.now(UTC).isoformat()
            write_sync_status(directory, status)
    return not failed


def sync_source(
    entry: dict[str, Any], directory: Path, *, sweeps: bool = True, out: Path | None = None, **options: Any
) -> list[str]:
    """Fetch one remembered source and process its new or changed files; return the files that failed.

    Bayesian runs are re-summarised per folder when any of its runs changed. Each raw sweep gets a result
    record under ``<dir>/labs/<label>/sweeps/<path>.json`` (both SBED runs, or the error); a record whose
    source size/mtime and :data:`~nvision.tools.lab_sources.PROCESSING_VERSION` still match is not redone.
    """
    from nvision.metrics.lab_process import load_bayesian_runs, write_statistics
    from nvision.metrics.lab_study import labs_dir
    from nvision.tools.lab_sources import FileIndex, resolve_source, sweep_record_is_current, sweep_record_path

    root, base = resolve_source(entry, directory)
    files = [root] if root.is_file() else sorted(root.rglob("*.mat"))
    typer.echo(f"Source '{entry['label']}': {len(files)} .mat file(s) in {root}")
    label_dir = labs_dir(directory) / entry["label"]
    index = FileIndex(label_dir)
    kinds = index.classify(files, base)
    typer.echo("  " + ", ".join(f"{n} {kind}" for kind, n in sorted(index.counts().items())))
    for f, kind in kinds.items():
        if kind == "unreadable":
            typer.echo(f"  Skipped (not a readable MATLAB file): {f.relative_to(base).as_posix()}")

    bayes_failures: list[str] = []
    by_folder: dict[Path, list[Path]] = {}
    for f in files:
        if kinds[f] == "bayesian":
            by_folder.setdefault(f.parent, []).append(f)
    for folder, folder_files in by_folder.items():
        rel = folder.relative_to(base)
        stats_dir = label_dir / rel / "convergence_statistics"
        # A folder processed before window_focus.csv or the search-phase fit existed is redone (the lab's own
        # summaries never get either).
        own_summary = (stats_dir / "window_summary.csv").exists()
        has_focus = (stats_dir / "window_focus.csv").exists() or own_summary
        has_search_fit = any(stats_dir.glob("window*MHz_search_fit.json")) or own_summary
        if not index.folder_changed(folder_files, base) and stats_dir.exists() and has_focus and has_search_fit:
            continue
        own = folder / "convergence_statistics"
        try:
            if (own / "window_summary.csv").exists():
                shutil.copytree(own, stats_dir, dirs_exist_ok=True)  # the lab's own script already summarised these
                typer.echo(f"Lab study {rel}: kept the lab's own statistics ({len(folder_files)} runs)")
            else:
                runs_by_folder, _ = load_bayesian_runs(folder_files)
                if folder not in runs_by_folder:
                    raise ValueError("none of its Bayesian ESR files could be read as a run")
                write_statistics(runs_by_folder[folder], stats_dir)
                typer.echo(f"Lab study {rel}: processed {len(folder_files)} runs")
        except Exception as exc:  # one bad folder must not stop the others or the sweeps; reported as a failure
            log.exception("lab study failed for %s", folder)
            typer.echo(f"FAILED: lab study {rel.as_posix()}: {type(exc).__name__}: {exc}")
            bayes_failures.append(f"{rel.as_posix()}/ (Bayesian runs)")
    index.save()

    sweep_files = [f for f in files if kinds[f] == "sweep"]
    todo = [f for f in sweep_files if not sweep_record_is_current(sweep_record_path(label_dir, f, base), f)]
    if sweep_files and not sweeps:
        typer.echo(f"Skipped {len(sweep_files)} raw sweep file(s) (--no-sweeps).")
        return bayes_failures
    typer.echo(f"Raw sweeps: {len(sweep_files)}, {len(todo)} new or changed.")
    failures: list[str] = list(bayes_failures)
    summaries: list[_MatlabRunSummary] = []
    for i, f in enumerate(todo, 1):
        rel = f.relative_to(base).as_posix()
        typer.echo(f"=== [{i}/{len(todo)}] {rel} ===")
        record_path = sweep_record_path(label_dir, f, base)
        try:
            record, summary = _process_sweep(f, run_name=f"{entry['label']}/{rel}", directory=directory, **options)
            summaries.append(summary)
        except Exception as exc:
            log.exception("matlab failed for %s", f)
            typer.echo(f"FAILED: {rel}: {exc}")
            failures.append(rel)
            record = {"error": f"{type(exc).__name__}: {exc}"}
        _write_sweep_record(record_path, f, base, record)
        if out is not None:
            _write_sweep_record(out / f"{Path(rel).with_suffix('')}.json", f, base, record)
        typer.echo("")
    if todo:
        typer.echo(f"Done: {len(todo) - len(failures) + len(bayes_failures)}/{len(todo)} succeeded.")
    if failures:
        typer.echo(f"Failed: {', '.join(failures)}")
    if summaries:
        _echo_convergence_breakdown(summaries)
    return failures


def _process_sweep(matlab_file: Path, *, run_name: str, directory: Path, **options: Any):
    """Both SBED runs of one sweep file: on all of it, and on the even sweeps against the odd ones."""
    full = _matlab_run_one(matlab_file=matlab_file, run_name=run_name, directory=directory, holdout=False, **options)
    held = _matlab_run_one(matlab_file=matlab_file, run_name=run_name, directory=directory, holdout=True, **options)
    return {"full": full.record, "holdout": held.record}, held


def _write_sweep_record(path: Path, matlab_file: Path, base: Path, record: dict[str, Any]) -> None:
    from nvision.tools.lab_sources import PROCESSING_VERSION

    stat = matlab_file.stat()
    rel = matlab_file.relative_to(base)
    payload = {
        "file": rel.as_posix(),
        "folder": rel.parent.as_posix(),
        "source_size": stat.st_size,
        "source_mtime": stat.st_mtime,
        "version": PROCESSING_VERSION,
        **record,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# contrast_total prior for real data. The positive-only sign is the simulated grid's (the ratio
# inversion in MatlabDataFile.load makes every resonance dip), but not its (0.1, 0.4)
# magnitude cap: the raw ratio isn't a calibrated population fraction, and both files in
# data/matlab/ sat pinned against 0.4. The capped fit made up the missing depth with a
# wider line — ESR_20251112_134403 fit hwhm 3.1 MHz and R^2 0.946, against 2.2 MHz
# and 0.976 (the unconstrained scipy ceiling) with the cap lifted, where contrast_total settles
# at ~0.54.
# 1.0 is the physical ceiling: a dip can't go below zero signal.
_REAL_DATA_CONTRAST_TOTAL_BOUNDS: tuple[float, float] = (0.1, 1.0)


@contextlib.contextmanager
def _real_data_contrast_total_threshold() -> Generator[None]:
    """Scale contrast_total's absolute convergence threshold to ``_REAL_DATA_CONTRAST_TOTAL_BOUNDS``.

    NVISION_CONTRAST_TOTAL_CONVERGENCE_THRESHOLD (0.01) is absolute and was calibrated against
    the simulated grid's (0.1, 0.4) range. Left as-is under the wider real-data range it
    demands a precision the recorded shots can't give (the 205-shot file never reported
    all-parameter convergence), so keep it the same *fraction of the prior range* for the
    duration of the run, and restore it afterwards so nothing else in the process sees it.
    """
    from nvision.sim.defaults import PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS
    from nvision.spectra.nv_center import (
        DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
        DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
        nv_center_lorentzian_bounds_for_domain,
    )

    sim_lo, sim_hi = nv_center_lorentzian_bounds_for_domain(
        DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
        DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
        hyperfine="unresolved",
        with_zeeman_splitting=True,
    )["contrast_total"]
    real_lo, real_hi = _REAL_DATA_CONTRAST_TOTAL_BOUNDS
    original = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS["contrast_total"]
    PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS["contrast_total"] = original * (real_hi - real_lo) / (sim_hi - sim_lo)
    try:
        yield
    finally:
        PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS["contrast_total"] = original


@_real_data_contrast_total_threshold()
def _matlab_run_one(
    *,
    matlab_file: Path,
    run_name: str,
    directory: Path,
    noise_std: float | None,
    max_steps: int,
    valid_shots: int | None,
    no_progress: bool,
    no_ui: bool,
    particles: int,
    infer_zero_field_freq: bool,
    holdout: bool,
) -> _MatlabRunSummary:
    """Run the SBED locator on a single ESR .mat file; *run_name* (source label / path) names it in the cache.

    Seeded from *run_name*, so re-processing the same file reproduces the same run.
    """
    from nvision.noises.over_frequency.gaussian_noise import OverFrequencyGaussianNoise
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
    from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
    from nvision.tools.matlab_loader import MatlabDataFile

    # --- Load data ---
    typer.echo(f"Loading: {matlab_file}")
    data = MatlabDataFile.load(matlab_file, valid_shots=valid_shots, noise_std_override=noise_std)
    seed = (NVISION_RNG_SEED + zlib.crc32(run_name.encode())) % 2**32
    data.rng = np.random.default_rng(seed)
    ref_data = None
    if holdout:
        data, ref_data = data.split_sweeps(noise_std)
        typer.echo(
            f"Holdout: SBED gets the {data.n_valid_shots} even sweeps, "
            f"the reference fit the {ref_data.n_valid_shots} odd ones"
        )

    drive_freq_min_phys = float(data.drive_freq_phys.min())
    drive_freq_max_phys = float(data.drive_freq_phys.max())
    n_drive_freq_points = len(data.drive_freq_phys)

    typer.echo(
        f"Loaded {n_drive_freq_points} frequencies: "
        f"{drive_freq_min_phys / 1e6:.1f} to {drive_freq_max_phys / 1e6:.1f} MHz  |  "
        f"valid shots: {data.n_valid_shots}  |  noise_std: {data.noise_std:.4f}"
    )

    # --- Build locator ---
    # contrast_total: real .mat signal ratios used to come out peaking above 1.0 instead of
    # dipping below it (whichever shot channel the instrument calls "baseline" can land on
    # either side of the driven channel). That inversion is corrected once, uniformly, in
    # MatlabDataFile.load (baseline / with_freq instead of with_freq / baseline), so the
    # positive-only sign applies directly — but the magnitude cap doesn't; see
    # _REAL_DATA_CONTRAST_TOTAL_BOUNDS.
    #
    # zero_field_freq is inferred (--infer-center-freq), not pinned to the simulated grid's 2870 MHz:
    # both files' sweeps are centered on 2870 MHz, but their doublets sit at 2871.62 and
    # 2871.13 MHz, and pinning drops R^2 from 0.976 to 0.824 on the narrower file.
    #
    # zeeman_shift's default ceiling is MAX_ZEEMAN_SHIFT (60 MHz), sized for the simulated
    # domain and wider than some real sweeps (one file here spans only 80 MHz in total).
    # Left alone it lets the fit park one of the two peaks outside the measured window and
    # "explain" only the half it can see — which is what the narrower file did: center
    # 2840 MHz, Zeeman shift 51.7 MHz, lower peak at 2789 MHz against data starting at 2830 MHz.
    # A quarter of the span keeps the doublet inside a sweep that was deliberately
    # recorded around it. Note this cannot be done by narrowing the "zero_field_freq" bound
    # instead: nv_center_smc_belief reuses that same entry as the drive-frequency domain, so
    # tightening it would stop the locator from ever measuring the wings.
    span = drive_freq_max_phys - drive_freq_min_phys
    locator_bounds = {
        "zero_field_freq": (drive_freq_min_phys, drive_freq_max_phys),
        "zeeman_shift": (0.0, span / 4.0),
        "contrast_total": _REAL_DATA_CONTRAST_TOTAL_BOUNDS,
    }

    ref = None
    lab_curve_fit = None
    if ref_data is not None:
        ref = _fit_holdout_reference(ref_data, locator_bounds, infer_zero_field_freq)
    else:
        # The lab's own averaged sweep (all shots): the files store that curve but no fit of it, so fit it here.
        # SBED draws from these same shots, so this is a yardstick for precision, not an independent check.
        lab_curve_fit = _fit_holdout_reference(
            data, locator_bounds, infer_zero_field_freq, label="Lab averaged sweep (all shots)"
        )

    # The SMC default (1000 particles, 1% exploration) is tuned for a live instrument,
    # where each measurement is expensive and the run must stay cheap. On these
    # 4-parameter real-data posteriors it is too thin to resolve the modes reliably:
    # repeated runs over both files in data/matlab/ landed on a badly wrong mode 1 run in
    # 5 and 3 in 5 respectively. 10k particles at 5% exploration hit the best achievable
    # fit on 5/5 runs of both files with run-to-run spread in the 4th decimal, for ~2s a
    # run; 40k was no better. Re-running a recorded file is cheap, so buy the robustness.
    locator = SequentialBayesianExperimentDesignLocator.create(
        builder=nv_center_smc_belief,
        parameter_bounds=locator_bounds,
        # The belief always infers the noise level through its conjugate prior; its prior window is
        # centred on the noise measured from (or overridden for) this recording.
        noise_model=OverFrequencyGaussianNoise(std=data.noise_std).to_noise_signal_model(),
        max_steps=max_steps,
        with_fixed_zero_field_freq=not infer_zero_field_freq,
        num_particles=particles,
        min_exploration_frac=0.05,
        seed=seed,
    )

    typer.echo(f"SBED locator ready (max_steps={max_steps}). Starting adaptive scan...\n")

    # --- Run (with or without artifact tracking) ---
    t0 = time.monotonic()
    ts_str = datetime.now(UTC).isoformat()

    # Always recorded: the convergence milestones are read off the recorded run (as for every
    # simulated locator, see nvision/metrics/milestones.py), not kept on the locator.
    run_result = _run_with_observer(locator, data, drive_freq_min_phys, drive_freq_max_phys, no_progress)

    elapsed = time.monotonic() - t0

    from nvision.metrics.milestones import detect_all_converged_step, detect_milestone

    primary_idx = detect_milestone(run_result, "zeeman_shift")
    primary_converged_step = None if primary_idx is None else primary_idx + 1
    all_converged_step = detect_all_converged_step(run_result)

    # --- Final report ---
    typer.echo("")
    if primary_converged_step is not None:
        typer.echo(f"Converged at step {primary_converged_step}.")
    else:
        typer.echo("Did not converge within the step budget.")

    if all_converged_step is not None:
        typer.echo(f"All parameters converged at step {all_converged_step}.")

    shot_reuse = data.shot_reuse
    typer.echo(
        f"Total steps: {locator.step_count}  |  draws per distinct recorded shot: {shot_reuse:.1f}  |  "
        f"elapsed: {elapsed:.1f}s\n"
    )

    final_est = locator.belief.estimates()
    final_unc = locator.belief.uncertainty()

    typer.echo("Final parameter estimates:")
    for param, val in final_est.items():
        unc_val = final_unc.get(param, float("nan"))
        unit = "MHz" if "zero_field_freq" in param or "hyperfine_shift" in param or "hwhm" in param else ""
        scale = 1e-6 if unit == "MHz" else 1.0
        typer.echo(f"  {param:20s}: {val * scale:.4f} +/- {unc_val * scale:.4f} {unit}".rstrip())

    holdout_cols: dict[str, float | int] = {}
    if ref is not None:
        from nvision.metrics.lab_reference import holdout_metrics

        holdout_cols = holdout_metrics(final_est, final_unc, ref)
        typer.echo("\nHoldout check (z = (estimate - ref) / combined sigma):")
        for name in ("zeeman_shift", "zero_field_freq"):
            if f"holdout_z_{name}" in holdout_cols:
                typer.echo(f"  {name:20s}: z = {holdout_cols[f'holdout_z_{name}']:+.2f}")
            elif f"ref_{name}_at_bound" in holdout_cols:
                typer.echo(f"  {name:20s}: no z (the reference fit sits on the prior bound: not resolved)")

    noise_name = "real-holdout" if holdout else "real"
    record: dict[str, Any] = {
        "run_name": run_name,
        "noise": noise_name,
        "n_drive_freq_points": n_drive_freq_points,
        "drive_freq_min_mhz": drive_freq_min_phys / 1e6,
        "drive_freq_max_mhz": drive_freq_max_phys / 1e6,
        "n_valid_shots": data.n_valid_shots,
        "noise_std": data.noise_std,
        "max_steps": max_steps,
        "total_steps": locator.step_count,
        "primary_converged_step": primary_converged_step,
        "all_converged_step": all_converged_step,
        "elapsed_s": round(elapsed, 2),
        "shot_reuse": shot_reuse,
        "estimates": {k: float(v) for k, v in final_est.items()},
        "uncertainties": {k: float(v) for k, v in final_unc.items()},
    }
    if holdout_cols:
        record["holdout"] = holdout_cols
    if lab_curve_fit is not None:
        record["lab_curve_fit"] = {
            "values": {k: float(v) for k, v in lab_curve_fit.values_phys.items()},
            "sigmas": {k: float(v) for k, v in lab_curve_fit.sigmas_phys.items()},
            "chi2_dof": lab_curve_fit.chi2_dof,
            "at_bound": list(lab_curve_fit.at_bound),
        }
    # Claimed sigma after each measurement, physical units (Hz): one list of length total_steps per parameter.
    record["sigma_history"] = {
        name: [float(snap.belief.uncertainty()[name]) for snap in run_result.snapshots]
        for name in ("zeeman_shift", "zero_field_freq")
    }
    # Fit of the article's search-phase error model to that history (MHz), done here once so serving only reads it.
    # A fixed parameter reports sigma 0 throughout: nothing to fit.
    record["search_fit"] = {}
    for name, hist_hz in record["sigma_history"].items():
        hist_mhz = np.asarray(hist_hz, dtype=float) / 1e6
        if np.all(hist_mhz > 0):
            fit = fit_search_curve(hist_mhz)
            fit["model"] = fit["model"].tolist()
            record["search_fit"][name] = fit
        else:
            record["search_fit"][name] = None

    if not no_ui:
        gen_name = f"MATLAB:{run_name}"
        _write_artifacts(
            locator=locator,
            run_result=run_result,
            data=data,
            directory=directory,
            gen_name=gen_name,
            drive_freq_min_phys=drive_freq_min_phys,
            drive_freq_max_phys=drive_freq_max_phys,
            max_steps=max_steps,
            elapsed=elapsed,
            ts_str=ts_str,
            primary_converged_step=primary_converged_step,
            all_converged_step=all_converged_step,
            noise_name=noise_name,
            extra_metrics={"shot_reuse": shot_reuse, **holdout_cols},
        )
        typer.echo(f"\nIn `nv serve`: Study 'MATLAB (real data)', generator '{gen_name}', noise '{noise_name}'.")

    return _MatlabRunSummary(
        file_name=run_name,
        primary_converged_step=primary_converged_step,
        all_converged_step=all_converged_step,
        total_steps=locator.step_count,
        holdout_z={
            k.removeprefix("holdout_z_"): float(v) for k, v in holdout_cols.items() if k.startswith("holdout_z_")
        }
        if ref is not None
        else None,
        record=record,
    )


def _fit_holdout_reference(
    ref_data: Any,
    locator_bounds: dict[str, tuple[float, float]],
    infer_zero_field_freq: bool,
    label: str = "Reference fit on held-out sweeps",
):
    """Fit *ref_data*'s shots with the locator's model and bounds; echo the result."""
    from nvision.metrics.lab_reference import CHI2_DOF_WARN, fit_reference
    from nvision.spectra.nv_center import NV_ZERO_FIELD_SPLITTING_HZ, nv_center_lorentzian_bounds_for_domain

    lo, hi = locator_bounds["zero_field_freq"]
    # The same merge nv_center_smc_belief does: domain defaults, overridden by the locator's own bounds.
    bounds = nv_center_lorentzian_bounds_for_domain(lo, hi, hyperfine="unresolved", with_zeeman_splitting=True)
    bounds.update(locator_bounds)
    ref = fit_reference(
        ref_data.drive_freq_phys,
        ref_data.shot_ratios,
        bounds,
        fixed_zero_field_freq_phys=None if infer_zero_field_freq else NV_ZERO_FIELD_SPLITTING_HZ,
    )
    typer.echo(
        f"{label} ({ref.n_bins} bins, {ref.n_shots} shots, shot std {ref.noise_std:.4f}), chi2/dof {ref.chi2_dof:.2f}:"
    )
    for name, value in ref.values_phys.items():
        scale, unit = (1.0, "") if name == "contrast_total" else (1e-6, " MHz")
        typer.echo(f"  {name:20s}: {value * scale:.4f} +/- {ref.sigmas_phys[name] * scale:.4f}{unit}")
    if ref.chi2_dof > CHI2_DOF_WARN:
        typer.echo("  WARNING: the model does not describe the held-out data; sigma_ref and z are too optimistic.")
    typer.echo("")
    return ref


def _run_with_observer(
    locator: Locator,
    data: Any,
    drive_freq_min_phys: float,
    drive_freq_max_phys: float,
    no_progress: bool,
) -> Any:
    """Run the measurement loop with Observer tracking; return RunResult."""
    from nvision.models.observer import Observer

    proxy = _MatlabSignalProxy(drive_freq_min_phys, drive_freq_max_phys)
    observer = Observer(
        true_signal=proxy, drive_freq_min_phys=drive_freq_min_phys, drive_freq_max_phys=drive_freq_max_phys
    )
    return observer.watch(_matlab_loop(locator, data, drive_freq_min_phys, drive_freq_max_phys, no_progress))


def _write_artifacts(
    *,
    locator: Locator,
    run_result: Any,
    data: Any,
    directory: Path,
    gen_name: str,
    drive_freq_min_phys: float,
    drive_freq_max_phys: float,
    max_steps: int,
    elapsed: float,
    ts_str: str,
    noise_name: str,
    extra_metrics: dict[str, float | int],
    primary_converged_step: int | None,
    all_converged_step: int | None,
) -> Path:
    """Write the run's cache entries under *directory* (and its standalone bundle under ``<dir>/matlab/runs``)."""
    from nvision.cache import CacheBridge
    from nvision.gui.report import prepare_static_ui_data
    from nvision.runner.cache import embed_graph_content
    from nvision.runner.convert import run_result_to_finalize_record, run_result_to_history_df
    from nvision.runner.metrics import generate_attempt_metrics
    from nvision.runner.plots import generate_attempt_plots
    from nvision.sim.combinations import CombinationGrid
    from nvision.tools.artifacts import (
        merge_locator_results_with_existing,
        prepare_artifact_tree,
        write_locator_results,
        write_plots_manifest,
        write_run_status,
    )
    from nvision.viz import Viz

    matlab_ui_root = directory / MATLAB_UI_DIRNAME
    (matlab_ui_root / "cache").mkdir(parents=True, exist_ok=True)

    slug = "matlab_" + re.sub(r"[^A-Za-z0-9._-]+", "_", f"{gen_name.removeprefix('MATLAB:')}_{noise_name}")
    out_dir = matlab_ui_root / "runs" / slug
    tree = prepare_artifact_tree(out_dir)

    write_run_status(out_dir, "running", total_tasks=1, completed_tasks=0, pid=os.getpid())

    strat_name = "Bayesian-SBED"
    repeat_id = 0

    proxy = _MatlabSignalProxy(drive_freq_min_phys, drive_freq_max_phys, data=data)
    experiment = _MatlabExperiment(data, proxy, drive_freq_min_phys, drive_freq_max_phys)

    # Build DataFrames from the RunResult
    history_df = run_result_to_history_df(run_result, repeat_id, drive_freq_min_phys, drive_freq_max_phys)

    locator_result = locator.result()
    finalize_record = run_result_to_finalize_record(run_result, locator_result, repeat_id)
    finalize_record["primary_converged_step"] = primary_converged_step
    finalize_record["all_converged_step"] = all_converged_step
    finalize_record["locator_steps"] = locator.step_count
    finalize_record["duration_ms"] = elapsed * 1000
    finalize_record["converged"] = primary_converged_step is not None
    finalize_df = pl.DataFrame([finalize_record])

    stop_reason = "converged" if primary_converged_step is not None else "max_steps"

    entry_base, main_result_row, current_history_df = generate_attempt_metrics(
        n_repeats=1,
        attempt_idx_in_combo=0,
        gen_name=gen_name,
        noise_name=noise_name,
        strat_name=strat_name,
        repeat_stop_reasons=[stop_reason],
        repeat_start_times=[time.perf_counter() - elapsed],
        repeat_timestamps=[ts_str],
        current_scan=experiment,
        final_history_df=history_df,
        finalize_results=finalize_df,
        strat_obj=None,
        max_steps=max_steps,
        seed=None,
        # No run_result: its milestone and stage metrics are measured against the true signal's model and
        # parameters, which real data does not have. The milestones come from _matlab_run_one instead.
        run_result=None,
    )
    main_result_row.update(primary_converged_step=primary_converged_step, all_converged_step=all_converged_step)

    # Generate scan + Bayesian plots
    viz = Viz(out_dir=out_dir)
    plot_manifest = generate_attempt_plots(
        viz=viz,
        entry_base=entry_base,
        attempt_idx_in_combo=0,
        current_scan=experiment,
        current_history_df=current_history_df,
        noise_obj=None,
        strat_obj=None,
        slug_base=slug,
        out_dir=out_dir,
        scans_dir=tree.scans_dir,
        bayes_dir=tree.bayes_dir,
        run_result=run_result,
    )

    # Alternative "actual averages per-drive-frequency-point" view — the per-bin mean/std/min/max
    # of every shot recorded in the .mat file, independent of which bins the locator
    # actually visited (contrast with the sampled-measurements scan plot above).
    if data.signal_mean is not None and data.signal_std is not None:
        from nvision.runner.plots_data import write_matlab_drive_freq_stats_data

        stats_path = tree.scans_dir / f"{slug}_drive_freq_stats.json.gz"
        stats_bytes = write_matlab_drive_freq_stats_data(
            data.drive_freq_phys, data.signal_mean, data.signal_std, data.signal_min, data.signal_max
        )
        if stats_bytes is not None:
            stats_entry = entry_base.copy()
            stats_entry["type"] = "matlab_drive_freq_stats"
            stats_entry["path"] = str(stats_path.relative_to(out_dir))
            stats_entry["_bytes"] = stats_bytes
            plot_manifest.append(stats_entry)

    main_result_row.update(extra_metrics)

    # Write locator_results.parquet (merge with existing)
    loc_df = pl.DataFrame([main_result_row])
    loc_df = merge_locator_results_with_existing(loc_df, out_dir, log)
    write_locator_results(loc_df, out_dir)

    # Persist into a SQLite cache so `nv serve --dir` can render this run in the normal
    # results UI (it builds its manifest from a cache, never by scanning artifact trees).
    # Written to *two* places:
    #  - artifacts/matlab/cache/ — MATLAB-only, so `nv serve --dir artifacts/matlab` stays
    #    fast (13 combos list in 0.07s) regardless of how large the simulation cache gets.
    #  - artifacts/cache/ — the shared cache the main `nv serve` (no --dir) reads, so MATLAB
    #    generators show up there too, alongside the simulated grid.
    # The shared write means the *next* full manifest rebuild over there (currently ~940k
    # entries / 36 GB) picks up this run — that rebuild is already slow regardless of
    # whether MATLAB combos are in it, and 13 small entries don't meaningfully add to it.
    # Best-effort per destination: a cache-write failure must not cost us the standalone
    # bundle written above, and a failure in one destination must not skip the other.
    embedded = embed_graph_content(plot_manifest, out_dir)
    combo_key = dict(
        generator=gen_name,
        noise=noise_name,
        strategy=strat_name,
        repeats=1,
        seed=0,
        max_steps=max_steps,
        timeout_s=0,
        repeat_offset=0,
    )
    for cache_dir in (matlab_ui_root / "cache", directory / "cache"):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            bridge = CacheBridge(cache_dir)
            try:
                repo = bridge.get_cache_for_category(CombinationGrid.generator_category(gen_name))
                # Every matlab-run replaces the previous result rather than resuming it.
                # Saving over an existing combination rewrites repeat 0 but leaves its
                # pointer's updated_at untouched (it only advances when the repeat count
                # grows), and the manifest shows the most recently *updated* generation per
                # file — so a rerun would stay hidden behind any other --max-steps
                # generation saved since. Purging first makes the save write a fresh
                # pointer stamped now.
                repo.purge_cached_combination(**combo_key)
                repo.save_cached_combination(**combo_key, results=[(embedded, main_result_row)], start_idx=0)
            finally:
                bridge.close()
            log.info("Persisted MATLAB run to %s", cache_dir)
        except Exception as exc:
            log.warning("Failed to persist MATLAB run to %s (other destinations unaffected): %s", cache_dir, exc)

    # Flush each entry's in-memory plot bytes to its .json.gz path. The normal
    # `nv run` pipeline does this via the SQLite cache + restore_graphs when
    # `nv serve` starts; matlab-run writes a standalone tree with no cache, so
    # it must write these directly — write_plots_manifest only strips _bytes,
    # it never writes the file itself.
    for plot_entry in plot_manifest:
        entry_bytes = plot_entry.get("_bytes")
        entry_path = plot_entry.get("path")
        if entry_bytes is not None and entry_path:
            file_path = out_dir / entry_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(entry_bytes)

    # Write plots manifest
    write_plots_manifest(plot_manifest, out_dir)

    try:
        prepare_static_ui_data(out_dir)
    except Exception as exc:
        log.warning(f"Failed to build HTML index: {exc}")

    write_run_status(out_dir, "done", total_tasks=1, completed_tasks=1)

    return out_dir
