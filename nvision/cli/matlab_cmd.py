"""CLI command for running SBED on real MATLAB ESR measurements."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import typer

from nvision.cli.app_instance import app
from nvision.models.locator import Locator

log = logging.getLogger(__name__)

# Artifacts subdirectory holding the MATLAB-only results cache, served with
# `nv serve --dir artifacts/matlab`. Kept apart from the shared simulation cache; see
# _write_artifacts for why. Distinct from the per-file bundles in artifacts/matlab_<stem>/.
MATLAB_UI_DIRNAME = "matlab"


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

    def __init__(self, freq_lo: float, freq_hi: float, data: Any = None) -> None:
        self._freq_lo = freq_lo
        self._freq_hi = freq_hi
        self._interp_freq: np.ndarray | None = None
        self._interp_signal: np.ndarray | None = None
        if data is not None:
            order = np.argsort(data.freq_hz)
            self._interp_freq = np.asarray(data.freq_hz, dtype=float)[order]
            signal = np.asarray(data.signal, dtype=float)[order]
            # NaN out any bin this run never actually measured. The file can hold a real
            # recorded mean for every bin, but a given run may have converged early or
            # stopped short of visiting all of them — np.interp propagates a NaN endpoint
            # across the whole segment touching it, so this breaks the curve over any
            # unvisited stretch instead of bridging it with an unmeasured value.
            visited = np.asarray(data.visited_mask, dtype=bool)[order]
            self._interp_signal = np.where(visited, signal, np.nan)

    def __call__(self, x: float) -> float:
        if self._interp_freq is None or self._interp_signal is None:
            return float("nan")
        return float(np.interp(x, self._interp_freq, self._interp_signal))

    def parameter_values(self) -> dict[str, float]:
        return {"frequency": float("nan")}

    def get_param_value(self, name: str) -> float:
        return float("nan")

    def all_bounds(self) -> dict[str, tuple[float, float]]:
        return {"frequency": (self._freq_lo, self._freq_hi)}


class _MatlabExperiment:
    """Minimal CoreExperiment duck-type backed by a MatlabDataFile."""

    def __init__(self, data: Any, true_signal: _MatlabSignalProxy, freq_lo: float, freq_hi: float) -> None:
        self.true_signal = true_signal
        self.noise = None  # real measurements: no CompositeNoise model, mirrors CoreExperiment.noise
        self.x_min = freq_lo
        self.x_max = freq_hi
        self._data = data
        self._freq_lo = freq_lo
        self._freq_hi = freq_hi

    def measure(self, x_unit: float, rng: Any = None):
        return self._data.measure(x_unit, self._freq_lo, self._freq_hi)

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
    freq_lo: float,
    freq_hi: float,
    no_progress: bool,
) -> Generator[Locator]:
    """Adaptive measurement loop — yields locator state after each observation.

    Each step measures whichever frequency the locator picks, drawing one of that
    frequency's recorded shots (see ``MatlabDataFile.measure``), and stops when the
    locator does — the run is not obliged to consume every point in the file.
    """
    while not locator.done():
        x_unit = locator.next()
        obs = data.measure(x_unit, freq_lo, freq_hi)
        locator.observe(obs)

        if not no_progress:
            phys_mhz = (freq_lo + x_unit * (freq_hi - freq_lo)) / 1e6
            est = locator.belief.estimates()
            unc = locator.belief.uncertainty()
            freq_est_mhz = est.get("frequency", float("nan")) / 1e6
            freq_unc_mhz = unc.get("frequency", float("nan")) / 1e6
            typer.echo(
                f"Step {locator.step_count:3d}: {phys_mhz:7.2f} MHz -> "
                f"signal={obs.signal_value:.4f}  "
                f"freq={freq_est_mhz:.2f} +/- {freq_unc_mhz:.2f} MHz"
            )

        yield locator


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command("matlab-run")
def matlab_run(
    matlab_file: Annotated[
        Path | None,
        typer.Argument(
            help="Path to the ESR .mat file (or bare filename resolved via data/matlab/). Omit when using --all."
        ),
    ] = None,
    all_files: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Run every .mat file in data/matlab/ (or --dir) instead of a single file. Ignores matlab_file.",
        ),
    ] = False,
    matlab_dir: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="Directory to scan for .mat files with --all (default: data/matlab/).",
        ),
    ] = None,
    noise_std: Annotated[
        float | None,
        typer.Option("--noise-std", help="Override the auto-estimated measurement noise std."),
    ] = None,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Maximum SBED measurement steps."),
    ] = 300,
    valid_shots: Annotated[
        int | None,
        typer.Option("--valid-shots", help="Number of shot columns to use (default: esr.currIter)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON result summary to this file."),
    ] = None,
    no_progress: Annotated[
        bool,
        typer.Option("--no-progress", help="Suppress per-step output."),
    ] = False,
    no_ui: Annotated[
        bool,
        typer.Option("--no-ui", help="Skip artifact writing (no web UI integration)."),
    ] = False,
    particles: Annotated[
        int,
        typer.Option("--particles", help="SMC particle count (default 10x the simulation default)."),
    ] = 10000,
    infer_frequency: Annotated[
        bool,
        typer.Option(
            "--infer-frequency/--no-infer-frequency",
            help=(
                "Fit the NV zero-field-splitting center instead of fixing it to the "
                "textbook 2.87 GHz. Real samples run 1-2 MHz off that value from strain/"
                "temperature, which is comparable to the Zeeman splitting itself — a fixed "
                "wrong center visibly distorts the fit. The simulated grid always uses the "
                "exact value it was generated with, so it doesn't need this."
            ),
        ),
    ] = True,
) -> None:
    """Run the SBED locator on real ESR measurements from a MATLAB file.

    Loads a .mat file recorded by the NVision lab instrument, then adaptively
    selects measurement frequencies using the Bayesian SBED strategy. Results
    are written to the artifact store so they appear in the ``nvision serve`` UI.

    With ``--all``, runs this same procedure over every ``.mat`` file in
    ``data/matlab/`` (or ``--dir``) in turn, continuing past any single file's
    failure so one bad recording doesn't abort the rest of the batch.
    """
    if all_files:
        from nvision.tools.matlab_loader import _MATLAB_DATA_DIR

        scan_dir = matlab_dir if matlab_dir is not None else _MATLAB_DATA_DIR
        mat_files = sorted(scan_dir.glob("*.mat"))
        if not mat_files:
            typer.echo(f"No .mat files found in {scan_dir}")
            raise typer.Exit(code=1)

        typer.echo(f"Found {len(mat_files)} .mat file(s) in {scan_dir}\n")
        failures: list[str] = []
        for i, f in enumerate(mat_files, 1):
            typer.echo(f"=== [{i}/{len(mat_files)}] {f.name} ===")
            file_out = (out / f"{f.stem}.json") if out is not None else None
            try:
                _matlab_run_one(
                    matlab_file=f,
                    noise_std=noise_std,
                    max_steps=max_steps,
                    valid_shots=valid_shots,
                    out=file_out,
                    no_progress=no_progress,
                    no_ui=no_ui,
                    particles=particles,
                    infer_frequency=infer_frequency,
                )
            except Exception as exc:
                log.exception("matlab-run failed for %s", f)
                typer.echo(f"FAILED: {f.name}: {exc}")
                failures.append(f.name)
            typer.echo("")

        n_ok = len(mat_files) - len(failures)
        typer.echo(f"Done: {n_ok}/{len(mat_files)} succeeded.")
        if failures:
            typer.echo(f"Failed: {', '.join(failures)}")
            raise typer.Exit(code=1)
        return

    if matlab_file is None:
        typer.echo("Error: provide a .mat file, or pass --all to run every file in data/matlab/.")
        raise typer.Exit(code=1)

    _matlab_run_one(
        matlab_file=matlab_file,
        noise_std=noise_std,
        max_steps=max_steps,
        valid_shots=valid_shots,
        out=out,
        no_progress=no_progress,
        no_ui=no_ui,
        particles=particles,
        infer_frequency=infer_frequency,
    )


# c_total prior for real data. The positive-only sign is the simulated grid's (the ratio
# inversion in MatlabDataFile.load makes every resonance dip), but not its (0.1, 0.4)
# magnitude cap: the raw ratio isn't a calibrated population fraction, and both files in
# data/matlab/ sat pinned against 0.4. The capped fit made up the missing depth with a
# wider line — ESR_20251112_134403 fit linewidth 3.1 MHz and R^2 0.946, against 2.2 MHz
# and 0.976 (the unconstrained scipy ceiling) with the cap lifted, where c_total settles
# at ~0.54.
# 1.0 is the physical ceiling: a dip can't go below zero signal.
_REAL_DATA_C_TOTAL_BOUNDS: tuple[float, float] = (0.1, 1.0)


@contextlib.contextmanager
def _real_data_c_total_threshold() -> Generator[None]:
    """Scale c_total's absolute convergence threshold to ``_REAL_DATA_C_TOTAL_BOUNDS``.

    NVISION_C_TOTAL_CONVERGENCE_THRESHOLD (0.01) is absolute and was calibrated against
    the simulated grid's (0.1, 0.4) range. Left as-is under the wider real-data range it
    demands a precision the recorded shots can't give (the 205-shot file never reported
    all-parameter convergence), so keep it the same *fraction of the prior range* for the
    duration of the run, and restore it afterwards so nothing else in the process sees it.
    """
    from nvision.sim.defaults import PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS
    from nvision.spectra.nv_center import (
        DEFAULT_NV_CENTER_FREQ_X_MAX,
        DEFAULT_NV_CENTER_FREQ_X_MIN,
        nv_center_lorentzian_bounds_for_domain,
    )

    sim_lo, sim_hi = nv_center_lorentzian_bounds_for_domain(
        DEFAULT_NV_CENTER_FREQ_X_MIN,
        DEFAULT_NV_CENTER_FREQ_X_MAX,
        with_hyperfine_splitting=False,
        with_zeeman_splitting=True,
    )["c_total"]
    real_lo, real_hi = _REAL_DATA_C_TOTAL_BOUNDS
    original = PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS["c_total"]
    PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS["c_total"] = original * (real_hi - real_lo) / (sim_hi - sim_lo)
    try:
        yield
    finally:
        PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS["c_total"] = original


@_real_data_c_total_threshold()
def _matlab_run_one(
    *,
    matlab_file: Path,
    noise_std: float | None,
    max_steps: int,
    valid_shots: int | None,
    out: Path | None,
    no_progress: bool,
    no_ui: bool,
    particles: int,
    infer_frequency: bool,
) -> None:
    """Run the SBED locator on a single ESR .mat file (the body of ``matlab-run``)."""
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
    from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
    from nvision.tools.matlab_loader import MatlabDataFile

    # --- Load data ---
    typer.echo(f"Loading: {matlab_file}")
    data = MatlabDataFile.load(matlab_file, valid_shots=valid_shots, noise_std_override=noise_std)

    freq_lo = float(data.freq_hz.min())
    freq_hi = float(data.freq_hz.max())
    n_freqs = len(data.freq_hz)

    typer.echo(
        f"Loaded {n_freqs} frequencies: {freq_lo / 1e6:.1f} to {freq_hi / 1e6:.1f} MHz  |  "
        f"valid shots: {data.n_valid_shots}  |  noise_std: {data.noise_std:.4f}"
    )

    # --- Build locator ---
    # c_total: real .mat signal ratios used to come out peaking above 1.0 instead of
    # dipping below it (whichever shot channel the instrument calls "baseline" can land on
    # either side of the driven channel). That inversion is corrected once, uniformly, in
    # MatlabDataFile.load (baseline / with_freq instead of with_freq / baseline), so the
    # positive-only sign applies directly — but the magnitude cap doesn't; see
    # _REAL_DATA_C_TOTAL_BOUNDS.
    #
    # frequency is inferred (--infer-frequency), not pinned to the simulated grid's 2870 MHz:
    # both files' sweeps are centered on 2870 MHz, but their doublets sit at 2871.62 and
    # 2871.13 MHz, and pinning drops R^2 from 0.976 to 0.824 on the narrower file.
    #
    # zeeman_split's default ceiling is MAX_ZEEMAN_SPLIT (60 MHz), sized for the simulated
    # domain and wider than some real sweeps (one file here spans only 80 MHz in total).
    # Left alone it lets the fit park one of the two peaks outside the measured window and
    # "explain" only the half it can see — which is what the narrower file did: center
    # 2840 MHz, split 51.7 MHz, lower peak at 2789 MHz against data starting at 2830 MHz.
    # A quarter of the span keeps the doublet inside a sweep that was deliberately
    # recorded around it. Note this cannot be done by narrowing the "frequency" bound
    # instead: nv_center_smc_belief reuses that same entry as the probe x-domain, so
    # tightening it would stop the locator from ever measuring the wings.
    span = freq_hi - freq_lo
    locator_bounds = {
        "frequency": (freq_lo, freq_hi),
        "zeeman_split": (0.0, span / 4.0),
        "c_total": _REAL_DATA_C_TOTAL_BOUNDS,
    }

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
        noise_std=data.noise_std,
        max_steps=max_steps,
        with_fixed_frequency=not infer_frequency,
        num_particles=particles,
        min_exploration_frac=0.05,
    )

    typer.echo(f"SBED locator ready (max_steps={max_steps}). Starting adaptive scan...\n")

    # --- Run (with or without artifact tracking) ---
    t0 = time.monotonic()
    ts_str = datetime.now(UTC).isoformat()

    if not no_ui:
        run_result = _run_with_observer(locator, data, freq_lo, freq_hi, no_progress)
    else:
        # Plain loop — no artifact tracking
        for _ in _matlab_loop(locator, data, freq_lo, freq_hi, no_progress):
            pass
        run_result = None

    elapsed = time.monotonic() - t0

    # --- Final report ---
    typer.echo("")
    if locator.splitting_converged_step is not None:
        typer.echo(f"Converged at step {locator.splitting_converged_step}.")
    else:
        typer.echo("Did not converge within the step budget.")

    if locator.all_converged_step is not None:
        typer.echo(f"All parameters converged at step {locator.all_converged_step}.")

    typer.echo(f"Total steps: {locator.step_count}  |  elapsed: {elapsed:.1f}s\n")

    final_est = locator.belief.estimates()
    final_unc = locator.belief.uncertainty()

    typer.echo("Final parameter estimates:")
    for param, val in final_est.items():
        unc_val = final_unc.get(param, float("nan"))
        unit = "MHz" if "frequency" in param or "split" in param or "linewidth" in param else ""
        scale = 1e-6 if unit == "MHz" else 1.0
        typer.echo(f"  {param:20s}: {val * scale:.4f} +/- {unc_val * scale:.4f} {unit}".rstrip())

    # --- Write artifacts ---
    if run_result is not None:
        out_dir = _write_artifacts(
            locator=locator,
            run_result=run_result,
            data=data,
            matlab_file=matlab_file,
            freq_lo=freq_lo,
            freq_hi=freq_hi,
            max_steps=max_steps,
            elapsed=elapsed,
            ts_str=ts_str,
        )
        typer.echo(f"\nArtifacts written to: {out_dir}")
        typer.echo(
            f"View in the results UI:  uv run nv serve --dir artifacts/{MATLAB_UI_DIRNAME}\n"
            "  -> http://localhost:18083 (its own port; 18080 is the main artifacts UI)\n"
            f"  -> Study: 'Default (ungrouped)', generator 'MATLAB:{Path(matlab_file).name}'\n"
            "  -> Press 'r' there to pick up later runs."
        )

    # --- Optional JSON output ---
    if out is not None:
        result = {
            "file": str(matlab_file),
            "n_freqs": n_freqs,
            "freq_lo_mhz": freq_lo / 1e6,
            "freq_hi_mhz": freq_hi / 1e6,
            "n_valid_shots": data.n_valid_shots,
            "noise_std": data.noise_std,
            "max_steps": max_steps,
            "total_steps": locator.step_count,
            "splitting_converged_step": locator.splitting_converged_step,
            "all_converged_step": locator.all_converged_step,
            "elapsed_s": round(elapsed, 2),
            "estimates": {k: float(v) for k, v in final_est.items()},
            "uncertainties": {k: float(v) for k, v in final_unc.items()},
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        typer.echo(f"\nResults written to: {out}")


def _run_with_observer(
    locator: Locator,
    data: Any,
    freq_lo: float,
    freq_hi: float,
    no_progress: bool,
) -> Any:
    """Run the measurement loop with Observer tracking; return RunResult."""
    from nvision.models.observer import Observer

    proxy = _MatlabSignalProxy(freq_lo, freq_hi)
    observer = Observer(true_signal=proxy, x_min=freq_lo, x_max=freq_hi)
    return observer.watch(_matlab_loop(locator, data, freq_lo, freq_hi, no_progress))


def _write_artifacts(
    *,
    locator: Locator,
    run_result: Any,
    data: Any,
    matlab_file: Path,
    freq_lo: float,
    freq_hi: float,
    max_steps: int,
    elapsed: float,
    ts_str: str,
) -> Path:
    """Write locator_results.csv, plots_manifest.json, and Bayesian plots."""
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
        write_locator_results_csv,
        write_plots_manifest,
        write_run_status,
    )
    from nvision.tools.paths import ARTIFACTS_ROOT
    from nvision.viz import Viz

    matlab_ui_root = ARTIFACTS_ROOT / MATLAB_UI_DIRNAME
    (matlab_ui_root / "cache").mkdir(parents=True, exist_ok=True)

    mat_stem = Path(matlab_file).stem
    slug = f"matlab_{mat_stem}"
    out_dir = ARTIFACTS_ROOT / slug
    tree = prepare_artifact_tree(out_dir)

    write_run_status(out_dir, "running", total_tasks=1, completed_tasks=0, pid=os.getpid())

    gen_name = f"MATLAB:{Path(matlab_file).name}"
    noise_name = "real"
    strat_name = "Bayesian-SBED"
    repeat_id = 0

    proxy = _MatlabSignalProxy(freq_lo, freq_hi, data=data)
    experiment = _MatlabExperiment(data, proxy, freq_lo, freq_hi)

    # Build DataFrames from the RunResult
    history_df = run_result_to_history_df(run_result, repeat_id, freq_lo, freq_hi)

    locator_result = locator.result()
    finalize_record = run_result_to_finalize_record(run_result, locator_result, repeat_id, freq_lo, freq_hi)
    finalize_record["splitting_converged_step"] = locator.splitting_converged_step
    finalize_record["all_converged_step"] = locator.all_converged_step
    finalize_record["locator_steps"] = locator.step_count
    finalize_record["duration_ms"] = elapsed * 1000
    finalize_df = pl.DataFrame([finalize_record])

    stop_reason = "converged" if locator.splitting_converged_step is not None else "max_steps"

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
        run_result=run_result,
    )

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

    # Write locator_results.csv (merge with existing)
    loc_df = pl.DataFrame([main_result_row])
    loc_df = merge_locator_results_with_existing(loc_df, out_dir, log)
    write_locator_results_csv(loc_df, out_dir)

    # Persist into a SQLite cache of MATLAB runs only, so `nv serve --dir` can render them
    # in the normal results UI (it builds its manifest from a cache, never by scanning
    # artifact trees). Deliberately NOT the shared artifacts/cache/: that one also holds
    # the simulated grid, which reached ~940k index entries over 36 GB here, and a manifest
    # rebuild across it takes ~20 minutes and several GB of RAM — for 13 MATLAB combos
    # that list in 0.07s on their own. Keeping them separate also means a reload of this UI
    # never touches, or waits on, the simulation cache. Best-effort: a cache-write failure
    # must not cost us the standalone bundle written above.
    try:
        bridge = CacheBridge(matlab_ui_root / "cache")
        try:
            repo = bridge.get_cache_for_category(CombinationGrid.generator_category(gen_name))
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
            # Every matlab-run replaces the previous result rather than resuming it. Saving
            # over an existing combination rewrites repeat 0 but leaves its pointer's
            # updated_at untouched (it only advances when the repeat count grows), and the
            # manifest shows the most recently *updated* generation per file — so a rerun
            # would stay hidden behind any other --max-steps generation saved since. Purging
            # first makes the save write a fresh pointer stamped now.
            repo.purge_cached_combination(**combo_key)
            repo.save_cached_combination(**combo_key, results=[(embedded, main_result_row)], start_idx=0)
        finally:
            bridge.close()
        log.info("Persisted MATLAB run to %s", matlab_ui_root / "cache")
    except Exception as exc:
        log.warning("Failed to persist MATLAB run to its cache (standalone bundle unaffected): %s", exc)

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
