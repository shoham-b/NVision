"""CLI command for running SBED on real MATLAB ESR measurements."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Generator, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from nvision.cli.app_instance import app
from nvision.models.locator import Locator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duck-typed helpers — satisfy TrueSignal / CoreExperiment interfaces without
# importing those concrete types (which carry heavy simulation dependencies).
# ---------------------------------------------------------------------------


class _MatlabSignalProxy:
    """Minimal TrueSignal duck-type for real measurements (no ground truth)."""

    model = None  # needed by some code paths that check hasattr(true_signal, 'model')

    def __init__(self, freq_lo: float, freq_hi: float) -> None:
        self._freq_lo = freq_lo
        self._freq_hi = freq_hi

    def __call__(self, x: float) -> float:
        return float("nan")

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

    def measure(self, x_unit: float, rng: Any = None):  # noqa: ANN001
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
) -> Generator[Locator, None, None]:
    """Adaptive measurement loop — yields locator state after each observation."""
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
def matlab_run(  # noqa: C901
    matlab_file: Annotated[
        Path,
        typer.Argument(help="Path to the ESR .mat file (or bare filename resolved via data/matlab/)."),
    ],
    noise_std: Annotated[
        float | None,
        typer.Option("--noise-std", help="Override the auto-estimated measurement noise std."),
    ] = None,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Maximum SBED measurement steps."),
    ] = 150,
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
) -> None:
    """Run the SBED locator on real ESR measurements from a MATLAB file.

    Loads a .mat file recorded by the NVision lab instrument, then adaptively
    selects measurement frequencies using the Bayesian SBED strategy. Results
    are written to the artifact store so they appear in the ``nvision serve`` UI.
    """
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
        f"Loaded {n_freqs} frequencies: {freq_lo/1e6:.1f} to {freq_hi/1e6:.1f} MHz  |  "
        f"valid shots: {data.n_valid_shots}  |  noise_std: {data.noise_std:.4f}"
    )

    # --- Build locator ---
    locator = SequentialBayesianExperimentDesignLocator.create(
        builder=nv_center_smc_belief,
        parameter_bounds={"frequency": (freq_lo, freq_hi)},
        noise_std=data.noise_std,
        max_steps=max_steps,
    )

    typer.echo(f"SBED locator ready (max_steps={max_steps}). Starting adaptive scan...\n")

    # --- Run (with or without artifact tracking) ---
    t0 = time.monotonic()
    ts_str = datetime.now(timezone.utc).isoformat()

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
    if locator.freq_converged_step is not None:
        typer.echo(f"Frequency converged at step {locator.freq_converged_step}.")
    else:
        typer.echo("Frequency did not converge within the step budget.")

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
        typer.echo(f"  {param:20s}: {val*scale:.4f} +/- {unc_val*scale:.4f} {unit}".rstrip())

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
        typer.echo("Open 'nvision serve' and refresh to see results in the UI.")

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
            "freq_converged_step": locator.freq_converged_step,
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


def _write_artifacts(  # noqa: C901
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
    from nvision.gui.report import prepare_static_ui_data
    from nvision.runner.convert import run_result_to_finalize_record, run_result_to_history_df
    from nvision.runner.metrics import generate_attempt_metrics
    from nvision.runner.plots import generate_attempt_plots
    from nvision.tools.artifacts import (
        merge_locator_results_with_existing,
        prepare_artifact_tree,
        write_locator_results_csv,
        write_plots_manifest,
        write_run_status,
    )
    from nvision.tools.paths import ARTIFACTS_ROOT
    from nvision.viz import Viz

    mat_stem = Path(matlab_file).stem
    slug = f"matlab_{mat_stem}"
    out_dir = ARTIFACTS_ROOT / slug
    tree = prepare_artifact_tree(out_dir)

    write_run_status(out_dir, "running", total_tasks=1, completed_tasks=0, pid=os.getpid())

    gen_name = f"MATLAB:{Path(matlab_file).name}"
    noise_name = "real"
    strat_name = "Bayesian-SBED"
    repeat_id = 0

    proxy = _MatlabSignalProxy(freq_lo, freq_hi)
    experiment = _MatlabExperiment(data, proxy, freq_lo, freq_hi)

    # Build DataFrames from the RunResult
    history_df = run_result_to_history_df(run_result, repeat_id, freq_lo, freq_hi)

    locator_result = locator.result()
    finalize_record = run_result_to_finalize_record(run_result, locator_result, repeat_id, freq_lo, freq_hi)
    finalize_record["freq_converged_step"] = locator.freq_converged_step
    finalize_record["all_converged_step"] = locator.all_converged_step
    finalize_record["locator_steps"] = locator.step_count
    finalize_record["duration_ms"] = elapsed * 1000
    finalize_df = pl.DataFrame([finalize_record])

    stop_reason = "converged" if locator.freq_converged_step is not None else "max_steps"

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
