"""One noisy evaluation of a constant vector, reusing the existing harness.

This module deliberately owns **no** scoring logic. The anytime log-error curve,
the calibration and catastrophic gates and the ``combined_score`` composition all
live in ``openevolve/sbed_acquisition/evaluator.py`` and are imported from there,
so a number produced here is directly comparable with the numbers the previous
LLM-driven search produced (baseline mean 0.521--0.585) and with anything that
harness produces in future. Re-implementing the score would quietly fork the
definition and make every historical comparison meaningless.

``evaluator.py`` is loaded **by path**, not as ``openevolve.sbed_acquisition``:
that directory is not a package, and OpenEvolve itself loads it by path too.

Import ordering matters. ``evaluator.py`` sets
``NVISION_CRLB_FEASIBILITY_MARGIN`` at module scope and that must happen before
``nvision.sim.defaults`` is imported anywhere (it snapshots the variable once, at
module load). Loading the evaluator is therefore the first thing this module
does, and nothing here imports ``nvision`` at module scope.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .candidate import BASELINE_PROGRAM, write_candidate

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = REPO_ROOT / "openevolve" / "sbed_acquisition" / "evaluator.py"

_evaluator = None


def evaluator():
    """Load (once) and return the ``sbed_acquisition`` evaluator module."""
    global _evaluator
    if _evaluator is None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        spec = importlib.util.spec_from_file_location("sbed_acquisition_evaluator", EVALUATOR_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load the evaluator from {EVALUATOR_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _evaluator = module
    return _evaluator


def default_grid() -> list[tuple[str, str, int]]:
    """The full evaluation grid the sbed_acquisition harness uses (54 repeats)."""
    return list(evaluator().EVAL_GRID)


def smoke_grid() -> list[tuple[str, str, int]]:
    """A deliberately tiny grid for wiring checks only.

    Far too small to rank candidates -- it exists to prove the render -> patch ->
    run -> score loop works end to end without spending 35 s per evaluation.
    """
    grid = evaluator().EVAL_GRID
    return [(grid[0][0], grid[0][1], 1), (grid[-1][0], grid[-1][1], 1)]


def scaled_grid(repeats: int) -> list[tuple[str, str, int]]:
    """The full combination list at a different repeat count per combination."""
    return [(generator, noise, repeats) for generator, noise, _ in evaluator().EVAL_GRID]


def evaluate_program(
    program_path: str | Path,
    grid: list[tuple[str, str, int]] | None = None,
    crn_seed: int | None = None,
) -> dict[str, float]:
    """Patch ``program_path`` onto the locator, run ``grid``, return metrics.

    ``crn_seed`` is an *optional* common-random-numbers control: when set, the
    global numpy RNG is re-seeded before each combination, so two different
    candidates evaluated at the same seed start each combination from the same
    stream. It reduces between-candidate variance but does not eliminate it --
    candidates consume different numbers of draws within a combination, so the
    streams desynchronise as a repeat proceeds. Treat it as variance reduction
    during search only: **final validation must be run unseeded**, or you are
    measuring performance on one particular random stream rather than in
    expectation.
    """
    ev = evaluator()
    grid = list(grid) if grid is not None else default_grid()

    module = ev._load_candidate(str(program_path))
    locator_cls, originals = ev._apply_patch(module)
    try:
        records: list[tuple[list[dict], dict]] = []
        for index, (generator, noise, repeats) in enumerate(grid):
            if crn_seed is not None:
                np.random.seed((crn_seed + 7919 * index) % (2**32))
            records.extend(ev._run_combo(generator, noise, repeats))
    finally:
        ev._restore(locator_cls, originals)
    return dict(ev._score(records))


def evaluate_params(
    params: dict[str, float] | None,
    grid: list[tuple[str, str, int]] | None = None,
    crn_seed: int | None = None,
    keep_dir: str | Path | None = None,
) -> dict[str, float]:
    """Render ``params`` into a candidate program and evaluate it.

    ``params is None`` means the **unmodified baseline**: the checked-in
    ``initial_program.py`` is used verbatim, with no rendering at all. That is
    the arm a winner has to beat, and it must not be a rendered default -- a
    rendered default shares any bug the renderer might have, which would cancel
    out of the comparison and hide it.
    """
    started = time.perf_counter()
    if params is None:
        metrics = evaluate_program(BASELINE_PROGRAM, grid=grid, crn_seed=crn_seed)
        metrics["elapsed_s"] = time.perf_counter() - started
        return metrics

    # mkdtemp + rmtree(ignore_errors=True), never TemporaryDirectory as a context
    # manager: on Windows a still-open file makes the manager's cleanup raise
    # WinError 32 *out of a successful evaluation*, which the caller can only read
    # as a failed candidate. Same reasoning as evaluator._run_combo -- see the long
    # comment there. A leaked temp file is harmless; a spuriously-zeroed candidate
    # poisons the surrogate model for the rest of the run.
    scratch = Path(keep_dir) if keep_dir is not None else Path(tempfile.mkdtemp(prefix="sbed_const_src_"))
    try:
        path = write_candidate(params, directory=scratch)
        metrics = evaluate_program(path, grid=grid, crn_seed=crn_seed)
    finally:
        if keep_dir is None:
            shutil.rmtree(scratch, ignore_errors=True)
    metrics["elapsed_s"] = time.perf_counter() - started
    return metrics
