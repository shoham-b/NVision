"""OpenEvolve evaluator for the SBED acquisition-function evolve target.

Loads a candidate program (see ``initial_program.py``), monkeypatches its
``_acquire`` / ``_eig_acquire`` / ``_dual_window_acquire`` onto the real
``SequentialBayesianExperimentDesignLocator`` class, runs a small in-process
evaluation grid directly through ``nvision.runner`` (no CLI/subprocess, no
cache writes -- mirrors the ``nv run-single --dry-run`` verification pattern
from AGENTS.md), and scores candidates on splitting-convergence sample
efficiency.

Per ``.claude/skills/locator-evaluation``, ``splitting_converged_step`` (not
the locator's own stop reason) is the metric that matters, and a speed
metric alone is a documented trap in this codebase's history (see project
memory ``sbed-voigt-fit-quality-fixes.md`` and the CRLB early-stop comments
in ``sbed_locator.py``): several past "improvements" here turned out to be
early-stop rules exploiting the benchmark's truth-centred prior rather than
real gains. So a candidate is only scored on speed *after* clearing hard
gates on convergence rate and catastrophic error rate -- a candidate that
converges faster by sacrificing correctness scores 0, not a partial credit.

Known limitation: the locator's own acquisition randomness
(``np.random.rand()``/``uniform()``/``choice()`` inside ``_acquire``) is not
seeded per repeat in this codebase -- only the ground-truth signal draw is
(see ``NVISION_RNG_SEED``). So re-evaluating the *same* candidate twice gives
different (though statistically similar) scores. This is a property of the
existing simulation harness, not something this evaluator works around;
``EVAL_GRID``'s repeat counts are chosen to average out most of that noise
without making each evaluation too slow.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import queue
import statistics
import sys
import tempfile
import uuid
from pathlib import Path

# Must be set before nvision.sim.defaults is first imported anywhere (it reads
# this env var once, at module load, into a module-level constant). The
# feasibility gate is an oracle check computed from the true signal params
# before any measurement is taken, independent of the acquisition strategy
# under evolution -- at its default margin (1.0) it skips a large fraction of
# repeats outright (0 measurements) for this generator/locator combination,
# which is pure noise for a search that can only ever affect what happens
# *after* that gate. Raising it lets those repeats actually run so the
# evolved strategy is judged on them instead of losing the sample.
os.environ.setdefault("NVISION_CRLB_FEASIBILITY_MARGIN", "50")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.evaluation_result import EvaluationResult  # noqa: E402

logging.getLogger("nvision").setLevel(logging.ERROR)

# "Catastrophic" matches the bar already used in sbed_locator.py's own
# CRLB-early-stop comments: a final zeeman_split error over 1 MHz is a wrong
# answer, not an imprecise one.
CATASTROPHIC_THRESHOLD_HZ = 1.0e6
CATASTROPHIC_CEILING = 0.25
CONVERGENCE_FLOOR = 0.5
# Divisor in the speed term below; ~30-50 steps is a typical splitting-converged
# step count for this generator at low noise (see openevolve/sbed_acquisition/README.md).
STEPS_SCALE = 60.0

# (generator, noise, repeats). Kept small and noise-light so one candidate
# evaluates in well under a minute: Gauss(0.1)+ was measured at ~70s/repeat
# and frequently fails to converge within budget at all (see README), which
# would make it pure noise for a fast search loop, not a harder test.
EVAL_GRID: list[tuple[str, str, int]] = [
    ("NVCenter-lorentzian", "Gauss(0.0)", 4),
    ("NVCenter-lorentzian", "Gauss(0.05)", 4),
]
REPEAT_TIMEOUT_S = 45


def _load_candidate(program_path: str):
    spec = importlib.util.spec_from_file_location(f"sbed_candidate_{uuid.uuid4().hex}", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_acquire", "_eig_acquire", "_dual_window_acquire"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"candidate program is missing a callable '{name}'")
    return module


def _apply_patch(module):
    from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator as Locator

    names = ("_acquire", "_eig_acquire", "_dual_window_acquire")
    originals = {name: getattr(Locator, name) for name in names}
    for name in names:
        setattr(Locator, name, getattr(module, name))
    return Locator, originals


def _restore(locator_cls, originals: dict) -> None:
    for name, fn in originals.items():
        setattr(locator_cls, name, fn)


def _run_combo(generator: str, noise: str, repeats: int) -> list[dict]:
    from nvision.models.task import LocatorTask
    from nvision.runner.executor import run_task
    from nvision.sim.combinations import CombinationGrid
    from nvision.tools.artifacts import prepare_artifact_tree
    from nvision.tools.utils import NVISION_RNG_SEED

    with tempfile.TemporaryDirectory(prefix="openevolve_sbed_eval_") as tmp:
        out_dir = Path(tmp)
        tree = prepare_artifact_tree(out_dir, clear_cache=True)
        grid = CombinationGrid()
        combo = grid.resolve(generator, noise, "Bayesian-SBED")
        if combo is None:
            raise ValueError(f"unknown combination {generator}/{noise}/Bayesian-SBED")

        task = LocatorTask(
            combination=combo,
            repeats=repeats,
            seed=NVISION_RNG_SEED,
            slug="openevolve-eval",
            out_dir=out_dir,
            scans_dir=tree.scans_dir,
            bayes_dir=tree.bayes_dir,
            loc_max_steps=500,
            sweep_max_steps=None,
            loc_timeout_s=REPEAT_TIMEOUT_S,
            use_cache=False,
            cache_dir=tree.cache_dir,
            log_queue=queue.Queue(-1),
            log_level=logging.ERROR,
            ignore_cache_strategy=None,
            require_cache=False,
            dry_run=True,
            progress_queue=queue.Queue(),
        )
        results = run_task(task)
        return [row for _entries, row in results]


def _score(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "combined_score": 0.0,
            "convergence_rate": 0.0,
            "catastrophic_rate": 1.0,
            "median_steps": math.inf,
        }

    converged_steps: list[int] = []
    catastrophic = 0
    for row in rows:
        step = row.get("splitting_converged_step")
        err = row.get("final_err_fc")
        if step is not None:
            converged_steps.append(step)
        if err is not None and err > CATASTROPHIC_THRESHOLD_HZ:
            catastrophic += 1

    n = len(rows)
    convergence_rate = len(converged_steps) / n
    catastrophic_rate = catastrophic / n

    if convergence_rate < CONVERGENCE_FLOOR or catastrophic_rate > CATASTROPHIC_CEILING:
        combined_score = 0.0
    else:
        median_steps = statistics.median(converged_steps)
        combined_score = convergence_rate / (1.0 + median_steps / STEPS_SCALE)

    return {
        "combined_score": combined_score,
        "convergence_rate": convergence_rate,
        "catastrophic_rate": catastrophic_rate,
        "median_steps": statistics.median(converged_steps) if converged_steps else float("nan"),
    }


def _evaluate_grid(program_path: str, grid: list[tuple[str, str, int]]) -> EvaluationResult:
    try:
        module = _load_candidate(program_path)
    except Exception as exc:
        return EvaluationResult(metrics={"combined_score": 0.0, "error": str(exc)[:500]})

    locator_cls, originals = _apply_patch(module)
    try:
        rows: list[dict] = []
        for generator, noise, repeats in grid:
            rows.extend(_run_combo(generator, noise, repeats))
    except Exception as exc:
        return EvaluationResult(metrics={"combined_score": 0.0, "error": str(exc)[:500]})
    finally:
        _restore(locator_cls, originals)

    return EvaluationResult(metrics=_score(rows))


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Cheap smoke test on the easy noiseless combo only. Not enabled by default
    (see config.yaml's ``cascade_evaluation: false``) but available if you want
    the speed/accuracy tradeoff of cascading -- see README."""
    return _evaluate_grid(program_path, EVAL_GRID[:1])


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation grid, used as the cascade's final stage."""
    return _evaluate_grid(program_path, EVAL_GRID)


def evaluate(program_path: str) -> EvaluationResult:
    """Non-cascade entry point (used by default; see config.yaml)."""
    return _evaluate_grid(program_path, EVAL_GRID)
