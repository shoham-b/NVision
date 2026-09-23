"""CLI for the SBED constant optimizer. Invoke through ``run.py``:

    uv run --no-sync python openevolve/sbed_constants/run.py smoke
    uv run --no-sync python openevolve/sbed_constants/run.py tune --n-init 20 --n-iter 130
    uv run --no-sync python openevolve/sbed_constants/run.py validate --params-json <best.json>

``smoke`` is a wiring check on a deliberately tiny grid -- it proves the
render -> monkeypatch -> run -> score loop works, and its scores mean nothing
else. ``tune`` is the real run. ``validate`` is mandatory before any result is
believed; see ``validate.py``'s docstring for why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .candidate import check_baseline_render, check_template, render
from .objective import evaluate_params, scaled_grid, smoke_grid
from .optimizer import run
from .space import BASELINE_PARAMS, format_params
from .validate import head_to_head, print_report

DEFAULT_OUT = Path(__file__).resolve().parent / "output"


def _grid_for(args: argparse.Namespace):
    return scaled_grid(args.repeats) if args.repeats is not None else None


def _cmd_smoke(args: argparse.Namespace) -> int:
    check_template()
    print("template check: all 8 constant sites matched and the rendered module compiles.")

    print("\n--- structural check: initial_program.py vs its render at the default constants ---")
    for line in check_baseline_render():
        print(f"  {line}")
    print("  OK: only the header and the 8 known substitution sites changed.")
    print("  (5e6 -> 5000000.0 is the same float; the only semantic change is the half_width")
    print("   non-overlap backstop, which cannot bind while dual_halfwidth <= dual_trigger.)")

    grid = smoke_grid()
    print(f"\nsmoke grid: {grid}\n")

    print("--- baseline (unmodified initial_program.py) ---")
    baseline = evaluate_params(None, grid=grid)
    print(f"combined_score={baseline['combined_score']:.4f}  elapsed={baseline['elapsed_s']:.1f}s")

    print("\n--- rendered defaults ---")
    rendered = evaluate_params(BASELINE_PARAMS, grid=grid)
    print(f"combined_score={rendered['combined_score']:.4f}  elapsed={rendered['elapsed_s']:.1f}s")
    print("(these two are single draws on a 2-repeat grid; they will differ a lot. The structural")
    print(" diff above, not this pair of numbers, is what shows the render matches the baseline.)")

    print("\n--- optimizer loop on the smoke grid ---")
    summary = run(
        n_init=args.n_init,
        n_iter=args.n_iter,
        reeval_every=2,
        seed=args.seed,
        grid=grid,
        log_path=DEFAULT_OUT / "smoke_log.jsonl",
    )
    print("\nsummary:")
    print(json.dumps(summary, indent=2, default=float))
    print("\nNOTE: smoke scores are from a 2-repeat grid and rank nothing. Wiring check only.")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run(
        n_init=args.n_init,
        n_iter=args.n_iter,
        reeval_every=args.reeval_every,
        seed=args.seed,
        grid=_grid_for(args),
        crn_seed=args.crn_seed,
        log_path=out_dir / "tune_log.jsonl",
        resume=args.resume,
    )
    (out_dir / "best.json").write_text(json.dumps(summary["best_params"], indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out_dir / "best_program.py").write_text(render(summary["best_params"]), encoding="utf-8", newline="\n")

    print("\n=== tuning summary ===")
    print(f"evaluations      : {summary['n_observations']}")
    print(f"wall clock       : {summary['elapsed_s'] / 60:.1f} min")
    print(f"fitted noise sd  : {summary['noise_sd']}")
    print(f"best params      : {format_params(summary['best_params'])}")
    print(f"posterior mean   : {summary['best_posterior_mean']:.4f} (sd {summary['best_posterior_sd']:.4f})")
    print(f"best observed    : {summary['best_observed_score']:.4f}  <-- biased, do not report this")
    print(f"baseline in-run  : {summary['baseline_observed_mean']:.4f} over {summary['baseline_n']} eval(s)")
    print(f"\nwrote {out_dir / 'best.json'}")
    print("NEXT STEP IS MANDATORY -- re-measure head-to-head before believing any of the above:")
    print(f"  uv run --no-sync python openevolve/sbed_constants/run.py validate --params-json {out_dir / 'best.json'}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = head_to_head(
        params,
        repeats_per_arm=args.repeats_per_arm,
        grid=_grid_for(args),
        log_path=out_dir / "validate_log.jsonl",
    )
    (out_dir / "validation.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print_report(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openevolve.sbed_constants", description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--repeats", type=int, default=None, help="repeats per grid combination (default: harness's 2)")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="fast end-to-end wiring check")
    smoke.add_argument("--n-init", type=int, default=4)
    smoke.add_argument("--n-iter", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(func=_cmd_smoke)

    tune = sub.add_parser("tune", help="run the noise-aware Bayesian optimization")
    tune.add_argument("--n-init", type=int, default=20, help="initial design size (includes the baseline point)")
    tune.add_argument("--n-iter", type=int, default=130, help="model-driven iterations after the initial design")
    tune.add_argument("--reeval-every", type=int, default=4, help="re-evaluate the incumbent every N iterations")
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--crn-seed", type=int, default=None, help="optional common-random-numbers seed (search only)")
    tune.add_argument("--resume", action="store_true", help="continue from an existing tune_log.jsonl")
    tune.set_defaults(func=_cmd_tune)

    validate = sub.add_parser("validate", help="mandatory head-to-head re-measurement against the baseline")
    validate.add_argument("--params-json", required=True, help="path to a best.json written by 'tune'")
    validate.add_argument("--repeats-per-arm", type=int, default=8)
    validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    np.random.seed(None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
