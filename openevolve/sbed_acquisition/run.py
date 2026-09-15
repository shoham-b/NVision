"""Launcher for the SBED-acquisition OpenEvolve run, using the Windows-safe
stdin-based Claude Code CLI backend (claude_code_stdin.py) instead of the
argv-based one that ships with openevolve.

Needed because `openevolve-run` only accepts a YAML config, and wiring in a
custom Python LLM backend requires setting `LLMModelConfig.init_client`,
which YAML cannot express. See claude_code_stdin.py's docstring for why this
backend exists (Windows command-line length limit).

Usage (same flags as `openevolve-run`, minus the positional program/evaluator
paths, which are fixed to this directory's files):

    uv run python openevolve/sbed_acquisition/run.py --iterations 30
    uv run python openevolve/sbed_acquisition/run.py --iterations 3   # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Must be imported at module level, not inside main()/the __main__ guard below:
# ProcessPoolExecutor workers on Windows use 'spawn', which re-executes this
# file from the top (everything above `if __name__ == "__main__":`) to
# reconstruct state needed to unpickle objects sent to the child -- including
# the init_client callable stashed on the config in build_config() below,
# which is pickled as a (module name, qualified name) reference. The worker
# can only resolve that reference if this same import has already happened
# by the time unpickling occurs, which this ordering guarantees.
import claude_code_stdin  # noqa: E402


def build_config(config_path: str):
    from openevolve.config import load_config

    config = load_config(config_path)
    for model_cfg in config.llm.models + config.llm.evaluator_models:
        if getattr(model_cfg, "provider", None) == "claude_code":
            model_cfg.init_client = claude_code_stdin.init_claude_code_stdin_client
    return config


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SBED-acquisition OpenEvolve search with a Windows-safe LLM backend."
    )
    parser.add_argument("--config", "-c", default=str(HERE / "config.yaml"))
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--iterations", "-i", type=int, default=None)
    parser.add_argument("--target-score", "-t", type=float, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    from openevolve import OpenEvolve

    config = build_config(args.config)
    openevolve = OpenEvolve(
        initial_program_path=str(HERE / "initial_program.py"),
        evaluation_file=str(HERE / "evaluator.py"),
        config=config,
        output_dir=args.output,
    )
    if args.checkpoint:
        openevolve.database.load(args.checkpoint)

    best_program = await openevolve.run(
        iterations=args.iterations,
        target_score=args.target_score,
        checkpoint_path=args.checkpoint,
    )

    print("\nEvolution complete!")
    print("Best program metrics:")
    for name, value in best_program.metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
