"""Launcher for the SBED constant optimizer.

    uv run --no-sync python openevolve/sbed_constants/run.py smoke
    uv run --no-sync python openevolve/sbed_constants/run.py tune --n-init 20 --n-iter 130
    uv run --no-sync python openevolve/sbed_constants/run.py validate --params-json <best.json>

This wrapper exists because ``python -m openevolve.sbed_constants`` does **not**
reach this directory. ``openevolve`` is an installed regular package in
``.venv/site-packages``, and a regular package always wins over a namespace
portion of the same name, so ``openevolve.*`` resolves into site-packages no
matter what is on ``sys.path``. Putting the *parent* directory on the path
instead makes this a top-level ``sbed_constants`` package, which resolves
unambiguously. The neighbouring ``sbed_acquisition/run.py`` uses the same
``sys.path``-insert-then-import pattern for the same underlying reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from sbed_constants.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
