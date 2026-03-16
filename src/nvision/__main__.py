from __future__ import annotations

import sys

from nvision.cli.main import app

# Ensure all subcommands are registered on import.
# Imported for side-effects only.
from nvision.cli import cache_cmd as _cache_cmd  # noqa: F401
from nvision.cli import gui as _gui  # noqa: F401
from nvision.cli import render as _render  # noqa: F401
from nvision.cli import run as _run  # noqa: F401


def main() -> None:
    """Run the Typer CLI."""
    app()


__all__ = ["main", "app"]

if __name__ == "__main__":  # pragma: no cover
    main()
