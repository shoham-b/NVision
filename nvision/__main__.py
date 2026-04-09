from __future__ import annotations

# Ensure all subcommands are registered on import.
# Imported for side-effects only.
from nvision.cli import cache_cmd as _cache_cmd  # noqa: F401
from nvision.cli import cases_cmd as _cases_cmd  # noqa: F401
from nvision.cli import render as _render  # noqa: F401
from nvision.cli import run as _run  # noqa: F401
from nvision.cli.main import app


def main() -> None:
    """Run the Typer CLI."""
    app()


__all__ = ["app", "main"]

if __name__ == "__main__":  # pragma: no cover
    main()
