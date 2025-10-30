from __future__ import annotations

import typer

from nvision.cli import cli


def main() -> None:
    """Run the Typer CLI."""
    typer.run(cli)


__all__ = []

if __name__ == "__main__":  # pragma: no cover
    main()
