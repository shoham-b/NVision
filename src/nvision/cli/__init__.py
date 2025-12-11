from __future__ import annotations

# Import subcommands to register them
from nvision.cli import cache_cmd, gui, run  # noqa: F401
from nvision.cli.main import app

__all__ = ["app"]
