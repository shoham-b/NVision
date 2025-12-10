from __future__ import annotations

from nvision.cli.main import app

# Import subcommands to register them
from nvision.cli import cache_cmd, gui, run

__all__ = ["app"]
