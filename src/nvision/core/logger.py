from __future__ import annotations

import logging
from logging.handlers import QueueHandler
from typing import Any

log = logging.getLogger("nvision")


def _ensure_worker_queue_logging(queue: Any, level: int) -> None:
    """Attach a multiprocessing QueueHandler exactly once per worker process."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_nvision_queue_handler_attached", False):
        root_logger.setLevel(level)
        return

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    queue_handler = QueueHandler(queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(level)
    root_logger._nvision_queue_handler_attached = True  # type: ignore[attr-defined]


def setup_logging(level_name: str = "INFO") -> None:
    """Convert string level to logging level int."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level)
    # Further configuration could go here if we weren't using RichHandler via Typer/Rich
