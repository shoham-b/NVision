from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LocatorTask:
    generator_name: str
    generator: Any
    noise_name: str
    noise: Any  # CompositeNoise | None
    strategy_name: str
    strategy: Any
    repeats: int
    seed: int
    slug: str
    out_dir: Path
    scans_dir: Path
    bayes_dir: Path
    loc_max_steps: int
    loc_timeout_s: int
    use_cache: bool
    cache_dir: Path
    log_queue: Any
    log_level: int
    ignore_cache_strategy: str | None
    require_cache: bool = False
    progress_queue: Any = None
    task_id: Any = None

    def __str__(self) -> str:
        return self.slug
