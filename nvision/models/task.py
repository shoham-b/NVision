from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nvision.models.locator import Locator
    from nvision.sim.combinations import Combination


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Normalized strategy descriptor used by the executor.

    ``locator_class`` is always a ``Locator`` subclass.
    ``locator_config`` is always a plain dict.
    """

    locator_class: type[Locator]
    locator_config: dict[str, Any]
    raw: Any

    @classmethod
    def from_raw(cls, strategy: Any) -> StrategySpec:
        from nvision.models.locator import Locator

        if isinstance(strategy, type) and issubclass(strategy, Locator):
            return cls(locator_class=strategy, locator_config={}, raw=strategy)
        if isinstance(strategy, dict):
            locator_class = strategy.get("class")
            locator_config = strategy.get("config", {})
            if isinstance(locator_class, type) and issubclass(locator_class, Locator):
                return cls(locator_class=locator_class, locator_config=dict(locator_config), raw=strategy)
            raise TypeError("Strategy dict must have 'class' as a Locator subclass")
        raise TypeError(f"Expected Locator class or dict strategy, got {type(strategy)}")


@dataclass(slots=True)
class LocatorTask:
    """A Combination plus runtime config — everything needed to execute a run.

    When a combination's repeats are split across multiple sub-tasks (for
    work-stealing parallelisation), each sub-task carries a ``repeat_offset``
    so that the executor knows which global repeat indices it owns without
    needing a separate total_repeats field.

    ``shard_index`` is the cross-pod counterpart: when the whole run is sharded
    across separate hosts (see ``nv run --shard-index/--shard-count``), it
    identifies which shard built this task, and is threaded into the worker's
    ``CacheBridge`` so the MySQL backend writes to that shard's own table
    (see ``nvision.cache.mysql.MySqlCache``). It travels on the task itself
    (not a parent-process env var) because ``ProcessPoolExecutor`` workers on
    Windows use spawn, not fork, and don't inherit parent process state.
    """

    combination: Combination
    repeats: int
    seed: int
    slug: str
    out_dir: Path
    scans_dir: Path
    bayes_dir: Path
    loc_max_steps: int
    sweep_max_steps: int | None
    loc_timeout_s: int
    use_cache: bool
    cache_dir: Path
    log_queue: Any
    log_level: int
    ignore_cache_strategy: str | None
    require_cache: bool = False
    dry_run: bool = False
    progress_queue: Any = None
    task_id: Any = None
    strategy_spec: StrategySpec = field(init=False, repr=False)
    repeat_offset: int = 0
    repeat_total: int = 0
    shard_index: str | None = None

    def __post_init__(self) -> None:
        self.strategy_spec = StrategySpec.from_raw(self.combination.strategy)

    @property
    def generator_name(self) -> str:
        return self.combination.generator_name

    @property
    def generator(self) -> object:
        return self.combination.generator

    @property
    def noise_name(self) -> str:
        return self.combination.noise_name

    @property
    def noise(self) -> Any:
        return self.combination.noise

    @property
    def strategy_name(self) -> str:
        return self.combination.strategy_name

    @property
    def strategy(self) -> Any:
        return self.combination.strategy

    def __str__(self) -> str:
        return self.slug
