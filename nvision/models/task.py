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
    """A Combination plus runtime config — everything needed to execute a run."""

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
    progress_queue: Any = None
    task_id: Any = None
    strategy_spec: StrategySpec = field(init=False, repr=False)

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
