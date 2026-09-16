# Classmethod Factory Pattern

## Overview

Locators use a `create()` classmethod for instantiation rather than a separate
`LocatorFactory` class. This is simpler, more Pythonic, and eliminates the factory
layer. This is still exactly how it works today — see `create()` on
`SbedLocator`, `SequentialBayesianLocator`, `SobolBayesianLocator`,
`GenericSweepLocator`, and `StagedSobolSweepLocator`.

---

## Before (Factory Pattern, historical)

```python
class SimpleSweepFactory(LocatorFactory):
    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps

    def create(self) -> Locator:
        belief = build_belief(...)
        return SimpleSweepLocator(belief, self.max_steps)

# Usage
factory = SimpleSweepFactory(max_steps=30)
locator = factory.create()
```

**Issues:**
- Extra layer of indirection
- Factory class just wraps configuration
- Not idiomatic Python

---

## After (Classmethod Pattern)

```python
class SimpleSweepLocator(Locator):
    @classmethod
    def create(cls, max_steps: int = 50, **kwargs) -> SimpleSweepLocator:
        """Create fresh locator with uniform prior."""
        belief = build_belief(...)
        return cls(belief, max_steps)

# Usage
locator = SimpleSweepLocator.create(max_steps=30)
```

**Benefits:**
- Simpler - no separate factory class
- More Pythonic - classmethods are standard factory pattern
- Configuration goes directly to create()
- Still creates fresh instances per repeat

---

## Abstract Base

The real interface (`nvision/models/locator.py`):

```python
class Locator(ABC):
    """Stateful locator for one repeat run. Created fresh per repeat via
    classmethod create(). Owns a belief updated incrementally each observation."""

    def __init__(self, belief: AbstractMarginalDistribution):
        self.belief = belief

    @classmethod
    @abstractmethod
    def create(cls, **config) -> Locator:
        """Create a fresh locator instance with a fresh belief. Called once per repeat."""

    @abstractmethod
    def next(self) -> float:
        """Propose next measurement position, using the current belief."""

    @abstractmethod
    def done(self) -> bool:
        """Check if localization is complete (belief.converged(), max steps, ...)."""

    @abstractmethod
    def result(self) -> dict[str, float]:
        """Extract final parameter estimates from the belief."""

    def observe(self, obs: Observation) -> None:
        """Update belief with a new observation (incremental Bayesian update)."""
        self.belief.update(obs)
```

Some concrete sweep locators (`GenericSweepLocator`, `StagedSobolSweepLocator`) add a
`finalize()` hook — called once after the last observation, before `result()` — to
flush deferred belief updates and run a batch model fit instead of paying a full
Bayesian update on every step. It's an addition on top of this ABC, not a
replacement for `done()`/`result()`; the runner calls it via `getattr(locator,
"finalize", None)` since most locators don't define it (see
`nvision/runner/executor.py`).

---

## Runner Usage

```python
for locator in run_loop(GenericSweepLocator, experiment, rng, max_steps=50):
    ...
```

`run_loop` (`nvision/runner/executor.py`) drives one repeat's measurement loop:

```python
def run_loop(
    locator_class: type[Locator],
    experiment: CoreExperiment,
    rng: random.Random,
    sweep_cache: SweepCache | None = None,
    n_shots: int = 1,
    **locator_config: Any,
) -> Iterator[Locator]:
    ...
    locator = locator_class.create(**locator_config)
    ...
```

## Strategy Specification

A `Combination`'s `strategy` field (and therefore `LocatorTask.strategy_spec`,
`nvision/models/task.py`) accepts either a bare `Locator` subclass, or a
`{"class": SomeLocator, "config": {...}}` dict when per-combination config needs to
travel with the class. `StrategySpec.from_raw` normalizes both into
`(locator_class, locator_config)` for the executor — this is how e.g.
`run_groups.py` and the CLI's combination grid attach `max_steps`,
`convergence_threshold`, etc. to a strategy without instantiating it upfront.

---

## Example: a real `create()` (trimmed from `GenericSweepLocator`)

```python
@classmethod
def create(
    cls,
    belief: AbstractMarginalDistribution,
    signal_model: SignalModel,
    max_steps: int,
    *,
    noise_std: float = 0.01,
    scan_param: str | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    **kwargs: Any,
) -> GenericSweepLocator:
    # Resolve the sweep domain from parameter_bounds when domain_lo/hi weren't
    # passed explicitly (see nvision/sim/locs/coarse/generic_sweep_locator.py
    # for the full frequency/scan_param resolution logic this elides).
    domain_lo = kwargs.get("domain_lo", 0.0)
    domain_hi = kwargs.get("domain_hi", 1.0)

    inst = cls(
        belief=belief,
        signal_model=signal_model,
        max_steps=max_steps,
        noise_std=noise_std,
        scan_param=scan_param,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
    )
    if parameter_bounds is not None:
        inst._parameter_bounds = dict(parameter_bounds)
    return inst
```

Read the real thing (`nvision/sim/locs/coarse/generic_sweep_locator.py`) for the
actual domain-resolution logic — this excerpt is trimmed for the pattern, not a
substitute for it, and will drift as that method evolves.

### Usage

```python
locator = GenericSweepLocator.create(belief=belief, signal_model=model, max_steps=200)

# Via runner
for locator in run_loop(GenericSweepLocator, experiment, rng, max_steps=200):
    ...

# Via a combination's strategy dict
strategy = {"class": GenericSweepLocator, "config": {"max_steps": 200}}
```

---

## Migration Guide

### For Existing Locators

**Step 1:** Remove any separate factory class

**Step 2:** Add `@classmethod create()` to the Locator subclass

```python
# Before
class MyLocatorFactory(LocatorFactory):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def create(self) -> Locator:
        belief = ...
        return MyLocator(belief, self.param1, self.param2)

class MyLocator(Locator):
    def __init__(self, belief, param1, param2):
        ...

# After
class MyLocator(Locator):
    @classmethod
    def create(cls, param1, param2, **kwargs):
        belief = ...
        return cls(belief, param1, param2)

    def __init__(self, belief, param1, param2):
        ...
```

**Step 3:** Update usage

```python
# Before
factory = MyLocatorFactory(param1=10, param2=20)
locator = factory.create()

# After
locator = MyLocator.create(param1=10, param2=20)
```

---

## Comparison

| Aspect | Factory Pattern | Classmethod Pattern |
|--------|----------------|---------------------|
| Lines of code | More (separate class) | Fewer (just method) |
| Complexity | Higher (two classes) | Lower (one class) |
| Configuration | Factory __init__ | classmethod kwargs |
| Idiomatic | Less (Java-style) | More (Pythonic) |
| Fresh instances | ✅ Factory.create() | ✅ Class.create() |
| Type hints | Factory + Locator | Just Locator |

---

## Summary

The classmethod pattern is:
- **Simpler**: No separate factory classes
- **Pythonic**: Classmethods are the standard way to create instances
- **Flexible**: Configuration passed directly as kwargs
- **Equivalent**: Still creates fresh instances per repeat

The factory layer was unnecessary indirection. The classmethod does the same job more elegantly.
