# Classmethod Factory Pattern

## Overview

Instead of separate `LocatorFactory` classes, locators now use a `create()` classmethod for instantiation. This is simpler, more Pythonic, and eliminates the factory layer.

---

## Before (Factory Pattern)

```python
class SimpleSweepFactory(LocatorFactory):
    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
    
    def create(self) -> Locator:
        model = BlackBoxSignalModel()
        belief = BeliefSignal(model=model, parameters=[...])
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
        model = BlackBoxSignalModel()
        belief = BeliefSignal(model=model, parameters=[...])
        return cls(belief, max_steps)

# Usage
locator = SimpleSweepLocator.create(max_steps=30)
```

**Benefits:**
- ✅ Simpler - no separate factory class
- ✅ More Pythonic - classmethods are standard factory pattern
- ✅ Configuration goes directly to create()
- ✅ Still creates fresh instances per repeat

---

## Abstract Base

```python
class Locator(ABC):
    """Abstract locator with classmethod factory."""
    
    @classmethod
    @abstractmethod
    def create(cls, **config):
        """Create fresh locator with fresh BeliefSignal.
        
        Subclasses implement this to create locators with
        properly initialized beliefs (uniform priors, etc).
        """
        pass
    
    @abstractmethod
    def next(self) -> float: ...
    
    @abstractmethod
    def done(self) -> bool: ...
    
    @abstractmethod
    def result(self) -> dict[str, float]: ...
```

---

## Runner Usage

### Before (with Factory)
```python
runner = Runner()
factory = SimpleSweepFactory(max_steps=50)
for locator in runner.run(factory, experiment, rng):
    ...
```

### After (with Classmethod)
```python
runner = Runner()
for locator in runner.run(SimpleSweepLocator, experiment, rng, max_steps=50):
    ...
```

The runner signature changed:
```python
def run(
    self,
    locator_class: Type[Locator],  # Pass class, not instance
    experiment: CoreExperiment,
    rng: random.Random,
    **locator_config,  # Config passed to create()
) -> Iterator[Locator]:
    locator = locator_class.create(**locator_config)
    # ...
```

---

## CLI Integration

### Task Configuration

**Option 1: Pass Locator class directly**
```python
task = LocatorTask(
    strategy_name="SimpleSweep",
    strategy=SimpleSweepLocator,  # Class itself
    # ...
)
```

**Option 2: Pass dict with class and config**
```python
task = LocatorTask(
    strategy_name="SimpleSweep",
    strategy={
        "class": SimpleSweepLocator,
        "config": {"max_steps": 50, "convergence_threshold": 0.01}
    },
    # ...
)
```

### CLI Detection

```python
# In src/nvision/cli/sim_runner.py
def run_simulation_batch(task):
    is_locator_class = (
        (isinstance(task.strategy, type) and issubclass(task.strategy, Locator)) or
        (isinstance(task.strategy, dict) and "class" in task.strategy)
    )
    
    if is_locator_class:
        # Native or adapted core architecture
        if isinstance(test_output, TrueSignal):
            return run_native_simulation_batch(task)
        else:
            return run_simulation_batch_with_core(task)
    else:
        # Legacy v2 architecture
        return run_simulation_batch_v2(task)
```

---

## Example: NV Center Locator

```python
class NVCenterBayesianLocator(Locator):
    """Bayesian locator for NV center ODMR signals."""
    
    def __init__(
        self,
        belief: BeliefSignal,
        max_steps: int = 150,
        acquisition: str = "eig",
    ):
        super().__init__(belief)
        self.max_steps = max_steps
        self.acquisition = acquisition
        self.step_count = 0
    
    @classmethod
    def create(
        cls,
        max_steps: int = 150,
        acquisition: str = "eig",
        n_grid_freq: int = 100,
        n_grid_linewidth: int = 50,
        n_grid_split: int = 50,
        **kwargs
    ) -> NVCenterBayesianLocator:
        """Create NV center locator with proper priors.
        
        Parameters
        ----------
        max_steps : int
            Maximum measurement steps
        acquisition : str
            Acquisition strategy: "eig", "ucb", "random"
        n_grid_* : int
            Grid resolution for each parameter
        **kwargs
            Additional config (ignored)
            
        Returns
        -------
        NVCenterBayesianLocator
            Fresh locator with uniform priors
        """
        from nvision.core.nv_models import NVCenterLorentzianModel
        
        model = NVCenterLorentzianModel()
        
        # Create uniform priors over physical parameter ranges
        belief = BeliefSignal(
            model=model,
            parameters=[
                ParameterWithPosterior(
                    name="frequency",
                    bounds=(2.6e9, 3.1e9),
                    grid=np.linspace(2.6e9, 3.1e9, n_grid_freq),
                    posterior=np.ones(n_grid_freq) / n_grid_freq,
                ),
                ParameterWithPosterior(
                    name="linewidth",
                    bounds=(1e6, 50e6),
                    grid=np.linspace(1e6, 50e6, n_grid_linewidth),
                    posterior=np.ones(n_grid_linewidth) / n_grid_linewidth,
                ),
                ParameterWithPosterior(
                    name="split",
                    bounds=(1e6, 200e6),
                    grid=np.linspace(1e6, 200e6, n_grid_split),
                    posterior=np.ones(n_grid_split) / n_grid_split,
                ),
                # ... other parameters
            ],
        )
        
        return cls(belief, max_steps, acquisition)
    
    def next(self) -> float:
        """Use acquisition function to propose next measurement."""
        if self.acquisition == "eig":
            return self._expected_information_gain()
        elif self.acquisition == "ucb":
            return self._upper_confidence_bound()
        else:
            return random.random()
    
    def done(self) -> bool:
        """Check convergence or max steps."""
        return (
            self.step_count >= self.max_steps or
            self.belief.converged(threshold=0.01)
        )
    
    def result(self) -> dict[str, float]:
        """Return final parameter estimates."""
        return self.belief.estimates()
```

### Usage

```python
# Direct instantiation
locator = NVCenterBayesianLocator.create(
    max_steps=200,
    acquisition="eig",
    n_grid_freq=150,
)

# Via runner
runner = Runner()
result = observer.watch(
    runner.run(
        NVCenterBayesianLocator,
        experiment,
        rng,
        max_steps=200,
        acquisition="eig",
    )
)

# Via CLI task
task = LocatorTask(
    strategy_name="NVCenter-Bayesian-EIG",
    strategy={
        "class": NVCenterBayesianLocator,
        "config": {
            "max_steps": 200,
            "acquisition": "eig",
            "n_grid_freq": 150,
        }
    },
    # ...
)
```

---

## Migration Guide

### For Existing Locators

**Step 1:** Remove LocatorFactory class

**Step 2:** Add `@classmethod create()` to Locator

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
