# NVision Core Architecture Refactor - Summary

## ✅ Completed: Tasks 1-6 + CLI Integration

### What Was Built

#### 1. Core Abstractions (`src/nvision/core/`)

**Parameter Hierarchy** (`signal.py`)
- `Parameter`: Base class with known value (zero uncertainty)
- `ParameterWithPosterior`: Extends Parameter with posterior distribution
- Unified interface: both have `.value`, `.mean()`, `.uncertainty()`

**Signal Models** (`signal.py`)
- `SignalModel`: Abstract base for signal shapes (Lorentzian, Voigt-Zeeman, etc.)
- `TrueSignal`: Ground truth with `list[Parameter]`
- `BeliefSignal`: Uncertain estimate with `list[ParameterWithPosterior]`
- **Incremental Bayesian update**: `belief.update(obs)` — no history replay

**Locator Pattern** (`locator.py`)
- `Locator`: Abstract base with `next()`, `done()`, `result()`, `observe()`
- `LocatorFactory`: Creates fresh locators per repeat with fresh `BeliefSignal`
- Stateful but short-lived: one locator instance per repeat

**Runner** (`runner.py`)
- Generator-based: `yield locator` after each observation
- Generic: no locator-specific logic
- Takes `TrueSignal` and `Noise` separately

**Observer** (`observer.py`)
- `StepSnapshot`: Captures `BeliefSignal` + `TrueSignal` at each step
- `RunResult`: Exposes trajectories as methods:
  - `uncertainty_trajectory(param)`
  - `estimate_trajectory(param)`
  - `error_trajectory(param)`
  - `entropy_trajectory()`
  - `convergence_step(param, threshold)`

**Observation** (`observation.py`)
- Simple dataclass: `x`, `signal_value`

#### 2. Concrete Signal Models (`src/nvision/core/models.py`)

- `LorentzianModel`: Single Lorentzian peak
- `VoigtZeemanModel`: NV center with hyperfine splitting (3 peaks)
- `GaussianModel`: Single Gaussian peak

All models work polymorphically with `list[Parameter]`.

#### 3. CLI Integration

**Automatic Architecture Detection** (`src/nvision/cli/sim_runner.py`)
```python
if isinstance(task.strategy, LocatorFactory):
    # New core architecture
    return run_simulation_batch_with_core(task)
else:
    # Legacy v2 architecture
    # ... existing code ...
```

**Adapters** (`src/nvision/sim/adapters.py`)
- `ScanBatchSignalModel`: Wraps black-box signal functions
- `scan_batch_to_true_signal()`: Converts legacy `ScanBatch` → `TrueSignal`
- Coordinate normalization: `normalize_x()`, `denormalize_x()`

**Core Sim Runner** (`src/nvision/cli/core_sim_runner.py`)
- `run_simulation_batch_with_core()`: Runs with new architecture
- `run_result_to_history_df()`: Converts trajectories to DataFrame
- `run_result_to_finalize_record()`: Extracts final estimates
- **Output format identical to v2**: Downstream visualization unchanged

**Example Locator** (`src/nvision/sim/locs/core/sweep_locator.py`)
- `SimpleSweepLocator`: Demonstrates new architecture
- `SimpleSweepFactory`: Creates locators with uniform priors
- Works with CLI out of the box

#### 4. Documentation

- `docs/core_architecture.md`: Parameter design philosophy
- `docs/cli_integration.md`: Integration guide
- `examples/core_architecture_demo.py`: Standalone core demo
- `examples/cli_integration_demo.py`: CLI integration demo

---

## Design Principles

### 1. Parameter is the Unit of Truth
```python
# Ground truth
Parameter(name="frequency", bounds=(2.6e9, 3.1e9), value=2.87e9)

# Uncertain estimate
ParameterWithPosterior(
    name="frequency",
    bounds=(2.6e9, 3.1e9),
    grid=np.linspace(2.6e9, 3.1e9, 100),
    posterior=np.array([...])  # distribution over grid
)
```

### 2. Same Model, Different Certainty
```python
model = LorentzianModel()

# TrueSignal knows exact values
true_signal = TrueSignal(model=model, parameters=[...])

# BeliefSignal has distributions
belief_signal = BeliefSignal(model=model, parameters=[...])

# Both compute signal the same way
true_signal(x)   # uses exact values
belief_signal(x)  # uses posterior means
```

### 3. Incremental Updates, No Replay
```python
# Old: replay full history every step O(n)
for obs in history:
    posterior = bayesian_update(prior, obs)

# New: incremental update O(1)
belief.update(obs)  # updates posterior in-place
```

### 4. Runner as Generator
```python
runner = Runner()
for locator in runner.run(factory, true_signal, noise, rng):
    print(f"Entropy: {locator.belief.entropy()}")
    # Can break early, inspect state, etc.
```

### 5. Observer Tracks Trajectories
```python
observer = Observer(true_signal)
result = observer.watch(runner.run(...))

# Full convergence trajectories available
errors = result.error_trajectory("frequency")
uncertainties = result.uncertainty_trajectory("frequency")
```

---

## Data Flow

```
User Code
    └─> LocatorFactory.create() → Locator (fresh per repeat)
        └─> Locator has BeliefSignal (uniform prior)
            └─> Runner.run() yields Locator after each obs
                └─> Observer.watch() accumulates StepSnapshot
                    └─> RunResult exposes trajectories
                        └─> viz/ consumes trajectories
                            └─> gui/ consumes plots
```

**One-way flow:**
1. Runner produces Locator states
2. Observer snapshots BeliefSignal + TrueSignal
3. RunResult exposes methods for trajectories
4. viz knows RunResult, nothing about Locator
5. gui knows viz outputs, nothing about RunResult

---

## What Works Now

### Standalone Core
```bash
uv run python examples/core_architecture_demo.py
```
✅ Creates TrueSignal with known parameters  
✅ Creates BeliefSignal with uniform prior  
✅ Runs localization with Runner  
✅ Observer tracks convergence  
✅ RunResult exposes trajectories  

Output:
```
Steps taken: 30
Final estimates: {'frequency': 0.518, 'linewidth': 0.012, ...}
Entropy reduced: 6.15 → 1.36
```

### CLI Integration
```bash
uv run python examples/cli_integration_demo.py
```
✅ LocatorFactory triggers new architecture  
✅ ScanBatch → TrueSignal conversion  
✅ Runner produces observations  
✅ Observer tracks trajectories  
✅ Output format matches v2 (DataFrame)  
✅ Downstream viz/caching unchanged  

Output:
```
History DataFrame shape: (90, 4)  # 30 steps × 3 repeats
Finalize Results: peak_x, x1_hat, entropy, converged
Peak estimate: 2.850000e+09
```

### CLI Command (when new locators available)
```bash
uv run python -m nvision run \
    --filter-category NVCenter \
    --filter-strategy SimpleSweep-Core \
    --repeats 10 \
    --seed 42
```
✅ Automatically detects LocatorFactory  
✅ Routes to new core architecture  
✅ Produces same outputs as v2  
✅ Caching and visualization work  

---

## What's Next (Tasks 7-16)

### 7. Implement Concrete SignalModel Subclasses
- **`LorentzianModel`** ✅ (already done)
- **`VoigtZeemanModel`** ✅ (already done)
- **`GaussianModel`** ✅ (already done)

### 8. Refactor NVCenterSweepLocator
- Implement new Locator ABC
- Create NVCenterSweepFactory
- BeliefSignal with Voigt-Zeeman model
- Move from v2 to core

### 9. Refactor OnePeakGoldenLocator
- Golden-section state lives in Locator self
- Fresh per repeat via factory
- BeliefSignal tracks uncertainty

### 10. Refactor NVCenterSequentialBayesianLocator
- Posterior moves into BeliefSignal.update()
- Remove history replay
- Use incremental updates

### 11-13. Delete Legacy Code
- Delete `_bayesian_adapter.py`
- Delete unused `LocatorStrategy` protocol
- Delete `NVCenterBayesianLocatorBase` and `NVCenterLocatorBase`

### 14. Update runner.py
- Remove isinstance checks for specific locator types
- Use generic Runner exclusively

### 15. Update viz/ to Consume RunResult
- Replace DataFrame-based plotting
- Use RunResult trajectory methods
- Single source of truth for convergence data

### 16. Update Tests
- Test Parameter hierarchy
- Test incremental Bayesian updates
- Test Runner/Observer pattern
- Test CLI integration

---

## Key Files Created/Modified

### New Files
- `src/nvision/core/observation.py`
- `src/nvision/core/signal.py`
- `src/nvision/core/locator.py`
- `src/nvision/core/runner.py`
- `src/nvision/core/observer.py`
- `src/nvision/core/models.py`
- `src/nvision/core/__init__.py`
- `src/nvision/sim/adapters.py`
- `src/nvision/cli/core_sim_runner.py`
- `src/nvision/sim/locs/core/__init__.py`
- `src/nvision/sim/locs/core/sweep_locator.py`
- `docs/core_architecture.md`
- `docs/cli_integration.md`
- `examples/core_architecture_demo.py`
- `examples/cli_integration_demo.py`

### Modified Files
- `src/nvision/cli/sim_runner.py` (added automatic detection)

### Unchanged (Backward Compatible)
- All existing v2 locators
- All visualization code
- All caching code
- CLI command interface

---

## Benefits Achieved

1. **Clean Abstractions**: Parameter, SignalModel, BeliefSignal, TrueSignal
2. **Type Safety**: Parameter vs ParameterWithPosterior enforced by type system
3. **Incremental Updates**: O(1) per observation, not O(n)
4. **Trajectory-First**: RunResult exposes convergence as first-class methods
5. **One-Way Data Flow**: Runner → Observer → RunResult → viz → gui
6. **Backward Compatible**: v2 locators still work
7. **Gradual Migration**: Can mix v2 and core locators
8. **Testable**: Each component testable in isolation
9. **Documented**: Architecture philosophy captured in docs

---

## Architecture Validation

✅ **Standalone core works**: `core_architecture_demo.py` runs successfully  
✅ **CLI integration works**: `cli_integration_demo.py` runs successfully  
✅ **Output compatibility**: DataFrames match v2 format  
✅ **Automatic routing**: LocatorFactory detected, routes to core  
✅ **Incremental updates**: BeliefSignal.update() works O(1)  
✅ **Trajectory tracking**: Observer accumulates snapshots  
✅ **Error computation**: RunResult computes errors vs TrueSignal  
✅ **Parameter polymorphism**: SignalModel accepts both Parameter types  

**Ready for tasks 7-16**: Refactor existing locators to new architecture.
