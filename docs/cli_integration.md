# CLI Integration with New Core Architecture

## Overview

The NVision CLI has been updated to support both the legacy v2 locator architecture and the new core architecture with `Runner`, `Observer`, `BeliefSignal`, and `TrueSignal`.

## Automatic Detection

The CLI automatically detects which architecture to use based on the strategy type:

```python
# In src/nvision/runner/batch.py
from nvision.runner.batch import run_simulation_batch
```

## Architecture Components

### 1. Adapters (`src/nvision/sim/adapters.py`)

Bridges legacy `ScanBatch` objects with the new core architecture:

- **`ScanBatchSignalModel`**: Wraps a black-box signal function as a `SignalModel`
- **`scan_batch_to_true_signal()`**: Converts `ScanBatch` to `TrueSignal`
- **`normalize_x()` / `denormalize_x()`**: Coordinate transformations

### 2. Batch Runner (`src/nvision/runner/batch.py`)

Runs simulations using the new core architecture:

- **`run_simulation_batch()`**: Main entry point
  - Runs localization with `Runner` and `Observer`
  - Converts `RunResult` into history/finalize DataFrames

- **`run_result_to_history_df()`**: Converts trajectory to history DataFrame
- **`run_result_to_finalize_record()`**: Extracts final estimates

### 3. Example Locator (`src/nvision/sim/locs/core/sweep_locator.py`)

Demonstrates implementing a locator with the new architecture:

```python
class SimpleSweepFactory(LocatorFactory):
    def create(self) -> Locator:
        model = BlackBoxSignalModel()
        belief = BeliefSignal(
            model=model,
            parameters=[
                ParameterWithPosterior(
                    name="peak_x",
                    bounds=(0.0, 1.0),
                    grid=np.linspace(0.0, 1.0, 100),
                    posterior=np.ones(100) / 100,
                ),
            ],
        )
        return SimpleSweepLocator(belief, self.max_steps)
```

## Data Flow

```
CLI Task Creation
    └─> Detect strategy type (LocatorFactory or Locator)
        ├─> LocatorFactory → New Core Architecture
        │   └─> run_simulation_batch_with_core()
        │       ├─> Convert ScanBatch → TrueSignal
        │       ├─> Runner.run(factory, true_signal, noise, rng)
        │       ├─> Observer.watch(runner) → RunResult
        │       └─> Convert RunResult → DataFrames
        │           ├─> history_df (trajectory)
        │           └─> finalize_df (estimates)
        │
        └─> Locator → Legacy V2 Architecture
            └─> run_simulation_batch() (existing)
```

## Output Format

Both architectures produce the same output format for downstream compatibility:

### History DataFrame
```
Columns: ['repeat_id', 'step', 'x', 'signal_values']
- One row per observation
- x in physical domain (denormalized)
```

### Finalize DataFrame
```
Columns: ['repeat_id', 'peak_x', 'x1_hat', 'entropy', 'converged', ...]
- One row per repeat
- Peak estimates in physical domain
- Uncertainties and convergence flags
```

## Example Usage

### Using Core Architecture

```python
from nvision.models.task import LocatorTask

# Create factory (triggers new architecture)
factory = SimpleSweepFactory(max_steps=50)

task = LocatorTask(
    generator_name="OnePeak",
    generator=generator,
    strategy_name="SimpleSweep-Core",
    strategy=factory,  # LocatorFactory → new architecture
    repeats=10,
    seed=42,
    # ... other params
)

# Run automatically routes to new architecture
results = run_simulation_batch(task)
```

### Using Legacy V2 Architecture

```python
from nvision.sim.locs import NVCenterSweepLocatorV2

# Create locator instance (triggers v2 architecture)
locator = NVCenterSweepLocatorV2(scan_points=50)

task = LocatorTask(
    strategy_name="NVCenter-Sweep",
    strategy=locator,  # Locator → v2 architecture
    # ...
)

# Run automatically routes to v2 architecture
results = run_simulation_batch(task)
```

## Benefits

1. **Backward Compatibility**: Existing v2 locators continue to work
2. **Gradual Migration**: New locators can use core architecture incrementally
3. **Consistent Output**: Both produce same DataFrame format
4. **Automatic Routing**: CLI detects and routes to appropriate runner
5. **Clean Abstractions**: Clear separation between architectures

## Testing

See `examples/cli_integration_demo.py` for a complete working example:

```bash
uv run python examples/cli_integration_demo.py
```

This demonstrates:
- Creating a `LocatorFactory`
- Running through CLI infrastructure
- Accessing trajectory data
- Extracting final estimates

## Next Steps

To implement a new locator using the core architecture:

1. Create a `SignalModel` subclass (or use existing like `LorentzianModel`)
2. Implement a `Locator` subclass with `next()`, `done()`, `result()`
3. Implement a `LocatorFactory` that creates fresh locators with `BeliefSignal`
4. Use in CLI by passing factory as `task.strategy`

The CLI will automatically use the new architecture and produce compatible outputs.
