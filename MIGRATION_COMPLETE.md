# Legacy to Core Architecture Migration

## Overview

Successfully migrated all legacy Manufacturer-based generators to the new core architecture with explicit `SignalModel` and `Parameter` objects.

## What Changed

### Phase 1: Core Signal Models (Completed)

Added to `src/nvision/core/models.py`:
- `ExponentialDecayModel` - Exponential decay signal model
- `CompositePeakModel` - Combines multiple peak models into a single signal

### Phase 2: Core Generators (Completed)

Completed implementations in `src/nvision/sim/gen/core_generators.py`:
- `OnePeakCoreGenerator` - Single Gaussian/Lorentzian peaks
- `TwoPeakCoreGenerator` - Two separated peaks with CompositePeakModel
- `MultiPeakCoreGenerator` - N peaks with configurable types
- `SymmetricTwoPeakCoreGenerator` - Symmetric peaks around center
- `NVCenterCoreGenerator` - Physics-based NV center ODMR signals

All generators:
- Produce `TrueSignal` with explicit `SignalModel` and `Parameter` objects
- Use lazy imports to avoid circular dependencies
- Support physics-based models (NVCenter) and generic peak shapes

### Phase 3: Updated Usage Sites (Completed)

Updated `src/nvision/sim/cases.py`:
- Replaced all legacy generators with core generators:
  - `OnePeak-gaussian` → `OnePeakCoreGenerator(peak_type="gaussian")`
  - `OnePeak-lorentzian` → `OnePeakCoreGenerator(peak_type="lorentzian")`
  - `TwoPeak-*` → `TwoPeakCoreGenerator(...)`
  - `NVCenter-*` → `NVCenterCoreGenerator(variant=...)`
- All generators now use explicit `x_min` and `x_max` in Hz (2.6e9 to 3.1e9)

Updated `src/nvision/sim/__init__.py`:
- Exports both core generators and legacy generators
- Legacy generators marked as deprecated in comments
- Core locators (SimpleSweepLocator) available via `nvision.sim.locs.core`

### Phase 4: Backward Compatibility (Completed)

Added deprecation warnings to all legacy generators:
- `OnePeakGenerator`
- `TwoPeakGenerator`
- `MultiPeakGenerator`
- `SymmetricTwoPeakGenerator`
- `NVCenterGenerator`

Each warns users to migrate to core generators but remains functional.

Added automatic adapter in `src/nvision/cli/sim_runner.py`:
- Detects when a generator produces `TrueSignal` but strategy is legacy v2
- Automatically converts `TrueSignal` to `ScanBatch` for backward compatibility
- Extracts `truth_positions` from `TrueSignal` parameters
- Enables seamless use of core generators with legacy locators

### Circular Import Resolution

Fixed circular dependency issues:
- Core generators use lazy imports via `_get_core_classes()` helper
- Avoids importing from `nvision.core` at module level
- `SimpleSweepLocator` not imported in `nvision.sim.__init__.py` to break cycle
- All imports work correctly at runtime

## Architecture Flow

```
Core Generators (produce TrueSignal)
         ↓
    Routing Logic (sim_runner.py)
         ↓
    ┌─────────────┬──────────────┐
    ↓             ↓              ↓
Native Path   Adapter Path   Legacy Path
(native_runner) (core_sim_runner) (sim_runner)
    ↓             ↓              ↓
CoreExperiment  TrueSignal→    ScanBatch
+ Runner        ScanBatch      + v2 Runner
```

## Testing

Validated:
- ✅ Native integration demo runs successfully
- ✅ CLI with core generators and legacy v2 locators works
- ✅ Automatic TrueSignal→ScanBatch adapter functions correctly
- ✅ All NVCenter variants generate correct signals
- ✅ No circular import errors

Test commands:
```bash
# Native architecture end-to-end
uv run python examples/native_integration_demo.py

# CLI with core generators + legacy locators
uv run python -m nvision run --repeats 2 --seed 42 --loc-max-steps 50 \
    --filter-strategy "NVCenter-Sweep-V2" --filter-category "NVCenter"
```

## Benefits

1. **Explicit Signal Models**: All signals now have clear mathematical representations
2. **Physics-Based NV Models**: `NVCenterLorentzianModel` and `NVCenterVoigtModel` with proper parameters
3. **Composable Architecture**: `CompositePeakModel` enables complex multi-peak signals
4. **Type Safety**: `TrueSignal` carries `SignalModel` + `Parameter` objects, not closures
5. **Backward Compatible**: Legacy code continues to work via automatic adapters
6. **Deprecation Path**: Clear warnings guide users to migrate

## Migration Guide for Users

### Before (Legacy):
```python
from nvision.sim import OnePeakGenerator, GaussianManufacturer

gen = OnePeakGenerator(
    x_min=2.6e9,
    x_max=3.1e9,
    manufacturer=GaussianManufacturer()
)
```

### After (Core):
```python
from nvision.sim.gen.core_generators import OnePeakCoreGenerator

gen = OnePeakCoreGenerator(
    x_min=2.6e9,
    x_max=3.1e9,
    peak_type="gaussian"
)
```

### NV Center Migration:
```python
# Before
from nvision.sim import NVCenterGenerator
gen = NVCenterGenerator(variant="voigt_zeeman")

# After
from nvision.sim.gen.core_generators import NVCenterCoreGenerator
gen = NVCenterCoreGenerator(
    x_min=2.6e9,
    x_max=3.1e9,
    variant="voigt",
    zero_field=False  # zeeman = not zero_field
)
```

## Next Steps

Recommended follow-up work:
1. Migrate legacy locators to use `Locator` ABC with `create()` classmethod
2. Update tests to use core generators
3. Create comprehensive examples for each core generator
4. Add parameter estimation validation in native runner
5. Eventually remove legacy generators after deprecation period

## Files Modified

Core architecture:
- `src/nvision/core/models.py` - Added ExponentialDecayModel, CompositePeakModel
- `src/nvision/sim/gen/core_generators.py` - Completed all generators, added lazy imports

Usage sites:
- `src/nvision/sim/cases.py` - Updated to use core generators
- `src/nvision/sim/__init__.py` - Export both core and legacy generators

Deprecation:
- `src/nvision/sim/gen/generators/one_peak_generator.py` - Added warning
- `src/nvision/sim/gen/generators/two_peak_generator.py` - Added warning
- `src/nvision/sim/gen/generators/multi_peak_generator.py` - Added warning
- `src/nvision/sim/gen/generators/symmetric_two_peak_generator.py` - Added warning
- `src/nvision/sim/gen/generators/nv_center_generator.py` - Added warning

Compatibility:
- `src/nvision/cli/sim_runner.py` - Added TrueSignal→ScanBatch adapter for legacy v2

---

**Status: ✅ Migration Complete and Tested**

All legacy generators have been migrated to the core architecture while maintaining full backward compatibility. The system now supports three execution paths (native, adapted, legacy) and automatically routes based on generator and locator types.
