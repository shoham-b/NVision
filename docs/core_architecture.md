# Core Architecture: Parameter Design

## Overview

The core architecture uses an inheritance-based parameter design where:
- `Parameter` is the base class representing a parameter with a known value (zero uncertainty)
- `ParameterWithPosterior` extends `Parameter` to add a probability distribution over values

This design reflects that **both TrueSignal and BeliefSignal work with the same parameters**, they just have different levels of certainty about the values.

## Class Hierarchy

```
Parameter (base class)
├── name: str
├── bounds: tuple[float, float]
├── value: float                    # single known value
├── mean() -> float                 # returns value
├── uncertainty() -> float          # returns 0.0
├── entropy() -> float              # returns 0.0
└── converged(threshold) -> bool    # always True

ParameterWithPosterior (subclass)
├── [inherits name, bounds from Parameter]
├── value: float                    # computed from posterior mean
├── grid: np.ndarray               # discretized range
├── posterior: np.ndarray          # probability distribution
├── mean() -> float                 # computes from posterior
├── uncertainty() -> float          # computes std from posterior
├── entropy() -> float              # Shannon entropy
└── converged(threshold) -> bool    # checks uncertainty < threshold
```

## Usage in Signals

### TrueSignal (Ground Truth)

Uses `Parameter` — known values, no uncertainty:

```python
true_signal = TrueSignal(
    model=LorentzianModel(),
    parameters=[
        Parameter(name="frequency", bounds=(0.2, 0.8), value=0.5),
        Parameter(name="linewidth", bounds=(0.01, 0.1), value=0.05),
        Parameter(name="amplitude", bounds=(0.1, 1.0), value=0.5),
        Parameter(name="background", bounds=(0.95, 1.05), value=1.0),
    ]
)
```

### BeliefSignal (Uncertain Estimate)

Uses `ParameterWithPosterior` — distributions that narrow over time:

```python
belief = BeliefSignal(
    model=LorentzianModel(),
    parameters=[
        ParameterWithPosterior(
            name="frequency",
            bounds=(0.2, 0.8),
            grid=np.linspace(0.2, 0.8, 50),
            posterior=np.ones(50) / 50,  # uniform prior
        ),
        # ... other parameters
    ]
)
```

## Benefits of This Design

1. **Same metadata, different certainty**: Both signals share parameter names and bounds, but differ in certainty
2. **Type safety**: `BeliefSignal` enforces `ParameterWithPosterior`, `TrueSignal` accepts any `Parameter`
3. **Polymorphism**: Both can be passed to `SignalModel.compute()` via the common `.value` property
4. **Clear semantics**: A parameter is inherently a value; uncertainty is an optional extension
5. **No duplication**: Parameter metadata (name, bounds) lives in one place

## SignalModel Interface

Signal models work with lists of parameters polymorphically:

```python
class SignalModel(ABC):
    @abstractmethod
    def compute(self, x: float, params: list[Parameter]) -> float:
        """Works with both Parameter and ParameterWithPosterior."""
        pass
    
    def _params_to_dict(self, params: list[Parameter]) -> dict[str, float]:
        """Helper to convert to dict for easier access."""
        return {p.name: p.value for p in params}
```

Example implementation:

```python
class LorentzianModel(SignalModel):
    def compute(self, x: float, params: list[Parameter]) -> float:
        p = self._params_to_dict(params)  # works with both types
        freq = p["frequency"]
        linewidth = p["linewidth"]
        # ...
```

## Key Properties

### Parameter (known value)
- `param.value` → the known value
- `param.mean()` → same as `value`
- `param.uncertainty()` → always 0.0
- Represents ground truth or a point estimate

### ParameterWithPosterior (uncertain value)
- `param.value` → posterior mean (auto-computed)
- `param.mean()` → computes from `grid` and `posterior`
- `param.uncertainty()` → standard deviation of posterior
- `param.entropy()` → Shannon entropy
- Represents a belief that narrows as evidence accumulates

## Evolution During Localization

```python
# Start: uniform prior, high uncertainty
belief.get_param("frequency").uncertainty()  # → 0.173

# After observations: narrowed posterior, low uncertainty
belief.update(Observation(x=0.5, signal_value=0.95))
belief.get_param("frequency").uncertainty()  # → 0.021

# Converged: very peaked posterior, near-zero uncertainty
belief.get_param("frequency").uncertainty()  # → 0.001
```

The design reflects the **philosophical truth**: we're always estimating the same parameters, we just gain certainty about their values over time.
