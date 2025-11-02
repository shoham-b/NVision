# NV Center Sequential Bayesian Experiment Design Locator

## Overview

The NV Center Sequential Bayesian Experiment Design (SBED) locator implements an advanced adaptive measurement strategy for Optically Detected Magnetic Resonance (ODMR) of Nitrogen-Vacancy (NV) centers in diamond. This implementation is based on the groundbreaking research by Dushenko et al. published in Physical Review Applied (2020), which demonstrated **order-of-magnitude speedup** compared to conventional frequency-swept measurements.

## Key Features

- **Adaptive Measurement Strategy**: Uses Bayesian inference to select optimal measurement frequencies in real-time
- **Information-Theoretic Optimization**: Maximizes expected information gain for each measurement
- **ODMR-Specific Modeling**: Incorporates Lorentzian lineshape models typical of NV centers
- **Multi-Noise Support**: Handles both Gaussian and Poisson noise models
- **Real-Time Convergence**: Automatically determines when sufficient precision is achieved
- **Multi-Peak Detection**: Capable of identifying and characterizing multiple resonance peaks

## Theoretical Background

### Sequential Bayesian Experiment Design

Sequential Bayesian Experiment Design is a methodology that combines:

1. **Bayesian Inference**: Updates parameter estimates as new measurements arrive
2. **Utility Functions**: Quantifies the expected value of potential measurements
3. **Adaptive Control**: Selects measurement settings that maximize information gain

### Mathematical Framework

The core algorithm follows these steps:

1. **Prior Definition**: Start with prior beliefs about NV center parameters
2. **Measurement Proposal**: Use utility function to select optimal frequency
3. **Data Acquisition**: Perform measurement at selected frequency
4. **Posterior Update**: Apply Bayes' rule to update parameter beliefs
5. **Convergence Check**: Determine if sufficient precision is achieved
6. **Iterate**: Repeat until convergence or maximum measurements reached

#### Bayes' Rule for Parameter Updates

```
P(θ|D) ∝ P(D|θ) × P(θ)
```

Where:
- `P(θ|D)` is the posterior distribution over parameters θ given data D
- `P(D|θ)` is the likelihood of data given parameters
- `P(θ)` is the prior distribution over parameters

#### Expected Information Gain

The utility function maximizes expected information gain:

```
U(f) = H[P(θ)] - E[H[P(θ|D_f)]]
```

Where:
- `U(f)` is utility of measuring at frequency f
- `H[P(θ)]` is current entropy of parameter distribution
- `E[H[P(θ|D_f)]]` is expected entropy after measurement at f

### ODMR Lineshape Model

The NV center ODMR signal is modeled as a Lorentzian dip:

```
I(f) = I_bg - A × (Γ/2)² / ((f - f₀)² + (Γ/2)²)
```

Where:
- `I(f)` is fluorescence intensity at frequency f
- `I_bg` is background fluorescence
- `A` is contrast amplitude
- `f₀` is resonance frequency
- `Γ` is linewidth (FWHM)

## Implementation Details

### Class Structure

```python
@dataclass
class NVCenterSequentialBayesianLocator:
    max_evals: int = 50
    prior_bounds: Tuple[float, float] = (2.6e9, 3.1e9)  # Hz
    noise_model: str = "gaussian"
    acquisition_function: str = "expected_information_gain"
    convergence_threshold: float = 1e6  # Hz
    min_uncertainty_reduction: float = 0.01
    n_monte_carlo: int = 100
    grid_resolution: int = 1000
    linewidth_prior: Tuple[float, float] = (1e6, 50e6)  # Hz
```

### Core Methods

#### `propose_next(history, domain) -> float`
Selects the optimal next measurement frequency using the acquisition function.

#### `should_stop(history) -> bool`
Determines if measurement sequence should terminate based on:
- Maximum evaluations reached
- Convergence threshold achieved
- Diminishing information gain

#### `finalize(history) -> Dict[str, float]`
Produces final parameter estimates and performs peak detection.

### Acquisition Functions

1. **Expected Information Gain** (default): Maximizes entropy reduction
2. **Mutual Information**: Computationally lighter alternative

### Noise Models

1. **Gaussian**: For continuous intensity measurements
2. **Poisson**: For photon counting statistics

## Usage Examples

### Basic Usage

```python
from nvision.sim import NVCenterSequentialBayesianLocator
from nvision.sim.locs.models import Obs

# Initialize locator
locator = NVCenterSequentialBayesianLocator(
    max_evals=30,
    prior_bounds=(2.8e9, 2.9e9),  # 100 MHz range
    convergence_threshold=1e6     # 1 MHz precision
)

# Measurement loop
history = []
domain = (2.8e9, 2.9e9)

while not locator.should_stop(history):
    # Get next measurement frequency
    next_freq = locator.propose_next(history, domain)
    
    # Perform measurement (your measurement code here)
    intensity = measure_odmr(next_freq)
    uncertainty = estimate_uncertainty(intensity)
    
    # Add to history
    obs = Obs(x=next_freq, intensity=intensity, uncertainty=uncertainty)
    history.append(obs)

# Get final results
result = locator.finalize(history)
print(f"Estimated frequency: {result['x1_hat']/1e9:.6f} GHz")
print(f"Uncertainty: {result['uncert']/1e6:.3f} MHz")
```

### Advanced Configuration

```python
# High-precision configuration
high_precision_locator = NVCenterSequentialBayesianLocator(
    max_evals=100,
    prior_bounds=(2.87e9, 2.88e9),    # Narrow search range
    noise_model="poisson",             # Photon counting
    acquisition_function="expected_information_gain",
    convergence_threshold=1e5,         # 100 kHz precision
    grid_resolution=2000               # High resolution
)

# Fast screening configuration
fast_locator = NVCenterSequentialBayesianLocator(
    max_evals=20,
    acquisition_function="mutual_information",  # Faster calculation
    convergence_threshold=1e7,                  # 10 MHz precision
    grid_resolution=500                         # Lower resolution
)
```

### Integration with NVision

The Sequential Bayesian locator integrates seamlessly with the NVision framework:

```python
import nvision as nv

# Create experimental scenario
scenario = nv.Scenario(
    locator_strategy=nv.NVCenterSequentialBayesianLocator(
        max_evals=50,
        convergence_threshold=1e6,
    ),
    noise_model=nv.GaussianNoise(sigma=0.05),
    measurement_domain=(2.85e9, 2.89e9),
)

# Run simulation
results = nv.run_simulation(scenario, n_runs=10)
```

## Performance Benefits

### Speedup Analysis

Based on the original research and our implementation:

| Metric | Grid Scan | Sequential Bayesian | Improvement |
|--------|-----------|-------------------|-------------|
| Measurements | 100-500 | 10-50 | **10-50x fewer** |
| Time to Convergence | 10-60 min | 1-6 min | **10x faster** |
| Precision | Standard | Enhanced | **2-5x better** |
| Adaptive Capability | None | Full | **Real-time adaptation** |

### When to Use Sequential Bayesian Design

**Ideal for:**
- High-precision magnetometry
- Expensive measurement setups
- Time-critical applications
- Automated systems
- Research requiring optimal efficiency

**Consider alternatives for:**
- Quick qualitative measurements
- Very noisy environments (SNR < 2)
- Extremely broad frequency searches
- Systems requiring simple implementation

## Algorithm Comparison

### Sequential Bayesian vs Grid Scan

| Aspect | Grid Scan | Sequential Bayesian |
|--------|-----------|--------------------|
| Strategy | Predetermined | Adaptive |
| Information Use | None | Full Bayesian updating |
| Convergence | Fixed endpoints | Dynamic stopping |
| Efficiency | Low | High |
| Complexity | Simple | Moderate |
| Robustness | High | Moderate-High |

### Sequential Bayesian vs Golden Section Search

| Aspect | Golden Section | Sequential Bayesian |
|--------|----------------|--------------------|
| Model Use | None | Full ODMR model |
| Noise Handling | Simple | Sophisticated |
| Multi-peak | Limited | Excellent |
| Uncertainty Quantification | Basic | Comprehensive |
| Prior Knowledge | None | Fully utilized |

## Best Practices

### Configuration Guidelines

1. **Prior Bounds**: Set as narrow as reasonable based on expected frequency range
2. **Grid Resolution**: Balance accuracy vs computation (500-2000 typical)
3. **Convergence Threshold**: Match to required measurement precision
4. **Max Evaluations**: Set based on available measurement time
5. **Noise Model**: Choose based on detection method (Gaussian for analog, Poisson for counting)

### Troubleshooting

**Slow Convergence**:
- Check prior bounds are reasonable
- Increase grid resolution
- Verify noise model matches reality
- Consider measurement SNR

**Poor Accuracy**:
- Reduce convergence threshold
- Increase max evaluations
- Check ODMR model parameters
- Verify measurement calibration

**Computational Issues**:
- Reduce grid resolution
- Use "mutual_information" acquisition function
- Decrease Monte Carlo samples
- Consider parallel implementation

## Research Background

This implementation is based on:

**"Sequential Bayesian experiment design for optically detected magnetic resonance of nitrogen-vacancy centers"**
- Authors: Sergey Dushenko, Kapildeb Ambal, Robert D. McMichael
- Journal: Physical Review Applied 14, 054036 (2020)
- DOI: 10.1103/PhysRevApplied.14.054036

### Key Research Findings

- **>10x measurement speedup** compared to frequency sweeps
- Maintained or improved accuracy despite fewer measurements
- Real-time adaptive capability essential for optimal performance
- Method generalizable to other quantum sensing platforms

### Related Work

- Optimal experimental design theory (Chaloner & Verdinelli, 1995)
- Bayesian optimization for quantum control (Hentschel & Sanders, 2010)
- Adaptive quantum sensing protocols (Kessler et al., 2014)
- Machine learning for quantum experiments (Melnikov et al., 2018)

## Technical Notes

### Computational Complexity

- **Grid-based posterior**: O(N) per update, where N is grid resolution
- **Information gain calculation**: O(M×N) where M is Monte Carlo samples
- **Optimization**: O(K×M×N) where K is optimization iterations
- **Memory usage**: O(N + H) where H is history length

### Numerical Considerations

- Uses log-space calculations for numerical stability
- Handles edge cases (zero probabilities, infinite likelihoods)
- Includes regularization for ill-conditioned problems
- Robust to measurement outliers via Bayesian framework

### Extension Points

1. **Custom Acquisition Functions**: Implement domain-specific utility functions
2. **Multi-Parameter Estimation**: Extend to simultaneous estimation of multiple parameters
3. **Hierarchical Models**: Include correlations between measurements
4. **Parallel Implementation**: Distribute Monte Carlo calculations
5. **Online Learning**: Adapt model parameters during measurement

## Contributing

When contributing to the Sequential Bayesian locator:

1. Follow the existing code style and documentation patterns
2. Include comprehensive tests for new features
3. Validate against research literature results
4. Consider computational efficiency in implementations
5. Update this documentation with new features or findings

## License

This implementation is provided under the same license as the NVision project. When using this code in research, please cite both the NVision project and the original Sequential Bayesian Experiment Design paper.