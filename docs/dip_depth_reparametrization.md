# Dip-Depth Reparametrization

## The Problem: Amplitude–Linewidth Degeneracy

The original Lorentzian dip model used a raw Hz² `amplitude` parameter $A$ in the numerator:

$$L(f) = \frac{A}{(f - f_0)^2 + \omega^2}$$

At resonance ($f = f_0$), the dip height evaluates to $A / \omega^2$. This means the
**observable** (dip height) depends on the **ratio** of two free parameters. Any pair
$(A, \omega)$ satisfying $A / \omega^2 = \text{const}$ produces the same peak height — forming a
**ridge** in the likelihood surface. This is the classic identifiability problem known as
*sloppy parameter combinations*.

In Bayesian inference this ridge causes the posterior to spread along the degenerate
direction ($A \propto \omega^2$), never converging. In practice the posterior drifts toward
the prior boundaries, producing wildly wrong point estimates for both `amplitude` and
`linewidth`.

## The Solution: `dip_depth`

We reparametrize using a unitless **contrast** parameter $D$ (`dip_depth`), defined so
that the Lorentzian has the form:

$$L(f) = \frac{D \cdot \omega^2}{(f - f_0)^2 + \omega^2}$$

At resonance this evaluates to exactly $D$, **independent of $\omega$**. The peak height
is now controlled by a single free parameter. The physical Hz² numerator can always be
recovered as:

$$A_{\text{phys}} = D \cdot \omega^2$$

but Bayesian inference operates on $(D, \omega)$ which are **identifiable** — each
parameter affects a different observable (depth vs. width).

## NV Center: Three Coupled Dips

The NV center ODMR signal consists of three Lorentzian dips from hyperfine splitting:

$$S(f) = B - L_{\text{left}}(f) - L_{\text{center}}(f) - L_{\text{right}}(f)$$

All three dips share the same `linewidth` $\omega$ and are tied to a single `dip_depth`
$D$ via the non-polarization factor `k_np`:

```
                     ┌─ Left dip:   depth = D / k_np,  center = f_B − Δ
                     │
dip_depth (D) ───────┼─ Center dip: depth = D,         center = f_B
                     │
                     └─ Right dip:  depth = D · k_np,   center = f_B + Δ
```

### Full Signal Equation

$$S(f) = B
  - \frac{(D / k_{np}) \cdot \omega^2}{(f - f_B + \Delta)^2 + \omega^2}
  - \frac{D \cdot \omega^2}{(f - f_B)^2 + \omega^2}
  - \frac{(D \cdot k_{np}) \cdot \omega^2}{(f - f_B - \Delta)^2 + \omega^2}$$

### Zero-Field Limit

When the hyperfine splitting vanishes ($\Delta \to 0$), the three dips merge into one
with a combined depth:

$$D_{\text{combined}} = D \cdot \left(k_{np} + 1 + \frac{1}{k_{np}}\right)$$

This is handled by the `split < 1e-10` branch in the kernel.

## Parameter Roles

| Parameter      | Symbol     | Role                                         | Identifiable? |
|----------------|------------|----------------------------------------------|:---:|
| `frequency`    | $f_B$      | Center of the main dip (location)            | ✓ |
| `linewidth`    | $\omega$   | Half-width at half-maximum of each dip       | ✓ |
| `split`        | $\Delta$   | Hyperfine splitting (dip separation)         | ✓ |
| `k_np`         | $k_{np}$   | Asymmetry ratio between left/right peaks     | ✓ |
| `dip_depth`    | $D$        | Contrast of the center dip at resonance      | ✓ |
| `background`   | $B$        | Baseline fluorescence level                  | ✓ |

Because `dip_depth` controls **absolute contrast** and `k_np` controls the **relative
ratio** between peaks, there is no degeneracy between them. Each parameter affects a
distinct geometric feature of the spectrum.

## Why `k_np` and `dip_depth` Don't Interfere

Consider measuring at the three resonance frequencies:

| Measurement at   | Observed dip                     |
|-------------------|----------------------------------|
| $f_B - \Delta$   | $D / k_{np}$                    |
| $f_B$            | $D$                              |
| $f_B + \Delta$   | $D \cdot k_{np}$               |

Three measurements, two unknowns ($D$ and $k_{np}$) — the system is **overdetermined**.
The ratio of left to right dip heights gives $k_{np}^2$; any single dip height then
gives $D$. There is no ridge.

## Implementation

### Numba Kernel (`numba_kernels.py`)

```python
@njit(cache=True)
def lorentzian_dip_term(x, center, linewidth, dip_depth):
    """dip_depth · ω² / ((x - center)² + ω²)"""
    d = (x - center) ** 2 + linewidth ** 2
    return (dip_depth * linewidth ** 2) / d

@njit(cache=True)
def nv_center_lorentzian_eval(x, freq, linewidth, split, k_np, dip_depth, background):
    left   = lorentzian_dip_term(x, freq - split, linewidth, dip_depth / k_np)
    center = lorentzian_dip_term(x, freq,         linewidth, dip_depth)
    right  = lorentzian_dip_term(x, freq + split, linewidth, dip_depth * k_np)
    return background - (left + center + right)
```

### Typed Parameters (`nv_center.py`)

```python
@dataclass(frozen=True)
class NVCenterLorentzianParams:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float      # unitless contrast [0, 1]
    background: float

    @property
    def physical_amplitude(self) -> float:
        """Recover the Hz² numerator: D · ω²."""
        return self.dip_depth * self.linewidth ** 2
```

### Generator (`core_generators.py`)

The `NVCenterCoreGenerator` samples `dip_depth` directly from `U(0.3, 0.95)`, ensuring
the true signal always has a clearly visible dip without approaching the background.

### Bayesian Priors (`belief_builders.py`)

Grid and SMC belief builders use `dip_depth ∈ [0.05, 1.5]` as the prior range. Because
$D$ is unitless and $O(1)$, it lives on the same scale as other normalized parameters,
preventing disproportionate weighting during inference.

## Voigt Extension

The Voigt model (Lorentzian convolved with Gaussian broadening) uses the same
reparametrization. `dip_depth` scales the Voigt profile amplitude, and the physical
amplitude is approximated as $D \cdot (\gamma_L)^2$ where $\gamma_L$ is the Lorentzian
half-width.
