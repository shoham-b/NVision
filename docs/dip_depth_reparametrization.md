# Amplitude Reparametrization: `dip_depth` → `c_total`

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

## The Fix: a Unitless Contrast Parameter

We reparametrize using a unitless **contrast** parameter, defined so that a single dip's
observable height no longer depends on its width — only on the contrast itself. Two
conventions exist in this codebase, used in different places:

- **`dip_depth`** (single, non-degenerate peak only — `nv_center_one_peak_lorentzian_bounds_for_domain`,
  the zero-field single-dip Lorentzian): $L(f) = D\cdot\omega^2 / [(f-f_0)^2+\omega^2]$, height $D$ at
  resonance.
- **`c_total`** (every multi-dip NV-center model — Lorentzian, Voigt, Saturation-Voigt hyperfine
  triplets, with or without Zeeman splitting): a **population-normalized** total contrast,
  described below. This is the current, standard convention for all three lineshapes and
  supersedes an earlier `dip_depth`-per-dip convention this file used to describe.

## NV Center: Population-Normalized `c_total`

The NV center ODMR signal from a hyperfine triplet is three dips sharing one linewidth and
one asymmetry ratio `k_np`, with population fractions that sum to the total contrast:

$$p_{\rm sum} = \frac{1}{k_{np}} + 1 + k_{np}, \qquad
p_L = c_{\rm total}\cdot\frac{1/k_{np}}{p_{\rm sum}}, \qquad
p_0 = c_{\rm total}\cdot\frac{1}{p_{\rm sum}}, \qquad
p_R = c_{\rm total}\cdot\frac{k_{np}}{p_{\rm sum}}$$

```
                     ┌─ Left dip:   depth = p_L = c_total·(1/k_np)/p_sum,  center = f_B − Δ
                     │
c_total ─────────────┼─ Center dip: depth = p_0 = c_total/p_sum,          center = f_B
                     │
                     └─ Right dip:  depth = p_R = c_total·k_np/p_sum,     center = f_B + Δ
```

Because $p_L + p_0 + p_R = c_{\rm total}$ by construction, `c_total` is the total observed
contrast split across the triplet according to `k_np` — it **cannot go negative** (unlike a
raw per-dip `dip_depth` that could be driven negative by an unconstrained fit), and its value
is unaffected by `k_np`'s value moving population between the three lines. This is why the
project settled on `c_total` over the older per-dip `dip_depth` convention when unifying
Lorentzian, Voigt, and Saturation-Voigt (2026-07-15): the same amplitude parameter, and the
same non-negativity guarantee, now applies uniformly across all three lineshapes.

### Full Signal Equation (Lorentzian dip term)

$$S(f) = B - \frac{p_L\cdot\omega^2}{(f-f_B+\Delta)^2+\omega^2} - \frac{p_0\cdot\omega^2}{(f-f_B)^2+\omega^2} - \frac{p_R\cdot\omega^2}{(f-f_B-\Delta)^2+\omega^2}$$

For Voigt/Saturation-Voigt, each Lorentzian dip term above is replaced by the height-normalized
pseudo-Voigt profile of [`sbed_and_smc.md` §7.2](equations/sbed_and_smc.md), with the same
$(p_L, p_0, p_R)$ population weights.

### Zero-Field Limit

When the hyperfine splitting vanishes ($\Delta \to 0$), the three dips merge into one with
depth $p_L+p_0+p_R = c_{\rm total}$ exactly (the population weights sum to the whole contrast
by construction, so no separate combined-depth formula is needed).

## Parameter Roles

| Parameter      | Symbol     | Role                                         | Identifiable? |
|----------------|------------|----------------------------------------------|:---:|
| `frequency`    | $f_B$      | Center of the main dip (location)            | ✓ |
| `linewidth` / `homogeneous_linewidth` | $\omega$ | Half-width at half-maximum of each dip | ✓ |
| `split`        | $\Delta$   | Hyperfine splitting (dip separation)         | ✓ |
| `k_np`         | $k_{np}$   | Asymmetry ratio between left/right peaks     | ✓ |
| `c_total`      | —          | Population-normalized total contrast         | ✓ |
| `background`   | $B$        | Baseline fluorescence level                  | ✓ |

Because `c_total` controls **absolute contrast** and `k_np` controls the **relative ratio**
between peaks, there is no degeneracy between them. Each parameter affects a distinct
geometric feature of the spectrum.

## Why `k_np` and `c_total` Don't Interfere

Consider measuring at the three resonance frequencies:

| Measurement at   | Observed dip                     |
|-------------------|----------------------------------|
| $f_B - \Delta$   | $p_L = c_{\rm total}(1/k_{np})/p_{\rm sum}$ |
| $f_B$            | $p_0 = c_{\rm total}/p_{\rm sum}$ |
| $f_B + \Delta$   | $p_R = c_{\rm total}\,k_{np}/p_{\rm sum}$ |

Three measurements, two unknowns ($c_{\rm total}$ and $k_{np}$) — the system is
**overdetermined**. The ratio of left to right dip heights gives $k_{np}^2$; any single dip
height combined with the known $p_{\rm sum}(k_{np})$ then gives $c_{\rm total}$. There is no
ridge.

## Implementation

### Numba Kernel (`numba_kernels.py`)

```python
@njit(cache=True)
def nv_center_lorentzian_eval(x, freq, linewidth, split, k_np, c_total, background):
    p_sum = (1.0 / k_np) + 1.0 + k_np
    p_L = c_total * (1.0 / k_np) / p_sum
    p_0 = c_total / p_sum
    p_R = c_total * k_np / p_sum
    left   = p_L * linewidth**2 / ((x - freq + split) ** 2 + linewidth**2)
    center = p_0 * linewidth**2 / ((x - freq) ** 2 + linewidth**2)
    right  = p_R * linewidth**2 / ((x - freq - split) ** 2 + linewidth**2)
    return background - (left + center + right)
```

The Voigt/Saturation-Voigt kernels (`nv_center_zeeman_pseudo_voigt_eval` and vectorized
variants) use the same $(p_L, p_0, p_R)$ split, replacing each Lorentzian dip term with the
pseudo-Voigt profile of `_pv_factors`/`_pv_norm`.

### Typed Parameters (`nv_center.py`)

```python
@dataclass(frozen=True)
class NVCenterLorentzianSpectrum:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    c_total: float      # unitless, population-normalized total contrast
```

### Generator (`nv_center_generator.py`)

`NVCenterCoreGenerator` samples `c_total` directly from `U(0.1, 0.4)` (or the pinned
`self.c_total` override), identically across the Lorentzian, Voigt, and Saturation-Voigt
branches.

### Bayesian Priors (`belief_builders.py`, `nv_center_generator.py`)

Grid and SMC belief builders use `c_total ∈ [0.1, 0.4]` as the prior range, with prior std
`0.3 × PRIOR_STD_FRACTION` (see `sim/gen/nv_center_generator.py`). Because `c_total` is
unitless and O(1), it lives on the same scale as other normalized parameters, preventing
disproportionate weighting during inference.

## Voigt Extension

The plain Voigt model (`NVCenterVoigtModel`, unified onto this convention 2026-07-15) uses the
same `c_total` amplitude convention and shares the exact hyperfine/Zeeman kernel machinery
with Lorentzian and Saturation-Voigt — only its own width decomposition differs. Voigt infers
`(homogeneous_linewidth, sigma_inhom)` directly (the physical Lorentzian HWHM and Gaussian
inhomogeneous-broadening std), reparametrized internally to the kernel-native
`(fwhm_total, lorentz_frac)` pair via `_voigt_reparam_scalar` — mirroring how Saturation-Voigt
already reparametrizes its own `(saturation, sigma_inhom)` via
`_saturation_voigt_reparam_scalar`. `sigma_inhom → 0` is the pure-Lorentzian limit
(`lorentz_frac → 1`). See [`sbed_and_smc.md` §7.2](equations/sbed_and_smc.md) for the pseudo-Voigt
profile itself, and [`equations/README.md`](equations/README.md) for the full inference-stack
equations.
