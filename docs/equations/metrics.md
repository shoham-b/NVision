# Evaluation Metrics — Equations and Approximations

Per-run scoring computed after a locator finishes (`metrics.py`, `sequential_bayesian_locator.py`).  The inference-side math (SMC, SBED, CRLB, convergence) is in [sbed_and_smc.md](sbed_and_smc.md).

> Scope: only the additive Gaussian measurement-noise path.

---

## 1. Point Estimate Error

Single-peak case:

$$\text{abs\_err\_x} = |\hat{f} - f_{\rm true}|$$

where f̂ is the posterior-mean frequency estimate and f_true is the ground-truth peak position.

Two-peak case (peaks sorted before pairing, so estimate k is compared to truth k):

$$\text{abs\_err\_x}_k = |\hat{f}_k - f_{{\rm true},k}|, \qquad \text{pair\_rmse} = \sqrt{\tfrac{1}{2}\left(\text{abs\_err\_x}_1^2 + \text{abs\_err\_x}_2^2\right)}$$

---

## 2. Uniform-Sampling Baseline (and why the sweep step count is what it is)

A uniform / Sobol sweep places points evenly across the whole frequency band.  Its step count is set by one principle: **a dip can only be found if enough sample points land inside it.**  If the narrowest feature is width w and the band is W wide, uniform points spaced W/n apart fall inside that feature only if the spacing is smaller than the feature — and to actually *resolve* it (not just clip an edge) you need several points across it.  That gives the master relation used everywhere below:

$$n = \frac{W}{w} \times (\text{samples per feature})$$

The two pieces are (a) the feature width w, taken from the signal model, and (b) how many samples we insist on placing inside it.

### 2.1 Feature width from the signal model (`signal_min_span`)

The "narrowest feature" is the smallest dip the model can produce, derived from its lineshape and parameter bounds.  For the NV-center models (`nv_center.py`):

| Model | `signal_min_span` | Reasoning |
|---|---|---|
| Lorentzian | 4 · linewidth_lo, linewidth_lo = 1e−4·W | A Lorentzian's HWHM **is** the `linewidth`, so its FWHM = 2·linewidth and it only returns near baseline by ≈ ±2·HWHM ⇒ a resolvable feature spans ≈ 4·linewidth. |
| Voigt | 2 · fwhm_total_lo, fwhm_total_lo = 70 kHz | `fwhm_total` is already the full width at half maximum, so the feature spans ≈ 2·FWHM to its wings. |
| One-peak Lorentzian | 4 · linewidth_lo | Same as the Lorentzian doublet, single dip. |

`signal_max_span` is the opposite extreme — the full envelope of the widest triplet, e.g. Lorentzian 2·split_hi + 4·linewidth_hi (two outer dips plus their wings).  It is used as a fallback when the minimum span is degenerate.

### 2.2 Sweep step *budget* (`compute_sweep_max_steps`)

This sets the `max_steps` ceiling a real sweep locator is allowed.  It demands `coverage_factor` samples inside the narrowest dip:

$$n_{\rm sweep} = \frac{W}{w_{\rm eff}} \times c_{\rm cov}, \qquad c_{\rm cov} = \texttt{NVISION\_SWEEP\_COVERAGE\_FACTOR} = 3.0$$

- **Why 3 samples per dip:** two points can straddle a dip without either landing near the minimum; three guarantees at least one interior sample close to the center, enough to register the dip and seed a fit.  It is the smallest count that reliably *detects* a feature rather than aliasing past it.
- **Span floor:** w_eff is clamped to ≥ 0.2 % of the domain (`max(min_span, 0.002·W)`).  `signal_min_span` can return an absurdly small lower bound from the parameter space (e.g. 0.01 % of W), which would demand tens of thousands of points; the floor caps that.
- **Hard clamps:** the result is clamped to [`NVISION_SWEEP_MIN_STEPS`, `NVISION_SWEEP_MAX_STEPS`] = [50, 500].  The floor guarantees minimal coverage of even a trivially wide dip; the ceiling caps cost on pathologically narrow ones.

### 2.3 Sweep step *baseline* for scoring (`_compute_expected_uniform_points`)

This is the benchmark SBED is compared against — how many uniform points a sweep would have needed on **this specific true signal** (not the parametric worst case).  It measures the real dip width instead of the model bound:

1. Evaluate the ground-truth signal on a fine grid of n = 20 000 points across the band W.
2. Background level = 95th percentile of the signal; dip depth d = background − min(y).
3. Mark points below background − max(0.2·d, 3e−6) as "in a dip"; extract contiguous segments ≥ 3 grid points.
4. Merge segments separated by ≤ 2 % of the domain.

Segments are classified as **merged** (one effective feature) when their total span is within 1.5× their total width:

$$\text{merged} \iff \text{span}_{\rm total} \leq 1.5 \cdot \text{width}_{\rm total}$$

The effective feature width is

$$w_{\rm eff} = \begin{cases} \text{span}_{\rm total} & \text{merged} \\ \text{width}_{\rm total} & \text{separate dips} \end{cases}$$

and the expected uniform step count is

$$\text{expected\_uniform\_points} = \frac{2\,W}{w_{\rm eff}}$$

- **Why 2 here (vs 3 in the budget):** this is a *lower bound* on what a sweep needs — the Nyquist minimum of ~2 samples across the narrowest feature, the theoretical floor for not aliasing past it.  The budget uses 3 because it must guarantee usable detection in practice; the baseline uses 2 because it answers "what is the least a sweep could get away with," making the SBED-vs-sweep comparison conservative (it does not inflate the sweep cost).
- **Merged vs separate:** if the hyperfine dips overlap into one blob, the binding constraint is the whole span; if they are cleanly separated, each narrow dip must be resolved individually, so the total dip width is what matters.
- **Fallback:** when no dip is resolvable, it falls back to 6·W / s_min (≈ 6 samples across the model's minimum span) or, failing that, `max_steps`.

### 2.4 Full derivation: why the sweep baseline ≈ 2500 steps

The 2500-step figure for the standard NV scan is not a constant — it is `expected_uniform_points` evaluated on the default signal.  Derived from first principles:

**Step 1 — Sampling principle (≥ 2 points per feature).**  A feature of width w is only resolvable if the uniform spacing δ puts at least two samples inside it (one point can sit anywhere on the dip; two guarantee you bracket it instead of aliasing past it).  This is the Nyquist limit:

$$\delta \le \frac{w}{2}$$

**Step 2 — Feature width from the lineshape (w = 4Ω).**  Ω denotes the `linewidth`, which for the Lorentzian lineshape used here is the **HWHM** (so the full width at half maximum is FWHM = 2Ω).  A dip's edges are detected at the 20 %-of-depth threshold (`threshold_frac = 0.2`).  A Lorentzian dip is

$$\text{drop}(\Delta) = \text{depth}\cdot\frac{\Omega^2}{\Omega^2 + \Delta^2}, \qquad \Delta = f - f_{\rm dip},\; \Omega = \text{linewidth (HWHM)}$$

Setting the drop equal to 20 % of depth gives the half-width:

$$\frac{\Omega^2}{\Omega^2 + \Delta^2} = 0.2 \;\Longrightarrow\; \Omega^2 + \Delta^2 = 5\Omega^2 \;\Longrightarrow\; \Delta = 2\Omega$$

so the full detected dip width is

$$w = 2\Delta = 4\Omega = 4 \cdot \text{linewidth}$$

**Step 3 — Combine (n = 2W/w).**  Substitute w into the spacing bound and count how many points of that spacing span the band W:

$$\delta \le 2\Omega, \qquad n = \frac{W}{\delta} = \frac{W}{2\Omega} = \frac{2W}{w}$$

The form n = 2W/w is exactly the code's `expected_uniform_points = 2·domain_width / dip_width`.

**Step 4 — The two physical inputs.**

| Quantity | Value | Source |
|---|---|---|
| Band W | 3.1 − 2.6 GHz = 500 MHz | `NVISION_NV_CENTER_FREQ_X_MIN/MAX` |
| Linewidth Ω | ≈ 100 kHz (effective narrow dip, HWHM) | true-signal linewidth |
| Dip width w = 4Ω | ≈ 400 kHz | Step 2 |

**Step 5 — Plug in.**

$$n = \frac{2W}{w} = \frac{2 \times 500\,\text{MHz}}{400\,\text{kHz}} = \frac{10^9}{4\times10^5} = 2500$$

equivalently n = W / (2Ω) = 500 MHz / 200 kHz = 2500.

**Step 6 — Scaling / caveats.**  The whole result reduces to n = W / (2Ω) ∝ W / Ω, so it is **config-dependent**: doubling the band → ≈ 5000; broadening the line to Ω = 1 MHz → ≈ 250.  The 2500 is the value for the default 500 MHz scan with a ~100 kHz line, not a fixed constant.  It surfaces as the sweep's `measurements_done` cap in the UI and as the `max_steps` chosen for sweep-baseline test runs.

The reported **`sobol_difference`** is

$$\text{sobol\_difference} = \text{expected\_uniform\_points} - \text{measurements}$$

— positive when SBED used fewer steps than a uniform sweep would have.

---

## 3. Failure Classification

A run's `failure_reason` (`None` = success) is assigned by the first matching rule:

| `failure_reason` | Condition |
|---|---|
| `None` (success) | `splitting_converged_step` is set, **or** the converged flag is True |
| `None` | Strategy is a sweep/Sobol/mixture baseline (no convergence gate) |
| `infeasible_crlb` | Stop reason was `infeasible_crlb` |
| `timeout` | Stop reason was `repeat_timeout` |
| `theory_budget` | `locator_steps > theory_step_budget` (§3.5 of [sbed_and_smc.md](sbed_and_smc.md)) |
| `None` | `locator_steps < max_steps` (stopped early for another reason) |
| `max_steps` | Otherwise — exhausted the step budget without converging |

---

## 4. Other Forwarded Metrics

Computed elsewhere and passed through `metrics.py` for reporting:

- **`measurements`** — total observations recorded for the repeat.
- **`final_est_<param>`** — final posterior-mean estimate for each parameter (`frequency`, `linewidth`/`homogeneous_linewidth`, `split`, `sigma_inhom`, `c_total`, `k_np`), populated dynamically off whatever `parameter_names()` the model returns rather than a hardcoded list.
- **`uncert`** — reported frequency uncertainty (see `reported_uncertainty` in [sbed_and_smc.md](sbed_and_smc.md) §2; floored at K_safety × CRLB_f).
- **`splitting_converged_step` / `all_converged_step`** — first step at which the primary parameter (`zeeman_split`/`split` when present, else `frequency`) / all target parameters converged.
- **`duration_ms`** — wall-clock runtime of the repeat.
- **Milestone metrics** (`steps_to_fb`, `err_fb_at_milestone`, `uncert_fb_at_milestone`) — from `calculate_zeeman_metrics`, now called with the same primary-parameter resolution (`resolve_primary_param`) so `fb` means splitting once frequency is fixed, rather than always being vacuous.
