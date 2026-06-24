# Step 1 — Batch MLE Re-Evaluation (Fisher-information uncertainty)

## Context

The dashboard's accuracy/uncertainty come from the locator's *online* belief — the
posterior accumulated step-by-step. For approximate inference (SMC) that online σ
drifts (particle degeneracy, resampling), so it can read tighter than the data
warrants, and the stop decision (fires when online σ < threshold) can trigger on an
over-tight σ.

Decided direction: score each run by **batch maximum-likelihood estimation** over all
its measurements, using the locator's *own* likelihood — not a foreign curve-fit, not
a prior-dependent Bayesian re-update.

- Estimate:   θ̂ = argmax_θ Σ_i log P(y_i | x_i, θ)
- Uncertainty: Σ ≈ −[∇² log L(θ̂)]⁻¹   (observed Fisher information; SE = sqrt(diagΣ))

This is an **offline scorer**; runtime stop logic is unchanged in this step.

## Why this estimator
- **Same likelihood as the locator** ⇒ "related," not foreign. Reuse
  `noise_model.composite_log_likelihood(predicted, residuals, …)` — the exact term the
  SMC weight update sums (`nvision/belief/smc_marginal.py:433`, `:537`).
- **vs online belief:** joint MLE removes sequential approximation error → exposes
  degeneracy-induced overconfidence (`online σ ≪ batch SE`).
- **vs curve-fit:** for Gaussian noise MLE == nonlinear least squares and `−H⁻¹` ==
  `curve_fit` `pcov`; for non-Gaussian noise (Poisson/drift in
  `nvision/spectra/noise_model.py`) MLE uses the true likelihood, where least squares
  would be wrong.
- **vs Bayesian batch:** no prior — reports what the data alone says (cleaner neutral
  scorer). Cost: Σ is a local Gaussian (Laplace) approx around one mode (loses
  multimodality — acceptable for a scorer).

## Validity guards (replace ad-hoc min-points / R² gates)
1. **Hessian positive-definite at θ̂.** Singular/indefinite ⇒ a likelihood-flat
   direction ⇒ unidentified parameter ⇒ Σ → ∞. Mark **N/A**, emit no number. This is
   the principled identifiability check and subsumes "too few measurements."
2. **Σ is asymptotic (large-N), optimistic for small N.** Flag short runs; don't trust
   tight Σ from few points.
3. **Local optimum.** Multi-start / warm-start from the belief mode to avoid spurious
   local maxima.
4. (Future, not now) sandwich covariance `H⁻¹ J H⁻¹` for model-mismatch experiments;
   plain `−H⁻¹` is correct under matched noise.

## Key facts (codebase)
- Likelihood term: `composite_log_likelihood` on every noise model
  (`nvision/spectra/noise_model.py:33/87/141/169/204`); builder
  `nvision/spectra/likelihood.py:50`.
- Forward models: `compute(x, params)` — `lorentzian.py:83`, `gaussian.py:73`,
  `voigt_zeeman.py:128`, `nv_center.py`, `composite.py:109`. Need a float-vector <->
  typed-params adapter to optimize/differentiate over the tracked params.
- Per-repeat measurements assembled at `nvision/runner/metrics.py:52` (`x`,
  `signal_values`); ground truth from `true_signal`. Online σ-at-stop available from
  the belief snapshot.
- Runtime stop is belief-σ-driven (`_target_params_converged` in `sbed_locator.py:294`,
  `sobol_bayesian_locator.py:130`, sequential locator) — NOT touched here.

## Changes

### 1. New module `nvision/metrics/batch_mle.py`
- `MleResult`: `params: dict[str,float]`, `se: dict[str,float]`, `success: bool`,
  `n_points: int`, `cond_number: float` (Hessian conditioning, for the N/A gate).
- `batch_mle(model, noise_model, xs, ys, fit_params, fixed_params, p0=None) -> MleResult`:
  - objective `nll(θ) = −Σ composite_log_likelihood(compute(x_i,θ), y_i−compute(x_i,θ), …)`.
  - minimize via `scipy.optimize.minimize` (or `least_squares` in the Gaussian case).
  - Hessian at θ̂ by **numerical second differences of the nll** (do NOT rely on the
    optimizer's `hess_inv`); `Σ = inv(H)`; `se = sqrt(diag(Σ))`.
  - PD/conditioning gate → `success=False`, inf SE on failure.
  - `p0` warm-start from belief mode; optional multi-start.

### 2. Per-repeat wiring (`nvision/runner/metrics.py`)
- Emit `mle_est_{param}`, `mle_se_{param}`, `mle_err_{param}` (= `|mle_est − truth|`),
  plus `mle_ok`. Keep online σ-at-stop as its own field (for calibration).

### 3. Headline = MLE; online σ retained only for calibration
- Dashboard error/uncertainty switch to MLE estimate / SE.
- Calibration panel (`static/app.js:4642`) reframed to its true purpose — auditing the
  stop — as **online σ-at-stop vs MLE SE** (overconfident stops ⇒ online σ ≪ SE).

### 4. Cross-repeat aggregation (`nvision/metrics/calculator.py` + `types.py`)
- Median MLE error + bootstrap median CI (cross-repeat, A-vs-B only). Drop
  `mean_steps_to_convergence`; no IQR. Always pair MLE error with measurement count in
  any ranking.

### 5. Tests (`tests/test_metrics.py`)
- Noiseless dense dip: `mle_err≈0`, small SE, PD Hessian.
- Gaussian case: `batch_mle` SE matches `scipy.curve_fit` pcov within tolerance
  (cross-check the Fisher computation).
- Non-identified case (few/one-shoulder points): `success=False`, inf SE, no crash.
- Injected-overconfident online σ: `MLE SE > online σ` (panel would flag).
- Aggregator: median + CI; outlier repeat does not move the median.

## Open items to confirm during build
- Optimize in unit space (via RescaleMap) vs physical space — pick the better-
  conditioned one; report SE back in physical units.
- Multi-start count / warm-start source.
- Hessian step size for numerical differentiation.

## Out of scope
Runtime stop/convergence changes; sandwich covariance; error normalization; baseline
integrity tests; cost-vs-error curves; UI reorganization (separate, pending).

## Verification
- `pytest tests/test_metrics.py -q`.
- Gaussian cross-check vs `curve_fit` pcov.
- Overconfidence check: MLE SE exceeds an injected-tight online σ.
- Confirm `viz/metrics.py` plots still generate (untouched raw lists).
