# SBED + SMC — Equations and Approximations

The SMC belief, its unit-cube extension, the SBED acquisition locator, the Gaussian Fisher/CRLB, and the convergence criteria form one inference stack and are documented together.  Symbols are defined at first use; defaults are the env-var values from `nvision/sim/defaults.py`.  Per-run evaluation metrics are in [metrics.md](metrics.md).

> Scope: only the additive Gaussian measurement-noise path. Rao-Blackwell (Inverse-Gamma) noise, Poisson likelihoods, and non-SBED locators are out of scope.

---

## 1. SMC Particle Filter (`smc_marginal.py`, Gaussian noise)

### 1.1 Bayesian Weight Update

For each new observation (x, y) the un-normalised weight of particle i is multiplied by its likelihood, then normalised:

$$\tilde{w}_i \leftarrow w_i \cdot p(y \mid \theta_i, x), \qquad w_i \leftarrow \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}$$

Under additive Gaussian noise with measurement std σ, the log-likelihood is

$$\log p(y \mid \theta_i, x) = -\frac{1}{2}\left(\frac{y - S(x, \theta_i)}{\sigma}\right)^2$$

where S(x, θ_i) is the signal model evaluated at particle i.  The constant −½·ln(2πσ²) normalization term is dropped because σ is the same for all particles, so it cancels under normalisation.  A `tempering_factor` β replaces σ with σ/√β (equivalent to scaling the log-likelihood by β).

**Numerically stable update** (in-place, avoids underflow):

```
log_w ← log(max(w, 1e-30)) + log_lik
log_w ← log_w − max(log_w)      # shift for numerical stability
w ← exp(log_w) / sum(exp(log_w))
```

### 1.2 Effective Sample Size (ESS)

$$\text{ESS} = \frac{1}{\sum_i w_i^2}$$

Resampling is triggered when ESS < r_thr · N, where r_thr = `ess_threshold` = 0.20 (default) and N = `num_particles`.

### 1.3 Systematic Resampling

Positions u_k = (k + U)/N for k = 0, …, N−1 with a single draw U ~ Uniform(0, 1/N) are mapped to particle indices by walking the CDF of the weights.  This has lower variance than multinomial resampling.

### 1.4 Liu-West Kernel (shrinkage + nudge)

After resampling, particles are contracted toward the pre-resample weighted mean μ and then jittered:

**Shrinkage (contraction):**

$$\theta_i \leftarrow a \cdot \theta_i + (1 - a) \cdot \mu$$

**Nudge:**

$$\theta_i \leftarrow \theta_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}\!\left(\mathbf{0},\, (1 - a^2)\Sigma\right)$$

where Σ is the pre-resample weighted covariance and a = `a_param` = 0.98 (default).

The nudge covariance C = (1−a²)Σ has its diagonal floored to enforce exploration:

$$C_{jj} \leftarrow \max\!\left(C_{jj},\, \left[(h_j - l_j) \cdot f_{\rm expl} \cdot e^{-t/25}\right]^2\right)$$

with f_expl = `min_exploration_frac` = 0.01, t = `_step_count`, and [l_j, h_j] the bound of parameter j.  An eigenvalue decomposition then regularises C (minimum eigenvalue clamped to max(1e−11, 1e−6·λ_max)) to restore positive-definiteness before the Cholesky draw.

**Particle rejuvenation:** A fraction f_rejuv = 0.05·e^(−t/25) of particles are replaced by fresh uniform prior draws to prevent mode collapse.

### 1.5 Weighted Statistics

**Weighted mean** (estimate):

$$\mu_j = \frac{\sum_i w_i \theta_{ij}}{\sum_i w_i}$$

**Weighted variance / std** (uncertainty):

$$\operatorname{Var}(\theta_j) = \frac{\sum_i w_i (\theta_{ij} - \mu_j)^2}{\sum_i w_i}, \qquad \sigma_j = \sqrt{\operatorname{Var}(\theta_j)}$$

**Weighted covariance matrix** (used for the nudge kernel and entropy):

$$\Sigma = \frac{\sum_i w_i (\theta_i - \mu)(\theta_i - \mu)^\top}{\sum_i w_i}$$

### 1.6 Expected Information Gain (EIG)

The SBED acquisition criterion uses a closed-form approximation based on predictive variance:

$$\text{EIG}(x) \approx \frac{1}{2} \ln\!\left(1 + \frac{\sigma^2_{\rm pred}(x)}{\sigma^2_{\rm noise}}\right)$$

where the **prediction variance** across (a subsample of) particles is

$$\sigma^2_{\rm pred}(x) = \left(\sum_i w_i S(x, \theta_i)^2\right) - \left(\sum_i w_i S(x, \theta_i)\right)^2$$

and the **noise variance** is the Gaussian measurement floor σ²_noise = max(σ², 1e−12).

This is the variance of the signal prediction under the current posterior — a proxy for the mutual information between the measurement outcome and the parameters.  Derivation: for a Gaussian observation y = S + η with prior predictive variance σ²_pred and noise variance σ²_noise, the information gain (entropy reduction) about S is exactly ½·ln(1 + σ²_pred/σ²_noise).

#### EIG Subsampling

When N > N_EIG (default 500), a stratified subsample of N_EIG particles is drawn before the EIG prediction matrix is built.  Variance estimation converges with ~200–500 particles, so the quality loss is negligible while the matrix shrinks from O(n_cand × N) to O(n_cand × N_EIG).

#### EIG Prediction-Matrix Cache (`NVISION_SMC_EIG_CACHE`)

Between resamples the particle positions and candidate grid are frozen, so the prediction matrix M[c, i] = S(x_c, θ_i) and its element-wise square M⁽²⁾ = M ⊙ M are invariant.  They are built once per epoch, and the per-step variance is recovered with two matrix-vector products against the current weights:

$$\sigma^2_{\rm pred} = M^{(2)} \mathbf{w} - (M \mathbf{w})^2$$

No model re-evaluation is needed per step; only the weight vector changes between resamples.

### 1.7 Entropy (Gaussian Approximation)

The posterior entropy is approximated by assuming the particles are Gaussian with empirical covariance Σ:

$$H \approx \tfrac{1}{2} \ln|\Sigma| + \tfrac{d}{2}(1 + \ln 2\pi)$$

where d is the number of parameters.  The log-determinant is computed via `slogdet` after adding a tiny diagonal ridge ε = 1e−12·(tr(Σ)/d + 1e−15).

### 1.8 Epoch Candidate Grid

After each resample a new slope-targeted grid is generated in frequency space.  Six slope points (two per dip, at dip center ± linewidth HWHM) define dense local windows:

$$\text{slope points} = \{f_B \pm \Delta f_{\rm hf}\} \pm \Omega_{\rm hw}$$

where f_B = posterior mean frequency, Δf_hf = posterior mean split, Ω_hw = posterior mean linewidth (HWHM).

Each window has half-width 3·σ_eff and step max(σ_eff/30, Δ_min), where

$$\sigma_{\rm eff} = \sqrt{\sigma_f^2 + \sigma_\Omega^2}$$

is the quadrature of posterior frequency and linewidth uncertainties, and Δ_min = `NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ` = 10 kHz.  A global coarse grid with n_global = ⌈W / s_min · P⌉ points (W = bandwidth, s_min = minimum feature width, P = `POINTS_PER_MIN_FEATURE` = 5) is merged with the local grids.

---

## 2. Unit-Cube Extension (`unit_cube_smc_marginal.py`)

Particles live in unit space [0, 1]ᵈ; physical values are recovered by affine rescaling.

### 2.1 Unit-to-Physical Mapping

For parameter j with physical bounds [l_j, h_j]:

$$\theta^{\rm phys}_j = l_j + u_j \cdot (h_j - l_j)$$

The frequency axis additionally keeps `_original_physical_x_bounds` = the full domain at construction, which never narrows, so the stored unit observation coordinate o.x always converts correctly:

$$f^{\rm phys} = l_{\rm orig} + o.x \cdot (h_{\rm orig} - l_{\rm orig})$$

even after the search window has been narrowed.

### 2.2 Physical Uncertainty

The unit-space std σᵘ_j is scaled by the physical range:

$$\sigma^{\rm phys}_j = \sigma^u_j \cdot (h_j - l_j)$$

### 2.3 Analytical CRLB for Frequency (Lorentzian, Gaussian noise)

For a Lorentzian signal measured under Gaussian noise with uniform measurement density ρ = N/W (measurements per Hz, W = bandwidth), the closed-form Cramér-Rao lower bound on frequency variance is:

$$\text{Var}^{\rm CRLB}(f) = \frac{2\sigma^2 \Omega}{\pi c^2 \rho}, \qquad \text{CRLB}_f = \sqrt{\text{Var}^{\rm CRLB}(f)}$$

where σ = noise std, Ω = linewidth FWHM (Hz), c = `c_total` (contrast).

**Derivation:** write the signal as S = c·L with L(x) = (Ω/2)² / [(Ω/2)² + (x−f)²] (HWHM = Ω/2).  The Gaussian Fisher information for one measurement at x is (∂S/∂f)²/σ².  The amplitude-free derivative integral evaluates to

$$\int_{-\infty}^{\infty}\left(\frac{\partial L}{\partial f}\right)^2 dx = \frac{\pi}{2\Omega},$$

so integrating over the uniform density ρ gives I(f) = (ρ/σ²)·c²·(π/2Ω) = π·c²·ρ / (2σ²Ω), hence Var^CRLB = 1/I(f).  (Here `linewidth` plays the role of the FWHM Ω, per the code's docstring.)

### 2.4 Focus-Window Narrowing (at each resample)

The search window narrows to the union of particle-predicted active regions.  Each particle i covers

$$[f_i - \Delta f_{\rm hf,i} - k\Omega_i,\quad f_i + \Delta f_{\rm hf,i} + k\Omega_i]$$

with cover factor k = `NVISION_SMC_FOCUSING_COVER_FACTOR` = 3.0.  The new bounds are the p-th / (1−p)-th percentiles of the left/right edges, with p = `NVISION_SMC_FOCUSING_TAIL_PERCENTILE` = 1.0 %.

When particles pile up at a unit boundary (> 15 % within 5 % of the edge), the window is instead expanded by max(cur_width, 10·Ω) in that direction.

---

## 3. SBED Locator (`sbed_locator.py`)

### 3.1 EIG Acquisition

The locator calls `belief.select_max_information_gain(candidates)`, which maximises the EIG of §1.6:

$$x^* = \arg\max_x \frac{1}{2}\ln\!\left(1 + \frac{\sigma^2_{\rm pred}(x)}{\sigma^2_{\rm noise}}\right)$$

Candidates come from the belief's slope-targeted epoch grid and are thinned to a minimum physical spacing of `candidate_step_hz` (default = 100 kHz = the frequency convergence threshold).

**Boltzmann chunk selection:** EIG scores are split into chunks of 64; each chunk's argmax competes via a softmax with temperature τ = 0.01,

$$P(\text{chunk } c) \propto \exp\!\left(\frac{\text{EIG}_c - \max_{c'}\text{EIG}_{c'}}{\tau}\right),$$

which avoids locking onto a single numerical-noise peak.

### 3.2 Exploration / Dip-Bias / EIG Mix

At each acquisition step a single uniform draw u selects the branch:

| Condition | Action | Notes |
|-----------|--------|-------|
| u < 0.1·e^(−t/25) | Uniform global sample | Decaying exploration probability |
| u < 0.20 | Sample near an empirical dip centroid ± 5 MHz jitter | Corrects posterior bias |
| otherwise | Full EIG maximisation | Main path |

with t = `inference_step_count`.  The factor e^(−t/25) makes global exploration decay exponentially so steps concentrate on EIG as the scan progresses.

### 3.3 Background Noise Estimation (MAD)

Measurements with |x − f̂| > k·|Ω̂| are classified as **background** (k = `NVISION_NOISE_BG_SPAN_FACTOR` = 3.0).  The Gaussian noise std is estimated robustly from the background scatter:

$$\hat\sigma_{\rm bg} = 1.4826 \cdot \operatorname{median}\!\left(\left|y_i^{\rm bg} - \operatorname{median}(y^{\rm bg})\right|\right)$$

The factor 1.4826 = 1/Φ⁻¹(0.75) makes the MAD a consistent estimator of σ for Gaussian data.  Returns `None` (triggering forced calibration) when fewer than `NVISION_NOISE_MIN_BG_POINTS` = 15 background points exist.

### 3.4 Forced Background Calibration

When σ̂_bg is unavailable, the locator samples uniformly from the two background regions

$$[f_{\rm lo},\, \hat{f} - k\hat\Omega] \;\cup\; [\hat{f} + k\hat\Omega,\, f_{\rm hi}]$$

with probability proportional to each region's width, until enough background points accumulate.

### 3.5 Theoretical Step Budget (backstop)

Once σ̂_bg is available and contrast ĉ > 0, a permissive backstop budget is computed from the uniform-sampling CRLB of §2.3:

$$n_{\rm theory} = \frac{2\hat\sigma_{\rm bg}^2\, \hat\Omega\, W}{\pi \hat{c}^2 T^2}$$

where T = `NVISION_FREQ_CONVERGENCE_THRESHOLD` = 100 kHz and W = bandwidth.  (This is exactly n such that Var^CRLB(f), evaluated at ρ = n/W, equals T².)  The applied limit is

$$N_{\rm budget} = \max(N_{\rm max},\; K_{\rm theory}\cdot n_{\rm theory} + 1)$$

with K_theory = `NVISION_SBED_STEPS_THEORY_FACTOR` = 20, so it only fires when something has genuinely gone wrong — EIG should converge far sooner.

### 3.6 Focus Window Confidence (`FocusWindowConfidence`)

Computed at each resample by merging two independent signals:

- **Empirical** (dip detector): dominant dip cluster bounds [l, r] and detector confidence.
- **Posterior** (particles): weighted-mean frequency f̄, std σ_f, and 16th/84th-percentile CI.

The window is flagged **stable** (`is_stable`) when all of:
1. Detector confidence ≥ `NVISION_DIP_CONFIDENCE` = 0.99
2. `methods_agree`: l ≤ f̄ ≤ r
3. σ_f < ρ_stab·(r − l), with `stability_ratio` ρ_stab = 0.5

---

## 4. Gaussian Fisher Information & CRLB (`fisher_information.py`, `abstract_marginal.py`)

### 4.1 Single-Observation Gaussian Fisher Matrix

$$\mathbf{I}(\theta; x) = \frac{1}{\sigma^2}\, \nabla_\theta S(x, \theta)\, \nabla_\theta S(x, \theta)^\top$$

### 4.2 Cumulative FIM and Marginal CRLB

The cumulative Fisher information over all observations is

$$\mathbf{I}_{\rm cum} = \sum_{n=1}^{N} \mathbf{I}(\theta; x_n)$$

and the marginal CRLB for parameter j is the diagonal of the inverse, computed via a ridge-regularised Moore-Penrose pseudo-inverse (ε = 1e−6):

$$\text{CRLB}_j = \sqrt{\left[(\mathbf{I}_{\rm cum} + \epsilon \mathbf{I})^+\right]_{jj}}$$

---

## 5. Convergence Criteria (`sequential_bayesian_locator.py`, `defaults.py`)

### 5.1 Per-Parameter Convergence

**Frequency** (absolute ceiling):

$$\sigma_f < T_f = 100\,\text{kHz} \quad (\texttt{NVISION\_FREQ\_CONVERGENCE\_THRESHOLD})$$

**Other parameters** — absolute ceiling if the env-var is set, otherwise relative to bound width:

$$\sigma_j < \text{threshold} \times (h_j - l_j), \qquad \text{threshold} = 0.01 = 1\%$$

### 5.2 Overall RMS Convergence

Even when every individual parameter passes, an overall RMS check is also required:

$$u_j = \frac{\sigma_j}{B_j}, \qquad B_j = \begin{cases} T_f / \text{threshold} & j = \text{frequency} \\ h_j - l_j & \text{otherwise} \end{cases}$$

$$\text{RMS} = \sqrt{\frac{1}{d}\sum_j u_j^2} < \text{threshold}$$

Both the per-parameter and RMS checks must pass simultaneously.

### 5.3 Convergence Patience Streak

A running counter `_convergence_streak` is incremented whenever `_target_params_converged()` returns `True` (evaluated once per resample, sharing a single uncertainty pass).  Convergence is declared only when the streak reaches `convergence_patience_steps` = 8 consecutive successes; any failure resets it to 0.

### 5.4 Dynamic CRLB Budget (base locator)

From `_acquisition_done()`, the steps needed to reach T_f at the current sampling density are estimated as

$$n_{\rm req} = t \cdot \left(\frac{\text{CRLB}_f}{T_f}\right)^2$$

(t = current step count; valid because CRLB_f ∝ 1/√N, so CRLB_f(t) = κ/√t and n_req = κ²/T_f²).

- If n_req > N_max·K_safety: stop immediately (infeasible).
- Otherwise the dynamic budget is N_dyn = min(N_max, K_safety·n_req + 1); stop when `inference_step_count` ≥ N_dyn.

with K_safety = `NVISION_FREQ_CRLB_SAFETY_FACTOR` = 4.0.

### 5.5 CRLB Early-Stop in SBED (`_check_crlb_early_stop`)

The background noise estimate rescales the FIM-based CRLBs (which were accumulated at the nominal noise std) — valid because the CRLB std scales linearly with σ:

$$\text{CRLB}^{\rm scaled}_j = \text{CRLB}^{\rm FIM}_j \cdot \frac{\hat\sigma_{\rm bg}}{\sigma_{\rm nominal}}$$

Convergence is then declared per parameter against the safety-factored CRLB:

$$\sigma_j < K_{\rm safety}\cdot \text{CRLB}^{\rm scaled}_j$$

- All target parameters pass the CRLB gate → sets `_is_converged = True`.
- Frequency passes (CRLB **or** absolute threshold) → records `freq_converged_step`.
- All parameters pass (CRLB **or** absolute threshold each) → records `all_converged_step`.

---

## 6. Key Default Constants

| Symbol | Name | Default | Unit |
|---|---|---|---|
| N | `NVISION_SMC_NUM_PARTICLES` | 1 000 | — |
| r_thr | `NVISION_SMC_ESS_THRESHOLD` | 0.20 | — |
| a | `NVISION_SMC_A_PARAM` | 0.98 | — |
| N_EIG | `NVISION_SMC_EIG_PARTICLES` | 500 | — |
| Δ_min | `NVISION_SMC_EPOCH_GRID_MIN_STEP_HZ` | 10 000 | Hz |
| T_f | `NVISION_FREQ_CONVERGENCE_THRESHOLD` | 100 000 | Hz |
| K_safety | `NVISION_FREQ_CRLB_SAFETY_FACTOR` | 4.0 | — |
| K_theory | `NVISION_SBED_STEPS_THEORY_FACTOR` | 20 | — |
| k_bg | `NVISION_NOISE_BG_SPAN_FACTOR` | 3.0 | linewidths |
| N_bg,min | `NVISION_NOISE_MIN_BG_POINTS` | 15 | — |
| patience | `NVISION_CONVERGENCE_PATIENCE` | 8 | steps |
| threshold | `NVISION_CONVERGENCE_THRESHOLD` | 0.01 | relative |
| p_conf | `NVISION_DIP_CONFIDENCE` | 0.99 | — |
| f_expl | `NVISION_SMC_MIN_EXPLORATION_FRAC` | 0.01 | — |
