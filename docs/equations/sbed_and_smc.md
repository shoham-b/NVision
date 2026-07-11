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

$$\text{Var}^{\rm CRLB}(f) = \frac{4\sigma^2 \Omega}{\pi c^2 \rho}, \qquad \text{CRLB}_f = \sqrt{\text{Var}^{\rm CRLB}(f)}$$

where σ = noise std, Ω = `linewidth` (Hz, **HWHM** — the code's `omega` denominator directly, not FWHM), c = `c_total` (contrast).

**Derivation:** write the signal as S = c·L with L(x) = Ω² / [Ω² + (x−f)²] (unit height at x=f, HWHM = Ω — matches `nv_center_lorentzian_eval`'s `1/(x_dim²+1)` with `x_dim=(x−f)/Ω` exactly). The Gaussian Fisher information for one measurement at x is (∂S/∂f)²/σ². The amplitude-free derivative integral evaluates to

$$\int_{-\infty}^{\infty}\left(\frac{\partial L}{\partial f}\right)^2 dx = \frac{\pi}{4\Omega},$$

(verified by direct numerical integration against the coded kernel), so integrating over the uniform density ρ gives I(f) = (ρ/σ²)·c²·(π/4Ω) = π·c²·ρ / (4σ²Ω), hence Var^CRLB = 1/I(f). An earlier version of this derivation used Ω to mean FWHM (HWHM = Ω/2) while the code's `linewidth` is the HWHM directly — that unit mismatch produced a Var^CRLB two times too small; both `crlb_frequency()` and the SBED `n_theory` backstop have been corrected to the `4σ²Ω` form above.

For the saturation-coupled Voigt model (§7), Ω is replaced by the general lineshape integral J = ∫(V′)²dx of the height-normalized pseudo-Voigt profile V = elf/(x²+γ_hom²) + egf·exp(−x²/2σ_inhom²) (the same `elf`, `egf` unit-height factors the pseudo-Voigt kernels use, from `_pv_factors(fwhm_total, lorentz_frac)`). Integrating each term independently (cross-term dropped — a single-dip approximation that is conservative, i.e. Var^CRLB is an overestimate for genuinely mixed lineshapes) gives closed forms for each piece:

$$\int\left(\frac{\partial}{\partial x}\frac{{\rm elf}}{x^2+\gamma_{\rm hom}^2}\right)^2 dx = {\rm elf}^2\cdot\frac{\pi}{4\gamma_{\rm hom}^5}, \qquad \int\left(\frac{\partial}{\partial x}\,{\rm egf}\cdot e^{-x^2/2\sigma_{\rm inhom}^2}\right)^2 dx = {\rm egf}^2\cdot\frac{\sqrt{\pi}}{2\sigma_{\rm inhom}}$$

$$J = {\rm elf}^2\frac{\pi}{4\gamma_{\rm hom}^5} + {\rm egf}^2\frac{\sqrt{\pi}}{2\sigma_{\rm inhom}}, \qquad \text{Var}^{\rm CRLB}(f) = \frac{\sigma^2}{\rho\, c_{\rm total}^2\, J}$$

where `c_total = c_max·s/(1+s)` is the realized (saturation-scaled) contrast from §7. Both terms were verified by direct numerical integration and reduce exactly to the Lorentzian J = π/(4Ω) as `sigma_inhom → 0` (`elf → γ_hom²`, `egf → 0`).

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

$$n_{\rm theory} = \frac{4\hat\sigma_{\rm bg}^2\, \hat\Omega\, W}{\pi \hat{c}^2 T^2}$$

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

**Saturation-Voigt models** — derived-quantity gating: the raw parameters
(`saturation`, `sigma_inhom`, `c_max`) are **not** checked individually (their relative
thresholds are physically inconsistent along the saturation axis since
$\Omega \propto \sqrt{1+s}$). Instead their uncertainties are propagated through the
local Jacobian onto two derived quantities that carry Lorentzian-equivalent semantics
(`saturation_voigt_derived_sigmas` in `sequential_bayesian_locator.py`):

$$\Omega(s, \sigma_{\rm inhom}) = \gamma_0\sqrt{1+s} + \sqrt{2\ln 2}\,\sigma_{\rm inhom},
\qquad
\sigma_\Omega = \sqrt{\left(\tfrac{\gamma_0}{2\sqrt{1+s}}\,\sigma_s\right)^2 + \left(\sqrt{2\ln 2}\,\sigma_{\sigma_{\rm inhom}}\right)^2}$$

$$C(s, c_{\max}) = c_{\max}\tfrac{s}{1+s},
\qquad
\sigma_C = \sqrt{\left(\tfrac{s}{1+s}\,\sigma_{c_{\max}}\right)^2 + \left(\tfrac{c_{\max}}{(1+s)^2}\,\sigma_s\right)^2}$$

$\sigma_\Omega$ is gated with **linewidth** semantics (absolute
`NVISION_LINEWIDTH_CONVERGENCE_THRESHOLD` if set, else relative to the effective-HWHM
range implied by the raw bounds); $\sigma_C$ with **c_total** semantics (relative to the
realized-contrast range). The SBED CRLB early-stop propagates per-parameter CRLBs
through the same Jacobians. **History note:** `all_converged_step` values recorded for
saturation-Voigt runs *before* this change used raw 1%-of-bound-width gating and are not
comparable.

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

---

## 7. Zeeman + Hyperfine Forward Model (`nv_center.py`, `voigt_zeeman.py`)

### 7.1 Physical Origin of Each Parameter

| Parameter | Physical origin |
|---|---|
| `frequency` | Zero-field splitting D ≈ 2.87 GHz between the ms=0 and ms=±1 levels |
| `zeeman_split` | External B-field along the NV axis (γ_NV ≈ 28 MHz/mT) splits ms=+1 from ms=−1 |
| `split` | ¹⁴N (nuclear spin I=1) hyperfine coupling, A∥ ≈ 2.16 MHz (`NV_N14_HYPERFINE_SPLIT_HZ`, `nv_center.py:49`) — splits each Zeeman line into a triplet (mI = −1, 0, +1). A ¹⁵N (I=½) sample would give a doublet instead. |
| `k_np` | Nuclear-spin-polarization asymmetry: population/depth ratio between the mI=−1 and mI=+1 hyperfine lines (`k_np=1` ⇒ unpolarized) |
| `fwhm_total`, `lorentz_frac` (or `saturation`, `sigma_inhom`) | Combined homogeneous + inhomogeneous linewidth — see §7.2 |
| `c_total` / `c_max` | ODMR contrast, set by microwave/optical saturation |

Both Zeeman groups share the same `(split, k_np)` — the spectrum is exactly symmetric about `frequency` by construction (`_zeeman_pv_pred`, `numba_kernels.py:1295`, uses one `(p_l, p_0, p_r)` population triple for both groups).

### 7.2 Homogeneous vs. Inhomogeneous Broadening

Modeled as a pseudo-Voigt — a height-normalized weighted *sum*, not the true Lorentzian⊛Gaussian convolution (see the model docstrings and `tests/spectra/test_pseudo_voigt_accuracy.py`):

$$V(dx) = \frac{\eta \cdot \dfrac{\gamma}{dx^2+\gamma^2} \;+\; (1-\eta)\cdot G_{\rm peak}\, e^{-dx^2/2\sigma^2}}{\text{center height}}$$

with $\gamma = \text{fwhm}_l/2$, $\sigma = \text{fwhm}_g/(2\sqrt{2\ln 2})$, $\text{fwhm}_l = \texttt{lorentz\_frac}\cdot\texttt{fwhm\_total}$, $\text{fwhm}_g = (1-\texttt{lorentz\_frac})\cdot\texttt{fwhm\_total}$, and η the Thompson-Cox-Hastings mixing weight (`_pv_factors`, `numba_kernels.py:1234`).

**Physical origin.** Homogeneous (Lorentzian) broadening is the Fourier transform of exponential-in-time decay (T2 dephasing, microwave power broadening). Inhomogeneous (Gaussian) broadening is an ensemble average over many small, independent static offsets (strain, field inhomogeneity, unresolved ¹³C hyperfine coupling) — Gaussian by the central limit theorem. In open-quantum-system terms: the ¹³C nuclear spin bath is the environment, and whether its contribution to the lineshape is Lorentzian or Gaussian depends on its correlation time τc relative to the measurement time T — τc ≪ T gives motional narrowing (Lorentzian), τc ≫ T gives a quasi-static, shot-to-shot-varying distribution (Gaussian). The saturation-coupled model's `NV_NATURAL_HWHM_HZ` (`nv_center.py:64`) seeds the homogeneous `gamma_hom` branch specifically; `sigma_inhom` is the independent Gaussian branch (`nv_center.py:688-706`).

### 7.3 The `k_np` / `split` Identifiability Limit

When the hyperfine triplet is unresolved (`fwhm_total` comparable to or larger than `split`), `k_np` becomes the near-null direction of the local sensitivity (finite-difference Jacobian of the Zeeman pseudo-Voigt kernel w.r.t. `(fwhm_total, split, k_np)`, fixed `zeeman_split`) — its effect on the observable curve vanishes exactly as `split → 0`:

| `split` | smallest singular value | condition number |
|---|---|---|
| 0.001 | 0.001 | 71 000 |
| 0.010 | 0.078 | 980 |
| 0.030 | 0.277 | 300 |
| 0.090 | 0.372 | 193 |

The weak singular vector is ≈ pure `k_np`; `split` and `fwhm_total` stay comparatively well-conditioned against each other throughout. This is a real, not merely practical, degeneracy: once the hyperfine structure is unresolved there is no information left about *how* population is split among the three sublevels — only its aggregate effect on width/depth survives.

### 7.4 Reduced Model (Zeeman-Only, Hyperfine Unresolved)

Per §7.3, `split`/`k_np` should be dropped as free parameters — not inferred, and not zeroed out either, since the hyperfine structure is still physically present, just unidentifiable — whenever the triplet isn't resolved. This reduced parameterization already exists in production:

- **Signal models**: `NVCenterLorentzianModel(with_zeeman_splitting=True, with_hyperfine_splitting=False)` → `NVCenterLorentzianZeemanSpectrum(frequency, linewidth, zeeman_split, c_total)`; `NVCenterSaturationVoigtModel(with_zeeman_splitting=True, with_hyperfine_splitting=False)` → `NVCenterSaturationVoigtZeemanSpectrum(frequency, saturation, sigma_inhom, zeeman_split, c_max)`.
- **Belief builder**: `with_zeeman_splitting=True, with_hyperfine_splitting=False` are the *defaults* of `nv_center_smc_belief()` (`belief_builders.py:262-263`).
- **Generator**: `NVCenterCoreGenerator` defaults the same two flags the same way (`nv_center_generator.py:55-56`).
- **Task wiring**: `combinations.py` switches `lineshape="saturation_voigt"` automatically when the generator name starts with `NVCenter-saturation_voigt` (`combinations.py:136-137`); the Lorentzian reduced path needs no override since it is already the belief builder's default.

So "only `zeeman_split` and the linewidth are visible" is the out-of-the-box configuration, not an opt-in — the opt-in is `with_hyperfine_splitting=True`, which should only be reached when §7.3's resolvability condition actually holds (e.g. gated on the current linewidth posterior, per the discussion this section is drawn from).
