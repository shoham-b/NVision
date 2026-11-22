# Evaluation metrics: what to measure, per stage

Read this before adding or changing any locator-comparison metric or results view. It records what the comparison
is for, which quantity drives each stage, and the rules that keep a metric meaningful. The stage detector itself
(search → localize → refine, the Fisher-efficiency rule) is defined in [locator_comparison.md](locator_comparison.md).

Naming: the article calls the middle stage *localize*; so do the UI and these docs. Stored column names keep the historical `focus_found_step` (and the locator's focus window keeps the name "focus", as in the article).

Terminology: **claimed uncertainty** σ̂_N is the standard deviation the locator itself reports after N measurements (the belief's posterior spread); it is compared against the true error and is not an accuracy measure on its own. Prior information J₀ is 4π²/W² for a global search over a uniform window of width W (article Eq. j0), 1/σ₀² for a Gaussian tracking prior; the plain 1/Var form is kept separately as `stage_prior_info_var`.

## What the comparison is for

**Bayesian SBED is the subject.** The other locators are a yardstick that shows what SBED gains. Scenario: the
line shape was profiled before, the magnetic field moved, and the new resonance must be found.

The yardstick is a ladder, each rung tuned before the next:

| Rung | Locator | Tuned / differs by |
|---|---|---|
| 0 | `SimpleSweep-optimal` (reference), `SimpleSweep-optimal-x0.5`, `-x2` | grid step (k points per **prior** FWHM): $m\,k^*$ with $m = 1, 0.5, 2$ and k* the analytically optimal fractional step of the task (cost $\min_k \overline{P(k,\varphi)}\,n(k)$, see [locator comparison](locator_comparison.md)); passes averaged over the whole budget |
| 1 | `Bayesian-SBED` | no coarse grid |
| 2 | `Bayesian-SBED-Drive` | also picks the MW drive amplitude (strong to search, near the slope optimum to refine) |

Expected story: the search stage is a tie (blind, cost of order n for everyone); SBED's distinctive win over
the sweeps is in the localize stage; in refine SBED measures at the Fisher-optimal points, while a
plain sweep keeps spreading its measurements over the whole dip.

## What drives each stage

Choose the axis from the stage's physics, not from whatever grid columns the results rows happen to have.

| Stage | What drives it | Axis | Metric |
|---|---|---|---|
| **Search** | whether the grid puts enough points inside the true dip, and how deep each point sees it | **grid points per true FWHM** = k × FWHM_true / FWHM_prior, with SNR = C / noise σ | measurements until a dip is seen (better: in grid **passes**), dip-found rate |
| **Localize** | going from "dip seen" to "located near the steepest slope" (within ε, where the sensitivity is still ≥ 90% of its best) | information per measurement ∝ SNR², so cost ∝ 1/SNR²; expected independent of the window (a hypothesis, checked in the Scaling view) | measurements in the localize stage, efficiency = Fisher bound / steps |
| **Refine** | Fisher-limited precision | measurements an oracle needs for σ = τ, which scale like (FWHM_true / (SNR · τ))² | Fisher efficiency η (`stage3_sens_eff`), efficiency = Fisher bound / steps (`stage3_efficiency`) |

`sigma_inhom` and the hyperfine isotope matter only through the dip's effective width: plot against the effective
FWHM, not against those columns.

**k is relative to the prior FWHM, not the true one.** The sweep spacing is FWHM_prior / k. When the prior is wide
(e.g. prior HWHM ≈ 2.5 MHz against a true 0.5 MHz) a "k = 1" grid puts ~0.2 points per true FWHM and can step over the
dip. k = 1 is therefore not studied (the optimal-step sweep scans k from 0.5, but its cost model prices such grids
out); compare spacings by points per true FWHM. The realised k* of every `SimpleSweep-optimal` repeat is recorded
(`sweep_k_opt`; `sweep_k` = m k* is the step actually swept; with the predicted passes `sweep_passes_pred` and cost `sweep_cost_pred`).

## The Fisher oracle behind the stage metrics

A stage is judged against the fewest measurements any locator could need, computed from the *true* signal
(`nvision/metrics/stages.py`, simulation only). The signal is the photoluminescence (PL) as a function of the drive
frequency x. Notation: Γ = FWHM of the dip (Γ = 2 HWHM), C = its depth, σ = noise standard deviation of **one** shot
(dimensionless, like the PL), SNR = C/σ, k = number of shots averaged into one measurement (`obs.n_shots`, `k_t` in
`stages.py`). One measurement of the primary parameter θ (the dip position, e.g. `zeeman_shift`) at drive frequency x
carries Fisher information

$$I(x) = \frac{k\,g(x)^2}{\sigma^2}, \qquad g(x) = \frac{\partial\,\mathrm{PL}(x;\theta)}{\partial\theta}\ [\mathrm{Hz}^{-1}].$$

Averaging k shots cuts the variance to σ²/k, so the information is multiplied by k. I has units Hz⁻² per measurement
(Fisher information is not unitless). Independent measurements add their information, so the Cramér-Rao bound after N of
them is σ_θ = 1/√(N I). The best point is the steepest slope of the dip: for a Lorentzian, |g\*| ≈ 1.3 C/Γ, so
I\* = k max g² / σ² ≈ 1.7 k SNR²/Γ². Hence the steps needed for a precision ε are N ≈ 1/(ε² I\*): **they fall as
1/SNR²**. `stage_fisher_star` stores I\* (Hz⁻² per step; shots enter as the factor k).

* **Localize ends at ε**, the half-width of the region around the best point where g² ≥ 0.9 max g² (η ≥ 1 − δ with
  δ = 0.1). For a Lorentzian η ≈ 1 − 18 (ε/Γ)², so ε = √(0.1/18) Γ ≈ 0.075 Γ; the code finds it numerically from the
  true curve (`stage_eps_target_phys`), so it holds for any line shape. Oracle: `stage2_oracle_steps` = (ε⁻² − J₀)/I\*,
  about 100/SNR² for a Lorentzian when J₀ is small (the constant depends on the SNR and width definitions; the exponent is
  the point). J₀ is the prior, see below.
  Not defined for drive-controlled models (the best point is a drive setting; pulse fringes can be narrower than the grid).
* **Refine starts when the claimed uncertainty σ̂ has stayed ≤ ε** for a few steps (`refine_start_step`; ε taken on the line as the run
  estimated it, so the boundary needs no true parameter and applies to an experiment). **It ends at the stop target τ**
  (`stage_tau_phys`). Oracle: `stage3_oracle_steps` = (τ⁻² − ε⁻²)/I\*. It scales like (FWHM_true / (SNR τ))²: the Scaling
  view's refine panel axis.
* **Search** has no Fisher bound (nothing is known about where the dip is); its scale is the number of dip-widths to cover,
  window / FWHM_true (`stage_window_fwhm`), times the passes needed to see the dip at the given SNR.
* **Efficiency** = oracle steps / steps spent (`stage2_efficiency`, `stage3_efficiency`): 1 = Fisher-limited. The oracle is a
  lower bound that assumes every measurement is at the best point and the other parameters are known, so it is not a target.
  `stage{2,3}_wasted_share` is the share of the stage's measurements with η < 1 − δ.
* The Fisher curve in the Stages view's claimed-uncertainty plots, (σ₀⁻² + j I\*)^(-1/2) after j in-stage measurements,
  starts from the stage's own claimed uncertainty σ̂ and is the same bound; below it the belief claims more precision than the data carry.

Per-run axes stored with them: `stage_fwhm_true_phys` (2 × `effective_hwhm` of the true signal; none for drive-controlled
models), `stage_prior_fwhm_phys` (mean prior FWHM), `stage_window_phys`, `stage_window_fwhm`.

**What the first check showed** (2026-10-27, read-only, on the stored pure-Lorentzian SNR × window rows, written before
`NVISION_DIP_P_FA` went to 0.1 and before measurements are confined to the focus window; re-check on fresh runs):
SBED's localize stage took 25-150× the bound and flattened at roughly 20-40 steps at high SNR instead of following 1/SNR²;
the claimed uncertainty σ̂ at localize start was already 0.05-0.25 Γ (not ≈ the FWHM) while the true error was 2-4× larger (overconfidence);
localize steps grew about 2× from window 5 to 10 × FWHM; SBED search steps were far flatter in the window than proportional.
With the localize stage's mean η only 0.3-0.5, the stage length looks set by exploration the tail rule counts, not by information.
(These were measured with the old boundaries: search→localize at a 0.5 false-alarm dip scan, refine = first step whose whole
remaining tail had mean η ≥ 0.9. In those rows the true error was already ≤ ε at "dip found" in about a third of SBED runs,
and refine was reached in about 15% of runs. The boundaries above replace both; the old `max_info_step` column is now
`refine_start_step`, so older rows show a dash and need a re-run.)

## The prior is information

A search window is a prior: a uniform window of width W says "the dip is inside, nowhere in particular", so it already
holds information before any shot. The Cramér-Rao line σ = 1/√(N I\*) is therefore not the reference for a Bayesian
posterior; the next sections give the prior, the stages it separates, and the bound that includes it.

### Information in the window: why σ₀² = W²/12

A uniform prior p = 1/W on [x₀ − W/2, x₀ + W/2] has variance

$$\sigma_0^2 = \int_{-W/2}^{W/2} \frac{u^2}{W}\,du = \frac{1}{W}\cdot\frac{2\,(W/2)^3}{3} = \frac{W^2}{12},
\qquad \sigma_0 = \frac{W}{\sqrt{12}} \approx 0.29\,W .$$

It is W² (Hz²), not W. The prior information scale is J₀ = 12/W² (Hz⁻²). A uniform prior has no finite Fisher
information at its edges, so the inverse variance is used as the convention: after N measurements 1/σ² = 12/W² + Σ I_j.
Fisher information is not unitless (Hz⁻²); the unitless quantity is the gain log₂(σ₀/σ) = log₂(W/(√12 σ)) bits.
Two prior informations are stored (`belief_prior_info_phys` in `nvision/metrics/stages.py`). `stage_prior_info` is the J₀
of the bound: for a global search (uniform prior, simulated prior unchanged) J₀ = 4π²/W² (Gill and Levit), a strict bound
on the worst-placed resonance, W being the width of the primary parameter's own prior range (`zeeman_shift`), not the
drive window; for a Gaussian tracking prior J₀ = 1/σ₀². `stage_prior_info_var` is always 1/Var(prior)
(`belief_prior_variance_phys`; 12/W² when uniform) and is the one for models of actual error (sweep, elimination). Neither
is read from any measurement: the claimed uncertainty σ̂ after the first measurement (used
before) mixed one measurement and the particle filter's own calibration, which is overconfident, into the prior. A prior
whose variance cannot be computed exactly raises instead of falling back.

* Window widths are not comparable without J₀: a narrower window starts lower for free. A focus window built from the
  search posterior was paid for with shots; a window placed with the truth (e.g. the Zeeman edge pad) is information the
  locator did not earn.
* The localize oracle is `(ε⁻² − J₀)/I*`, not "from scratch". Refine already started from ε⁻². Together:
  `(τ⁻² − J₀)/I*` for the whole run.

### When the three stages are distinct

The stages are regimes of one continuous accumulation of information; the boundaries are thresholds (δ = 0.1, the dip
false-alarm rate, the stop target τ). Whether all three show depends on the run:

* **Search** is visible only for large W/Γ: its plateau lasts about W/Γ points. If σ₀ ≲ Γ, i.e. W ≲ 3.5 Γ, there is no
  plateau, only localize and refine.
* **Localize** needs a gap between "dip seen" and ε = 0.075 Γ. At low SNR search and localize merge; at very high SNR localize is a
  few steps.
* **Refine** needs τ well below ε: its length is (τ⁻² − ε⁻²)/I\*. With τ ≈ ε the run stops on the cliff and has no −½
  tail. It also needs the locally linear regime (within ≈ 0.075 Γ of the best point).

Localize and refine are expected to be independent of W; only search grows with it. So a large window shows three stages and a
small one two.

### The CRLB with the real prior

$$\sigma_{\mathrm{CRLB}}(N) = \Big(\frac{12}{W^2} + N\,I^*\Big)^{-1/2}.$$

It starts at W/√12, bends where N I\* ≈ 12/W², then has slope −½ (van Trees form). A claimed uncertainty σ̂ below the plain
1/√(N I\*) at small N is expected; only a true error (averaged over repeats) below this curve would indicate a defect.
The Stages view's Whole run chart draws it from the run's own start (J₀ from the claimed uncertainty σ̂ after the first measurement,
every measurement counted); the localize and refine charts restart from each stage's own claimed uncertainty σ̂. The Information tab starts it from the primary parameter's own prior instead (see below): W is the drive window, not that prior, and uses J₀ = 4π²/W² for a global search.

The stages are regimes of one continuous accumulation of information, J = 1/σ² from J₀. The Whole run scope therefore
also shows two stage-free charts (`renderHlInfoCurve`), both from the claimed uncertainty σ̂: the **prior share** J₀/J (1 → 0, dashed =
J₀/(J₀ + N I\*)) and the **information rate** (J(N) − J(N′))/((N − N′) I\*), where 1 is every measurement at the
Fisher-optimal point and a value above 1 means the claimed uncertainty σ̂ fell faster than any measurements could justify
(overconfidence). Both are computed from the claimed uncertainty σ̂, so they inherit its overconfidence; they say nothing about whether
the dip was found, and search (detection) is not a Fisher regime, so the first steps need care. The stage boundaries and
columns remain for the per-stage views.

A fit of three straight log-log segments to the claimed uncertainty σ̂ is not used: the bound above is the reference. On the
2026-11-02 check (3 use cases, SNR 5, 50 repeats each), the existing `refine_start_step` sat on the cliff, not on a −1/2
tail, in many runs of the (since removed) Smart locators.

### Sensitivity and bandwidth

η = δB √T is a constant only when σ ∝ T^(−1/2) exactly (refine). With the gyromagnetic ratio γ_e ≈ 28 GHz/T,
θ = γ_e B along the NV axis, so σ_B = σ_θ/γ_e. The operational quantity is the first passage time of the true error to a
target, `stage_true_passage_step` = T(τ); the bandwidth is 1/T and `stage_eta_eff` = τ √T (Hz √steps) is the effective
sensitivity *at that target*. It tends to the refine sensitivity as τ → 0 and is not a constant of the locator. Per-step
time is not in the data yet, so T is in steps.

Stored only for runs recorded after these columns were added; re-run to get them (the cache has no belief snapshots).

## Information against the bound (the Information tab)

The stage boundaries are not used to judge a locator any more. The claim is the single bound σ(n) ≥ (J₀ + n·I\*)^(-1/2),
and every locator is judged by its distance to it. All of it is computed in the browser from the per-step `series.e` /
`series.u` and the stored `stage_fisher_star` and `stage_prior_info`. `stage_prior_info` is written at run time, so
rows recorded before this change carry J₀ = 12/W² (and no `stage_prior_info_var`). `nv metrics recalc` fixes them
without a re-run (`rederive_prior_info_fields`): for a uniform `zeeman_shift` prior J₀ = (π²/3) · stored, the old value
moves to `stage_prior_info_var`, and `stage2_oracle_steps` / `stage2_efficiency` are recomputed; rows already current
are skipped and any other primary parameter raises. `nv render` alone only re-reads what is stored. Rows from before
J₀ came from the prior carry the claimed-σ value and cannot be rescaled: re-run those.

* **J₀ is the prior of the primary parameter, not the drive window.** `stage_window_phys` is the swept drive-frequency
  range; the primary parameter (e.g. `zeeman_shift`) has its own prior, and a window-based J₀ of the drive range
  put the "bound" far too low, so locators appeared to beat it. J₀ = `stage_prior_info` = 4π²/W² for the uniform
  global-search prior of width W on `zeeman_shift` (1/σ₀² for a Gaussian tracking prior), computed from the locator's
  prior itself, not from a claimed uncertainty σ̂. Every run of a use case starts from the same prior, so they all store the same J₀;
  the browser still takes the median over the use case's runs, so all locators share one ceiling.
* **Information vs the bound**: per repeat and step, z = error² · (J₀ + n·I\*); R(n) = 1 / mean(z) over the repeats still
  running (a run that has stopped keeps its final error; a step needs at least half of a use case's repeats), and the median
  over use cases when several are selected. R = 1 is the bound, below it measurements are wasted. The mean square is the
  quantity the bound is on, so a few catastrophic runs dominate it on purpose (rule 8). Because the global-search J₀ is a
  worst-case bound, R of the true error is **not** expected to sit near 1 early on: the uniform average has up to 1.8× the
  bound's variance while n·I\* ≲ J₀ (R ≈ 0.55 there), so a low R at small n is looseness of the bound, not wasted
  measurements. R should stay ≤ 1 up to about 1 % (the uniform average can undercut the bound that much); a clear excess
  above that is a defect in J₀ or I\*. Read the gap between locators, and the approach to 1 once n·I\* ≫ J₀. The same quantity from the claimed uncertainty σ̂ (dotted) above the true one is overconfidence.
  A median-based "typical run" version was tried and dropped: median error² is not tied to the variance for non-Gaussian
  errors (it read up to 1.65), so it is no bound.
* **Cost to reach a precision**: T(τ) = the first measurement from which the true error stays ≤ τ = σ₀ / gain (gains
  1.5 … 80), median over the runs that got there, shown only where at least half did (a locator's curve ends where most
  of its runs stop short; the share is in the hover). The bound's cost is (τ⁻² − J₀)/I\* = (gain² − 1) J₀/I\*. Targets are
  in units of σ₀, not of the FWHM, because a tight prior can already beat 0.1 FWHM.
* **Lag n₀ and efficiency e**: the line T = n₀ + bound/e through a use case's rungs (at least three), summarised by the
  median over use cases. n₀ is the measurements spent beyond the bound (search plus localize, with no thresholds), e the share
  of the best information rate afterwards. The ladder is a median-run statistic against a mean-square bound, so e a little
  above 1 can occur, and a fit through three or four rungs is noisy; read n₀ and e together with the R chart.
* **Search stays** as the one real stage (detecting the dip is not a Fisher regime). The stage columns in `stages.py`
  and the Scaling view are untouched.

## Search against the bounds (Information tab, Search scope)

During the search no strategy, adaptive or not, beats the search bounds of the article (Sec. global and the SM prior
section). The Search scope draws them next to the measured search, per use case (rule 7) and for the search locators only
(`Bayesian-SBED` and the `SimpleSweep-optimal` family). Everything is computed by
`nvision/metrics/search_bounds.py` (formulas, unit tests in `tests/test_search_bounds.py`) from data already in the cache
through `GET /api/search-bounds` (`nvision/cli/search_bounds_api.py`), so `nv serve` on an existing cache is enough and no
re-run is needed. The panels are drawn by `static/search-bounds-view.js`.

* **W is the width the generator draws `zeeman_shift` from** (`zeeman_min_phys` .. `zeeman_max_phys`, read from the run's
  `true_params.bounds` and checked against the generator), not the drive window and not a van Trees prior. The bounds average over
  the dip uniform in this range, so `theta` varying per repeat is correct. A use case whose line shape differs between
  repeats, whose primary parameter is not `zeeman_shift`, or whose W is not wider than 2t is reported as "not judged" with the reason.
* **kappa_bar** = k ∫dip² dx / (2σ²W), the mean KL rate per measurement. The integral is taken on the true line (rebuilt from the
  generator, averaged over 33 values of `zeeman_shift` across the drawn range, on the drive window), so it holds for Voigt and
  hyperfine multiplets; the Lorentzian closed form πγk·SNR²/(4W) is shown for reference. k = 1 (the simulator never sets
  `n_shots`) and is cross-checked per run against the stored `stage_fisher_star` (k = I*σ²/max g² must be 1 within 5 %).
  SNR = line contrast / σ with the line contrast of `sim.presets.line_contrast`; the run's own SNR label can be another convention
  in older rows (`snr_label`, reported, not used).
* **Bound curves vs n**: finite reach `(W − nΔd)³/(12W)` (eq. search-mean), change of measure
  `W²/(8e)·e^{−nκ̄}/(1+ln2+nκ̄)²` (eq. search-bound) and the elimination model eq. mean-curve (`I_max` = the stored
  `stage_fisher_star`). Δd comes from eq. sm-reach (a fixed point, z = √(2 ln(W/Δd)) + `Z_BETA`); with no single-measurement
  reach (√k·SNR ≤ z) there is no Δd and the finite-reach and model curves are omitted, with Δd = W when one measurement covers the window.
  Where √k·SNR/z ≤ 4/3 the best probe averages and Δd = Γ/√3 (eq. sm-fixed-cost). Curves are drawn in units of the prior variance W²/12.
* **Mean squared true error vs n**: the mean over all runs of (true |estimate − truth|)²; a stopped run keeps its final error, a step
  is drawn only while at least half of the runs still run. The stored series is downsampled (80 points), so a run's error at n is
  that of its last kept step ≤ n.
* **Survival and κ_eff**: S(n) = share of runs with |error| > t, t = `SURVIVAL_TOLERANCE_HWHM` (one HWHM). κ_eff = −d ln S/dn by
  least squares over the grid points with 0.05 ≤ S ≤ 0.9. The bound is S ≥ exp(−(nκ̄+ln2)/(1−2t/W)) (drawn), so κ_eff/κ̄ may exceed 1
  without violating it: it is compared with 1/(1−2t/W). The predicted sweep value 1/(2.4 ln(W/Δd)) is shown for the sweeps.
* **Detection**: `dip_detected_step` (median, p75, share that never detected, counted at the budget so these are lower bounds) in
  units of 1/κ̄ and of W/Δd (at high SNR, 1/κ̄ < W/Δd, compare with W/(2Δd)).

* **Lab data.** The functions in `search_bounds.py` take a reference centre (`abs_errors`), widths, `k` and SNR or σ as plain arguments and
  never read the true signal, so the single-run outputs (bound curves, detection step against 1/κ̄ and W/Δd) apply to one lab run; the
  ensemble outputs (mean err², S(n), κ_eff) need many θ. The lab could emulate θ uniform in the window by replaying recorded sweeps with a
  randomly shifted window (not implemented, no lab wiring yet).
* **Old SNR labels.** Rows recorded before the `snr-per-line-contrast-v1` fingerprint change (cached `368f4f5d4fdc`; `018d00aed340` and later use the article SNR)
  have σ = contrast_total/x for `SNR(x)`, i.e. article SNR x/2. The panel always shows the article SNR computed from σ and states when the run label differs.

Reading it on the current cache (2026-11-15): the SNR × window study's drawn W is only 1.5–4.4 Γ, so the search is a few
probe cells and the bounds are far below the measured search (κ_eff/κ̄ ≈ 0.01–0.09 for all four locators, none near 1/(1−2t/W));
a real test of the search claim needs the larger windows. Rows of the fixed-step sweeps (`SimpleSweep-<k>perFWHM`, only in older caches) carry a stored J₀ that is not 4π²/W²
(older prior), flagged in the panel; nothing here reads it.
## Testing the article's claims (`nvision/metrics/claims.py`)

Code: `claims.py` (pure maths), `claims_report.py` (one locator in one use case), `GET /api/claims?generator=..&noise=..`
(`nvision/cli/claims_api.py`). Everything is computed from stored data (the per-step series `s`, `e`, `u`, and the stored
scalars), so `nv serve` on an existing cache is enough: **no re-run, no `nv render`, no physics fingerprint or cache schema
change.** Each number is a measurement that can fail its claim; a window or a use case that cannot support a number returns
`None` and says why. The `(n0, q)` line, the sweep theory, the search bounds, localize and the posterior series are the
existing sections of this page; this section adds only what they did not make:

| Claim (article) | Metric | Fails when |
|---|---|---|
| Gap before any measurement $\sqrt{J_0/J_{0,\mathrm{var}}} = \pi/\sqrt3$ | `prior_gap`, `truth_spread_ratio` (rms of the drawn truths about the window centre over $W/\sqrt{12}$) | the stored columns disagree, or the truths are not uniform over the prior (ratio far from 1) |
| van Trees: run-averaged true error never below $(J_0 + nI_{\max})^{-1/2}$ | `bound_ratio` $R(n)$, bootstrap interval, `bound_violated` | the lower end of $R$'s interval is above 1.01 anywhere (a defect in $J_0$ or $I_{\max}$) |
| Four regimes: search + localize, collapse, converge, refine | `regimes`: median detection step, median Gaussian onset, converge end (first $n$ with rms error $\le 1.1/\sqrt{q n I_{\max}}$, $q$ from the true-error fit), plateau level over $\sigma_0$, tail slope on $n \ge 2 n_{\rm det}$ | no detection, no Gaussian peak, no $(n_0,q)$ fit, or the error never gets within 10 % of the data limit (`why`); the slope is steeper than $-1/2$ by $n/(n-n_0)$, so read it with $n_0$ |
| $1/\mathrm{MSE}$ linear in $n$ late in the run, slope $q \le 1$ | `fit_with_gate`: `refine_rates.fit_rate_line` on `late_window`, bootstrap intervals over repeats, and a **linearity gate** (the $q$ of the two halves of the window must have overlapping intervals) | `linear` is false: the window is not the late regime and $q$ is then not a refine rate (it is still returned, flagged) |
| Information rate above 1 means overconfident | `rate_ratio` = $q_{\rm claimed}/q_{\rm true}$ over the same window | the ratio is near 1 (the claim would be contradicted); above 1 confirms it |
| Coverage of $\hat\sigma_N$: 68 % / 95 %, tested only once the posterior is Gaussian | `coverage_curve` per $n$, runs counted from their Gaussian onset (`within_k`, share of runs `gaussian_share`) and all runs (`within_k_all`), Wilson intervals; `coverage_from_z` from the signed series `z` (also the mean and std of $z$); `coverage_from_pit` from the series `p` (quantile coverage, below) | the interval excludes 0.683 / 0.954 |
| Sensitivity $\eta/\sqrt q$ | `sensitivity_ratio` = $\sqrt{I_{\max}\,\mathrm{MSE}\,n}$ on the window | by construction it equals $1/\sqrt q \cdot \sqrt{n/(n-n_0)}$, so it only cross-checks the fit |

Every run measures the whole budget, so a run's last stored value is held after its last stored step (the rule of
`refine_rates.inverse_mse_curve`) and all repeats are present at every $n$. Series have at most 80 stored points per run and
are interpolated linearly. A sweep's `e` / `u` come from the SMC belief it carries (its own fit is not stored).

**Quantile coverage (PIT).** The cache keeps no particles, so the posterior CDF at the truth, $p=\sum_i w_i\,[\theta_i\le\theta_{\rm true}]$, is a run-time column (`nvision/metrics/posterior.py`): the per-step series `p` (every step, downsampled like `z`) and the per-run `posterior_pit_final` (`p` at the last step; both in `_COMPARE_FIELDS`). A calibrated posterior gives $p\sim U(0,1)$ whatever its shape. `claims.coverage_from_pit` takes the $p$ of all runs at one step and returns the share inside the central 68 % ($0.16\le p\le0.84$) and 95 % ($0.025\le p\le0.975$) intervals with Wilson intervals, plus the mean and std of $p$ (1/2 and $1/\sqrt{12}$ when calibrated). Unlike `coverage_from_z` it needs no Gaussian peak. Runs recorded before the column existed (and lab/MATLAB runs) have none; a re-run fills it.

## Refine rates and the speedup over a sweep

Code: `nvision/metrics/refine_rates.py` (pure maths), `nvision/metrics/refine_rates_runs.py` (rebuilding cached runs),
`GET /api/refine-rates` (`nvision/cli/refine_rates_api.py`), UI `static/refine-rates.js` (the "Refine & speedup" section
at the bottom of the Information tab). Theory: `articles/sections/theory.tex`, Sec. "Speedup over a drive-frequency
sweep". Locators in scope: Bayesian-SBED and the `SimpleSweep-optimal` family (the realised k is read from `sweep_k`). It reads the cache only, so it works
after `nv render` / `nv serve` with no re-run (it needs the live server, not a static export).

After the dip is found a locator gains information about the primary parameter at `q · I_max` per measurement,
`1 / mean(err²)(n) = q · I_max · (n − n₀)`, with `I_max` = `stage_fisher_star` (per measurement, every locator takes one
shot per measurement: no strategy sets `n_shots`, so measurements are shots). Two different `q`:

* **`q_place`** = the mean oracle efficiency η_t of the measurements after `dip_detected_step` (per run, then median and mean
  over runs; runs with no dip, or ending at the detection, are left out and counted). Exact: it is where the locator
  measured, judged with the true signal. Rebuilt from the cache (below).
* **`q_err`** = slope of `1 / mean(err²)` in units of `I_max`, the information the *estimate* gained; `n₀` its intercept.
  `q_err / q_place` is what the particle filter keeps of the information the placement offered (1 = nothing lost). The
  fit is the least-squares line over a window `[max(median dip step, n_hi / 2), n_hi]`, `n_hi` the largest `n` where at
  least half the repeats still run (stopped runs keep their final error); `R²` is reported with it and a poor `R²` means a
  constant rate does not describe that locator there. The error is interpolated linearly between the ~80 stored
  samples of a run (a step hold would lag by up to several percent of `n`, which biases `n₀`).

**Every run measures its whole budget.** The convergence rule is recorded but never stops a run (since the
`stratified-zeeman-v1` fingerprint), so every repeat measures the shared budget (`NVISION_MAX_MEASUREMENTS`; a sweep a
whole number of passes of it) and has an error at every `n`: the fit window is common to every run. Every repeat also
stores where it measured (`meas_drive_freq_unit`, float32 unit coordinates of the drive window), so `q_place` uses all
repeats, not only the first `NVISION_GRAPH_REPEATS` that keep a scan figure. Rows recorded before that change stopped early
and stored positions for only the first repeats; they are stale under the current fingerprint.

```
uv run nv run --run-group all --repeats 50
```

**How `η` is rebuilt for a cached run** (`refine_rates_runs.build_run_trace`). The cache has no belief snapshots, so:
the true signal is regenerated with the executor's own seed (`generator.generate(Random(repeat_seed_int(signal_repeat_key(
seed, generator, repeat))))`) and compared with the stored `true_params` (a mismatch raises: the generator changed since the
run); SBED's positions come from `meas_drive_freq_unit` or the scan figure (`measurements.x`, float16 rescaled into the window,
~1e-3 of it); a sweep's from its design (its scan figure is never stored): `linspace(lo, hi, n)` repeated pass after pass with
`n = ceil(W / (FWHM_prior / k)) + 1`, `FWHM_prior` from the generator's own prior means. `η_t = g²(x_t) / max g²` with
`g = ∂signal/∂primary` (`stages.primary_sensitivity_at`, the function `compute_stage_metrics` uses). End-to-end check: the mean
`η` over each stage's measurements must reproduce the stored `stage{1,2,3}_sens_eff` to 0.03 (sweeps match to rounding,
SBED to about 1 % from the float16 positions), otherwise the repeat is left out and listed with the reason. Repeats that cannot
be rebuilt are never given a fallback.

`q_place` averages η over everything after the detection (localize included), while the line is fitted late, so the
filter-loss ratio uses `q_place_window`, the median over runs of the mean η over the fitted window `[n_lo, n_hi]`
(`q_err_over_place_window`); `q_err / q_place` is kept but mixes the two ranges.

**Theory references, per use case.** `q_sweep = I_sweep / I_max`: the Lorentzian closed form `8π/27 · Γ/W` and the exact
`mean(g²) / max(g²)` over the drive window of the rebuilt true line (any shape: Voigt, a hyperfine triplet, where the
Lorentzian form is not valid and `Γ` = `stage_fwhm_true_phys` is the single-line FWHM, not the triplet's extent); likewise
`R∞ = 27/(8π) · W/Γ` and `max / mean`. Both are drawn and labelled. The steepest point of the exact sensitivity, in units of the
true HWHM `γ` from the nearest line centre, is marked on the histogram next to the Lorentzian `±1/√3`. Line centres are the
dips (local minima deeper than 30 % of the range) of the noise-free true signal.

**Panels** (Information tab, below Whole run): the table (median over the use cases of each use case's number, rule 7: runs
with positions, `q_place`, `q_err`, `q_err / q_place`, `n₀`, `R²`, capped share, and the two theory `q_sweep`); the cost to reach
τ (median first passage, the share that got there in the hover, p75) with the prediction `n₀ + τ⁻² / (q_err · I_max)` dotted
(measured cost is a median over single runs, the prediction the mean square reaching τ², so they agree only roughly); the ratio
`T_sweep(τ) / T_SBED(τ)` with the **best sweep spacing at each target** as yardstick and `R∞` (theory, exact, measured) as
horizontal lines; `R∞` measured against `W / Γ`, one colour per SNR, with the theory line (solid dots: `q_err(SBED) / q_err(best
sweep)`; open: the same from `q_place`); and the histogram of `drive_freq` relative to the nearest TRUE line centre in `γ`,
before and after `dip_detected_step` (the SBED after-detection histogram should peak at the steepest point, the sweep's is
flat). The older cost-ladder `n₀` / `e` of the Information table are kept (median-run statistic against the bound); the new
`n₀` / `q_err` are the mean-square version the article uses.

**Stale if** the generator, the seed derivation or the sweep grid changes: the true-signal check then raises and the section
lists the repeats. No cache or render invalidation: nothing here is stored.
## Localize phase (`nvision/metrics/localize.py`)

The run is split by observable markers, not by claimed-sigma thresholds: **search** until the run's own dip test fires
(`dip_detected_step` = τ), **localize** from τ to the lock, **refine** afterwards (q of the section above). The claim it
tests is that the Bayesian strategy gains in localize too: from the same starting point, its own detection, it reaches
the line in fewer measurements than a sweep.

* **Lock step** (stored per run at run time, in `stages.py`): ε = `LOCK_TOLERANCE_HWHM` (0.5) × the true HWHM
  (`lock_tolerance_phys`); `lock_step_true` = the first measurement from which the true error stays ≤ ε to the end of the
  run, `lock_step_claimed` the same for the claimed σ̂ (the lab version). Computed on the full per-step trace, not the
  downsampled series. `None` for drive-controlled models and rows recorded before it was added (re-run).
* **Localize cost** L = lock − τ per run (0 if it locked before detecting; `None` if it never detected or never locked).
  The "Localize" table of the Refine & speedup section shows the median and p75 over the runs that locked, the share that
  locked, the best sweep's median L over SBED's, and the bound below.
* **Bound**: van Trees restarted at τ with the posterior as prior, E[err²] ≥ (J_τ + n I_max)⁻¹, so reaching ε² takes at least
  max(0, ε⁻² − J_τ) / I_max measurements after τ, shown for J_τ (`posterior_J_tau_phys`) and J_τ at twice the bandwidth
  (`posterior_J_tau_2h_phys`), since J_τ is good to a factor 2. It bounds the mean square, not a run's first passage: a
  reference, not a per-run test. L / bound is the median over runs of each run's L over its own bound.
* The functions take an error array and a tolerance, never the true signal, so the lab can pass the error against the
  holdout centre or the claimed σ̂.

## Posterior metrics (run-time, `nvision/metrics/posterior.py`)

Computed from each step's particle cloud of the primary parameter (`zeeman_shift`) while the run is recorded, because the cache keeps no belief snapshots. Runs recorded earlier have none of these: **re-run** (`nv render` cannot fill them, and a plain re-run hits the cache, so use `--no-cache`, which replaces the combination's cached repeats). They read the beliefs only; no physics, SMC or SBED behaviour and no cache key or physics fingerprint changes.

Per step *t* (the belief after measurement *t*), in the scan entry's `series` next to `s`/`e`/`u` (downsampled with them):

* `z`: signed (estimate − truth) / σ̂, σ̂ = `belief.uncertainty()`. Calibrated and Gaussian ⇒ N(0, 1); this is what rule 1's coverage check wants instead of |error|.
* `j`: Fisher information ∫ p′(θ)²/p(θ) dθ of the posterior density, Hz⁻²; `h`: the KDE bandwidth it was computed with, Hz. `g`, `j`, `h` exist only at the stored steps (see Cost).
* `g`: negentropy ½ ln(2πe V) − H of the smoothed density (V its own variance, H its differential entropy), nats: 0 for a Gaussian, > 0 for anything else (a Gaussian maximises entropy at fixed variance). The reference is the smoothed density's own variance, not σ̂²: the KDE adds its bandwidth to the variance, so against σ̂² a Gaussian cloud would come out negative.
* `m`: posterior mass farther than 3 true FWHM from the posterior mode (mode of a weighted histogram with bins of FWHM/4, within FWHM/8 of the KDE's); `None` for drive-controlled models (no single FWHM).
* `x`: 1 if measurement *t* was placed more than 3 FWHM from the mode of the belief before it (belief *t − 1*), else 0; `None` at *t* = 0. The mode is mapped to the drive axis (the visible dip sits at `zero_field_freq + zeeman_shift`) and compared with the drive frequency measured. Tests the claim that a Gaussian-EIG acquisition wastes localize measurements on residual far mass.

Per run, in the results row and the API's compare fields (`_COMPARE_FIELDS`):

* `posterior_J_tau_phys` = J_τ: `j` at the run's own detection step `dip_detected_step` (snapshot τ − 1), the information of the posterior van Trees is restarted from (`articles/sections/sm_prior.tex`). `posterior_kde_bandwidth_tau_phys` is the bandwidth it was computed with and `posterior_J_tau_2h_phys` the same J with twice that bandwidth (the smoothing bracket, see below), `posterior_negentropy_tau` and `posterior_far_mass_tau` the `g`, `m` at τ. `None` if the run never detected a dip.
* `posterior_far_placed_steps`, `posterior_far_placed_share` (over measurements 2…n) and `posterior_far_placed_share_stage2` (over the localize stage only); `posterior_undefined_steps` (see below). For a sweep `x` is high by construction (it measures everywhere); the far-placement test is about SBED.

**Density estimate.** Weighted Gaussian KDE on a grid spanning the particle support (±5 bandwidths, clipped to the parameter's bounds), by weighted binning and FFT convolution; p′ uses the derivative-of-Gaussian kernel. The density is *reflected* at the bounds (the SMC reflects particles that leave them), not clipped.
* **Pilot bandwidth: improved Sheather–Jones** (Botev et al. 2010) with the Kish effective sample size 1/Σw², *not* Silverman. Silverman takes σ from the whole cloud, so for the two mirror peaks right after detection it smooths each peak to the width of the cloud and returns J ≈ 1/h², an artefact of the bandwidth (about 8× too low in the unit test); ISJ resolves the peaks.
* **Then a sample-point adaptive (Abramson) bandwidth** h_i = h·(pilot(x_i)/g)^(−1/2), g the weighted geometric mean of the pilot, clamped to [1, 100]·h and quantised to half-octave steps. One global h tuned to narrow peaks leaves a diffuse tail (a few sparse particles) as spikes, and p′²/p of those inflates J: with two narrow peaks plus 15 % diffuse mass the global-h J was 2.3× the exact one (the tail contributed as much as the peaks), the adaptive one is within about 10 %. `posterior_kde_bandwidth_tau_phys` stores the pilot h.
* A KDE of a Gaussian cloud is N(μ, σ² + h²), so J for a Gaussian posterior is 1/(σ² + h²), a few percent below 1/σ².

**J is set partly by the bandwidth, and can jump before a resample.** J of a smoothed density depends on how much it is smoothed: in the reference run (NVCenter-lorentzian, Gauss(0.01), Bayesian-SBED) J at the detection step τ = 7 was 7.5e-14 Hz⁻² at the ISJ bandwidth h, 4.4e-14 at 2h and 1.2e-13 at h/2, so J_τ is good to about a factor 2, not to 10 %. `posterior_J_tau_2h_phys` stores the 2h value as the bracket; quote J_τ with it, and the `h` series shows where the bandwidth moves. Right before a resample the weights are degenerate (ESS 219–294 of 1000 at steps 15–17) and the cloud is clumped at the scale of the last Liu–West jitter; there ISJ can drop h by 3× (h/σ̂ 0.22 → 0.085 at step 16) and J jumps with it (5.5e-13 → 2.4e-12 → 3.6e-12, then 1.0e-12 once resampled). That spike is **bandwidth-limited, not structure**: J at 2h on the same snapshots (8e-13, 1.0e-12) is at the resampled level, and J falls roughly as 1/h² there. A fixed-bandwidth and an adaptive KDE agree (they share h), as do 2^11 and 2^14 ISJ search bins, so neither tests this. Do not read a single step's `j` without its `h`; J_τ for a τ that lands in such a window is an upper end of the bracket. Unit tests with importance-weighted clouds (ESS 100–500) show J is not inflated by low ESS per se (Gaussian: 1.06× exact; mirror peaks: 0.8×).

**`x` over-counts right after detection.** The measurement that resolves the side of the two mirror peaks is placed at the other peak, so it counts as "far" although it is the intended one. The localize-stage share is therefore an upper bound for waste on residual far mass.

**Undefined steps.** A step has no posterior density when no measurement had updated its belief (a flat prior: a sweep's belief until its first flush) or when its cloud has no structure for the bandwidth rule (the posterior is still the flat prior after the first measurements; the ISJ equation has no root). Its `g`, `j`, `h` (and `m`, `x` of the next step, for an un-updated belief) are `None` and it is counted in `posterior_undefined_steps` (the second kind only among the steps a density is computed for). This happens early in a run: in a 2-repeat budget-length SBED write one repeat had 7 such steps of 80 stored, which aborted the repeat before this was recorded as missing. Any other failure raises. A run that never detects a dip (`dip_detected_step` null, e.g. a fixed-budget run with the stop rule off) has no τ fields but still has every per-step series; nothing here assumes the run ends by its stop rule (budget-length SBED runs, 2500 steps, were checked, including a real write of two repeats on two runners).

**Real-data (MATLAB/lab) runs get none of these fields**: `nv matlab` calls the metrics with no `run_result` (its stage and milestone metrics need a true signal too), and the posterior series attach is a no-op there.

**Cost.** A KDE costs about 25 ms (68 s for a 2500-step SBED run when computed at every step), so `j`, `g` and `h` are computed only at the steps the stored series keeps (at most 80, the same indices as `s`) and at τ; `z`, `m` and `x` need no density (a weighted histogram with bins of a quarter FWHM gives the mode) and cover every step. Each distinct belief is summarised once (sweeps share a belief copy across buffered steps). Measured with three runs sharing the machine, startup included: Bayesian-SBED 190 steps 3.6 s of 34 s wall (~10 %), Bayesian-SBED at SNR(1) 2500 steps 3.0 s of 49 s, SimpleSweep 1768 steps (239 distinct beliefs) 2.8 s of 33 s.

## Stuck-belief check (run-time, `nvision/metrics/fit_gap.py`)

A particle-filter belief can freeze on a wrong line shape (e.g. the wrong split between the Lorentzian and Gaussian
widths) and stay there: its claimed σ keeps falling while the true error stays put, so neither the claimed σ nor a
1/√N curve of it shows the failure. `belief_fit_gap` detects it without the truth, so it also works on lab data. The
final belief's own model is fitted to every observation it absorbed (bounded least squares on the unit cube, from the
belief mean, the best particle, and a one-at-a-time ladder over each non-primary parameter), and

$$D = \frac{SS_{\rm belief} - SS_{\rm fit}}{\hat\sigma^2_{\rm fit}}, \qquad \hat\sigma^2_{\rm fit} = \frac{SS_{\rm fit}}{n-k}.$$

A belief that has converged sits within noise of the best fit (D of order k, a few); D > `STUCK_FIT_GAP` (20) flags a
stuck one. `belief_fit_gap_shift_phys` is how far the primary would move to the fit (Hz). The fit's Jacobian uses a
relative step of 1e-3 because the models compute in float32 (the default step reads a zero gradient).

What it showed (2026-11-17, window 5 linewidths, SNR 20, 20 paired repeats per arm, dry runs): with a = 0.98 the true
parameters beat the belief by Δχ² > 20 in 14 of 20 runs; D flagged all 14 and none of the 6 healthy runs, and tracked
the truth-based gap (e.g. 1723 → 6207, 138 → 144, 39 → 33). The cause was Liu-West rejuvenation too weak to leave the
wrong shape (jitter ≈ 0.2 × the cloud's spread at a = 0.98). An a × ESS grid (a ∈ {0.98 … 0.80}, ESS ∈ {0.2, 0.5, 0.8},
SNR 20 and 2) showed a ≤ 0.85 removes it at any ESS (stuck 0-5 %, coverage 0.9-1.0 at SNR 20; SNR 2 not hurt), so
`NVISION_SMC_A_PARAM` defaults to 0.85. Checked on windows 1.6 and 3 (SNR 10) and 5 (SNR 5): stuck 30/35/20 % →
5/10/0 %, RMS error 3-5x lower; the 1.6-linewidth window stays somewhat overconfident (2σ coverage 0.70).
Final configuration (2026-11-18): the Gaussian-share-0 window-grid signals use a Lorentzian belief (no `sigma_inhom`;
with a = 0.98 it still stuck in 20-35 % of runs on width and contrast, with a = 0.85 in none), 10,000 particles
(smoother claimed σ, ~4x the run time, 1.4 GB peak per 2500-step repeat) and resampling below 0.5 N_p (on the
1.6-linewidth window, SNR 10, it cut the typical run's largest log-σ step from 1.5 to 0.02 and raised 2σ coverage
from 0.80 to 1.00; 10 repeats per arm). A rare single σ spike (log-step ≈ 3-6, one seed per cell) is the same at
ESS 0.2 and 0.5, so it is not from resampling; not yet traced.
Lab runs (`nv matlab`) store it in the run record and print it; replayed shots are reused (`shot_reuse` draws per
distinct shot) and count as independent observations, so there D is inflated by about `shot_reuse` and the printed
flag compares against `STUCK_FIT_GAP × shot_reuse`.
Sweeps: their SMC belief is not resampled during the sweep and their reported estimate is the curve fit, so a large D
there describes the belief only, not the sweep's answer.

## Rules

1. **Never use the accuracy at the end of a run or stage as a quality metric.** Every run measures the same budget, so the error it ends with mostly reflects the budget, not the strategy. What matters is that the claimed
   σ is right: report the share of runs with |error| ≤ 1σ and ≤ 2σ against the Gaussian 68% / 95%, or |error| /
   claimed uncertainty σ̂. Below nominal = overconfident (the belief is known to be; see the locator memory notes).
2. **Every locator gets the same budget.** `NVISION_MAX_MEASUREMENTS` (default 2500, for every signal and every window
   of the SNR x window study) is the SBED family's `max_steps` and caps a plain sweep's total measurements. Every run measures all of it: the convergence rule
   is recorded (`primary_converged_step`) but never stops a run. A run whose rule never fired (`primary_converged_step` is
   null) is capped: count it at the budget, as a lower bound, and show the capped share.
3. **Show the raw metrics before ranking anything.** No arbitrary reference locator (e.g. "divided by k = 3"), no
   "cheapest k" summaries, until the plain per-locator metrics have been looked at.
4. **Use physical axes.** SNR = C / noise σ with C = contrast_total / 2 (∞ without noise) and the effective FWHM, per the stage table
   above.
5. **Runs are paired.** Repeat *i* has the same true signal for every locator and noise level
   (`signal_repeat_key`), so per-repeat comparisons are valid; measurement noise differs, as it must.
6. **Stops are compared only at matched reliability.** Comparing measurements-to-stop is fair only if the locators'
   claimed uncertainty σ̂ are equally honest (rule 1); otherwise one could look faster by stopping riskier.
7. **Metrics are per use case.** A use case is one signal at one noise level: window, line shape and SNR (generator ×
   noise for other studies). Compute every headline number (median and p75 cost, capped share, σ coverage, stage
   durations) within one use case first; never read it off runs pooled across windows, shapes or SNRs, since the cost
   spans orders of magnitude between them. A selection of several use cases is summarised as the **median over the
   use cases of each use case's own number** (every use case counts once, whatever its number of repeats) and is
   labelled with what it covers ("Median over 45 use cases · window 5, 10, 20 × FWHM · … · SNR 1…20"); the per-use-case
   numbers stay one click away. Such a median mixes easy and hard use cases, so it is a summary of the selection, not a
   property of any one signal. Trends across use cases are still shown as curves, one point per use case.
8. **Medians hide the tail.** Show p75 and the capped share next to the median cost; the spacings differ more in the
   tail than at the median.

## Where it lives

* Per-run metrics: `nvision/metrics/stages.py` (stage boundaries, η), `nvision/metrics/milestones.py` (stop step),
  `nvision/metrics/series.py` (per-step series), `nvision/metrics/posterior.py` (J_τ, z, negentropy, far mass),
  `nvision/metrics/compare.py` (`nv compare` tables),
  `nvision/metrics/refine_rates.py` / `refine_rates_runs.py` (refine rates and the speedup, `/api/refine-rates`).
* Results UI views: **Information** (the former Stages tab; scope selector Whole run / Search; **Whole run** also carries
  **Cost & search**: every locator side by side, chips to show/hide, a median-over-selection block and a block per use
  case, and cost and search vs SNR across use cases, `/api/run-rows`; the Whole run charts are the information-vs-bound
  ratio R(n), the cost to reach a precision against the bound, a table with the lag n₀, efficiency e and cost at 5× gain,
  the claimed-σ and true-error curves, the information rate and the paired per-repeat difference against SBED; Search keeps
  its stage metrics; see "Information against the bound" below),
  **Scaling** (fixed locator, many use cases: search and localize steps vs window, localize steps vs SNR, refine steps vs
  (FWHM/(SNR τ))², each with its predicted scaling dotted and the Fisher bound dashed, a panel only when the selection spans
  at least two values of its axis), **Sanity** (σ coverage vs SNR "Is the claimed uncertainty σ̂ right?", then the stop-quality panels, then the failed runs; the failed-run
  count is in the tab label), and **Run** for one repeat. Stages and Sanity default to the median over the selected use cases.
  See [ui_architecture.md](ui_architecture.md).

## Not done yet

* The oracle columns exist only for runs recorded after they were added; older runs show a dash (the stage metrics are
  computed from the belief snapshots at run time and the cache keeps no snapshots, so there is nothing to backfill from).
  Re-run to get them.
* Search is judged by its steps against window / FWHM_true in the Scaling view; there is no analytic "passes" model for it
  yet, and the prior FWHM / grid step enters only through `stage_prior_fwhm_phys`.
* The boundaries inherit the claimed uncertainty σ̂'s overconfidence (on the SNR × window study σ is ≈ 2 ε after the first measurement
  while the true error is ≈ 5 ε, and a few runs start refine with the error still several ε). They are as good as the
  Sanity coverage; a σ-free observable for "position known to within ε" is still to find.
* The search→localize factor `FOCUS_SIGMA_DROP` = 0.7 is arbitrary (0.9 fires before the data have localised anything; 0.25 fires later than the true error reaches ε, at
  step 37 / 200); the stage boundaries also depend on δ = 0.1; a one-off sensitivity check (δ = 0.05 / 0.1 /
  0.2) that the locator ranking survives is still to do.
* The signed z = (estimate − truth) / σ̂ is stored from the posterior-metrics runs on (series `z`, see above); earlier runs keep only
  |error|, so σ honesty there is checked against a half-normal, not N(0, 1). No Sanity/Information panel reads `z` yet.
* SBED-Drive against every fixed drive amplitude, including the best one in hindsight.
