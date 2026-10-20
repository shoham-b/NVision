# Glossary: NV signal vocabulary (`drive_freq`, `zero_field_freq`, `resonance_freq`, `detuning`, `hwhm`, ...)

The word "frequency" used to name several different things in the SBED/SMC code, and a few parameters
carried names that were not physical. Use the terms below exactly; never write plain "frequency" for any
of them in new code, comments or docs.

## Frequencies and shifts

| Term | Symbol / names | What it is | Space |
|---|---|---|---|
| **`drive_freq`** | `drive_freq_phys`, `drive_freq_unit`, `drive_freq_bounds_phys` (belief), `drive_freq_min_phys`/`drive_freq_max_phys` (experiment, locators), `FocusWindow`, `FocusRegion` | The microwave drive frequency swept by the experiment, i.e. the axis that is *measured along*. A coordinate, not a parameter of the signal. Its full range is fixed per run; a sweep locator's `FocusWindow` is a narrowing sub-interval of it, and the Bayesian locator's `FocusRegion` a set of narrowing sections of it. | `_phys`: Hz. `_unit`: `[0, 1]` over the full drive-frequency range. |
| **`candidate_drive_freq`** | `candidate_drive_freq_phys`, `candidate_drive_freq_unit`, `get_candidate_drive_freq_phys()` | A *drive frequency* that may be measured next. The acquisition step scores candidates by expected information gain and picks one. | Same two spaces as `drive_freq`. |
| **`zero_field_freq`** | model parameter `zero_field_freq` (was `center_freq`), `with_fixed_zero_field_freq`, `infer_zero_field_freq`, `zero_field_freq_crlb()`, `NVISION_ZERO_FIELD_FREQ_*` | The resonance at zero Zeeman shift: the zero-field splitting D (about 2.87 GHz), to first order the midpoint of the two Zeeman lines. A property of the sample (temperature and strain shift it); no line sits exactly at it once `zeeman_shift` is non-zero. Fixed to D by default (a known instrument constant), a free particle dimension only with `with_fixed_zero_field_freq=False`. | `_phys`: Hz. `_unit`: `[0, 1]` over its prior bounds. |
| **`resonance_freq`** | per-line value, not a stored parameter | The transition frequency of one line: `zero_field_freq ± zeeman_shift`, plus the line's hyperfine offset (`± hyperfine_shift`). | Hz |
| **`detuning`** | `detuning = drive_freq - resonance_freq` | Distance of the drive from one line's resonance. Every line profile is a function of its own `detuning` and `hwhm`. | Hz (a frequency *difference*) |
| **`zeeman_shift`** | `zeeman_shift` (was `zeeman_split`) | The Zeeman shift of each of the two line groups: they sit at `zero_field_freq ± zeeman_shift`, so their full separation is `2 * zeeman_shift` (about gamma_e * B along the NV axis). Not a frequency of the signal and never called one. The default *primary parameter* of an NV run. | `_phys`: Hz (a frequency *difference*). |
| **`hyperfine_shift`** | `hyperfine_shift` (was `split`) | Offset of the outer hyperfine lines from their group's centre (N-14: lines at `-hyperfine_shift`, 0, `+hyperfine_shift`; N-15: two lines at `+/- hyperfine_shift`, spacing `2 * hyperfine_shift`). Unrelated to `zeeman_shift`. | `_phys`: Hz. |

## Line shape

| Term | Symbol / names | What it is |
|---|---|---|
| **`pl_normalized`** | the measured/modelled signal | The photoluminescence rate divided by its off-resonance level, so it is 1 away from every line. NV models have no free background (it is the constant 1); only the legacy Lorentzian/Gaussian/Voigt-Zeeman spectra keep a free `background`. |
| **`contrast`** | per-line `contrast` (was `dip_depth`) | Fractional drop of `pl_normalized` at a line's resonance (ODMR contrast). |
| **`contrast_total`** | `contrast_total` (was `c_total`) | Total contrast summed over all lines of the model (the line weights sum to it, so it does not depend on the line count). |
| **`hwhm`** | `hwhm` (was `linewidth`) | Half width at half maximum of the Lorentzian line. It is an *effective* width: the intrinsic floor `1/(pi*T2)` (`NV_T2_HWHM_HZ`; `T2*` where nothing separates static dephasing), broadened by microwave power. The model only contains the T2 floor and the saturation term `hwhm = NV_T2_HWHM_HZ * sqrt(1 + s)`; optical-pumping broadening is not modelled separately and is absorbed into the fitted value. A single spectrum cannot separate the contributions. |
| **`hwhm_lorentz`** | `hwhm_lorentz` (was `homogeneous_linewidth`) | The Lorentzian HWHM inside the Voigt models. |
| **`sigma_inhom`** | `sigma_inhom` | Standard deviation of the Gaussian inhomogeneous width. Conversions: `FWHM = 2 * HWHM` (Lorentzian), `FWHM = 2.355 * sigma` (Gaussian). |

`k_np` (nuclear-spin polarisation asymmetry of the outer hyperfine lines) is unchanged. A *dip* is the
signature of one resonance in `pl_normalized`; `find_dips` and the focus sections talk about dips, the
physics about resonances.

### Microwave drive (drive-controlled experiments)

| Term | Symbol / names | What it is | Space |
|---|---|---|---|
| **drive** | `DriveSetting`, `Observation.drive`, `locator.next_drive()`, `TrueSignal.baseline_drive` | The microwave drive applied to one measurement: amplitude, plus the pulse duration for pulsed ODMR. A per-measurement *setting*, not a model parameter. See [drive control](amplitude_control.md). | — |
| **`drive_amp_unit`** | `DriveSetting.drive_amp_unit`, `DRIVE_AMP_MIN_UNIT` | Microwave field amplitude as a fraction of the source's maximum (linear in field, so proportional to the Rabi frequency). | `[DRIVE_AMP_MIN_UNIT, 1]`, already normalized; there is no `_phys` amplitude. |
| **`pulse_duration_s`** | `DriveSetting.pulse_duration_s` | Pulsed ODMR only: the pulse length, the locator's π-pulse calibration `t = 1/(2·Ω_est)`. | seconds |
| **`drive_amp_sat`**, **`rabi_per_amp`** | model parameters | The sample's inferred *response* to the drive: CW amplitude at saturation `s = 1`; pulsed Rabi frequency at full amplitude (Hz). Nuisance parameters, excluded from convergence. | `drive_amp_sat`: fraction of max field; `rabi_per_amp`: Hz |

## Why the distinction matters

* `zero_field_freq` and `drive_freq` are different objects even though, by construction of the
  NV bound builders, the `zero_field_freq` prior range spans exactly the drive-frequency range. Code that needs
  that range uses `drive_freq_bounds_phys` (belief) or `experiment.drive_freq_min_phys` / `drive_freq_max_phys`
  (runner), never `physical_param_bounds["zero_field_freq"]`. See [core architecture](core_architecture.md) and
  `nvision/belief/coordinate.py` for the rescale-vs-focus split.
* Every variable, getter and setter touching these carries `_phys` or `_unit`
  (see `AGENTS.md`): `candidate_drive_freq_phys` vs `candidate_drive_freq_unit`, `drive_freq_min_phys`, ...
* The convergence "milestone" metrics refer to the *primary parameter* (`zeeman_shift`, else
  `split`, else `zero_field_freq`; `resolve_primary_param`), not to a frequency. They are named
  `primary_*` (`primary_converged_step`, `steps_to_primary`, `err_primary_at_milestone`, ...) and
  the shift-specific ones `*_shift` (`err_shift_at_milestone`, `final_err_shift`, ...).

## What deliberately still says "frequency"

* `OverFrequencyGaussianNoise`, `CompositeOverFrequencyNoise`, `over_frequency_noise`,
  `frequency_noise_model`: noise that varies *along `drive_freq`*. These are persisted noise names and are unchanged.
* `formatHz` (UI): formats any Hz quantity; its unit-type tag `'frequency'` is a *unit*, not a parameter.
* Axis titles in plots and prose about the physical
  spectrum.
* MATLAB `.mat` field `esr.frequency`: an external file format, read as-is by `nvision/tools/matlab_loader.py`.

## Rename migration (`frequency` → `center_freq`)

The rename touches persisted names, so **all cached results must be fully re-run** (not just
`nv render`):

* model parameter `frequency` → `center_freq` (specs, bounds, typed params, `true_params`, cache payloads);
* metrics: `err_fb*`/`uncert_fb*`/`steps_to_fb`/`fb_at_milestone` → `*_primary*`;
  `err_fc*`/`final_err_fc`/`fc_at_milestone` → `*_split*`; `splitting_converged_step` → `primary_converged_step`;
  `sobol_freq_*` → `sobol_primary_*`; `scan_param` → `center_param` (the model parameter locating the signal on `drive_freq`);
  MATLAB stats plot `matlab_freq_stats` → `matlab_drive_freq_stats`, `freq_hz` → `drive_freq_phys`;
* `PHYSICS_CONFIG_FINGERPRINT` is bumped (`center-freq-rename-v1`) and `CACHE_SCHEMA_VERSION` is 12;
  `nv serve` hides (with a logged warning) combinations written under an older schema instead of
  half-rendering them. They stay on disk untouched.
* Environment variables (old names are **no longer read**; the program refuses to start if one is
  still set, see `nvision/tools/renamed_env.py`; values and units are unchanged):

  | Old | New |
  |---|---|
  | `NVISION_FREQ_CONVERGENCE_THRESHOLD` | `NVISION_CENTER_FREQ_CONVERGENCE_THRESHOLD` |
  | `NVISION_FREQ_CRLB_SAFETY_FACTOR` | `NVISION_CENTER_FREQ_CRLB_SAFETY_FACTOR` |
  | `NVISION_NV_CENTER_FREQ_DELTA_HZ` (and the short-lived `NVISION_NV_PROBE_DELTA_HZ`) | `NVISION_NV_DRIVE_FREQ_DELTA_HZ` |
  | `NVISION_SWEEP_FIT_FREQ_STARTS` | `NVISION_SWEEP_FIT_CENTER_FREQ_STARTS` |

  The Python constants `DEFAULT_NV_CENTER_FREQ_X_MIN/MAX` are now `DEFAULT_NV_DRIVE_FREQ_MIN_PHYS/MAX_PHYS`.

## Probe axis → `drive_freq` (second rename)

The generic names for the measured axis are gone: `x`/`xs` (where it is the drive frequency), `x_min`/`x_max` →
`drive_freq_min_phys`/`drive_freq_max_phys`, `physical_x_bounds`/`probe_x_phys` → `drive_freq_bounds_phys`,
`probe_lo/hi_phys` and `_full_domain_lo/hi` → `drive_freq_min/max_phys`, `candidate_x*` → `candidate_drive_freq*`,
`probe_axis_param` → `center_param`, `probe_axis_phys`/`_mhz` → `drive_freq_phys`/`_mhz`,
`matlab_probe_stats` → `matlab_drive_freq_stats`, `NVISION_NV_PROBE_DELTA_HZ` → `NVISION_NV_DRIVE_FREQ_DELTA_HZ`.
Cache schema 13 / fingerprint `drive-freq-rename-v1`: re-run, not `nv render`.

`Observation.x` → `Observation.drive_freq_unit`, `ObservationHistory.xs` → `.drive_freqs_unit` (always unit) plus
`.drive_freqs_phys` (Hz, needs `drive_freq_bounds_phys` at construction). The unit/physical guessing in
`refocus/window.py`, `StagedSobolSweepLocator.observe` and `convert.run_result_to_history_df` is gone: every locator now
records unit observations, `GridMarginalDistribution` maps them to Hz itself via its own `drive_freq_bounds_phys`, and
`infer_focus_window_physical` raises unless its history spans the requested domain. Side effect: the staged-Sobol
locator's belief now receives unit (not physical) observations, as every other locator's belief does.

`Locator.next()` is a unit drive frequency: always `drive_freq_unit` in `[0, 1]` over the full range (what
`CoreExperiment.measure` takes), never Hz. SBED works in Hz inside `_acquire` and maps it in `next()`. Callers use
`Locator.next_drive_freq_unit()`, which raises `ValueError` for anything outside `[0, 1]` (so a physical-Hz value cannot slip
through).

## Physical-names rename (third rename)

Persisted names changed, so **all cached results must be fully re-run** (not just `nv render`):

| Old | New |
|---|---|
| `center_freq` | `zero_field_freq` |
| `linewidth` / `homogeneous_linewidth` | `hwhm` / `hwhm_lorentz` |
| `zeeman_split` | `zeeman_shift` |
| `split` (hyperfine) / `hf_split` | `hyperfine_shift` |
| `dip_depth` / `c_total` | `contrast` / `contrast_total` |
| `NV_NATURAL_HWHM_HZ` | `NV_T2_HWHM_HZ` |
| `MIN/MAX_LINEWIDTH`, `MIN/MAX_SPLIT` | `MIN/MAX_HWHM`, `MIN/MAX_HYPERFINE_SHIFT` |
| metrics `err_split*`, `final_err_split`, `split_at_milestone`, `milestone_error_*_split` | `err_shift*`, `final_err_shift`, `shift_at_milestone`, `milestone_error_*_shift` (`shift` = `zeeman_shift`, else `hyperfine_shift`) |
| `docs/dip_depth_reparametrization.md` | `docs/contrast_reparametrization.md` |

`PHYSICS_CONFIG_FINGERPRINT` carries the new tag `physical-names-v1`, and `CACHE_SCHEMA_VERSION` is 14 (`nv serve`
hides older combinations with a warning). Every renamed `NVISION_*` environment variable is listed in
`nvision/tools/renamed_env.py` (old names are refused at start-up), e.g. `NVISION_CENTER_FREQ_CONVERGENCE_THRESHOLD` ->
`NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD`, `NVISION_MIN_LINEWIDTH` -> `NVISION_MIN_HWHM`,
`NVISION_ZEEMAN_SPLIT_CONVERGENCE_THRESHOLD` -> `NVISION_ZEEMAN_SHIFT_CONVERGENCE_THRESHOLD`.
