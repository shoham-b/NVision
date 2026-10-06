# Glossary: `drive_freq`, candidate, `center_freq`, `zeeman_split`

The word "frequency" used to name four different things in the SBED/SMC code. They are now
four distinct terms. Use them exactly; never write plain "frequency" for any of them in new
code, comments or docs.

| Term | Symbol / names | What it is | Space |
|---|---|---|---|
| **`drive_freq`** | `drive_freq_phys`, `drive_freq_unit`, `drive_freq_bounds_phys` (belief), `drive_freq_min_phys`/`drive_freq_max_phys` (experiment, locators), `FocusWindow` | The microwave drive frequency swept by the experiment, i.e. the axis that is *measured along*. A coordinate, not a parameter of the signal: signal = f(`drive_freq`; `center_freq`, `zeeman_split`, ...). Its full range is fixed per run; the locator's `FocusWindow` is a narrowing sub-interval of it. | `_phys`: Hz. `_unit`: `[0, 1]` over the full drive-frequency range. |
| **candidate_drive_freq** | `candidate_drive_freq_phys`, `candidate_drive_freq_unit`, `get_candidate_drive_freq_phys()` | A *drive frequency* that may be measured next. The acquisition step scores candidates by expected information gain and picks one. | Same two spaces as `drive_freq`. |
| **`center_freq`** | model parameter `center_freq` (was `frequency`), `with_fixed_center_freq`, `infer_center_freq`, `crlb_center_freq()`, `NVISION_CENTER_FREQ_*` | The zero-field dip-centre *model parameter* ``f_B`` (≈ 2.87 GHz, the NV zero-field splitting D). A property of the signal. Fixed to D by default (a known instrument constant, not a particle dimension); a free particle dimension only with `with_fixed_center_freq=False`. | `_phys`: Hz. `_unit`: `[0, 1]` over its prior bounds. |
| **`zeeman_split`** | `zeeman_split` | The Zeeman half-separation of the two dip groups about `center_freq`. It is **not** a frequency of the signal centre and never called one. It is the default *primary parameter* of an NV run. | `_phys`: Hz (a frequency *difference*). |

`split` (the hyperfine splitting) is a fifth, separate parameter and is unrelated to `zeeman_split`.

## Why the distinction matters

* `center_freq` and `drive_freq` are different objects even though, by construction of the
  NV bound builders, the `center_freq` prior range spans exactly the drive-frequency range. Code that needs
  that range uses `drive_freq_bounds_phys` (belief) or `experiment.drive_freq_min_phys` / `drive_freq_max_phys`
  (runner), never `physical_param_bounds["center_freq"]`. See [core architecture](core_architecture.md) and
  `nvision/belief/coordinate.py` for the rescale-vs-focus split.
* Every variable, getter and setter touching these carries `_phys` or `_unit`
  (see `AGENTS.md`): `candidate_drive_freq_phys` vs `candidate_drive_freq_unit`, `drive_freq_min_phys`, ...
* The convergence "milestone" metrics refer to the *primary parameter* (`zeeman_split`, else
  `split`, else `center_freq`; `resolve_primary_param`), not to a frequency. They are named
  `primary_*` (`primary_converged_step`, `steps_to_primary`, `err_primary_at_milestone`, ...) and
  the splitting-specific ones `*_split` (`err_split_at_milestone`, `final_err_split`, ...).

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

Not yet renamed: `Observation.x` (and `ObservationHistory.xs`) is a *unit* coordinate in the SBED/SMC path but a
*physical* one in the sweep locators, so it cannot honestly be called `drive_freq_unit` until that is split.
